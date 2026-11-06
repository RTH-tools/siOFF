"""Pure core for off-target analysis.

The entry points are free of CLI concerns (no stdout, no Typer/Click
exceptions):

- [`compute_off_targets`][sioff.core.off_targets.compute_off_targets] -- the
  one engine. A predictions file, a folder of them, an in-memory frame or the
  hits of an in-process search is split into one part per siRNA, and every
  siRNA is scored as its own task; a generator yielding
  ``(pl.DataFrame, meta)`` per siRNA.
- [`compute_off_targets_single`][sioff.core.off_targets.compute_off_targets_single] -- a single prediction source (an in-memory
  DataFrame, one RIsearch2 output file, or RIsearch run in-process); returns
  ``(pl.DataFrame, meta)``. Being replaced by `compute_off_targets`.
- [`compute_off_targets_directory`][sioff.core.off_targets.compute_off_targets_directory] -- a directory of per-siRNA prediction
  files; a generator yielding ``(pl.DataFrame, meta)`` per siRNA. Being
  replaced by `compute_off_targets`.

The core writes no output files, with two exceptions. The in-process search
builds the RIsearch index ``<target>.idx`` next to ``target_fasta`` when no
``target_index`` is given (and reuses it later). And there are temporary
files: the Arrow-IPC copy of the transcriptome that every task memory-maps,
the per-siRNA split parts, and accessibility profiles spilled to disk so
workers can memory-map them. They live inside the generator's ``try/finally`` and are
removed when the run ends, on error too, so callers should consume the
generator fully (or close it) for deterministic cleanup. The accessibility
cache built next to ``-a`` profiles is kept: it is reused by later runs.

``meta`` carries the partition-function state downstream formatters need:
``z_per_sirna`` and ``on_target_weights``.

The public wrappers [`sioff.off_targets`][sioff.off_targets], [`sioff.accessibility`][sioff.accessibility],
[`sioff.index`][sioff.index] and [`sioff.search`][sioff.search] in `sioff.api` add path
coercion and mode dispatch on top of this layer.
"""

import copy
import ctypes
import gc
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Mapping, Optional, Union, cast

import polars as pl
import pyarrow as pa
import pyarrow.ipc
from loguru import logger

from sioff.core.split import merge_splits, split_file, split_frame
from sioff.models import RISEARCH_COLUMNS, RISEARCH_SCHEMA
from sioff.services.accessibility import (
    PARQUET_SUFFIX,
    AccessibilityError,
    GenomeAccessibilityService,
    ensure_accessibility_cache,
    has_profiles,
)
from sioff.services.annotation_parser import AnnotationParser
from sioff.services.intersection_service import (
    IntersectionService,
    sort_for_preload,
)
from sioff.services.probability import ProbabilityService
from sioff.services.profiling import PipelineProfiler
from sioff.services.risearch_parser import RIsearchParser
from sioff.services.risearch_service import RIsearchService
from sioff.services.sirna_table import SirnaRecord, read_sirna_table

# ---------------------------------------------------------------------------
# Per-worker state for directory-mode parallel processing (spawn-safe).
# _init_worker() runs once per spawned worker process and populates these
# module-level globals.  spawn (not fork) is used to avoid Rayon/Polars
# thread-pool deadlocks that occur when forking a process that already has
# background threads active.
# ---------------------------------------------------------------------------
_WORKER_DF_TRANS: Optional[pl.DataFrame] = None
_WORKER_INTERSECTOR: Optional[object] = None
_WORKER_PROB_SERVICE: Optional[object] = None
_WORKER_ON_TARGET_MAP: dict = {}
_WORKER_SELF_HYB_EMIN: dict = {}
# --summary-only: score per siRNA with ProbabilityService(aggregate_only=True)
# and return only the metadata, no per-site table.
_WORKER_SUMMARY_ONLY: bool = False
_WORKER_DETAILED_REPORT: bool = False
# The globals _init_worker sets, with their defaults. An in-process run of
# compute_off_targets keeps its own copy and puts it back before each task.
_WORKER_STATE_DEFAULTS: dict[str, Any] = {
    "_WORKER_DF_TRANS": None,
    "_WORKER_INTERSECTOR": None,
    "_WORKER_PROB_SERVICE": None,
    "_WORKER_ON_TARGET_MAP": {},
    "_WORKER_SELF_HYB_EMIN": {},
    "_WORKER_SUMMARY_ONLY": False,
    "_WORKER_DETAILED_REPORT": False,
}


def _prepare_accessibility_cache(
    accessibility_dir: Path, accessibility_cache_dir: Optional[Path], workers: int
) -> Path:
    """Check the profile source and build the uint8 cache before any scoring.

    Parameters
    ----------
    accessibility_dir : pathlib.Path
        Directory with the npy cache or per-chromosome Parquet profiles.
    accessibility_cache_dir : pathlib.Path, optional
        Where the cache lives; defaults to ``accessibility_dir``.
    workers : int
        Processes used to pack the profiles.

    Returns
    -------
    pathlib.Path
        The cache directory the service should read.

    Raises
    ------
    AccessibilityError
        When neither directory holds any profile, when a profile cannot be
        read, or when the cache directory is not writable.
    """
    cache_dir = (
        Path(accessibility_cache_dir)
        if accessibility_cache_dir is not None
        else Path(accessibility_dir)
    )
    if not has_profiles(accessibility_dir) and not has_profiles(cache_dir):
        where = (
            str(accessibility_dir)
            if cache_dir == Path(accessibility_dir)
            else f"{accessibility_dir} or {cache_dir}"
        )
        raise AccessibilityError(
            f"No accessibility profiles in {where}. Pass the directory written by "
            "'sioff accessibility' (-a/--accessibility-dir)."
        )
    t0 = time.perf_counter()
    built: list[str] = []
    write_error: Optional[OSError] = None
    try:
        built = ensure_accessibility_cache(
            Path(accessibility_dir), cache_dir, workers=workers
        )
    except AccessibilityError as e:
        # The build pool wraps a worker's OSError; any other failure stands.
        if not isinstance(e.__cause__, OSError):
            raise
        write_error = e.__cause__
    except OSError as e:
        write_error = e
    if write_error is not None:
        raise AccessibilityError(
            f"Cannot write the accessibility cache to {cache_dir}: {write_error}. "
            "Pass a writable cache directory (--accessibility-cache-dir)."
        ) from write_error
    if built:
        logger.info(
            f"[accessibility] packed accessibility cache for {len(built)} record(s) "
            f"in {cache_dir} in {time.perf_counter() - t0:.1f}s"
        )
    else:
        logger.info(f"[accessibility] cache in {cache_dir} is complete")
    return cache_dir


def _drop_unexpressed(df_trans: pl.DataFrame) -> pl.DataFrame:
    """Drop annotation rows whose expression is <= 0.

    A site on such a transcript has Boltzmann weight 0 and changes no partition
    function, but on a typical RNA-seq annotation (GENCODE v47 + H1299 TPM) more
    than half the exon lines are unexpressed, so half of every siRNA's site x
    transcript rows were intersected, looked up and scored for nothing. The
    legacy pipeline's GTF path skips them as well.

    Parameters
    ----------
    df_trans : polars.DataFrame
        Annotation with an ``exp_value`` column.

    Returns
    -------
    polars.DataFrame
        The rows with ``exp_value > 0``.
    """
    kept = df_trans.filter(pl.col("exp_value") > 0)
    dropped = df_trans.height - kept.height
    if dropped:
        logger.info(
            f"Dropped {dropped:,} of {df_trans.height:,} annotation rows with "
            "expression <= 0 (weight 0 in every partition function)"
        )
    return kept


def _apply_self_hyb_emin(df: pl.DataFrame, self_hyb_emin: dict) -> pl.DataFrame:
    """Set ``raw_e_min`` to the legacy E_min: self-hybridisation or lower hit.

    The legacy pipeline starts E_min at the siRNA's self-hybridisation energy
    and lowers it to any expressed site that binds more strongly (G:U wobbles
    can beat the perfect complement). Overriding ``raw_e_min`` with the
    self-hybridisation energy alone moved the alpha/gamma clamp threshold for
    those siRNAs: 11 of 2000 in the RIsearch3 off-target benchmark differed
    from the legacy pipeline by up to 0.069 in P_off_target. Call after the
    intersection, so the sites are the expressed ones the legacy pipeline reads.

    Parameters
    ----------
    df : polars.DataFrame
        One siRNA's predictions, with ``energy`` and ``raw_e_min`` columns.
    self_hyb_emin : dict
        ``sirna_id -> self-hybridisation E_min`` (empty without ``-s``).

    Returns
    -------
    polars.DataFrame
        ``df`` with ``raw_e_min`` replaced when the siRNA has a
        self-hybridisation E_min; unchanged otherwise.
    """
    if not self_hyb_emin or "raw_e_min" not in df.columns or df.height == 0:
        return df
    sirna_id = str(df["sirna_id"][0])
    if sirna_id not in self_hyb_emin:
        return df
    e_min = min(float(self_hyb_emin[sirna_id]), float(cast(float, df["energy"].min())))
    return df.with_columns(pl.lit(e_min).cast(pl.Float32).alias("raw_e_min"))


def _read_shared_ipc(path: Path) -> pl.DataFrame:
    """Polars view of an uncompressed Arrow IPC file, memory-mapped (no copy).

    Parameters
    ----------
    path : pathlib.Path
        Arrow IPC file written uncompressed.

    Returns
    -------
    polars.DataFrame
        Columns backed by the memory map, so processes reading the same file
        share its pages.
    """
    table = pyarrow.ipc.open_file(pa.memory_map(str(path))).read_all()
    return cast(pl.DataFrame, pl.from_arrow(table, rechunk=False))


_HEAVY_COLUMN_PREFIXES = ("boltzmann_weight", "Z_sirna")


def _drop_heavy_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop the raw Boltzmann weights, partition functions and ``E_min``.

    These per-interaction columns are intermediate to the probabilities; they
    are kept only under ``detailed_report``.

    Returns
    -------
    polars.DataFrame
        ``df`` without those columns.
    """
    drop_cols = [
        c for c in df.columns if c.startswith(_HEAVY_COLUMN_PREFIXES) or c == "E_min"
    ]
    return df.drop(drop_cols) if drop_cols else df


def _init_worker(
    ipc_path: str,
    accessibility_dir: str,
    polars_max_threads: int,
    on_target_map: dict,
    self_hyb_emin: dict,
    temperature: float = 37.0,
    accessibility_cache_dir: Optional[str] = None,
    summary_only: bool = False,
    detailed_report: bool = False,
    in_process: bool = False,
) -> None:
    """Initialise per-worker state; called once per spawned worker process.

    The transcriptome is loaded from a pre-written Arrow IPC file (memory-mapped)
    rather than being re-parsed from the original GTF/GFF3, so all workers share
    the same physical memory pages via the OS page cache. With
    ``accessibility_dir`` the service memory-maps the uint8 cache the parent
    built and checked in ``accessibility_cache_dir`` (``None``: in
    ``accessibility_dir``) before starting the pool, so the worker does not
    check it against the Parquet again.

    With ``in_process`` the state is set up in the calling process instead, for
    a run without a pool. That process's polars and pyarrow thread pools are
    already sized, so ``polars_max_threads`` is not applied.
    """
    if not in_process:
        os.environ["POLARS_MAX_THREADS"] = str(polars_max_threads)
        # pyarrow reads the shared transcriptome IPC and the cache-building
        # Parquet, so keep its pool inside the per-worker budget.
        pa.set_cpu_count(max(1, polars_max_threads))

    global _WORKER_DF_TRANS, _WORKER_INTERSECTOR, _WORKER_PROB_SERVICE
    global _WORKER_ON_TARGET_MAP, _WORKER_SELF_HYB_EMIN, _WORKER_SUMMARY_ONLY
    global _WORKER_DETAILED_REPORT

    if ipc_path:
        # Memory-map the transcriptome the parent wrote (sorted, uncompressed,
        # classic Arrow string layout) through pyarrow: the columns stay views of
        # the shared page cache, and preload_transcriptome slices them without a
        # copy. polars 2.0's read_ipc copies the file into each worker (~0.85 GB
        # private per worker for a GENCODE GTF, against ~0.12 GB this way).
        _WORKER_DF_TRANS = _read_shared_ipc(Path(ipc_path))
        _WORKER_INTERSECTOR = IntersectionService()
        _WORKER_INTERSECTOR.preload_transcriptome(_WORKER_DF_TRANS)
    else:
        # Set even when there is nothing to load: in-process, an earlier run's
        # transcriptome may still be here.
        _WORKER_DF_TRANS = None
        _WORKER_INTERSECTOR = None

    acc_service = None
    if accessibility_dir:
        acc_service = GenomeAccessibilityService(
            Path(accessibility_dir),
            max_cached=4,
            cache_dir=(
                Path(accessibility_cache_dir)
                if accessibility_cache_dir is not None
                else None
            ),
            verify_pack=False,
        )

    _WORKER_PROB_SERVICE = ProbabilityService(acc_service, temperature=temperature)

    # Store per-run constants so they don't need to be pickled per task.
    _WORKER_ON_TARGET_MAP = on_target_map
    _WORKER_SELF_HYB_EMIN = self_hyb_emin
    _WORKER_SUMMARY_ONLY = summary_only
    _WORKER_DETAILED_REPORT = detailed_report


def _process_one_sirna(
    file_path_str: str,
    alpha_gamma_pairs: list,
    theta_vals: list,
    sense_only: bool,
    predictions_type: str,
) -> tuple:
    """Process a single siRNA file through the full pipeline in a worker process.

    Reads worker state from the module-level globals populated by
    ``_init_worker()``.

    Returns
    -------
    tuple
        ``(pyarrow.Table, meta)``, or ``(None, {})`` if no predictions remain
        after filtering and intersection.
    """
    _m = sys.modules[__name__]

    df_trans = _m._WORKER_DF_TRANS
    intersector = _m._WORKER_INTERSECTOR
    prob_service = _m._WORKER_PROB_SERVICE
    on_target_map = _m._WORKER_ON_TARGET_MAP
    self_hyb_emin = _m._WORKER_SELF_HYB_EMIN

    _t_start = time.perf_counter()

    parser = RIsearchParser()
    df = parser.load_single_file(Path(file_path_str))
    _t_load = time.perf_counter()

    if sense_only:
        df = df.filter(pl.col("strand") == "+")
    if df.height == 0:
        return None, {}

    # Intersect with transcriptome (sequential — each worker owns its CPU).
    # intersect() already deduplicates per (chrom, strand) pair internally.
    _intersect_subtimings: dict = {}
    if df_trans is not None and intersector is not None:
        df = intersector.intersect(
            df,
            df_trans,
            mode=predictions_type,
            workers=1,
            _timings=_intersect_subtimings,
        )
        if df.height == 0:
            return None, {}
    df = _apply_self_hyb_emin(df, self_hyb_emin)
    _t_intersect = time.perf_counter()

    # Probabilities. E_min is the file's raw_e_min (full-scan minimum), or the
    # self-hybridisation value _apply_self_hyb_emin put there.
    e_min = (
        dict(df.group_by("sirna_id").agg(pl.col("raw_e_min").min()).iter_rows())
        if "raw_e_min" in df.columns
        else None
    )
    df, meta = prob_service.calculate_probabilities_per_sirna(
        df,
        alpha_gamma_pairs=alpha_gamma_pairs,
        theta_values=theta_vals,
        on_target_map=on_target_map,
        e_min=e_min,
        aggregate_only=_m._WORKER_SUMMARY_ONLY,
    )
    _t_prob = time.perf_counter()

    # Drop heavy intermediate columns before returning (only without
    # detailed_report, which keeps them)
    if not _m._WORKER_DETAILED_REPORT:
        df = _drop_heavy_columns(df)

    arrow = df.to_arrow()
    _t_end = time.perf_counter()

    # Return freed pages to OS to prevent RSS growth across sequential siRNAs
    # in the same worker. Python's allocator retains pages by default; gc +
    # malloc_trim release them, keeping per-siRNA memory cost constant.
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    meta["_timings"] = {
        "load_s": _t_load - _t_start,
        "intersect_s": _t_intersect - _t_load,
        "prob_s": _t_prob - _t_intersect,
        "serialize_s": _t_end - _t_prob,
        "total_s": _t_end - _t_start,
        **_intersect_subtimings,
    }

    return arrow, meta


class WorkerFailureError(RuntimeError):
    """Raised by the off-target generators when worker tasks failed.

    Successful siRNAs are still yielded first; the error is raised once the
    pool has drained (or immediately for a failed accessibility contig, whose
    loss would silently corrupt every siRNA). ``failures`` holds
    ``(task_name, error_message)`` pairs in completion order: in
    `compute_off_targets` a task is an input file that could not be split
    (named by its file name) or a siRNA that could not be scored (named by its
    ``sirna_id``).
    """

    def __init__(self, stage: str, failures: list[tuple[str, str]], n_total: int):
        self.stage = stage
        self.failures = failures
        self.n_total = n_total
        unit = "contigs" if stage.startswith("pass 2") else "siRNAs"
        name, msg = failures[0]
        super().__init__(
            f"{len(failures)}/{n_total} {unit} failed in {stage}; "
            f"first error: {name}: {msg}"
        )

    @property
    def n_failed(self) -> int:
        """Number of tasks that failed."""
        return len(self.failures)

    @property
    def failed_names(self) -> list[str]:
        """Names (siRNA files or contigs) of the failed tasks, in completion order."""
        return [name for name, _ in self.failures]


def _stream_sirnas(
    pool: ProcessPoolExecutor,
    all_files: list[Path],
    alpha_gamma_pairs: list,
    theta_vals: list,
    sense_only: bool,
    predictions_type: str,
) -> Generator[tuple[pl.DataFrame, dict], None, None]:
    """One task per siRNA; each worker looks accessibility up in the shared cache.

    Yields
    ------
    tuple[polars.DataFrame, dict]
        ``(df, meta)`` per siRNA, in completion order.

    Raises
    ------
    WorkerFailureError
        After the last successful siRNA, when any task failed.
    """
    failures: list[tuple[str, str]] = []
    futures = {
        pool.submit(
            _process_one_sirna,
            str(f),
            alpha_gamma_pairs,
            theta_vals,
            sense_only,
            predictions_type,
        ): f
        for f in all_files
    }

    for future in as_completed(futures):
        # Pop the future immediately so its held result (arrow table +
        # metadata) can be GC'd as soon as the loop variable rotates.
        f = futures.pop(future)
        try:
            arrow_table, batch_metadata = future.result()
        except Exception as exc:
            logger.error(f"Worker failed for {f.name}: {exc}")
            failures.append((f.name, str(exc)))
            continue

        if arrow_table is None:
            continue

        # arrow_table is a pyarrow Table, so from_arrow returns a DataFrame
        # (not a Series).
        df_chunk = cast(pl.DataFrame, pl.from_arrow(arrow_table))
        del arrow_table  # Arrow copy no longer needed; Polars owns the data
        yield df_chunk, batch_metadata
    if failures:
        raise WorkerFailureError("per-siRNA workers", failures, len(all_files))


def _build_alpha_gamma_pairs(alpha: str, gamma: str) -> list[tuple[float, float]]:
    """Parse semicolon-separated alpha/gamma strings into (alpha, gamma) pairs.

    Always includes the baseline (1.0, 1.0). Enforces alpha <= gamma. Deduplicates.

    Returns
    -------
    list[tuple[float, float]]
        Ordered, deduplicated ``(alpha, gamma)`` pairs starting with ``(1.0, 1.0)``.
    """
    alpha_vals = [float(x) for x in alpha.split(";") if x.strip()]
    gamma_vals = [float(x) for x in gamma.split(";") if x.strip()]
    pairs: list[tuple[float, float]] = [(1.0, 1.0)]
    for a in alpha_vals:
        for g in gamma_vals:
            if a == 1.0 and g == 1.0:
                continue
            if a <= g:
                pairs.append((a, g))
    return list(dict.fromkeys(pairs))


def _parse_theta(theta: str) -> list[float]:
    """Parse a semicolon-separated theta string into a list of floats.

    Returns
    -------
    list[float]
        One value per non-empty ``;``-separated field; empty for ``""``.
    """
    return [float(x) for x in theta.split(";") if x.strip()]


def _downcast_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Downcast columns to minimize Arrow IPC file size.

    - Coordinates: start/end → UInt32 (covers positions up to 4.3B)
    - Strings: sirna_id, transcript_id, chrom, strand, gene_id → Categorical
    - Floats: energy, opening_energy, dG_total, P_off_target, exp_value → Float32

    Complexity: O(n) time, O(1) extra space.

    Returns
    -------
    polars.DataFrame
        The same frame with the listed columns cast to narrower dtypes.
    """
    casts = {}
    for col in ["start", "end"]:
        if col in df.columns:
            casts[col] = pl.UInt32
    for col in ["sirna_id", "transcript_id", "chrom", "strand", "gene_id"]:
        if col in df.columns:
            casts[col] = pl.Categorical
    for col in [
        "energy",
        "opening_energy",
        "dG_total",
        "P_off_target",
        "exp_value",
    ]:
        if col in df.columns:
            casts[col] = pl.Float32
    if casts:
        df = df.cast(casts)
    return df


# ---------------------------------------------------------------------------
# Single-file / inline-RIsearch core
# ---------------------------------------------------------------------------
def _coerce_predictions(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate an in-memory predictions frame and cast it to RISEARCH_SCHEMA.

    Extra columns are dropped and compatible dtypes (Int64 coordinates, Float64
    energies, Categorical strings) are cast, so a frame that went through Parquet
    or a user's own manipulation still works.

    Returns
    -------
    polars.DataFrame
        ``frame`` restricted to ``RISEARCH_COLUMNS`` and cast to ``RISEARCH_SCHEMA``.

    Raises
    ------
    ValueError
        When one or more of ``RISEARCH_COLUMNS`` is missing; the message names them.
    """
    missing = [c for c in RISEARCH_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"predictions is missing column(s) {', '.join(missing)}; expected the "
            f"sioff.search schema {RISEARCH_COLUMNS}"
        )
    return frame.select(RISEARCH_COLUMNS).cast(pl.Schema(RISEARCH_SCHEMA))


def compute_off_targets_single(
    *,
    predictions: Optional[pl.DataFrame] = None,
    risearch_file: Optional[Path] = None,
    sirna_fasta: Optional[Path] = None,
    target_fasta: Optional[Path] = None,
    target_index: Optional[Path] = None,
    gtf_file: Optional[Path] = None,
    feature_type: str = "exon",
    expression_metric: str = "RPKM",
    transcriptome_format: str = "auto",
    accessibility: Optional[Mapping[str, pl.DataFrame]] = None,
    accessibility_dir: Optional[Path] = None,
    accessibility_cache_dir: Optional[Path] = None,
    genome_file: Optional[Path] = None,
    window_size: int = 80,
    max_span: int = 40,
    unpaired_prob: int = 30,
    temperature: float = 37.0,
    matrix: Union[str, Path] = "t04",
    on_target_ids_file: Optional[Path] = None,
    alpha: str = "1.0",
    gamma: str = "1.0",
    theta: str = "",
    sense_only: bool = False,
    predictions_type: str = "gw",
    detailed_report: bool = False,
    n_workers: int = 1,
    profiler: Optional[PipelineProfiler] = None,
    accessibility_progress_callback=None,
) -> tuple[pl.DataFrame, dict]:
    """Compute off-target probabilities for one set of RIsearch predictions.

    The predictions come from exactly one of: ``predictions`` (an in-memory
    DataFrame in the ``sioff.search`` schema -- ``sirna_id, chrom, start, end,
    strand, energy``), ``risearch_file`` (a RIsearch2 output file), or
    ``sirna_fasta`` + ``target_fasta`` (run RIsearch in-process; ``sirna_fasta``
    is validated for duplicate IDs first and ``target_fasta`` is indexed unless
    ``target_index`` is given).

    Accessibility profiles likewise come from at most one of ``accessibility``
    (the ``dict[chrom -> DataFrame]`` that [`sioff.accessibility`][sioff.accessibility] returns),
    ``accessibility_dir`` (per-chromosome Parquet files) or ``genome_file``
    (fold on the fly into a temporary directory that lives only for the
    probability computation). With none of them, opening
    energies are treated as zero.

    When ``gtf_file`` is given, the predictions are intersected with the
    annotation before probabilities are computed.

    Writes no files and prints nothing.

    Parameters
    ----------
    predictions : polars.DataFrame, optional
        In-memory predictions, typically the value returned by
        [`sioff.search`][sioff.search]. Must contain the columns ``sirna_id, chrom, start,
        end, strand, energy``; extra columns are dropped and compatible dtypes
        are cast to the RIsearch schema. Mutually exclusive with
        ``risearch_file``.
    risearch_file : pathlib.Path, optional
        Pre-computed RIsearch2 output file (CLI ``-r/--risearch-file``).
        Mutually exclusive with ``predictions``.
    sirna_fasta : pathlib.Path, optional
        siRNA FASTA with one or more sequences (CLI ``-s/--sirna-fasta``). Used
        as the RIsearch query when neither ``predictions`` nor
        ``risearch_file`` is given; requires ``target_fasta``. Duplicate
        sequence IDs are rejected.
    target_fasta : pathlib.Path, optional
        Target FASTA (genome or transcriptome) for the in-process RIsearch run
        (CLI ``--target-fasta``/``--genome``).
    target_index : pathlib.Path, optional
        Pre-built RIsearch index for ``target_fasta`` (CLI
        ``-idx/--target-index``); speeds up repeated runs. When omitted the
        index is built (or reused) next to ``target_fasta``.
    gtf_file : pathlib.Path, optional
        Transcriptome annotation, GTF or GFF3 (CLI ``-t/--transcriptome``).
        When given, predictions are intersected with its features and annotated
        with ``transcript_id``, ``gene_id`` and ``exp_value``.
    feature_type : str, default "exon"
        Feature type to select from a GTF/GFF3 annotation (CLI ``--feature``).
    expression_metric : str, default "RPKM"
        Annotation attribute used as the expression score (CLI
        ``--expression-metric``).
    transcriptome_format : str, default "auto"
        Annotation format: ``"auto"``, ``"gtf"`` or ``"gff3"``; BED is rejected
        (CLI ``--transcriptome-format``). ``"auto"`` detects the
        format from the file, distinguishing GTF from GFF3 by the attribute
        column.
    accessibility : Mapping[str, polars.DataFrame], optional
        In-memory accessibility profiles, ``dict[chrom -> DataFrame]`` with the
        columns ``position, strand, u1..uN`` and both strands stacked, as
        returned by [`sioff.accessibility`][sioff.accessibility]. Mutually exclusive with
        ``accessibility_dir``.
    accessibility_dir : pathlib.Path, optional
        The npy cache from ``sioff accessibility`` or per-chromosome Parquet
        files (CLI ``-a/--accessibility-dir``); for Parquet the cache of every
        contig is built (or rebuilt, when older than its Parquet) in
        ``accessibility_cache_dir`` before the predictions are read. Mutually
        exclusive with ``accessibility``.
    accessibility_cache_dir : pathlib.Path, optional
        Where the cache is read from and built; default ``accessibility_dir``
        (CLI ``--accessibility-cache-dir``). Created if absent.
    genome_file : pathlib.Path, optional
        Genome FASTA to fold on the fly when no pre-computed profiles are
        supplied (CLI ``-f/--fasta``). Profiles are written to a temporary
        directory that is removed before the function returns.
    window_size : int, default 80
        RNAplfold window size ``W`` for on-the-fly folding (CLI ``-W/--window``).
    max_span : int, default 40
        Maximum base-pair span ``L`` for on-the-fly folding (CLI ``-L/--span``).
    unpaired_prob : int, default 30
        Maximum unpaired-stretch length ``u`` for on-the-fly folding (CLI
        ``-u/--unpaired``); the profiles hold columns ``u1..u{unpaired_prob}``.
    temperature : float, default 37.0
        Folding temperature in degrees Celsius (CLI ``-T/--temperature``).
        Affects both on-the-fly accessibility and the partition function.
    matrix : str or pathlib.Path, default "t04"
        Energy parameter set for every RIsearch call made here (in-process
        search and the self-hybridisation E_min of ``sirna_fasta``): a bundled
        id or the path of a DSM TSV table (CLI ``-z/--matrix``). Use the set the
        predictions were searched with: E_min is the reference energy of the
        alpha/gamma clamp.
    on_target_ids_file : pathlib.Path, optional
        Two-column TSV mapping ``sirna_id`` to on-target ``transcript_id``
        (CLI ``-oi/--on-target-ids``), used by the per-siRNA model to flag and
        weight each siRNA's own target.
    alpha : str, default "1.0"
        Alpha clamping parameter(s) as a ``";"``-separated string, e.g.
        ``"0.8;1.0"`` (CLI ``--alpha``). Every ``alpha`` is paired with every
        ``gamma``; pairs with ``alpha > gamma`` are dropped, duplicates are
        removed and the baseline ``(1.0, 1.0)`` is always included. Each
        non-baseline pair adds ``:alpha=A,gamma=G``-suffixed columns.
    gamma : str, default "1.0"
        Gamma clamping parameter(s), same format as ``alpha`` (CLI ``--gamma``).
    theta : str, default ""
        Theta scaling parameter(s) as a ``";"``-separated string, e.g.
        ``"0.5;0.7"`` (CLI ``--theta``). Empty means no theta scaling. Each
        value adds ``:theta=T``-suffixed columns.
    sense_only : bool, default False
        Keep only sense-strand (``+``) predictions (CLI ``--sense-only``).
    predictions_type : str, default "gw"
        ``"gw"`` for genome-wide predictions (interval intersection with the
        annotation) or ``"tw"`` for transcriptome-wide predictions (join on
        transcript ID) (CLI ``--type``).
    detailed_report : bool, default False
        Keep the raw ``boltzmann_weight*``, ``Z_sirna*`` and ``E_min`` columns
        for every interaction in the returned table; by default they are
        dropped (CLI ``--detailed-report``).
    n_workers : int, default 1
        Number of parallel threads for the genome-wide intersection (CLI
        ``-j/--workers``).
    profiler : PipelineProfiler, optional
        `sioff.services.profiling.PipelineProfiler` that records wall
        time and memory per stage (CLI ``--profile``). A disabled profiler is
        used when omitted.
    accessibility_progress_callback : callable, optional
        Called as ``callback(advance=n, description=str)`` after each batch
        of ``n`` records is folded on the fly; only used with ``genome_file``.

    Returns
    -------
    tuple[polars.DataFrame, dict]
        ``(df, meta)``. ``df`` holds the input columns ``sirna_id, chrom, start,
        end, strand, energy``, plus ``trans_start, trans_end, gene_id,
        transcript_id, exp_value`` when intersected with ``gtf_file``, plus the
        probability columns ``opening_energy``, ``dG_total`` and
        ``P_off_target``, plus ``is_on_target``. ``dG_total``, ``energy`` and
        ``P_off_target`` repeat with a ``:alpha=A,gamma=G`` or ``:theta=T``
        suffix for each extra parameter set. With ``detailed_report`` the
        table also holds ``E_min``, ``boltzmann_weight*`` and ``Z_sirna*``
        (each repeated per parameter set). ``meta`` always contains
        ``"_report"`` (a dict with ``n_loaded``, ``energy_min``,
        ``energy_max``, ``sense_only``, ``n_features``, ``n_intersected`` and
        ``preview``, the raw prediction head), plus ``"n_sirnas"``,
        ``"z_per_sirna"``, ``"on_target_weights"`` and ``"on_target_count"``.

    Raises
    ------
    AccessibilityError
        When ``accessibility_dir`` (and ``accessibility_cache_dir``) holds no
        profile, or the cache cannot be written.
    ValueError
        When both ``predictions`` and ``risearch_file`` are given; when both
        ``accessibility`` and ``accessibility_dir`` are given; when none of
        ``predictions``, ``risearch_file`` or ``sirna_fasta`` is given; when
        ``sirna_fasta`` is given without ``target_fasta``; when
        ``on_target_ids_file`` cannot be parsed. Also propagated when
        ``predictions`` lacks a required column or ``sirna_fasta`` contains
        duplicate IDs.

    See Also
    --------
    sioff.off_targets : Public wrapper accepting ``str`` paths and dispatching
        directories to [`compute_off_targets_directory`][sioff.core.off_targets.compute_off_targets_directory].
    sioff.accessibility : Produces the in-memory ``accessibility`` mapping.
    sioff.search : Produces the in-memory ``predictions`` frame.
    compute_off_targets_directory : Directory-of-files variant.
    """
    profiler = profiler if profiler is not None else PipelineProfiler(enabled=False)
    risearch_parser = RIsearchParser()

    if predictions is not None and risearch_file is not None:
        raise ValueError(
            "predictions and risearch_file are two sources for the same input; "
            "pass one of them"
        )
    if accessibility is not None and accessibility_dir is not None:
        raise ValueError(
            "accessibility and accessibility_dir are two sources for the same "
            "input; pass one of them"
        )
    is_running_risearch = (
        predictions is None and risearch_file is None and sirna_fasta is not None
    )
    if predictions is None and risearch_file is None and sirna_fasta is None:
        raise ValueError(
            "Must provide predictions (a DataFrame), risearch_file (a file) or "
            "sirna_fasta"
        )
    if is_running_risearch and target_fasta is None:
        raise ValueError(
            "sirna_fasta requires target_fasta when running RIsearch dynamically"
        )

    # Check the profiles and build the uint8 cache before the predictions are
    # loaded, searched or intersected, so a bad -a fails at once.
    cache_dir: Optional[Path] = None
    if accessibility_dir:
        cache_dir = _prepare_accessibility_cache(
            Path(accessibility_dir), accessibility_cache_dir, n_workers
        )

    # --- Acquire predictions ---
    if predictions is not None:
        # Mode 0: in-memory DataFrame, typically the output of sioff.search
        df = _coerce_predictions(predictions)
    elif is_running_risearch:
        # Mode 1: integrated RIsearch execution
        # Guaranteed by is_running_risearch + the target_fasta check above.
        assert sirna_fasta is not None and target_fasta is not None
        risearch_service = RIsearchService()
        risearch_service.validate_sirna_fasta(sirna_fasta)  # raises ValueError on dups
        index_path = (
            target_index
            if target_index is not None
            else risearch_service.index_target(target_fasta)
        )
        df = risearch_service.run_search(
            query_path=sirna_fasta,
            index_path=index_path,
            target_fasta=target_fasta,
            matrix=matrix,
        )
    else:
        # Mode 2: pre-computed RIsearch file
        # In this branch risearch_file is non-None (else the checks above raised).
        assert risearch_file is not None
        with profiler.stage("Load predictions") as _s:
            df = risearch_parser.load(Path(risearch_file))
            _s.rows_out = df.height

    logger.info(
        f"Loaded {df.height:,} predictions for {df['sirna_id'].n_unique()} siRNAs"
    )

    if sense_only:
        df = df.filter(pl.col("strand") == "+")

    # Diagnostics the CLI prints (loaded count / energy range / intersection
    # counts). Collected here because the wrapper no longer holds these frames.
    _summary_ris = risearch_parser.summary(df)
    report: dict = {
        "n_loaded": _summary_ris["row_count"],
        "energy_min": _summary_ris["energy_min"],
        "energy_max": _summary_ris["energy_max"],
        "sense_only": sense_only,
        "n_features": None,
        "n_intersected": None,
        # Raw predictions head (6 narrow columns) for the CLI's --verbose preview;
        # the final frame has too many columns to render without truncation.
        "preview": df.head(5),
    }

    # --- Transcriptome + intersection ---
    if gtf_file:
        trans_parser = AnnotationParser()
        with profiler.stage("Load transcriptome") as _s:
            df_trans = trans_parser.load_gtf(
                Path(gtf_file),
                feature=feature_type,
                score_col=expression_metric,
                format=transcriptome_format,
            )
            df_trans = _drop_unexpressed(df_trans)
            _s.rows_out = df_trans.height
        logger.info(
            f"Transcriptome: {df_trans.height:,} rows, {df_trans['gene_id'].n_unique()} genes"
        )
        with profiler.stage("Intersection", rows_in=df.height) as _s:
            intersector = IntersectionService()
            df = intersector.intersect(
                df, df_trans, mode=predictions_type, workers=n_workers
            )
            _s.rows_out = df.height
        report["n_features"] = df_trans.height
        report["n_intersected"] = df.height
        logger.info(f"After intersection: {df.height:,} rows")

    alpha_gamma_pairs = _build_alpha_gamma_pairs(alpha, gamma)
    theta_vals = _parse_theta(theta)

    on_target_map: dict[str, str] = {}
    if on_target_ids_file is not None:
        try:
            mapping_df = pl.read_csv(
                on_target_ids_file, separator="\t", has_header=False
            )
            on_target_map = dict(
                zip(
                    mapping_df.to_series(0),
                    mapping_df.to_series(1),
                )
            )
        except Exception as e:
            raise ValueError(f"Error parsing on-target mapping file: {e}") from e

    def _finish(prob_service: ProbabilityService, frame: pl.DataFrame):
        with profiler.stage("Probabilities (per-siRNA)", rows_in=frame.height) as _s:
            frame, meta = prob_service.calculate_probabilities_per_sirna(
                frame,
                alpha_gamma_pairs=alpha_gamma_pairs,
                theta_values=theta_vals,
                on_target_map=on_target_map if on_target_map else None,
            )
            _s.rows_out = frame.height
        logger.info(f"After probabilities: {frame.height:,} rows")

        if not detailed_report:
            frame = _drop_heavy_columns(frame)
        meta["_report"] = report
        return frame, meta

    # --- Accessibility service selection, then probabilities ---
    if accessibility is not None:
        acc_service = GenomeAccessibilityService.from_frames(
            accessibility, max_cached=4
        )
        prob_service = ProbabilityService(acc_service, temperature=temperature)
        return _finish(prob_service, df)
    elif accessibility_dir:
        # _prepare_accessibility_cache checked (or rebuilt) the pack above.
        acc_service = GenomeAccessibilityService(
            Path(accessibility_dir),
            max_cached=4,
            cache_dir=cache_dir,
            verify_pack=False,
        )
        prob_service = ProbabilityService(acc_service, temperature=temperature)
        return _finish(prob_service, df)
    elif genome_file:
        # Compute accessibility on-the-fly into a temp dir; the dir must outlive the
        # probability computation (it reads the profiles), so do it
        # inside the context manager before it is torn down.
        with tempfile.TemporaryDirectory(prefix="risearch_accessibility_") as temp_dir:
            acc_service = GenomeAccessibilityService(Path(temp_dir), max_cached=4)
            acc_service.compute_genome_accessibility(
                Path(genome_file),
                window_size=window_size,
                max_span=max_span,
                unpaired_prob=unpaired_prob,
                progress_callback=accessibility_progress_callback,
                temperature=temperature,
                output_format="npy",
            )
            prob_service = ProbabilityService(acc_service, temperature=temperature)
            return _finish(prob_service, df)
    else:
        return _finish(ProbabilityService(None, temperature=temperature), df)


# ---------------------------------------------------------------------------
# Directory core (generator, one DataFrame per siRNA)
# ---------------------------------------------------------------------------
def compute_off_targets_directory(
    *,
    input_dir: Path,
    sirna_fasta: Optional[Path] = None,
    gtf_file: Optional[Path] = None,
    feature_type: str = "exon",
    expression_metric: str = "RPKM",
    transcriptome_format: str = "auto",
    accessibility_dir: Optional[Path] = None,
    accessibility_cache_dir: Optional[Path] = None,
    temperature: float = 37.0,
    matrix: Union[str, Path] = "t04",
    on_target_ids_file: Optional[Path] = None,
    alpha: str = "1.0",
    gamma: str = "1.0",
    theta: str = "",
    sense_only: bool = False,
    predictions_type: str = "gw",
    n_workers: int = 1,
    summary_only: bool = False,
    detailed_report: bool = False,
) -> Generator[tuple[pl.DataFrame, dict], None, None]:
    """Compute off-target probabilities for a directory of per-siRNA files.

    A generator yielding ``(df, meta)`` for each RIsearch prediction file in
    ``input_dir``, in completion order. Each file is processed in a spawned
    worker process (``spawn`` rather than ``fork`` avoids Rayon/Polars
    thread-pool deadlocks): it is loaded, optionally filtered to the sense
    strand, intersected with the transcriptome, and scored with the per-siRNA
    partition function. Files with no remaining predictions are skipped
    silently. Files whose worker fails are logged and skipped, and once every
    successful siRNA has been yielded a `WorkerFailureError` summarising them
    is raised, so a run where tasks failed never looks like an empty success.
    Heavy intermediate columns (``boltzmann_weight*``, ``Z_sirna*``, ``E_min``)
    are dropped from the yielded frames.

    The process pool and the temporary Arrow-IPC copy of the transcriptome
    (memory-mapped by every worker) are held open across yields and cleaned up
    in a ``finally``. Consume the generator fully, call ``.close()`` on it, or
    wrap it in `contextlib.closing` for deterministic teardown.

    Accessibility profiles are read from ``accessibility_dir`` only; in-memory
    profiles and on-the-fly folding are not supported in this mode. Writes no
    files other than the temporary IPC copy and the accessibility cache, and
    prints nothing.

    Parameters
    ----------
    input_dir : pathlib.Path
        Directory of RIsearch2 output files, one per siRNA (CLI
        ``-r/--risearch-file`` pointing at a directory).
    sirna_fasta : pathlib.Path, optional
        siRNA FASTA (CLI ``-s/--sirna-fasta``). When given, a self-hybridisation
        ``E_min`` is computed for every siRNA; the clamp's ``E_min`` is the lower
        of it and the strongest expressed site (the legacy ``E_min`` semantics).
    gtf_file : pathlib.Path, optional
        Transcriptome annotation, GTF or GFF3 (CLI ``-t/--transcriptome``).
        Loaded once, serialised to a temporary Arrow IPC file and memory-mapped
        by each worker for the intersection.
    feature_type : str, default "exon"
        Feature type to select from a GTF/GFF3 annotation (CLI ``--feature``).
    expression_metric : str, default "RPKM"
        Annotation attribute used as the expression score (CLI
        ``--expression-metric``).
    transcriptome_format : str, default "auto"
        Annotation format: ``"auto"``, ``"gtf"`` or ``"gff3"``; BED is rejected
        (CLI ``--transcriptome-format``).
    accessibility_dir : pathlib.Path, optional
        The npy cache from ``sioff accessibility`` or per-chromosome Parquet
        files (CLI ``-a/--accessibility-dir``); for Parquet the cache is built
        (or rebuilt, when older than its Parquet) in ``accessibility_cache_dir``
        first, before the transcriptome is parsed or the pool starts. Without
        it, opening energies are treated as zero.
    accessibility_cache_dir : pathlib.Path, optional
        Where the cache is read from and built; default ``accessibility_dir``
        (CLI ``--accessibility-cache-dir``); created if absent. Every contig
        that has a Parquet but no current cache is built here, in the parent
        (``min(n_workers, 4)`` build processes, one strand's cache in memory
        each), and each worker then memory-maps the finished files.
    temperature : float, default 37.0
        Temperature in degrees Celsius for the partition function (CLI
        ``-T/--temperature``).
    matrix : str or pathlib.Path, default "t04"
        Energy parameter set for every RIsearch call made here (in-process
        search and the self-hybridisation E_min of ``sirna_fasta``): a bundled
        id or the path of a DSM TSV table (CLI ``-z/--matrix``). Use the set the
        predictions were searched with: E_min is the reference energy of the
        alpha/gamma clamp.
    on_target_ids_file : pathlib.Path, optional
        Two-column TSV mapping ``sirna_id`` to on-target ``transcript_id``
        (CLI ``-oi/--on-target-ids``); lines starting with ``#`` are ignored.
    alpha : str, default "1.0"
        Alpha clamping parameter(s) as a ``";"``-separated string, e.g.
        ``"0.8;1.0"`` (CLI ``--alpha``). Every ``alpha`` is paired with every
        ``gamma``; pairs with ``alpha > gamma`` are dropped, duplicates are
        removed and the baseline ``(1.0, 1.0)`` is always included.
    gamma : str, default "1.0"
        Gamma clamping parameter(s), same format as ``alpha`` (CLI ``--gamma``).
    theta : str, default ""
        Theta scaling parameter(s) as a ``";"``-separated string, e.g.
        ``"0.5;0.7"`` (CLI ``--theta``). Empty means no theta scaling.
    sense_only : bool, default False
        Keep only sense-strand (``+``) predictions (CLI ``--sense-only``).
    predictions_type : str, default "gw"
        ``"gw"`` for genome-wide predictions (interval intersection with the
        annotation) or ``"tw"`` for transcriptome-wide predictions (join on
        transcript ID) (CLI ``--type``).
    n_workers : int, default 1
        Number of worker processes (CLI ``-j/--workers``), capped at the number
        of input files. Polars threads per worker are set to
        ``n_workers // n_processes`` so the total stays within the budget.
    summary_only : bool, default False
        Only the per-siRNA sums are needed (CLI ``--summary-only``): workers
        score with ``aggregate_only`` and ``df`` is empty, so no per-site table
        is built or sent back to this process. ``meta`` is unchanged and
        carries ``"n_rows"``, the number of scored site x transcript rows.
    detailed_report : bool, default False
        Keep the raw ``boltzmann_weight*``, ``Z_sirna*`` and ``E_min`` columns
        in each ``df`` (CLI ``--detailed-report``). Has no effect with
        ``summary_only``, whose ``df`` is empty.

    Yields
    ------
    tuple[polars.DataFrame, dict]
        ``(df, meta)`` for one siRNA. ``df`` holds the input columns
        ``sirna_id, chrom, start, end, strand, energy``, the annotation columns
        ``trans_start, trans_end, gene_id, transcript_id, exp_value`` when
        ``gtf_file`` is given, ``opening_energy``, ``dG_total``,
        ``is_on_target`` and ``P_off_target``, plus ``dG_total``, ``energy`` and
        ``P_off_target`` repeated with a ``:alpha=A,gamma=G`` or ``:theta=T``
        suffix for each extra parameter set (empty with ``summary_only``). With
        ``detailed_report`` it also holds ``E_min``, ``boltzmann_weight*`` and
        ``Z_sirna*``.
        ``meta`` contains ``"n_sirnas"``, ``"n_rows"``,
        ``"z_per_sirna"``, ``"on_target_weights"``, ``"on_target_count"`` and
        ``"_timings"`` (per-stage wall times for the worker: ``load_s``,
        ``intersect_s``, ``prob_s``, ``serialize_s``, ``total_s`` and the
        intersection sub-timings).

    Raises
    ------
    FileNotFoundError
        When ``input_dir`` contains no RIsearch prediction files.
    WorkerFailureError
        After the last successful siRNA, when any worker task failed.
    AccessibilityError
        When the accessibility cache cannot be built (unreadable Parquet,
        unwritable cache directory) or ``accessibility_dir`` holds no profile.

    See Also
    --------
    sioff.off_targets : Public wrapper; dispatches here when ``risearch_file``
        is a directory.
    compute_off_targets_single : Single-source variant with the full option set.
    """
    input_dir = Path(input_dir)
    risearch_parser = RIsearchParser()
    all_files = risearch_parser.list_directory_files(input_dir)
    if not all_files:
        raise FileNotFoundError(f"No RIsearch files found in {input_dir}")
    # Check the profiles and build the uint8 cache first: a bad -a then fails
    # before the GTF parse and self-hybridisation, and before the IPC temp dir
    # exists. Workers only ever memory-map the finished files.
    cache_dir: Optional[Path] = None
    if accessibility_dir is not None:
        cache_dir = _prepare_accessibility_cache(
            Path(accessibility_dir), accessibility_cache_dir, n_workers
        )

    on_target_map: dict[str, str] = {}
    if on_target_ids_file is not None:
        with open(on_target_ids_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        on_target_map[parts[0]] = parts[1]

    # Self-hybridisation E_min per siRNA (matches the legacy E_min semantics).
    self_hyb_emin: dict[str, float] = {}
    if sirna_fasta is not None:
        self_hyb_emin = RIsearchService().self_hybridization_emin_batch(
            sirna_fasta, matrix=matrix
        )

    alpha_gamma_pairs = _build_alpha_gamma_pairs(alpha, gamma)
    theta_vals = _parse_theta(theta)

    n_proc = min(n_workers, len(all_files))
    # Polars threads per worker: keeps N_workers × threads ≤ available cores.
    polars_threads_per_worker = max(1, n_workers // n_proc)

    # spawn: each worker starts a fresh interpreter, avoiding fork+Rayon deadlocks.
    _ctx = multiprocessing.get_context("spawn")

    # Serialize transcriptome to a temp Arrow IPC file so workers can memory-map
    # it instead of re-parsing the GTF/GFF3. Workers share pages via the OS cache.
    # Written last before the pool: only the try/finally around the pool removes
    # it, so nothing that can fail may run in between.
    _ipc_tmp: Optional[str] = None
    _ipc_path: Optional[Path] = None
    if gtf_file:
        trans_parser = AnnotationParser()
        df_trans = trans_parser.load_gtf(
            Path(gtf_file),
            feature=feature_type,
            score_col=expression_metric,
            format=transcriptome_format,
        )
        df_trans = _drop_unexpressed(df_trans)
        # Sorted by (chrom, strand, start) so each worker's partitions are
        # zero-copy slices of the memory-mapped file, and uncompressed so the
        # file can be memory-mapped at all; see preload_transcriptome.
        df_trans = sort_for_preload(df_trans)
        _ipc_tmp = tempfile.mkdtemp(prefix="risearch_ipc_")
        _ipc_path = Path(_ipc_tmp) / "transcriptome.arrow"
        try:
            df_trans.write_ipc(
                _ipc_path,
                compression="uncompressed",
                compat_level=pl.CompatLevel.oldest(),
            )
        except BaseException:
            shutil.rmtree(_ipc_tmp, ignore_errors=True)
            raise

    # Polars sizes its thread pool when it is imported, which in a spawned
    # worker happens while the initializer is being unpickled -- before the
    # initializer runs. The budget therefore has to be in the environment the
    # children inherit. The parent's own pool is already sized, so this does
    # not affect it; restored on exit.
    _prev_polars_threads = os.environ.get("POLARS_MAX_THREADS")
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads_per_worker)
    _init_args = (
        str(_ipc_path) if gtf_file else "",
        str(accessibility_dir) if accessibility_dir else "",
        polars_threads_per_worker,
        on_target_map,
        self_hyb_emin,
        temperature,
        str(cache_dir) if cache_dir is not None else None,
        summary_only,
        detailed_report,
        False,  # in_process
    )

    try:
        with ProcessPoolExecutor(
            max_workers=n_proc,
            mp_context=_ctx,
            initializer=_init_worker,
            initargs=_init_args,
        ) as pool:
            yield from _stream_sirnas(
                pool,
                all_files,
                alpha_gamma_pairs,
                theta_vals,
                sense_only,
                predictions_type,
            )
    finally:
        # Clean up the temp Arrow IPC dir used to share the transcriptome.
        if _ipc_tmp:
            shutil.rmtree(_ipc_tmp, ignore_errors=True)
        if _prev_polars_threads is None:
            os.environ.pop("POLARS_MAX_THREADS", None)
        else:
            os.environ["POLARS_MAX_THREADS"] = _prev_polars_threads


# ---------------------------------------------------------------------------
# The one engine: every input split into per-siRNA parts, one task per siRNA
# ---------------------------------------------------------------------------
_NO_TRANSCRIPTOME_WARNING = (
    "--sirna-table on-target IDs are matched after intersection; "
    "without -t/--transcriptome no row can be on-target"
)


def _score_part(
    sirna_id: str,
    part_path: str,
    perfect_match: Optional[float],
    alpha_gamma_pairs: list,
    theta_vals: list,
    sense_only: bool,
    predictions_type: str,
) -> tuple:
    """Score one siRNA's split part with the state `_init_worker` set up.

    The part is loaded, filtered to the sense strand with ``sense_only`` and
    intersected with the transcriptome. With a ``perfect_match`` energy, E_min
    is the lower of it and the strongest hit left (the legacy rule); without
    one, E_min is the minimum over the hits left.

    Returns
    -------
    tuple
        ``(pyarrow.Table, meta)``, or ``(None, {})`` if no predictions remain
        after filtering and intersection.
    """
    _m = sys.modules[__name__]
    df_trans = _m._WORKER_DF_TRANS
    intersector = _m._WORKER_INTERSECTOR
    prob_service = _m._WORKER_PROB_SERVICE

    _t_start = time.perf_counter()
    df = pl.read_parquet(part_path)
    _t_load = time.perf_counter()

    if sense_only:
        df = df.filter(pl.col("strand") == "+")
    if df.height == 0:
        return None, {}

    _intersect_subtimings: dict = {}
    if df_trans is not None and intersector is not None:
        df = intersector.intersect(
            df,
            df_trans,
            mode=predictions_type,
            workers=1,
            _timings=_intersect_subtimings,
        )
        if df.height == 0:
            return None, {}
    e_min = None
    if perfect_match is not None:
        e_min = {sirna_id: min(perfect_match, float(cast(float, df["energy"].min())))}
    _t_intersect = time.perf_counter()

    df, meta = prob_service.calculate_probabilities_per_sirna(
        df,
        alpha_gamma_pairs=alpha_gamma_pairs,
        theta_values=theta_vals,
        on_target_map=_m._WORKER_ON_TARGET_MAP,
        e_min=e_min,
        aggregate_only=_m._WORKER_SUMMARY_ONLY,
    )
    _t_prob = time.perf_counter()

    if not _m._WORKER_DETAILED_REPORT:
        df = _drop_heavy_columns(df)
    arrow = df.to_arrow()
    _t_end = time.perf_counter()

    # Return freed pages to the OS so memory stays flat across the siRNAs a
    # worker scores in turn (see _process_one_sirna).
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    meta["_timings"] = {
        "load_s": _t_load - _t_start,
        "intersect_s": _t_intersect - _t_load,
        "prob_s": _t_prob - _t_intersect,
        "serialize_s": _t_end - _t_prob,
        "total_s": _t_end - _t_start,
        **_intersect_subtimings,
    }
    return arrow, meta


class _Executor:
    """Run the engine's tasks in one spawned pool, or in this process.

    The first phase in which more than one task can run at once
    (``min(n_workers, n_tasks) > 1``) starts the pool, and the later phase
    reuses it; processes are spawned as tasks need them, up to ``n_workers``.
    Until then tasks run in this process, where `_init_worker`
    (``in_process=True``) sets the worker state up on first use. This run's
    copy of that state is put back before each task, so two runs consumed in
    turn in one process never see each other's.

    Either way, a task's exception is logged and collected in ``failures``
    instead of raised.
    """

    def __init__(self, n_workers: int, init_kwargs: dict[str, Any]) -> None:
        self.n_workers = max(1, n_workers)
        self.failures: list[tuple[str, str]] = []
        self._init_kwargs = init_kwargs
        self._pool: Optional[ProcessPoolExecutor] = None
        self._state: Optional[dict[str, Any]] = None
        # Set before the first _init_worker call: one that fails part-way has
        # still set some of the globals, and close() must reset them too.
        self._init_attempted = False
        self._env_saved = False
        self._prev_polars_threads: Optional[str] = None

    def run(
        self,
        fn: Callable[..., Any],
        tasks: Mapping[str, tuple],
        *,
        threads_per_worker: int,
        uses_worker_state: bool,
    ) -> Iterator[tuple[str, Any]]:
        """Run ``fn(*args)`` for each named task.

        Parameters
        ----------
        fn : callable
            A top-level function, so a pool can pickle it by reference.
        tasks : Mapping[str, tuple]
            ``{task name: args}``.
        threads_per_worker : int
            Polars threads per worker process, if this call starts the pool.
        uses_worker_state : bool
            ``fn`` reads the state `_init_worker` sets up; in-process, it is
            only set up for such a task.

        Yields
        ------
        tuple[str, Any]
            ``(task name, result)`` for each task that succeeded, in
            completion order.
        """
        if self._pool is None and min(self.n_workers, len(tasks)) > 1:
            self._start_pool(threads_per_worker)
        pool = self._pool
        if pool is not None:
            futures = {}
            for name, args in tasks.items():
                try:
                    futures[pool.submit(fn, *args)] = name
                except BrokenProcessPool as exc:
                    # A worker died in an earlier phase (say, killed for
                    # memory): its tasks failed, and so do these.
                    self._fail(name, exc)
            for future in as_completed(futures):
                # Pop at once so a held result can be freed as the loop moves on.
                name = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    self._fail(name, exc)
                    continue
                yield name, result
        else:
            for name, args in tasks.items():
                try:
                    if uses_worker_state:
                        self._enter_worker_state()
                    result = fn(*args)
                except Exception as exc:
                    self._fail(name, exc)
                    continue
                yield name, result

    def close(self) -> None:
        """Shut the pool down and drop the in-process worker state."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._env_saved:
            if self._prev_polars_threads is None:
                os.environ.pop("POLARS_MAX_THREADS", None)
            else:
                os.environ["POLARS_MAX_THREADS"] = self._prev_polars_threads
            self._env_saved = False
        if self._init_attempted:
            # Copies, so the defaults' own dicts never become live state.
            globals().update(
                {name: copy.copy(v) for name, v in _WORKER_STATE_DEFAULTS.items()}
            )
            self._state = None
            self._init_attempted = False

    def _start_pool(self, threads_per_worker: int) -> None:
        # Polars sizes its thread pool when it is imported, which in a spawned
        # worker happens while the initializer is being unpickled -- before the
        # initializer runs. The budget therefore has to be in the environment
        # the children inherit. The parent's own pool is already sized, so this
        # does not affect it; restored in close().
        self._prev_polars_threads = os.environ.get("POLARS_MAX_THREADS")
        self._env_saved = True
        os.environ["POLARS_MAX_THREADS"] = str(threads_per_worker)
        # spawn: each worker starts a fresh interpreter, avoiding fork+Rayon
        # deadlocks.
        self._pool = ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=partial(
                _init_worker, polars_max_threads=threads_per_worker, **self._init_kwargs
            ),
        )

    def _enter_worker_state(self) -> None:
        module_globals = globals()
        if self._state is None:
            self._init_attempted = True
            _init_worker(
                polars_max_threads=self.n_workers, in_process=True, **self._init_kwargs
            )
            self._state = {
                name: module_globals[name] for name in _WORKER_STATE_DEFAULTS
            }
        else:
            module_globals.update(self._state)

    def _fail(self, name: str, exc: Exception) -> None:
        logger.error(f"Worker failed for {name}: {exc}")
        self.failures.append((name, str(exc)))


def _input_files(path: Path) -> list[Path]:
    """List the prediction files behind ``-r``: the file, or the files in the folder.

    Returns
    -------
    list[pathlib.Path]
        ``[path]`` for a file; every ``*.gz``, ``*.tsv``, ``*.out`` and
        ``*.parquet`` in a directory, sorted.

    Raises
    ------
    FileNotFoundError
        When ``path`` does not exist, or is a directory without prediction
        files.
    """
    if path.is_dir():
        files = RIsearchParser().list_directory_files(path)
        if not files:
            raise FileNotFoundError(f"No RIsearch files found in {path}")
        return files
    if not path.is_file():
        raise FileNotFoundError(f"RIsearch2 output file not found: {path}")
    return [path]


def _check_ignored_arguments(
    *,
    risearch_file: Optional[Path],
    predictions: Optional[pl.DataFrame],
    target_fasta: Optional[Path],
    target_index: Optional[Path],
    accessibility: Optional[Mapping[str, pl.DataFrame]],
    accessibility_dir: Optional[Path],
    accessibility_cache_dir: Optional[Path],
    genome_file: Optional[Path],
) -> None:
    """Refuse argument combinations in which one argument would go unused.

    Raises
    ------
    ValueError
        When both ``accessibility`` and ``accessibility_dir`` are given; when
        ``genome_file`` is given with either of them; when
        ``accessibility_cache_dir`` is given with ``accessibility`` or
        ``genome_file`` but without ``accessibility_dir``; and when
        ``target_fasta`` or ``target_index`` is given with ``risearch_file``
        or ``predictions``. The message names both arguments and what to drop.
    """
    if accessibility is not None and accessibility_dir is not None:
        raise ValueError(
            "accessibility and accessibility_dir are two sources for the same "
            "input; pass one of them"
        )
    profiles = "accessibility" if accessibility is not None else "accessibility_dir"
    if genome_file is not None and (
        accessibility is not None or accessibility_dir is not None
    ):
        raise ValueError(
            f"genome_file and {profiles} are two sources for the same profiles; "
            f"drop genome_file to use {profiles}, or drop {profiles} to fold the "
            "genome"
        )
    if accessibility_cache_dir is not None and accessibility_dir is None:
        spilled = None
        if accessibility is not None:
            spilled = "accessibility is packed"
        elif genome_file is not None:
            spilled = "genome_file is folded"
        if spilled is not None:
            raise ValueError(
                "accessibility_cache_dir only holds the cache of accessibility_dir; "
                f"{spilled} in a temporary directory, so drop accessibility_cache_dir"
            )
    targets = [
        name
        for name, value in (
            ("target_fasta", target_fasta),
            ("target_index", target_index),
        )
        if value is not None
    ]
    source = "risearch_file" if risearch_file is not None else "predictions"
    if targets and (risearch_file is not None or predictions is not None):
        names = " and ".join(targets)
        raise ValueError(
            f"{names} {'is' if len(targets) == 1 else 'are'} only used by the "
            f"in-process search (sirna_fasta) and cannot be combined with "
            f"{source}; drop {names}"
        )


def _check_profile_names(profiles: Mapping[str, pl.DataFrame]) -> None:
    """Refuse in-memory profiles that cannot be spilled one file per chromosome.

    Each key becomes ``<key>.accessibility.parquet`` in the spill directory,
    so a key that is not a plain file name would fail there, or be written
    outside it and never packed.

    Raises
    ------
    ValueError
        When there are no profiles, or when a key is empty, ``.`` or ``..``,
        or holds ``/``, the path separator or a NUL; the message names every
        such key.
    """
    if not profiles:
        raise ValueError(
            "accessibility holds no profiles; pass one per chromosome, as "
            "sioff.accessibility returns"
        )
    bad = [
        chrom
        for chrom in profiles
        if chrom in ("", ".", "..") or any(c in chrom for c in ("/", os.sep, "\0"))
    ]
    if bad:
        raise ValueError(
            "accessibility keys name the chromosomes and become file names; "
            f"these cannot: {', '.join(map(repr, bad))}"
        )


def _search_in_process(
    sirna_fasta: Path,
    target_fasta: Path,
    target_index: Optional[Path],
    matrix: Union[str, Path],
) -> pl.DataFrame:
    """Search the siRNAs against the target with RIsearch, in this process.

    The target is indexed first unless ``target_index`` is given; the index is
    built (or reused) next to ``target_fasta``.

    Returns
    -------
    polars.DataFrame
        The hits in the ``sioff.search`` schema.
    """
    service = RIsearchService()
    index_path = (
        target_index if target_index is not None else service.index_target(target_fasta)
    )
    return service.run_search(
        query_path=sirna_fasta,
        index_path=index_path,
        target_fasta=target_fasta,
        matrix=matrix,
    )


def _split_inputs(
    executor: _Executor, files: list[Path], root: Path
) -> dict[str, Path]:
    """Split every input file into per-siRNA parts, one task per file.

    A file that cannot be read is collected in ``executor.failures``; the
    siRNAs of the other files go on.

    Returns
    -------
    dict[str, pathlib.Path]
        ``{sirna_id: part}``, sorted by ``sirna_id``.
    """
    # Each file gets its own directory: parts are named 0.parquet, 1.parquet...
    tasks = {f.name: (str(f), str(root / str(i))) for i, f in enumerate(files)}
    # The pool started here also runs the scoring, whose task count is not
    # known yet: one polars thread per worker keeps n_workers processes within
    # the budget.
    splits = dict(
        executor.run(split_file, tasks, threads_per_worker=1, uses_worker_state=False)
    )
    # Merged in input order and each split in sirna_id order, so the siRNA a
    # duplicate error names does not depend on which file finished first.
    merged = merge_splits(
        [(name, dict(sorted(splits[name].items()))) for name in tasks if name in splits]
    )
    return dict(sorted(merged.items()))


def _check_sirna_table(
    parts: Mapping[str, Path],
    table: Mapping[str, SirnaRecord],
    needs_e_min: bool,
    has_transcriptome: bool,
) -> None:
    """Check the predicted siRNAs against the siRNA table, before any scoring.

    Raises
    ------
    ValueError
        When an alpha/gamma pair needs E_min and a siRNA with predictions has
        no table row (or there is no table), so no perfect-match energy.
    """
    missing = sorted(sid for sid in parts if sid not in table)
    if missing and needs_e_min:
        raise ValueError(
            f"{len(missing)} siRNA(s) have predictions but no row in --sirna-table, "
            f"which alpha/gamma clamping needs for E_min: {', '.join(missing[:10])}"
        )
    # Without a table (optional at alpha = gamma = 1) nobody expects a row.
    if missing and table:
        logger.warning(
            f"{len(missing)} siRNA(s) have predictions but no row in --sirna-table, "
            f"so they have no on-target: {', '.join(missing[:10])}"
        )
    unused = [sid for sid in table if sid not in parts]
    if unused:
        logger.info(f"{len(unused)} siRNA(s) in --sirna-table have no predictions")
    if not has_transcriptome and any(rec.on_target_id for rec in table.values()):
        logger.warning(_NO_TRANSCRIPTOME_WARNING)


def compute_off_targets(
    *,
    predictions: Optional[pl.DataFrame] = None,
    risearch_file: Optional[Path] = None,
    sirna_fasta: Optional[Path] = None,
    target_fasta: Optional[Path] = None,
    target_index: Optional[Path] = None,
    gtf_file: Optional[Path] = None,
    feature_type: str = "exon",
    expression_metric: str = "RPKM",
    transcriptome_format: str = "auto",
    accessibility: Optional[Mapping[str, pl.DataFrame]] = None,
    accessibility_dir: Optional[Path] = None,
    accessibility_cache_dir: Optional[Path] = None,
    genome_file: Optional[Path] = None,
    window_size: int = 80,
    max_span: int = 40,
    unpaired_prob: int = 30,
    temperature: float = 37.0,
    matrix: Union[str, Path] = "t04",
    sirna_table: Optional[Path] = None,
    alpha: str = "1.0",
    gamma: str = "1.0",
    theta: str = "",
    sense_only: bool = False,
    predictions_type: str = "gw",
    n_workers: int = 1,
    summary_only: bool = False,
    detailed_report: bool = False,
    on_split: Optional[Callable[[int], None]] = None,
    accessibility_progress_callback=None,
) -> Generator[tuple[pl.DataFrame, dict], None, None]:
    """Compute off-target probabilities, one siRNA at a time.

    A generator yielding ``(df, meta)`` per siRNA, in completion order. Every
    input takes the same path: a file is treated as a folder holding one file,
    and every input file is split into one Parquet part per ``sirna_id``. A
    single ``sioff search`` TSV holding every siRNA and a folder of per-siRNA
    files therefore run the same way, and so do the ``predictions`` frame and
    the hits of the in-process search, which are split in this process. A
    ``sirna_id`` found in two input files is an error. Each siRNA is then
    scored as its own task: its part is
    loaded, filtered to the sense strand with ``sense_only``, intersected with
    the transcriptome and scored with the per-siRNA partition function.
    siRNAs with no predictions left after filtering and intersection are
    skipped.

    **E_min** (the reference of the alpha/gamma clamp) follows the legacy
    rule: the lower of the siRNA's perfect-match energy, computed from its
    ``sirna_table`` sequence with ``matrix``, and the strongest hit left
    after ``sense_only`` and the intersection. It is only computed when an
    alpha/gamma pair other than (1, 1) is, and every siRNA with predictions
    then needs a table row. At the baseline E_min is never used.

    **Tasks** run in a pool of spawned worker processes (``spawn`` rather
    than ``fork`` avoids Rayon/Polars thread-pool deadlocks), started only
    when more than one task can run at once. Otherwise they run in this
    process. Splitting the input files and scoring the siRNAs share the pool.
    A failed task is logged and skipped; once every successful siRNA has been
    yielded, a `WorkerFailureError` summarising the failures is raised, so a
    run where tasks failed never looks like an empty success.

    **Checks and warnings** are made in this process, before any siRNA is
    scored: a siRNA missing from the table (an error when E_min is needed, a
    warning otherwise), table rows without predictions (an info line), and
    on-target IDs without ``gtf_file`` (a warning). After the last siRNA, the
    siRNAs whose on-target matched no row are named in one warning.

    **Accessibility** reaches the workers as a memory-mapped packed cache,
    whatever its source: the cache of ``accessibility_dir``, the
    ``accessibility`` profiles written to a temporary directory and packed
    there, or ``genome_file`` folded into one. Without any of them, opening
    energies are treated as zero.

    **Temporary files**: the transcriptome is written once to an Arrow IPC
    file that every task memory-maps, the split parts go to another temporary
    directory, and so do the packed ``accessibility`` profiles or the
    ``genome_file`` fold. All of them are removed in a ``finally``, on error
    too. Consume the generator fully, call ``.close()`` on it, or wrap it in
    `contextlib.closing` for deterministic teardown.

    Parameters
    ----------
    predictions : polars.DataFrame, optional
        In-memory predictions, typically the value returned by
        [`sioff.search`][sioff.search]. Must contain the columns ``sirna_id,
        chrom, start, end, strand, energy``; extra columns are dropped and
        compatible dtypes are cast to the RIsearch schema.
    risearch_file : pathlib.Path, optional
        A predictions file, or a directory of them (CLI
        ``-r/--risearch-file``): RIsearch2 output (plain or ``.gz``), a
        ``sioff search`` TSV, or Parquet. From a directory every ``*.gz``,
        ``*.tsv``, ``*.out`` and ``*.parquet`` is read, each holding one siRNA
        or many.
    sirna_fasta : pathlib.Path, optional
        Query FASTA for an in-process RIsearch run against ``target_fasta``
        (CLI ``-s/--sirna-fasta``), searched with ``matrix``; duplicate
        sequence IDs are rejected. It is only the query: it cannot be
        combined with ``risearch_file`` or ``predictions``, and the
        perfect-match energies come from ``sirna_table``.
    target_fasta : pathlib.Path, optional
        Target FASTA (genome or transcriptome) for the in-process search
        (CLI ``--target-fasta``/``--genome``); required with ``sirna_fasta``
        and refused with ``risearch_file`` or ``predictions``.
    target_index : pathlib.Path, optional
        Pre-built RIsearch index for ``target_fasta`` (CLI
        ``-idx/--target-index``); refused with ``risearch_file`` or
        ``predictions``. When omitted the index is built (or reused) next to
        ``target_fasta``.
    gtf_file : pathlib.Path, optional
        Transcriptome annotation, GTF or GFF3 (CLI ``-t/--transcriptome``).
        Rows with expression <= 0 are dropped; predictions are intersected
        with the rest and annotated with ``transcript_id``, ``gene_id`` and
        ``exp_value``. Read before ``genome_file`` is folded.
    feature_type : str, default "exon"
        Feature type to select from a GTF/GFF3 annotation (CLI ``--feature``).
    expression_metric : str, default "RPKM"
        Annotation attribute used as the expression score (CLI
        ``--expression-metric``).
    transcriptome_format : str, default "auto"
        Annotation format: ``"auto"``, ``"gtf"`` or ``"gff3"``; BED is
        rejected (CLI ``--transcriptome-format``).
    accessibility : Mapping[str, polars.DataFrame], optional
        In-memory accessibility profiles, ``dict[chrom -> DataFrame]`` with
        the columns ``position, strand, u1..uN`` and both strands stacked, as
        returned by [`sioff.accessibility`][sioff.accessibility]. They are
        written to a temporary directory, one ``<chrom>.accessibility.parquet``
        per key, and packed there, so a key cannot be empty, ``.`` or ``..``,
        or hold ``/``, the path separator or a NUL. Mutually exclusive with
        ``accessibility_dir``.
    accessibility_dir : pathlib.Path, optional
        The npy cache from ``sioff accessibility`` or per-chromosome Parquet
        files (CLI ``-a/--accessibility-dir``). For Parquet the cache is built
        (or rebuilt, when older than its Parquet) in
        ``accessibility_cache_dir`` first, before anything else is read.
        Mutually exclusive with ``accessibility``.
    accessibility_cache_dir : pathlib.Path, optional
        Where the cache of ``accessibility_dir`` is read from and built;
        default ``accessibility_dir`` (CLI ``--accessibility-cache-dir``).
        Created if absent. Refused with ``accessibility`` or ``genome_file``,
        whose cache lives in a temporary directory.
    genome_file : pathlib.Path, optional
        Genome FASTA to fold during the run (CLI ``-f/--fasta``), straight
        into a packed cache in a temporary directory. Refused with
        ``accessibility`` or ``accessibility_dir``.
    window_size : int, default 80
        RNAplfold window size for folding ``genome_file`` (CLI
        ``-W/--window``).
    max_span : int, default 40
        Maximum base-pair span for folding ``genome_file`` (CLI
        ``-L/--span``).
    unpaired_prob : int, default 30
        Maximum unpaired-stretch length for folding ``genome_file`` (CLI
        ``-u/--unpaired``).
    temperature : float, default 37.0
        Temperature in degrees Celsius for the partition function (CLI
        ``-T/--temperature``).
    matrix : str or pathlib.Path, default "t04"
        Energy parameter set for the perfect-match energies: a bundled id or
        the path of a DSM TSV table (CLI ``-z/--matrix``). Use the set the
        predictions were searched with: E_min is the reference energy of the
        alpha/gamma clamp.
    sirna_table : pathlib.Path, optional
        Tab-separated siRNA table with a header (CLI ``--sirna-table``):
        ``sirna_id``, ``on_target_id`` (a transcript or gene ID; may be empty)
        and ``sequence`` (the guide strand). Gives each siRNA's on-target and
        its perfect-match energy. Required when an alpha/gamma pair other than
        (1, 1) is computed.
    alpha : str, default "1.0"
        Alpha clamping parameter(s) as a ``";"``-separated string, e.g.
        ``"0.8;1.0"`` (CLI ``--alpha``). Every ``alpha`` is paired with every
        ``gamma``; pairs with ``alpha > gamma`` are dropped, duplicates are
        removed and the baseline ``(1.0, 1.0)`` is always included.
    gamma : str, default "1.0"
        Gamma clamping parameter(s), same format as ``alpha`` (CLI
        ``--gamma``).
    theta : str, default ""
        Theta scaling parameter(s) as a ``";"``-separated string, e.g.
        ``"0.5;0.7"`` (CLI ``--theta``). Empty means no theta scaling.
    sense_only : bool, default False
        Keep only sense-strand (``+``) predictions (CLI ``--sense-only``).
    predictions_type : str, default "gw"
        ``"gw"`` for genome-wide predictions (interval intersection with the
        annotation) or ``"tw"`` for transcriptome-wide predictions (join on
        transcript ID) (CLI ``--type``).
    n_workers : int, default 1
        Worker processes for splitting and scoring (CLI ``-j/--workers``). A
        pool is started only when more than one task can run at once.
    summary_only : bool, default False
        Only the per-siRNA sums are needed (CLI ``--summary-only``): siRNAs
        are scored with ``aggregate_only`` and ``df`` is empty. ``meta`` is
        unchanged and carries ``"n_rows"``, the number of scored rows.
    detailed_report : bool, default False
        Keep the raw ``boltzmann_weight*``, ``Z_sirna*`` and ``E_min``
        columns in each ``df`` (CLI ``--detailed-report``).
    on_split : callable, optional
        Called once with the number of siRNAs to score, after the inputs are
        split and checked and before any siRNA is scored.
    accessibility_progress_callback : callable, optional
        Called as ``callback(advance=n, description=str)`` after each batch
        of ``n`` records of ``genome_file`` is folded.

    Yields
    ------
    tuple[polars.DataFrame, dict]
        ``(df, meta)`` for one siRNA. ``df`` holds the input columns
        ``sirna_id, chrom, start, end, strand, energy``, the annotation
        columns ``trans_start, trans_end, gene_id, transcript_id, exp_value``
        when ``gtf_file`` is given, ``opening_energy``, ``dG_total``,
        ``is_on_target`` and ``P_off_target``, plus ``dG_total``, ``energy``
        and ``P_off_target`` repeated with a ``:alpha=A,gamma=G`` or
        ``:theta=T`` suffix for each extra parameter set (empty with
        ``summary_only``). With ``detailed_report`` it also holds ``E_min``,
        ``boltzmann_weight*`` and ``Z_sirna*``. ``meta`` contains
        ``"n_sirnas"``, ``"n_rows"``, ``"z_per_sirna"``,
        ``"on_target_weights"``, ``"on_target_count"``, ``"n_on_target_rows"``
        and ``"_timings"`` (per-stage wall times of the task: ``load_s``,
        ``intersect_s``, ``prob_s``, ``serialize_s``, ``total_s`` and the
        intersection sub-timings).

    Raises
    ------
    ValueError
        When an argument would go unused: both ``predictions`` and
        ``risearch_file``, or both ``accessibility`` and
        ``accessibility_dir``; ``sirna_fasta``, ``target_fasta`` or
        ``target_index`` with ``risearch_file`` or ``predictions``;
        ``genome_file`` with ``accessibility`` or ``accessibility_dir``;
        ``accessibility_cache_dir`` with ``accessibility`` or ``genome_file``.
        Also when ``accessibility`` is empty or has a key that cannot be a
        file name; when no prediction source is given; when ``sirna_fasta``
        is given without ``target_fasta``; when ``predictions`` lacks a
        required column; when ``sirna_fasta`` holds a duplicate ID; when the
        siRNA table cannot be read; when a ``sirna_id`` occurs in two input
        files; and when an alpha/gamma pair other than (1, 1) is computed and
        a siRNA with predictions has no row in ``sirna_table`` (or there is
        no table).
    FileNotFoundError
        When ``risearch_file`` does not exist, or is a directory without
        prediction files; when ``sirna_fasta``, ``target_fasta``,
        ``target_index`` or ``gtf_file`` does not exist. All of these are
        checked before any profile is packed or folded.
    RIsearchError
        `sioff.services.risearch_service.RIsearchError`, when the in-process
        search or indexing ``target_fasta`` fails.
    AccessibilityError
        When ``accessibility_dir`` (and ``accessibility_cache_dir``) holds no
        profile, when the cache cannot be built, or when ``accessibility``
        cannot be packed or ``genome_file`` cannot be folded.
    WorkerFailureError
        After the last successful siRNA, when an input file could not be
        split or a siRNA could not be scored.

    See Also
    --------
    sioff.off_targets : Public wrapper.
    """
    # 1. Argument combinations, the inputs and the accessibility cache.
    if predictions is not None and risearch_file is not None:
        raise ValueError(
            "predictions and risearch_file are two sources for the same input; "
            "pass one of them"
        )
    if sirna_fasta is not None and (
        risearch_file is not None or predictions is not None
    ):
        raise ValueError(
            "sirna_fasta is the query for the in-process search and cannot be "
            "combined with risearch_file or predictions; the perfect-match energy "
            "for E_min now comes from the siRNA table"
        )
    _check_ignored_arguments(
        risearch_file=risearch_file,
        predictions=predictions,
        target_fasta=target_fasta,
        target_index=target_index,
        accessibility=accessibility,
        accessibility_dir=accessibility_dir,
        accessibility_cache_dir=accessibility_cache_dir,
        genome_file=genome_file,
    )
    if accessibility is not None:
        _check_profile_names(accessibility)
    input_files: list[Path] = []
    frame: Optional[pl.DataFrame] = None
    if risearch_file is not None:
        input_files = _input_files(Path(risearch_file))
    elif predictions is not None:
        frame = _coerce_predictions(predictions)
    elif sirna_fasta is None:
        raise ValueError(
            "Must provide predictions (a DataFrame), risearch_file (a file "
            "or a directory) or sirna_fasta"
        )
    elif target_fasta is None:
        raise ValueError(
            "sirna_fasta requires target_fasta when running RIsearch dynamically"
        )
    else:
        # The search itself runs in step 5; duplicate query IDs fail now.
        RIsearchService().validate_sirna_fasta(Path(sirna_fasta))
    # Cheap, so a typo fails before a long genome fold or cache build.
    for path, what in (
        (gtf_file, "Transcriptome file"),
        (target_fasta, "Target FASTA"),
        (target_index, "Index"),
    ):
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"{what} not found: {path}")
    # Workers only ever memory-map the finished cache. In-memory profiles and
    # a fold get theirs in step 4, inside the try, so its temporary directory
    # is always removed.
    acc_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    if accessibility_dir is not None:
        acc_dir = Path(accessibility_dir)
        cache_dir = _prepare_accessibility_cache(
            acc_dir, accessibility_cache_dir, n_workers
        )

    # 2. The siRNA table: on-targets and perfect-match sequences.
    table = read_sirna_table(Path(sirna_table)) if sirna_table is not None else {}
    on_target_map = {
        sid: rec.on_target_id for sid, rec in table.items() if rec.on_target_id
    }

    # 3. Parameters.
    alpha_gamma_pairs = _build_alpha_gamma_pairs(alpha, gamma)
    theta_vals = _parse_theta(theta)
    needs_e_min = any(pair != (1.0, 1.0) for pair in alpha_gamma_pairs)

    spill_dir: Optional[str] = None
    ipc_dir: Optional[str] = None
    split_root: Optional[str] = None
    executor: Optional[_Executor] = None
    try:
        # 4. The worker inputs. The transcriptome first: it is quick next to
        # a fold, so a bad annotation fails before the genome is folded. It
        # is sorted by (chrom, strand, start) so each task's partitions are
        # zero-copy slices of the memory-mapped file, and uncompressed so it
        # can be memory-mapped at all; see preload_transcriptome.
        ipc_path = ""
        if gtf_file is not None:
            df_trans = AnnotationParser().load_gtf(
                Path(gtf_file),
                feature=feature_type,
                score_col=expression_metric,
                format=transcriptome_format,
            )
            df_trans = sort_for_preload(_drop_unexpressed(df_trans))
            ipc_dir = tempfile.mkdtemp(prefix="risearch_ipc_")
            ipc_path = str(Path(ipc_dir) / "transcriptome.arrow")
            df_trans.write_ipc(
                ipc_path,
                compression="uncompressed",
                compat_level=pl.CompatLevel.oldest(),
            )
            del df_trans
        # Then the profiles: held in memory, they are written out as Parquet
        # and packed; a genome is folded straight into a packed cache. -a
        # needs neither.
        if accessibility is not None:
            spill_dir = tempfile.mkdtemp(prefix="sioff_acc_")
            for chrom, profile in accessibility.items():
                profile.write_parquet(
                    Path(spill_dir) / f"{chrom}{PARQUET_SUFFIX}", compression="lz4"
                )
            acc_dir = Path(spill_dir)
            cache_dir = _prepare_accessibility_cache(acc_dir, None, n_workers)
        elif genome_file is not None:
            spill_dir = tempfile.mkdtemp(prefix="sioff_fold_")
            acc_dir = Path(spill_dir)
            GenomeAccessibilityService(acc_dir).compute_genome_accessibility(
                Path(genome_file),
                window_size=window_size,
                max_span=max_span,
                unpaired_prob=unpaired_prob,
                temperature=temperature,
                output_format="npy",
                progress_callback=accessibility_progress_callback,
            )
        executor = _Executor(
            n_workers,
            {
                "ipc_path": ipc_path,
                "accessibility_dir": str(acc_dir) if acc_dir is not None else "",
                "on_target_map": on_target_map,
                "self_hyb_emin": {},
                "temperature": temperature,
                "accessibility_cache_dir": (
                    str(cache_dir) if cache_dir is not None else None
                ),
                "summary_only": summary_only,
                "detailed_report": detailed_report,
            },
        )

        # 5. Split phase: input files in tasks, a frame here.
        split_root = tempfile.mkdtemp(prefix="sioff_split_")
        if input_files:
            parts = _split_inputs(executor, input_files, Path(split_root))
        else:
            if frame is None:
                # Step 1 checked both FASTAs and the query's IDs.
                assert sirna_fasta is not None and target_fasta is not None
                frame = _search_in_process(
                    Path(sirna_fasta),
                    Path(target_fasta),
                    Path(target_index) if target_index is not None else None,
                    matrix,
                )
            parts = dict(sorted(split_frame(frame, Path(split_root) / "0").items()))
            frame = None  # the parts hold the rows now
        n_split_failures = len(executor.failures)

        # 6. Checks, before any scoring.
        _check_sirna_table(parts, table, needs_e_min, gtf_file is not None)
        if on_split is not None:
            on_split(len(parts))

        # 7. Perfect-match energies (every siRNA has a table row by now).
        perfect_match: dict[str, float] = {}
        if needs_e_min:
            service = RIsearchService()
            perfect_match = {
                sid: service.self_hybridization_emin(
                    table[sid].sequence, sid, matrix=matrix
                )
                for sid in parts
                if sid in table
            }

        # 8. Scoring phase.
        tasks = {
            sid: (
                sid,
                str(part),
                perfect_match.get(sid),
                alpha_gamma_pairs,
                theta_vals,
                sense_only,
                predictions_type,
            )
            for sid, part in parts.items()
        }
        # Polars threads per worker, if the pool starts here: keeps
        # processes x threads within n_workers.
        n_proc = max(1, min(executor.n_workers, len(tasks)))
        n_on_target_rows: dict[str, int] = {}
        for sid, (arrow_table, meta) in executor.run(
            _score_part,
            tasks,
            threads_per_worker=max(1, executor.n_workers // n_proc),
            uses_worker_state=True,
        ):
            if arrow_table is None:
                # --sense-only and the intersection left no row, so none
                # can be its on-target either.
                n_on_target_rows[sid] = 0
                continue
            n_on_target_rows.update(meta["n_on_target_rows"])
            df_chunk = cast(pl.DataFrame, pl.from_arrow(arrow_table))
            del arrow_table  # Polars owns the data now
            yield df_chunk, meta

        # 9. On-targets that matched no row. Without a transcriptome nothing
        # can match, and the check above already said so once.
        if gtf_file is not None:
            unmatched = sorted(
                sid
                for sid, n in n_on_target_rows.items()
                if n == 0 and sid in on_target_map
            )
            if unmatched:
                logger.warning(
                    f"{len(unmatched)} siRNA(s) have no on-target row (W_on = 0): "
                    f"{', '.join(unmatched[:10])}"
                )
        if executor.failures:
            # An input file that could not be split counts as one task: how
            # many siRNAs it held is unknown.
            raise WorkerFailureError(
                "per-siRNA workers", executor.failures, n_split_failures + len(tasks)
            )
    finally:
        # 10. Pool first (its workers hold the files open), then the files,
        # even if shutting the pool down fails.
        try:
            if executor is not None:
                executor.close()
        finally:
            for tmp in (split_root, ipc_dir, spill_dir):
                if tmp is not None:
                    shutil.rmtree(tmp, ignore_errors=True)
