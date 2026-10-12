"""Pure core for off-target analysis.

Two entry points, both free of CLI concerns (no stdout, no Typer/Click
exceptions, no file writes):

- [`compute_off_targets_single`][sioff.core.off_targets.compute_off_targets_single] -- a single prediction source (an in-memory
  DataFrame, one RIsearch2 output file, or RIsearch run in-process); returns
  ``(pl.DataFrame, meta)``.
- [`compute_off_targets_directory`][sioff.core.off_targets.compute_off_targets_directory] -- a directory of per-siRNA prediction
  files; a generator yielding ``(pl.DataFrame, meta)`` per siRNA. The process
  pool and the temporary Arrow-IPC transcriptome live inside the generator's
  ``try/finally``, so callers should consume it fully (or close it) for
  deterministic cleanup.

``meta`` carries the partition-function state downstream formatters need:
``z_per_sirna``/``on_target_weights`` (multi-siRNA), or ``z_total`` etc.
(single-siRNA). When ``legacy_format`` is requested the single core also stashes
the rendered legacy report under ``meta["legacy_text"]`` (it reuses the already
built accessibility service, avoiding a second on-the-fly fold).

The public wrappers [`sioff.off_targets`][sioff.off_targets], [`sioff.accessibility`][sioff.accessibility],
[`sioff.index`][sioff.index] and [`sioff.search`][sioff.search] in `sioff.api` add path
coercion and mode dispatch on top of this layer.
"""

import ctypes
import gc
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, Mapping, Optional, cast

import polars as pl
from loguru import logger

from sioff.models import RISEARCH_COLUMNS, RISEARCH_SCHEMA
from sioff.services.accessibility import (
    GenomeAccessibilityService,
    ensure_accessibility_cache,
)
from sioff.services.annotation_parser import AnnotationParser
from sioff.services.intersection_service import IntersectionService
from sioff.services.probability import ProbabilityService
from sioff.services.profiling import PipelineProfiler
from sioff.services.risearch_parser import RIsearchParser
from sioff.services.risearch_service import RIsearchService

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


class WorkerPoolError(RuntimeError):
    """Raised by `compute_off_targets_directory` when worker tasks failed.

    Raised once every file was attempted and at least one worker task failed
    (an exception in the worker, or a worker killed, e.g. by the OOM killer).
    Successful siRNAs are yielded before this is raised, so a caller that
    streams results to disk keeps them; the CLI reports the summary and exits
    non-zero instead of pretending there was nothing to process.

    Attributes
    ----------
    failures : dict[str, str]
        File name -> error message, in completion order.
    n_total : int
        Number of files submitted to the pool.
    """

    def __init__(self, failures: dict[str, str], n_total: int):
        self.failures = failures
        self.n_total = n_total
        first_name, first_err = next(iter(failures.items()))
        super().__init__(
            f"{len(failures)}/{n_total} siRNA files failed in worker processes; "
            f"first error ({first_name}): {first_err}"
        )

    @property
    def n_failed(self) -> int:
        """Number of files whose worker task failed."""
        return len(self.failures)


def _init_worker(
    ipc_path: str,
    accessibility_dir: str,
    polars_max_threads: int,
    on_target_map: dict,
    self_hyb_emin: dict,
    temperature: float = 37.0,
    accessibility_cache_dir: str = "",
) -> None:
    """Initialise per-worker state; called once per spawned worker process.

    The transcriptome is loaded from a pre-written Arrow IPC file (memory-mapped)
    rather than being re-parsed from the original BED/GTF, and accessibility
    profiles are memory-mapped from the uint8 cache the parent built before
    starting the pool. All workers share the same physical memory pages via
    the OS page cache.
    """
    os.environ["POLARS_MAX_THREADS"] = str(polars_max_threads)

    global _WORKER_DF_TRANS, _WORKER_INTERSECTOR, _WORKER_PROB_SERVICE
    global _WORKER_ON_TARGET_MAP, _WORKER_SELF_HYB_EMIN

    if ipc_path:
        _WORKER_DF_TRANS = pl.read_ipc(Path(ipc_path), memory_map=True)
        _WORKER_INTERSECTOR = IntersectionService()
        _WORKER_INTERSECTOR.preload_transcriptome(_WORKER_DF_TRANS)

    acc_service = None
    if accessibility_dir:
        acc_service = GenomeAccessibilityService(
            Path(accessibility_dir),
            max_cached=4,
            cache_dir=Path(accessibility_cache_dir)
            if accessibility_cache_dir
            else None,
        )

    _WORKER_PROB_SERVICE = ProbabilityService(acc_service, temperature=temperature)

    # Store per-run constants so they don't need to be pickled per task.
    _WORKER_ON_TARGET_MAP = on_target_map
    _WORKER_SELF_HYB_EMIN = self_hyb_emin


def _process_one_sirna(
    file_path_str: str,
    alpha_gamma_pairs: list,
    theta_vals: list,
    on_target_expression: float,
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

    # Apply self-hybridisation E_min override (one value per siRNA file)
    if self_hyb_emin and "raw_e_min" in df.columns:
        sirna_id = str(df["sirna_id"][0])
        if sirna_id in self_hyb_emin:
            df = df.with_columns(
                pl.lit(self_hyb_emin[sirna_id]).cast(pl.Float32).alias("raw_e_min")
            )

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
    _t_intersect = time.perf_counter()

    # Probabilities
    df, meta = prob_service.calculate_probabilities_per_sirna(
        df,
        alpha_gamma_pairs=alpha_gamma_pairs,
        theta_values=theta_vals,
        on_target_map=on_target_map,
        on_target_expression=on_target_expression,
    )
    _t_prob = time.perf_counter()

    # Drop heavy intermediate columns before returning
    drop_cols = [
        c
        for c in df.columns
        if c.startswith("boltzmann_weight") or c.startswith("Z_sirna") or c == "E_min"
    ]
    if drop_cols:
        df = df.drop(drop_cols)

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
    on_target_file: Optional[Path] = None,
    on_target_risearch_file: Optional[Path] = None,
    query_file: Optional[Path] = None,
    on_target_expression: float = 1000.0,
    on_target_accessibility: Optional[Path] = None,
    on_target_ids_file: Optional[Path] = None,
    alpha: str = "1.0",
    gamma: str = "1.0",
    theta: str = "",
    sense_only: bool = False,
    predictions_type: str = "gw",
    legacy_format: bool = False,
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
    probability and legacy-report computation). With none of them, opening
    energies are treated as zero.

    When ``gtf_file`` is given, the predictions are intersected with the
    annotation before probabilities are computed. The probability model is
    chosen automatically: the per-siRNA partition function is used when the
    frame holds more than one ``sirna_id`` or when any non-baseline ``alpha`` /
    ``gamma`` / ``theta`` value is requested; otherwise the single-siRNA model
    (which alone supports ``on_target_file``, ``on_target_risearch_file``,
    ``query_file``, ``on_target_accessibility`` and ``legacy_format``).

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
        Transcriptome annotation, GTF/GFF3 or BED (CLI ``-t/--transcriptome``).
        When given, predictions are intersected with its features and annotated
        with ``transcript_id``, ``gene_id`` and ``exp_value``.
    feature_type : str, default "exon"
        Feature type to select from a GTF/GFF3 annotation (CLI ``--feature``).
    expression_metric : str, default "RPKM"
        Annotation attribute used as the expression score (CLI
        ``--expression-metric``).
    transcriptome_format : str, default "auto"
        Annotation format: ``"auto"``, ``"gtf"``, ``"gff3"``, ``"bed6"`` or
        ``"bed7"`` (CLI ``--transcriptome-format``). ``"auto"`` detects the
        format from the file, distinguishing GTF from GFF3 by the attribute
        column.
    accessibility : Mapping[str, polars.DataFrame], optional
        In-memory accessibility profiles, ``dict[chrom -> DataFrame]`` with the
        columns ``position, strand, u1..uN`` and both strands stacked, as
        returned by [`sioff.accessibility`][sioff.accessibility]. Mutually exclusive with
        ``accessibility_dir``.
    accessibility_dir : pathlib.Path, optional
        Directory of per-chromosome ``{chrom}.accessibility.parquet`` files
        written by ``sioff accessibility`` (CLI ``-a/--accessibility-dir``), or
        of the ``.access.npy`` cache files built from them.
        Mutually exclusive with ``accessibility``.
    accessibility_cache_dir : pathlib.Path, optional
        Where the memory-mapped ``<chrom>_<plus|minus>.access.npy`` cache is
        read from and, when missing, built (CLI
        ``--accessibility-cache-dir``). Created if absent. Default:
        ``accessibility_dir`` itself.
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
    on_target_file : pathlib.Path, optional
        On-target sequence FASTA for the single-siRNA partition function (CLI
        ``-on/--on-target``). Requires ``query_file``; adds an ``onTarget`` row
        to the result and populates ``meta["p_on_target"]``.
    on_target_risearch_file : pathlib.Path, optional
        Pre-computed RIsearch output for the on-target hybridisation (CLI
        ``-on-ris/--on-target-risearch-file``); skips the on-the-fly RIsearch
        run against ``on_target_file``.
    query_file : pathlib.Path, optional
        siRNA query FASTA used for the on-target computation (CLI
        ``-q/--query``); required when ``on_target_file`` is set. Its stem
        names the siRNA in the legacy report.
    on_target_expression : float, default 1000.0
        Expression level assigned to the on-target (CLI
        ``-oexp/--on-target-expression``).
    on_target_accessibility : pathlib.Path, optional
        Accessibility Parquet for the on-target, same schema as the
        ``sioff accessibility`` output (CLI ``--on-target-accessibility``).
        Falls back to on-the-fly folding when omitted.
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
    legacy_format : bool, default False
        Also render the legacy ``gw.results``-style report, aggregated by
        transcript, into ``meta["legacy_text"]`` (CLI ``--legacy-format``).
    detailed_report : bool, default False
        Include per-transcript off-target probabilities in the legacy report
        (CLI ``--detailed-report``). Only meaningful with ``legacy_format``.
    n_workers : int, default 1
        Number of parallel threads for the genome-wide intersection (CLI
        ``-j/--workers``).
    profiler : PipelineProfiler, optional
        `sioff.services.profiling.PipelineProfiler` that records wall
        time and memory per stage (CLI ``--profile``). A disabled profiler is
        used when omitted.
    accessibility_progress_callback : callable, optional
        Called as ``callback(advance=1, description=str)`` after each
        chromosome is folded on the fly; only used with ``genome_file``.

    Returns
    -------
    tuple[polars.DataFrame, dict]
        ``(df, meta)``. ``df`` holds the input columns ``sirna_id, chrom, start,
        end, strand, energy``, plus ``trans_start, trans_end, gene_id,
        transcript_id, exp_value`` when intersected with ``gtf_file``, plus the
        probability columns ``opening_energy``, ``dG_total`` and
        ``P_off_target``. The per-siRNA model additionally adds ``E_min``,
        ``is_on_target``, ``boltzmann_weight``, ``Z_sirna`` and
        ``Z_sirna_noacc``, and repeats ``dG_total``, ``energy``,
        ``boltzmann_weight``, ``Z_sirna*`` and ``P_off_target`` with a
        ``:alpha=A,gamma=G`` or ``:theta=T`` suffix for each extra parameter
        set. ``meta`` always contains ``"_report"`` (a dict with ``n_loaded``,
        ``energy_min``, ``energy_max``, ``sense_only``, ``n_features``,
        ``n_intersected`` and ``preview``, the raw prediction head) and, when
        ``legacy_format`` is set, ``"legacy_text"``. The per-siRNA model adds
        ``"n_sirnas"``, ``"z_per_sirna"``, ``"on_target_weights"`` and
        ``"on_target_count"``; the single-siRNA model adds ``"z_total"``,
        ``"z_off_target"``, ``"w_on_target"``, ``"dG_on_target"``,
        ``"has_on_target"`` and, with an on-target, ``"p_on_target"``.

    Raises
    ------
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

    unique_sirnas = df["sirna_id"].unique() if "sirna_id" in df.columns else []
    has_custom_params = len(alpha_gamma_pairs) > 1 or len(theta_vals) > 0
    is_multi_sirna = len(unique_sirnas) > 1 or has_custom_params

    on_target_map: dict[str, str] = {}
    if on_target_ids_file is not None:
        try:
            mapping_df = pl.read_csv(
                on_target_ids_file, separator="\t", has_header=False
            )
            on_target_map = dict(
                zip(
                    mapping_df.get_column("column_1"),
                    mapping_df.get_column("column_2"),
                )
            )
        except Exception as e:
            raise ValueError(f"Error parsing on-target mapping file: {e}") from e

    def _finish(prob_service: ProbabilityService, frame: pl.DataFrame):
        if is_multi_sirna:
            with profiler.stage(
                "Probabilities (per-siRNA)", rows_in=frame.height
            ) as _s:
                frame, meta = prob_service.calculate_probabilities_per_sirna(
                    frame,
                    alpha_gamma_pairs=alpha_gamma_pairs,
                    theta_values=theta_vals,
                    on_target_map=on_target_map if on_target_map else None,
                    on_target_expression=on_target_expression,
                )
                _s.rows_out = frame.height
        else:
            with profiler.stage(
                "Probabilities (single-siRNA)", rows_in=frame.height
            ) as _s:
                frame, meta = prob_service.calculate_probabilities(
                    frame,
                    on_target_path=on_target_file,
                    query_path=query_file,
                    on_target_expression=on_target_expression,
                    on_target_accessibility_path=on_target_accessibility,
                    on_target_risearch_path=on_target_risearch_file,
                )
                _s.rows_out = frame.height
        logger.info(f"After probabilities: {frame.height:,} rows")

        # Legacy report must be rendered here (reuses prob_service's accessibility
        # service, avoiding a second on-the-fly fold). Replicates the CLI's call,
        # which uses calculate_legacy_format's default alpha/gamma pairs.
        if legacy_format:
            sirna_id = query_file.stem if query_file else "siRNA"
            meta["legacy_text"] = prob_service.calculate_legacy_format(
                frame,
                sirna_id=sirna_id,
                on_target_path=on_target_file,
                query_path=query_file,
                on_target_expression=on_target_expression,
                on_target_accessibility_path=on_target_accessibility,
                on_target_risearch_path=on_target_risearch_file,
                verbose=detailed_report,
            )
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
        acc_service = GenomeAccessibilityService(
            Path(accessibility_dir), max_cached=4, cache_dir=accessibility_cache_dir
        )
        prob_service = ProbabilityService(acc_service, temperature=temperature)
        return _finish(prob_service, df)
    elif genome_file:
        # Compute accessibility on-the-fly into a temp dir; the dir must outlive the
        # probability + legacy computation (both read the profiles), so do them
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
    on_target_ids_file: Optional[Path] = None,
    on_target_expression: float = 1000.0,
    alpha: str = "1.0",
    gamma: str = "1.0",
    theta: str = "",
    sense_only: bool = False,
    predictions_type: str = "gw",
    n_workers: int = 1,
) -> Generator[tuple[pl.DataFrame, dict], None, None]:
    """Compute off-target probabilities for a directory of per-siRNA files.

    A generator yielding ``(df, meta)`` for each RIsearch prediction file in
    ``input_dir``, in completion order. Each file is processed in a spawned
    worker process (``spawn`` rather than ``fork`` avoids Rayon/Polars
    thread-pool deadlocks): it is loaded, optionally filtered to the sense
    strand, intersected with the transcriptome, and scored with the per-siRNA
    partition function. Files with no remaining predictions are skipped
    silently. A file whose worker fails is logged and skipped, and once every
    file has been attempted the generator raises `WorkerPoolError` summarising
    the failures (after yielding every successful siRNA). Heavy intermediate
    columns (``boltzmann_weight*``, ``Z_sirna*``, ``E_min``) are dropped from
    the yielded frames.

    The process pool and the temporary Arrow-IPC copy of the transcriptome
    (memory-mapped by every worker) are held open across yields and cleaned up
    in a ``finally``. Consume the generator fully, call ``.close()`` on it, or
    wrap it in `contextlib.closing` for deterministic teardown.

    Accessibility profiles are read from ``accessibility_dir`` only; in-memory
    profiles and on-the-fly folding are not supported in this mode. Writes no
    files (other than the temporary IPC copy) and prints nothing.

    Parameters
    ----------
    input_dir : pathlib.Path
        Directory of RIsearch2 output files, one per siRNA (CLI
        ``-r/--risearch-file`` pointing at a directory).
    sirna_fasta : pathlib.Path, optional
        siRNA FASTA (CLI ``-s/--sirna-fasta``). When given, a self-hybridisation
        ``E_min`` is computed for every siRNA and overrides the per-file
        ``raw_e_min`` before clamping (matching the legacy ``E_min`` semantics).
    gtf_file : pathlib.Path, optional
        Transcriptome annotation, GTF/GFF3 or BED (CLI ``-t/--transcriptome``).
        Loaded once, serialised to a temporary Arrow IPC file and memory-mapped
        by each worker for the intersection.
    feature_type : str, default "exon"
        Feature type to select from a GTF/GFF3 annotation (CLI ``--feature``).
    expression_metric : str, default "RPKM"
        Annotation attribute used as the expression score (CLI
        ``--expression-metric``).
    transcriptome_format : str, default "auto"
        Annotation format: ``"auto"``, ``"gtf"``, ``"gff3"``, ``"bed6"`` or
        ``"bed7"`` (CLI ``--transcriptome-format``).
    accessibility_dir : pathlib.Path, optional
        Directory of per-chromosome ``{chrom}.accessibility.parquet`` files
        written by ``sioff accessibility`` (CLI ``-a/--accessibility-dir``), or
        of the ``.access.npy`` cache files built from them. Without it,
        opening energies are treated as zero.
    accessibility_cache_dir : pathlib.Path, optional
        Where the memory-mapped ``<chrom>_<plus|minus>.access.npy`` cache
        lives (CLI ``--accessibility-cache-dir``); created if absent. Every
        contig that has a Parquet but no cache is built here, in the parent,
        before the worker pool starts (``min(n_workers, 4)`` build processes,
        one strand's cache in memory each). Default: ``accessibility_dir``.
    temperature : float, default 37.0
        Temperature in degrees Celsius for the partition function (CLI
        ``-T/--temperature``).
    on_target_ids_file : pathlib.Path, optional
        Two-column TSV mapping ``sirna_id`` to on-target ``transcript_id``
        (CLI ``-oi/--on-target-ids``); lines starting with ``#`` are ignored.
    on_target_expression : float, default 1000.0
        Expression level assigned to the on-target (CLI
        ``-oexp/--on-target-expression``).
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

    Yields
    ------
    tuple[polars.DataFrame, dict]
        ``(df, meta)`` for one siRNA. ``df`` holds the input columns
        ``sirna_id, chrom, start, end, strand, energy``, the annotation columns
        ``trans_start, trans_end, gene_id, transcript_id, exp_value`` when
        ``gtf_file`` is given, ``opening_energy``, ``dG_total``,
        ``is_on_target`` and ``P_off_target``, plus ``dG_total``, ``energy`` and
        ``P_off_target`` repeated with a ``:alpha=A,gamma=G`` or ``:theta=T``
        suffix for each extra parameter set. ``meta`` contains ``"n_sirnas"``,
        ``"z_per_sirna"``, ``"on_target_weights"``, ``"on_target_count"`` and
        ``"_timings"`` (per-stage wall times for the worker: ``load_s``,
        ``intersect_s``, ``prob_s``, ``serialize_s``, ``total_s`` and the
        intersection sub-timings).

    Raises
    ------
    FileNotFoundError
        When ``input_dir`` contains no RIsearch prediction files.
    WorkerPoolError
        After the last file, when one or more worker tasks failed.

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

    # Serialize transcriptome to a temp Arrow IPC file so workers can memory-map
    # it instead of re-parsing the BED/GTF. Workers share pages via the OS cache.
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
        _ipc_tmp = tempfile.mkdtemp(prefix="risearch_ipc_")
        _ipc_path = Path(_ipc_tmp) / "transcriptome.arrow"
        df_trans.write_ipc(_ipc_path)

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
        self_hyb_emin = RIsearchService().self_hybridization_emin_batch(sirna_fasta)

    # Build the memory-mapped accessibility cache once, here, so workers only
    # ever map finished files (and never each decode a whole chromosome).
    if accessibility_dir:
        _t0 = time.perf_counter()
        built = ensure_accessibility_cache(
            Path(accessibility_dir),
            accessibility_cache_dir,
            workers=n_workers,
        )
        if built:
            logger.info(
                f"Built accessibility cache for {len(built)} contig(s) in "
                f"{time.perf_counter() - _t0:.1f}s"
            )

    alpha_gamma_pairs = _build_alpha_gamma_pairs(alpha, gamma)
    theta_vals = _parse_theta(theta)

    n_proc = min(n_workers, len(all_files))
    # Polars threads per worker: keeps N_workers × threads ≤ available cores.
    polars_threads_per_worker = max(1, n_workers // n_proc)

    # spawn: each worker starts a fresh interpreter, avoiding fork+Rayon deadlocks.
    _ctx = multiprocessing.get_context("spawn")
    _init_args = (
        str(_ipc_path) if gtf_file else "",
        str(accessibility_dir) if accessibility_dir else "",
        polars_threads_per_worker,
        on_target_map,
        self_hyb_emin,
        temperature,
        str(accessibility_cache_dir) if accessibility_cache_dir else "",
    )

    try:
        with ProcessPoolExecutor(
            max_workers=n_proc,
            mp_context=_ctx,
            initializer=_init_worker,
            initargs=_init_args,
        ) as pool:
            futures = {
                pool.submit(
                    _process_one_sirna,
                    str(f),
                    alpha_gamma_pairs,
                    theta_vals,
                    on_target_expression,
                    sense_only,
                    predictions_type,
                ): f
                for f in all_files
            }

            n_total = len(futures)
            failures: dict[str, str] = {}
            for future in as_completed(futures):
                # Pop the future immediately so its held result (arrow table +
                # metadata) can be GC'd as soon as the loop variable rotates.
                f = futures.pop(future)
                try:
                    arrow_table, batch_metadata = future.result()
                except Exception as exc:
                    logger.error(f"Worker failed for {f.name}: {exc}")
                    failures[f.name] = str(exc)
                    continue

                if arrow_table is None:
                    continue

                # arrow_table is a pyarrow Table, so from_arrow returns a DataFrame
                # (not a Series).
                df_chunk = cast(pl.DataFrame, pl.from_arrow(arrow_table))
                del arrow_table  # Arrow copy no longer needed; Polars owns the data
                yield df_chunk, batch_metadata

            if failures:
                raise WorkerPoolError(failures, n_total)
    finally:
        # Clean up the temp Arrow IPC dir used to share the transcriptome.
        if _ipc_tmp:
            shutil.rmtree(_ipc_tmp, ignore_errors=True)
