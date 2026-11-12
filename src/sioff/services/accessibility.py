import glob
import itertools
import math
import multiprocessing
import os
import shutil
import zipfile
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Optional, Union, cast

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import RNA
from loguru import logger


class AccessibilityError(Exception):
    """Base exception for accessibility service."""

    pass


class MissingStrandError(AccessibilityError):
    """The contig's profile has no rows for the strand looked up."""


class ProfileNotFoundError(AccessibilityError):
    """No accessibility profile exists for the contig looked up."""


_COMPLEMENT_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")
_GAS_CONSTANT = 0.001987  # kcal / (mol·K)


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT_TABLE)[::-1]


def site_cells(
    strand: str,
    starts: np.ndarray,
    ends: np.ndarray,
    seq_len: int | np.ndarray,
    n_u: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map RIsearch hits to cells of one strand's opening-energy profile.

    The single place that turns a hit into a profile index; every lookup path
    (dense, in-memory frames, memory-mapped cache, on-target) goes through it.

    A profile row ``i`` (0-based) and column ``k`` hold the energy to open the
    ``k + 1`` bases ENDING at position ``i + 1`` of the molecule that was folded
    (RNAplfold's ``-u`` convention). A hit ``[s, e]`` (1-based, inclusive, on the
    chromosome) needs the stretch covering exactly ``[s, e]`` on its own
    transcript, with the column clamped to the widest ``u``:

    * ``'+'``: the forward sequence; the stretch ends at ``e`` -> row ``e - 1``.
    * ``'-'``: the reverse complement, which ``_fold_full_chromosome`` folds and
      writes in its own coordinates (RC position ``p`` = chromosome position
      ``N - p + 1``). ``[s, e]`` is RC ``[N - e + 1, N - s + 1]``, ending at
      ``N - s + 1`` -> row ``N - s``. ``seq_len`` must therefore be the full
      contig length ``N``; the writer emits every position, so the strand's
      maximum position is ``N``.

    Returns ``(rows, cols, in_bounds)`` as int64/int64/bool arrays. Rows of
    out-of-bounds hits (``s < 1`` or ``e > N``) are clipped into range so the
    gather stays valid; callers must use ``in_bounds`` to discard them.

    ``seq_len`` is one length for every hit, or one length per hit (the packed
    cache looks hits of many records up in one call).
    """
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    in_bounds = (s >= 1) & (e <= seq_len)
    cols = np.maximum(np.minimum(e - s + 1, n_u) - 1, 0)
    if strand == "+":
        rows = e - 1
    elif strand == "-":
        rows = seq_len - s
    else:
        raise AccessibilityError(f"strand must be '+' or '-', got {strand!r}")
    rows = np.clip(rows, 0, np.maximum(seq_len - 1, 0))
    return rows, cols, in_bounds


def check_strand_lengths(
    chrom: str, plus_len: Optional[int], minus_len: Optional[int], source: str
) -> bool:
    """Warn when a contig's two strand profiles have different lengths.

    `site_cells` maps a minus-strand hit to row ``N - s`` using the minus
    profile's length as the contig length ``N``. That holds because the writer
    emits every position of both strands; a profile that lost positions at an
    end (truncated or hand-edited Parquet, a future writer change) would shift
    every minus-strand lookup on the contig without any other symptom. Both
    strands of one contig must have the same length, so a mismatch is the
    cheapest observable sign of that. Unknown lengths (``None``) and empty
    strands (``0``, a single-strand profile; see `_warn_missing_strand`) are
    skipped.

    Returns ``True`` when the lengths agree or cannot be compared.
    """
    if not plus_len or not minus_len or plus_len == minus_len:
        return True
    logger.warning(
        f"Accessibility profile for {chrom} ({source}): '+' covers {plus_len} "
        f"positions but '-' covers {minus_len}. Minus-strand opening energies are "
        "looked up relative to the contig length, so they are unreliable for this "
        "contig; regenerate the profile with `sioff accessibility`."
    )
    return False


_OUT_OF_BOUNDS_ENERGY = (
    10.0  # the `np.full` default of annotate_opening_energy_vectorized
)

# Names kept per kind in a lookup's default-energy counters.
_MAX_DEFAULT_EXAMPLES = 10


def new_default_energy_stats() -> dict:
    """Counters of the hits one lookup left at the default opening energy.

    `GenomeAccessibilityService.annotate_opening_energy_vectorized` adds to
    one when given it and never logs: a worker sees only its own siRNA, so
    the engine sums the counters and reports once, in the parent process.
    """
    return {
        "n_hits": 0,  # hits looked up
        "n_found": 0,  # hits that got a profile value
        "n_unknown": 0,  # hits on a sequence without a profile
        "n_out_of_bounds": 0,  # hits outside their sequence's profile
        "unknown": [],  # up to _MAX_DEFAULT_EXAMPLES sequence names
        "out_of_bounds": [],  # up to _MAX_DEFAULT_EXAMPLES sequence names
    }


def add_default_examples(examples: list, names: Iterable[str]) -> None:
    """Append the new ``names`` to ``examples`` until it holds the maximum."""
    for name in names:
        if len(examples) >= _MAX_DEFAULT_EXAMPLES:
            return
        if name not in examples:
            examples.append(str(name))


# Strands `_warn_missing_strand` has reported in this process.
_WARNED_MISSING_STRANDS: set[str] = set()


def _warn_missing_strand(chrom: str, strand: str) -> None:
    """Warn that hits on a strand the profile lacks keep the default energy.

    A profile folded with ``--strand +`` has no ``-`` rows, so without
    ``--sense-only`` every minus-strand hit would get the default opening
    energy silently. Reported once per process and strand: a worker meets it
    on every contig of such a profile.
    """
    if strand in _WARNED_MISSING_STRANDS:
        return
    _WARNED_MISSING_STRANDS.add(strand)
    logger.warning(
        f"The accessibility profile for {chrom} has no {strand!r} strand (folded "
        "with `sioff accessibility --strand +`?): hits on that strand keep the "
        f"default opening energy of {_OUT_OF_BOUNDS_ENERGY} kcal/mol, here and on "
        "any other contig without it. Pass --sense-only (sense_only=True) to drop "
        "them, or recompute the profiles with --strand both."
    )


def fold_sequence(
    sequence: str,
    window_size: int = 80,
    max_span: int = 40,
    unpaired_prob: int = 30,
    temperature: float = 37.0,
) -> np.ndarray:
    """Fold a single sequence and return opening energies (no instance state).

    Module-level so callers that only need a profile (the in-memory core, the
    on-target path) can fold without constructing a service or touching disk.

    Returns a 2D float32 array [seq_len, unpaired_prob]; result[pos, u-1] is the
    opening energy for length u at 0-based position pos.
    """
    seq_len = len(sequence)
    if seq_len == 0:
        return np.array([], dtype=np.float32)

    rna_seq = sequence.upper().replace("T", "U")
    w = min(window_size, seq_len)
    max_span_adj = min(max_span, seq_len)
    RT = _GAS_CONSTANT * (temperature + 273.15)

    profile = np.full((seq_len, unpaired_prob), 25.5, dtype=np.float32)

    try:
        md = RNA.md()
        md.temperature = temperature
        md.window_size = w
        md.max_bp_span = max_span_adj
        fc = RNA.fold_compound(rna_seq, md, RNA.OPTION_WINDOW)
        probs_matrix: list[list[float] | None] = [None] * (seq_len + 2)

        def _cb(v, v_size, i, maxsize, what, data, _pm=probs_matrix):
            if (what & RNA.PROBS_WINDOW_UP) and v is not None:
                _pm[i] = list(v)

        fc.probs_window(unpaired_prob, RNA.PROBS_WINDOW_UP, _cb)
        for i in range(1, seq_len + 1):
            if i < len(probs_matrix):
                row = probs_matrix[i]
                if row is not None:
                    for u in range(1, unpaired_prob + 1):
                        if u < len(row):
                            p = row[u]
                            if p is not None and p > 0:
                                profile[i - 1, u - 1] = -RT * np.log(p)

        logger.debug(f"Computed accessibility for sequence of length {seq_len}")
    except Exception as e:
        logger.error(f"ViennaRNA error computing accessibility: {e}")
        raise AccessibilityError(f"Failed to compute accessibility: {e}") from e

    return profile


OUTPUT_FORMATS = ("npy", "parquet")
# "+" folds the sequence as given only: a transcriptome FASTA already holds each
# target in its sense orientation, so its reverse complement is never hit.
FOLD_STRANDS = ("both", "+")


def _check_fold_strand(strand: str) -> None:
    if strand not in FOLD_STRANDS:
        raise AccessibilityError(
            f"strand must be one of {FOLD_STRANDS}, got {strand!r}"
        )


def _strands_to_fold(sequence: str, strand: str) -> tuple[tuple[str, str], ...]:
    """``(strand, sequence)`` pairs to fold: ``+``, then ``-`` unless ``strand="+"``."""
    _check_fold_strand(strand)
    plus = (("+", sequence),)
    return plus if strand == "+" else plus + (("-", _reverse_complement(sequence)),)


def _fold_strand_chunks(
    seq: str,
    window_size: int,
    max_span: int,
    unpaired_prob: int,
    chunk_size: int,
    temperature: float,
):
    """Fold one strand in chunks; yield ``(pos0, u_matrix)`` per chunk.

    ``pos0`` is the chunk's first 0-based position and ``u_matrix`` its
    float32 ``[n_pos, unpaired_prob]`` opening energies, rounded to 0.1.
    """
    RT = _GAS_CONSTANT * (temperature + 273.15)
    seq_len = len(seq)
    rna_seq = seq.upper().replace("T", "U")
    pos = 0

    while pos < seq_len:
        ext_start = max(0, pos - window_size)
        ext_end = min(seq_len, pos + chunk_size + window_size)
        chunk_seq = rna_seq[ext_start:ext_end]
        chunk_len = len(chunk_seq)

        data_start_in_chunk = pos - ext_start
        data_end_in_chunk = min(seq_len, pos + chunk_size) - ext_start

        w = min(window_size, chunk_len)
        l_adj = min(max_span, chunk_len)

        try:
            md = RNA.md()
            md.temperature = temperature
            md.window_size = w
            md.max_bp_span = l_adj
            fc = RNA.fold_compound(chunk_seq, md, RNA.OPTION_WINDOW)
            probs_matrix: list[list[float] | None] | None = [None] * (chunk_len + 2)

            # `probs_matrix` is non-None here (just assigned); cast the
            # default so the callback param is a list, not `list | None`.
            def _cb(v, v_size, i, maxsize, what, data, _pm=cast(list, probs_matrix)):
                if (what & RNA.PROBS_WINDOW_UP) and v is not None:
                    _pm[i] = list(v)

            fc.probs_window(unpaired_prob, RNA.PROBS_WINDOW_UP, _cb)
            probs_ok = True
        except Exception:
            probs_ok = False
            probs_matrix = None

        n_pos = data_end_in_chunk - data_start_in_chunk
        u_matrix = np.full((n_pos, unpaired_prob), 25.5, dtype=np.float32)

        if probs_ok and probs_matrix is not None:
            for i in range(n_pos):
                chunk_pos_1based = data_start_in_chunk + i + 1
                if chunk_pos_1based >= len(probs_matrix):
                    continue
                row_probs = probs_matrix[chunk_pos_1based]
                if row_probs is None:
                    continue
                for u in range(1, unpaired_prob + 1):
                    if u >= len(row_probs):
                        break
                    p = row_probs[u]
                    if p is not None and p > 0:
                        ev = -RT * math.log(p)
                        u_matrix[i, u - 1] = round(ev * 10.0) / 10.0
        else:
            u_matrix[:] = 10.0  # folding error

        yield pos, u_matrix
        pos += chunk_size


def _fold_full_chromosome(
    sequence: str,
    chrom: str,
    window_size: int,
    max_span: int,
    unpaired_prob: int,
    output_path: str,
    chunk_size: int = 1_000_000,
    temperature: float = 37.0,
    output_format: str = "parquet",
    strand: str = "both",
) -> str:
    """Fold both strands of a chromosome in chunks and stream them to disk.

    ``output_format="parquet"`` writes ``output_path`` (the
    ``{chrom}.accessibility.parquet`` schema); ``"npy"`` writes the uint8
    cache files of `build_accessibility_cache` for both strands into the
    directory ``output_path``, byte for byte what that function builds from
    the Parquet, and returns the plus-strand ``.npy`` path.

    ``strand="+"`` folds the forward strand only: the Parquet holds no ``-``
    rows, and the npy output an empty ``(0, u)`` ``-`` cache, as
    `build_accessibility_cache` makes from that Parquet.

    Top-level (not a method) so ProcessPoolExecutor can pickle it; only the
    output path string is pickled back, avoiding O(N×u) OOM on large chromosomes.
    """
    if output_format not in OUTPUT_FORMATS:
        raise AccessibilityError(
            f"output_format must be one of {OUTPUT_FORMATS}, got {output_format!r}"
        )
    strands = _strands_to_fold(sequence, strand)

    def chunks(seq: str):
        return _fold_strand_chunks(
            seq, window_size, max_span, unpaired_prob, chunk_size, temperature
        )

    if output_format == "npy":
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for s, seq in strands:
            _write_strand_cache(out_dir, chrom, s, len(seq), unpaired_prob, chunks(seq))
        if strand == "+":
            _write_strand_cache(out_dir, chrom, "-", 0, unpaired_prob, ())
        return str(cache_paths(out_dir, chrom, "+")[0])

    schema = pa.schema(
        [("position", pa.int32()), ("strand", pa.utf8())]
        + [(f"u{i}", pa.float32()) for i in range(1, unpaired_prob + 1)]
    )
    with pq.ParquetWriter(output_path, schema) as writer:
        for s, seq in strands:
            for pos0, u_matrix in chunks(seq):
                n_pos = len(u_matrix)
                arrays = [
                    pa.array(np.arange(pos0 + 1, pos0 + n_pos + 1, dtype=np.int32)),
                    pa.array([s] * n_pos, type=pa.utf8()),
                ] + [pa.array(u_matrix[:, u]) for u in range(unpaired_prob)]
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))

    return output_path


# ---------------------------------------------------------------------------
# Memory-mapped per-(contig, strand) cache
#
# `_fold_full_chromosome` writes one Parquet file per contig, or (npy) this
# cache directly through `_write_strand_cache`; the CLI default, npy, folds a
# FASTA into the packed layout below (`_pack_fasta_records`). Reading a whole
# chromosome back into a dense float32 array costs 30 GB per strand on chr1,
# and off-target batch runs do that once per worker process. The cache below
# re-encodes each strand as a position-major uint8 matrix that every worker
# memory-maps, so the OS page cache holds a single shared copy and a worker's
# resident set is only the pages its own hits touch.
#
# Encoding: code = round(E * 10), 0..255, so 25.5 (the "not foldable" sentinel
# and the value of positions absent from the Parquet) is code 255. The
# Parquet values are already rounded to 0.1 and every consumer quantises to
# 0.1 again, so the codes reproduce them bit for bit: fl32(code)/fl32(10) is
# the same float32 as the writer's fl32(round(ev*10)/10) because k/10 is never
# close enough to a float32 rounding boundary for double rounding to occur.
# Entries a code cannot represent exactly - opening energies above 25.5
# (they exist: chr1 peaks at 35.0), negative or unrounded values - go to a
# sorted sidecar of (flat index, exact float32), consulted only where the
# code is 255. Real data put ~1e-6 of the entries there, so the cache stays
# lossless for any input while costing one byte per value. The byte layout
# equals the legacy pipeline's ``*.open.acc.bin`` (position-major, 30 columns).
# ---------------------------------------------------------------------------

PARQUET_SUFFIX = ".accessibility.parquet"
CACHE_SUFFIX = ".access.npy"
EXCEPTIONS_SUFFIX = ".access.exceptions.npz"
_STRAND_NAMES = {"+": "plus", "-": "minus"}
_SCALE = np.float32(10.0)
_SENTINEL_CODE = 255
_DEFAULT_BATCH_ROWS = 1_000_000

# Packed layout: every record of a strand in one pair of files,
# ``accessibility.<plus|minus>.packed.npy`` (uint8 ``[total_positions, n_u]``,
# the records' code rows back to back, same encoding as the per-record cache)
# and ``accessibility.<plus|minus>.packed.npz`` (``names``, ``offsets``,
# ``lengths``, ``index``, ``value``; ``index`` is the global flat index
# ``row * n_u + col`` of the exact-value exceptions, sorted).
PACKED_STEM = "accessibility"
PACKED_SUFFIX = ".packed.npy"
PACKED_INDEX_SUFFIX = ".packed.npz"
_PACK_TASK_POSITIONS = 2_000_000

# What reading a damaged cache file raises: ``np.load`` of a truncated
# ``.npz`` fails in ``zipfile``, of a truncated ``.npy`` with ValueError.
_CORRUPT_CACHE_ERRORS = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile)


def cache_paths(cache_dir: Path, chrom: str, strand: str) -> tuple[Path, Path]:
    """Return ``(codes .npy, exceptions .npz)`` paths for one contig strand."""
    if strand not in _STRAND_NAMES:
        raise AccessibilityError(f"Unknown strand {strand!r} (expected '+' or '-')")
    stem = f"{chrom}_{_STRAND_NAMES[strand]}"
    return (
        Path(cache_dir) / f"{stem}{CACHE_SUFFIX}",
        Path(cache_dir) / f"{stem}{EXCEPTIONS_SUFFIX}",
    )


def packed_paths(cache_dir: Path, strand: str) -> tuple[Path, Path]:
    """Return ``(codes .npy, index .npz)`` paths of one strand's packed cache."""
    if strand not in _STRAND_NAMES:
        raise AccessibilityError(f"Unknown strand {strand!r} (expected '+' or '-')")
    stem = f"{PACKED_STEM}.{_STRAND_NAMES[strand]}"
    return (
        Path(cache_dir) / f"{stem}{PACKED_SUFFIX}",
        Path(cache_dir) / f"{stem}{PACKED_INDEX_SUFFIX}",
    )


def has_profiles(directory: Union[str, Path]) -> bool:
    """Report whether a directory holds any accessibility profile.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory to inspect.

    Returns
    -------
    bool
        True when it holds a ``*.packed.npy`` or ``*.access.npy`` cache file or
        a ``*.accessibility.parquet`` file; False otherwise, including when the
        directory does not exist.
    """
    d = Path(directory)
    if not d.is_dir():
        return False
    return any(
        any(d.glob(f"*{suffix}"))
        for suffix in (PACKED_SUFFIX, CACHE_SUFFIX, PARQUET_SUFFIX)
    )


def _chrom_of(parquet_path: Path) -> str:
    name = parquet_path.name
    if not name.endswith(PARQUET_SUFFIX):
        raise AccessibilityError(
            f"{parquet_path} is not a '<chrom>{PARQUET_SUFFIX}' accessibility file"
        )
    return name[: -len(PARQUET_SUFFIX)]


def _u_columns(names: Iterable[str]) -> list[str]:
    return sorted(
        [c for c in names if c.startswith("u") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )


def _encode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode float32 kcal/mol to uint8 codes; also flag the inexact entries.

    Uses the very ``np.round(v * 10)`` the consumers apply, so a decoded code
    re-quantises to the same value the dense in-memory path produces.
    """
    k = np.round(values * _SCALE)
    codes = np.clip(k, 0, _SENTINEL_CODE).astype(np.uint8)
    decoded = codes.astype(np.float32) / _SCALE
    exact = (decoded == values) & ~np.signbit(values)
    return codes, exact


def _decode(codes: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) / _SCALE


def _stat_value(v):
    return v.decode() if isinstance(v, bytes) else v


def _strand_row_groups(pf: pq.ParquetFile, strand: str) -> tuple[list[int], int]:
    """Row groups that may hold ``strand`` rows, and the strand's max position.

    `_fold_full_chromosome` writes single-strand row groups with statistics,
    so normally this costs no I/O. Row groups without usable statistics (or
    with mixed strands) are scanned on their ``position``/``strand`` columns.
    """
    md = pf.metadata
    names = pf.schema_arrow.names
    i_pos, i_strand = names.index("position"), names.index("strand")
    row_groups: list[int] = []
    scan: list[int] = []
    max_pos = 0
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        st_s = rg.column(i_strand).statistics
        st_p = rg.column(i_pos).statistics
        single = (
            st_s is not None
            and st_s.has_min_max
            and _stat_value(st_s.min) == _stat_value(st_s.max)
        )
        if single:
            if _stat_value(st_s.min) != strand:
                continue
            row_groups.append(i)
            if st_p is not None and st_p.has_min_max:
                max_pos = max(max_pos, int(st_p.max))
            else:
                scan.append(i)
        else:
            row_groups.append(i)
            scan.append(i)
    for i in scan:
        t = pf.read_row_group(i, columns=["position", "strand"])
        here = t.column("position").to_numpy()[
            t.column("strand").to_numpy(zero_copy_only=False) == strand
        ]
        if here.size:
            max_pos = max(max_pos, int(here.max()))
    return row_groups, max_pos


def _tmp_path(target: Path) -> Path:
    """This process's temporary name for ``target``, with the same suffix.

    Keeping the suffix matters: ``np.savez`` appends ``.npz`` to a name
    without it.
    """
    return target.with_name(f"{target.name}.{os.getpid()}.tmp{target.suffix}")


def _write_atomically(target: Path, write) -> None:
    """``write(tmp_path)`` then rename, so readers never see a partial file.

    A concurrent builder of the same target writes its own temporary file and
    the last rename wins; both renames deliver a complete, identical file.
    """
    tmp = _tmp_path(target)
    try:
        write(tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def _fill_strand(
    pf: pq.ParquetFile,
    strand: str,
    u_cols: list[str],
    codes: np.ndarray,
    batch_rows: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Stream one strand's rows of ``pf`` into ``codes``.

    ``codes`` is a 2-D view prefilled with 255 whose row 0 is position 1.
    Returns ``(exc_index, exc_value, n_rows)``: the exact-value exceptions
    (flat indices local to ``codes``) and the number of rows written.
    """
    n_u = len(u_cols)
    row_groups, _ = _strand_row_groups(pf, strand)
    exc_index: list[np.ndarray] = []
    exc_value: list[np.ndarray] = []
    n_rows = 0
    if row_groups:
        batches = pf.iter_batches(
            batch_size=batch_rows,
            row_groups=row_groups,
            columns=["position", "strand", *u_cols],
        )
        for batch in batches:
            mask = batch.column("strand").to_numpy(zero_copy_only=False) == strand
            if not mask.any():
                continue
            pos0 = batch.column("position").to_numpy().astype(np.int64) - 1
            values = np.stack(
                [batch.column(c).to_numpy() for c in u_cols], axis=1
            ).astype(np.float32, copy=False)
            if not mask.all():
                pos0, values = pos0[mask], values[mask]
            block, exact = _encode(values)
            codes[pos0, :] = block
            n_rows += len(pos0)
            if not exact.all():
                r, c = np.nonzero(~exact)
                exc_index.append(pos0[r] * n_u + c)
                exc_value.append(values[r, c])
    return exc_index, exc_value, n_rows


def _build_strand(
    pf: pq.ParquetFile,
    chrom: str,
    strand: str,
    u_cols: list[str],
    cache_dir: Path,
    batch_rows: int,
) -> Path:
    """Build one strand's cache files; the code matrix is freed on return."""
    n_u = len(u_cols)
    _, max_pos = _strand_row_groups(pf, strand)
    codes = np.full((max_pos, n_u), _SENTINEL_CODE, dtype=np.uint8)
    exc_index, exc_value, n_rows = _fill_strand(pf, strand, u_cols, codes, batch_rows)
    index, value = _sorted_exceptions(chrom, strand, exc_index, exc_value, n_rows * n_u)
    npy_path, npz_path = cache_paths(cache_dir, chrom, strand)
    _write_atomically(npz_path, lambda tmp: np.savez(tmp, index=index, value=value))
    _write_atomically(npy_path, lambda tmp: np.save(tmp, codes))
    logger.info(
        f"Built accessibility cache {npy_path.name}: {max_pos:,} positions × {n_u}, "
        f"{len(index):,} exact-value exceptions"
    )
    return npy_path


def _sorted_exceptions(
    chrom: str,
    strand: str,
    exc_index: list[np.ndarray],
    exc_value: list[np.ndarray],
    n_values: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate one strand's exception chunks, sorted by flat index."""
    index = np.concatenate(exc_index) if exc_index else np.empty(0, dtype=np.int64)
    value = np.concatenate(exc_value) if exc_value else np.empty(0, dtype=np.float32)
    order = np.argsort(index, kind="stable")
    index, value = index[order], value[order]
    if n_values and len(index) > 0.01 * n_values:
        logger.warning(
            f"{chrom} {strand}: {len(index):,} of {n_values:,} values are not "
            "multiples of 0.1 kcal/mol (or exceed 25.5); the exceptions sidecar "
            "will be large. Write accessibility with 'sioff accessibility' for a "
            "compact cache."
        )
    return index, value


def _encode_chunks(
    codes: np.ndarray, chunks: Iterable[tuple[int, np.ndarray]], row0: int = 0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Encode folded ``(pos0, float32 block)`` chunks into ``codes`` from row ``row0``.

    Chunk ``pos0`` lands at row ``row0 + pos0``. Returns the entries a code
    cannot hold exactly, one array per chunk that has any: their flat indices
    in ``codes`` (``row * n_u + col``, int64) and their float32 values.
    """
    n_u = codes.shape[1]
    exc_index: list[np.ndarray] = []
    exc_value: list[np.ndarray] = []
    for pos0, values in chunks:
        start = row0 + pos0
        block, exact = _encode(values)
        codes[start : start + len(values)] = block
        if not exact.all():
            r, c = np.nonzero(~exact)
            exc_index.append((start + r).astype(np.int64) * n_u + c)
            exc_value.append(values[r, c])
    return exc_index, exc_value


def _write_strand_cache(
    cache_dir: Path,
    chrom: str,
    strand: str,
    n_pos: int,
    n_u: int,
    chunks: Iterable[tuple[int, np.ndarray]],
) -> Path:
    """Encode folded ``(pos0, float32 block)`` chunks straight into one strand's cache.

    The ``.npy`` is filled through a memory map, so peak memory is one chunk,
    and the files equal what `build_accessibility_cache` makes from the
    Parquet of the same chunks. As there, the sidecar lands before the codes
    and each file is renamed into place complete.
    """
    npy_path, npz_path = cache_paths(cache_dir, chrom, strand)
    tmp = npy_path.with_name(f"{npy_path.name}.{os.getpid()}.tmp{npy_path.suffix}")
    try:
        codes = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.uint8, shape=(n_pos, n_u)
        )
        exc_index, exc_value = _encode_chunks(codes, chunks)
        codes.flush()
        del codes
        index, value = _sorted_exceptions(
            chrom, strand, exc_index, exc_value, n_pos * n_u
        )
        _write_atomically(npz_path, lambda t: np.savez(t, index=index, value=value))
        os.replace(tmp, npy_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    logger.info(
        f"Wrote accessibility cache {npy_path.name}: {n_pos:,} positions × {n_u}, "
        f"{len(index):,} exact-value exceptions"
    )
    return npy_path


def build_accessibility_cache(
    parquet_path: Union[str, Path],
    cache_dir: Union[str, Path],
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> dict[str, Path]:
    """Re-encode one contig's Parquet into the per-strand uint8 cache files.

    This is the v0.2.0 per-record layout, kept as the reference the packed
    cache (`pack_parquet_profiles`) is tested against; the pipeline itself
    packs a Parquet directory with `ensure_accessibility_cache`.

    Streams the Parquet by row group (never more than ``batch_rows`` rows in
    flight) into a preallocated ``uint8[max_position, n_u]`` per strand, so
    peak memory is one strand's cache (chr1: 7.5 GB) plus a batch. Positions
    absent from the Parquet keep code 255 (25.5 kcal/mol), as in the dense
    in-memory path; positions must be unique within a strand, as the writer
    guarantees. A strand with no rows gets an empty ``(0, n_u)`` cache so the
    build stays idempotent, and queries on it raise like the dense path.

    Files are written to a temporary name and renamed into place. Each
    ``.npy`` is dated no earlier than the Parquet it was read from, so the
    age check of `_cache_is_stale` holds even when the Parquet's mtime lies
    ahead of this machine's clock.

    Returns
    -------
    dict[str, pathlib.Path]
        ``{"+": <plus .npy>, "-": <minus .npy>}``.
    """
    parquet_path = Path(parquet_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    chrom = _chrom_of(parquet_path)
    # Taken before reading: a Parquet rewritten mid-build stays newer.
    folded = parquet_path.stat().st_mtime_ns
    pf = pq.ParquetFile(parquet_path)
    u_cols = _u_columns(pf.schema_arrow.names)
    if not u_cols:
        raise AccessibilityError(f"{parquet_path} has no u1..uN columns")
    built = {
        strand: _build_strand(pf, chrom, strand, u_cols, cache_dir, batch_rows)
        for strand in ("+", "-")
    }
    check_strand_lengths(
        chrom,
        int(np.load(built["+"], mmap_mode="r").shape[0]),
        int(np.load(built["-"], mmap_mode="r").shape[0]),
        str(parquet_path),
    )
    for npy_path in built.values():
        _date_no_earlier_than(npy_path, folded)
    return built


def _warn_if_disk_is_small(directory: Path, nbytes: int) -> None:
    """Warn when ``directory`` has less free space than a pack about to be written.

    The packed ``.npy`` is a sparse file filled through a memory map, so a
    full disk surfaces as SIGBUS (the process dies, no error message) rather
    than as an ``OSError``. Only a warning: a compressing filesystem stores
    the codes in far less than their logical size.
    """
    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        return
    if free < nbytes:
        logger.warning(
            f"Writing the packed accessibility cache into {directory} needs "
            f"{nbytes / 1e9:.1f} GB but only {free / 1e9:.1f} GB is free; the run "
            "may die when the disk fills (a bus error, as the cache is written "
            "through a memory map)."
        )


def _stage_index(
    npz_path: Path,
    names: list[str],
    offsets: Union[np.ndarray, Sequence[int]],
    lengths: Union[np.ndarray, Sequence[int]],
    index: np.ndarray,
    value: np.ndarray,
) -> Path:
    """Write a packed ``.npz`` index under this process's temporary name.

    A write that fails part-way (a full disk) removes its partial file: the
    caller only learns the temporary name once the index is complete.
    """
    tmp = _tmp_path(npz_path)
    try:
        np.savez(
            tmp,
            names=np.array(names, dtype=str),
            offsets=np.asarray(offsets, dtype=np.int64),
            lengths=np.asarray(lengths, dtype=np.int64),
            index=index,
            value=value,
        )
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _batch_records(
    sizes: Sequence[int], target: int = _PACK_TASK_POSITIONS
) -> list[list[int]]:
    """Group record indices into batches of roughly ``target`` positions.

    Records are taken longest first and added to the current batch while its
    total stays within ``target``; a record of ``target`` positions or more
    is a batch of its own. Every index appears exactly once.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    total = 0
    for i in sorted(range(len(sizes)), key=lambda k: -sizes[k]):
        size = sizes[i]
        if size >= target:
            batches.append([i])
            continue
        if current and total + size > target:
            batches.append(current)
            current, total = [], 0
        current.append(i)
        total += size
    if current:
        batches.append(current)
    return batches


def _pack_parquet_batch(
    jobs: list[tuple[str, str, int, int]],
    u_cols: list[str],
    tmp_paths: dict[str, str],
    batch_rows: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fill the records of one batch into the packed temp ``.npy`` files.

    Each job is ``(parquet path, strand, row offset, length)``. Returns, per
    strand, only the exceptions as ``(global flat index int64, value float32)``.
    Top-level so spawned workers can pickle it.
    """
    n_u = len(u_cols)
    maps = {s: np.load(p, mmap_mode="r+") for s, p in tmp_paths.items()}
    found: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        s: ([], []) for s in maps
    }
    # One open ParquetFile at a time: a batch can hold thousands of records,
    # and a descriptor per record would exhaust the open-file limit.
    for path, group in itertools.groupby(jobs, key=lambda j: j[0]):
        with pq.ParquetFile(path) as pf:
            for _, strand, off, n in group:
                view = maps[strand][off : off + n]
                view[:] = _SENTINEL_CODE
                idx, val, _ = _fill_strand(pf, strand, u_cols, view, batch_rows)
                found[strand][0].extend(i + off * n_u for i in idx)
                found[strand][1].extend(val)
    for mm in maps.values():
        mm.flush()
    return _concat_exceptions(found)


def _concat_exceptions(
    found: Mapping[str, tuple[list[np.ndarray], list[np.ndarray]]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per strand, a worker's exception chunks as one (int64, float32) pair."""
    return {
        s: (
            np.concatenate(i).astype(np.int64) if i else np.empty(0, np.int64),
            np.concatenate(v).astype(np.float32) if v else np.empty(0, np.float32),
        )
        for s, (i, v) in found.items()
    }


def pack_parquet_profiles(
    parquet_dir: Path,
    cache_dir: Path,
    workers: int = 1,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> list[str]:
    """Pack every ``*.accessibility.parquet`` of a directory into one pair per strand.

    Records are laid out in sorted file-name order. Each strand's index uses
    that strand's own maximum position per record, so ``lengths`` may differ
    between the plus and minus ``.npz``. A strand with no rows in any Parquet
    gets no pair, and an existing pair for it in ``cache_dir`` is deleted.

    The code matrix is filled through a memory map by up to
    ``min(workers, 4)`` spawned processes, so it is never held in memory.
    Every strand's files are written under temporary names first and only
    then renamed into place, each ``.npz`` before its ``.npy``, so a failure
    leaves the previous pack whole. Each ``.npy`` is dated no earlier than
    the newest Parquet (read before packing).

    Parameters
    ----------
    parquet_dir : pathlib.Path
        Directory of ``<record>.accessibility.parquet`` files.
    cache_dir : pathlib.Path
        Output directory, created if missing.
    workers : int
        Number of worker processes (capped at 4 and at the number of batches).
    batch_rows : int
        Maximum Parquet rows in flight per worker.

    Returns
    -------
    list[str]
        The record names, in packed order.

    Raises
    ------
    AccessibilityError
        If there is no Parquet, a file has no ``u`` columns, or the files'
        ``u`` columns differ.
    """
    parquet_dir, cache_dir = Path(parquet_dir), Path(cache_dir)
    files = sorted(parquet_dir.glob(f"*{PARQUET_SUFFIX}"), key=lambda p: p.name)
    if not files:
        raise AccessibilityError(f"No *{PARQUET_SUFFIX} files in {parquet_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Taken before reading: a Parquet rewritten mid-pack stays newer.
    newest = max(f.stat().st_mtime_ns for f in files)
    names = [_chrom_of(f) for f in files]
    u_cols: list[str] = []
    lengths: dict[str, list[int]] = {s: [] for s in _STRAND_NAMES}
    for f, name in zip(files, names):
        pf = pq.ParquetFile(f)
        cols = _u_columns(pf.schema_arrow.names)
        if not cols:
            raise AccessibilityError(f"{f} has no u1..uN columns")
        if u_cols and cols != u_cols:
            raise AccessibilityError(
                f"{f.name} has {len(cols)} u columns, but {files[0].name} has "
                f"{len(u_cols)}; pack Parquet files folded with the same -u"
            )
        u_cols = cols
        for strand in _STRAND_NAMES:
            lengths[strand].append(_strand_row_groups(pf, strand)[1])
        check_strand_lengths(name, lengths["+"][-1], lengths["-"][-1], str(f))
    n_u = len(u_cols)

    offsets = {
        s: np.concatenate([[0], np.cumsum(n, dtype=np.int64)[:-1]]).astype(np.int64)
        for s, n in lengths.items()
    }
    totals = {s: int(sum(n)) for s, n in lengths.items()}
    strands = [s for s in _STRAND_NAMES if totals[s]]

    _warn_if_disk_is_small(cache_dir, sum(totals.values()) * n_u)
    tmp_paths: dict[str, Path] = {}
    staged: list[tuple[Path, Path]] = []
    try:
        for strand in strands:
            tmp = _tmp_path(packed_paths(cache_dir, strand)[0])
            tmp_paths[strand] = tmp
            mm = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=np.uint8, shape=(totals[strand], n_u)
            )
            mm.flush()
            del mm

        sizes = [lengths["+"][i] + lengths["-"][i] for i in range(len(files))]
        batches = _batch_records(sizes)
        tasks = [
            (
                [
                    (str(files[i]), s, int(offsets[s][i]), lengths[s][i])
                    for i in batch
                    for s in strands
                    if lengths[s][i]
                ],
                u_cols,
                {s: str(p) for s, p in tmp_paths.items()},
                batch_rows,
            )
            for batch in batches
        ]
        n_workers = min(workers, 4, len(tasks))
        if n_workers <= 1:
            results = [_pack_parquet_batch(*t) for t in tasks]
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                results = list(pool.map(_pack_parquet_batch, *zip(*tasks)))

        label = "accessibility (packed)"
        n_exceptions: dict[str, int] = {}
        for strand in strands:
            index, value = _sorted_exceptions(
                label,
                strand,
                [r[strand][0] for r in results if strand in r],
                [r[strand][1] for r in results if strand in r],
                totals[strand] * n_u,
            )
            n_exceptions[strand] = len(index)
            npy_path, npz_path = packed_paths(cache_dir, strand)
            npz_tmp = _stage_index(
                npz_path, names, offsets[strand], lengths[strand], index, value
            )
            staged += [(npz_tmp, npz_path), (tmp_paths[strand], npy_path)]
        for tmp, final in staged:
            os.replace(tmp, final)
        # A strand no Parquet has any more loses its old pair only now, once
        # the new pack is in place: a failure above leaves the old pack whole.
        for strand in _STRAND_NAMES:
            if not totals[strand]:
                for path in packed_paths(cache_dir, strand):
                    path.unlink(missing_ok=True)
        for strand in strands:
            npy_path = packed_paths(cache_dir, strand)[0]
            _date_no_earlier_than(npy_path, newest)
            logger.info(
                f"Packed {len(names):,} records into {npy_path.name}: "
                f"{totals[strand]:,} positions × {n_u}, {n_exceptions[strand]:,} "
                "exact-value exceptions"
            )
    finally:
        for tmp in [*tmp_paths.values(), *(t for t, _ in staged)]:
            tmp.unlink(missing_ok=True)
    return names


def _fold_records_into_pack(
    tmp_npy: dict[str, str],
    records: list[tuple[str, str, int]],
    window_size: int,
    max_span: int,
    unpaired_prob: int,
    chunk_size: int,
    temperature: float,
    strand: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fold one batch of FASTA records into the packed temp ``.npy`` files.

    Each record is ``(name, sequence, row offset)``. Its strands are folded in
    chunks as `_fold_full_chromosome` folds them, and each chunk's codes land
    at row ``offset + pos0`` of that strand's map, ``tmp_npy[strand]``, opened
    ``r+``. A zero-length record has no rows and is skipped. Returns, per
    strand, only the exceptions as ``(global flat index int64, value
    float32)``. Top-level so worker processes can pickle it.
    """
    maps = {s: np.load(p, mmap_mode="r+") for s, p in tmp_npy.items()}
    found: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        s: ([], []) for s in maps
    }
    for _, sequence, offset in records:
        if not sequence:
            continue
        for s, seq in _strands_to_fold(sequence, strand):
            chunks = _fold_strand_chunks(
                seq, window_size, max_span, unpaired_prob, chunk_size, temperature
            )
            index, value = _encode_chunks(maps[s], chunks, offset)
            found[s][0].extend(index)
            found[s][1].extend(value)
    for mm in maps.values():
        mm.flush()
    return _concat_exceptions(found)


def _remove_interrupted_fold_temps(cache_dir: Path) -> None:
    """Delete packed temporary files a killed fold left in ``cache_dir``.

    Temporary names carry the writer's pid, so a run killed by SIGKILL, a
    batch-system time limit or SIGBUS (disk full) leaves files of the full
    cache size that no later run would replace. One ``sioff accessibility``
    run writes a directory at a time, so any such file is an orphan. The
    Parquet packer does not do this: two off-target runs may pack one
    directory concurrently.
    """
    for strand in _STRAND_NAMES:
        for target in packed_paths(cache_dir, strand):
            pattern = f"{glob.escape(target.name)}.*.tmp{target.suffix}"
            for orphan in cache_dir.glob(pattern):
                logger.info(f"Removing {orphan}, left by an interrupted run")
                orphan.unlink(missing_ok=True)


def _pack_fasta_records(
    records: Sequence[tuple[str, str]],
    cache_dir: Path,
    window_size: int,
    max_span: int,
    unpaired_prob: int,
    chunk_size: int,
    temperature: float,
    strand: str,
    workers: int,
    on_batch: Callable[[int], None],
) -> Path:
    """Fold FASTA ``(name, sequence)`` records into one packed pair per strand.

    Records keep their FASTA order, and both strands share ``offsets`` and
    ``lengths`` (a reverse complement is as long as its record). Each written
    strand's ``.npy`` is preallocated under a temporary name and filled
    through memory maps by `_fold_records_into_pack`, on `_batch_records`
    batches: inline when ``workers`` or the number of batches is 1, else in
    a spawned pool of ``min(workers, batches)`` processes. Only exceptions
    come back to this process, and ``on_batch`` gets the record count of
    every finished batch. Both strands' files are written under temporary
    names before any is renamed into place (each ``.npz`` before its
    ``.npy``), so a failure leaves the previous pack whole; with
    ``strand="+"`` a minus pair left by an earlier run is deleted afterwards.
    Temporary files never outlive the call, and those of an earlier,
    killed run are removed first.

    Returns the plus strand's ``.npy`` path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    _remove_interrupted_fold_temps(cache_dir)
    names = [name for name, _ in records]
    lengths = np.array([len(seq) for _, seq in records], dtype=np.int64)
    offsets = (np.cumsum(lengths) - lengths).astype(np.int64)
    total = int(lengths.sum())
    strands = ["+"] if strand == "+" else ["+", "-"]
    _warn_if_disk_is_small(cache_dir, total * unpaired_prob * len(strands))

    tmp_paths: dict[str, Path] = {}
    staged: list[tuple[Path, Path]] = []
    try:
        for s in strands:
            tmp = _tmp_path(packed_paths(cache_dir, s)[0])
            tmp_paths[s] = tmp
            mm = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=np.uint8, shape=(total, unpaired_prob)
            )
            mm.flush()
            del mm

        tmp_npy = {s: str(p) for s, p in tmp_paths.items()}
        tasks = [
            (
                tmp_npy,
                [(names[i], records[i][1], int(offsets[i])) for i in batch],
                window_size,
                max_span,
                unpaired_prob,
                chunk_size,
                temperature,
                strand,
            )
            for batch in _batch_records(lengths.tolist())
        ]
        results: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
        n_workers = min(workers, len(tasks))
        if n_workers <= 1:
            for task in tasks:
                results.append(_fold_records_into_pack(*task))
                on_batch(len(task[1]))
        else:
            # spawn, as for the Parquet packer and the off-target workers: no
            # fork of a parent whose polars/Rayon threads may be running.
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = {
                    pool.submit(_fold_records_into_pack, *task): len(task[1])
                    for task in tasks
                }
                try:
                    for fut in as_completed(futures):
                        results.append(fut.result())
                        on_batch(futures[fut])
                except BaseException:
                    # A failed run must not go on folding the rest of the FASTA.
                    pool.shutdown(cancel_futures=True)
                    raise

        n_exceptions: dict[str, int] = {}
        for s in strands:
            index, value = _sorted_exceptions(
                "accessibility (packed)",
                s,
                [r[s][0] for r in results],
                [r[s][1] for r in results],
                total * unpaired_prob,
            )
            n_exceptions[s] = len(index)
            npy_path, npz_path = packed_paths(cache_dir, s)
            npz_tmp = _stage_index(npz_path, names, offsets, lengths, index, value)
            staged += [(npz_tmp, npz_path), (tmp_paths[s], npy_path)]
        for tmp, final in staged:
            os.replace(tmp, final)
        for s in strands:
            logger.info(
                f"Folded {len(names):,} records into "
                f"{packed_paths(cache_dir, s)[0].name}: {total:,} positions × "
                f"{unpaired_prob}, {n_exceptions[s]:,} exact-value exceptions"
            )
        if strand == "+":
            # A minus pair from an earlier --strand both run would otherwise
            # keep answering for records this run did not fold.
            for path in packed_paths(cache_dir, "-"):
                path.unlink(missing_ok=True)
    finally:
        for tmp in [*tmp_paths.values(), *(t for t, _ in staged)]:
            tmp.unlink(missing_ok=True)
    return packed_paths(cache_dir, "+")[0]


def _date_no_earlier_than(npy_path: Path, mtime_ns: int) -> None:
    """Move a fresh cache file's mtime up to ``mtime_ns`` if it is earlier.

    A Parquet written on a node whose clock runs ahead would otherwise make
    the cache just built from it look stale, and every run rebuild it. Best
    effort: if the time cannot be set, the next run rebuilds once more.
    """
    try:
        if npy_path.stat().st_mtime_ns < mtime_ns:
            os.utime(npy_path, ns=(mtime_ns, mtime_ns))
    except OSError:
        pass


def _cache_is_valid(npy_path: Path, npz_path: Path) -> bool:
    """Both files present and the ``.npy`` header is a 2-D uint8 array.

    Only the header is checked (dtype, rank), never the content. Whether the
    cache still matches its Parquet is `_cache_is_stale`'s question.
    """
    if not (npy_path.is_file() and npz_path.is_file()):
        return False
    try:
        arr = np.load(npy_path, mmap_mode="r")
    except (OSError, ValueError, EOFError):
        return False
    return arr.ndim == 2 and arr.dtype == np.uint8


def _cache_is_stale(cache_dir: Path, chrom: str, parquet_path: Path) -> bool:
    """Either strand's ``.npy`` in ``cache_dir`` is older than the contig's Parquet.

    Re-folding into the same directory (new ``-W/-L/-u``, a corrected genome)
    rewrites the Parquet but leaves the cache built from the old one, which
    would otherwise keep answering. Only modification times are compared: a
    Parquet copied without its mtime costs an unneeded rebuild, and one given
    an mtime older than its cache goes unnoticed. A missing ``.npy`` is not
    stale, it is missing (see `_cache_is_valid`).
    """
    try:
        folded = parquet_path.stat().st_mtime_ns
    except OSError:
        return False
    for strand in _STRAND_NAMES:
        try:
            if cache_paths(cache_dir, chrom, strand)[0].stat().st_mtime_ns < folded:
                return True
        except OSError:
            continue
    return False


def _warn_stale(chrom: str, cache_dir: Path, parquet_path: Path) -> None:
    logger.warning(
        f"Accessibility cache for {chrom} in {cache_dir} is older than "
        f"{parquet_path}; rebuilding it"
    )


def _pack_is_current(pack_dir: Path, parquet_files: list[Path]) -> bool:
    """Every packed pair in ``pack_dir`` holds every record of these Parquet files.

    For each strand that has any packed file, all three must hold: both files
    are present and pass the header checks, the ``.npz`` ``names`` contain
    every Parquet record (order is ignored: a pack folded from a FASTA keeps
    FASTA order), and the ``.npy`` is no older than the newest Parquet. Names
    without a Parquet (one deleted later, or a FASTA fold of more records)
    are harmless. So a FASTA re-fold into a Parquet directory is kept while
    the Parquet is older than it. A
    directory with no packed file is not current; nor is one whose files
    cannot be read. Content is never read, only the ``names`` list, so this
    stays cheap on a genome-sized pack.
    """
    want = {_chrom_of(f) for f in parquet_files}
    try:
        newest = max(f.stat().st_mtime_ns for f in parquet_files)
    except (OSError, ValueError):
        return False
    found = False
    for strand in _STRAND_NAMES:
        npy_path, npz_path = packed_paths(pack_dir, strand)
        if not (npy_path.exists() or npz_path.exists()):
            continue
        found = True
        if not _cache_is_valid(npy_path, npz_path):
            return False
        try:
            with np.load(npz_path) as side:
                names = np.asarray(side["names"]).tolist()
            if not want.issubset(names) or npy_path.stat().st_mtime_ns < newest:
                return False
        except _CORRUPT_CACHE_ERRORS:
            return False
    return found


def ensure_accessibility_cache(
    parquet_dir: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    workers: int = 1,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> list[str]:
    """Pack a directory of Parquet profiles into the memory-mapped cache.

    Meant to run once in the parent process before an off-target worker pool
    starts, so workers only ever memory-map finished files. In order:

    1. A directory without Parquet (only caches) packs nothing.
    2. A packed pair in ``cache_dir`` or ``parquet_dir`` (searched in that
       order, the first one holding any packed file) that holds every
       record of these Parquet files, in any order, and is no older than the
       newest of them, is kept: a pack of a later FASTA fold into the same
       directory included.
    3. With no pack, and a valid, current per-record cache (the v0.2.0 layout
       of `build_accessibility_cache`) in ``cache_dir`` for every Parquet
       contig, that cache is reused and nothing is packed, so a genome-sized
       cache is never duplicated.
    4. Otherwise `pack_parquet_profiles` packs every Parquet, in sorted
       file-name order. A pack that existed but no longer matches its Parquet
       (the directory was re-folded, a record was added, or a
       pack file cannot be read) is rebuilt with a warning.

    Parameters
    ----------
    parquet_dir : str or pathlib.Path
        Directory of ``<chrom>.accessibility.parquet`` files.
    cache_dir : str or pathlib.Path, optional
        Where the packed files go; created if missing. Defaults to
        ``parquet_dir``.
    workers : int, default 1
        Processes filling the packed matrix: ``min(workers, 4)``.
    batch_rows : int, default 1_000_000
        Rows decoded per Parquet batch while streaming.

    Returns
    -------
    list[str]
        Names of the records that were packed, in pack order; empty when
        nothing had to be built.

    Raises
    ------
    AccessibilityError
        When the profiles cannot be packed; the original exception (an
        ``OSError`` for an unwritable ``cache_dir``) is its ``__cause__``.
    OSError
        When ``cache_dir`` cannot be created.
    """
    parquet_dir = Path(parquet_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else parquet_dir
    files = sorted(parquet_dir.glob(f"*{PARQUET_SUFFIX}"))
    if not files:
        return []
    cache_dir.mkdir(parents=True, exist_ok=True)

    stale_in: Optional[Path] = None
    for directory in dict.fromkeys((cache_dir, parquet_dir)):
        if any(p.exists() for s in _STRAND_NAMES for p in packed_paths(directory, s)):
            if _pack_is_current(directory, files):
                return []
            stale_in = directory
            break
    if stale_in is None and all(
        all(_cache_is_valid(*cache_paths(cache_dir, _chrom_of(f), s)) for s in "+-")
        and not _cache_is_stale(cache_dir, _chrom_of(f), f)
        for f in files
    ):
        return []
    if stale_in is not None:
        logger.warning(
            f"Packed accessibility cache in {stale_in} does not match the "
            f"Parquet profiles in {parquet_dir}; rebuilding it"
        )
    logger.info(
        f"Packing {len(files)} accessibility profile(s) into {cache_dir} "
        f"with {max(1, min(workers, 4))} process(es)"
    )
    try:
        return pack_parquet_profiles(parquet_dir, cache_dir, workers, batch_rows)
    except Exception as e:
        raise AccessibilityError(
            f"Failed to pack accessibility profiles from {parquet_dir}: {e}"
        ) from e


_UNRESOLVED = object()


class _DenseProfile:
    """One strand as a dense float32 ``[n_pos, n_u]`` array (in-memory frames)."""

    __slots__ = ("arr",)

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    @property
    def n_pos(self) -> int:
        return self.arr.shape[0]

    @property
    def n_u(self) -> int:
        return self.arr.shape[1]

    def gather(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return self.arr[rows, cols].astype(np.float32)

    def region(self, start: int, end: int) -> np.ndarray:
        return self.arr[start:end]


class _CachedProfile:
    """One strand as a memory-mapped uint8 code matrix plus its exceptions."""

    __slots__ = ("codes", "exc_index", "exc_value")

    def __init__(self, codes: np.ndarray, exc_index: np.ndarray, exc_value: np.ndarray):
        self.codes = codes
        self.exc_index = exc_index
        self.exc_value = exc_value

    @property
    def n_pos(self) -> int:
        return self.codes.shape[0]

    @property
    def n_u(self) -> int:
        return self.codes.shape[1]

    def _patch(self, values: np.ndarray, flat: np.ndarray, where: np.ndarray) -> None:
        """Overwrite ``values[where]`` with the exact exceptions stored for ``flat``."""
        if self.exc_index.size == 0 or where.size == 0:
            return
        pos = np.searchsorted(self.exc_index, flat)
        pos_c = np.minimum(pos, self.exc_index.size - 1)
        hit = self.exc_index[pos_c] == flat
        values[where[hit]] = self.exc_value[pos_c[hit]]

    def gather(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        # Touch the mapping in position order: sequential page faults let the
        # kernel read ahead instead of seeking per hit.
        order = np.argsort(rows, kind="stable")
        codes = np.asarray(self.codes[rows[order], cols[order]])
        values = _decode(codes)
        sel = np.flatnonzero(codes == _SENTINEL_CODE)
        if sel.size:
            flat = rows[order][sel].astype(np.int64) * self.n_u + cols[order][sel]
            self._patch(values, flat, sel)
        out = np.empty_like(values)
        out[order] = values
        return out

    def region(self, start: int, end: int) -> np.ndarray:
        values = _decode(np.asarray(self.codes[start:end]))
        lo = np.searchsorted(self.exc_index, start * self.n_u)
        hi = np.searchsorted(self.exc_index, end * self.n_u)
        if hi > lo:
            flat = values.reshape(-1)
            flat[self.exc_index[lo:hi] - start * self.n_u] = self.exc_value[lo:hi]
        return values


_Profile = Union[_DenseProfile, _CachedProfile]


def _name_keys(names: np.ndarray) -> np.ndarray:
    """Record names as UTF-8 bytes (``S`` dtype), the packed reader's lookup keys.

    numpy's ``str`` dtype takes four bytes per character of the longest name:
    385k GENCODE IDs of up to 151 characters come to 233 MB per copy, a
    quarter of that as bytes. UTF-8 orders like the code points, so sorting
    and searching the bytes agrees with doing so on the strings.
    """
    names = np.asarray(names)
    if names.dtype.kind == "S":
        return names
    return np.char.encode(names.astype(str), "utf-8")


class _PackedStrand:
    """One strand of a packed cache: every record's code rows back to back.

    ``codes`` is the memory-mapped ``[total, n_u]`` matrix; record ``i`` owns
    rows ``offsets[i] : offsets[i] + lengths[i]``. ``exc_index`` holds the exact
    exceptions as global flat indices (``row * n_u + col``). ``whole`` is a
    profile over the full matrix, so a gather at global rows needs no per-record
    object. ``names`` are UTF-8 bytes (see `_name_keys`); ``by_name`` orders
    them, so lookups need no sorted copy. Both strands of a pack hold the same
    names, and the service passes one strand's arrays to the other.
    """

    def __init__(
        self,
        codes: np.ndarray,
        names: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        exc_index: np.ndarray,
        exc_value: np.ndarray,
        by_name: Optional[np.ndarray] = None,
    ):
        self.codes = codes
        self.names = _name_keys(names)
        self.offsets = offsets
        self.lengths = lengths
        self.exc_index = exc_index
        self.exc_value = exc_value
        self.whole = _CachedProfile(codes, exc_index, exc_value)
        # Names are in file-name or FASTA order, which need not be sorted.
        self._by_name = (
            np.argsort(self.names, kind="stable") if by_name is None else by_name
        )

    def record_ids(self, chroms: np.ndarray) -> np.ndarray:
        """Record number of each name in ``chroms``; ``-1`` for an unknown one."""
        keys = _name_keys(np.asarray(chroms, dtype=str))
        if self.names.size == 0:
            return np.full(keys.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.names, keys, sorter=self._by_name)
        rid = self._by_name[np.minimum(pos, self.names.size - 1)]
        return np.where(self.names[rid] == keys, rid, -1).astype(np.int64)

    def record(self, name: str) -> Optional[_CachedProfile]:
        """A view of one record's rows and exceptions; ``None`` if unknown."""
        rid = int(self.record_ids(np.array([name]))[0])
        if rid < 0:
            return None
        off, n = int(self.offsets[rid]), int(self.lengths[rid])
        n_u = self.codes.shape[1]
        lo, hi = np.searchsorted(self.exc_index, [off * n_u, (off + n) * n_u])
        return _CachedProfile(
            self.codes[off : off + n],
            self.exc_index[lo:hi] - off * n_u,
            self.exc_value[lo:hi],
        )


def _load_packed_strand(
    directory: Path, strand: str, share: Optional[_PackedStrand] = None
) -> _PackedStrand:
    """Map one strand's packed cache; only the ``.npz`` index is read into memory.

    ``share`` is the other strand of the pack, if loaded: when its names are
    the same, its name arrays are reused, so a process holds them once.
    """
    npy_path, npz_path = packed_paths(directory, strand)

    def corrupt(why: str) -> AccessibilityError:
        return AccessibilityError(
            f"Corrupt packed accessibility cache for strand {strand!r} "
            f"({npy_path}, {npz_path}): {why}. Delete the files to rebuild them."
        )

    try:
        codes = np.load(npy_path, mmap_mode="r")
        with np.load(npz_path) as side:
            names = np.asarray(side["names"])
            offsets = np.ascontiguousarray(side["offsets"], dtype=np.int64)
            lengths = np.ascontiguousarray(side["lengths"], dtype=np.int64)
            exc_index = np.ascontiguousarray(side["index"], dtype=np.int64)
            exc_value = np.ascontiguousarray(side["value"], dtype=np.float32)
        keys = _name_keys(names)
    except _CORRUPT_CACHE_ERRORS as e:
        raise corrupt(f"{type(e).__name__}: {e}") from e
    del names
    if codes.ndim != 2 or codes.dtype != np.uint8:
        raise corrupt(
            f"codes are {codes.dtype} with shape {codes.shape}, not 2-D uint8"
        )
    if not (keys.shape == offsets.shape == lengths.shape) or keys.ndim != 1:
        raise corrupt("names, offsets and lengths differ in length")
    if exc_index.shape != exc_value.shape:
        raise corrupt("the exception index and values differ in length")
    if keys.size and int(offsets[-1] + lengths[-1]) != codes.shape[0]:
        raise corrupt(
            f"records cover {int(offsets[-1] + lengths[-1])} rows but the matrix "
            f"has {codes.shape[0]}"
        )
    by_name = None
    if share is not None and np.array_equal(share.names, keys):
        keys, by_name = share.names, share._by_name
    return _PackedStrand(codes, keys, offsets, lengths, exc_index, exc_value, by_name)


class GenomeAccessibilityService:
    """
    Service to pre-compute and query genomic accessibility profiles.

    Profiles are stored as per-chromosome Parquet files produced by
    compute_genome_accessibility() in ``data_dir`` and served from a uint8
    cache, ``np.load(mmap_mode="r")``-ed so every process shares one page-cache
    copy and resident memory is only the pages it touches. The cache is the
    packed pair of `pack_parquet_profiles` (one ``.packed.npy`` plus index
    ``.packed.npz`` per strand, every record in it), or, for a directory
    written by v0.2.0, the per-record files of `build_accessibility_cache`. It
    is looked up in ``cache_dir`` (default: ``data_dir``), then in ``data_dir``,
    and a pack is preferred over per-record files. A pack that does not match
    the Parquet in ``data_dir`` (a Parquet record missing from it, or a Parquet newer
    than the pack) is rebuilt on first use, as is a contig whose per-record
    cache is missing or older than its Parquet (batch runs pack up front with
    `ensure_accessibility_cache` so worker processes only map finished files);
    a directory holding the complete cache needs no Parquet at all. The LRU of
    ``max_cached`` slots bounds the number of per-record mappings loaded per
    process. ``verify_pack=False`` skips the pack-versus-Parquet check: pass it
    only when `ensure_accessibility_cache` has just run on the same
    directories, as the off-target pipeline does before building services.

    `from_frames()` builds a service over in-memory per-chromosome
    DataFrames instead (the ``dict[chrom -> DataFrame]`` that
    `sioff.accessibility` returns); such a service never reads or writes
    disk, has ``data_dir is None`` and keeps the dense float32 profiles.
    """

    def __init__(
        self,
        data_dir: Optional[Path],
        max_cached: int = 4,
        frames: Optional[Mapping[str, pl.DataFrame]] = None,
        cache_dir: Optional[Path] = None,
        verify_pack: bool = True,
    ):
        self.data_dir = data_dir
        self._verify_pack = verify_pack
        if self.data_dir is not None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        # Where the uint8 cache is built and first looked for; None only for a
        # frame-backed service, which has no on-disk profiles.
        self.cache_dir: Optional[Path] = (
            Path(cache_dir)
            if cache_dir is not None
            else (Path(data_dir) if data_dir is not None else None)
        )
        self._frames: Mapping[str, pl.DataFrame] = frames if frames is not None else {}
        self._max_cached = max_cached
        self._profiles: OrderedDict[str, _Profile] = OrderedDict()
        self._strands_checked: set[str] = set()
        # Contigs whose cache was found no older than their Parquet, or was
        # (re)built here; `_cached_from_disk` does not compare mtimes again.
        self._current: set[str] = set()
        # Directory holding the packed cache, resolved once; `_UNRESOLVED` until
        # then (resetting it to that re-resolves).
        self._packed_root: object = _UNRESOLVED
        self._packed_strands: dict[str, Optional[_PackedStrand]] = {}
        # Records whose two packed strands differ in length; None until computed.
        self._packed_mismatches: Optional[dict[str, tuple[int, int]]] = None

    def _reset_packed(self) -> None:
        """Forget the resolved pack directory and every loaded strand.

        Called after a lazy (re)pack: keeping a cached ``None`` strand or a
        memory map of the replaced file would go on serving the old answer.
        """
        self._packed_root = _UNRESOLVED
        self._packed_strands = {}
        self._packed_mismatches = None

    def _packed_dir(self) -> Optional[Path]:
        """The first searched directory holding packed files, or ``None``.

        When ``data_dir`` holds Parquet the pack must match it (see
        `_pack_is_current`), checked once per process unless the service was
        built with ``verify_pack=False``; one that does not is ignored, so the
        next lookup goes through `_cached_from_disk`, which repacks it.
        """
        if self._packed_root is _UNRESOLVED:
            root: Optional[Path] = None
            for directory in self._cache_dirs():
                if any(
                    p.is_file()
                    for s in _STRAND_NAMES
                    for p in packed_paths(directory, s)
                ):
                    root = directory
                    break
            if root is not None and self.data_dir is not None and self._verify_pack:
                files = sorted(self.data_dir.glob(f"*{PARQUET_SUFFIX}"))
                if files and not _pack_is_current(root, files):
                    logger.info(
                        f"Packed accessibility cache in {root} does not match the "
                        f"Parquet profiles in {self.data_dir}; not using it"
                    )
                    root = None
            if root is not None:
                logger.info(f"Using the packed accessibility cache in {root}")
            self._packed_root = root
        return cast(Optional[Path], self._packed_root)

    def _packed_strand(self, strand: str) -> Optional[_PackedStrand]:
        """The packed cache of one strand, loaded once; ``None`` if it has none.

        A pair with only one of its two files (an interrupted pack) is corrupt,
        not missing. The two strands share one copy of the record names.
        """
        if strand not in self._packed_strands:
            root = self._packed_dir()
            loaded = None
            if root is not None and any(
                p.is_file() for p in packed_paths(root, strand)
            ):
                other = next(
                    (p for p in self._packed_strands.values() if p is not None), None
                )
                loaded = _load_packed_strand(root, strand, share=other)
            self._packed_strands[strand] = loaded
        return self._packed_strands[strand]

    def _packed_length_mismatches(self) -> dict[str, tuple[int, int]]:
        """``{record: (plus length, minus length)}`` for records whose strands differ.

        Computed once per loaded pack. Records empty on either strand are left
        out, as `check_strand_lengths` skips them.
        """
        if self._packed_mismatches is None:
            plus, minus = self._packed_strand("+"), self._packed_strand("-")
            found: dict[str, tuple[int, int]] = {}
            if plus is not None and minus is not None:
                rid = (
                    np.arange(plus.names.size)
                    if plus.names is minus.names
                    else minus.record_ids(np.char.decode(plus.names, "utf-8"))
                )
                known = rid >= 0
                minus_len = np.zeros_like(plus.lengths)
                minus_len[known] = minus.lengths[rid[known]]
                differ = (
                    (plus.lengths != minus_len) & (plus.lengths > 0) & (minus_len > 0)
                )
                for i in np.flatnonzero(differ):
                    found[plus.names[i].decode("utf-8")] = (
                        int(plus.lengths[i]),
                        int(minus_len[i]),
                    )
            self._packed_mismatches = found
        return self._packed_mismatches

    def _require_dir(self) -> Path:
        """The Parquet directory; a frame-backed service has none."""
        if self.data_dir is None:
            raise AccessibilityError(
                "this accessibility service serves in-memory frames and has no "
                "data directory"
            )
        return self.data_dir

    @classmethod
    def from_frames(
        cls, frames: Mapping[str, pl.DataFrame], max_cached: int = 4
    ) -> "GenomeAccessibilityService":
        """Serve profiles from in-memory DataFrames, one per chromosome.

        Each frame has the Parquet schema ``[position, strand, u1..uN]`` with
        both strands stacked, exactly what `sioff.accessibility` returns.
        """
        return cls(None, max_cached=max_cached, frames=frames)

    def compute_genome_accessibility(
        self,
        genome_path: Path,
        window_size: int = 80,
        max_span: int = 40,
        unpaired_prob: int = 30,
        workers: int = 1,
        progress=None,
        progress_callback=None,
        temperature: float = 37.0,
        output_format: str = "parquet",
        strand: str = "both",
        chunk_size: int = 1_000_000,
    ) -> Dict[str, Path]:
        """
        Compute accessibility for all sequences in the genome FASTA.

        Record IDs (the first word of each header) must be unique; duplicates
        raise before anything is written. With ``output_format="parquet"``
        one task per record goes to a ProcessPoolExecutor, and each saves a
        ``{record}.accessibility.parquet`` with columns [position, strand, u1,
        u2, ..., u{unpaired_prob}]. With ``"npy"`` the records are folded, in
        batches, straight into the packed uint8 memory-map cache in the data
        directory, ``accessibility.{plus,minus}.packed.npy`` plus a
        ``.packed.npz`` index each, records in FASTA order (see
        `_pack_fasta_records`), which ``off-targets`` reads with no
        conversion step; a re-run replaces the pair, and a FASTA without
        records is refused. ``strand="+"`` folds each sequence as given only
        (see `_fold_full_chromosome`); the npy output then has no minus pair,
        and one from an earlier run is deleted.

        Args:
            genome_path: Path to genome FASTA file.
            window_size: -W parameter (default 80).
            max_span: -L parameter (default 40).
            unpaired_prob: -u parameter (default 30).
            workers: Number of parallel folding processes.
            progress: Rich Progress instance (optional).
            progress_callback: Simple callback(advance, description) (optional).
                For "npy" it is called once per finished batch, with
                ``advance`` the batch's record count.
            temperature: Folding temperature in °C (default 37.0).
            output_format: "parquet" (default) or "npy".
            strand: "both" (default) or "+" to skip the reverse complement.
            chunk_size: Positions folded per RNAplfold call on long records
                (default 1,000,000).

        Returns:
            Dictionary mapping each record to its Parquet path, or, for
            "npy", every record to the plus strand's ``.packed.npy`` path.
        """
        if output_format not in OUTPUT_FORMATS:
            raise AccessibilityError(
                f"output_format must be one of {OUTPUT_FORMATS}, got {output_format!r}"
            )
        _check_fold_strand(strand)
        from sioff.services.helpers import read_fasta

        records = list(read_fasta(genome_path))
        dups = [n for n, k in Counter(n for n, _ in records).items() if k > 1]
        if dups:
            more = "…" if len(dups) > 10 else ""
            raise AccessibilityError(
                f"Duplicate record IDs in {genome_path}: {', '.join(dups[:10])}{more}"
            )
        out_dir = self._require_dir()

        logger.info(
            f"Computing full-genome accessibility for {len(records)} "
            f"record(s) using {workers} worker(s)"
        )

        if output_format == "npy":
            n = len(records)
            if not n:
                # A pack of no records is of no use, and `_annotate_packed`
                # cannot index an empty one.
                raise AccessibilityError(f"No records in {genome_path}")
            prog_task = None
            if progress is not None:
                prog_task = progress.add_task("Folding records...", total=n)
            done = 0

            def on_batch(advance: int) -> None:
                nonlocal done
                done += advance
                description = f"Folded {done}/{n} records"
                if progress is not None and prog_task is not None:
                    progress.update(prog_task, advance=advance, description=description)
                elif progress_callback:
                    progress_callback(advance=advance, description=description)

            plus = _pack_fasta_records(
                records,
                out_dir,
                window_size,
                max_span,
                unpaired_prob,
                chunk_size,
                temperature,
                strand,
                workers,
                on_batch,
            )
            # Mappings of a pair this service had loaded are now stale.
            self._reset_packed()
            self._profiles.clear()
            return {name: plus for name, _ in records}

        # Longest-job-first to minimise tail latency.
        chromosomes = sorted(records, key=lambda cs: len(cs[1]), reverse=True)

        total_tasks = len(chromosomes)
        prog_task = None
        if progress is not None:
            prog_task = progress.add_task("Folding chromosomes...", total=total_tasks)

        results: Dict[str, Path] = {}
        futures = {}

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for chrom, sequence in chromosomes:
                fut = pool.submit(
                    _fold_full_chromosome,
                    sequence,
                    chrom,
                    window_size,
                    max_span,
                    unpaired_prob,
                    str(out_dir / f"{chrom}{PARQUET_SUFFIX}"),
                    chunk_size,
                    temperature,
                    output_format=output_format,
                    strand=strand,
                )
                futures[fut] = chrom

            for fut in as_completed(futures):
                chrom = futures[fut]
                try:
                    path = fut.result()
                except Exception as e:
                    logger.error(f"Error folding {chrom}: {e}")
                    raise

                results[chrom] = Path(path)
                logger.info(f"Completed {chrom} → {path}")

                if progress is not None and prog_task is not None:
                    progress.update(
                        prog_task,
                        advance=1,
                        description=f"Folded {chrom}",
                    )
                elif progress_callback:
                    progress_callback(advance=1, description=f"Folded {chrom}")

        return results

    def compute_sequence_accessibility(
        self,
        sequence: str,
        window_size: int = 80,
        max_span: int = 40,
        unpaired_prob: int = 30,
        temperature: float = 37.0,
    ) -> np.ndarray:
        """
        Compute accessibility for a single sequence (e.g., on-target).

        Uses ViennaRNA's RNA.fold_compound with RNA.OPTION_WINDOW to compute unpaired probabilities,
        then converts to opening energies.

        Args:
            sequence: RNA/DNA sequence string.
            window_size: -W parameter (default 80).
            max_span: -L parameter (default 40).
            unpaired_prob: -u parameter (default 30).
            temperature: Folding temperature in °C (default 37.0).

        Returns:
            2D numpy array [seq_len, unpaired_prob] of opening energies.
            Use result[pos, u-1] to get opening energy for length u at position pos.
        """
        return fold_sequence(
            sequence,
            window_size=window_size,
            max_span=max_span,
            unpaired_prob=unpaired_prob,
            temperature=temperature,
        )

    def _find_profile(self, chrom: str) -> Path | None:
        """Return path to the chromosome's accessibility Parquet, or None."""
        p = self._require_dir() / f"{chrom}{PARQUET_SUFFIX}"
        return p if p.exists() else None

    @staticmethod
    def _dense_profile(full: pl.DataFrame, strand: str, source: str) -> _DenseProfile:
        """The dense in-memory path: one strand of a frame as float32[max_pos, n_u].

        Positions absent from the frame keep the 25.5 kcal/mol sentinel.
        """
        df = full.filter(pl.col("strand") == strand).sort("position")
        if df.height == 0:
            raise MissingStrandError(f"No data for strand {strand!r} in {source}")
        u_cols = _u_columns(df.columns)
        max_pos = int(cast(int, df["position"].max()))
        arr = np.full((max_pos, len(u_cols)), 25.5, dtype=np.float32)
        pos_0 = df["position"].to_numpy() - 1  # 1-based -> 0-based
        for col_idx, col_name in enumerate(u_cols):
            arr[pos_0, col_idx] = df[col_name].to_numpy()
        return _DenseProfile(arr)

    def _cache_dirs(self) -> list[Path]:
        """Directories searched for a cache, in order: ``cache_dir``, ``data_dir``."""
        return list(
            dict.fromkeys(d for d in (self.cache_dir, self.data_dir) if d is not None)
        )

    def _find_cache(self, chrom: str, strand: str) -> Optional[tuple[Path, Path]]:
        """``(npy, npz)`` of the strand's per-record cache, or ``None``.

        The first searched directory holding both files; a lone ``.npy`` does
        not count, so it cannot shadow a complete cache further down the list.
        """
        for directory in self._cache_dirs():
            npy_path, npz_path = cache_paths(directory, chrom, strand)
            if npy_path.is_file() and npz_path.is_file():
                return npy_path, npz_path
        return None

    def _open_cache(self, chrom: str, strand: str) -> _CachedProfile | None:
        """Memory-map the (chrom, strand) cache if both its files exist."""
        found = self._find_cache(chrom, strand)
        if found is None:
            return None
        npy_path, npz_path = found
        try:
            codes = np.load(npy_path, mmap_mode="r")
            with np.load(npz_path) as side:
                exc_index = np.ascontiguousarray(side["index"], dtype=np.int64)
                exc_value = np.ascontiguousarray(side["value"], dtype=np.float32)
        except _CORRUPT_CACHE_ERRORS as e:
            raise AccessibilityError(
                f"Corrupt accessibility cache for {chrom} {strand} "
                f"({npy_path}, {npz_path}): {e}. Delete the files to rebuild them."
            ) from e
        if codes.ndim != 2 or codes.dtype != np.uint8:
            raise AccessibilityError(
                f"Corrupt accessibility cache for {chrom} {strand}: {npy_path} is "
                f"{codes.dtype} with shape {codes.shape}, expected 2-D uint8. "
                "Delete the file to rebuild it."
            )
        return _CachedProfile(codes, exc_index, exc_value)

    def _cached_from_disk(self, chrom: str, strand: str) -> _CachedProfile:
        """Map the per-record cache for (chrom, strand), packing it if needed.

        Reached when no current pack exists. A missing cache, or one older than
        the contig's Parquet in ``data_dir`` (see `_cache_is_stale`, warned
        about), makes `ensure_accessibility_cache` pack the Parquet directory;
        the contig is then served from the pack.
        The age check runs until the contig's cache is known current, so the
        second strand and later re-mappings after LRU eviction skip it.
        """
        parquet = self._find_profile(chrom)
        stale_in: Optional[Path] = None
        if parquet is not None and chrom not in self._current:
            found = self._find_cache(chrom, strand)
            if found is not None and _cache_is_stale(found[0].parent, chrom, parquet):
                stale_in = found[0].parent
        profile = None if stale_in is not None else self._open_cache(chrom, strand)
        if profile is None:
            if parquet is None:
                searched = " or ".join(
                    str(cache_paths(d, chrom, strand)[0]) for d in self._cache_dirs()
                )
                raise ProfileNotFoundError(
                    f"Profile for {chrom} not found: neither the cache "
                    f"{searched} nor the Parquet "
                    f"{self._require_dir() / f'{chrom}{PARQUET_SUFFIX}'} exists."
                )
            if stale_in is not None:
                _warn_stale(chrom, stale_in, parquet)
            else:
                logger.info(
                    f"Accessibility cache for {chrom} missing in {self.cache_dir}; "
                    f"packing the profiles in {self.data_dir}"
                )
            try:
                ensure_accessibility_cache(
                    cast(Path, self.data_dir), cast(Path, self.cache_dir)
                )
            except (AccessibilityError, OSError) as e:
                cause = e.__cause__ if isinstance(e, AccessibilityError) else e
                if not isinstance(cause, OSError):
                    raise
                raise AccessibilityError(
                    f"Cannot write the accessibility cache for {chrom} to "
                    f"{self.cache_dir}: {cause}. Pass a writable cache directory "
                    "(--accessibility-cache-dir)."
                ) from cause
            self._reset_packed()
            packed_root = self._packed_dir()
            if packed_root is not None:
                self._current.add(chrom)
                return self._packed_profile(chrom, strand, packed_root)[0]
            profile = self._open_cache(chrom, strand)
            if profile is None:
                raise AccessibilityError(
                    f"Accessibility cache for {chrom} {strand} still missing after "
                    f"building from {parquet}"
                )
        if parquet is not None:
            self._current.add(chrom)
        if profile.n_pos == 0:
            raise MissingStrandError(f"No data for strand {strand!r} in {chrom}")
        return profile

    def _packed_profile(
        self, chrom: str, strand: str, root: Path
    ) -> tuple[_CachedProfile, str]:
        """One record's view of the packed cache, and the ``.npy`` it comes from."""
        packed = self._packed_strand(strand)
        if packed is None:
            raise MissingStrandError(
                f"No data for strand {strand!r} in the packed cache {root}"
            )
        npy = str(packed_paths(root, strand)[0])
        profile = packed.record(chrom)
        if profile is None:
            raise ProfileNotFoundError(
                f"Profile for {chrom} not found in the packed cache {npy}"
            )
        if profile.n_pos == 0:
            raise MissingStrandError(f"No data for strand {strand!r} in {chrom}")
        return profile, npy

    def _ensure_profile(self, chrom: str, strand: str) -> str:
        """
        Ensure the profile for (chrom, strand) is loaded, applying LRU eviction.

        Returns the profile_key string.

        In-memory frames are materialised as a dense float32 array; on-disk
        profiles are memory-mapped from the uint8 cache, built from the Parquet
        first if needed. Positions absent from the source keep the default
        sentinel value of 25.5 kcal/mol in both cases.

        Complexity: O(1) amortized. Eviction drops the oldest cached profile
        (an array, or a mapping) when the cache exceeds max_cached slots.
        """
        profile_key = f"{chrom}_{strand}"

        if profile_key in self._profiles:
            self._profiles.move_to_end(profile_key)
            return profile_key

        if chrom in self._frames:
            source: str = f"in-memory frame for {chrom}"
            profile: _Profile = self._dense_profile(self._frames[chrom], strand, source)
        else:
            if self.data_dir is None:
                raise ProfileNotFoundError(
                    f"Profile for {chrom} not found among the in-memory "
                    f"accessibility frames ({', '.join(sorted(self._frames)) or 'none'})."
                )
            packed_root = None if chrom in self._frames else self._packed_dir()
            if packed_root is not None:
                profile, source = self._packed_profile(chrom, strand, packed_root)
            else:
                profile = self._cached_from_disk(chrom, strand)
                packed_root = self._packed_dir()
                found = (
                    None if packed_root is not None else self._find_cache(chrom, strand)
                )
                source = str(
                    packed_paths(packed_root, strand)[0]
                    if packed_root is not None
                    else found[0]
                    if found is not None
                    else "the accessibility cache"
                )

        self._profiles[profile_key] = profile
        logger.info(f"Loaded accessibility profile for {chrom} {strand} from {source}")
        if chrom not in self._strands_checked:
            self._strands_checked.add(chrom)
            check_strand_lengths(chrom, *self._strand_lengths(chrom), source)

        while len(self._profiles) > self._max_cached:
            evicted_key, _ = self._profiles.popitem(last=False)
            logger.debug(f"Evicted profile cache entry: {evicted_key}")

        return profile_key

    def _strand_lengths(self, chrom: str) -> tuple[Optional[int], Optional[int]]:
        """``(plus, minus)`` profile lengths from the cheapest available source.

        In-memory frames: a group-by on ``position``. Cache: the ``.npy``
        headers, read from ``cache_dir`` first, then ``data_dir``. Any failure
        yields ``None`` for that strand: this feeds a warning only and must
        never break a lookup.
        """
        try:
            if chrom in self._frames:
                agg = (
                    self._frames[chrom].group_by("strand").agg(pl.col("position").max())
                )
                lens = dict(zip(agg["strand"].to_list(), agg["position"].to_list()))
                return lens.get("+"), lens.get("-")
            if self._packed_dir() is not None:
                packed_lens: list[Optional[int]] = []
                for strand in ("+", "-"):
                    packed = self._packed_strand(strand)
                    rid = (
                        -1
                        if packed is None
                        else int(packed.record_ids(np.array([chrom]))[0])
                    )
                    packed_lens.append(
                        None if packed is None or rid < 0 else int(packed.lengths[rid])
                    )
                return packed_lens[0], packed_lens[1]
            out: list[Optional[int]] = []
            for strand in ("+", "-"):
                found = self._find_cache(chrom, strand)
                out.append(
                    None
                    if found is None
                    else int(np.load(found[0], mmap_mode="r").shape[0])
                )
            return out[0], out[1]
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            logger.debug(f"could not determine strand lengths for {chrom}: {exc}")
            return None, None

    def query(self, chrom: str, start: int, end: int, strand: str = "+") -> np.ndarray:
        """Return profile rows ``[start, end)`` (0-based, half-open) of one strand.

        Always a float32 ``[end - start, n_u]`` array in kcal/mol, whichever
        kind of profile backs the chromosome. Rows are in the coordinates of the
        molecule that was folded: chromosome positions for ``'+'``, reverse-
        complement positions for ``'-'`` (see `site_cells`). To look up the
        energy of a hit given in chromosome coordinates use `query_single` or
        `annotate_opening_energy_vectorized`.
        """
        profile_key = self._ensure_profile(chrom, strand)
        profile = self._profiles[profile_key]
        seq_len = profile.n_pos

        if start < 0 or end > seq_len:
            raise AccessibilityError(
                f"Query {start}-{end} out of bounds for {chrom} {strand} (len {seq_len})"
            )

        return profile.region(start, end)

    def query_single(
        self, chrom: str, start: int, end: int, strand: str = "+"
    ) -> float:
        """
        Fast-path: return a single opening energy value for a prediction.

        Picks the cell `site_cells` assigns to the hit (the stretch covering
        ``[start, end]`` on its own transcript, interaction length as the
        u-column) and quantizes to 0.1 kcal/mol.

        Args:
            chrom: Chromosome name.
            start: 1-based start position (RIsearch convention).
            end:   1-based end position (inclusive, RIsearch convention).
            strand: '+' or '-'.

        Returns:
            Opening energy as float, quantized to 0.1 kcal/mol.

        Complexity: O(1) time, O(1) space per call (profile already loaded).
        """
        profile_key = self._ensure_profile(chrom, strand)
        profile = self._profiles[profile_key]
        seq_len = profile.n_pos

        rows, cols, in_bounds = site_cells(
            strand, np.array([start]), np.array([end]), seq_len, profile.n_u
        )
        if not in_bounds[0]:
            return 10.0
        raw_val = float(profile.gather(rows, cols)[0])

        return int(round(raw_val * 10.0)) / 10.0

    def annotate_opening_energy_vectorized(
        self, df: pl.DataFrame, stats: Optional[dict] = None
    ) -> pl.DataFrame:
        """Add `opening_energy` column by vectorized numpy gather per (chrom, strand) group.

        Hits on an unknown contig, on ``onTarget``, on a strand other than
        ``+``/``-`` or outside their contig keep 10.0; so do hits on a strand
        the profile lacks, with a warning. Nothing is logged for the others:
        ``stats`` (see `new_default_energy_stats`) counts the hits looked up,
        found, on an unknown contig and outside their contig, for the caller
        to report once per run.

        Raises
        ------
        AccessibilityError
            When a cache cannot be read (corrupt or truncated files), or a
            missing one cannot be built or written.
        """
        n_rows = df.height
        opening_energies = np.full(n_rows, 10.0, dtype=np.float32)

        starts = df["start"].to_numpy().astype(np.int64)
        ends = df["end"].to_numpy().astype(np.int64)
        strands = df["strand"].to_numpy()

        if n_rows == 0:
            return df.with_columns(pl.Series("opening_energy", [], dtype=pl.Float32))
        if stats is not None:
            stats["n_hits"] += n_rows

        if not self._frames and self._packed_dir() is not None:
            # Map each distinct contig once: a per-hit array of names costs
            # the longest name's width for every hit. Nulls read as "None",
            # as `str` makes them on the per-group path below.
            chrom = df["chrom"].cast(pl.String).fill_null("None")
            uniq = chrom.unique(maintain_order=True)
            inv = chrom.replace_strict(
                uniq, np.arange(uniq.len(), dtype=np.int64), return_dtype=pl.Int64
            ).to_numpy()
            self._annotate_packed(
                np.asarray(uniq.to_list(), dtype=str),
                inv,
                starts,
                ends,
                strands,
                opening_energies,
                stats,
            )
            return df.with_columns(
                pl.Series("opening_energy", opening_energies, dtype=pl.Float32)
            )

        chroms = df["chrom"].to_numpy()
        groups: dict[tuple, list] = defaultdict(list)
        for i in range(n_rows):
            c = str(chroms[i])
            if c == "onTarget":
                continue
            groups[(c, str(strands[i]))].append(i)

        for (chrom, strand), row_indices in groups.items():
            if strand not in ("+", "-"):
                continue
            try:
                profile_key = self._ensure_profile(chrom, strand)
            except MissingStrandError:
                _warn_missing_strand(chrom, strand)
                continue
            except ProfileNotFoundError:
                if stats is not None:
                    stats["n_unknown"] += len(row_indices)
                    add_default_examples(stats["unknown"], [chrom])
                continue

            profile = self._profiles[profile_key]
            seq_len = profile.n_pos

            idx = np.array(row_indices, dtype=np.int64)
            row_idx, col_idx, in_bounds = site_cells(
                strand, starts[idx], ends[idx], seq_len, profile.n_u
            )
            raw_vals = profile.gather(row_idx, col_idx)

            # Quantize to 0.1 resolution (match legacy query_single behaviour)
            raw_vals = (np.round(raw_vals * 10.0) / 10.0).astype(np.float32)
            opening_energies[idx[in_bounds]] = raw_vals[in_bounds]
            if stats is not None:
                n_in = int(in_bounds.sum())
                stats["n_found"] += n_in
                if n_in < idx.size:
                    stats["n_out_of_bounds"] += int(idx.size) - n_in
                    add_default_examples(stats["out_of_bounds"], [chrom])

        return df.with_columns(
            pl.Series("opening_energy", opening_energies, dtype=pl.Float32)
        )

    def _annotate_packed(
        self,
        names: np.ndarray,
        inv: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        strands: np.ndarray,
        out: np.ndarray,
        stats: Optional[dict] = None,
    ) -> None:
        """Fill ``out`` with every hit's opening energy from the packed cache.

        Hit ``i`` is on contig ``names[inv[i]]``; each distinct name is looked
        up once. One vectorised pass per strand: hits are mapped to records,
        turned into global rows (``record offset + site_cells`` row) and gathered
        in a single call. Unknown records, ``onTarget``, other strands and hits
        outside their record keep the default already in ``out``. Records with
        '-' hits whose strands differ in length are warned about once; '+'
        lookups do not depend on the minus length, so '+' hits alone stay
        silent.
        """
        on_target = names == "onTarget"
        # Unequal strand lengths only shift '-' lookups (`site_cells` takes the
        # contig length from the minus profile), so only records with '-' hits
        # are checked, and a '+'-only run never loads the minus index.
        minus_hit = np.zeros(names.size, dtype=bool)
        minus_hit[inv[strands == "-"]] = True
        self._check_packed_lengths(names[minus_hit & ~on_target])
        for strand in ("+", "-"):
            sel = np.flatnonzero(strands == strand)
            sel = sel[~on_target[inv[sel]]]
            if sel.size == 0:
                continue
            packed = self._packed_strand(strand)
            if packed is None:
                _warn_missing_strand(str(names[inv[sel[0]]]), strand)
                continue
            rid = packed.record_ids(names)[inv[sel]]
            known = rid >= 0
            lengths = (
                np.where(known, packed.lengths[np.maximum(rid, 0)], 0)
                if packed.lengths.size
                else np.zeros(rid.shape, dtype=np.int64)
            )
            empty = known & (lengths == 0)
            if empty.any():
                _warn_missing_strand(str(names[inv[sel[np.argmax(empty)]]]), strand)
            rows, cols, in_bounds = site_cells(
                strand, starts[sel], ends[sel], lengths, packed.codes.shape[1]
            )
            ok = known & in_bounds & (lengths > 0)
            if stats is not None:
                # From the masks, never from the values: 10.0 is also an
                # ordinary opening energy.
                stats["n_found"] += int(ok.sum())
                for key, mask in (
                    ("unknown", ~known),
                    ("out_of_bounds", known & (lengths > 0) & ~in_bounds),
                ):
                    n = int(mask.sum())
                    if n:
                        stats[f"n_{key}"] += n
                        named = np.zeros(names.size, dtype=bool)
                        named[inv[sel[mask]]] = True
                        add_default_examples(
                            stats[key], names[named][:_MAX_DEFAULT_EXAMPLES].tolist()
                        )
            if not ok.any():
                continue
            global_rows = rows[ok] + packed.offsets[rid[ok]]
            vals = packed.whole.gather(global_rows, cols[ok])
            out[sel[ok]] = (np.round(vals * 10.0) / 10.0).astype(np.float32)

    def _check_packed_lengths(self, names: np.ndarray) -> None:
        """`check_strand_lengths` for the packed records in ``names``, once each.

        Diagnostics only, like `_strand_lengths`: a strand that cannot be
        loaded here is reported by the lookup itself, never by this check.
        """
        if names.size == 0:
            return
        try:
            mismatched = self._packed_length_mismatches()
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            logger.debug(f"could not compare packed strand lengths: {exc}")
            self._packed_mismatches = {}
            return
        if not mismatched:
            return
        root = cast(Path, self._packed_dir())
        source = str(packed_paths(root, "+")[0])
        for name in names.tolist():
            if name in mismatched and name not in self._strands_checked:
                self._strands_checked.add(name)
                check_strand_lengths(name, *mismatched[name], source)
