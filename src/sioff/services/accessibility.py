import math
import multiprocessing
import os
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
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


_COMPLEMENT_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")
_GAS_CONSTANT = 0.001987  # kcal / (mol·K)


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT_TABLE)[::-1]


def site_cells(
    strand: str,
    starts: np.ndarray,
    ends: np.ndarray,
    seq_len: int,
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
    rows = np.clip(rows, 0, max(seq_len - 1, 0))
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
# `_fold_full_chromosome` writes one Parquet file per contig, or (npy, the
# CLI default) this cache directly through `_write_strand_cache`. Reading a whole
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


def cache_paths(cache_dir: Path, chrom: str, strand: str) -> tuple[Path, Path]:
    """Return ``(codes .npy, exceptions .npz)`` paths for one contig strand."""
    if strand not in _STRAND_NAMES:
        raise AccessibilityError(f"Unknown strand {strand!r} (expected '+' or '-')")
    stem = f"{chrom}_{_STRAND_NAMES[strand]}"
    return (
        Path(cache_dir) / f"{stem}{CACHE_SUFFIX}",
        Path(cache_dir) / f"{stem}{EXCEPTIONS_SUFFIX}",
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
        True when it holds a ``*.access.npy`` cache file or a
        ``*.accessibility.parquet`` file; False otherwise, including when the
        directory does not exist.
    """
    d = Path(directory)
    if not d.is_dir():
        return False
    return any(d.glob(f"*{CACHE_SUFFIX}")) or any(d.glob(f"*{PARQUET_SUFFIX}"))


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


def _write_atomically(target: Path, write) -> None:
    """``write(tmp_path)`` then rename, so readers never see a partial file.

    A concurrent builder of the same target writes its own temporary file and
    the last rename wins; both renames deliver a complete, identical file.
    """
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp{target.suffix}")
    try:
        write(tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


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
    row_groups, max_pos = _strand_row_groups(pf, strand)
    codes = np.full((max_pos, n_u), _SENTINEL_CODE, dtype=np.uint8)
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
    exc_index: list[np.ndarray] = []
    exc_value: list[np.ndarray] = []
    try:
        codes = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.uint8, shape=(n_pos, n_u)
        )
        for pos0, values in chunks:
            block, exact = _encode(values)
            codes[pos0 : pos0 + len(values)] = block
            if not exact.all():
                r, c = np.nonzero(~exact)
                exc_index.append((pos0 + r).astype(np.int64) * n_u + c)
                exc_value.append(values[r, c])
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
    except (OSError, ValueError):
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


def ensure_accessibility_cache(
    parquet_dir: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    contigs: Optional[Iterable[str]] = None,
    workers: int = 1,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> list[str]:
    """Build the uint8 cache for every Parquet contig that lacks a current one.

    Meant to run once in the parent process before an off-target worker pool
    starts, so workers only ever memory-map finished files. A contig is
    skipped when both strand caches in ``cache_dir`` pass `_cache_is_valid`
    and neither ``.npy`` is older than its Parquet, which makes repeated calls
    cheap. A cache older than its Parquet (the directory was re-folded) is
    rebuilt with a warning. A directory without any Parquet (only ``.npy``
    caches) builds nothing.

    Parameters
    ----------
    parquet_dir : str or pathlib.Path
        Directory of ``<chrom>.accessibility.parquet`` files.
    cache_dir : str or pathlib.Path, optional
        Where the ``.npy``/``.npz`` files go; created if missing. Defaults to
        ``parquet_dir``.
    contigs : iterable of str, optional
        Restrict the build to these contig names; default all Parquet present.
    workers : int, default 1
        Contigs built in parallel: ``min(workers, 4)`` spawned processes, each
        holding one strand's cache (chr1: 7.5 GB) at its peak.
    batch_rows : int, default 1_000_000
        Rows decoded per Parquet batch while streaming.

    Returns
    -------
    list[str]
        Names of the contigs that were (re)built, largest file first.

    Raises
    ------
    AccessibilityError
        When a contig's cache cannot be built, naming its Parquet; the
        original exception (an ``OSError`` for an unwritable ``cache_dir``)
        is its ``__cause__``.
    OSError
        When ``cache_dir`` cannot be created.
    """
    parquet_dir = Path(parquet_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else parquet_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parquet_dir.glob(f"*{PARQUET_SUFFIX}"))
    if contigs is not None:
        wanted = set(contigs)
        files = [f for f in files if _chrom_of(f) in wanted]
    missing: list[Path] = []
    for f in files:
        chrom = _chrom_of(f)
        if not all(_cache_is_valid(*cache_paths(cache_dir, chrom, s)) for s in "+-"):
            missing.append(f)
        elif _cache_is_stale(cache_dir, chrom, f):
            _warn_stale(chrom, cache_dir, f)
            missing.append(f)
    if not missing:
        return []
    missing.sort(key=lambda f: f.stat().st_size, reverse=True)
    names = [_chrom_of(f) for f in missing]

    n_proc = max(1, min(workers, 4, len(missing)))
    logger.info(
        f"Building accessibility cache for {len(missing)} contig(s) in {cache_dir} "
        f"with {n_proc} process(es)"
    )
    if n_proc == 1:
        for f in missing:
            try:
                build_accessibility_cache(f, cache_dir, batch_rows)
            except Exception as e:
                raise AccessibilityError(
                    f"Failed to build accessibility cache for {f.name}: {e}"
                ) from e
        return names

    # spawn: a fresh interpreter per worker avoids fork + Arrow thread-pool deadlocks.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx) as pool:
        futures = {
            pool.submit(build_accessibility_cache, f, cache_dir, batch_rows): f
            for f in missing
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                raise AccessibilityError(
                    f"Failed to build accessibility cache for {futures[fut].name}: {e}"
                ) from e
    return names


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


class GenomeAccessibilityService:
    """
    Service to pre-compute and query genomic accessibility profiles.

    Profiles are stored as per-chromosome Parquet files produced by
    compute_genome_accessibility() in ``data_dir`` and served from the uint8
    cache described at `build_accessibility_cache`: one
    ``<chrom>_<plus|minus>.access.npy`` per strand plus an exceptions sidecar,
    ``np.load(mmap_mode="r")``-ed so every process shares one page-cache copy
    and resident memory is only the pages it touches. The cache is looked up in
    ``cache_dir`` (default: ``data_dir``), then in ``data_dir``. A contig whose
    cache is missing, or older than its Parquet, is built from the Parquet
    into ``cache_dir`` on first use (batch runs build it up front with
    `ensure_accessibility_cache` so worker processes only map finished files);
    a directory holding the complete cache needs no Parquet at all. The LRU of
    ``max_cached`` slots bounds the number of mappings loaded per process.

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
    ):
        self.data_dir = data_dir
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
    ) -> Dict[str, Path]:
        """
        Compute accessibility for all sequences in the genome FASTA.

        Dispatches one task per chromosome to a ProcessPoolExecutor for
        parallel folding. With ``output_format="parquet"`` saves one Parquet
        file per chromosome containing columns [position, strand, u1, u2, ...,
        u{unpaired_prob}]; with ``"npy"`` writes the uint8 memory-map cache
        (``{chrom}_{plus,minus}.access.npy`` plus ``.access.exceptions.npz``
        sidecars, see `build_accessibility_cache`) instead, which
        ``off-targets`` reads with no conversion step.
        ``strand="+"`` folds each sequence as given only (see
        `_fold_full_chromosome`).

        Args:
            genome_path: Path to genome FASTA file.
            window_size: -W parameter (default 80).
            max_span: -L parameter (default 40).
            unpaired_prob: -u parameter (default 30).
            workers: Number of parallel folding workers.
            progress: Rich Progress instance (optional).
            progress_callback: Simple callback(advance, description) (optional).
            temperature: Folding temperature in °C (default 37.0).
            output_format: "parquet" (default) or "npy".
            strand: "both" (default) or "+" to skip the reverse complement.

        Returns:
            Dictionary mapping 'chrom' to the output Parquet path, or to the
            plus-strand ``.access.npy`` path for "npy".
        """
        if output_format not in OUTPUT_FORMATS:
            raise AccessibilityError(
                f"output_format must be one of {OUTPUT_FORMATS}, got {output_format!r}"
            )
        _check_fold_strand(strand)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from sioff.services.helpers import read_fasta

        # Read all chromosomes; longest-job-first to minimise tail latency.
        chromosomes = sorted(
            list(read_fasta(genome_path)),
            key=lambda cs: len(cs[1]),
            reverse=True,
        )

        logger.info(
            f"Computing full-genome accessibility for {len(chromosomes)} "
            f"chromosome(s) using {workers} worker(s)"
        )

        total_tasks = len(chromosomes)
        prog_task = None
        if progress is not None:
            prog_task = progress.add_task("Folding chromosomes...", total=total_tasks)

        results: Dict[str, Path] = {}
        futures = {}

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for chrom, sequence in chromosomes:
                out_path = str(
                    self._require_dir()
                    if output_format == "npy"
                    else self._require_dir() / f"{chrom}{PARQUET_SUFFIX}"
                )
                fut = pool.submit(
                    _fold_full_chromosome,
                    sequence,
                    chrom,
                    window_size,
                    max_span,
                    unpaired_prob,
                    out_path,
                    1_000_000,
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

    def _npy_path(self, chrom: str, strand: str) -> Path:
        """The ``.npy`` the lookup uses: the first searched directory that has it.

        Falls back to the ``cache_dir`` path, where a lazy build would write.
        """
        dirs = self._cache_dirs()
        for directory in dirs:
            npy = cache_paths(directory, chrom, strand)[0]
            if npy.is_file():
                return npy
        return cache_paths(dirs[0], chrom, strand)[0]

    def _cache_location(self, chrom: str, strand: str) -> Optional[Path]:
        """The first searched directory holding both files of the strand's cache."""
        for directory in self._cache_dirs():
            npy_path, npz_path = cache_paths(directory, chrom, strand)
            if npy_path.is_file() and npz_path.is_file():
                return directory
        return None

    def _open_cache(self, chrom: str, strand: str) -> _CachedProfile | None:
        """Memory-map the (chrom, strand) cache if both its files exist."""
        directory = self._cache_location(chrom, strand)
        if directory is None:
            return None
        npy_path, npz_path = cache_paths(directory, chrom, strand)
        try:
            codes = np.load(npy_path, mmap_mode="r")
            with np.load(npz_path) as side:
                exc_index = np.ascontiguousarray(side["index"], dtype=np.int64)
                exc_value = np.ascontiguousarray(side["value"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as e:
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
        """Map the cache for (chrom, strand), building it from Parquet if needed.

        Built when absent, and rebuilt (with a warning) when the cache found is
        older than the contig's Parquet in ``data_dir`` (see `_cache_is_stale`).
        The age check runs until the contig's cache is known current, so the
        second strand and later re-mappings after LRU eviction skip it.
        """
        parquet = self._find_profile(chrom)
        stale_in: Optional[Path] = None
        if parquet is not None and chrom not in self._current:
            found = self._cache_location(chrom, strand)
            if found is not None and _cache_is_stale(found, chrom, parquet):
                stale_in = found
        profile = None if stale_in is not None else self._open_cache(chrom, strand)
        if profile is None:
            if parquet is None:
                searched = " or ".join(
                    str(cache_paths(d, chrom, strand)[0]) for d in self._cache_dirs()
                )
                raise AccessibilityError(
                    f"Profile for {chrom} not found: neither the cache "
                    f"{searched} nor the Parquet "
                    f"{self._require_dir() / f'{chrom}{PARQUET_SUFFIX}'} exists."
                )
            if stale_in is not None:
                _warn_stale(chrom, stale_in, parquet)
            else:
                logger.info(
                    f"Accessibility cache for {chrom} missing in {self.cache_dir}; "
                    f"building it from {parquet}"
                )
            try:
                build_accessibility_cache(parquet, cast(Path, self.cache_dir))
            except OSError as e:
                raise AccessibilityError(
                    f"Cannot write the accessibility cache for {chrom} to "
                    f"{self.cache_dir}: {e}. Pass a writable cache directory "
                    "(--accessibility-cache-dir)."
                ) from e
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
                raise AccessibilityError(
                    f"Profile for {chrom} not found among the in-memory "
                    f"accessibility frames ({', '.join(sorted(self._frames)) or 'none'})."
                )
            profile = self._cached_from_disk(chrom, strand)
            source = str(self._npy_path(chrom, strand))

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
            out: list[Optional[int]] = []
            for strand in ("+", "-"):
                npy = self._npy_path(chrom, strand)
                out.append(
                    int(np.load(npy, mmap_mode="r").shape[0]) if npy.is_file() else None
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

    def annotate_opening_energy_vectorized(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add `opening_energy` column by vectorized numpy gather per (chrom, strand) group."""
        n_rows = df.height
        opening_energies = np.full(n_rows, 10.0, dtype=np.float32)

        chroms = df["chrom"].to_numpy()
        starts = df["start"].to_numpy().astype(np.int64)
        ends = df["end"].to_numpy().astype(np.int64)
        strands = df["strand"].to_numpy()

        groups: dict[tuple, list] = defaultdict(list)
        for i in range(n_rows):
            c = str(chroms[i])
            if c == "onTarget":
                continue
            groups[(c, str(strands[i]))].append(i)

        for (chrom, strand), row_indices in groups.items():
            try:
                profile_key = self._ensure_profile(chrom, strand)
            except MissingStrandError:
                _warn_missing_strand(chrom, strand)
                continue
            except Exception:
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

        return df.with_columns(
            pl.Series("opening_energy", opening_energies, dtype=pl.Float32)
        )
