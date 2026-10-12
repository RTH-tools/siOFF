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
import pyarrow.compute as pc
import pyarrow.parquet as pq
import RNA
from loguru import logger


class AccessibilityError(Exception):
    """Base exception for accessibility service."""

    pass


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
    (eager, in-memory frames, memory-mapped cache, chromosome-major gather,
    on-target) goes through it.

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
    cheapest observable sign of that. Unknown lengths (``None``) are skipped.

    Returns ``True`` when the lengths agree or cannot be compared.
    """
    if plus_len is None or minus_len is None or plus_len == minus_len:
        return True
    logger.warning(
        f"Accessibility profile for {chrom} ({source}): '+' covers {plus_len} "
        f"positions but '-' covers {minus_len}. Minus-strand opening energies are "
        "looked up relative to the contig length, so they are unreliable for this "
        "contig; regenerate the profile with `sioff accessibility`."
    )
    return False


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


def _fold_full_chromosome(
    sequence: str,
    chrom: str,
    window_size: int,
    max_span: int,
    unpaired_prob: int,
    output_path: str,
    chunk_size: int = 1_000_000,
    temperature: float = 37.0,
) -> str:
    """Fold both strands of a chromosome in chunks and stream to Parquet.

    Top-level (not a method) so ProcessPoolExecutor can pickle it; only the
    output path string is pickled back, avoiding O(N×u) OOM on large chromosomes.
    """
    RT = _GAS_CONSTANT * (temperature + 273.15)
    seq_len = len(sequence)

    schema = pa.schema(
        [("position", pa.int32()), ("strand", pa.utf8())]
        + [(f"u{i}", pa.float32()) for i in range(1, unpaired_prob + 1)]
    )

    def _fold_strand(writer: pq.ParquetWriter, seq: str, strand_char: str) -> None:
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
                def _cb(
                    v, v_size, i, maxsize, what, data, _pm=cast(list, probs_matrix)
                ):
                    if (what & RNA.PROBS_WINDOW_UP) and v is not None:
                        _pm[i] = list(v)

                fc.probs_window(unpaired_prob, RNA.PROBS_WINDOW_UP, _cb)
                probs_ok = True
            except Exception:
                probs_ok = False
                probs_matrix = None

            n_pos = data_end_in_chunk - data_start_in_chunk
            positions = np.arange(pos + 1, pos + n_pos + 1, dtype=np.int32)
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

            arrays = [
                pa.array(positions),
                pa.array([strand_char] * n_pos, type=pa.utf8()),
            ] + [pa.array(u_matrix[:, u - 1]) for u in range(1, unpaired_prob + 1)]
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))

            pos += chunk_size

    with pq.ParquetWriter(output_path, schema) as writer:
        _fold_strand(writer, sequence, "+")
        _fold_strand(writer, _reverse_complement(sequence), "-")

    return output_path


# ---------------------------------------------------------------------------
# Memory-mapped per-(contig, strand) cache
#
# `_fold_full_chromosome` writes one Parquet file per contig. Reading a whole
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
    re-quantises to the same value the eager path produced.
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
        here = pc.filter(t.column("position"), pc.equal(t.column("strand"), strand))
        m = pc.max(here).as_py() if len(here) else None
        if m is not None:
            max_pos = max(max_pos, int(m))
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
            mask = pc.equal(batch.column("strand"), strand).to_numpy(
                zero_copy_only=False
            )
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

    index = np.concatenate(exc_index) if exc_index else np.empty(0, dtype=np.int64)
    value = np.concatenate(exc_value) if exc_value else np.empty(0, dtype=np.float32)
    order = np.argsort(index, kind="stable")
    index, value = index[order], value[order]
    n_exc = len(index)
    if n_rows and n_exc > 0.01 * n_rows * n_u:
        logger.warning(
            f"{chrom} {strand}: {n_exc:,} of {n_rows * n_u:,} values are not "
            "multiples of 0.1 kcal/mol (or exceed 25.5); the exceptions sidecar "
            "will be large. Write accessibility with 'sioff accessibility' for a "
            "compact cache."
        )

    npy_path, npz_path = cache_paths(cache_dir, chrom, strand)
    _write_atomically(npz_path, lambda tmp: np.savez(tmp, index=index, value=value))
    _write_atomically(npy_path, lambda tmp: np.save(tmp, codes))
    logger.info(
        f"Built accessibility cache {npy_path.name}: {max_pos:,} positions × {n_u}, "
        f"{n_exc:,} exact-value exceptions"
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
    absent from the Parquet keep code 255 (25.5 kcal/mol), as the eager path
    did; positions must be unique within a strand, as the writer guarantees.
    A strand with no rows gets an empty ``(0, n_u)`` cache so the build stays
    idempotent, and queries on it raise like the eager path.

    Files are written to a temporary name and renamed into place.

    Returns
    -------
    dict[str, pathlib.Path]
        ``{"+": <plus .npy>, "-": <minus .npy>}``.
    """
    parquet_path = Path(parquet_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    chrom = _chrom_of(parquet_path)
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
    return built


def _layout_strand_lengths(
    layout: list[tuple[int, str, int, int, bool]],
) -> tuple[Optional[int], Optional[int]]:
    """``(plus, minus)`` maximum positions from a `_row_group_layout`."""
    lens: dict[str, int] = {}
    for _, strand, _, pos_max, _ in layout:
        lens[strand] = max(lens.get(strand, 0), int(pos_max))
    return lens.get("+"), lens.get("-")


def _cache_is_valid(npy_path: Path, npz_path: Path) -> bool:
    """Both files present and the ``.npy`` header is a 2-D uint8 array.

    Only the header is checked (dtype, rank); the row count is not compared
    with the Parquet, so a cache built from an older Parquet is trusted.
    """
    if not (npy_path.is_file() and npz_path.is_file()):
        return False
    try:
        arr = np.load(npy_path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return arr.ndim == 2 and arr.dtype == np.uint8


def ensure_accessibility_cache(
    parquet_dir: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    contigs: Optional[Iterable[str]] = None,
    workers: int = 1,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> list[str]:
    """Build the uint8 cache for every Parquet contig that lacks one.

    Meant to run once in the parent process before an off-target worker pool
    starts, so workers only ever memory-map finished files. Contigs whose two
    strand caches already exist and pass `_cache_is_valid` are skipped, which
    makes repeated calls cheap. A directory without any Parquet (only
    ``.npy`` caches) builds nothing.

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
    """
    parquet_dir = Path(parquet_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else parquet_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parquet_dir.glob(f"*{PARQUET_SUFFIX}"))
    if contigs is not None:
        wanted = set(contigs)
        files = [f for f in files if _chrom_of(f) in wanted]
    missing = [
        f
        for f in files
        if not all(
            _cache_is_valid(*cache_paths(cache_dir, _chrom_of(f), s)) for s in "+-"
        )
    ]
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
            build_accessibility_cache(f, cache_dir, batch_rows)
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
    """One strand as the eager float32 ``[n_pos, n_u]`` array (in-memory frames)."""

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
    compute_genome_accessibility() in ``data_dir`` and served one of two ways:

    - eager (``cache_dir=None``, the default): a chromosome's Parquet is read
      whole, one strand is filtered out and materialised as a dense float32
      ``[max_pos, n_u]`` array (chr1: 30 GB per strand). Nothing is written.
    - memory-mapped (``cache_dir`` given): the uint8 cache described at
      `build_accessibility_cache` (one ``<chrom>_<plus|minus>.access.npy``
      per strand, plus an exceptions sidecar) in ``cache_dir`` is
      ``np.load(mmap_mode="r")``-ed, so every process shares one page-cache
      copy and resident memory is only the pages it touches. A contig whose
      cache is missing is built from its Parquet on first use (batch runs
      build it up front with `ensure_accessibility_cache` so worker processes
      only map finished files); a ``cache_dir`` holding the complete cache
      needs no Parquet at all. Pass ``cache_dir=data_dir`` to keep the cache
      next to the Parquet.

    Both answer bit-for-bit alike. The LRU of ``max_cached`` slots bounds the
    number of loaded profiles (arrays or mappings) per process.

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
        # None: eager Parquet loader; a path: the memory-mapped uint8 cache there.
        self.cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir is not None else None
        )
        self._frames: Mapping[str, pl.DataFrame] = frames if frames is not None else {}
        self._max_cached = max_cached
        self._profiles: OrderedDict[str, _Profile] = OrderedDict()
        self._strands_checked: set[str] = set()

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
    ) -> Dict[str, Path]:
        """
        Compute accessibility for all sequences in the genome FASTA.

        Dispatches one task per chromosome to a ProcessPoolExecutor for
        parallel folding. Saves one Parquet file per chromosome containing
        columns [position, strand, u1, u2, ..., u{unpaired_prob}].

        Args:
            genome_path: Path to genome FASTA file.
            window_size: -W parameter (default 80).
            max_span: -L parameter (default 40).
            unpaired_prob: -u parameter (default 30).
            workers: Number of parallel folding workers.
            progress: Rich Progress instance (optional).
            progress_callback: Simple callback(advance, description) (optional).

        Returns:
            Dictionary mapping 'chrom' to the output Parquet path.
        """
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
                out_path = str(self._require_dir() / f"{chrom}.accessibility.parquet")
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
        """The eager path: one strand of a whole-chromosome frame as float32[max_pos, n_u].

        Positions absent from the frame keep the 25.5 kcal/mol sentinel.
        """
        df = full.filter(pl.col("strand") == strand).sort("position")
        if df.height == 0:
            raise AccessibilityError(f"No data for strand {strand!r} in {source}")
        u_cols = _u_columns(df.columns)
        max_pos = int(cast(int, df["position"].max()))
        arr = np.full((max_pos, len(u_cols)), 25.5, dtype=np.float32)
        pos_0 = df["position"].to_numpy() - 1  # 1-based -> 0-based
        for col_idx, col_name in enumerate(u_cols):
            arr[pos_0, col_idx] = df[col_name].to_numpy()
        return _DenseProfile(arr)

    def _dense_from_parquet(self, chrom: str, strand: str) -> tuple[_DenseProfile, str]:
        """Eager loader: read the chromosome's Parquet whole and densify one strand."""
        path = self._find_profile(chrom)
        if not path:
            raise AccessibilityError(
                f"Profile for {chrom} not found in {self.data_dir}. "
                f"Expected {self._require_dir() / f'{chrom}{PARQUET_SUFFIX}'}."
            )
        source = str(path)
        return self._dense_profile(pl.read_parquet(path), strand, source), source

    def _open_cache(self, chrom: str, strand: str) -> _CachedProfile | None:
        """Memory-map the (chrom, strand) cache if both its files exist."""
        cache_dir = self.cache_dir
        if cache_dir is None:
            return None
        npy_path, npz_path = cache_paths(cache_dir, chrom, strand)
        if not (npy_path.is_file() and npz_path.is_file()):
            return None
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
        """Map the cache for (chrom, strand), building it from Parquet if absent."""
        profile = self._open_cache(chrom, strand)
        if profile is None:
            parquet = self._find_profile(chrom)
            if parquet is None:
                expected_npy = cache_paths(cast(Path, self.cache_dir), chrom, strand)[0]
                raise AccessibilityError(
                    f"Profile for {chrom} not found: neither the cache "
                    f"{expected_npy} nor the Parquet "
                    f"{self._require_dir() / f'{chrom}{PARQUET_SUFFIX}'} exists."
                )
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
        if profile.n_pos == 0:
            raise AccessibilityError(f"No data for strand {strand!r} in {chrom}")
        return profile

    def _ensure_profile(self, chrom: str, strand: str) -> str:
        """
        Ensure the profile for (chrom, strand) is loaded, applying LRU eviction.

        Returns the profile_key string.

        In-memory frames and, without ``cache_dir``, on-disk Parquet are
        materialised as a dense float32 array (the eager path); with
        ``cache_dir`` on-disk profiles are memory-mapped from the uint8 cache,
        built from the Parquet first if needed. Positions absent from the
        source keep the default sentinel value of 25.5 kcal/mol in all cases.

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
            if self.cache_dir is None:
                profile, source = self._dense_from_parquet(chrom, strand)
            else:
                profile = self._cached_from_disk(chrom, strand)
                source = str(cache_paths(self.cache_dir, chrom, strand)[0])

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

        In-memory frames: a group-by on ``position``. Parquet: row-group
        statistics (no data read). Cache: the ``.npy`` headers. Any failure
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
            if self.cache_dir is not None:
                out: list[Optional[int]] = []
                for strand in ("+", "-"):
                    npy = cache_paths(self.cache_dir, chrom, strand)[0]
                    out.append(
                        int(np.load(npy, mmap_mode="r").shape[0])
                        if npy.is_file()
                        else None
                    )
                return out[0], out[1]
            path = self._find_profile(chrom)
            if path is None:
                return None, None
            return _layout_strand_lengths(_row_group_layout(pq.ParquetFile(path)))
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


# ---------------------------------------------------------------------------
# Streaming gather: one contig Parquet, read row group by row group.
# ---------------------------------------------------------------------------
_OUT_OF_BOUNDS_ENERGY = (
    10.0  # the `np.full` default of annotate_opening_energy_vectorized
)
_MISSING_POSITION_ENERGY = 25.5  # the `np.full` default of _ensure_profile


def _row_group_layout(pf: pq.ParquetFile) -> list[tuple[int, str, int, int, bool]]:
    """Describe each row group as ``(index, strand, pos_min, pos_max, mixed)``.

    Taken from the Parquet column statistics when they are present and the
    row group holds a single strand (what ``_fold_full_chromosome`` writes);
    otherwise the two key columns of that row group are read (cheap: ~5 B/row)
    and a mixed-strand row group yields one entry per strand with
    ``mixed=True`` so the reader masks rows by strand.
    """
    md = pf.metadata
    names = pf.schema_arrow.names
    i_pos = names.index("position")
    i_strand = names.index("strand")
    layout: list[tuple[int, str, int, int, bool]] = []
    for rg in range(md.num_row_groups):
        rgm = md.row_group(rg)
        if rgm.num_rows == 0:
            continue
        ps = rgm.column(i_pos).statistics
        ss = rgm.column(i_strand).statistics
        if (
            ps is not None
            and ps.has_min_max
            and ss is not None
            and ss.has_min_max
            and ss.min == ss.max
        ):
            strand = ss.min.decode() if isinstance(ss.min, bytes) else str(ss.min)
            layout.append((rg, strand, int(ps.min), int(ps.max), False))
            continue
        tbl = pf.read_row_group(rg, columns=["position", "strand"])
        pos = tbl.column("position").to_numpy()
        strands = tbl.column("strand").to_numpy(zero_copy_only=False)
        uniq = np.unique(strands)
        for strand in uniq:
            m = strands == strand if len(uniq) > 1 else slice(None)
            p = pos[m]
            layout.append((rg, str(strand), int(p.min()), int(p.max()), len(uniq) > 1))
    return layout


def gather_opening_energies(
    parquet_path: Path,
    strands: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Opening energies for hits on one contig, streamed from its Parquet file.

    Bit-for-bit equivalent to ``annotate_opening_energy_vectorized`` restricted
    to one contig, but never materialises the dense ``[max_pos, u]`` profile:
    the needed positions are sorted, the row-group statistics select the row
    groups that hold them, and each of those is read once with only the ``u``
    columns the hits in it need (peak memory ~ one row group).

    Parameters
    ----------
    parquet_path : pathlib.Path
        ``{chrom}.accessibility.parquet`` written by ``sioff accessibility``.
    strands : numpy.ndarray
        ``'+'``/``'-'`` per hit (any array comparable to those strings).
    starts, ends : numpy.ndarray
        1-based inclusive RIsearch coordinates per hit.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        ``energies`` (float32, one per hit, quantised to 0.1; 10.0 when out of
        bounds or when the strand has no rows; 25.5 when the position is absent
        from the file) and ``stats`` with ``row_groups_total``,
        ``row_groups_read``, ``bytes_read`` (compressed column-chunk bytes
        actually read) and ``bytes_file`` (all column chunks of the file).
    """
    pf = pq.ParquetFile(parquet_path)
    md = pf.metadata
    names = pf.schema_arrow.names
    u_cols = sorted(
        [c for c in names if c.startswith("u") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    n_u = len(u_cols)
    col_index = {c: names.index(c) for c in names}
    layout = _row_group_layout(pf)
    check_strand_lengths(
        _chrom_of(parquet_path), *_layout_strand_lengths(layout), str(parquet_path)
    )

    stats = {
        "row_groups_total": md.num_row_groups,
        "row_groups_read": 0,
        "bytes_read": 0,
        "bytes_file": sum(
            md.row_group(rg).column(c).total_compressed_size
            for rg in range(md.num_row_groups)
            for c in range(md.num_columns)
        ),
    }

    starts = np.asarray(starts).astype(np.int64)
    ends = np.asarray(ends).astype(np.int64)
    strands = np.asarray(strands)
    out = np.full(len(starts), _OUT_OF_BOUNDS_ENERGY, dtype=np.float32)

    for strand in ("+", "-"):
        sel = np.flatnonzero(strands == strand)
        if sel.size == 0:
            continue
        rgs = [e for e in layout if e[1] == strand]
        if not rgs:
            # The eager path raises "No data for strand" and skips the group.
            continue
        seq_len = max(e[3] for e in rgs)

        # Same cells as annotate_opening_energy_vectorized.
        row_idx, col_idx, in_bounds = site_cells(
            strand, starts[sel], ends[sel], seq_len, n_u
        )

        ib = np.flatnonzero(in_bounds)
        vals = np.full(ib.size, _MISSING_POSITION_ENERGY, dtype=np.float32)
        want_pos = row_idx[ib] + 1  # back to the file's 1-based positions
        want_col = col_idx[ib]
        order = np.argsort(want_pos, kind="stable")
        sorted_pos = want_pos[order]

        for rg, _, pos_min, pos_max, mixed in rgs:
            lo = int(np.searchsorted(sorted_pos, pos_min, side="left"))
            hi = int(np.searchsorted(sorted_pos, pos_max, side="right"))
            if lo == hi:
                continue
            sub = order[lo:hi]
            cols_needed = np.unique(want_col[sub])
            read_cols = ["position"] + (["strand"] if mixed else [])
            read_cols += [u_cols[int(c)] for c in cols_needed]
            tbl = pf.read_row_group(rg, columns=read_cols)
            rgm = md.row_group(rg)
            stats["row_groups_read"] += 1
            stats["bytes_read"] += sum(
                rgm.column(col_index[c]).total_compressed_size for c in read_cols
            )

            pos = tbl.column("position").to_numpy()
            mask = None
            if mixed:
                mask = tbl.column("strand").to_numpy(zero_copy_only=False) == strand
                pos = pos[mask]
            reorder = None
            if pos.size > 1 and not np.all(pos[1:] >= pos[:-1]):
                reorder = np.argsort(pos, kind="stable")
                pos = pos[reorder]

            j = np.searchsorted(pos, sorted_pos[lo:hi], side="right") - 1
            hit = (j >= 0) & (pos[np.maximum(j, 0)] == sorted_pos[lo:hi])
            sub_cols = want_col[sub]
            for c in cols_needed:
                arr = tbl.column(u_cols[int(c)]).to_numpy()
                if mask is not None:
                    arr = arr[mask]
                if reorder is not None:
                    arr = arr[reorder]
                m = hit & (sub_cols == c)
                vals[sub[m]] = arr[j[m]]

        # Quantize to 0.1 resolution (match legacy query_single behaviour)
        vals = (np.round(vals * 10.0) / 10.0).astype(np.float32)
        out[sel[ib]] = vals

    return out, stats
