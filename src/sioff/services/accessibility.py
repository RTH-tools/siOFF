import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, cast

import numpy as np
import polars as pl
import pyarrow as pa
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


class GenomeAccessibilityService:
    """
    Service to pre-compute and query genomic accessibility profiles.

    Profiles are stored as per-chromosome Parquet files produced by
    compute_genome_accessibility() and loaded on demand into float32 numpy
    arrays. An LRU cache (max_cached slots) bounds resident memory.

    `from_frames()` builds a service over in-memory per-chromosome
    DataFrames instead (the ``dict[chrom -> DataFrame]`` that
    `sioff.accessibility` returns); such a service never reads or writes
    disk and has ``data_dir is None``.
    """

    def __init__(
        self,
        data_dir: Optional[Path],
        max_cached: int = 4,
        frames: Optional[Mapping[str, pl.DataFrame]] = None,
    ):
        self.data_dir = data_dir
        if self.data_dir is not None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        self._frames: Mapping[str, pl.DataFrame] = frames if frames is not None else {}
        self._max_cached = max_cached
        self._profiles: OrderedDict[str, np.ndarray] = OrderedDict()

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
        p = self._require_dir() / f"{chrom}.accessibility.parquet"
        return p if p.exists() else None

    def _ensure_profile(self, chrom: str, strand: str) -> str:
        """
        Ensure the profile for (chrom, strand) is loaded, applying LRU eviction.

        Returns the profile_key string.

        Loads the chromosome's Parquet file, filters to the requested strand,
        and materialises a float32 numpy array indexed by 0-based position.
        Positions absent from the Parquet (e.g. un-foldable regions) keep the
        default sentinel value of 25.5 kcal/mol.

        Complexity: O(1) amortized. Eviction frees the oldest cached profile
        when the cache exceeds max_cached slots.
        """
        profile_key = f"{chrom}_{strand}"

        if profile_key in self._profiles:
            self._profiles.move_to_end(profile_key)
            return profile_key

        if chrom in self._frames:
            source: str = f"in-memory frame for {chrom}"
            full = self._frames[chrom]
        else:
            if self.data_dir is None:
                raise AccessibilityError(
                    f"Profile for {chrom} not found among the in-memory "
                    f"accessibility frames ({', '.join(sorted(self._frames)) or 'none'})."
                )
            path = self._find_profile(chrom)
            if not path:
                raise AccessibilityError(
                    f"Profile for {chrom} not found in {self.data_dir}. "
                    f"Expected {self._require_dir() / f'{chrom}.accessibility.parquet'}."
                )
            source = str(path)
            full = pl.read_parquet(path)

        df = full.filter(pl.col("strand") == strand).sort("position")
        if df.height == 0:
            raise AccessibilityError(f"No data for strand {strand!r} in {source}")

        u_cols = sorted(
            [c for c in df.columns if c.startswith("u") and c[1:].isdigit()],
            key=lambda x: int(x[1:]),
        )
        max_pos = int(cast(int, df["position"].max()))
        n_u = len(u_cols)
        arr = np.full((max_pos, n_u), 25.5, dtype=np.float32)
        pos_0 = df["position"].to_numpy() - 1  # 1-based → 0-based
        for col_idx, col_name in enumerate(u_cols):
            arr[pos_0, col_idx] = df[col_name].to_numpy()

        self._profiles[profile_key] = arr
        logger.info(f"Loaded accessibility profile for {chrom} {strand} from {source}")

        while len(self._profiles) > self._max_cached:
            evicted_key, _ = self._profiles.popitem(last=False)
            logger.debug(f"Evicted profile cache entry: {evicted_key}")

        return profile_key

    def query(self, chrom: str, start: int, end: int, strand: str = "+") -> np.ndarray:
        """Return opening energies for a region (0-based, half-open [start, end))."""
        profile_key = self._ensure_profile(chrom, strand)
        profile = self._profiles[profile_key]
        seq_len = len(profile)

        if start < 0 or end > seq_len:
            raise AccessibilityError(
                f"Query {start}-{end} out of bounds for {chrom} {strand} (len {seq_len})"
            )

        return profile[start:end]

    def query_single(
        self, chrom: str, start: int, end: int, strand: str = "+"
    ) -> float:
        """
        Fast-path: return a single opening energy value for a prediction.

        Mirrors the old pipeline's get_opening_energy() logic:
        - Picks the value at the 3'-end position using the interaction length
          as the u-column index.
        - Quantizes to 0.1 resolution (match legacy behavior).

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
        seq_len = len(profile)

        start0 = start - 1
        end0 = end  # half-open

        if start0 < 0 or end0 > seq_len:
            return 10.0

        interaction_len = end0 - start0
        matrix_width = profile.shape[1]
        col_idx = min(interaction_len, matrix_width) - 1
        if col_idx < 0:
            col_idx = 0
        row_idx = end0 - 1 if strand == "+" else start0
        raw_val = float(profile[row_idx, col_idx])

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
            seq_len = len(profile)

            idx = np.array(row_indices, dtype=np.int64)
            s = starts[idx]
            e = ends[idx]
            start0 = s - 1
            end0 = e
            interaction_len = end0 - start0
            in_bounds = (start0 >= 0) & (end0 <= seq_len)

            matrix_width = profile.shape[1]
            col_idx = np.minimum(interaction_len, matrix_width) - 1
            col_idx = np.maximum(col_idx, 0)
            row_idx = np.where(strand == "+", end0 - 1, start0)
            row_idx = np.clip(row_idx, 0, seq_len - 1)
            raw_vals = profile[row_idx, col_idx].astype(np.float32)

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

        # Same arithmetic as annotate_opening_energy_vectorized.
        s = starts[sel]
        e = ends[sel]
        start0 = s - 1
        end0 = e
        interaction_len = end0 - start0
        in_bounds = (start0 >= 0) & (end0 <= seq_len)
        col_idx = np.minimum(interaction_len, n_u) - 1
        col_idx = np.maximum(col_idx, 0)
        row_idx = end0 - 1 if strand == "+" else start0
        row_idx = np.clip(row_idx, 0, seq_len - 1)

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
