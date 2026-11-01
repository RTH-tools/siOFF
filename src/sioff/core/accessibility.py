"""Pure core for accessibility computation, returning in-memory DataFrames.

Folds each sequence in the FASTA with ViennaRNA and returns one Polars DataFrame
per chromosome (schema ``[position, strand, u1..u{unpaired_prob}]`` -- the same
schema the CLI streams to ``{chrom}.accessibility.parquet``), holding nothing on
disk. For genome-scale inputs prefer the CLI's streaming path; the in-memory dict
keeps every profile resident at once.

The public wrapper is [`sioff.accessibility`][sioff.accessibility]; its result feeds the
``accessibility`` argument of [`sioff.off_targets`][sioff.off_targets].
"""

from pathlib import Path

import numpy as np
import polars as pl

from sioff.services.accessibility import (
    _check_fold_strand,
    _strands_to_fold,
    fold_sequence,
)
from sioff.services.helpers import read_fasta


def _profile_to_df(
    profile: np.ndarray, strand: str, unpaired_prob: int
) -> pl.DataFrame:
    """Turn a [seq_len, u] opening-energy array into the parquet-equivalent schema.

    Returns
    -------
    polars.DataFrame
        Columns ``position``, ``strand`` and ``u1..u{unpaired_prob}``.
    """
    n_pos = 0 if profile.size == 0 else profile.shape[0]
    columns: dict[str, pl.Series] = {
        "position": pl.Series("position", np.arange(1, n_pos + 1, dtype=np.int32)),
        "strand": pl.Series("strand", [strand] * n_pos, dtype=pl.Utf8),
    }
    for u in range(1, unpaired_prob + 1):
        col = (
            profile[:, u - 1].astype(np.float32)
            if n_pos
            else np.array([], dtype=np.float32)
        )
        columns[f"u{u}"] = pl.Series(f"u{u}", col, dtype=pl.Float32)
    return pl.DataFrame(columns)


def compute_accessibility(
    genome: Path,
    window_size: int = 80,
    max_span: int = 40,
    unpaired_prob: int = 30,
    temperature: float = 37.0,
    strand: str = "both",
) -> dict[str, pl.DataFrame]:
    """Compute per-chromosome accessibility profiles in memory.

    Every sequence in ``genome`` is folded twice with ViennaRNA's RNAplfold
    model, once as given (``+`` strand) and once as its reverse complement
    (``-`` strand); ``strand="+"`` skips the reverse complement. The resulting
    opening energies are returned as one Polars DataFrame per chromosome with
    columns ``[position, strand, u1..u{unpaired_prob}]``, the strands stacked
    (``+`` then ``-``), matching the on-disk Parquet schema written by
    ``sioff accessibility``. ``position``
    is 1-based along the folded sequence and ``u{k}`` is the opening energy
    (kcal/mol) of the ``k``-nt stretch ending at that position. Writes no
    files and keeps every profile resident at once, so prefer the CLI's
    streaming path for genome-scale inputs.

    Parameters
    ----------
    genome : pathlib.Path
        FASTA file whose records are folded; each record ID becomes a key of
        the result (CLI positional ``genome``).
    window_size : int, default 80
        RNAplfold window size ``W`` (CLI ``-W/--window``).
    max_span : int, default 40
        Maximum base-pair span ``L`` (CLI ``-L/--span``).
    unpaired_prob : int, default 30
        Maximum unpaired-stretch length ``u`` (CLI ``-u/--unpaired``); the
        output holds columns ``u1..u{unpaired_prob}``.
    temperature : float, default 37.0
        Folding temperature in degrees Celsius (CLI ``-T/--temperature``).
    strand : {"both", "+"}, default "both"
        ``"+"`` folds each sequence as given only, for a transcriptome FASTA
        whose records are already sense; the frames then hold no ``-`` rows
        (CLI ``--strand``).

    Returns
    -------
    dict[str, polars.DataFrame]
        Mapping from chromosome (FASTA record ID) to its profile frame with
        columns ``position`` (Int32), ``strand`` (Utf8, ``"+"`` or ``"-"``) and
        ``u1..u{unpaired_prob}`` (Float32 opening energies).

    Raises
    ------
    FileNotFoundError
        When ``genome`` does not exist.
    AccessibilityError
        `sioff.services.accessibility.AccessibilityError`, when ``strand`` is
        not ``"both"`` or ``"+"``.

    See Also
    --------
    sioff.accessibility : Public wrapper accepting ``str`` paths.
    sioff.off_targets : Consumes the returned mapping via its ``accessibility``
        argument.
    """
    _check_fold_strand(strand)
    genome = Path(genome)
    if not genome.exists():
        raise FileNotFoundError(f"Genome FASTA not found: {genome}")

    profiles: dict[str, pl.DataFrame] = {}
    for chrom, sequence in read_fasta(genome):
        profiles[chrom] = pl.concat(
            [
                _profile_to_df(
                    fold_sequence(
                        seq,
                        window_size=window_size,
                        max_span=max_span,
                        unpaired_prob=unpaired_prob,
                        temperature=temperature,
                    ),
                    s,
                    unpaired_prob,
                )
                for s, seq in _strands_to_fold(sequence, strand)
            ]
        )
    return profiles
