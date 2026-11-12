"""Service for executing RIsearch and validating siRNA inputs.

Uses the risearch PyO3 bindings for in-process execution — no subprocess or
intermediate TSV files.
"""

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import polars as pl
from Bio import SeqIO
from loguru import logger

from sioff.services.risearch_parser import _FIRST_WORD

# NOTE: `risearch` (the PyO3 bindings) is imported lazily inside the methods that
# use it (index_target / run_search), not at module top. This keeps `import sioff`
# and all non-RIsearch code paths (off-targets, accessibility) working when the
# optional `risearch` extra is not installed.


class RIsearchError(Exception):
    """Raised when RIsearch execution fails."""

    pass


# Nearest-neighbour scoring models (DSM ids) bundled with risearch 3.0.0a4. The
# tables are compiled into the extension module and not enumerable at runtime,
# so the set is mirrored here to fail early with a useful message; risearch is
# still the final authority and raises "unknown DSM id" for anything it does not
# know. Turner 1999 (`t99`) was dropped upstream in 3.0.0a3. Besides these ids
# risearch also loads a custom long-form DSM table from a TSV path
# (`q1 q2 t1 t2 delta_g_kcal_per_mol`, see risearch's tools/dsm); siOFF only
# checks that the file exists and leaves the format to risearch.
#
#   t04          Turner 2004            RNA-RNA (default)
#   slh04        SantaLucia-Hicks 2004  DNA-DNA
#   s95-rna-dna  Sugimoto 1995          RNA query / DNA target
#   s95-dna-rna  Sugimoto 1995          DNA query / RNA target
#
# Note `s95` alone is the data directory, not an id — risearch rejects it.
VALID_DSM_IDS = frozenset({"t04", "slh04", "s95-rna-dna", "s95-dna-rna"})

# Temperatures risearch accepts, in °C (it raises outside this range, but only
# once the search is under way).
TEMPERATURE_RANGE = (0, 50)

# Columns added to the hits by `alignment=True`: risearch's query coordinates
# (same convention as `start`/`end`) and its pairing string.
ALIGNMENT_COLUMNS = ("q_start", "q_end", "alignment")


class RIsearchService:
    """Wrapper for the risearch PyO3 bindings.

    An internal registry maps each index path to its source target FASTA so
    that run_search() can resolve integer target indices to sequence names.
    """

    def __init__(self) -> None:
        self._target_registry: Dict[str, Path] = {}

    def validate_sirna_fasta(self, path: Path) -> List[str]:
        """Validate siRNA FASTA for existence, format, and unique IDs. Returns ordered ID list."""
        if not path.exists():
            raise FileNotFoundError(f"siRNA FASTA file not found: {path}")

        ids: List[str] = []
        seen: set[str] = set()
        for record in SeqIO.parse(path, "fasta"):
            if record.id in seen:
                raise ValueError(f"Duplicate siRNA ID: '{record.id}'")
            seen.add(record.id)
            ids.append(record.id)

        logger.info(f"Validated {len(ids)} siRNA(s) from {path.name}")
        return ids

    def index_target(
        self, target_path: Path, index_path: Optional[Path] = None
    ) -> Path:
        """Create or reuse a RIsearch index for a target FASTA.

        Registers target_path so run_search() can resolve sequence names.
        Reuses an existing index if it is newer than the target.
        """
        if not target_path.exists():
            raise FileNotFoundError(f"Target FASTA not found: {target_path}")

        if index_path is None:
            index_path = target_path.with_suffix(".idx")

        self._target_registry[str(index_path)] = target_path

        if index_path.exists():
            if index_path.stat().st_mtime > target_path.stat().st_mtime:
                logger.info(f"Reusing existing index: {index_path}")
                return index_path
            logger.info(f"Index outdated, rebuilding: {index_path}")

        try:
            import risearch

            risearch.index(target_path, index_path)
            logger.info(
                f"Created index: {index_path} ({index_path.stat().st_size} bytes)"
            )
            return index_path
        except Exception as e:
            raise RIsearchError(f"RIsearch index failed: {e}") from e

    def run_search(
        self,
        query_path: Path,
        index_path: Path,
        target_fasta: Optional[Path] = None,
        seed_length: int = 6,
        max_extension: int = 20,
        energy_threshold: float = -10.0,
        seed_start: Optional[int] = None,
        seed_end: Optional[int] = None,
        seed_wobble: bool = True,
        matrix: Union[str, Path] = "t04",
        mismatches: int = 0,
        mismatch_prefix: int = 1,
        mismatch_suffix: int = 0,
        temperature: int = 37,
        penalty: float = 0.0,
        seed_energy: float = 0.0,
        no_max_prune: bool = False,
        no_dedup: bool = False,
        alignment: bool = False,
        threads: Optional[int] = None,
        sense_only: bool = False,
    ) -> pl.DataFrame:
        """Run RIsearch and return hits as a DataFrame (sirna_id, chrom, start, end, strand, energy).

        target_fasta is required when the index was not built in this session
        via index_target() — needed to resolve integer target indices to names.

        Every option of ``risearch.search`` is passed explicitly, with siOFF's
        own default, so an upstream default change cannot move our numbers
        (see ``penalty`` below for the time it did). The defaults reproduce the
        searches siOFF ran before these options were exposed.

        Seed geometry and energy-model parameters mirror the RIsearch2 CLI:

        ``seed_start``/``seed_end``/``seed_length``
            The ``-s n:m/l`` seed specification — the seed must fall within query
            positions ``n..m`` and be ``l`` nt long. Leaving start/end as ``None``
            constrains length only, which is RIsearch's default behaviour.
        ``seed_wobble``
            ``False`` is the CLI's ``--noGUseed``: forbid G:U wobble pairs *inside
            the seed*. This affects seed location only, not the energy model, so it
            is a candidate-generation knob rather than a rescoring one. Measured on
            3,826 fly S2 3'UTRs x 10 miRNAs it removes **84.7%** of hits
            (783,836 -> 119,928).
        ``matrix``
            the ``-z`` nearest-neighbour energy parameter set; one of
            ``VALID_DSM_IDS`` (``t04`` by default) or the path of a custom
            long-form DSM TSV table.

        The other ``risearch.search`` options:

        ``mismatches``, ``mismatch_prefix``, ``mismatch_suffix``
            Mismatches allowed inside a seed, and the matches required at its
            5' and 3' ends around them (risearch's Python defaults, 1 and 0;
            its CLI ``--mismatch-max`` instead sets both to the mismatch count).
            A mismatched seed holding an uninterrupted run of the full seed
            length is dropped, since the perfect seed already covers it.
        ``temperature``
            Energy-model temperature in °C, an integer in ``TEMPERATURE_RANGE``;
            37 is risearch's own default. A custom DSM table ignores it.
        ``penalty``
            kcal/mol added per extended nucleotide, included in the reported
            energy (risearch >= 3.0.0a4). RIsearch2's C ``-d`` took dacal/mol,
            so its recommended 30 is 0.3 here. Its upstream default was 3.5 in
            0.0.0a1 and 0.0 from 3.0.0a2 on; siOFF's is 0.0 either way.
        ``seed_energy``
            Accepted by risearch for compatibility; no effect in 3.0.0a4/a5.
        ``no_max_prune``
            Also extend non-maximal seeds (redundant hits).
        ``no_dedup``
            Report every seed's hit instead of one best hit per duplex.
        ``alignment``
            Add ``ALIGNMENT_COLUMNS``: the query coordinates and risearch's
            pairing string (``P`` pair, ``W`` G:U, ``U`` mismatch, ``T`` / ``Q``
            extra target / query base, query 5' to 3'). Off by default: the
            hits are the same either way, and skipping the traceback is faster.
        ``threads``
            Worker threads; ``None`` uses ``RAYON_NUM_THREADS``, else every core.

        One option is siOFF's own, not risearch's:

        ``sense_only``
            Keep only ``+``-strand hits, for a target FASTA whose records are
            already in sense orientation (a transcriptome). risearch indexes
            both strands and has no option to search one, so the ``-`` hits are
            dropped after the search: the returned frame shrinks, but the
            search's time and peak memory do not.
        """

        if not query_path.exists():
            raise FileNotFoundError(f"Query FASTA not found: {query_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")

        resolved_target = target_fasta or self._target_registry.get(str(index_path))
        if resolved_target is None:
            raise RIsearchError(
                "Cannot resolve target sequence names: pass target_fasta= or "
                "build the index via index_target() first."
            )

        if str(matrix) not in VALID_DSM_IDS and not Path(matrix).is_file():
            raise RIsearchError(
                f"unknown matrix {str(matrix)!r}; expected one of "
                f"{', '.join(sorted(VALID_DSM_IDS))} or the path of an existing "
                "DSM TSV table"
            )
        if (seed_start is None) != (seed_end is None):
            raise RIsearchError(
                "seed_start and seed_end must be given together (the '-s n:m/l' form)"
            )
        if seed_start is not None and seed_end is not None:
            if seed_start < 1:
                raise RIsearchError(f"seed_start is 1-based, got {seed_start}")
            if seed_end < seed_start:
                raise RIsearchError(f"seed_end {seed_end} < seed_start {seed_start}")
            # Catch the RIsearch2 '-s 2:8/8' case up front. The C binary prints
            # "Invalid seed length (exceeds interval)" and then raises SIGABRT rather
            # than exiting cleanly; the Rust bindings raise, but only per query and
            # only once the search is already under way.
            window = seed_end - seed_start + 1
            if seed_length > window:
                raise RIsearchError(
                    f"seed length {seed_length} exceeds the {window}-nt window "
                    f"{seed_start}:{seed_end} — positions {seed_start}-{seed_end} "
                    f"span {window} nt"
                )
        for name, value in (
            ("mismatches", mismatches),
            ("mismatch_prefix", mismatch_prefix),
            ("mismatch_suffix", mismatch_suffix),
        ):
            if value < 0:
                raise RIsearchError(f"{name} must be 0 or more, got {value}")
        low, high = TEMPERATURE_RANGE
        if not low <= temperature <= high:
            raise RIsearchError(
                f"temperature must be {low}-{high} °C (risearch's range), got {temperature}"
            )
        if threads is not None and threads < 1:
            raise RIsearchError(f"threads must be at least 1, got {threads}")

        try:
            import risearch

            store = risearch.TargetRegistry.open(index_path)
            raw = risearch.search(
                query_path,
                store,
                seed_length=seed_length,
                seed_start=seed_start,
                seed_end=seed_end,
                max_extension=max_extension,
                energy_threshold=energy_threshold,
                seed_wobble=seed_wobble,
                matrix=matrix,
                mismatches=mismatches,
                mismatch_prefix=mismatch_prefix,
                mismatch_suffix=mismatch_suffix,
                temperature=temperature,
                # Passed, never inherited. `penalty`'s upstream default changed
                # 3.5 -> 0.0 between pins, and 0.0.0a1 applied it with the wrong
                # sign, so simply omitting it moved our E_min by 7 kcal/mol on a
                # pin bump with no siOFF change. siOFF's default 0.0 matches the
                # original C default (`extPen = 0`). Exposed since risearch
                # 3.0.0a4 settled its meaning (kcal/mol per extended nucleotide,
                # in the reported energy) — see RTH-tools/risearch#27.
                penalty=penalty,
                seed_energy=seed_energy,
                no_max_prune=no_max_prune,
                no_dedup=no_dedup,
                alignment=alignment,
                threads=threads,
            )
        except Exception as e:
            raise RIsearchError(f"RIsearch search failed: {e}") from e

        if sense_only:
            raw = raw.filter(pl.col("strand") == "+")

        extra = list(ALIGNMENT_COLUMNS) if alignment else []
        if raw.is_empty():
            logger.info("Search complete: 0 hits")
            return pl.DataFrame(
                schema={
                    "sirna_id": pl.Utf8,
                    "chrom": pl.Utf8,
                    "start": pl.Int32,
                    "end": pl.Int32,
                    "strand": pl.Utf8,
                    "energy": pl.Float32,
                    **(
                        {"q_start": pl.Int32, "q_end": pl.Int32, "alignment": pl.Utf8}
                        if alignment
                        else {}
                    ),
                }
            )

        if {"query_name", "target_name"} <= set(raw.columns):
            # RIsearch's own names. Its indices count the records it kept,
            # and it drops a record that is empty after normalisation, so
            # looking them up in another parse of the FASTA named every hit
            # on a later record after an earlier one.
            sirna_ids = _first_words(raw["query_name"])
            chroms = _first_words(raw["target_name"])
        else:
            # A risearch whose hits carry the indices only.
            query_names = pl.Series(_fasta_names(str(query_path)), dtype=pl.Utf8)
            target_names = pl.Series(_fasta_names(str(resolved_target)), dtype=pl.Utf8)
            sirna_ids = query_names.gather(raw["query_idx"])
            chroms = target_names.gather(raw["target_idx"])

        df = (
            raw.with_columns([sirna_ids.alias("sirna_id"), chroms.alias("chrom")])
            .rename({"t_start": "start", "t_end": "end"})
            .select(["sirna_id", "chrom", "start", "end", "strand", "energy", *extra])
            .cast({"start": pl.Int32, "end": pl.Int32, "energy": pl.Float32})
        )
        if alignment:
            df = df.cast({"q_start": pl.Int32, "q_end": pl.Int32})

        logger.info(f"Search complete: {df.height} hits")
        return df

    def self_hybridization_emin(
        self,
        sequence: str,
        sirna_id: str = "query",
        matrix: Union[str, Path] = "t04",
        temperature: int = 37,
    ) -> float:
        """Compute E_min by searching a siRNA against itself.

        seed_length = len(seq) - 1 ensures only near-full-length hybridisations,
        replicating ``risearch2.x -q siRNA.fa -i siRNA.pksuf -s len-1 -e 0``.
        ``matrix`` is the energy parameter set (bundled id or DSM TSV path) and
        ``temperature`` the energy-model temperature in °C (an integer in
        ``TEMPERATURE_RANGE``); both should be the ones the predictions were
        searched with, since E_min is the reference energy their alpha/gamma
        clamp is measured against.
        Returns 0.0 if no hits found.
        """
        seq_dna = sequence.upper().replace("U", "T")
        with tempfile.TemporaryDirectory(prefix="risearch_self_") as tmpdir:
            fasta_path = Path(tmpdir) / "sirna.fa"
            index_path = Path(tmpdir) / "sirna.idx"
            fasta_path.write_text(f">{sirna_id}\n{seq_dna}\n")
            self.index_target(fasta_path, index_path)
            df = self.run_search(
                fasta_path,
                index_path,
                target_fasta=fasta_path,
                seed_length=len(seq_dna) - 1,
                energy_threshold=0.0,
                matrix=matrix,
                temperature=temperature,
            )
            if df.is_empty():
                logger.warning(
                    f"No self-hybridisation hits for {sirna_id}, using E_min=0.0"
                )
                return 0.0
            emin = float(cast(float, df["energy"].min()))
            logger.debug(f"Self-hyb E_min {sirna_id}: {emin:.4f} kcal/mol")
            return emin


def _first_words(names: pl.Series) -> pl.Series:
    """Cut each name at its first whitespace: the ID of a FASTA header.

    RIsearch names a record by its whole header line; the annotation, the
    accessibility profiles and the siRNA table use the first word. Done per
    distinct name, since a search returns many hits on few names.
    """
    distinct = names.unique()
    words = distinct.str.extract(_FIRST_WORD, 1).fill_null(distinct)
    return names.replace_strict(distinct, words, return_dtype=pl.Utf8)


@lru_cache(maxsize=32)
def _fasta_names(path_str: str) -> tuple:
    """Return sequence IDs from a FASTA file (cached by path string)."""
    return tuple(r.id for r in SeqIO.parse(path_str, "fasta"))
