"""Service for executing RIsearch and validating siRNA inputs.

Uses the risearch PyO3 bindings for in-process execution — no subprocess or
intermediate TSV files.
"""

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import polars as pl
from Bio import SeqIO
from loguru import logger

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
    ) -> pl.DataFrame:
        """Run RIsearch and return hits as a DataFrame (sirna_id, chrom, start, end, strand, energy).

        target_fasta is required when the index was not built in this session
        via index_target() — needed to resolve integer target indices to names.

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
                # Pinned, not inherited. `penalty`'s upstream default changed
                # 3.5 -> 0.0 between pins, and 0.0.0a1 applied it with the wrong
                # sign, so simply omitting it moved our E_min by 7 kcal/mol on a
                # pin bump with no siOFF change. 0.0 matches the original C
                # default (`extPen = 0`). Not exposed as a flag: its semantics
                # differ across versions (units, sign, and whether it reaches
                # the reported energy at all) — see RTH-tools/risearch#27.
                penalty=0.0,
            )
        except Exception as e:
            raise RIsearchError(f"RIsearch search failed: {e}") from e

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
                }
            )

        query_names = pl.Series(_fasta_names(str(query_path)), dtype=pl.Utf8)
        target_names = pl.Series(_fasta_names(str(resolved_target)), dtype=pl.Utf8)

        df = (
            raw.with_columns(
                [
                    query_names.gather(raw["query_idx"]).alias("sirna_id"),
                    target_names.gather(raw["target_idx"]).alias("chrom"),
                ]
            )
            .rename({"t_start": "start", "t_end": "end"})
            .select(["sirna_id", "chrom", "start", "end", "strand", "energy"])
            .cast({"start": pl.Int32, "end": pl.Int32, "energy": pl.Float32})
        )

        logger.info(f"Search complete: {df.height} hits")
        return df

    def self_hybridization_emin(self, sequence: str, sirna_id: str = "query") -> float:
        """Compute E_min by searching a siRNA against itself.

        seed_length = len(seq) - 1 ensures only near-full-length hybridisations,
        replicating ``risearch2.x -q siRNA.fa -i siRNA.pksuf -s len-1 -e 0``.
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
            )
            if df.is_empty():
                logger.warning(
                    f"No self-hybridisation hits for {sirna_id}, using E_min=0.0"
                )
                return 0.0
            emin = float(cast(float, df["energy"].min()))
            logger.debug(f"Self-hyb E_min {sirna_id}: {emin:.4f} kcal/mol")
            return emin

    def self_hybridization_emin_batch(self, fasta_path: Path) -> Dict[str, float]:
        """Compute self-hybridisation E_min for all siRNAs in a FASTA file.

        Each siRNA is searched against itself (seed_length = len - 1, energy ≤ 0),
        replicating the old pipeline's minimum-energy anchor for alpha/gamma clamping.
        """
        if not fasta_path.exists():
            raise FileNotFoundError(f"siRNA FASTA not found: {fasta_path}")

        records = list(SeqIO.parse(fasta_path, "fasta"))
        logger.info(
            f"Computing self-hybridisation E_min for {len(records)} siRNA(s)..."
        )
        result = {r.id: self.self_hybridization_emin(str(r.seq), r.id) for r in records}
        logger.info(f"Self-hybridisation complete: {len(result)} siRNA(s) processed")
        return result

    def search_single_sirna(
        self, query_path: Path, target_path: Path
    ) -> Tuple[float, int, int, str]:
        """Run RIsearch for a single siRNA and return (energy, start, end, strand) of the best hit.

        Returns (0.0, 0, 0, "+") if no hits found or search fails.
        """
        with tempfile.TemporaryDirectory(prefix="risearch_") as tmpdir:
            index_path = Path(tmpdir) / "target.idx"
            try:
                self.index_target(target_path, index_path)
                df = self.run_search(query_path, index_path)
                if df.is_empty():
                    logger.warning("No valid hits found")
                    return 0.0, 0, 0, "+"
                best = df.sort("energy").row(0, named=True)
                return (
                    float(best["energy"]),
                    int(best["start"]),
                    int(best["end"]),
                    str(best["strand"]),
                )
            except RIsearchError as e:
                logger.error(f"RIsearch failed: {e}")
                return 0.0, 0, 0, "+"


@lru_cache(maxsize=32)
def _fasta_names(path_str: str) -> tuple:
    """Return sequence IDs from a FASTA file (cached by path string)."""
    return tuple(r.id for r in SeqIO.parse(path_str, "fasta"))
