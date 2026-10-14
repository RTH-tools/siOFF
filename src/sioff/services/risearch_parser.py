"""Service for parsing RIsearch2 output files."""

import gzip
from pathlib import Path

import polars as pl

from sioff.models import RISEARCH_COLUMNS, RISEARCH_SCHEMA

_VALID_SUFFIXES = {".gz", ".tsv", ".out", ".parquet"}


def _has_search_header(path: Path) -> bool:
    """True for the headered TSV written by ``sioff search -o``."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return fh.readline().rstrip("\r\n").split("\t") == RISEARCH_COLUMNS


def _check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"RIsearch2 output file not found: {path}")


class RIsearchParser:
    """Parser for RIsearch2 prediction output files."""

    def load(self, path: Path) -> pl.DataFrame:
        """Load RIsearch2 TSV output (or ``sioff search -o`` output) into a validated DataFrame."""
        _check_file(path)
        if _has_search_header(path):
            df = pl.read_csv(path, separator="\t", infer_schema=False)
        else:
            # RIsearch2 layout, no header: pick columns by position. Read them
            # as text, then name and cast: polars >= 2.0 rejects `new_columns`
            # shorter than the file is wide.
            df = pl.read_csv(
                path,
                separator="\t",
                has_header=False,
                columns=[0, 3, 4, 5, 6, 7],
                infer_schema=False,
                truncate_ragged_lines=True,
            )
            df = df.rename(dict(zip(df.columns, RISEARCH_COLUMNS, strict=True)))
        return df.cast(pl.Schema(RISEARCH_SCHEMA)).filter(
            pl.col("sirna_id").is_not_null()
        )

    def summary(self, df: pl.DataFrame) -> dict:
        """Generate summary statistics for loaded predictions."""
        return {
            "row_count": df.height,
            "energy_min": df["energy"].min(),
            "energy_max": df["energy"].max(),
            "chromosomes": df["chrom"].unique().to_list(),
            "strands": df["strand"].unique().to_list(),
        }

    def load_single_file(self, file_path: Path) -> pl.DataFrame:
        """Load one per-siRNA RIsearch file and attach raw_e_min.

        Parquet files from convert_risearch_to_parquet.py already contain
        named columns and pre-computed raw_e_min — returned directly.
        """
        if file_path.suffix == ".parquet":
            return pl.read_parquet(file_path)

        df = self.load(file_path)
        raw_emin = df.group_by("sirna_id").agg(
            pl.col("energy").min().alias("raw_e_min")
        )
        return df.join(raw_emin, on="sirna_id", how="left")

    def list_directory_files(self, directory: Path) -> list[Path]:
        """List all RIsearch output files in a directory (*.gz, *.tsv, *.out, *.parquet)."""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        return sorted(f for f in directory.iterdir() if f.suffix in _VALID_SUFFIXES)
