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
        """Load RIsearch2 output, ``sioff search -o`` output or Parquet into a validated DataFrame."""
        _check_file(path)
        if path.suffix == ".parquet":
            # Only the six prediction columns are read: a stored E_min column
            # from older converters is dropped, since E_min is passed in.
            df = pl.read_parquet(path).select(RISEARCH_COLUMNS)
        elif _has_search_header(path):
            df = pl.read_csv(path, separator="\t", infer_schema=False).select(
                RISEARCH_COLUMNS
            )
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

    def list_directory_files(self, directory: Path) -> list[Path]:
        """List all RIsearch output files in a directory (*.gz, *.tsv, *.out, *.parquet)."""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        return sorted(f for f in directory.iterdir() if f.suffix in _VALID_SUFFIXES)
