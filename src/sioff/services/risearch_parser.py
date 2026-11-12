"""Service for parsing RIsearch2 output files."""

import gzip
from pathlib import Path

import polars as pl

from sioff.models import RISEARCH_COLUMNS, RISEARCH_SCHEMA

_VALID_SUFFIXES = {".gz", ".tsv", ".out", ".parquet"}
_FIRST_WORD = r"^\s*(\S+)"


def first_word_targets(df: pl.DataFrame) -> pl.DataFrame:
    """Name every target (``chrom``) by the first word of its FASTA header.

    The ``risearch`` command line writes the whole header line as the target
    name (``rna-XM_1.2 CDS=163-1434``). The annotation (GTF seqname, or
    ``transcript_id`` with ``--type tw``), the accessibility profiles and
    ``sioff search`` all name a record by the header's first word, so a whole
    header matched none of them: the hit was dropped by the annotation join
    or kept the opening energy of an unknown record, without a message.

    ``sirna_id`` is left as it is: the siRNA table and the ``.summary`` files
    are keyed by the full ID, and a mismatch there is reported.

    Raises ``ValueError`` when two different names share a first word: cut,
    they would be one record, and their hits could not be told apart.
    """
    # The distinct names, not the rows: a genome-wide file has millions of
    # hits on a few names, and this runs on every load.
    names = df["chrom"].unique().drop_nulls().sort()
    if not names.str.contains(r"\s").any():
        return df
    # A name of whitespace only has no first word and is kept.
    words = names.str.extract(_FIRST_WORD, 1).fill_null(names)
    shared = (
        pl.DataFrame({"name": names, "word": words})
        .group_by("word", maintain_order=True)
        .agg(pl.col("name"))
        .filter(pl.col("name").list.len() > 1)
    )
    if shared.height:
        word, full = shared.row(0)
        raise ValueError(
            f"Target names {full[0]!r} and {full[1]!r} share the first word "
            f"{word!r}. A target is named by the first word of its FASTA "
            "header, as in the annotation and the accessibility profiles, so "
            "their hits cannot be told apart; give the records distinct first "
            "words"
        )
    return df.with_columns(
        pl.col("chrom").replace_strict(
            names, words, default=None, return_dtype=pl.String
        )
    )


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
        """Load RIsearch2 output, ``sioff search -o`` output or Parquet into a validated DataFrame.

        Targets are named by the first word of their FASTA header (see
        `first_word_targets`).
        """
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
        return first_word_targets(
            df.cast(pl.Schema(RISEARCH_SCHEMA)).filter(pl.col("sirna_id").is_not_null())
        )

    def list_directory_files(self, directory: Path) -> list[Path]:
        """List all RIsearch output files in a directory (*.gz, *.tsv, *.out, *.parquet)."""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        return sorted(f for f in directory.iterdir() if f.suffix in _VALID_SUFFIXES)
