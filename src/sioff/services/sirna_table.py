"""Reader for the ``--sirna-table`` TSV (``sirna_id``, ``on_target_id``, ``sequence``)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl

SIRNA_TABLE_COLUMNS = ("sirna_id", "on_target_id", "sequence")

_NUCLEOTIDES = frozenset("ACGT")


@dataclass(frozen=True)
class SirnaRecord:
    """One row of the siRNA table.

    Attributes
    ----------
    sirna_id : str
        Unique siRNA identifier.
    on_target_id : str or None
        Transcript the siRNA is designed against; None when the cell is empty.
    sequence : str
        Guide sequence, upper-case, in the DNA alphabet (U is turned into T).
    """

    sirna_id: str
    on_target_id: Optional[str]
    sequence: str


def read_sirna_table(path: Path) -> dict[str, SirnaRecord]:
    """Read and validate an siRNA table.

    Columns are matched by name, in any order; extra columns are ignored and
    lines starting with ``#`` are skipped.

    Parameters
    ----------
    path : Path
        Tab-separated file with a header row.

    Returns
    -------
    dict[str, SirnaRecord]
        Records keyed by ``sirna_id``, in file order.

    Raises
    ------
    ValueError
        If the file cannot be parsed, a required column is missing, a
        ``sirna_id`` repeats, or a sequence is empty or not made of A, C, G,
        T or U. Every message starts with ``"{path}: "``.
    """
    try:
        df = pl.read_csv(path, separator="\t", comment_prefix="#", infer_schema=False)
    except pl.exceptions.NoDataError as exc:
        raise ValueError(f"{path}: the table has no header row") from exc

    missing = [c for c in SIRNA_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {', '.join(missing)}; "
            f"expected {', '.join(SIRNA_TABLE_COLUMNS)}"
        )

    records: dict[str, SirnaRecord] = {}
    for sirna_id, on_target_id, raw in df.select(SIRNA_TABLE_COLUMNS).iter_rows():
        if sirna_id is None or not sirna_id.strip():
            raise ValueError(f"{path}: a row has an empty sirna_id")
        if sirna_id in records:
            raise ValueError(f"{path}: sirna_id '{sirna_id}' appears more than once")
        sequence = (raw or "").strip().upper().replace("U", "T")
        if not sequence:
            raise ValueError(f"{path}: siRNA '{sirna_id}' has an empty sequence")
        if not set(sequence) <= _NUCLEOTIDES:
            raise ValueError(
                f"{path}: siRNA '{sirna_id}' has a sequence outside ACGTU: {raw!r}"
            )
        records[sirna_id] = SirnaRecord(sirna_id, on_target_id or None, sequence)
    return records
