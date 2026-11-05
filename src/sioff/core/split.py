"""Split prediction inputs into one Parquet part per siRNA.

Every prediction input (a file, a folder of files or an in-memory frame) is
turned into per-siRNA Parquet parts, so each siRNA can be scored as its own
task. Parts are named by index (``0.parquet``, ``1.parquet``, ...), never by
``sirna_id``, because an ID may contain characters that are unsafe in a file
name.

[`split_file`][sioff.core.split.split_file] is a top-level function so it can
be pickled by reference and run in spawned worker processes.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from sioff.models import RISEARCH_COLUMNS
from sioff.services.risearch_parser import RIsearchParser


def _write_parts(df: pl.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, Path] = {}
    for i, part in enumerate(df.select(RISEARCH_COLUMNS).partition_by("sirna_id")):
        path = out_dir / f"{i}.parquet"
        part.write_parquet(path, compression="lz4")
        parts[part["sirna_id"][0]] = path
    return parts


def split_file(path: str, out_dir: str) -> dict[str, str]:
    """Split one predictions file into one Parquet part per siRNA.

    Parameters
    ----------
    path
        RIsearch2 output (plain or ``.gz``), ``sioff search`` TSV, or Parquet.
    out_dir
        Directory for the parts; created if missing.

    Returns
    -------
    dict[str, str]
        ``{sirna_id: part path}``.
    """
    df = RIsearchParser().load(Path(path))
    return {k: str(v) for k, v in _write_parts(df, Path(out_dir)).items()}


def split_frame(df: pl.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Split an in-memory predictions frame into one Parquet part per siRNA.

    Parameters
    ----------
    df
        Predictions already coerced to the RIsearch schema.
    out_dir
        Directory for the parts; created if missing.

    Returns
    -------
    dict[str, Path]
        ``{sirna_id: part path}``.
    """
    return _write_parts(df, out_dir)


def merge_splits(splits: Sequence[tuple[str, Mapping[str, str]]]) -> dict[str, Path]:
    """Merge the splits of several sources into one ``{sirna_id: part}`` map.

    Parameters
    ----------
    splits
        ``(source name, its split)`` pairs.

    Returns
    -------
    dict[str, Path]
        ``{sirna_id: part path}`` over all sources.

    Raises
    ------
    ValueError
        If one ``sirna_id`` occurs in two sources; the message names the siRNA
        and both sources.
    """
    merged: dict[str, Path] = {}
    origin: dict[str, str] = {}
    for source, split in splits:
        for sirna_id, part in split.items():
            if sirna_id in merged:
                raise ValueError(
                    f"siRNA {sirna_id!r} occurs in two inputs: "
                    f"{origin[sirna_id]!r} and {source!r}"
                )
            merged[sirna_id] = Path(part)
            origin[sirna_id] = source
    return merged
