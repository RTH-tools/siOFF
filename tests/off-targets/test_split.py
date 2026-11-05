"""Tests for splitting prediction inputs into per-siRNA Parquet parts."""

import gzip
import shutil
from pathlib import Path

import polars as pl
import pytest

from sioff.core.split import merge_splits, split_file, split_frame
from sioff.models import RISEARCH_COLUMNS
from sioff.services.risearch_parser import RIsearchParser

DATA = Path(__file__).parents[2] / "examples" / "data"
PREDICTIONS = DATA / "predictions.out"


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(RISEARCH_COLUMNS).sort(RISEARCH_COLUMNS)


def _concat(parts: dict) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in parts.values()])


def test_load_reads_parquet_and_drops_raw_e_min(tmp_path):
    loaded = RIsearchParser().load(PREDICTIONS)
    path = tmp_path / "x.parquet"
    loaded.with_columns(raw_e_min=pl.lit(-1.0)).write_parquet(path)
    out = RIsearchParser().load(path)
    assert out.columns == RISEARCH_COLUMNS
    assert out.equals(loaded)


def test_split_file_writes_one_part_per_sirna(tmp_path):
    parts = split_file(str(PREDICTIONS), str(tmp_path / "o"))
    assert sorted(parts) == [f"siRNA_{i}" for i in range(1, 6)]
    assert all(isinstance(p, str) and p.endswith(".parquet") for p in parts.values())
    assert sorted(Path(p).name for p in parts.values()) == [
        f"{i}.parquet" for i in range(5)
    ]
    for sid, p in parts.items():
        assert pl.read_parquet(p)["sirna_id"].unique().to_list() == [sid]
    expected = _sorted(RIsearchParser().load(PREDICTIONS))
    assert _sorted(_concat(parts)).equals(expected)


def test_split_file_reads_gz_and_search_tsv(tmp_path):
    loaded = RIsearchParser().load(PREDICTIONS)
    gz = tmp_path / "p.out.gz"
    with open(PREDICTIONS, "rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    tsv = tmp_path / "p.tsv"
    loaded.write_csv(tsv, separator="\t")
    for i, f in enumerate([gz, tsv]):
        parts = split_file(str(f), str(tmp_path / f"o{i}"))
        assert len(parts) == 5
        assert _sorted(_concat(parts)).equals(_sorted(loaded))


def test_split_frame_matches_split_file(tmp_path):
    loaded = RIsearchParser().load(PREDICTIONS)
    from_frame = split_frame(loaded, tmp_path / "f")
    from_file = split_file(str(PREDICTIONS), str(tmp_path / "g"))
    assert all(isinstance(p, Path) for p in from_frame.values())
    assert sorted(from_frame) == sorted(from_file)
    for sid in from_frame:
        assert _sorted(pl.read_parquet(from_frame[sid])).equals(
            _sorted(pl.read_parquet(from_file[sid]))
        )


def test_merge_splits_rejects_a_sirna_in_two_sources():
    with pytest.raises(ValueError, match=r"siRNA_1.*a\.out\.gz.*a\.parquet"):
        merge_splits(
            [
                ("a.out.gz", {"siRNA_1": "x/0.parquet"}),
                ("a.parquet", {"siRNA_1": "y/0.parquet"}),
            ]
        )


def test_merge_splits_merges_disjoint_sources():
    merged = merge_splits([("a", {"s1": "x/0.parquet"}), ("b", {"s2": "y/0.parquet"})])
    assert merged == {"s1": Path("x/0.parquet"), "s2": Path("y/0.parquet")}


def test_odd_sirna_ids_are_safe(tmp_path):
    df = RIsearchParser().load(PREDICTIONS).with_columns(sirna_id=pl.lit("a/b"))
    parts = split_frame(df, tmp_path / "o")
    assert list(parts) == ["a/b"]
    assert parts["a/b"].parent == tmp_path / "o"
    assert pl.read_parquet(parts["a/b"])["sirna_id"].unique().to_list() == ["a/b"]
