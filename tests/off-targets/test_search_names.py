"""``sioff search`` names each hit's target and query as RIsearch itself does.

RIsearch returns a hit's target as an index into the records it kept, and it
drops a record that is empty after normalisation (all gaps). siOFF used to
look that index up in its own parse of the FASTA, which keeps such a record,
so every hit on a later record got the name of an earlier one. The names now
come from RIsearch's own ``target_name``/``query_name``, cut at the first
whitespace as everywhere else in siOFF.
"""

import random
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("risearch")

import sioff  # noqa: E402
from sioff.services.risearch_service import RIsearchService  # noqa: E402

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _sequence(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


REC_A = _sequence(200, 1)
REC_C = _sequence(200, 2)
#: A perfect guide against positions 61-81 of recC.
GUIDE = REC_C[60:81].translate(_COMPLEMENT)[::-1]


def _inputs(tmp_path: Path, gap_record: bool) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "t.fa"
    target.write_text(
        f">recA first record\n{REC_A}\n"
        + (f">gaponly an all-gap record\n{'-' * 30}\n" if gap_record else "")
        + f">recC third record\n{REC_C}\n"
    )
    query = tmp_path / "q.fa"
    query.write_text(f">siC guide against recC\n{GUIDE}\n")
    return query, target


def _search(query: Path, target: Path) -> pl.DataFrame:
    return sioff.search(
        query, sioff.index(target), target=target, energy_threshold=-25
    ).filter(pl.col("strand") == "+")


def test_hit_after_a_skipped_record_keeps_its_own_name(tmp_path):
    """It was named ``gaponly``: the record RIsearch had dropped."""
    control = _search(*_inputs(tmp_path / "control", gap_record=False))
    assert control.select("sirna_id", "chrom").rows() == [("siC", "recC")]

    hits = _search(*_inputs(tmp_path / "gap", gap_record=True))

    assert hits.select("sirna_id", "chrom").rows() == [("siC", "recC")]
    assert hits.select("start", "end", "energy").equals(
        control.select("start", "end", "energy")
    )


def test_names_are_the_first_word_of_the_header(tmp_path):
    hits = _search(*_inputs(tmp_path, gap_record=False))
    assert hits["sirna_id"].unique().to_list() == ["siC"]
    assert hits["chrom"].unique().to_list() == ["recC"]


def test_names_fall_back_to_the_fasta_order_without_name_columns(tmp_path, monkeypatch):
    """A risearch whose hits carry only the indices: the previous lookup."""
    import risearch

    query, target = _inputs(tmp_path, gap_record=False)
    index = sioff.index(target)
    real = risearch.search
    monkeypatch.setattr(
        risearch,
        "search",
        lambda *a, **k: real(*a, **k).drop("query_name", "target_name"),
    )
    hits = RIsearchService().run_search(
        query, Path(index), target_fasta=target, energy_threshold=-25
    )
    assert hits.filter(pl.col("strand") == "+").select("sirna_id", "chrom").rows() == [
        ("siC", "recC")
    ]
