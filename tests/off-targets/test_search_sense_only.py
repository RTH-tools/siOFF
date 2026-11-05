"""`sense_only` keeps a search's `+`-strand hits and nothing else.

It is siOFF's own option, not risearch's: risearch indexes both strands and
has no option to search one, so siOFF drops the `-` hits after the search.
"""

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

pytest.importorskip("risearch")

import risearch  # noqa: E402

import sioff  # noqa: E402
from sioff.cli import app  # noqa: E402
from sioff.core import risearch as core  # noqa: E402
from sioff.services.risearch_service import (  # noqa: E402
    ALIGNMENT_COLUMNS,
    RIsearchService,
)

DATA = Path(__file__).parent / "data"
QUERY = DATA / "sirnas.fa"
GENOME = DATA / "genome.fa"
BASE_COLUMNS = ["sirna_id", "chrom", "start", "end", "strand", "energy"]


@pytest.fixture(scope="module")
def index(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("idx") / "genome.idx"
    RIsearchService().index_target(GENOME, path)
    return path


def _service(index: Path, **kw) -> pl.DataFrame:
    return RIsearchService().run_search(QUERY, index, target_fasta=GENOME, **kw)


def _core(index: Path, **kw) -> pl.DataFrame:
    return core.run_search(QUERY, index, target=GENOME, **kw)


def _api(index: Path, **kw) -> pl.DataFrame:
    return sioff.search(QUERY, index, target=GENOME, **kw)


LAYERS = [_service, _core, _api]


def test_fixture_has_hits_on_both_strands(index):
    """Otherwise the tests below would pass without filtering anything."""
    plain = _api(index, energy_threshold=0.0)
    assert set(plain["strand"].unique()) == {"+", "-"}


@pytest.mark.parametrize("layer", LAYERS, ids=lambda f: f.__name__.strip("_"))
def test_sense_only_keeps_exactly_the_plus_hits(layer, index):
    plain = _api(index, energy_threshold=0.0)
    sense = layer(index, energy_threshold=0.0, sense_only=True)
    assert sense.columns == BASE_COLUMNS
    assert sense.schema == plain.schema
    assert sense.sort(BASE_COLUMNS).equals(
        plain.filter(pl.col("strand") == "+").sort(BASE_COLUMNS)
    )


def test_sense_only_keeps_the_alignment_columns(index):
    plain = _api(index, energy_threshold=0.0, alignment=True)
    sense = _api(index, energy_threshold=0.0, alignment=True, sense_only=True)
    columns = BASE_COLUMNS + list(ALIGNMENT_COLUMNS)
    assert sense.columns == columns
    assert sense.sort(columns).equals(
        plain.filter(pl.col("strand") == "+").sort(columns)
    )


def test_sense_only_is_not_passed_to_risearch(index, monkeypatch):
    captured: dict = {}
    real_search = risearch.search

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_search(*args, **kwargs)

    monkeypatch.setattr(risearch, "search", spy)
    _api(index, energy_threshold=0.0, sense_only=True)
    assert captured, "risearch.search was not called"
    assert "sense_only" not in captured


@pytest.mark.parametrize("alignment", [False, True])
def test_empty_sense_only_result_has_the_columns(index, alignment):
    empty = _api(index, energy_threshold=-1000.0, alignment=alignment, sense_only=True)
    assert empty.is_empty()
    assert empty.columns == BASE_COLUMNS + (
        list(ALIGNMENT_COLUMNS) if alignment else []
    )


def test_cli_sense_only(index, tmp_path):
    out = tmp_path / "hits.tsv"
    result = CliRunner().invoke(
        app,
        [
            "search",
            str(QUERY),
            str(index),
            "-t",
            str(GENOME),
            "-E",
            "0",
            "-o",
            str(out),
            "--sense-only",
        ],
    )
    assert result.exit_code == 0, result.output
    hits = pl.read_csv(out, separator="\t")
    assert hits.height > 0
    assert set(hits["strand"].unique()) == {"+"}
