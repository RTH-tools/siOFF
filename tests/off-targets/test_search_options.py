"""Every `risearch.search` option reaches risearch through siOFF.

siOFF passes each option explicitly, with its own default (see
test_penalty_pinned.py for how an inherited default once moved our numbers),
and every layer — `RIsearchService.run_search`, `sioff.core.risearch.run_search`,
`sioff.search` and the `sioff search` CLI — forwards what the caller sets.
"""

import inspect
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
    RIsearchError,
    RIsearchService,
)

DATA = Path(__file__).parent / "data"
QUERY = DATA / "sirnas.fa"
GENOME = DATA / "genome.fa"
BASE_COLUMNS = ["sirna_id", "chrom", "start", "end", "strand", "energy"]

# siOFF's defaults for the options it did not pass before; they reproduce the
# searches siOFF ran then (risearch's own defaults, except `alignment`, which
# does not change the hits).
SIOFF_DEFAULTS = {
    "mismatches": 0,
    "mismatch_prefix": 1,
    "mismatch_suffix": 0,
    "temperature": 37,
    "penalty": 0.0,
    "seed_energy": 0.0,
    "no_max_prune": False,
    "no_dedup": False,
    "alignment": False,
    "threads": None,
}

# A non-default value for each, and the same as `sioff search` flags.
CHANGED = {
    "mismatches": 1,
    "mismatch_prefix": 2,
    "mismatch_suffix": 1,
    "temperature": 25,
    "penalty": 0.3,
    "seed_energy": -1.0,
    "no_max_prune": True,
    "no_dedup": True,
    "alignment": True,
    "threads": 2,
}
CHANGED_FLAGS = [
    "--mismatches=1",
    "--mismatch-prefix=2",
    "--mismatch-suffix=1",
    "--temperature=25",
    "--penalty=0.3",
    "--seed-energy=-1.0",
    "--no-max-prune",
    "--no-dedup",
    "--alignment",
    "--threads=2",
]


@pytest.fixture(scope="module")
def index(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("idx") / "genome.idx"
    RIsearchService().index_target(GENOME, path)
    return path


@pytest.fixture
def seen(monkeypatch) -> dict:
    """Spy on risearch.search; the dict fills with the kwargs of the last call."""
    captured: dict = {}
    real_search = risearch.search

    def spy(*args, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return real_search(*args, **kwargs)

    monkeypatch.setattr(risearch, "search", spy)
    return captured


def _service(index: Path, **kw) -> pl.DataFrame:
    return RIsearchService().run_search(QUERY, index, target_fasta=GENOME, **kw)


def _core(index: Path, **kw) -> pl.DataFrame:
    return core.run_search(QUERY, index, target=GENOME, **kw)


def _api(index: Path, **kw) -> pl.DataFrame:
    return sioff.search(QUERY, index, target=GENOME, **kw)


LAYERS = [_service, _core, _api]


def test_every_risearch_option_is_passed(index, seen):
    """A new upstream option fails this test until siOFF passes it too."""
    _service(index)
    options = {
        name
        for name, p in inspect.signature(risearch.search).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert options - set(seen) == set()


def test_sioff_defaults_are_pinned(index, seen):
    _service(index)
    assert {k: seen[k] for k in SIOFF_DEFAULTS} == SIOFF_DEFAULTS


@pytest.mark.parametrize("layer", LAYERS, ids=lambda f: f.__name__.strip("_"))
def test_layer_forwards_every_option(layer, index, seen):
    layer(index, **CHANGED)
    assert {k: seen[k] for k in CHANGED} == CHANGED


def test_cli_forwards_every_option(index, seen, tmp_path):
    out = tmp_path / "hits.tsv"
    result = CliRunner().invoke(
        app,
        ["search", str(QUERY), str(index), "-t", str(GENOME), "-o", str(out)]
        + CHANGED_FLAGS,
    )
    assert result.exit_code == 0, result.output
    assert {k: seen[k] for k in CHANGED} == CHANGED
    assert pl.read_csv(out, separator="\t").columns == BASE_COLUMNS + list(
        ALIGNMENT_COLUMNS
    )


def test_default_schema_is_unchanged(index):
    assert _api(index, energy_threshold=0.0).columns == BASE_COLUMNS


def test_alignment_adds_columns_and_keeps_the_hits(index):
    plain = _api(index, energy_threshold=0.0)
    aligned = _api(index, energy_threshold=0.0, alignment=True)
    assert aligned.columns == BASE_COLUMNS + list(ALIGNMENT_COLUMNS)
    assert aligned["alignment"].null_count() == 0
    assert (
        aligned.select(BASE_COLUMNS).sort(BASE_COLUMNS).equals(plain.sort(BASE_COLUMNS))
    )


def test_empty_result_with_alignment_has_the_columns(index):
    empty = _api(index, energy_threshold=-1000.0, alignment=True)
    assert empty.is_empty()
    assert empty.columns == BASE_COLUMNS + list(ALIGNMENT_COLUMNS)


def test_options_reach_the_engine(index):
    """Not only passed: they change the result as RIsearch documents."""
    perfect = _api(index, energy_threshold=0.0)
    assert _api(index, energy_threshold=0.0, mismatches=1).height > perfect.height
    assert (
        _api(index, energy_threshold=0.0, temperature=25)["energy"].min()
        < perfect["energy"].min()
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"mismatches": -1},
        {"mismatch_prefix": -1},
        {"mismatch_suffix": -1},
        {"temperature": 51},
        {"temperature": -1},
        {"threads": 0},
    ],
    ids=lambda d: next(iter(d.items())).__repr__(),
)
def test_bad_values_fail_before_the_search(bad, index, seen):
    with pytest.raises(RIsearchError):
        _api(index, **bad)
    assert seen == {}, "risearch.search was called"


def test_cli_rejects_a_bad_temperature(index):
    result = CliRunner().invoke(
        app,
        ["search", str(QUERY), str(index), "-t", str(GENOME), "--temperature=60"],
    )
    assert result.exit_code == 1
    assert "temperature" in result.output
