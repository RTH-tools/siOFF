"""siOFF must pin `penalty` explicitly rather than inherit upstream's default.

Inheriting it is exactly how a pin bump silently changed our numbers once:
`penalty` defaulted to 3.5 in risearch 0.0.0a1 and 0.0 in 3.0.0a2, and siOFF
never passed it, so bumping the pin moved E_min from -40.2956 to -33.2956
without a line of siOFF changing.

Pinning it means a future upstream default change is a no-op for us, and any
deliberate change is a visible diff in this repo. See RTH-tools/risearch#27 for
why the parameter is not comparable across versions anyway (units, sign and
effect on the reported energy all differ).
"""

from pathlib import Path

import pytest

pytest.importorskip("risearch")

import risearch  # noqa: E402

from sioff.services.risearch_service import RIsearchService  # noqa: E402

DATA = Path(__file__).parent / "data"


@pytest.fixture
def captured_kwargs(monkeypatch, tmp_path):
    """Run a search with risearch.search spied on, and return its kwargs."""
    seen: dict = {}
    real_search = risearch.search

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_search(*args, **kwargs)

    monkeypatch.setattr(risearch, "search", spy)

    index = tmp_path / "g.idx"
    service = RIsearchService()
    service.index_target(DATA / "genome.fa", index)
    service.run_search(
        query_path=DATA / "sirnas.fa",
        index_path=index,
        target_fasta=DATA / "genome.fa",
    )
    return seen


def test_penalty_is_passed_explicitly(captured_kwargs):
    """The whole point: never let upstream's default decide our energies."""
    assert "penalty" in captured_kwargs, (
        "risearch.search was called without `penalty` — siOFF would inherit "
        "whatever upstream defaults to, which has already changed once (3.5 -> 0.0)"
    )


def test_pinned_penalty_is_zero(captured_kwargs):
    """0.0 matches the C reference default (`extPen = 0`, main.c:76).

    Changing this changes published off-target energies, so it should be a
    deliberate, reviewed edit rather than a silent inheritance.
    """
    assert captured_kwargs["penalty"] == 0.0


def test_the_other_scoring_knobs_are_pinned_too(captured_kwargs):
    """Same reasoning as penalty: `seed_wobble` also flipped upstream.

    Its default went True -> False between the two pins. Leaving any scoring
    input implicit reintroduces the failure mode.
    """
    for knob in (
        "seed_length",
        "max_extension",
        "energy_threshold",
        "seed_wobble",
        "matrix",
    ):
        assert knob in captured_kwargs, f"{knob} must be passed explicitly"
