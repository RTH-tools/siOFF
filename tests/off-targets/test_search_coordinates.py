"""``sioff search`` reports a hit where it lies: 1-based, inclusive.

risearch returns ``t_start``/``t_end`` and ``q_start``/``q_end`` counted from
0. Everything in siOFF that reads a hit (the opening-energy lookup, the
overlap with annotation features, a RIsearch2 output file) counts from 1, as
RIsearch2 does. siOFF used to pass risearch's numbers on unchanged, so every
hit of a search was scored on the stretch one position lower on the target,
and a hit on a target's first nucleotide (``start == 0``) fell outside the
profile and took the default opening energy.

No test stated where a hit should be: the search tests checked columns and
names, the lookup tests built 1-based hits by hand, and the tests that
compared routes compared one search with another. Here the sites are planted
at known positions and the expected coordinates are written out.
"""

import random
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("risearch")

import sioff  # noqa: E402

_COMPLEMENT = str.maketrans("ACGT", "TGCA")

N = 300
_RNG = random.Random(11)
TARGET = "".join(_RNG.choice("ACGT") for _ in range(N))

#: Planted sites as (first, last) nucleotide of the target, counted from 1.
#: ``mid`` is a site whose opening energy differs by kcal/mol from that of the
#: stretch one position away, on either strand.
SITES = {"first": (1, 21), "mid": (62, 82), "last": (280, 300)}
STRANDS = {"plus": "+", "minus": "-"}
CASES = [(strand, site) for strand in STRANDS for site in SITES]


def _window(start: int, end: int) -> str:
    """The target's nucleotides ``start..end``, counted from 1."""
    return TARGET[start - 1 : end]


def _guide(strand: str, site: str) -> str:
    """A 21-mer that pairs perfectly with the site on that strand."""
    window = _window(*SITES[site])
    # The forward window itself pairs with the minus strand.
    return window.translate(_COMPLEMENT)[::-1] if strand == "plus" else window


@pytest.fixture(scope="module")
def fastas(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("coordinates")
    target = root / "target.fa"
    target.write_text(f">tx1\n{TARGET}\n")
    query = root / "guides.fa"
    query.write_text(
        "".join(f">{strand}_{site}\n{_guide(strand, site)}\n" for strand, site in CASES)
    )
    return query, target


@pytest.fixture(scope="module")
def hits(fastas) -> pl.DataFrame:
    query, target = fastas
    return sioff.search(
        query, sioff.index(target), target=target, energy_threshold=-25, alignment=True
    )


@pytest.fixture(scope="module")
def profile(fastas) -> pl.DataFrame:
    """Opening energies of tx1: one row per ``position`` (from 1) and strand."""
    _, target = fastas
    return sioff.accessibility(target, strand="both")["tx1"]


def _planted(frame: pl.DataFrame, strand: str, site: str) -> pl.DataFrame:
    """The guide's hit on the strand it was designed against."""
    return frame.filter(
        (pl.col("sirna_id") == f"{strand}_{site}")
        & (pl.col("strand") == STRANDS[strand])
    )


def _opening_energy(profile: pl.DataFrame, strand: str, start: int, end: int) -> float:
    """Energy to open ``start..end`` on that strand, read from the profile.

    ``u<k>`` at ``position`` is the energy to open the ``k`` bases ending
    there on the molecule that was folded: the target for ``+``, its reverse
    complement for ``-``, where target position ``c`` is position
    ``N - c + 1``.
    """
    ends_at = end if strand == "plus" else N - start + 1
    row = profile.filter(
        (pl.col("strand") == STRANDS[strand]) & (pl.col("position") == ends_at)
    )
    return row[f"u{end - start + 1}"][0]


@pytest.mark.parametrize(("strand", "site"), CASES)
def test_target_coordinates_count_from_one(hits, strand, site):
    assert _planted(hits, strand, site).select("chrom", "start", "end").rows() == [
        ("tx1", *SITES[site])
    ]


@pytest.mark.parametrize(("strand", "site"), CASES)
def test_query_coordinates_count_from_one(hits, strand, site):
    """A perfect 21-mer pairs over its nucleotides 1 to 21."""
    assert _planted(hits, strand, site).select("q_start", "q_end").rows() == [(1, 21)]


@pytest.mark.parametrize("strand", STRANDS)
def test_the_mid_site_tells_a_shifted_lookup_apart(profile, strand):
    """Guard for the scoring tests below: without it they could not fail."""
    start, end = SITES["mid"]
    own = _opening_energy(profile, strand, start, end)
    one_lower = _opening_energy(profile, strand, start - 1, end - 1)
    assert abs(own - one_lower) > 1.0


def _scored_opening_energy(scored: pl.DataFrame, strand: str, site: str) -> float:
    planted = _planted(scored, strand, site).sort("energy").head(1)
    return planted["opening_energy"][0]


@pytest.mark.parametrize(("strand", "site"), CASES)
def test_search_hits_are_scored_on_their_own_stretch(hits, profile, strand, site):
    scored = sioff.off_targets(
        predictions=hits.drop("q_start", "q_end", "alignment"),
        accessibility={"tx1": profile},
    )

    # Opening energies are reported to 0.1 kcal/mol.
    assert _scored_opening_energy(scored, strand, site) == pytest.approx(
        _opening_energy(profile, strand, *SITES[site]), abs=0.051
    )


@pytest.mark.parametrize(("strand", "site"), CASES)
def test_in_process_search_scores_hits_on_their_own_stretch(
    fastas, profile, strand, site
):
    query, target = fastas
    scored = sioff.off_targets(
        sirna_fasta=query, target_fasta=target, accessibility={"tx1": profile}
    )

    assert _scored_opening_energy(scored, strand, site) == pytest.approx(
        _opening_energy(profile, strand, *SITES[site]), abs=0.051
    )
