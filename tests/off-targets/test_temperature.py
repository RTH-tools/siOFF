"""What ``temperature`` of ``sioff.off_targets`` reaches.

Every energy the run computes itself is taken at ``temperature``: the RT of
the partition function, the ``genome_file`` fold, the in-process search of
``sirna_fasta`` and the perfect-match energy of E_min. The last two used to
run at risearch's default 37 °C whatever ``temperature`` said, so
``temperature=28`` weighed 37 °C duplex energies with a 28 °C RT.

Inputs computed earlier keep the temperature they were computed at:
``accessibility_dir`` profiles and ``risearch_file`` / ``predictions``.

Every fixture is hand-written and lives in ``tmp_path``; the index the search
writes next to its target FASTA lands there too.
"""

from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("risearch")

import sioff  # noqa: E402
from sioff.services.accessibility import GenomeAccessibilityService  # noqa: E402

# A 21-nt guide strand (5'->3') and its perfect target site on the + strand.
GUIDE = "UAUCGGACAUUGCGAACUCAA"
SITE = GUIDE.translate(str.maketrans("ACGU", "TGCA"))[::-1]  # TTGAGTTCGCAATGTCCGATA
FLANK_A = "ACTCTACATTCACTATCATACAATCTTACA"
FLANK_B = "TCACATTACTCTAACTTATCATCTACATAC"
FLANK_C = "CATTCATACACTATTCTACTTACATCACTA"
# One perfect site and one 12-nt partial site.
GENOME_WITH_PERFECT_SITE = FLANK_A + SITE + FLANK_B + SITE[-12:] + FLANK_C
# Partial sites only (12 and 10 nt): no hit is as strong as the perfect match.
GENOME_WITHOUT_PERFECT_SITE = FLANK_A + SITE[-12:] + FLANK_B + SITE[-10:] + FLANK_C

SITE_KEY = ["sirna_id", "chrom", "start", "end", "strand"]


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _inputs(tmp_path: Path, genome: str) -> tuple[Path, Path, Path]:
    """Query FASTA, target FASTA and siRNA table (no on-target) in ``tmp_path``."""
    return (
        _write(tmp_path / "sirnas.fa", f">s1\n{GUIDE}\n"),
        _write(tmp_path / "genome.fa", f">chrT\n{genome}\n"),
        _write(
            tmp_path / "sirnas.tsv",
            f"sirna_id\ton_target_id\tsequence\ns1\t\t{GUIDE}\n",
        ),
    )


def _sites(frame: pl.DataFrame) -> pl.DataFrame:
    """The hits and their duplex energies (4 decimals), in a fixed order."""
    return frame.select(*SITE_KEY, pl.col("energy").round(4)).sort(SITE_KEY)


def _perfect_match(tmp_path: Path, temperature: int) -> float:
    """The guide against itself at ``temperature``: the recipe of
    ``RIsearchService.self_hybridization_emin`` (seed = length - 1, threshold 0)."""
    fasta = _write(
        tmp_path / f"self_{temperature}.fa", f">s1\n{GUIDE.replace('U', 'T')}\n"
    )
    hits = sioff.search(
        fasta,
        sioff.index(fasta),
        target=fasta,
        seed_length=len(GUIDE) - 1,
        energy_threshold=0.0,
        temperature=temperature,
    )
    return float(hits["energy"].min())


def test_in_process_search_runs_at_the_requested_temperature(tmp_path):
    sirnas, genome, _ = _inputs(tmp_path, GENOME_WITH_PERFECT_SITE)
    index = sioff.index(genome)
    at_37 = _sites(sioff.search(sirnas, index, target=genome, temperature=37))
    at_28 = _sites(sioff.search(sirnas, index, target=genome, temperature=28))
    # Control: on this fixture the search temperature changes the hits.
    assert not at_28.equals(at_37)

    scored = _sites(
        sioff.off_targets(sirna_fasta=sirnas, target_fasta=genome, temperature=28.0)
    )

    assert scored.equals(at_28), (
        "off_targets(sirna_fasta=..., temperature=28.0) did not score the hits of "
        f"a 28 degree search (it scored the 37 degree hits: {scored.equals(at_37)}).\n"
        f"scored:\n{scored}\nsearch at 28:\n{at_28}\nsearch at 37:\n{at_37}"
    )


def test_perfect_match_energy_is_computed_at_the_requested_temperature(tmp_path):
    sirnas, genome, table = _inputs(tmp_path, GENOME_WITHOUT_PERFECT_SITE)
    # The predictions are searched at 28 degrees, as a user who sets every
    # temperature by hand would do: only the perfect-match energy is left.
    hits_28 = sioff.search(sirnas, sioff.index(genome), target=genome, temperature=28)
    pm_37, pm_28 = _perfect_match(tmp_path, 37), _perfect_match(tmp_path, 28)
    # Control: the perfect match depends on the temperature and beats every
    # hit, so E_min = min(perfect match, strongest hit) is the perfect match.
    assert pm_28 < pm_37 < hits_28["energy"].min()

    report = sioff.off_targets(
        predictions=hits_28, sirna_table=table, temperature=28.0, detailed_report=True
    )
    (e_min,) = report["E_min"].unique().to_list()

    assert e_min == pytest.approx(pm_28, abs=1e-3), (
        f"E_min of a temperature=28.0 run is {e_min:.4f}: the perfect match at "
        f"28 degrees is {pm_28:.4f}, at 37 degrees {pm_37:.4f}"
    )


def test_precomputed_profiles_keep_the_temperature_they_were_folded_at(tmp_path):
    """Pins the current behaviour, which is by design: ``accessibility_dir``
    profiles are read as folded; only ``genome_file`` is folded at ``temperature``."""
    sirnas, genome, _ = _inputs(tmp_path, GENOME_WITH_PERFECT_SITE)
    hits = sioff.search(sirnas, sioff.index(genome), target=genome, temperature=28)
    folded = {}
    for degrees in (37.0, 28.0):
        folded[degrees] = tmp_path / f"acc_{degrees:.0f}"
        GenomeAccessibilityService(folded[degrees]).compute_genome_accessibility(
            genome, temperature=degrees, output_format="npy"
        )

    def opening(temperature: float, **source) -> list[float]:
        frame = sioff.off_targets(predictions=hits, temperature=temperature, **source)
        return frame.sort(SITE_KEY)["opening_energy"].to_list()

    from_37_profiles = opening(28.0, accessibility_dir=folded[37.0])
    from_28_profiles = opening(28.0, accessibility_dir=folded[28.0])
    # Control: the two folds give different opening energies at these hits.
    assert from_37_profiles != from_28_profiles
    # -a: the profile decides, whatever temperature the run is given.
    assert from_37_profiles == opening(37.0, accessibility_dir=folded[37.0])
    assert from_28_profiles == opening(37.0, accessibility_dir=folded[28.0])
    # -f: the only source that is folded at the run's temperature.
    assert opening(28.0, genome_file=genome) == from_28_profiles
    assert opening(37.0, genome_file=genome) == from_37_profiles


@pytest.mark.parametrize("temperature", [28.5, 60.0, -1.0])
def test_in_process_search_refuses_a_temperature_risearch_cannot_take(
    tmp_path, temperature
):
    """RIsearch takes whole degrees from 0 to 50: no silent rounding."""
    sirnas, genome, _ = _inputs(tmp_path, GENOME_WITH_PERFECT_SITE)
    with pytest.raises(ValueError, match="whole number of degrees from 0 to 50"):
        sioff.off_targets(
            sirna_fasta=sirnas, target_fasta=genome, temperature=temperature
        )


def test_perfect_match_energy_refuses_a_fractional_temperature(tmp_path):
    sirnas, genome, table = _inputs(tmp_path, GENOME_WITHOUT_PERFECT_SITE)
    hits = sioff.search(sirnas, sioff.index(genome), target=genome, temperature=28)
    with pytest.raises(ValueError, match="whole number of degrees from 0 to 50"):
        sioff.off_targets(
            predictions=hits, sirna_table=table, temperature=28.5, alpha="0.8"
        )


def test_fractional_temperature_is_fine_when_risearch_computes_nothing(tmp_path):
    """Predictions given, no E_min: only RT uses the temperature."""
    sirnas, genome, table = _inputs(tmp_path, GENOME_WITH_PERFECT_SITE)
    hits = sioff.search(sirnas, sioff.index(genome), target=genome, temperature=28)
    scored = sioff.off_targets(predictions=hits, sirna_table=table, temperature=28.5)
    assert scored.height == hits.height
