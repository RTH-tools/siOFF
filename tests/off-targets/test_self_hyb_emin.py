"""E_min is the legacy rule: the perfect-match energy or a stronger remaining hit.

The legacy pipeline starts E_min at the siRNA's self-hybridisation (perfect
match) energy and lowers it to any expressed site that binds more strongly;
with G:U wobbles a genomic site can beat the perfect complement. Using the
perfect-match energy alone moved the alpha/gamma clamp threshold: in the
RIsearch3 off-target benchmark 11 of 2000 siRNAs differed from the legacy
pipeline by up to 0.069 in P_off_target (alpha = gamma = 0.8, no
accessibility), each with a hit 0.1-0.4 kcal/mol below its perfect match.

The engine takes the perfect-match energy from the siRNA table's sequence and
the second term from the hits left after --sense-only and the intersection.
"""

from pathlib import Path
from typing import Optional

import polars as pl
import pytest

from sioff.core.off_targets import compute_off_targets
from sioff.services.risearch_service import RIsearchService


def _gtf(path: Path) -> Path:
    path.write_text(
        'chr1\tt\texon\t100\t300\t.\t+\t.\tgene_id "g1"; transcript_id "t1"; TPM "4.0";\n'
        'chr1\tt\texon\t500\t700\t.\t+\t.\tgene_id "g2"; transcript_id "t2"; TPM "0";\n'
        'chr1\tt\texon\t100\t300\t.\t-\t.\tgene_id "g3"; transcript_id "t3"; TPM "4.0";\n'
    )
    return path


def _hits(path: Path, minus: bool) -> Path:
    # -22.5 and -12.0 on an expressed exon; -30.0 on an unexpressed one and
    # -35.0 outside any exon, neither of which the legacy pipeline reads.
    # With `minus`, -40.0 on an expressed minus-strand exon.
    path.write_text(
        "s1\t1\t19\tchr1\t160\t180\t+\t-22.5\n"
        "s1\t1\t19\tchr1\t200\t220\t+\t-12.0\n"
        "s1\t1\t19\tchr1\t600\t620\t+\t-30.0\n"
        "s1\t1\t19\tchr1\t900\t920\t+\t-35.0\n"
        + ("s1\t1\t19\tchr1\t160\t180\t-\t-40.0\n" if minus else "")
    )
    return path


def _e_min(
    monkeypatch,
    tmp_path: Path,
    perfect_match: Optional[float],
    *,
    transcriptome: bool = True,
    sense_only: bool = False,
    minus: bool = False,
) -> float:
    """E_min of s1, read from the detailed report.

    With a perfect-match energy the run computes an alpha/gamma pair and has a
    table; without one it is a baseline run without a table.
    """
    monkeypatch.setattr(
        RIsearchService,
        "self_hybridization_emin",
        lambda self, sequence, sirna_id="query", matrix="t04": perfect_match,
    )
    table = tmp_path / "sirnas.tsv"
    table.write_text("sirna_id\ton_target_id\tsequence\ns1\t\tACGUACGUACGUACGUACG\n")
    with_pm = perfect_match is not None
    frames = [
        df
        for df, _ in compute_off_targets(
            risearch_file=_hits(tmp_path / "hits.out", minus),
            gtf_file=_gtf(tmp_path / "a.gtf") if transcriptome else None,
            expression_metric="TPM",
            sirna_table=table if with_pm else None,
            alpha="0.8" if with_pm else "1.0",
            gamma="0.9" if with_pm else "1.0",
            sense_only=sense_only,
            detailed_report=True,
        )
    ]
    (e_min,) = pl.concat(frames)["E_min"].unique().to_list()
    return e_min


def test_perfect_match_below_every_hit_is_e_min(monkeypatch, tmp_path):
    # The unexpressed (-30.0) and unannotated (-35.0) hits do not lower it.
    assert _e_min(monkeypatch, tmp_path, -26.0) == pytest.approx(-26.0)


def test_a_hit_below_the_perfect_match_is_e_min(monkeypatch, tmp_path):
    assert _e_min(monkeypatch, tmp_path, -22.2) == pytest.approx(-22.5)


@pytest.mark.parametrize(
    "perfect_match, expected", [(-26.0, -26.0), (None, -22.5)], ids=["pm", "no-pm"]
)
def test_sense_only_ignores_a_stronger_minus_hit(
    monkeypatch, tmp_path, perfect_match, expected
):
    both = _e_min(monkeypatch, tmp_path, perfect_match, minus=True)
    sense = _e_min(monkeypatch, tmp_path, perfect_match, minus=True, sense_only=True)
    assert both == pytest.approx(-40.0), "the minus-strand hit counts without the flag"
    assert sense == pytest.approx(expected)


def test_without_transcriptome_every_hit_counts(monkeypatch, tmp_path):
    # No intersection: the hit outside every exon (-35.0) is the strongest.
    e_min = _e_min(monkeypatch, tmp_path, -26.0, transcriptome=False)
    assert e_min == pytest.approx(-35.0)
