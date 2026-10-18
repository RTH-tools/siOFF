"""With `-s`, E_min is the self-hybridisation energy or a stronger expressed site.

The legacy pipeline starts E_min at the siRNA's self-hybridisation energy and
lowers it to any expressed site that binds more strongly; with G:U wobbles a
genomic site can beat the perfect complement. siOFF used the self-hybridisation
energy alone, so for those siRNAs the alpha/gamma clamp threshold moved: in the
RIsearch3 off-target benchmark 11 of 2000 siRNAs differed from the legacy
pipeline by up to 0.069 in P_off_target (alpha = gamma = 0.8, no accessibility),
each with a hit 0.1-0.4 kcal/mol below its self-hybridisation E_min.
"""

from pathlib import Path

import polars as pl
import pytest

from sioff.core.off_targets import _apply_self_hyb_emin, compute_off_targets_directory
from sioff.services.risearch_service import RIsearchService


def _df(energies: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sirna_id": ["s1"] * len(energies),
            "energy": energies,
            "raw_e_min": [min(energies) - 5.0] * len(energies),
        },
        schema_overrides={"energy": pl.Float32, "raw_e_min": pl.Float32},
    )


def test_self_hyb_wins_when_no_site_binds_more_strongly():
    out = _apply_self_hyb_emin(_df([-20.0, -15.0]), {"s1": -26.46})
    assert out["raw_e_min"].unique().to_list() == [pytest.approx(-26.46)]


def test_a_stronger_site_lowers_e_min():
    out = _apply_self_hyb_emin(_df([-26.76, -15.0]), {"s1": -26.46})
    assert out["raw_e_min"].unique().to_list() == [pytest.approx(-26.76)]


def test_without_self_hyb_the_file_minimum_is_kept():
    df = _df([-20.0])
    assert _apply_self_hyb_emin(df, {}).equals(df)
    assert _apply_self_hyb_emin(df, {"other": -30.0}).equals(df)


def _gtf(path: Path) -> Path:
    path.write_text(
        'chr1\tt\texon\t100\t300\t.\t+\t.\tgene_id "g1"; transcript_id "t1"; TPM "4.0";\n'
        'chr1\tt\texon\t500\t700\t.\t+\t.\tgene_id "g2"; transcript_id "t2"; TPM "0";\n'
    )
    return path


def _hits(d: Path) -> Path:
    d.mkdir()
    # -22.5 on an expressed exon; -30.0 on an unexpressed one and -35.0 outside
    # any exon, neither of which the legacy pipeline reads.
    (d / "risearch_s1.out").write_text(
        "s1\t1\t19\tchr1\t160\t180\t+\t-22.5\n"
        "s1\t1\t19\tchr1\t200\t220\t+\t-12.0\n"
        "s1\t1\t19\tchr1\t600\t620\t+\t-30.0\n"
        "s1\t1\t19\tchr1\t900\t920\t+\t-35.0\n"
    )
    return d


def _z(monkeypatch, tmp_path: Path, hits: Path, gtf: Path, self_hyb: float) -> dict:
    monkeypatch.setattr(
        RIsearchService,
        "self_hybridization_emin_batch",
        lambda self, fasta_path, matrix="t04": {"s1": self_hyb},
    )
    fasta = tmp_path / "s.fa"
    fasta.write_text(">s1\nACGUACGUACGUACGUACG\n")
    (meta,) = [
        m
        for _, m in compute_off_targets_directory(
            input_dir=hits,
            sirna_fasta=fasta,
            gtf_file=gtf,
            expression_metric="TPM",
            alpha="0.8",
            gamma="0.9",
        )
    ]
    z = dict(meta["z_per_sirna"]["s1"])
    z.pop("sirna_id", None)
    return z


def test_directory_mode_takes_the_strongest_expressed_site(monkeypatch, tmp_path):
    hits, gtf = _hits(tmp_path / "hits"), _gtf(tmp_path / "a.gtf")
    weaker_self = _z(monkeypatch, tmp_path, hits, gtf, self_hyb=-20.0)
    at_site = _z(monkeypatch, tmp_path, hits, gtf, self_hyb=-22.5)
    unexpressed = _z(monkeypatch, tmp_path, hits, gtf, self_hyb=-30.0)
    # E_min = min(-20.0, -22.5) = -22.5, the strongest expressed site.
    assert weaker_self == pytest.approx(at_site, rel=1e-6)
    # The unexpressed and unannotated sites do not lower it.
    assert unexpressed != pytest.approx(at_site, rel=1e-6)
