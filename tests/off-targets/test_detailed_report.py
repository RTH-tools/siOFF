"""`--detailed-report` keeps the raw weights; the default output drops them."""

import shutil
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

import sioff
from sioff.cli import app

runner = CliRunner()

EX = Path(__file__).parents[2] / "examples" / "data"
SUFFIX = ":alpha=0.8,gamma=0.8"


def _sirna_table(tmp_path: Path) -> Path:
    """The example siRNAs, no on-targets: alpha/gamma needs their sequences."""
    lines = (EX / "sirnas.fa").read_text().split()
    table = tmp_path / "sirna_table.tsv"
    table.write_text(
        "sirna_id\ton_target_id\tsequence\n"
        + "".join(
            f"{lines[i][1:]}\t\t{lines[i + 1]}\n" for i in range(0, len(lines), 2)
        )
    )
    return table


def _run(form: str, tmp_path: Path, detailed_report: bool) -> pl.DataFrame:
    if form == "file":
        source = EX / "predictions.out"
    else:
        source = tmp_path / "d"
        source.mkdir()
        shutil.copy(EX / "predictions.out", source / "predictions.out")
    return sioff.off_targets(
        risearch_file=source,
        gtf_file=EX / "annotation.gtf",
        accessibility_dir=EX / "accessibility",
        sirna_table=_sirna_table(tmp_path),
        alpha="0.8",
        gamma="0.8",
        detailed_report=detailed_report,
    )


@pytest.mark.parametrize("form", ["file", "directory"])
def test_detailed_report_keeps_weights_and_z(tmp_path, form):
    df = _run(form, tmp_path, detailed_report=True)
    for sfx in ("", SUFFIX):
        for col in (
            f"boltzmann_weight{sfx}",
            f"boltzmann_weight_noacc{sfx}",
            f"Z_sirna{sfx}",
            f"Z_sirna_noacc{sfx}",
        ):
            assert col in df.columns
    assert "E_min" in df.columns
    sums = df.group_by("sirna_id").agg(
        pl.col("boltzmann_weight").sum(), pl.col("Z_sirna").first()
    )
    assert sums["boltzmann_weight"].to_list() == pytest.approx(
        sums["Z_sirna"].to_list(), rel=1e-5
    )


@pytest.mark.parametrize("form", ["file", "directory"])
def test_default_output_drops_weights(tmp_path, form):
    df = _run(form, tmp_path, detailed_report=False)
    assert not [c for c in df.columns if c.startswith(("boltzmann_weight", "Z_sirna"))]
    assert "E_min" not in df.columns


def test_detailed_report_with_summary_only_is_an_error(tmp_path, plain):
    result = runner.invoke(
        app,
        [
            "off-targets",
            "-r",
            str(EX / "predictions.out"),
            "--detailed-report",
            "--summary-only",
            "-o",
            str(tmp_path / "o.tsv"),
        ],
    )
    assert result.exit_code == 1
    assert "--detailed-report needs the per-interaction table" in plain(result.stdout)
