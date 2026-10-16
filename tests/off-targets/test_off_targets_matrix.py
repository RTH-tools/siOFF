"""`off-targets -z/--matrix` reaches every RIsearch call the command makes.

E_min, the reference energy of the alpha/gamma clamp, is each siRNA's
self-hybridisation minimum (`-s/--sirna-fasta`). It used to be computed with
the bundled `t04` table whatever the predictions had been searched with, so
predictions from RIsearch2 (whose `t04` differs slightly from RIsearch3's)
were clamped against a shifted threshold: in the RIsearch3 off-target
benchmark every one of 298 checked siRNAs had a self-hybridisation E_min
0.06-0.13 kcal/mol away from the legacy pipeline's, and with RIsearch2's
`t04` exported to a DSM TSV all 298 matched.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sioff.cli import app
from sioff.config import config_to_kwargs, load_config
from sioff.services.risearch_service import RIsearchService

DATA = Path(__file__).parent / "data"
runner = CliRunner()


def _first_sequence() -> tuple[str, str]:
    lines = (DATA / "sirnas.fa").read_text().split()
    return lines[0].lstrip(">"), lines[1]


class TestSelfHybridisationMatrix:
    def test_matrix_changes_self_hybridisation_emin(self):
        sid, seq = _first_sequence()
        svc = RIsearchService()
        t04 = svc.self_hybridization_emin(seq, sid)
        slh04 = svc.self_hybridization_emin(seq, sid, matrix="slh04")
        assert t04 < 0 and slh04 < 0
        assert t04 != pytest.approx(slh04)

    def test_batch_passes_the_matrix_to_every_siRNA(self, tmp_path):
        sid, seq = _first_sequence()
        fa = tmp_path / "one.fa"
        fa.write_text(f">{sid}\n{seq}\n")
        svc = RIsearchService()
        batch = svc.self_hybridization_emin_batch(fa, matrix="slh04")
        assert batch == {
            sid: pytest.approx(svc.self_hybridization_emin(seq, sid, matrix="slh04"))
        }

    def test_unknown_matrix_is_rejected(self):
        sid, seq = _first_sequence()
        with pytest.raises(Exception, match="matrix"):
            RIsearchService().self_hybridization_emin(seq, sid, matrix="t99")


class TestPlumbing:
    def test_directory_mode_uses_the_matrix_for_emin(self, monkeypatch, tmp_path):
        seen = {}

        def fake_batch(self, fasta_path, matrix="t04"):
            seen["matrix"] = matrix
            raise RuntimeError("stop after E_min")

        monkeypatch.setattr(
            RIsearchService, "self_hybridization_emin_batch", fake_batch
        )
        from sioff.core.off_targets import compute_off_targets_directory

        (tmp_path / "in").mkdir()
        (tmp_path / "in" / "risearch_a.out").write_text(
            (DATA / "risearch_siRNAID.out").read_text()
        )
        with pytest.raises(RuntimeError, match="stop after E_min"):
            list(
                compute_off_targets_directory(
                    input_dir=tmp_path / "in",
                    sirna_fasta=DATA / "sirnas.fa",
                    matrix="slh04",
                )
            )
        assert seen["matrix"] == "slh04"

    def test_cli_lists_the_option(self, plain):
        # Rich colours the help on CI (FORCE_COLOR), splitting "--matrix" with
        # ANSI codes; compare the plain text.
        result = runner.invoke(app, ["off-targets", "--help"], terminal_width=200)
        assert result.exit_code == 0
        out = plain(result.output)
        assert "--matrix" in out and "-z" in out

    def test_cli_passes_the_option_to_the_directory_core(self, monkeypatch, tmp_path):
        seen = {}

        def fake_directory(**kwargs):
            seen.update(kwargs)
            return iter(())

        monkeypatch.setattr(
            "sioff.core.off_targets.compute_off_targets_directory", fake_directory
        )
        (tmp_path / "in").mkdir()
        (tmp_path / "in" / "risearch_a.out").write_text(
            (DATA / "risearch_siRNAID.out").read_text()
        )
        runner.invoke(
            app,
            [
                "off-targets",
                "-r",
                str(tmp_path / "in"),
                "-s",
                str(DATA / "sirnas.fa"),
                "-z",
                "slh04",
                "-o",
                str(tmp_path / "out.tsv"),
            ],
        )
        assert seen.get("matrix") == "slh04"

    def test_config_accepts_matrix_and_resolves_a_relative_table(self, tmp_path):
        (tmp_path / "table.tsv").write_text("q1\tq2\tt1\tt2\tdelta_g_kcal_per_mol\n")
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(
            "command: off-targets\noff_targets:\n  risearch_file: hits.out\n  matrix: table.tsv\n"
        )
        kwargs = config_to_kwargs(load_config(cfg_file), "off-targets")
        assert Path(kwargs["matrix"]) == tmp_path / "table.tsv"

    def test_config_keeps_a_bundled_id(self, tmp_path):
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(
            "command: off-targets\noff_targets:\n  risearch_file: hits.out\n  matrix: slh04\n"
        )
        assert (
            config_to_kwargs(load_config(cfg_file), "off-targets")["matrix"] == "slh04"
        )
