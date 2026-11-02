"""Tests for the CLI module."""

import subprocess
import sys
from pathlib import Path

import pytest

from typer.testing import CliRunner

from sioff.cli import app

runner = CliRunner()


def test_accessibility_strategy_option_is_gone(plain) -> None:
    tsv_path = Path(__file__).parent / "data" / "risearch_siRNAID.out"
    result = runner.invoke(
        app,
        ["off-targets", "-r", str(tsv_path), "--accessibility-strategy", "memmap"],
    )
    assert result.exit_code == 2
    assert "No such option" in plain(result.output)


class TestCLIMain:
    """Tests for the main command."""

    def test_run_valid_file(self, plain) -> None:
        """CLI loads valid TSV and prints summary."""
        tsv_path = Path(__file__).parent / "data" / "risearch_siRNAID.out"
        result = runner.invoke(app, ["off-targets", "-r", str(tsv_path)])

        assert result.exit_code == 0
        assert "60 predictions" in plain(result.stdout)
        assert "Energy range" in plain(result.stdout)

    def test_run_missing_file(self) -> None:
        """CLI exits with error for missing file."""
        result = runner.invoke(app, ["off-targets", "-r", "/nonexistent/file.tsv"])

        assert result.exit_code != 0

    def test_run_shows_chromosomes(self, plain) -> None:
        """CLI output includes chromosome information."""
        # Use verbose to see the dataframe head where chromosomes/targets are listed
        tsv_path = Path(__file__).parent / "data" / "risearch_siRNAID.out"
        result = runner.invoke(app, ["off-targets", "-r", str(tsv_path), "-v"])

        assert "transcript_3" in plain(result.stdout)

    def test_run_shows_strands(self, plain) -> None:
        """CLI output includes strand information."""
        tsv_path = Path(__file__).parent / "data" / "risearch_siRNAID.out"
        result = runner.invoke(app, ["off-targets", "-r", str(tsv_path), "-v"])

        assert "+" in plain(result.stdout)


class TestCLIStartup:
    """`sioff --help` must not pay for the heavy runtime dependencies.

    Everything a command needs at run time (polars, numpy, pyarrow, omegaconf,
    Biopython, the core and services) is imported inside the command function,
    so building the help text costs only typer and the signatures.
    """

    HEAVY = (
        "polars",
        "numpy",
        "pyarrow",
        "omegaconf",
        "Bio",
        "sioff.api",
        "sioff.core",
        "sioff.services",
        "sioff.config",
        "sioff.models",
    )

    @pytest.mark.parametrize(
        ("argv", "extra"),
        [
            # Top-level help never reaches the group callback, so loguru (pulled
            # in by setup_logging) must stay out too.
            (["--help"], ("loguru",)),
            # Click runs the group callback — and with it setup_logging — before
            # a subcommand parses its own --help, and clears the pending args
            # first, so there is no clean way to skip it. loguru is tolerated
            # here; the heavy list above is not.
            (["off-targets", "--help"], ()),
            (["accessibility", "--help"], ()),
            (["index", "--help"], ()),
            (["search", "--help"], ()),
        ],
        ids=lambda a: " ".join(a) if isinstance(a, list) else "",
    )
    def test_help_does_not_import_heavy_deps(
        self, argv: list[str], extra: tuple[str, ...]
    ) -> None:
        # A subprocess gives a clean sys.modules; the test process has already
        # imported everything through other tests.
        code = (
            "import sys\n"
            "from typer.testing import CliRunner\n"
            "from sioff.cli import app\n"
            f"result = CliRunner().invoke(app, {argv!r})\n"
            "assert result.exit_code == 0, result.output\n"
            f"heavy = {self.HEAVY + extra!r}\n"
            "print(','.join(sorted(m for m in heavy if m in sys.modules)))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "", (
            f"`sioff {' '.join(argv)}` imported: {proc.stdout.strip()}"
        )
