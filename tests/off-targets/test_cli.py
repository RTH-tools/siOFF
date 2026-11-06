"""Tests for the CLI module."""

import subprocess
import sys
from pathlib import Path

import pytest

import polars as pl
from typer.testing import CliRunner

from sioff.cli import app

runner = CliRunner()
TSV = Path(__file__).parent / "data" / "risearch_siRNAID.out"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "data"


def test_accessibility_strategy_option_is_gone(plain) -> None:
    tsv_path = Path(__file__).parent / "data" / "risearch_siRNAID.out"
    result = runner.invoke(
        app,
        ["off-targets", "-r", str(tsv_path), "--accessibility-strategy", "memmap"],
    )
    assert result.exit_code == 2
    assert "No such option" in plain(result.output)


@pytest.mark.parametrize(
    "flag",
    [
        "--on-target",
        "--on-target-risearch-file",
        "--query",
        "--on-target-expression",
        "--on-target-accessibility",
    ],
)
def test_fasta_on_target_options_are_gone(plain, flag) -> None:
    tsv_path = Path(__file__).parent / "data" / "risearch_siRNAID.out"
    result = runner.invoke(app, ["off-targets", "-r", str(tsv_path), flag, "x"])
    assert result.exit_code == 2
    assert "No such option" in plain(result.output)


class TestCLIMain:
    """Tests for the main command."""

    def test_run_valid_file(self, tmp_path: Path) -> None:
        """No -t, so the output has one row per prediction."""
        out = tmp_path / "o.tsv"
        result = runner.invoke(app, ["off-targets", "-r", str(TSV), "-o", str(out)])

        assert result.exit_code == 0, result.output
        assert pl.read_csv(out, separator="\t").height == 60

    def test_run_missing_file(self) -> None:
        """CLI exits with error for missing file."""
        result = runner.invoke(app, ["off-targets", "-r", "/nonexistent/file.tsv"])

        assert result.exit_code != 0

    def test_run_shows_chromosomes(self, tmp_path: Path) -> None:
        """The written table keeps each prediction's chromosome."""
        out = tmp_path / "o.tsv"
        result = runner.invoke(app, ["off-targets", "-r", str(TSV), "-o", str(out)])

        assert result.exit_code == 0, result.output
        assert "transcript_3" in pl.read_csv(out, separator="\t")["chrom"].to_list()

    def test_run_shows_strands(self, tmp_path: Path) -> None:
        """The written table keeps each prediction's strand."""
        out = tmp_path / "o.tsv"
        result = runner.invoke(app, ["off-targets", "-r", str(TSV), "-o", str(out)])

        assert result.exit_code == 0, result.output
        assert set(pl.read_csv(out, separator="\t")["strand"]) == {"+", "-"}


def test_sirna_table_option_replaces_on_target_ids(tmp_path: Path, plain) -> None:
    result = runner.invoke(
        app, ["off-targets", "-r", str(TSV), "--on-target-ids", str(TSV)]
    )
    assert result.exit_code == 2
    assert "No such option" in plain(result.output)

    help_text = plain(
        runner.invoke(app, ["off-targets", "--help"], terminal_width=200).output
    )
    assert "--sirna-table" in help_text
    assert "--on-target-ids" not in help_text

    table = tmp_path / "sirna_table.tsv"
    table.write_text("sirna_id\ton_target_id\tsequence\nsiRNAID\ttranscript_3\tACGU\n")
    out = tmp_path / "o.tsv"
    result = runner.invoke(
        app,
        ["off-targets", "-r", str(TSV), "--sirna-table", str(table), "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert pl.read_csv(out, separator="\t").height == 60


def test_file_input_honours_output_format(tmp_path: Path) -> None:
    out = tmp_path / "o.tsv"
    result = runner.invoke(
        app,
        ["off-targets", "-r", str(TSV), "-o", str(out), "--output-format", "parquet"],
    )
    assert result.exit_code == 0, result.output
    assert not out.exists()
    assert pl.read_parquet(tmp_path / "o.parquet").height == 60


def test_rows_are_sorted_within_each_sirna(tmp_path: Path) -> None:
    out = tmp_path / "out" / "o.tsv"
    result = runner.invoke(
        app,
        [
            "off-targets",
            "-r",
            str(EXAMPLES / "predictions.out"),
            "-t",
            str(EXAMPLES / "annotation.gtf"),
            "-j",
            "2",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    df = pl.read_csv(out, separator="\t")
    # One contiguous block per siRNA ...
    blocks = df["sirna_id"].rle()
    assert blocks.struct.field("value").n_unique() == blocks.len() == 5
    # ... each non-increasing in P_off_target.
    for (sid,), block in df.group_by("sirna_id", maintain_order=True):
        p = block["P_off_target"]
        assert (p.diff().drop_nulls() <= 0).all(), sid


@pytest.mark.parametrize("output", ["out/o.tsv", "out"])
def test_single_sirna_run_writes_a_summary(tmp_path: Path, output: str) -> None:
    """One .summary per siRNA: next to -o, or inside it when it has no suffix."""
    result = runner.invoke(
        app, ["off-targets", "-r", str(TSV), "-o", str(tmp_path / output)]
    )
    assert result.exit_code == 0, result.output
    assert [p.relative_to(tmp_path) for p in tmp_path.rglob("*.summary")] == [
        Path("out") / "siRNAID.summary"
    ]


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


# ---------------------------------------------------------------------------
# -o names the table; a stale table goes; errors name the flags
# ---------------------------------------------------------------------------
def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize(
    ("output", "fmt", "table"),
    [
        ("results.txt", "tsv", "results.txt.tsv"),
        ("x.tsv", "parquet", "x.parquet"),
        ("x.parquet", "csv", "x.csv"),
        ("out", "tsv", "out.tsv"),
    ],
)
def test_output_table_keeps_the_name_given(tmp_path: Path, output, fmt, table) -> None:
    """A table suffix is replaced by --output-format's; any other name is kept."""
    result = runner.invoke(
        app,
        ["off-targets", "-r", str(TSV), "-o", str(tmp_path / output)]
        + ["--output-format", fmt],
    )
    assert result.exit_code == 0, result.output
    summary = "out/siRNAID.summary" if output == "out" else "siRNAID.summary"
    assert _files(tmp_path) == sorted([table, summary])


def test_dotted_output_names_do_not_collide(tmp_path: Path) -> None:
    for name in ("run.v1", "run.v2"):
        result = runner.invoke(
            app, ["off-targets", "-r", str(TSV), "-o", str(tmp_path / name)]
        )
        assert result.exit_code == 0, result.output
    assert pl.read_csv(tmp_path / "run.v1.tsv", separator="\t").height == 60
    assert pl.read_csv(tmp_path / "run.v2.tsv", separator="\t").height == 60


def test_run_that_scores_nothing_leaves_no_stale_table(tmp_path: Path, plain) -> None:
    """The fixture's chromosomes (transcript_1..5) are not in the example GTF."""
    out = tmp_path / "o.tsv"
    out.write_text("a stale table from an earlier run\n")
    result = runner.invoke(
        app,
        ["off-targets", "-r", str(TSV), "-t", str(EXAMPLES / "annotation.gtf")]
        + ["-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert not out.exists()
    assert "No predictions to process" in plain(result.output)


def test_table_path_that_is_not_a_file_is_an_error(tmp_path: Path, plain) -> None:
    (tmp_path / "o.tsv").mkdir()
    result = runner.invoke(
        app, ["off-targets", "-r", str(TSV), "-o", str(tmp_path / "o.tsv")]
    )
    assert result.exit_code == 1
    text = " ".join(plain(result.output).split())
    assert "exists and is not a regular file" in text
    assert (tmp_path / "o.tsv").is_dir()


def test_errors_name_the_cli_flags(plain) -> None:
    result = runner.invoke(
        app,
        ["off-targets", "-r", str(TSV), "-f", str(EXAMPLES / "genome.fa")]
        + ["-a", str(EXAMPLES / "accessibility")],
    )
    assert result.exit_code == 1
    text = " ".join(plain(result.output).split())
    assert (
        "-f/--fasta and -a/--accessibility-dir are two sources for the same "
        "profiles; drop -f/--fasta to use -a/--accessibility-dir" in text
    )
    assert "genome_file" not in text and "accessibility_dir" not in text


def test_flag_names_are_the_options_of_the_command() -> None:
    """Every flag an error message names is spelled as the command declares it."""
    import typer.main

    from sioff.commands.off_targets import _FLAG_NAMES

    command = typer.main.get_command(app).commands["off-targets"]
    options = {param.name: set(param.opts) for param in command.params}
    for name, flags in _FLAG_NAMES.items():
        assert set(flags.split("/")) <= options[name], name
