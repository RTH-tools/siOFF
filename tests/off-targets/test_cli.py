"""Tests for the CLI module."""

import shutil
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


# ---------------------------------------------------------------------------
# -o never resolves to an input: removing a stale table must not delete it
# ---------------------------------------------------------------------------
def _squash(text: str) -> str:
    """No whitespace at all: Rich wraps long paths, even inside a word."""
    return "".join(text.split())


def _refuses_to_overwrite(result, plain, flag: str, path: Path) -> None:
    assert result.exit_code == 1, result.output
    text = _squash(plain(result.output))
    assert _squash(f"-o/--output resolves to the {flag} input {path}") in text, text


def test_output_equal_to_the_predictions_is_refused(tmp_path: Path, plain) -> None:
    hits = tmp_path / "hits.tsv"
    shutil.copy(TSV, hits)
    result = runner.invoke(app, ["off-targets", "-r", str(hits), "-o", str(hits)])
    _refuses_to_overwrite(result, plain, "-r/--risearch-file", hits)
    assert hits.read_bytes() == TSV.read_bytes()


def test_output_equal_to_the_sirna_table_is_refused(tmp_path: Path, plain) -> None:
    table = tmp_path / "table.tsv"
    table.write_text("sirna_id\ton_target_id\tsequence\nsiRNAID\t\tACGU\n")
    before = table.read_bytes()
    result = runner.invoke(
        app,
        ["off-targets", "-r", str(TSV), "--sirna-table", str(table)]
        + ["-o", str(table)],
    )
    _refuses_to_overwrite(result, plain, "--sirna-table", table)
    assert table.read_bytes() == before


@pytest.mark.parametrize("spelling", ["dot-dot", "symlink"])
def test_output_spelling_the_predictions_differently_is_refused(
    tmp_path: Path, plain, spelling: str
) -> None:
    hits = tmp_path / "hits.tsv"
    shutil.copy(TSV, hits)
    if spelling == "dot-dot":
        (tmp_path / "x").mkdir()
        output = tmp_path / "x" / ".." / "hits.tsv"
    else:
        output = tmp_path / "link.tsv"
        output.symlink_to(hits)
    result = runner.invoke(app, ["off-targets", "-r", str(hits), "-o", str(output)])
    _refuses_to_overwrite(result, plain, "-r/--risearch-file", hits)
    assert hits.read_bytes() == TSV.read_bytes()
    assert output.exists()


def _per_sirna_folder(tmp_path: Path) -> Path:
    """The example predictions as one RIsearch2 .tsv per siRNA."""
    preds = tmp_path / "preds"
    preds.mkdir()
    lines = (EXAMPLES / "predictions.out").read_text().splitlines(keepends=True)
    for sid in sorted({line.split("\t", 1)[0] for line in lines}):
        rows = [line for line in lines if line.startswith(sid + "\t")]
        (preds / f"{sid}.tsv").write_text("".join(rows))
    return preds


@pytest.mark.parametrize("name", ["siRNA_1.tsv", "new.tsv"])
def test_output_the_folder_would_read_is_refused(
    tmp_path: Path, plain, name: str
) -> None:
    """A table in the -r folder would be an input: now (siRNA_1) or next run."""
    preds = _per_sirna_folder(tmp_path)
    before = {p.name: p.read_bytes() for p in preds.iterdir()}
    result = runner.invoke(
        app, ["off-targets", "-r", str(preds), "-o", str(preds / name)]
    )
    assert result.exit_code == 1, result.output
    expected = (
        f"-o/--output resolves to {preds / name} inside the -r/--risearch-file "
        f"folder {preds}, which would be read as input"
    )
    assert _squash(expected) in _squash(plain(result.output)), result.output
    assert {p.name: p.read_bytes() for p in preds.iterdir()} == before


def test_output_the_folder_ignores_is_allowed(tmp_path: Path) -> None:
    """The folder reader skips .csv, so a rerun does not read the table back."""
    preds = _per_sirna_folder(tmp_path)
    out = preds / "out.csv"
    for _ in range(2):
        result = runner.invoke(
            app,
            ["off-targets", "-r", str(preds), "-o", str(out)]
            + ["--output-format", "csv"],
        )
        assert result.exit_code == 0, result.output
        assert pl.read_csv(out)["sirna_id"].n_unique() == 5


# ---------------------------------------------------------------------------
# An argument error keeps the previous table; -o is resolved before its guards
# ---------------------------------------------------------------------------
def _argument_error(case: str, tmp_path: Path) -> list[str]:
    """off-targets arguments the engine refuses before scoring anything."""
    if case == "-r with -s":
        return ["-r", str(EXAMPLES / "predictions.out")] + [
            "-s",
            str(EXAMPLES / "sirnas.fa"),
        ]
    if case == "alpha/gamma without a table":
        return ["-r", str(EXAMPLES / "predictions.out")] + [
            "--alpha",
            "0.8",
            "--gamma",
            "0.8",
        ]
    if case == "cache dir without -a":
        return ["-r", str(EXAMPLES / "predictions.out")] + [
            "--accessibility-cache-dir",
            str(tmp_path / "cache"),
        ]
    # A sirna_id in two files of the folder.
    folder = tmp_path / "dup"
    folder.mkdir()
    shutil.copy(EXAMPLES / "predictions.out", folder / "x.out")
    shutil.copy(EXAMPLES / "predictions.out", folder / "y.out")
    return ["-r", str(folder)]


@pytest.mark.parametrize(
    "case",
    [
        "-r with -s",
        "alpha/gamma without a table",
        "cache dir without -a",
        "duplicate sirna_id",
    ],
)
def test_an_argument_error_keeps_the_previous_table(tmp_path: Path, case) -> None:
    out = tmp_path / "results.tsv"
    out.write_text("a table from an earlier run\n")
    before = out.read_bytes()
    result = runner.invoke(
        app, ["off-targets", *_argument_error(case, tmp_path), "-o", str(out)]
    )
    assert result.exit_code == 1, result.output
    assert out.read_bytes() == before


def test_output_through_a_missing_dir_cannot_reach_the_predictions(
    tmp_path: Path, plain
) -> None:
    """``ghost/..`` exists only once -o's parent is created; resolve it first."""
    hits = tmp_path / "hits.tsv"
    shutil.copy(TSV, hits)
    output = tmp_path / "ghost" / ".." / "hits.tsv"
    result = runner.invoke(app, ["off-targets", "-r", str(hits), "-o", str(output)])
    _refuses_to_overwrite(result, plain, "-r/--risearch-file", hits)
    assert hits.read_bytes() == TSV.read_bytes()
    assert not (tmp_path / "ghost").exists()


def test_output_through_a_missing_dir_cannot_land_in_the_folder(
    tmp_path: Path, plain
) -> None:
    preds = _per_sirna_folder(tmp_path)
    before = {p.name: p.read_bytes() for p in preds.iterdir()}
    output = preds / "ghost2" / ".." / "siRNA_1.tsv"
    result = runner.invoke(app, ["off-targets", "-r", str(preds), "-o", str(output)])
    assert result.exit_code == 1, result.output
    assert "inside the -r/--risearch-file folder" in " ".join(
        plain(result.output).split()
    )
    assert {p.name: p.read_bytes() for p in preds.iterdir()} == before
