"""Python 3.15 / polars 2.0 compatibility.

Python 3.15 stopped storing a function's implicit ``None`` return in
``co_consts``. polars < 2.0 relies on that to recognise the empty-bodied
``Series`` methods it forwards to ``Expr``, so on 3.15 ``Series.unique()`` and
friends silently returned ``None``. polars 2.0 fixes that but renames
headerless CSV columns (``column_0, ...`` instead of ``column_1, ...``) and
rejects ``new_columns`` shorter than the file is wide. These tests pin down
every place siOFF depended on either behaviour.
"""

from pathlib import Path

import polars as pl
import pytest

import sioff
from sioff.models import RISEARCH_COLUMNS, RISEARCH_SCHEMA
from sioff.services.risearch_parser import RIsearchParser

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "data"


def test_series_methods_forwarded_to_expr_return_a_series() -> None:
    """The polars < 2.0 failure on 3.15: these returned None instead of a Series."""
    s = pl.Series("a", ["x", "y", "x"])
    assert isinstance(s.unique(), pl.Series)
    assert sorted(s.unique().to_list()) == ["x", "y"]


def test_parser_summary_lists_chromosomes_and_strands(tmp_path: Path) -> None:
    path = tmp_path / "pred.out"
    path.write_text(
        "si1\t1\t21\tchr1\t10\t30\t+\t-20.5\nsi1\t1\t21\tchr2\t50\t70\t-\t-15.0\n"
    )
    parser = RIsearchParser()
    summary = parser.summary(parser.load(path))
    assert sorted(summary["chromosomes"]) == ["chr1", "chr2"]
    assert sorted(summary["strands"]) == ["+", "-"]


def test_parser_reads_risearch2_rows_wider_than_eight_columns(tmp_path: Path) -> None:
    """Columns are picked by position; trailing extra columns are ignored."""
    path = tmp_path / "pred.out"
    path.write_text("si1\t1\t21\tchr1\t10\t30\t+\t-20.5\textra\tmore\n")
    df = RIsearchParser().load(path)
    assert df.columns == RISEARCH_COLUMNS
    assert df.schema == pl.Schema(RISEARCH_SCHEMA)
    assert df.row(0) == ("si1", "chr1", 10, 30, "+", -20.5)


def test_load_single_file_reads_headerless_risearch2(tmp_path: Path) -> None:
    path = tmp_path / "si1.out"
    path.write_text(
        "si1\t1\t21\tchr1\t10\t30\t+\t-20.5\nsi1\t1\t21\tchr1\t50\t70\t-\t-25.0\n"
    )
    df = RIsearchParser().load_single_file(path)
    assert df["raw_e_min"].to_list() == [-25.0, -25.0]
    assert df["start"].dtype == pl.Int32


def test_parser_reads_sioff_search_output(tmp_path: Path) -> None:
    """``sioff search -o`` writes a headered 6-column TSV; ``-r`` must accept it."""
    path = tmp_path / "predictions.tsv"
    path.write_text(
        "\t".join(RISEARCH_COLUMNS) + "\n" + "si1\tchr1\t10\t30\t+\t-20.5\n"
    )
    df = RIsearchParser().load(path)
    assert df.schema == pl.Schema(RISEARCH_SCHEMA)
    assert df.row(0) == ("si1", "chr1", 10, 30, "+", -20.5)


def test_on_target_ids_file_is_parsed(tmp_path: Path) -> None:
    """The headerless sirna_id -> transcript_id map was read by polars 1.x column names."""
    kwargs = dict(
        risearch_file=EXAMPLES / "predictions.out",
        gtf_file=EXAMPLES / "annotation.gtf",
        accessibility_dir=EXAMPLES / "accessibility",
    )
    mapping = tmp_path / "on_target.tsv"
    mapping.write_text("siRNA_4\ttranscript_6\n")

    df = sioff.off_targets(**kwargs, on_target_ids_file=mapping)

    tagged = df.filter(pl.col("is_on_target")).select("sirna_id", "transcript_id")
    assert tagged.height > 0
    assert set(tagged.iter_rows()) == {("siRNA_4", "transcript_6")}


def test_malformed_on_target_ids_file_raises_value_error(tmp_path: Path) -> None:
    mapping = tmp_path / "on_target.tsv"
    mapping.write_text("only_one_column\n")
    with pytest.raises(ValueError, match="on-target mapping"):
        sioff.off_targets(
            risearch_file=EXAMPLES / "predictions.out",
            gtf_file=EXAMPLES / "annotation.gtf",
            on_target_ids_file=mapping,
        )


def test_directory_mode_with_annotation(tmp_path: Path) -> None:
    """Workers read the annotation from an Arrow IPC file (`read_ipc`)."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    lines = (EXAMPLES / "predictions.out").read_text().splitlines(keepends=True)
    for sirna in ("siRNA_1", "siRNA_4"):
        rows = [line for line in lines if line.startswith(sirna + "\t")]
        (pred_dir / f"{sirna}.out").write_text("".join(rows))

    frames = list(
        sioff.off_targets(
            risearch_file=pred_dir,
            gtf_file=EXAMPLES / "annotation.gtf",
            n_workers=2,
        )
    )
    df = pl.concat(frames, how="diagonal_relaxed")
    assert set(df["sirna_id"]) == {"siRNA_1", "siRNA_4"}
    assert df["gene_id"].null_count() == 0


def test_legacy_format_merges_alpha_gamma_columns(tmp_path: Path) -> None:
    """`--legacy-format --detailed-report` joins one column pair per (alpha, gamma) set."""
    from typer.testing import CliRunner

    from sioff.cli import app

    pred = tmp_path / "siRNA_4.out"
    lines = (EXAMPLES / "predictions.out").read_text().splitlines(keepends=True)
    pred.write_text("".join(line for line in lines if line.startswith("siRNA_4\t")))
    out = tmp_path / "out.tsv"
    result = CliRunner().invoke(
        app,
        [
            "off-targets",
            "-r",
            str(pred),
            "-t",
            str(EXAMPLES / "annotation.gtf"),
            "--legacy-format",
            "--detailed-report",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.with_suffix(".results").read_text()
    assert "## End of off-target info" in text
    assert "None" not in text
