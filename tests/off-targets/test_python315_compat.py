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


def test_parser_reads_sioff_search_output(tmp_path: Path) -> None:
    """``sioff search -o`` writes a headered 6-column TSV; ``-r`` must accept it."""
    path = tmp_path / "predictions.tsv"
    path.write_text(
        "\t".join(RISEARCH_COLUMNS) + "\n" + "si1\tchr1\t10\t30\t+\t-20.5\n"
    )
    df = RIsearchParser().load(path)
    assert df.schema == pl.Schema(RISEARCH_SCHEMA)
    assert df.row(0) == ("si1", "chr1", 10, 30, "+", -20.5)


def test_sirna_table_on_targets_are_tagged(tmp_path: Path) -> None:
    """The siRNA table is read by column name, whatever polars calls headerless columns."""
    table = tmp_path / "sirna_table.tsv"
    table.write_text(
        "sirna_id\ton_target_id\tsequence\nsiRNA_4\ttranscript_6\tACGUACGU\n"
    )

    df = sioff.off_targets(
        risearch_file=EXAMPLES / "predictions.out",
        gtf_file=EXAMPLES / "annotation.gtf",
        accessibility_dir=EXAMPLES / "accessibility",
        sirna_table=table,
    )

    tagged = df.filter(pl.col("is_on_target")).select("sirna_id", "transcript_id")
    assert tagged.height > 0
    assert set(tagged.iter_rows()) == {("siRNA_4", "transcript_6")}


def test_sirna_table_without_its_columns_raises_value_error(tmp_path: Path) -> None:
    table = tmp_path / "sirna_table.tsv"
    table.write_text("only_one_column\n")
    with pytest.raises(ValueError, match="missing column"):
        sioff.off_targets(
            risearch_file=EXAMPLES / "predictions.out",
            gtf_file=EXAMPLES / "annotation.gtf",
            sirna_table=table,
        )


def test_directory_mode_with_annotation(tmp_path: Path) -> None:
    """Workers read the annotation from an Arrow IPC file (`read_ipc`)."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    lines = (EXAMPLES / "predictions.out").read_text().splitlines(keepends=True)
    for sirna in ("siRNA_1", "siRNA_4"):
        rows = [line for line in lines if line.startswith(sirna + "\t")]
        (pred_dir / f"{sirna}.out").write_text("".join(rows))

    df = sioff.off_targets(
        risearch_file=pred_dir,
        gtf_file=EXAMPLES / "annotation.gtf",
        n_workers=2,
    )
    assert set(df["sirna_id"]) == {"siRNA_1", "siRNA_4"}
    assert df["gene_id"].null_count() == 0
