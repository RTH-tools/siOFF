"""Tests for AnnotationParser format flag logic."""

import tempfile
from pathlib import Path

import pytest

from sioff.services.annotation_parser import AnnotationParser


@pytest.fixture
def parser():
    return AnnotationParser()


@pytest.mark.parametrize(
    "suffix,fmt",
    [(".bed", "auto"), (".bed.gz", "auto"), (".txt", "bed6"), (".txt", "bed7")],
)
def test_bed_annotations_are_rejected(parser, tmp_path, suffix, fmt):
    """BED has one ID column, so it cannot say which rows are one transcript."""
    path = tmp_path / f"ann{suffix}"
    path.write_text("chr1\t100\t200\tprot1\t0.5\t+\n")
    with pytest.raises(ValueError, match="BED annotations are not supported"):
        parser.load_gtf(path, format=fmt)


def test_gtf_row_without_transcript_id_falls_back_to_gene_id(parser, tmp_path):
    path = tmp_path / "ann.gtf"
    path.write_text('chr1\t.\texon\t100\t200\t.\t+\t.\tgene_id "g1"; RPKM "5.0";\n')
    df = parser.load_gtf(path, format="gtf")
    assert df["transcript_id"].to_list() == ["g1"]


def test_load_gtf_explicit_gtf(parser):
    """Test explicit GTF format."""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(
            'chr1\t.\texon\t100\t200\t.\t+\t.\tgene_id "g1"; transcript_id "t1"; RPKM "50.0";\n'
        )
        path = Path(f.name)

    try:
        # Explicit GTF, even with .txt extension
        df = parser.load_gtf(path, format="gtf")
        assert df.height == 1
        assert df["gene_id"][0] == "g1"
        assert df["exp_value"][0] == 50.0
    finally:
        path.unlink()
