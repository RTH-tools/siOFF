"""Tests for the annotation parser service."""

from pathlib import Path

import polars as pl
import pytest

from sioff.services.annotation_parser import AnnotationParser


@pytest.fixture
def parser() -> AnnotationParser:
    """Create a parser instance."""
    return AnnotationParser()


@pytest.fixture
def test_gtf_path() -> Path:
    """Path to the test GTF file."""
    return Path(__file__).parent / "data" / "expression_data.gtf"


class TestAnnotationParser:
    """Tests for loading GTF files."""

    def test_load_gtf_returns_dataframe(
        self, parser: AnnotationParser, test_gtf_path: Path
    ) -> None:
        """Parser returns a Polars DataFrame."""
        df = parser.load_gtf(test_gtf_path)
        assert isinstance(df, pl.DataFrame)

    def test_load_gtf_filters_exons(
        self, parser: AnnotationParser, test_gtf_path: Path
    ) -> None:
        """Parser filters annotations by feature type (default 'exon')."""
        # The test file has transcripts and exons. We expect only exons.
        # Let's count quickly manually from user's file view or rely on logic.
        # User file snippet shows interleaved transcript/exon lines.
        # Assuming we filter for exons, we should get fewer rows than total.
        df = parser.load_gtf(test_gtf_path)
        # Total lines in file is 119
        # Many lines are "transcript". "exons" > 0.
        assert df.height > 0
        assert df.height < 119

    def test_load_gtf_extracts_attributes(
        self, parser: AnnotationParser, test_gtf_path: Path
    ) -> None:
        """Parser correctly extracts gene_id, transcript_id, and score."""
        df = parser.load_gtf(test_gtf_path)

        # Check first row (should be an exon based on file snippet)
        # Row 2 in file is first exon: gene_id "gene_1"; transcript_id "transcript_21"; RPKM "1000"
        # Polars order should be preserved (CSV reading is sequential)

        row = df.row(0, named=True)
        assert row["gene_id"] == "gene_1"
        assert row["transcript_id"] == "transcript_21"
        assert row["exp_value"] == 1000.0

    def test_load_gtf_unknown_score_column_raises(
        self, parser: AnnotationParser, test_gtf_path: Path
    ) -> None:
        """A score attribute that matches no row is an error, not silently 0.0.

        This previously asserted `(df["exp_value"] == 0.0).all()`. That encoded the
        silent-wrong-answer path: a misnamed score column zeroed every weight in
        ``W_i = Expression_i * exp(-dG/RT)`` while the run exited 0. Total
        extraction failure now raises; partial nulls still fill to 0.0.
        """
        with pytest.raises(ValueError, match="NONEXISTENT"):
            parser.load_gtf(test_gtf_path, score_col="NONEXISTENT")

    def test_load_gtf_missing_file(self, parser: AnnotationParser) -> None:
        """Parser raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parser.load_gtf(Path("/nonexistent/file.gtf"))


TRANSCRIPT_ROWS = (
    'chr1\tt\ttranscript\t100\t300\t.\t+\t.\tgene_id "g1"; transcript_id "t1"; TPM "4";\n'
    'chr1\tt\ttranscript\t500\t700\t.\t-\t.\tgene_id "g2"; transcript_id "t2"; TPM "7";\n'
    'chr1\tt\tCDS\t120\t280\t.\t+\t.\tgene_id "g1"; transcript_id "t1"; TPM "4";\n'
)


class TestFeatureWithoutRows:
    """A feature type that no row of the file has is an error.

    The filter used to leave an empty frame, which the all-null guard skips
    (it has no row to look at), so the run went on against an empty
    annotation, scored nothing and exited 0.
    """

    def test_default_exon_on_a_transcript_level_file_raises(
        self, parser: AnnotationParser, tmp_path: Path
    ) -> None:
        path = tmp_path / "transcripts.gtf"
        path.write_text(TRANSCRIPT_ROWS)
        with pytest.raises(ValueError) as excinfo:
            parser.load_gtf(path, score_col="TPM")
        message = str(excinfo.value)
        assert "No 'exon' rows in transcripts.gtf" in message
        # The types the file does hold, most frequent first, with their counts.
        assert "transcript (2), CDS (1)" in message
        assert "--feature" in message

    def test_the_guard_is_not_masked_by_a_wrong_expression_attribute(
        self, parser: AnnotationParser, tmp_path: Path
    ) -> None:
        """No row is left to miss the attribute, so the feature is the error."""
        path = tmp_path / "transcripts.gtf"
        path.write_text(TRANSCRIPT_ROWS)
        with pytest.raises(ValueError, match="No 'exon' rows"):
            parser.load_gtf(path, score_col="RPKM")

    def test_a_feature_the_file_has_still_loads(
        self, parser: AnnotationParser, tmp_path: Path
    ) -> None:
        path = tmp_path / "transcripts.gtf"
        path.write_text(TRANSCRIPT_ROWS)
        df = parser.load_gtf(path, feature="transcript", score_col="TPM")
        assert df["transcript_id"].to_list() == ["t1", "t2"]

    def test_gff3_feature_without_rows_raises(
        self, parser: AnnotationParser, tmp_path: Path
    ) -> None:
        path = tmp_path / "ann.gff3"
        path.write_text(
            "##gff-version 3\n"
            "chr1\t.\tmRNA\t100\t300\t.\t+\t.\tID=t1;gene_id=g1;TPM=4\n"
        )
        with pytest.raises(ValueError, match=r"No 'exon' rows.*mRNA \(1\)"):
            parser.load_gtf(path, score_col="TPM")
