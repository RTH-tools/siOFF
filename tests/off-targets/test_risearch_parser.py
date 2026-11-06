"""Tests for the RIsearch parser service."""

from pathlib import Path

import polars as pl
import pytest

from sioff.services.risearch_parser import RIsearchParser


@pytest.fixture
def parser() -> RIsearchParser:
    """Create a parser instance."""
    return RIsearchParser()


@pytest.fixture
def test_tsv_path() -> Path:
    """Path to the test RIsearch2 output file."""
    return Path(__file__).parent / "data" / "risearch_siRNAID.out"


class TestRIsearchParser:
    """Tests for RIsearchParser.load()."""

    def test_load_returns_dataframe(
        self, parser: RIsearchParser, test_tsv_path: Path
    ) -> None:
        """Parser returns a Polars DataFrame."""
        df = parser.load(test_tsv_path)
        assert isinstance(df, pl.DataFrame)

    def test_load_correct_row_count(
        self, parser: RIsearchParser, test_tsv_path: Path
    ) -> None:
        """Parser loads all 71 predictions from test file."""
        df = parser.load(test_tsv_path)
        assert df.height == 60

    def test_load_correct_columns(
        self, parser: RIsearchParser, test_tsv_path: Path
    ) -> None:
        """Parser creates expected column names."""
        df = parser.load(test_tsv_path)
        expected_cols = [
            "sirna_id",
            "chrom",
            "start",
            "end",
            "strand",
            "energy",
        ]
        assert df.columns == expected_cols

    def test_load_energy_is_float(
        self, parser: RIsearchParser, test_tsv_path: Path
    ) -> None:
        """Energy column has Float32 dtype."""
        df = parser.load(test_tsv_path)
        assert df["energy"].dtype == pl.Float32

    def test_load_energy_values_negative(
        self, parser: RIsearchParser, test_tsv_path: Path
    ) -> None:
        """All energy values are negative (favorable binding)."""
        df = parser.load(test_tsv_path)
        assert (df["energy"] < 0).all()

    def test_load_file_not_found(self, parser: RIsearchParser) -> None:
        """Parser raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parser.load(Path("/nonexistent/file.tsv"))
