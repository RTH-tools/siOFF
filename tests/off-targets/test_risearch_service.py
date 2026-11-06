"""Tests for RIsearchService."""

from pathlib import Path
import tempfile

import polars as pl
import pytest

# The `risearch` PyO3 bindings are an optional extra (`uv sync --extra risearch`).
# When absent — e.g. in CI, which installs core only — skip this whole module.
pytest.importorskip("risearch")

from sioff.services.risearch_service import RIsearchService  # noqa: E402


@pytest.fixture
def service() -> RIsearchService:
    """Create a RIsearchService instance."""
    return RIsearchService()


@pytest.fixture
def sirna_single_path() -> Path:
    """Path to single siRNA FASTA."""
    return Path(__file__).parent / "data" / "sirna.fa"


@pytest.fixture
def sirna_multi_path() -> Path:
    """Path to multi-siRNA FASTA."""
    return Path(__file__).parent / "data" / "sirnas.fa"


@pytest.fixture
def genome_path() -> Path:
    """Path to test genome FASTA."""
    return Path(__file__).parent / "data" / "genome.fa"


class TestValidateSirnaFasta:
    """Tests for validate_sirna_fasta()."""

    def test_validates_single_sirna(
        self, service: RIsearchService, sirna_single_path: Path
    ) -> None:
        """Single siRNA file returns list with one ID."""
        ids = service.validate_sirna_fasta(sirna_single_path)
        assert len(ids) == 1
        assert ids[0] == "siRNAID"  # Expected ID from sirna.fa

    def test_validates_multiple_sirnas(
        self, service: RIsearchService, sirna_multi_path: Path
    ) -> None:
        """Multi-siRNA file returns list with all IDs."""
        ids = service.validate_sirna_fasta(sirna_multi_path)
        assert len(ids) == 100  # User created 100 siRNAs
        assert ids[0] == "siRNA1"
        assert ids[-1] == "siRNA100"

    def test_rejects_duplicate_ids(self, service: RIsearchService) -> None:
        """Raises ValueError for duplicate siRNA IDs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
            f.write(">siRNA1\nAUGC\n>siRNA1\nGCUA\n")
            f.flush()
            dup_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Duplicate siRNA ID"):
                service.validate_sirna_fasta(dup_path)
        finally:
            dup_path.unlink()

    def test_missing_file_raises(self, service: RIsearchService) -> None:
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            service.validate_sirna_fasta(Path("/nonexistent/sirna.fa"))


class TestIndexTarget:
    """Tests for index_target()."""

    def test_creates_index(self, service: RIsearchService, genome_path: Path) -> None:
        """Creates index file from genome FASTA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "genome.idx"
            result = service.index_target(genome_path, index_path)

            assert result.exists()
            assert result.stat().st_size > 0

    def test_reuses_existing_index(
        self, service: RIsearchService, genome_path: Path
    ) -> None:
        """Reuses index if newer than source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "genome.idx"

            # Create first index
            service.index_target(genome_path, index_path)
            first_mtime = index_path.stat().st_mtime

            # Second call should reuse
            service.index_target(genome_path, index_path)
            assert index_path.stat().st_mtime == first_mtime


class TestRunSearch:
    """Tests for run_search()."""

    def test_search_returns_dataframe(
        self,
        service: RIsearchService,
        sirna_single_path: Path,
        genome_path: Path,
    ) -> None:
        """Search returns a Polars DataFrame with RISEARCH_SCHEMA columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "genome.idx"

            service.index_target(genome_path, index_path)
            result = service.run_search(sirna_single_path, index_path)

            assert isinstance(result, pl.DataFrame)
            assert set(
                ["sirna_id", "chrom", "start", "end", "strand", "energy"]
            ).issubset(set(result.columns))


def test_search_accepts_a_custom_dsm_tsv_table(tmp_path):
    """End-to-end through risearch: a long-form DSM TSV replaces a bundled id.

    The table is tiny (one stack plus the two initiation cells risearch
    requires), so it yields no hits; the point is that the path travels from
    siOFF to risearch untouched and is loaded as a table, not looked up as an
    id.
    """
    pytest.importorskip("risearch")
    data = Path(__file__).parent / "data"
    table = tmp_path / "custom.tsv"
    table.write_text(
        "q1\tq2\tt1\tt2\tdelta_g_kcal_per_mol\n"
        "A\tC\tU\tG\t-2.1805\n"
        "A\t-\tU\t-\t3.0\n"
        "-\tA\t-\tU\t2.0\n"
    )
    index = tmp_path / "g.idx"
    service = RIsearchService()
    service.index_target(data / "genome.fa", index)
    df = service.run_search(
        query_path=data / "sirnas.fa",
        index_path=index,
        target_fasta=data / "genome.fa",
        matrix=table,
    )
    assert df.columns == ["sirna_id", "chrom", "start", "end", "strand", "energy"]
