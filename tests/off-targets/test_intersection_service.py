"""Tests for the Intersection Service."""

import polars as pl
import pytest

from sioff.services.intersection_service import IntersectionService


@pytest.fixture
def service() -> IntersectionService:
    """Create service instance."""
    return IntersectionService()


@pytest.fixture
def risearch_df() -> pl.DataFrame:
    """Dummy RIsearch dataframe."""
    return pl.DataFrame(
        {
            "sirna_id": ["si1", "si2", "si3", "si4"],
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "start": [100, 200, 300, 100],
            "end": [120, 220, 320, 120],
            "strand": ["+", "+", "-", "+"],
            "energy": [-15.0, -10.0, -12.0, -10.0],
        }
    )


@pytest.fixture
def transcriptome_df() -> pl.DataFrame:
    """Dummy transcriptome dataframe."""
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [90, 250, 50],
            "end": [150, 350, 150],
            "strand": ["+", "+", "+"],
            "gene_id": ["g1", "g2", "g3"],
            "transcript_id": ["t1", "t2", "t3"],
            "exp_value": [10.0, 20.0, 30.0],
        }
    )


class TestIntersectionService:
    """Tests for intersect logic."""

    def test_intersect_containment(
        self,
        service: IntersectionService,
        risearch_df: pl.DataFrame,
        transcriptome_df: pl.DataFrame,
    ) -> None:
        """Test simple containment intersection."""
        # si1 (100-120) is inside t1 (90-150) on chr1 (+) -> MATCH
        # si4 (100-120) is inside t3 (50-150) on chr2 (+) -> MATCH

        result = service.intersect(risearch_df, transcriptome_df)

        # Check si1 match
        si1_match = result.filter(pl.col("sirna_id") == "si1")
        assert si1_match.height == 1
        assert si1_match["gene_id"][0] == "g1"

        # Check si4 match
        si4_match = result.filter(pl.col("sirna_id") == "si4")
        assert si4_match.height == 1
        assert si4_match["gene_id"][0] == "g3"

    def test_intersect_excludes_non_overlap(
        self,
        service: IntersectionService,
        risearch_df: pl.DataFrame,
        transcriptome_df: pl.DataFrame,
    ) -> None:
        """Test filtering of non-overlapping regions."""
        # si2 (200-220) is NOT in t1 (90-150) nor t2 (250-350) -> NO MATCH

        result = service.intersect(risearch_df, transcriptome_df)
        assert result.filter(pl.col("sirna_id") == "si2").height == 0

    def test_intersect_strands(
        self,
        service: IntersectionService,
        risearch_df: pl.DataFrame,
        transcriptome_df: pl.DataFrame,
    ) -> None:
        """Test strict strand matching."""
        # si3 (300-320) is on '-', t2 (250-350) is on '+' -> NO MATCH despite coordinate overlap

        result = service.intersect(risearch_df, transcriptome_df)
        assert result.filter(pl.col("sirna_id") == "si3").height == 0


@pytest.mark.parametrize("workers", [1, 3])
def test_intersect_row_order_follows_input(service: IntersectionService, workers):
    """Rows come out grouped by (chrom, strand) in first-appearance order.

    The row order feeds the float32 summation order of the partition
    functions, so it must not depend on hash order or thread scheduling (it
    used to: P_off_target drifted by a few ULP between runs on the same input).
    """
    preds = pl.DataFrame(
        {
            "sirna_id": ["s"] * 6,
            "chrom": ["chr2", "chr1", "chr1", "chr2", "chr1", "chr1"],
            "start": [100, 100, 200, 110, 300, 210],
            "end": [120, 120, 220, 130, 320, 230],
            "strand": ["+", "-", "+", "+", "-", "+"],
            "energy": [-15.0, -14.0, -13.0, -12.0, -11.0, -10.0],
        }
    )
    trans = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr1"],
            "start": [50, 50, 50],
            "end": [500, 500, 500],
            "strand": ["+", "-", "+"],
            "gene_id": ["g2p", "g1m", "g1p"],
            "transcript_id": ["t2p", "t1m", "t1p"],
            "exp_value": [1.0, 1.0, 1.0],
        }
    )
    result = service.intersect(preds, trans, workers=workers)
    assert result.height == 6
    assert list(zip(result["chrom"], result["strand"])) == [
        ("chr2", "+"),
        ("chr2", "+"),
        ("chr1", "-"),
        ("chr1", "-"),
        ("chr1", "+"),
        ("chr1", "+"),
    ]
    # Within a pair, input order is kept.
    assert result["energy"].to_list() == [-15.0, -12.0, -14.0, -11.0, -13.0, -10.0]
