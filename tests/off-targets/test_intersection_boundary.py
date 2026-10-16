"""
Tests for IntersectionService overlap semantics.

Focuses on the specific behaviours that were fixed:
  1. Any-overlap (not containment-only)
  2. Boundary-touching intervals do NOT overlap (BED half-open)
"""

import polars as pl

from sioff.services.intersection_service import IntersectionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pred(start, end, chrom="chr1", strand="+", sid="si1", energy=-15.0):
    return pl.DataFrame(
        {
            "sirna_id": [sid],
            "chrom": [chrom],
            "start": [start],
            "end": [end],
            "strand": [strand],
            "energy": [energy],
        }
    )


def trans(start, end, chrom="chr1", strand="+", gene="g1", tid="t1", exp=10.0):
    return pl.DataFrame(
        {
            "chrom": [chrom],
            "start": [start],
            "end": [end],
            "strand": [strand],
            "gene_id": [gene],
            "transcript_id": [tid],
            "exp_value": [exp],
        }
    )


def hits(svc, p, t):
    return svc.intersect(p, t).height


# ---------------------------------------------------------------------------
# Any-overlap tests
# ---------------------------------------------------------------------------


class TestAnyOverlap:
    """Partial overlap in either direction must be detected."""

    def test_pred_overlaps_right_end_of_transcript(self):
        # pred [10,20], trans [5,15]: pred_start=10 < trans_end=15 ✓  pred_end=20 > trans_start=5 ✓
        svc = IntersectionService()
        assert hits(svc, pred(10, 20), trans(5, 15)) == 1

    def test_pred_overlaps_left_end_of_transcript(self):
        # pred [10,20], trans [15,30]: pred_start=10 < trans_end=30 ✓  pred_end=20 > trans_start=15 ✓
        svc = IntersectionService()
        assert hits(svc, pred(10, 20), trans(15, 30)) == 1

    def test_pred_contained_inside_transcript(self):
        svc = IntersectionService()
        assert hits(svc, pred(10, 20), trans(5, 30)) == 1

    def test_transcript_contained_inside_pred(self):
        svc = IntersectionService()
        assert hits(svc, pred(5, 30), trans(10, 20)) == 1

    def test_exact_same_coordinates(self):
        svc = IntersectionService()
        assert hits(svc, pred(10, 20), trans(10, 20)) == 1


# ---------------------------------------------------------------------------
# Boundary-touching tests (the NCLS +1 fix)
# ---------------------------------------------------------------------------


class TestBoundaryTouching:
    """
    RIsearch hits and GTF/GFF3 features are 1-based closed intervals [start, end]:
    both end positions belong to the interval. Two intervals that share a
    coordinate share that nucleotide and overlap; intervals one position apart
    (end + 1 == start) are adjacent and do not. The index used to treat both as
    half-open, which dropped every hit whose last base is an exon's first base
    (and the reverse) -- 2,184 site x transcript rows for one Huesken siRNA on
    GENCODE v47, which the legacy pipeline kept.
    """

    def test_pred_end_equals_trans_start_is_a_hit(self):
        # pred [10, 20], trans [20, 30]: both contain position 20
        svc = IntersectionService()
        assert hits(svc, pred(10, 20), trans(20, 30)) == 1

    def test_pred_start_equals_trans_end_is_a_hit(self):
        # pred [20, 30], trans [10, 20]: both contain position 20
        svc = IntersectionService()
        assert hits(svc, pred(20, 30), trans(10, 20)) == 1

    def test_adjacent_intervals_no_hit(self):
        # pred [1, 5], trans [6, 10]: adjacent, no shared position
        svc = IntersectionService()
        assert hits(svc, pred(1, 5), trans(6, 10)) == 0
        assert hits(svc, pred(6, 10), trans(1, 5)) == 0

    def test_single_position_intervals(self):
        svc = IntersectionService()
        assert hits(svc, pred(7, 7), trans(7, 7)) == 1
        assert hits(svc, pred(7, 7), trans(8, 8)) == 0

    def test_one_base_overlap_is_a_hit(self):
        # pred [9, 11], trans [10, 20]: positions 10 and 11 shared
        svc = IntersectionService()
        assert hits(svc, pred(9, 11), trans(10, 20)) == 1

    def test_gencode_example(self):
        # hit chr10:100219251-100219270 (-) and exon 100219270-100219359 of
        # ENST00000370397.8 share position 100219270.
        svc = IntersectionService()
        p = pred(100219251, 100219270, chrom="chr10", strand="-")
        t = trans(100219270, 100219359, chrom="chr10", strand="-")
        assert hits(svc, p, t) == 1


# ---------------------------------------------------------------------------
# Strand specificity tests
# ---------------------------------------------------------------------------


class TestStrandSpecificity:
    def test_same_strand_matches(self):
        svc = IntersectionService()
        assert hits(svc, pred(10, 20, strand="+"), trans(5, 30, strand="+")) == 1
        assert hits(svc, pred(10, 20, strand="-"), trans(5, 30, strand="-")) == 1

    def test_opposite_strand_no_match(self):
        svc = IntersectionService()
        assert hits(svc, pred(10, 20, strand="+"), trans(5, 30, strand="-")) == 0
        assert hits(svc, pred(10, 20, strand="-"), trans(5, 30, strand="+")) == 0
