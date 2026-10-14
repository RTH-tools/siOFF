"""A binding site that overlaps several annotation rows of the same ID.

A BED annotation has one ID column, so siOFF uses it as both gene_id and
transcript_id. Files keyed by gene but carrying one row per transcript exon
(e.g. GENCODE exons with per-transcript TPM, column 4 = gene ID) then have many
overlapping rows with the same ID and different values. Each distinct row is a
distinct feature of that ID, so the site's weight is the sum of the values of the
distinct rows covering it; an exact duplicate line is the same feature listed
twice and counts once. Before this rule siOFF kept whichever row the interval
index returned first, so the weight of an on-target site could be 0 or the
largest isoform's TPM depending on row order.

With GTF input gene_id and transcript_id differ, and two rows with the same
transcript_id are two exons of one transcript: a site touching both still
counts that transcript once.
"""

import polars as pl
import pytest

from sioff.services.annotation_parser import AnnotationParser
from sioff.services.intersection_service import IntersectionService


def _site(start: int = 160, end: int = 170) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sirna_id": ["si1"],
            "chrom": ["chr1"],
            "start": [start],
            "end": [end],
            "strand": ["+"],
            "energy": [-20.0],
        }
    )


def _bed_like(rows: list[tuple[int, int, str, float]]) -> pl.DataFrame:
    """Rows as the BED loader returns them: gene_id == transcript_id == column 4."""
    return pl.DataFrame(
        {
            "chrom": ["chr1"] * len(rows),
            "start": [r[0] for r in rows],
            "end": [r[1] for r in rows],
            "strand": ["+"] * len(rows),
            "gene_id": [r[2] for r in rows],
            "transcript_id": [r[2] for r in rows],
            "exp_value": [r[3] for r in rows],
        },
        schema_overrides={"exp_value": pl.Float32},
    )


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_bed_rows_of_one_id_are_summed_and_order_does_not_matter(workers, order):
    rows = [
        (100, 200, "G", 0.0),
        (100, 200, "G", 5.0),
        (150, 200, "G", 2.0),
        (100, 200, "G", 5.0),  # exact duplicate of the second row: counts once
        (300, 400, "G", 9.0),  # does not cover the site
    ]
    rows = rows[order:] + rows[:order]
    result = IntersectionService().intersect(_site(), _bed_like(rows), workers=workers)
    assert result.height == 1
    assert result["gene_id"][0] == "G"
    assert result["exp_value"][0] == pytest.approx(7.0)


@pytest.mark.parametrize("workers", [1, 2])
def test_bed_different_ids_stay_separate(workers):
    rows = [(100, 200, "G1", 3.0), (100, 200, "G2", 4.0), (120, 200, "G1", 1.0)]
    result = (
        IntersectionService()
        .intersect(_site(), _bed_like(rows), workers=workers)
        .sort("gene_id")
    )
    assert result["gene_id"].to_list() == ["G1", "G2"]
    assert result["exp_value"].to_list() == pytest.approx([4.0, 4.0])


@pytest.mark.parametrize("workers", [1, 2])
def test_gtf_two_exons_of_one_transcript_count_once(workers):
    # A site spanning an exon boundary touches two exon rows of transcript T1.
    trans = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 165, 100],
            "end": [164, 300, 300],
            "strand": ["+", "+", "+"],
            "gene_id": ["G", "G", "G"],
            "transcript_id": ["T1", "T1", "T2"],
            "exp_value": [6.0, 6.0, 1.0],
        },
        schema_overrides={"exp_value": pl.Float32},
    )
    result = (
        IntersectionService()
        .intersect(_site(), trans, workers=workers)
        .sort("transcript_id")
    )
    assert result["transcript_id"].to_list() == ["T1", "T2"]
    assert result["exp_value"].to_list() == pytest.approx([6.0, 1.0])


def test_bed6_file_end_to_end(tmp_path):
    bed = tmp_path / "ann.bed"
    bed.write_text(
        "chr1\t100\t200\tG\t0.0\t+\n"
        "chr1\t100\t200\tG\t89.5\t+\n"
        "chr1\t150\t200\tG\t1.5\t+\n"
    )
    trans = AnnotationParser().load_gtf(bed, format="bed6")
    result = IntersectionService().intersect(_site(), trans)
    assert result.height == 1
    assert result["exp_value"][0] == pytest.approx(91.0)
