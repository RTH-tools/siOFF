"""A binding site counts once per transcript.

Annotations are GTF/GFF3, which carry gene_id and transcript_id. Two exon rows
with the same transcript_id are one molecule: a site touching both (e.g. across a
short intron) counts that transcript once. Rows of different transcripts are
different molecules, so each transcript containing the site counts with its own
expression, and the partition function adds them up.
"""

import polars as pl
import pytest

from sioff.services.intersection_service import IntersectionService


def _site() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sirna_id": ["si1"],
            "chrom": ["chr1"],
            "start": [160],
            "end": [170],
            "strand": ["+"],
            "energy": [-20.0],
        }
    )


def _exons(rows: list[tuple[int, int, str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["chr1"] * len(rows),
            "start": [r[0] for r in rows],
            "end": [r[1] for r in rows],
            "strand": ["+"] * len(rows),
            "gene_id": [r[2] for r in rows],
            "transcript_id": [r[3] for r in rows],
            "exp_value": [r[4] for r in rows],
        },
        schema_overrides={"exp_value": pl.Float32},
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_two_exons_of_one_transcript_count_once(workers):
    # The site spans the 2-nt intron between T1's exons 100-164 and 166-300.
    exons = _exons([(100, 164, "G", "T1", 6.0), (166, 300, "G", "T1", 6.0)])
    result = IntersectionService().intersect(_site(), exons, workers=workers)
    assert result["transcript_id"].to_list() == ["T1"]
    assert result["exp_value"].to_list() == pytest.approx([6.0])


@pytest.mark.parametrize("workers", [1, 2])
def test_each_transcript_of_a_gene_counts(workers):
    exons = _exons(
        [
            (100, 164, "G", "T1", 6.0),
            (166, 300, "G", "T1", 6.0),
            (100, 300, "G", "T2", 1.0),
            (150, 200, "G", "T3", 0.5),
        ]
    )
    result = (
        IntersectionService()
        .intersect(_site(), exons, workers=workers)
        .sort("transcript_id")
    )
    assert result["transcript_id"].to_list() == ["T1", "T2", "T3"]
    assert result["exp_value"].to_list() == pytest.approx([6.0, 1.0, 0.5])
    assert set(result["gene_id"].to_list()) == {"G"}
