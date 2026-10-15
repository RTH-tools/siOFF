"""preload_transcriptome slices a presorted table instead of copying it.

The off-target driver writes the transcriptome sorted by (chrom, strand, start)
to an uncompressed Arrow IPC file that every worker memory-maps; partitions
that are zero-copy slices keep that table shared between the workers. The
intersection result must not depend on whether the input was presorted.
"""

import polars as pl
import pytest

from sioff.services.intersection_service import (
    IntersectionService,
    _sorted_partition_slices,
    sort_for_preload,
)


def _trans() -> pl.DataFrame:
    # deliberately unsorted: chromosomes interleaved, starts descending
    return pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr1", "chr2", "chr1", "chr1"],
            "start": [500, 300, 100, 100, 200, 150],
            "end": [600, 400, 250, 200, 260, 170],
            "strand": ["+", "+", "+", "+", "-", "+"],
            "gene_id": ["g4", "g2", "g1", "g3", "g5", "g6"],
            "transcript_id": ["t4", "t2", "t1", "t3", "t5", "t6"],
            "exp_value": [4.0, 2.0, 1.0, 3.0, 5.0, 6.0],
        },
        schema_overrides={"exp_value": pl.Float32},
    )


def _preds() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sirna_id": ["s"] * 5,
            "chrom": ["chr1", "chr1", "chr2", "chr1", "chr2"],
            "start": [110, 160, 150, 210, 550],
            "end": [130, 165, 170, 220, 560],
            "strand": ["+", "+", "+", "-", "+"],
            "energy": [-12.0, -13.0, -14.0, -15.0, -16.0],
        }
    )


def test_unsorted_input_is_not_sliced():
    assert _sorted_partition_slices(_trans()) is None


def test_sorted_input_gives_one_slice_per_pair():
    slices = _sorted_partition_slices(sort_for_preload(_trans()))
    assert slices == {
        ("chr1", "+"): (0, 3),
        ("chr1", "-"): (3, 1),
        ("chr2", "+"): (4, 2),
    }


def test_preload_uses_views_of_the_sorted_input():
    trans = sort_for_preload(_trans())
    svc = IntersectionService()
    svc.preload_transcriptome(trans)
    part = svc._trans_partition[("chr1", "+")]
    assert part["start"].to_list() == [100, 150, 300]
    # a slice shares the parent's buffers: same chunk, offset into it
    assert part["start"].to_numpy().base is not None


@pytest.mark.parametrize("workers", [1, 2])
def test_result_does_not_depend_on_presorting(workers):
    key = ["chrom", "start", "transcript_id"]
    a, b = IntersectionService(), IntersectionService()
    a.preload_transcriptome(_trans())
    b.preload_transcriptome(sort_for_preload(_trans()))
    ra = a.intersect(_preds(), _trans(), workers=workers).sort(key)
    rb = b.intersect(_preds(), sort_for_preload(_trans()), workers=workers).sort(key)
    assert ra.height == 6  # the 160-165 site overlaps t1 and t6
    assert ra.equals(rb)
