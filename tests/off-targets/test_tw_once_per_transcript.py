"""--type tw: a hit counts once per transcript, whatever rows the annotation has.

A transcriptome-wide hit names its transcript, and the annotation only says
how much of that transcript there is. The join used to make one scored row
per matching annotation row, so with exon rows a transcript with N exons had
each of its hits N times, with N times the Boltzmann weight; the genome-wide
path already kept one row per (site, transcript).

Now the annotation is reduced to one row per transcript before the join, and
``feature_type`` defaults to ``transcript`` in this mode. Every test goes
through the public ``sioff.off_targets`` and asserts outcomes (rows, P, Z).
"""

import math
from pathlib import Path

import polars as pl
import pytest

import sioff
from sioff.models import PredictionsMode
from sioff.services.intersection_service import IntersectionService

RT = 0.001987 * (37.0 + 273.15)
ENERGY = -20.0
RPKM = 10.0
#: Boltzmann weight of one hit: Expression * exp(-dG / RT).
W = RPKM * math.exp(-ENERGY / RT)


def _row(feature, start, end, strand, gene, tx, rpkm=RPKM) -> str:
    return (
        f"chr1\ttest\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t"
        f'gene_id "{gene}"; transcript_id "{tx}"; RPKM "{rpkm:g}";\n'
    )


def _annotation(path: Path, n_exons_t1: int = 2, transcript_rows: bool = True) -> Path:
    """T1 with ``n_exons_t1`` exon rows, T2 with one; a transcript row each."""
    rows = [
        _row("exon", 101 + 200 * i, 200 + 200 * i, "-", "G1", "T1")
        for i in range(n_exons_t1)
    ]
    rows.append(_row("exon", 9001, 9200, "+", "G2", "T2"))
    if transcript_rows:
        rows.append(_row("transcript", 101, 100 + 200 * n_exons_t1, "-", "G1", "T1"))
        rows.append(_row("transcript", 9001, 9200, "+", "G2", "T2"))
    path.write_text("".join(rows))
    return path


def _hits() -> pl.DataFrame:
    """One transcriptome-wide hit per transcript, identical energy."""
    return pl.DataFrame(
        {
            "sirna_id": ["si1", "si1"],
            "chrom": ["T1", "T2"],  # tw: the target name is the transcript ID
            "start": [50, 50],
            "end": [70, 70],
            "strand": ["+", "+"],
            "energy": [ENERGY, ENERGY],
        }
    )


def _score(annotation: Path, **kwargs) -> pl.DataFrame:
    return sioff.off_targets(
        predictions=_hits(),
        gtf_file=annotation,
        predictions_type="tw",
        n_workers=1,
        **kwargs,
    ).sort("transcript_id")


@pytest.mark.parametrize("n_exons_t1", [2, 5])
def test_hit_counts_once_per_transcript_with_exon_rows(tmp_path, n_exons_t1):
    """It used to be 1 + n rows, and P(T1) = n / (n + 1)."""
    df = _score(
        _annotation(tmp_path / "a.gtf", n_exons_t1),
        feature_type="exon",
        detailed_report=True,
    )
    assert df["transcript_id"].to_list() == ["T1", "T2"]
    assert df["P_off_target"].to_list() == pytest.approx([0.5, 0.5])
    assert df["boltzmann_weight"].to_list() == pytest.approx([W, W], rel=1e-9)
    assert df["Z_sirna"].to_list() == pytest.approx([2 * W, 2 * W], rel=1e-9)


def test_exon_rows_and_transcript_rows_give_the_same_table(tmp_path):
    annotation = _annotation(tmp_path / "a.gtf")
    by_exon = _score(annotation, feature_type="exon")
    by_transcript = _score(annotation, feature_type="transcript")
    columns = ["sirna_id", "chrom", "transcript_id", "gene_id", "exp_value"]
    assert by_exon.select(columns).equals(by_transcript.select(columns))
    assert by_exon["P_off_target"].to_list() == pytest.approx(
        by_transcript["P_off_target"].to_list()
    )


def test_on_target_weight_is_not_multiplied_by_exon_rows(tmp_path):
    """The on-target (T1, two exon rows) holds half of Z, not two thirds."""
    table = tmp_path / "sirna_table.tsv"
    table.write_text(
        "sirna_id\ton_target_id\tsequence\nsi1\tT1\tUGCGAACCCCGCCCAUAAAUU\n"
    )
    df = _score(_annotation(tmp_path / "a.gtf"), feature_type="exon", sirna_table=table)
    on_target = df.filter(pl.col("is_on_target"))
    assert on_target.height == 1
    assert on_target["P_off_target"].sum() == pytest.approx(0.5)


def test_a_transcript_listed_twice_counts_once(tmp_path):
    """Selecting transcript rows does not by itself make them unique."""
    gtf = tmp_path / "a.gtf"
    gtf.write_text(
        _row("transcript", 101, 500, "-", "G1", "T1")
        + _row("transcript", 101, 500, "-", "G1", "T1")
        + _row("transcript", 9001, 9200, "+", "G2", "T2")
    )
    df = _score(gtf)
    assert df["transcript_id"].to_list() == ["T1", "T2"]
    assert df["P_off_target"].to_list() == pytest.approx([0.5, 0.5])


def test_feature_defaults_to_transcript_in_tw(tmp_path):
    """One row per transcript and no feature_type: the file of the bug report."""
    gtf = tmp_path / "transcripts.gtf"
    gtf.write_text(
        _row("transcript", 101, 500, "-", "G1", "T1")
        + _row("transcript", 9001, 9200, "+", "G2", "T2")
    )
    assert _score(gtf)["transcript_id"].to_list() == ["T1", "T2"]


def test_default_transcript_without_such_rows_names_the_types_present(tmp_path):
    gtf = _annotation(tmp_path / "exons.gtf", transcript_rows=False)
    with pytest.raises(ValueError, match=r"No 'transcript' rows.*exon \(3\)"):
        _score(gtf)
    # The exon rows hold all that is needed: asking for them works.
    assert _score(gtf, feature_type="exon")["transcript_id"].to_list() == ["T1", "T2"]


def test_feature_still_defaults_to_exon_in_gw(tmp_path):
    gtf = tmp_path / "transcripts.gtf"
    gtf.write_text(_row("transcript", 101, 500, "+", "G1", "T1"))
    hits = _hits().with_columns(chrom=pl.lit("chr1"), start=150, end=170)
    with pytest.raises(ValueError, match="No 'exon' rows"):
        sioff.off_targets(predictions=hits, gtf_file=gtf)


@pytest.mark.parametrize("values", [(10, 7), (10, 0)])
def test_rows_of_one_transcript_that_disagree_on_expression_are_refused(
    tmp_path, values
):
    """Which one would weigh the hit? Not a guess: the transcripts are named."""
    gtf = tmp_path / "a.gtf"
    gtf.write_text(
        _row("exon", 101, 200, "-", "G1", "T1", values[0])
        + _row("exon", 301, 400, "-", "G1", "T1", values[1])
        + _row("exon", 9001, 9200, "+", "G2", "T2")
    )
    with pytest.raises(ValueError) as excinfo:
        _score(gtf, feature_type="exon")
    message = str(excinfo.value)
    assert "1 transcript" in message and "T1" in message and "T2" not in message
    assert "RPKM" in message


def test_tw_table_has_the_genome_wide_columns(tmp_path):
    """No leftovers of the join (chrom_trans, start_trans, ...)."""
    annotation = _annotation(tmp_path / "a.gtf", n_exons_t1=2)
    tw = _score(annotation, feature_type="exon")
    gw = sioff.off_targets(
        predictions=_hits().with_columns(
            chrom=pl.lit("chr1"), start=150, end=170, strand=pl.lit("-")
        ),
        gtf_file=annotation,
    )
    assert tw.columns == gw.columns
    # The extent of the transcript's selected rows.
    assert tw.select("transcript_id", "trans_start", "trans_end").rows() == [
        ("T1", 101, 400),
        ("T2", 9001, 9200),
    ]


def test_the_join_alone_counts_a_hit_once_per_transcript():
    """Callers of the service get the rule too, without the engine's reduction."""
    annotation = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [101, 301, 9001],
            "end": [200, 400, 9200],
            "strand": ["-", "-", "+"],
            "gene_id": ["G1", "G1", "G2"],
            "transcript_id": ["T1", "T1", "T2"],
            "exp_value": [10.0, 10.0, 10.0],
        },
        schema_overrides={"start": pl.Int32, "end": pl.Int32, "exp_value": pl.Float32},
    )
    hits = _hits().cast({"start": pl.Int32, "end": pl.Int32, "energy": pl.Float32})
    joined = IntersectionService().intersect(hits, annotation, mode=PredictionsMode.TW)
    assert joined["transcript_id"].to_list() == ["T1", "T2"]
    assert joined.columns == [
        *hits.columns,
        "trans_start",
        "trans_end",
        "gene_id",
        "transcript_id",
        "exp_value",
    ]
