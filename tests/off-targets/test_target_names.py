"""A target is named by the first word of its FASTA header.

The ``risearch`` command line names a target by its whole header line
(``rna-TX1.1 CDS=10-200``), while everything a prediction is matched against
is keyed by the header's first word: the annotation (``transcript_id`` with
``--type tw``, the seqname with ``--type gw``), the accessibility profiles
(``sioff accessibility``) and the hits of ``sioff search``.

Kept verbatim, such a name matched nothing: the hit was dropped by the
annotation join, or kept the 10.0 kcal/mol opening energy of an unknown
record, and the run exited 0. Each engine test runs the first-word input as a
control and asserts the row count, the opening energies and the probabilities.
"""

from pathlib import Path

import polars as pl
import pytest

import sioff
from sioff.core.off_targets import _coerce_predictions
from sioff.services.risearch_parser import RIsearchParser

# Two headers "id description", one single-word header; 240 nt each.
TRANSCRIPTS_FASTA = """\
>rna-TX1.1 CDS=10-200
CATAAGTGCGCTATCCGACTCGCTTGGGAATGGGCCATCAGATGTGTTACCTAACAACCG
ACGGATACACGGCGTGTGTGAAATGACCACCCGCTTGTGTTCGAAGCGCGTTTTAAATTC
TGTTGGCGCGCCGACACGGTACCGAAACGTCCAAATTGAGCAATAATCCTCTGATCCAAG
TCGACTTCGGCTGCTGTGAATCGGCATTTAAACCGGTGAGTTGGTATCCGCTACCTCTCC
>rna-TX2.1 CDS=5-150
GCGAGGGGTGCGCAAGAAGCTAGAGTCTTTTCTAATGGGTATCGTAGGCCAGTTGGTCTG
CCAGTATTCCACAGATTCATTCACTATGGTGGAAAAATCCTTGCGTACAAAACACAGACC
AACACTTGCCATCTCAGCAAGGTCAACGATTCCGATTTAATAGTTGCCCCAAGGCGTTGA
TATAAGTGTACGGGCTTAGAGTTCAGGTGGGAAATCCTCTACAAATTGAACCCGTATGGA
>rna-NC3.1
ACCTAGCTGGACGATAGAATATTAACTAGGAATTGGAGACTTGGAGCGGAAAGGTTCAGG
GGGTGATAGCGTGGGTTCCAGCTATACTAGAAGTCCGGATAATTTATTTCCATGTCTGGA
GGTCACAGCAAACTGTTAGACTGAAAGATTGGCAATTAAGCTAACCACATTATTTGCGAC
GTACTCGATGCACGTCAGCTAACCTACTGATATTCCTAGCCATGCGTATGAGGACCAGAA
"""

# One row per transcript, hence feature_type="transcript".
TRANSCRIPTS_GTF = "".join(
    f'{tid}\tsrc\ttranscript\t1\t240\t.\t+\t.\tgene_id "{gene}"; transcript_id "{tid}"; RPKM "{rpkm}";\n'
    for tid, gene, rpkm in (
        ("rna-TX1.1", "gene1", 10),
        ("rna-TX2.1", "gene2", 20),
        ("rna-NC3.1", "gene3", 30),
    )
)

# (sirna_id, target as the risearch command line writes it, start, end, energy)
TW_HITS = [
    ("si1", "rna-TX1.1 CDS=10-200", 51, 71, -20.5),
    ("si1", "rna-TX2.1 CDS=5-150", 101, 121, -15.25),
    ("si1", "rna-NC3.1", 31, 51, -12.0),
    ("si2", "rna-TX1.1 CDS=10-200", 150, 170, -18.0),
    ("si2", "rna-TX2.1 CDS=5-150", 200, 220, -11.0),
    ("si3", "rna-NC3.1", 120, 140, -14.0),
    ("si3", "rna-TX2.1 CDS=5-150", 60, 80, -13.5),
]

GENOME_GTF = (
    'chr1\tsrc\texon\t41\t100\t.\t+\t.\tgene_id "gA"; transcript_id "tA"; RPKM "5";\n'
    'chr1\tsrc\texon\t141\t200\t.\t+\t.\tgene_id "gB"; transcript_id "tB"; RPKM "7";\n'
)
GW_HITS = [
    ("si1", "chr1 some description", 50, 70, -20.5),
    ("si1", "chr1 some description", 150, 170, -15.25),
    ("si2", "chr1 some description", 60, 80, -18.0),
]

UNKNOWN_RECORD_ENERGY = 10.0  # what a hit on a record without a profile keeps


def _first_word(name: str) -> str:
    return name.split()[0]


def _risearch2_tsv(path: Path, hits, name=lambda n: n) -> Path:
    """Write hits in the RIsearch2 layout: 8 tab-separated columns, no header."""
    path.write_text(
        "".join(
            f"{sid}\t1\t21\t{name(target)}\t{start}\t{end}\t+\t{energy}\n"
            for sid, target, start, end, energy in hits
        )
    )
    return path


def _frame(hits, name=lambda n: n) -> pl.DataFrame:
    """The same hits as an in-memory frame in the ``sioff.search`` schema."""
    return pl.DataFrame(
        {
            "sirna_id": [h[0] for h in hits],
            "chrom": [name(h[1]) for h in hits],
            "start": [h[2] for h in hits],
            "end": [h[3] for h in hits],
            "strand": ["+"] * len(hits),
            "energy": [h[4] for h in hits],
        }
    )


def _source(tmp_path: Path, hits, form: str, names: str) -> dict:
    """The predictions argument of ``sioff.off_targets`` for one input form."""
    name = _first_word if names == "first_word" else (lambda n: n)
    if form == "frame":  # through _coerce_predictions
        return {"predictions": _frame(hits, name)}
    if form == "file":  # through RIsearchParser.load, split per siRNA
        return {"risearch_file": _risearch2_tsv(tmp_path / f"{names}.tsv", hits, name)}
    # A folder of one-siRNA Parquet files: scored in place, never split.
    folder = tmp_path / f"{names}_parquet"
    folder.mkdir()
    for (sid,), part in (
        _frame(hits, name).partition_by("sirna_id", as_dict=True).items()
    ):
        part.write_parquet(folder / f"{sid}.parquet")
    return {"risearch_file": folder}


def _numbers(df: pl.DataFrame) -> list[tuple]:
    return sorted(
        (sid, chrom, start, round(float(oe), 4), round(float(p), 7))
        for sid, chrom, start, oe, p in zip(
            df["sirna_id"],
            df["chrom"],
            df["start"],
            df["opening_energy"],
            df["P_off_target"],
        )
    )


@pytest.fixture
def transcripts(tmp_path: Path) -> dict:
    fasta = tmp_path / "transcripts.fa"
    fasta.write_text(TRANSCRIPTS_FASTA)
    gtf = tmp_path / "transcripts.gtf"
    gtf.write_text(TRANSCRIPTS_GTF)
    return {"fasta": fasta, "gtf": gtf}


class TestParser:
    def test_load_cuts_the_target_at_the_first_whitespace(self, tmp_path):
        df = RIsearchParser().load(_risearch2_tsv(tmp_path / "hits.tsv", TW_HITS))
        assert df["chrom"].to_list() == [_first_word(h[1]) for h in TW_HITS]
        assert df.height == len(TW_HITS)

    def test_load_cuts_at_a_tab_in_a_parquet_name(self, tmp_path):
        path = tmp_path / "hits.parquet"
        _frame([("si1", "chr1\tdna:chromosome", 5, 25, -12.0)]).write_parquet(path)
        assert RIsearchParser().load(path)["chrom"].to_list() == ["chr1"]

    def test_load_leaves_single_word_names_alone(self, tmp_path):
        path = _risearch2_tsv(tmp_path / "hits.tsv", TW_HITS, _first_word)
        df = RIsearchParser().load(path)
        assert df["chrom"].to_list() == [_first_word(h[1]) for h in TW_HITS]

    def test_load_keeps_the_sirna_id_whole(self, tmp_path):
        """The siRNA table and the .summary files are keyed by the full ID."""
        path = tmp_path / "hits.tsv"
        path.write_text("si1 guide A\t1\t21\trna-TX1.1 CDS=10-200\t51\t71\t+\t-20.5\n")
        df = RIsearchParser().load(path)
        assert df["sirna_id"].to_list() == ["si1 guide A"]
        assert df["chrom"].to_list() == ["rna-TX1.1"]

    def test_coerce_predictions_cuts_the_target_too(self):
        df = _coerce_predictions(_frame(TW_HITS))
        assert df["chrom"].to_list() == [_first_word(h[1]) for h in TW_HITS]

    @pytest.mark.parametrize("form", ["file", "frame"])
    def test_two_targets_sharing_a_first_word_are_refused(self, tmp_path, form):
        """Cut, they would be one record: neither the join nor a profile could tell."""
        hits = [
            ("si1", "rna-TX1.1 copy=a", 51, 71, -20.5),
            ("si1", "rna-TX1.1 copy=b", 51, 71, -15.0),
        ]
        with pytest.raises(ValueError) as excinfo:
            if form == "file":
                RIsearchParser().load(_risearch2_tsv(tmp_path / "hits.tsv", hits))
            else:
                _coerce_predictions(_frame(hits))
        message = str(excinfo.value)
        assert "'rna-TX1.1 copy=a'" in message and "'rna-TX1.1 copy=b'" in message
        assert "first word 'rna-TX1.1'" in message


@pytest.mark.parametrize("form", ["file", "frame", "parquet_folder"])
def test_tw_hits_named_by_the_whole_header_are_scored(tmp_path, transcripts, form):
    """Used to score 2 of the 7 hits: the two on the single-word record."""
    common = dict(
        gtf_file=transcripts["gtf"],
        feature_type="transcript",
        predictions_type="tw",
        genome_file=transcripts["fasta"],  # folded in the run, keyed by first word
    )
    control = sioff.off_targets(
        **_source(tmp_path, TW_HITS, form, "first_word"), **common
    )
    assert control.height == len(TW_HITS)
    assert (control["opening_energy"] != UNKNOWN_RECORD_ENERGY).all()

    scored = sioff.off_targets(
        **_source(tmp_path, TW_HITS, form, "whole_header"), **common
    )

    assert scored.height == len(TW_HITS)
    assert (scored["opening_energy"] != UNKNOWN_RECORD_ENERGY).all()
    assert _numbers(scored) == _numbers(control)


@pytest.mark.parametrize("form", ["file", "frame"])
def test_whole_header_hits_get_their_opening_energy(tmp_path, transcripts, form):
    """Without an annotation no hit is dropped; 5 of 7 used to keep the 10.0."""
    common = dict(genome_file=transcripts["fasta"])
    control = sioff.off_targets(
        **_source(tmp_path, TW_HITS, form, "first_word"), **common
    )
    scored = sioff.off_targets(
        **_source(tmp_path, TW_HITS, form, "whole_header"), **common
    )
    assert scored.height == control.height == len(TW_HITS)
    assert (scored["opening_energy"] != UNKNOWN_RECORD_ENERGY).all()
    assert _numbers(scored) == _numbers(control)


@pytest.mark.parametrize("form", ["file", "frame"])
def test_gw_hits_named_by_the_whole_header_are_scored(tmp_path, form):
    """'chr1 some description' never equalled the GTF seqname 'chr1': 0 rows."""
    gtf = tmp_path / "genome.gtf"
    gtf.write_text(GENOME_GTF)
    control = sioff.off_targets(
        **_source(tmp_path, GW_HITS, form, "first_word"), gtf_file=gtf
    )
    assert control.height == len(GW_HITS)

    scored = sioff.off_targets(
        **_source(tmp_path, GW_HITS, form, "whole_header"), gtf_file=gtf
    )

    assert scored.height == len(GW_HITS)
    assert scored["chrom"].unique().to_list() == ["chr1"]
    assert _numbers(scored) == _numbers(control)


def test_profile_key_with_whitespace_is_refused():
    """A hit's target is cut to its first word, so such a key matches no hit.

    It would leave every hit on that record with the opening energy of an
    unknown one, without a message.
    """
    profile = pl.DataFrame({"position": [1], "strand": ["+"], "u1": [0.5]})
    with pytest.raises(ValueError) as excinfo:
        sioff.off_targets(
            predictions=_frame(TW_HITS),
            accessibility={"rna-TX1.1 CDS=10-200": profile, "rna-NC3.1": profile},
        )
    message = str(excinfo.value)
    assert "'rna-TX1.1 CDS=10-200'" in message and "'rna-NC3.1'" not in message
    assert "first word" in message
