"""Hits that keep the default opening energy are reported.

``GenomeAccessibilityService.annotate_opening_energy_vectorized`` leaves its
10.0 kcal/mol default for a hit on a sequence without a profile and for a hit
outside its sequence's profile. That divides the hit's Boltzmann weight by
about 1e7, so the site all but leaves the partition function, and nothing
used to say so (a missing *strand* was warned about, a missing *record* not).

Each worker scores one siRNA, so the counts travel in the siRNA's ``meta``
and the parent warns once after the last siRNA; a run in which no hit at all
has a profile ends with an error. No number changes.
"""

import math
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pytest
from loguru import logger

import sioff
from sioff.core.off_targets import compute_off_targets
from sioff.services.accessibility import (
    AccessibilityError,
    GenomeAccessibilityService,
    build_accessibility_cache,
)

# Three hand-written 120-nt records.
RECORDS = {
    "recA": (
        "GGCATCGATTACGGCTAGCTTAGGCATCGATCGGATTACGCTAGCATCGGACTAGCTAGA"
        "TCGGATACGCTTAGCGGCGCGCCGCTAAGCGATATCGATTAGCTAGGCTAACGTATCGGA"
    ),
    "recB": (
        "TTAGCGATCGATCGGCTAAGCTAGCTTTAGCGAGCTAGCTAGGATCGATTTAGCATGCAT"
        "GCATGCTAGCTAAAGCTAGCTCGCTAAAGCTAGCTTAGCCGATCGATCGCTAACGATCGA"
    ),
    "recC": (
        "ACGTACGATCGTTAGCAAGCTTGCAACGATCGGGCCCGATCGTTGCAAGCTTGCTAACGA"
        "TCGTACGTAGCTAGGGATCCCTAGCTACGTACGATATCGCGATATCGATTTAAATCGATA"
    ),
}
RT_37 = 0.001987 * (37.0 + 273.15)

# (sirna_id, chrom, start, end, strand, energy); 1-based inclusive, as RIsearch.
HITS = [
    ("siRNA_1", "recA", 100, 120, "+", -20.0),  # profile exists
    ("siRNA_1", "recB", 43, 63, "+", -20.0),  # profile exists
    ("siRNA_1", "recC", 93, 113, "+", -20.0),  # no profile for recC
    ("siRNA_1", "recC", 74, 94, "+", -20.0),  # no profile for recC
    ("siRNA_1", "recA", 7, 27, "-", -18.0),  # profile exists
    ("siRNA_2", "recA", 1, 21, "+", -19.0),  # a siRNA without any recC hit
    ("siRNA_2", "recB", 31, 51, "+", -17.0),
    ("siRNA_3", "recC", 56, 76, "+", -16.0),  # a siRNA with recC hits only
    ("siRNA_3", "recC", 93, 113, "-", -15.0),
]


def _frame(rows) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["sirna_id", "chrom", "start", "end", "strand", "energy"],
        orient="row",
    )


def _fasta(path: Path, records: dict[str, str]) -> Path:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records.items()))
    return path


def _fold(fasta: Path, out_dir: Path, output_format: str) -> Path:
    """What ``sioff accessibility -f fasta -o out_dir --output-format ...`` writes."""
    GenomeAccessibilityService(out_dir).compute_genome_accessibility(
        fasta, output_format=output_format
    )
    return out_dir


def _profiles(kind: str, tmp_path: Path, records: dict[str, str]) -> Path:
    """An ``-a`` directory holding profiles for ``records``, in one of three layouts."""
    fasta = _fasta(tmp_path / "fold.fa", records)
    if kind == "packed":  # the default output of `sioff accessibility`
        return _fold(fasta, tmp_path / "acc_packed", "npy")
    if kind == "parquet":  # `sioff accessibility --output-format parquet`
        return _fold(fasta, tmp_path / "acc_parquet", "parquet")
    if kind == "per_record_cache":  # the v0.2.0 <record>_<strand>.access.npy layout
        source = _fold(fasta, tmp_path / "acc_parquet_src", "parquet")
        cache = tmp_path / "acc_per_record"
        for parquet in sorted(source.glob("*.accessibility.parquet")):
            build_accessibility_cache(parquet, cache)
        return cache
    raise ValueError(kind)


@contextmanager
def _warnings():
    """WARNING (and higher) loguru messages logged in this process inside the block."""
    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    try:
        yield seen
    finally:
        logger.remove(handler)


def _two_record_profiles(kind: str, tmp_path: Path) -> Path:
    return _profiles(kind, tmp_path, {n: RECORDS[n] for n in ("recA", "recB")})


def _assert_recc_got_the_default(df: pl.DataFrame) -> None:
    """Precondition: recC was scored with 10.0, recA/recB with their own energies."""
    recc = df.filter(pl.col("chrom") == "recC")
    known = df.filter(pl.col("chrom") != "recC")
    assert recc.height == 4, df
    assert recc["opening_energy"].to_list() == [10.0] * 4, recc
    assert known.height == 5, df
    assert (known["opening_energy"] != 10.0).all(), known


# --- the report ---


@pytest.mark.parametrize("kind", ["packed", "parquet", "per_record_cache"])
def test_unknown_record_is_reported(kind: str, tmp_path: Path) -> None:
    """A run with hits on a record without a profile warns and names the record.

    ``packed`` and ``parquet`` reach ``_annotate_packed`` (the engine packs a
    Parquet directory before scoring); ``per_record_cache`` reaches the
    per-group loop and its ``except ProfileNotFoundError: continue``.
    """
    acc_dir = _two_record_profiles(kind, tmp_path)
    with _warnings() as seen:
        df = sioff.off_targets(
            predictions=_frame(HITS), accessibility_dir=acc_dir, n_workers=1
        )
    _assert_recc_got_the_default(df)
    (message,) = [m for m in seen if "recC" in m]
    assert message.startswith("4 of 9 scored hits (2 siRNA(s)) are on sequences")
    assert "10.0 kcal/mol" in message


def test_unknown_record_is_reported_for_in_memory_profiles(tmp_path: Path) -> None:
    """The same through ``sioff.accessibility`` -> ``off_targets(accessibility=...)``."""
    fasta = _fasta(tmp_path / "fold.fa", {n: RECORDS[n] for n in ("recA", "recB")})
    profiles = sioff.accessibility(genome=fasta)
    assert sorted(profiles) == ["recA", "recB"]
    with _warnings() as seen:
        df = sioff.off_targets(
            predictions=_frame(HITS), accessibility=profiles, n_workers=1
        )
    _assert_recc_got_the_default(df)
    assert any("recC" in message for message in seen), (
        f"no WARNING names recC. WARNING messages logged: {seen}"
    )


def test_unknown_record_is_reported_by_the_parent_when_a_pool_scores(
    tmp_path: Path,
) -> None:
    """With a worker pool the report must come from the parent process.

    Spawned workers log to their own stderr sink (``_init_worker``), which the
    caller's handlers never see, and each worker knows only its own siRNAs.
    """
    acc_dir = _two_record_profiles("packed", tmp_path)
    with _warnings() as seen:
        df = sioff.off_targets(
            predictions=_frame(HITS), accessibility_dir=acc_dir, n_workers=2
        )
    _assert_recc_got_the_default(df)
    assert any("recC" in message for message in seen), (
        f"no WARNING in the parent names recC. WARNING messages logged: {seen}"
    )


def test_run_whose_hits_all_lack_a_profile_is_an_error(tmp_path: Path) -> None:
    """Profiles ``chrUn1``/``chrUn2``, every hit on ``Un1``/``Un2``.

    The names of the profiles and of the predictions do not match at all:
    every opening energy would be the default, which is no accessibility. It
    used to succeed silently with 10.0 on every row.
    """
    acc_dir = _profiles(
        "packed", tmp_path, {"chrUn1": RECORDS["recA"], "chrUn2": RECORDS["recB"]}
    )
    hits = _frame(
        [
            ("siRNA_1", "Un1", 100, 120, "+", -20.0),
            ("siRNA_1", "Un2", 43, 63, "+", -19.0),
            ("siRNA_2", "Un1", 1, 21, "+", -19.0),
        ]
    )
    with pytest.raises(AccessibilityError) as excinfo:
        sioff.off_targets(predictions=hits, accessibility_dir=acc_dir, n_workers=1)
    message = str(excinfo.value)
    assert "None of the 3 scored hits" in message
    assert "Un1, Un2" in message


def test_the_stream_yields_every_sirna_before_that_error(tmp_path: Path) -> None:
    """Like a worker failure: raised after the last siRNA, results kept."""
    acc_dir = _profiles("packed", tmp_path, {"chrUn1": RECORDS["recA"]})
    hits = _frame([("siRNA_1", "Un1", 100, 120, "+", -20.0)])
    seen = []
    with pytest.raises(AccessibilityError):
        for df, _meta in compute_off_targets(
            predictions=hits, accessibility_dir=acc_dir
        ):
            seen.append(df)
    assert [df["opening_energy"].to_list() for df in seen] == [[10.0]]


def test_out_of_bounds_hit_is_reported(tmp_path: Path) -> None:
    """A hit ending past its record's profile keeps 10.0: say so, naming the record."""
    acc_dir = _two_record_profiles("packed", tmp_path)
    n = len(RECORDS["recA"])
    hits = _frame(
        [
            ("siRNA_1", "recA", n - 20, n, "+", -20.0),  # ends at the last position
            ("siRNA_1", "recA", n - 19, n + 1, "+", -20.0),  # one past the end
        ]
    )
    with _warnings() as seen:
        df = sioff.off_targets(predictions=hits, accessibility_dir=acc_dir, n_workers=1)
    by_end = dict(zip(df["end"].to_list(), df["opening_energy"].to_list()))
    assert by_end[n] != 10.0 and by_end[n + 1] == 10.0, df
    assert any("recA" in message for message in seen), (
        f"the hit recA:{n - 19}-{n + 1} lies outside the {n}-nt profile of recA and "
        "was scored with the default opening energy 10.0 kcal/mol, but no WARNING "
        f"names recA. WARNING messages logged: {seen}"
    )


# --- what the report must leave alone ---


def test_pin_default_is_10_kcal_and_scales_the_weight_by_exp_minus_10_over_rt(
    tmp_path: Path,
) -> None:
    """The numbers a report-only fix must leave alone."""
    acc_dir = _two_record_profiles("packed", tmp_path)
    df = sioff.off_targets(
        predictions=_frame(HITS),
        accessibility_dir=acc_dir,
        n_workers=1,
        detailed_report=True,
    )
    _assert_recc_got_the_default(df)
    recc = df.filter(pl.col("chrom") == "recC")
    ratio = (recc["boltzmann_weight"] / recc["boltzmann_weight_noacc"]).to_list()
    assert ratio == pytest.approx([math.exp(-10.0 / RT_37)] * 4, rel=1e-12)
    # recC:93-113 has the same hybridisation energy as recA:100-120 (-20.0), yet
    # its probability is below 1e-5 of that hit's.
    p = {
        (r["chrom"], r["start"]): r["P_off_target"]
        for r in df.filter(pl.col("sirna_id") == "siRNA_1").to_dicts()
    }
    assert p[("recC", 93)] / p[("recA", 100)] < 1e-5


def test_pin_a_run_whose_hits_all_have_a_profile_stays_quiet(tmp_path: Path) -> None:
    """Guard for the fix: no warning when nothing kept the default."""
    acc_dir = _profiles("packed", tmp_path, RECORDS)
    with _warnings() as seen:
        df = sioff.off_targets(
            predictions=_frame(HITS), accessibility_dir=acc_dir, n_workers=1
        )
    assert df.height == 9
    assert (df["opening_energy"] != 10.0).all(), df
    assert seen == []
