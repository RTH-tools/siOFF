"""Packed accessibility cache: one ``.npy``/``.npz`` pair per strand.

The packed pair holds every record's code rows back to back; each record is
looked up through ``offsets``/``lengths`` in the ``.npz``. The reference for
the packer is the per-record cache of `build_accessibility_cache`, built here
into ``tmp_path`` from the same Parquet files.
"""

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from loguru import logger

import sioff.services.accessibility as acc
from sioff.services.accessibility import (
    AccessibilityError,
    _batch_records,
    _fold_full_chromosome,
    build_accessibility_cache,
    cache_paths,
    pack_parquet_profiles,
    packed_paths,
)

HAIRPIN = "CGGGGCCCGGGGCGGGCCGAAAGGCCCGCCCCGGGCCCCG" * 3


def _random_seq(n: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("ACGT"), n))


@pytest.fixture(scope="module")
def parquet_records(tmp_path_factory):
    """Three folded records: one with exceptions, one plain, one shorter than W."""
    rng = np.random.default_rng(5)
    seqs = {
        "recA": _random_seq(1600, rng),
        "recB": _random_seq(300, rng) + HAIRPIN + _random_seq(300, rng),
        "recC": _random_seq(25, rng),
    }
    pdir = tmp_path_factory.mktemp("packed_parquets")
    for name, seq in seqs.items():
        _fold_full_chromosome(
            seq, name, 80, 40, 30, str(pdir / f"{name}.accessibility.parquet"), 700
        )
    return pdir, seqs


@pytest.mark.parametrize("workers", [1, 2])
def test_pack_from_parquet_equals_per_record_caches(
    parquet_records, tmp_path, workers, monkeypatch
):
    pdir, seqs = parquet_records
    calls: list[int] = []
    if workers > 1:
        # One batch per record, so the spawned process pool really runs.
        def one_each(sizes, target=0):
            calls.append(len(sizes))
            return [[i] for i in range(len(sizes))]

        monkeypatch.setattr(acc, "_batch_records", one_each)
    names = pack_parquet_profiles(pdir, tmp_path / "packed", workers=workers)
    assert names == sorted(seqs)
    assert bool(calls) == (workers > 1)
    assert sorted(p.name for p in (tmp_path / "packed").iterdir()) == sorted(
        p.name for s in "+-" for p in packed_paths(tmp_path / "packed", s)
    )
    for strand in "+-":
        npy, npz = packed_paths(tmp_path / "packed", strand)
        codes = np.load(npy, mmap_mode="r")
        with np.load(npz) as side:
            assert side["names"].tolist() == names
            n_u = codes.shape[1]
            for i, name in enumerate(names):
                build_accessibility_cache(
                    pdir / f"{name}.accessibility.parquet", tmp_path / "rec"
                )
                rec_npy, rec_npz = cache_paths(tmp_path / "rec", name, strand)
                off, n = int(side["offsets"][i]), int(side["lengths"][i])
                np.testing.assert_array_equal(codes[off : off + n], np.load(rec_npy))
                lo, hi = np.searchsorted(side["index"], [off * n_u, (off + n) * n_u])
                with np.load(rec_npz) as rec:
                    np.testing.assert_array_equal(
                        side["index"][lo:hi] - off * n_u, rec["index"]
                    )
                    np.testing.assert_array_equal(side["value"][lo:hi], rec["value"])
            assert side["index"].size > 0  # recB's hairpins
            i_b = names.index("recB")
            off, n = int(side["offsets"][i_b]), int(side["lengths"][i_b])
            assert off > 0
            lo, hi = np.searchsorted(side["index"], [off * n_u, (off + n) * n_u])
            assert hi > lo, strand


def test_strand_absent_from_every_parquet_gets_no_pair(tmp_path):
    rng = np.random.default_rng(1)
    pdir = tmp_path / "pq"
    pdir.mkdir()
    _fold_full_chromosome(
        _random_seq(400, rng),
        "rec",
        80,
        40,
        30,
        str(pdir / "rec.accessibility.parquet"),
        700,
        strand="+",
    )
    out = tmp_path / "packed"
    out.mkdir()
    stale_npy, stale_npz = packed_paths(out, "-")
    stale_npy.write_bytes(b"old")
    stale_npz.write_bytes(b"old")
    assert pack_parquet_profiles(pdir, out) == ["rec"]
    assert sorted(p.name for p in out.iterdir()) == sorted(
        p.name for p in packed_paths(out, "+")
    )
    with np.load(packed_paths(out, "+")[1]) as side:
        assert side["lengths"].tolist() == [400]


def test_unequal_strand_lengths_are_indexed_per_strand_and_warned(tmp_path):
    rng = np.random.default_rng(2)
    n = 500
    full = tmp_path / "full.accessibility.parquet"
    _fold_full_chromosome(_random_seq(n, rng), "rec", 80, 40, 30, str(full), 700)
    pdir = tmp_path / "pq"
    pdir.mkdir()
    df = pl.read_parquet(full)
    df = df.filter(~((pl.col("strand") == "-") & (pl.col("position") == n)))
    df.write_parquet(pdir / "rec.accessibility.parquet")
    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    try:
        pack_parquet_profiles(pdir, tmp_path / "packed")
    finally:
        logger.remove(handler)
    for strand, expected in (("+", n), ("-", n - 1)):
        with np.load(packed_paths(tmp_path / "packed", strand)[1]) as side:
            assert side["lengths"].tolist() == [expected]
    assert any("'+' covers" in m for m in seen)


def test_batch_records_groups_short_records_and_isolates_long_ones():
    sizes = [5, 3_000_000, 4, 1_999_999, 1]
    batches = _batch_records(sizes, target=2_000_000)
    assert batches == [[1], [3], [0, 2, 4]]
    assert sorted(i for b in batches for i in b) == list(range(len(sizes)))
    assert _batch_records([], target=10) == []
    many = _batch_records([7, 3, 9, 2, 6, 4], target=10)
    assert sorted(i for b in many for i in b) == list(range(6))
    assert all(sum([7, 3, 9, 2, 6, 4][i] for i in b) <= 10 for b in many)


def test_mismatched_u_columns_are_refused(tmp_path):
    rng = np.random.default_rng(3)
    pdir = tmp_path / "pq"
    pdir.mkdir()
    for name, n_u in (("a", 10), ("b", 30)):
        _fold_full_chromosome(
            _random_seq(300, rng),
            name,
            80,
            40,
            n_u,
            str(pdir / f"{name}.accessibility.parquet"),
            700,
        )
    with pytest.raises(AccessibilityError, match="b.accessibility.parquet"):
        pack_parquet_profiles(pdir, tmp_path / "packed")
    assert not (tmp_path / "packed").exists() or not any(
        (tmp_path / "packed").iterdir()
    )


def test_packed_paths_rejects_unknown_strand(tmp_path):
    with pytest.raises(AccessibilityError):
        packed_paths(tmp_path, "x")


def test_packing_closes_each_parquet_before_the_next(tmp_path, monkeypatch):
    """A batch of many records must not hold one open descriptor per record."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        pytest.skip("needs /proc")
    rng = np.random.default_rng(4)
    pdir = tmp_path / "pq"
    pdir.mkdir()
    for i in range(50):
        _fold_full_chromosome(
            _random_seq(100, rng),
            f"r{i:02d}",
            40,
            20,
            10,
            str(pdir / f"r{i:02d}.accessibility.parquet"),
            700,
        )
    peak = 0
    real = acc._fill_strand

    def watching(*args, **kwargs):
        nonlocal peak
        held = 0
        for fd in fd_dir.iterdir():
            try:
                held += os.readlink(fd).startswith(str(pdir) + os.sep)
            except OSError:
                pass
        peak = max(peak, held)
        return real(*args, **kwargs)

    monkeypatch.setattr(acc, "_fill_strand", watching)
    assert len(pack_parquet_profiles(pdir, tmp_path / "packed", workers=1)) == 50
    # The open file plus its batch reader; one per record would reach 50.
    assert peak <= 3


# --------------------------------------------------------------------------
# Reading the packed cache
# --------------------------------------------------------------------------


def _per_record_dir(pdir, names, target):
    for name in names:
        build_accessibility_cache(pdir / f"{name}.accessibility.parquet", target)
    return target


def _frame(chrom, hits):
    return pl.DataFrame(
        {
            "chrom": [chrom] * len(hits),
            "start": [h[1] for h in hits],
            "end": [h[2] for h in hits],
            "strand": [h[0] for h in hits],
        },
        schema={
            "chrom": pl.Utf8,
            "start": pl.Int32,
            "end": pl.Int32,
            "strand": pl.Utf8,
        },
    )


def _annotate(service, chrom, hits):
    out = service.annotate_opening_energy_vectorized(_frame(chrom, hits))
    return out["opening_energy"].to_numpy()


@pytest.fixture
def packed_and_reference(parquet_records, tmp_path):
    pdir, seqs = parquet_records
    packed = tmp_path / "packed"
    pack_parquet_profiles(pdir, packed)
    ref = _per_record_dir(pdir, seqs, tmp_path / "rec")
    return pdir, seqs, packed, ref


def test_packed_lookup_equals_per_record_lookup(packed_and_reference, tmp_path):
    pdir, seqs, packed, ref = packed_and_reference
    empty = tmp_path / "empty"
    svc_p = acc.GenomeAccessibilityService(empty, cache_dir=packed)
    svc_r = acc.GenomeAccessibilityService(empty, cache_dir=ref)
    rng = np.random.default_rng(9)
    for name, seq in seqs.items():
        n = len(seq)
        hits = []
        for strand in "+-":
            for _ in range(200):
                length = int(rng.integers(1, 41))
                if length > n:
                    length = n
                s = int(rng.integers(1, n - length + 2))
                hits.append((strand, s, s + length - 1))
            hits += [
                (strand, 1, 1),
                (strand, n, n),
                (strand, 1, min(n, 40)),
                (strand, max(1, n - 39), n),
                (strand, 0, 5),
                (strand, n - 2, n + 3),
            ]
        got = _annotate(svc_p, name, hits)
        want = _annotate(svc_r, name, hits)
        np.testing.assert_array_equal(got, want)
        if n > 40:
            assert (got != 10.0).any()
        single = np.array(
            [svc_p.query_single(name, s, e, st) for st, s, e in hits], dtype=np.float32
        )
        single_ref = np.array(
            [svc_r.query_single(name, s, e, st) for st, s, e in hits],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(single, single_ref)
        for strand in "+-":
            np.testing.assert_array_equal(
                svc_p.query(name, 0, n, strand), svc_r.query(name, 0, n, strand)
            )


def test_hit_past_a_record_end_does_not_read_the_next_record(
    packed_and_reference, tmp_path
):
    pdir, seqs, packed, ref = packed_and_reference
    empty = tmp_path / "empty"
    svc_p = acc.GenomeAccessibilityService(empty, cache_dir=packed)
    svc_r = acc.GenomeAccessibilityService(empty, cache_dir=ref)
    n_a = len(seqs["recA"])
    hits = [("+", n_a - 4, n_a), ("+", n_a - 3, n_a + 1), ("-", 1, 5)]
    got = _annotate(svc_p, "recA", hits)
    want = _annotate(svc_r, "recA", hits)
    assert got[0] == want[0] != 10.0
    assert got[1] == 10.0
    assert got[2] == want[2] != 10.0


def test_unknown_chrom_keeps_default_and_logs_nothing(
    packed_and_reference, tmp_path, warnings_seen
):
    _, _, packed, _ = packed_and_reference
    svc = acc.GenomeAccessibilityService(tmp_path / "empty", cache_dir=packed)
    got = _annotate(svc, "nope", [("+", 5, 20), ("-", 5, 20)])
    assert (got == 10.0).all()
    assert not warnings_seen


@pytest.fixture
def warnings_seen(monkeypatch):
    monkeypatch.setattr(acc, "_WARNED_MISSING_STRANDS", set())
    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    yield seen
    logger.remove(handler)


def test_missing_minus_pair_warns_once_and_keeps_default(
    parquet_records, tmp_path, warnings_seen
):
    pdir, seqs = parquet_records
    plus_dir = tmp_path / "plus_parquet"
    plus_dir.mkdir()
    for name, seq in seqs.items():
        _fold_full_chromosome(
            seq,
            name,
            80,
            40,
            30,
            str(plus_dir / f"{name}.accessibility.parquet"),
            700,
            strand="+",
        )
    packed = tmp_path / "packed"
    pack_parquet_profiles(plus_dir, packed)
    assert not packed_paths(packed, "-")[0].exists()
    svc = acc.GenomeAccessibilityService(tmp_path / "empty", cache_dir=packed)
    hits = [("+", 50, 60), ("-", 50, 60)]
    got = _annotate(svc, "recA", hits)
    assert got[0] != 10.0 and got[1] == 10.0
    _annotate(svc, "recA", hits)
    missing = [m for m in warnings_seen if "no '-' strand" in m]
    assert len(missing) == 1, warnings_seen


def test_packed_cache_wins_over_per_record_files_in_the_same_dir(
    packed_and_reference, tmp_path
):
    pdir, seqs, packed, ref = packed_and_reference
    both = tmp_path / "both"
    pack_parquet_profiles(pdir, both)
    for name in seqs:
        for strand in "+-":
            npy, _ = cache_paths(ref, name, strand)
            codes = np.load(npy)
            np.save(cache_paths(both, name, strand)[0], np.zeros_like(codes))
            (cache_paths(both, name, strand)[1]).write_bytes(
                cache_paths(ref, name, strand)[1].read_bytes()
            )
    hits = [("+", 100, 120), ("-", 100, 120), ("+", 300, 340)]
    want = _annotate(
        acc.GenomeAccessibilityService(tmp_path / "e1", cache_dir=packed), "recA", hits
    )
    got = _annotate(
        acc.GenomeAccessibilityService(tmp_path / "e2", cache_dir=both), "recA", hits
    )
    np.testing.assert_array_equal(got, want)
    assert (got != 10.0).any() and (got != 0.0).any()


def test_cache_dir_pack_is_read_before_profile_dir_pack(parquet_records, tmp_path):
    pdir, seqs = parquet_records
    right = tmp_path / "right"
    pack_parquet_profiles(pdir, right)
    other_src = tmp_path / "other_parquet"
    other_src.mkdir()
    _fold_full_chromosome(
        _random_seq(400, np.random.default_rng(77)),
        "recA",
        80,
        40,
        30,
        str(other_src / "recA.accessibility.parquet"),
        700,
    )
    wrong = tmp_path / "wrong"
    pack_parquet_profiles(other_src, wrong)
    hits = [("+", 100, 120), ("-", 100, 120)]
    got = _annotate(
        acc.GenomeAccessibilityService(wrong, cache_dir=right), "recA", hits
    )
    want = _annotate(
        acc.GenomeAccessibilityService(tmp_path / "e", cache_dir=right), "recA", hits
    )
    other = _annotate(
        acc.GenomeAccessibilityService(tmp_path / "e2", cache_dir=wrong), "recA", hits
    )
    np.testing.assert_array_equal(got, want)
    assert not np.array_equal(got, other)


def test_corrupt_packed_index_is_reported(packed_and_reference, tmp_path):
    pdir, seqs, packed, ref = packed_and_reference
    np.savez(packed_paths(packed, "+")[1], names=np.array(["x"]))
    svc = acc.GenomeAccessibilityService(tmp_path / "empty", cache_dir=packed)
    with pytest.raises(AccessibilityError, match="Corrupt packed accessibility cache"):
        svc.query("recA", 0, 5, "+")


def test_site_cells_accepts_per_hit_lengths():
    rows, cols, ok = acc.site_cells("-", [1, 1], [5, 5], np.array([10, 20]), 30)
    assert rows.tolist() == [9, 19]
    for i, n in enumerate((10, 20)):
        r1, c1, o1 = acc.site_cells("-", [1], [5], n, 30)
        assert (rows[i], cols[i], ok[i]) == (r1[0], c1[0], o1[0])
