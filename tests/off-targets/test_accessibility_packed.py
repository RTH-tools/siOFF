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
                held += os.readlink(fd).endswith(".accessibility.parquet")
            except OSError:
                pass
        peak = max(peak, held)
        return real(*args, **kwargs)

    monkeypatch.setattr(acc, "_fill_strand", watching)
    assert len(pack_parquet_profiles(pdir, tmp_path / "packed", workers=1)) == 50
    # The open file plus its batch reader; one per record would reach 50.
    assert peak <= 3
