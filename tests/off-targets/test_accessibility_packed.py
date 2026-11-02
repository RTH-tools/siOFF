"""Packed accessibility cache: one ``.npy``/``.npz`` pair per strand.

The packed pair holds every record's code rows back to back; each record is
looked up through ``offsets``/``lengths`` in the ``.npz``. The reference for
the packer is the per-record cache of `build_accessibility_cache`, built here
into ``tmp_path`` from the same Parquet files.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from loguru import logger
from typer.testing import CliRunner

import sioff
import sioff.services.accessibility as acc
from sioff.cli import app
from sioff.services.accessibility import (
    AccessibilityError,
    _batch_records,
    _fold_full_chromosome,
    build_accessibility_cache,
    cache_paths,
    ensure_accessibility_cache,
    has_profiles,
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


# --------------------------------------------------------------------------
# A Parquet directory is packed; the pack goes stale with its Parquet.
# --------------------------------------------------------------------------

PACKED_NAMES = sorted(p.name for s in "+-" for p in packed_paths(Path("."), s))


@pytest.fixture
def pq_copy(parquet_records, tmp_path):
    """A private copy of the Parquet fixture directory, safe to pack in place."""
    pdir, seqs = parquet_records
    d = tmp_path / "pq"
    shutil.copytree(pdir, d)
    return d, seqs


def _listing(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


def _pack_names(d: Path, strand: str = "+") -> list[str]:
    with np.load(packed_paths(d, strand)[1]) as side:
        return side["names"].tolist()


def test_ensure_packs_a_parquet_dir(pq_copy):
    d, seqs = pq_copy
    assert ensure_accessibility_cache(d) == sorted(seqs)
    assert _listing(d) == sorted(
        [f"{n}.accessibility.parquet" for n in seqs] + PACKED_NAMES
    )
    assert not any(".access." in n for n in _listing(d))
    assert ensure_accessibility_cache(d) == []


def test_ensure_reuses_a_complete_per_record_cache(pq_copy):
    d, seqs = pq_copy
    for name in seqs:
        build_accessibility_cache(d / f"{name}.accessibility.parquet", d)
    assert ensure_accessibility_cache(d) == []
    assert not any(n.endswith(acc.PACKED_SUFFIX) for n in _listing(d))


def _refold_recB(d):
    rng = np.random.default_rng(99)
    _fold_full_chromosome(
        _random_seq(700, rng),
        "recB",
        60,
        40,
        30,
        str(d / "recB.accessibility.parquet"),
        700,
    )
    return ["recA", "recB", "recC"]


def _delete_recC(d):
    (d / "recC.accessibility.parquet").unlink()
    return ["recA", "recB"]


def _add_recD(d):
    shutil.copy(d / "recA.accessibility.parquet", d / "recD.accessibility.parquet")
    return ["recA", "recB", "recC", "recD"]


@pytest.mark.parametrize("change", [_refold_recB, _delete_recC, _add_recD])
def test_ensure_repacks_when_the_parquet_dir_changes(pq_copy, warnings_seen, change):
    d, seqs = pq_copy
    ensure_accessibility_cache(d)
    later = packed_paths(d, "+")[0].stat().st_mtime + 10
    want = change(d)
    for f in d.glob("*.accessibility.parquet"):
        os.utime(f, (later, later))
    assert ensure_accessibility_cache(d) == want
    assert any("does not match the Parquet profiles" in m for m in warnings_seen)
    assert _pack_names(d) == want
    assert ensure_accessibility_cache(d) == []


def test_mixed_generation_pair_is_repacked(pq_copy, warnings_seen):
    d, seqs = pq_copy
    ensure_accessibility_cache(d)
    # The minus pair is left over from an older pack; the plus pair is fresh.
    npy = packed_paths(d, "-")[0]
    old = npy.stat().st_mtime - 100
    os.utime(npy, (old, old))
    assert ensure_accessibility_cache(d) == sorted(seqs)
    assert any("does not match the Parquet profiles" in m for m in warnings_seen)
    assert ensure_accessibility_cache(d) == []


def test_service_packs_lazily(pq_copy):
    d, seqs = pq_copy
    ref = _per_record_dir(d, ["recB"], d.parent / "rec")
    want = acc.GenomeAccessibilityService(d.parent / "e", cache_dir=ref).query_single(
        "recB", 10, 20, "+"
    )
    got = acc.GenomeAccessibilityService(d).query_single("recB", 10, 20, "+")
    assert got == want
    assert [n for n in _listing(d) if n.endswith((".packed.npy", ".packed.npz"))] == (
        PACKED_NAMES
    )
    assert not any(".access." in n for n in _listing(d))


def test_service_repacks_a_stale_pack_and_serves_the_new_one(pq_copy):
    d, seqs = pq_copy
    ensure_accessibility_cache(d)
    later = packed_paths(d, "+")[0].stat().st_mtime + 10
    _delete_recC(d)
    for f in d.glob("*.accessibility.parquet"):
        os.utime(f, (later, later))
    service = acc.GenomeAccessibilityService(d)
    service.query_single("recB", 10, 20, "+")
    assert _pack_names(d) == ["recA", "recB"]
    assert service._packed_strand("+") is not None
    assert "recB" in service._packed_strand("+").names.tolist()


def test_half_present_pair_is_corrupt(packed_and_reference, tmp_path):
    pdir, seqs, packed, ref = packed_and_reference
    packed_paths(packed, "+")[1].unlink()
    svc = acc.GenomeAccessibilityService(tmp_path / "empty", cache_dir=packed)
    with pytest.raises(AccessibilityError, match="Corrupt packed accessibility cache"):
        svc.query("recA", 0, 5, "+")


def test_has_profiles_sees_a_packed_dir(parquet_records, tmp_path):
    pdir, seqs = parquet_records
    assert not has_profiles(tmp_path)
    pack_parquet_profiles(pdir, tmp_path)
    for f in tmp_path.glob("*.accessibility.parquet"):
        f.unlink()
    assert has_profiles(tmp_path)


def _synthetic_hits(seqs):
    rng = np.random.default_rng(13)
    frames = []
    for name, seq in seqs.items():
        n = 100
        length = rng.integers(5, 25, n)
        start = np.array([rng.integers(1, max(2, len(seq) - int(w))) for w in length])
        frames.append(
            pl.DataFrame(
                {
                    "chrom": [name] * n,
                    "start": start,
                    "end": start + length - 1,
                    "strand": rng.choice(["+", "-"], n),
                    "energy": rng.uniform(-30.0, -8.0, n),
                }
            )
        )
    df = pl.concat(frames).with_columns(
        pl.Series("sirna_id", [f"siRNA{i % 3}" for i in range(300)]),
        pl.col("start").cast(pl.Int32),
        pl.col("end").cast(pl.Int32),
        pl.col("energy").cast(pl.Float32),
    )
    return df.select("sirna_id", "chrom", "start", "end", "strand", "energy")


@pytest.mark.parametrize("form", ["single", "directory"])
def test_off_targets_packed_equals_per_record(pq_copy, tmp_path, form):
    d, seqs = pq_copy
    hits = _synthetic_hits(seqs)
    per_record = _per_record_dir(d, seqs, tmp_path / "rec")
    empty = tmp_path / "empty"
    empty.mkdir()
    keys = ["sirna_id", "chrom", "start", "end", "strand", "energy"]

    def run(acc_dir, cache_dir):
        if form == "single":
            out = sioff.off_targets(
                predictions=hits,
                accessibility_dir=acc_dir,
                accessibility_cache_dir=cache_dir,
            )
        else:
            hits_dir = tmp_path / "hits"
            hits_dir.mkdir(exist_ok=True)
            for (sid,), part in hits.group_by("sirna_id"):
                part.write_parquet(hits_dir / f"risearch_{sid}.parquet")
            out = pl.concat(
                list(
                    sioff.off_targets(
                        risearch_file=hits_dir,
                        accessibility_dir=acc_dir,
                        accessibility_cache_dir=cache_dir,
                        n_workers=2,
                    )
                )
            )
        return out.sort(keys)

    packed = run(d, None)
    assert any(n.endswith(acc.PACKED_SUFFIX) for n in _listing(d))
    reference = run(empty, per_record)
    assert (packed["opening_energy"] != 10.0).any()
    assert (reference["opening_energy"] != 10.0).any()
    assert_frame_equal(packed, reference, check_exact=True)


# --------------------------------------------------------------------------
# `sioff accessibility` folds a FASTA straight into the packed pair.
# --------------------------------------------------------------------------

FOLD_80 = ["-W", "80", "-L", "40", "-u", "30"]


def _write_fasta(path: Path, records) -> Path:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records))
    return path


def _fold_cli(fa: Path, out: Path, *extra: str):
    result = CliRunner().invoke(
        app, ["accessibility", "-f", str(fa), "-o", str(out), *FOLD_80, *extra]
    )
    assert result.exit_code == 0, result.output
    return result


def _record_slices(directory: Path, strand: str) -> dict:
    """``{name: (codes, exception index local to the record, values)}`` of a pack."""
    npy, npz = packed_paths(directory, strand)
    codes = np.load(npy, mmap_mode="r")
    n_u = codes.shape[1]
    slices = {}
    with np.load(npz) as side:
        index, value = side["index"], side["value"]
        for name, off, n in zip(
            side["names"].tolist(), side["offsets"].tolist(), side["lengths"].tolist()
        ):
            lo, hi = np.searchsorted(index, [off * n_u, (off + n) * n_u])
            slices[name] = (
                np.asarray(codes[off : off + n]),
                index[lo:hi] - off * n_u,
                value[lo:hi],
            )
    return slices


def _per_record_slice(rec_dir: Path, name: str, strand: str):
    npy, npz = cache_paths(rec_dir, name, strand)
    with np.load(npz) as side:
        return np.load(npy), side["index"], side["value"]


def _assert_same_record(got, want, label) -> None:
    (codes_g, index_g, value_g), (codes_w, index_w, value_w) = got, want
    assert codes_g.dtype == codes_w.dtype == np.uint8, label
    assert codes_g.shape == codes_w.shape, label
    assert codes_g.tobytes() == codes_w.tobytes(), label
    assert index_g.dtype == index_w.dtype == np.int64, label
    assert value_g.dtype == value_w.dtype == np.float32, label
    np.testing.assert_array_equal(index_g, index_w, err_msg=str(label))
    np.testing.assert_array_equal(value_g, value_w, err_msg=str(label))


def _assert_index(directory: Path, strand: str, names, lengths) -> None:
    with np.load(packed_paths(directory, strand)[1]) as side:
        assert side["names"].tolist() == list(names)
        for key in ("offsets", "lengths", "index"):
            assert side[key].dtype == np.int64, key
        assert side["value"].dtype == np.float32
        assert side["lengths"].tolist() == list(lengths)
        assert (
            side["offsets"].tolist()
            == np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
        )
        assert (np.diff(side["index"]) > 0).all()


def test_cli_writes_four_files_and_two_with_strand_plus(
    parquet_records, tmp_path, plain
):
    _, seqs = parquet_records
    fa = _write_fasta(tmp_path / "multi.fa", seqs.items())
    for extra, strands in (((), "+-"), (("--strand", "+"), "+")):
        out = tmp_path / f"out_{len(strands)}"
        text = plain(_fold_cli(fa, out, *extra).output)
        assert _listing(out) == sorted(
            p.name for s in strands for p in packed_paths(out, s)
        )
        assert "Processed 3 record(s)" in text
        assert "accessibility.plus.packed.npy" in text
        assert "recA" not in text


@pytest.mark.parametrize("workers", [1, 2])
def test_fold_pack_equals_parquet_pack(parquet_records, tmp_path, workers, monkeypatch):
    pdir, seqs = parquet_records
    order = ["recB", "recA", "recC"]
    fa = _write_fasta(tmp_path / "multi.fa", [(n, seqs[n]) for n in order])
    calls: list[int] = []
    if workers > 1:
        # One batch per record, so the process pool really runs and its
        # workers write their r+ slices of the same files concurrently.
        def one_each(sizes, target=0):
            calls.append(len(sizes))
            return [[i] for i in range(len(sizes))]

        monkeypatch.setattr(acc, "_batch_records", one_each)
    out = tmp_path / "fold"
    got = acc.GenomeAccessibilityService(out).compute_genome_accessibility(
        fa, 80, 40, 30, workers=workers, output_format="npy", chunk_size=700
    )
    assert calls == ([3] if workers > 1 else [])
    assert list(got) == order
    assert set(got.values()) == {packed_paths(out, "+")[0]}
    assert _listing(out) == PACKED_NAMES

    ref = tmp_path / "ref"
    pack_parquet_profiles(pdir, ref)
    for strand in "+-":
        _assert_index(out, strand, order, [len(seqs[n]) for n in order])
        folded, packed = _record_slices(out, strand), _record_slices(ref, strand)
        assert folded.keys() == packed.keys()
        for name in order:
            _assert_same_record(folded[name], packed[name], (name, strand))
        assert folded["recB"][1].size > 0, strand  # the hairpins' exceptions


def test_zero_length_record_is_indexed_and_not_folded(parquet_records, tmp_path):
    pdir, seqs = parquet_records
    # recB's exceptions sit at a nonzero offset, after recC and the empty record.
    order = ["recC", "empty", "recB"]
    fa = tmp_path / "gap.fa"
    fa.write_text(f">recC\n{seqs['recC']}\n>empty\n>recB\n{seqs['recB']}\n")
    out = tmp_path / "fold"
    got = acc.GenomeAccessibilityService(out).compute_genome_accessibility(
        fa, 80, 40, 30, output_format="npy", chunk_size=700
    )
    assert list(got) == order
    ref = tmp_path / "ref"
    pack_parquet_profiles(pdir, ref)
    for strand in "+-":
        _assert_index(out, strand, order, [25, 0, len(seqs["recB"])])
        folded, packed = _record_slices(out, strand), _record_slices(ref, strand)
        assert folded["empty"][0].shape == (0, 30)
        for name in ("recC", "recB"):
            _assert_same_record(folded[name], packed[name], (name, strand))
        assert folded["recB"][1].size > 0, strand


def test_empty_fasta_writes_no_pack(tmp_path):
    fa = tmp_path / "empty.fa"
    fa.write_text("")
    out = tmp_path / "out"
    with pytest.raises(AccessibilityError, match="No records in"):
        acc.GenomeAccessibilityService(out).compute_genome_accessibility(
            fa, 40, 20, 10, output_format="npy"
        )
    assert not any(out.iterdir())


@pytest.mark.parametrize("output_format", ["npy", "parquet"])
def test_duplicate_record_ids_are_refused(tmp_path, output_format, plain):
    seq = _random_seq(60, np.random.default_rng(6))
    fa = _write_fasta(
        tmp_path / "dup.fa", [("dupX", seq), ("ok", seq), ("dupX second copy", seq)]
    )
    out = tmp_path / "out"
    with pytest.raises(AccessibilityError, match=r"Duplicate record IDs in .*: dupX$"):
        acc.GenomeAccessibilityService(out).compute_genome_accessibility(
            fa, 40, 20, 10, output_format=output_format
        )
    assert not any(out.iterdir())

    many = _write_fasta(
        tmp_path / "many.fa", [(f"d{i:02d}", seq) for i in range(12)] * 2
    )
    with pytest.raises(AccessibilityError) as err:
        acc.GenomeAccessibilityService(out).compute_genome_accessibility(
            many, 40, 20, 10, output_format=output_format
        )
    assert str(err.value).endswith(
        ": " + ", ".join(f"d{i:02d}" for i in range(10)) + "…"
    )
    assert not any(out.iterdir())

    cli_out = tmp_path / "cli"
    result = CliRunner().invoke(
        app,
        ["accessibility", "-f", str(fa), "-o", str(cli_out), "-W", "40", "-L", "20"]
        + ["-u", "10", "--output-format", output_format],
    )
    assert result.exit_code == 1
    assert "dupX" in plain(result.output)
    assert not cli_out.exists() or not any(cli_out.iterdir())


def test_strand_plus_rerun_removes_a_stale_minus_pair(
    parquet_records, tmp_path, monkeypatch
):
    _, seqs = parquet_records
    fa = _write_fasta(tmp_path / "multi.fa", seqs.items())
    out = tmp_path / "out"
    _fold_cli(fa, out)
    assert _listing(out) == PACKED_NAMES
    _fold_cli(fa, out, "--strand", "+")
    assert _listing(out) == sorted(p.name for p in packed_paths(out, "+"))

    # A CLI run resets loguru's handlers, so start capturing after it.
    monkeypatch.setattr(acc, "_WARNED_MISSING_STRANDS", set())
    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    try:
        got = _annotate(
            acc.GenomeAccessibilityService(out), "recA", [("+", 50, 60), ("-", 50, 60)]
        )
    finally:
        logger.remove(handler)
    assert got[0] != 10.0 and got[1] == 10.0
    assert [m for m in seen if "no '-' strand" in m], seen


def test_record_ids_with_pipes_and_slashes(tmp_path):
    seq = _random_seq(300, np.random.default_rng(11))
    names = ["ENST0001.1|ENSG0001.1|tx", "a/b", "plain"]
    fa = tmp_path / "ids.fa"
    fa.write_text(
        f">{names[0]} gene=x\n{seq}\n>{names[1]}\n{seq}\n>{names[2]}\n{seq}\n"
    )
    out = tmp_path / "out"
    got = acc.GenomeAccessibilityService(out).compute_genome_accessibility(
        fa, 40, 20, 10, output_format="npy"
    )
    assert list(got) == names
    assert _listing(out) == PACKED_NAMES
    reader = acc.GenomeAccessibilityService(out)
    for strand in "+-":
        for start, end in ((100, 115), (10, 18), (280, 300)):
            want = reader.query_single("plain", start, end, strand)
            assert want != 10.0
            for name in names[:2]:
                assert reader.query_single(name, start, end, strand) == want


def test_many_short_records(tmp_path):
    rng = np.random.default_rng(21)
    records = [
        (f"t{i:03d}", _random_seq(int(rng.integers(30, 121)), rng)) for i in range(300)
    ]
    names = [n for n, _ in records]
    lengths = [len(s) for _, s in records]
    fa = _write_fasta(tmp_path / "many.fa", records)
    advances: list[int] = []
    descriptions: list[str] = []

    def callback(advance, description):
        advances.append(advance)
        descriptions.append(description)

    out = tmp_path / "out"
    got = acc.GenomeAccessibilityService(out).compute_genome_accessibility(
        fa,
        40,
        20,
        10,
        workers=2,
        output_format="npy",
        progress_callback=callback,
    )
    assert list(got) == names
    assert _listing(out) == PACKED_NAMES
    assert sum(advances) == 300
    assert len(advances) == len(_batch_records(lengths)) <= 2
    assert descriptions[-1] == "Folded 300/300 records"

    slices = {s: _record_slices(out, s) for s in "+-"}
    for strand in "+-":
        _assert_index(out, strand, names, lengths)
    for i in rng.choice(300, 20, replace=False):
        name, seq = records[int(i)]
        rec = tmp_path / "rec"
        _fold_full_chromosome(seq, name, 40, 20, 10, str(rec), output_format="npy")
        for strand in "+-":
            _assert_same_record(
                slices[strand][name],
                _per_record_slice(rec, name, strand),
                (name, strand),
            )


def test_empty_pack_keeps_the_default(tmp_path: Path) -> None:
    """A pack with no records answers every hit with the 10.0 default."""
    for strand in "+-":
        npy, npz = packed_paths(tmp_path, strand)
        np.save(npy, np.zeros((0, 30), dtype=np.uint8))
        np.savez(
            npz,
            names=np.array([], dtype="U1"),
            offsets=np.array([], dtype=np.int64),
            lengths=np.array([], dtype=np.int64),
            index=np.array([], dtype=np.int64),
            value=np.array([], dtype=np.float32),
        )
    service = acc.GenomeAccessibilityService(tmp_path)
    hits = pl.DataFrame(
        {
            "chrom": ["chrA", "chrB", "onTarget"],
            "start": [10, 5, 1],
            "end": [30, 25, 20],
            "strand": ["+", "-", "+"],
        }
    )
    out = service.annotate_opening_energy_vectorized(hits)
    assert out["opening_energy"].to_list() == [10.0, 10.0, 10.0]
