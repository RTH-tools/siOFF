"""The memory-mapped accessibility cache answers exactly like the eager path.

`GenomeAccessibilityService` reads, by default (``cache_dir=None``), a whole
chromosome's Parquet and keeps a dense float32 copy per strand (chr1: 30 GB,
one private copy per worker process). With ``cache_dir`` it instead serves a
compact uint8 cache (``<chrom>_<plus|minus>.access.npy``, value
``round(E * 10)``) that every worker memory-maps, so the OS page cache holds
one shared copy. Entries that the code cannot represent exactly (opening
energies above 25.5, negative or unrounded values) live in a small sorted
sidecar, which keeps the cache lossless for any input. Only that memory-mapped
mode (``--accessibility-strategy memmap``) ever writes cache files.

The reference for every test here is the frame-backed service: it runs the
very code the Parquet path ran before the cache existed (filter one strand,
materialise a dense float32 array), so equality against it is equality against
the old eager path.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sioff.services.accessibility import (
    AccessibilityError,
    GenomeAccessibilityService,
    cache_paths,
    ensure_accessibility_cache,
)

N_U = 30
CONTIGS = {"chrA": 700, "chrB": 300}


def _synthetic_frames(seed: int = 7) -> dict[str, pl.DataFrame]:
    """Two contigs, both strands, written-like values (``round(E*10)/10`` as float32).

    Includes the sentinels (25.5 unfoldable, 10.0 folding error), opening
    energies above 25.5 (which exist in real data and do not fit a uint8 code)
    and a few missing positions (which the eager path fills with 25.5).
    """
    rng = np.random.default_rng(seed)
    frames: dict[str, pl.DataFrame] = {}
    for chrom, n_pos in CONTIGS.items():
        parts = []
        for strand in "+-":
            positions = np.arange(1, n_pos + 1, dtype=np.int32)
            positions = positions[rng.random(n_pos) > 0.02]
            n = len(positions)
            k = rng.integers(0, 256, size=(n, N_U))
            vals = (k / 10.0).astype(np.float32)
            vals[rng.random(vals.shape) < 0.01] = np.float32(25.5)
            vals[rng.random(vals.shape) < 0.01] = np.float32(10.0)
            over = rng.random(vals.shape) < 0.005
            vals[over] = (rng.integers(256, 400, size=int(over.sum())) / 10.0).astype(
                np.float32
            )
            cols: dict[str, pl.Series] = {
                "position": pl.Series("position", positions, dtype=pl.Int32),
                "strand": pl.Series("strand", [strand] * n, dtype=pl.Utf8),
            }
            for u in range(1, N_U + 1):
                cols[f"u{u}"] = pl.Series(f"u{u}", vals[:, u - 1], dtype=pl.Float32)
            parts.append(pl.DataFrame(cols))
        frames[chrom] = pl.concat(parts)
    return frames


def _write_like_sioff(frame: pl.DataFrame, path: Path, rows_per_group: int) -> None:
    """Write the way ``_fold_full_chromosome`` does: several single-strand row groups."""
    schema = pa.schema(
        [("position", pa.int32()), ("strand", pa.utf8())]
        + [(f"u{i}", pa.float32()) for i in range(1, N_U + 1)]
    )
    with pq.ParquetWriter(path, schema) as writer:
        for strand in "+-":
            part = frame.filter(pl.col("strand") == strand)
            table = part.to_arrow().cast(schema)
            for off in range(0, table.num_rows, rows_per_group):
                writer.write_table(table.slice(off, rows_per_group))


@pytest.fixture(scope="module")
def frames() -> dict[str, pl.DataFrame]:
    return _synthetic_frames()


@pytest.fixture(scope="module")
def eager(frames) -> GenomeAccessibilityService:
    return GenomeAccessibilityService.from_frames(frames)


@pytest.fixture
def parquet_dir(tmp_path: Path, frames) -> Path:
    d = tmp_path / "parquet"
    d.mkdir()
    # One contig through the real writer's layout (many row groups with
    # statistics), the other through polars' writer (one row group).
    _write_like_sioff(frames["chrA"], d / "chrA.accessibility.parquet", 128)
    frames["chrB"].write_parquet(d / "chrB.accessibility.parquet")
    return d


def _random_hits(seed: int = 11, n_per_group: int = 3000) -> pl.DataFrame:
    """Hits on both strands with lengths 1..40, boundaries and out-of-bounds rows."""
    rng = np.random.default_rng(seed)
    chroms, starts, ends, strands = [], [], [], []
    for chrom, n_pos in CONTIGS.items():
        for strand in "+-":
            s = rng.integers(-5, n_pos + 6, size=n_per_group)
            ln = rng.integers(1, 41, size=n_per_group)
            e = s + ln - 1
            # Exact boundaries on top of the random ones.
            s = np.concatenate([s, [1, 1, n_pos, n_pos - 20, 0, n_pos - 18]])
            e = np.concatenate([e, [1, 30, n_pos, n_pos, 5, n_pos + 1]])
            chroms += [chrom] * len(s)
            strands += [strand] * len(s)
            starts.append(s)
            ends.append(e)
    # Rows the eager path leaves at the 10.0 default: unknown contig, on-target.
    chroms += ["chrZ", "onTarget"]
    strands += ["+", "+"]
    starts.append(np.array([10, 10]))
    ends.append(np.array([30, 30]))
    return pl.DataFrame(
        {
            "chrom": chroms,
            "start": pl.Series(np.concatenate(starts), dtype=pl.Int32),
            "end": pl.Series(np.concatenate(ends), dtype=pl.Int32),
            "strand": strands,
        }
    )


def _bits(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def _assert_same_answers(
    reference: GenomeAccessibilityService, candidate: GenomeAccessibilityService
) -> None:
    hits = _random_hits()
    a = reference.annotate_opening_energy_vectorized(hits)["opening_energy"]
    b = candidate.annotate_opening_energy_vectorized(hits)["opening_energy"]
    assert a.dtype == pl.Float32 and b.dtype == pl.Float32
    np.testing.assert_array_equal(_bits(a.to_numpy()), _bits(b.to_numpy()))

    sample = hits.filter(pl.col("chrom") != "onTarget").sample(400, seed=3)
    for chrom, s, e, strand in sample.iter_rows():
        assert reference.query_single(chrom, s, e, strand) == candidate.query_single(
            chrom, s, e, strand
        )

    for chrom, n_pos in CONTIGS.items():
        for strand in "+-":
            ra = reference.query(chrom, 0, n_pos, strand)
            rb = candidate.query(chrom, 0, n_pos, strand)
            assert rb.dtype == np.float32 and rb.shape == ra.shape
            np.testing.assert_array_equal(_bits(ra), _bits(rb))
            np.testing.assert_array_equal(
                _bits(reference.query(chrom, 17, 23, strand)),
                _bits(candidate.query(chrom, 17, 23, strand)),
            )


# --- (i) the built cache is the eager array, re-encoded ----------------------


def test_cache_build_equals_eager_array(parquet_dir: Path, eager) -> None:
    built = ensure_accessibility_cache(parquet_dir, parquet_dir, workers=2)
    assert sorted(built) == ["chrA", "chrB"]

    for chrom, n_pos in CONTIGS.items():
        for strand in "+-":
            npy, exceptions = cache_paths(parquet_dir, chrom, strand)
            assert npy.exists() and exceptions.exists()
            codes = np.load(npy, mmap_mode="r")
            assert codes.dtype == np.uint8 and codes.shape == (n_pos, N_U)

            dense = eager.query(chrom, 0, n_pos, strand)
            expected = np.clip(np.round(dense * 10.0), 0, 255).astype(np.uint8)
            np.testing.assert_array_equal(np.asarray(codes), expected)

            side = np.load(exceptions)
            over = np.flatnonzero(dense.ravel() > 25.5)
            np.testing.assert_array_equal(side["index"], over)
            np.testing.assert_array_equal(
                _bits(side["value"]), _bits(dense.ravel()[over])
            )


def test_cache_build_is_idempotent_and_can_be_restricted(parquet_dir: Path) -> None:
    assert ensure_accessibility_cache(parquet_dir, parquet_dir, contigs=["chrB"]) == [
        "chrB"
    ]
    npy_b, _ = cache_paths(parquet_dir, "chrB", "+")
    npy_a, _ = cache_paths(parquet_dir, "chrA", "+")
    assert npy_b.exists() and not npy_a.exists()
    before = npy_b.stat().st_mtime_ns

    assert sorted(ensure_accessibility_cache(parquet_dir, parquet_dir)) == ["chrA"]
    assert ensure_accessibility_cache(parquet_dir, parquet_dir) == []
    assert npy_b.stat().st_mtime_ns == before


# --- (ii) lookups through the memmap equal the eager path ----------------------


def test_memmap_service_matches_eager(parquet_dir: Path, eager) -> None:
    ensure_accessibility_cache(parquet_dir, parquet_dir)
    cached = GenomeAccessibilityService(parquet_dir, cache_dir=parquet_dir)
    _assert_same_answers(eager, cached)


def test_eager_parquet_service_matches_and_writes_nothing(
    parquet_dir: Path, eager
) -> None:
    """Without ``cache_dir`` the service is the original Parquet loader."""
    _assert_same_answers(eager, GenomeAccessibilityService(parquet_dir))
    assert not [f for f in parquet_dir.iterdir() if ".access." in f.name], (
        "the eager loader must never write cache files"
    )


def test_service_builds_a_missing_cache_lazily(
    tmp_path: Path, parquet_dir: Path, eager
) -> None:
    cache_dir = tmp_path / "cache" / "nested"
    assert not cache_dir.exists()
    cached = GenomeAccessibilityService(parquet_dir, cache_dir=cache_dir)
    _assert_same_answers(eager, cached)
    assert cache_paths(cache_dir, "chrA", "-")[0].exists()
    assert not list(parquet_dir.glob("*.npy")), "cache written next to the Parquet"


# --- (iii) a directory with only the .npy files works -------------------------


def test_npy_only_directory_works(tmp_path: Path, parquet_dir: Path, eager) -> None:
    ensure_accessibility_cache(parquet_dir, parquet_dir)
    npy_only = tmp_path / "npy_only"
    npy_only.mkdir()
    for f in parquet_dir.iterdir():
        if ".access." in f.name:
            shutil.copy(f, npy_only / f.name)
    assert not list(npy_only.glob("*.parquet"))

    assert ensure_accessibility_cache(npy_only, npy_only) == []
    _assert_same_answers(
        eager, GenomeAccessibilityService(npy_only, cache_dir=npy_only)
    )

    empty = tmp_path / "empty"
    empty.mkdir()
    _assert_same_answers(eager, GenomeAccessibilityService(empty, cache_dir=npy_only))


# --- (iv) neither cache nor Parquet -> AccessibilityError ----------------------


@pytest.mark.parametrize("memmap", [False, True])
def test_missing_cache_and_parquet_raise(tmp_path: Path, memmap: bool) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    service = GenomeAccessibilityService(empty, cache_dir=empty if memmap else None)
    with pytest.raises(AccessibilityError, match="chrQ"):
        service.query("chrQ", 0, 10, "+")

    hits = pl.DataFrame(
        {"chrom": ["chrQ"], "start": [10], "end": [30], "strand": ["+"]}
    )
    # The vectorised annotator keeps the old contract: unknown contig -> 10.0.
    out = service.annotate_opening_energy_vectorized(hits)
    assert out["opening_energy"].to_list() == [10.0]


def test_corrupt_cache_file_is_reported(parquet_dir: Path) -> None:
    ensure_accessibility_cache(parquet_dir, parquet_dir)
    npy, _ = cache_paths(parquet_dir, "chrA", "+")
    np.save(npy, np.zeros((5, 3), dtype=np.float32))
    with pytest.raises(AccessibilityError, match="chrA"):
        GenomeAccessibilityService(parquet_dir, cache_dir=parquet_dir).query(
            "chrA", 0, 1, "+"
        )


def test_strand_absent_from_parquet_raises_like_eager(tmp_path: Path) -> None:
    d = tmp_path / "plus_only"
    d.mkdir()
    frame = pl.DataFrame(
        {
            "position": pl.Series([1, 2, 3], dtype=pl.Int32),
            "strand": ["+", "+", "+"],
            "u1": pl.Series([0.1, 0.2, 0.3], dtype=pl.Float32),
        }
    )
    frame.write_parquet(d / "chrP.accessibility.parquet")
    eager = GenomeAccessibilityService.from_frames({"chrP": frame})
    cached = GenomeAccessibilityService(d, cache_dir=d)
    np.testing.assert_array_equal(
        eager.query("chrP", 0, 3, "+"), cached.query("chrP", 0, 3, "+")
    )
    with pytest.raises(AccessibilityError, match="strand"):
        eager.query("chrP", 0, 3, "-")
    with pytest.raises(AccessibilityError, match="strand"):
        cached.query("chrP", 0, 3, "-")
    # Idempotent even though one strand has no data.
    assert ensure_accessibility_cache(d, d) == []


def test_build_is_atomic(parquet_dir: Path) -> None:
    ensure_accessibility_cache(parquet_dir, parquet_dir)
    leftovers = [f for f in os.listdir(parquet_dir) if ".tmp" in f]
    assert leftovers == []


# --- the option reaches both public forms --------------------------------------


def _rounded_profiles() -> dict[str, pl.DataFrame]:
    """sioff.accessibility profiles rounded to 0.1, as the Parquet writer does."""
    import sioff

    profiles = sioff.accessibility(
        Path(__file__).parent / "data" / "genome.fa",
        window_size=40,
        max_span=20,
        unpaired_prob=10,
    )
    return {
        chrom: frame.with_columns(
            [
                ((pl.col(c) * 10.0).round(0) / 10.0).cast(pl.Float32)
                for c in frame.columns
                if c.startswith("u")
            ]
        )
        for chrom, frame in profiles.items()
    }


def test_api_cache_dir_single_file_and_directory(tmp_path: Path) -> None:
    import sioff

    profiles = _rounded_profiles()
    acc_dir = tmp_path / "acc"
    acc_dir.mkdir()
    for chrom, frame in profiles.items():
        frame.write_parquet(acc_dir / f"{chrom}.accessibility.parquet")
    predictions = Path(__file__).parent / "data" / "risearch_genome_example.out"

    from_frames = sioff.off_targets(risearch_file=predictions, accessibility=profiles)

    # Default strategy on a single file: the eager loader, no cache files.
    default = sioff.off_targets(risearch_file=predictions, accessibility_dir=acc_dir)
    assert isinstance(default, pl.DataFrame)
    assert default.sort(default.columns).equals(from_frames.sort(from_frames.columns))
    assert not [f for f in acc_dir.iterdir() if ".access." in f.name], (
        "the default strategy wrote cache files"
    )

    cache_dir = tmp_path / "cache"
    single = sioff.off_targets(
        risearch_file=predictions,
        accessibility_dir=acc_dir,
        accessibility_cache_dir=cache_dir,
        accessibility_strategy="memmap",
    )
    assert isinstance(single, pl.DataFrame)
    assert single.sort(single.columns).equals(from_frames.sort(from_frames.columns))
    assert cache_paths(cache_dir, "chr1", "+")[0].exists()
    assert not list(acc_dir.glob("*.npy")), "cache written into the Parquet dir"

    # Directory form: the cache is built up front, before the pool starts.
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / predictions.name).write_bytes(predictions.read_bytes())
    cache_dir2 = tmp_path / "cache2"
    gen = sioff.off_targets(
        risearch_file=in_dir,
        accessibility_dir=acc_dir,
        accessibility_cache_dir=cache_dir2,
        accessibility_strategy="memmap",
        n_workers=1,
    )
    first = next(gen)
    assert cache_paths(cache_dir2, "chr1", "-")[0].exists()
    frames = [first, *gen]
    assert len(frames) == 1
    batch = frames[0]
    common = [c for c in single.columns if c in batch.columns]
    assert batch.select(common).sort(common).equals(single.select(common).sort(common))
