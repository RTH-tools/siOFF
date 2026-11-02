"""The memory-mapped accessibility cache answers exactly like the dense profile.

A directory-backed `GenomeAccessibilityService` always serves a compact uint8
cache (``<chrom>_<plus|minus>.access.npy``, value ``round(E * 10)``) that every
worker memory-maps, so the OS page cache holds one shared copy. Without a
``cache_dir`` the cache lives next to the Parquet and is built on first use.
Entries that the code cannot represent exactly (opening energies above 25.5,
negative or unrounded values) live in a small sorted sidecar, which keeps the
cache lossless for any input.

The reference for every test here is the frame-backed service
(`GenomeAccessibilityService.from_frames`): it filters one strand and
materialises a dense float32 array, the very code the Parquet path ran before
the cache became the only on-disk path.
"""

import contextlib
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

# The off_targets tests run on the example data (read-only; folds go to
# tmp_path): a 6 kb genome whose hits lie inside it on both strands, so the
# compared opening energies are looked up rather than the 10.0 default.
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "data"
GENOME = EXAMPLES / "genome.fa"
PREDICTIONS = EXAMPLES / "predictions.out"
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


def test_default_cache_dir_is_the_profile_dir(parquet_dir: Path, eager) -> None:
    """With no cache_dir the service builds and reads the cache next to the Parquet."""
    _assert_same_answers(eager, GenomeAccessibilityService(parquet_dir))
    assert cache_paths(parquet_dir, "chrA", "-")[0].exists()


def test_cache_in_profile_dir_is_found_with_another_cache_dir(
    tmp_path: Path, parquet_dir: Path, eager
) -> None:
    """An npy-only -a plus an unrelated --accessibility-cache-dir: read from -a."""
    ensure_accessibility_cache(parquet_dir, parquet_dir)
    npy_only = tmp_path / "npy_only"
    shutil.copytree(
        parquet_dir, npy_only, ignore=shutil.ignore_patterns("*.accessibility.parquet")
    )
    elsewhere = tmp_path / "scratch"
    _assert_same_answers(
        eager, GenomeAccessibilityService(npy_only, cache_dir=elsewhere)
    )
    assert not elsewhere.exists() or not any(elsewhere.iterdir())


def test_cache_dir_is_read_before_the_profile_dir(
    tmp_path: Path, parquet_dir: Path, eager
) -> None:
    """Both directories hold a complete cache: --accessibility-cache-dir wins."""
    cache_dir = tmp_path / "cache"
    ensure_accessibility_cache(parquet_dir, cache_dir)
    other = tmp_path / "other_fold"
    other.mkdir()
    for chrom, frame in _synthetic_frames(seed=8).items():
        frame.write_parquet(other / f"{chrom}.accessibility.parquet")
    npy_only = tmp_path / "npy_only"
    ensure_accessibility_cache(other, npy_only)
    in_a, in_cache_dir = (cache_paths(d, "chrA", "+")[0] for d in (npy_only, cache_dir))
    assert in_a.read_bytes() != in_cache_dir.read_bytes(), "the caches must differ"

    _assert_same_answers(
        eager, GenomeAccessibilityService(npy_only, cache_dir=cache_dir)
    )


def test_not_found_names_both_directories(tmp_path: Path) -> None:
    profile_dir, cache_dir = tmp_path / "acc", tmp_path / "cache"
    service = GenomeAccessibilityService(profile_dir, cache_dir=cache_dir)
    with pytest.raises(AccessibilityError, match="chrQ") as exc_info:
        service.query("chrQ", 0, 10, "+")
    for directory in (cache_dir, profile_dir):
        assert str(cache_paths(directory, "chrQ", "+")[0]) in str(exc_info.value)


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


# --- (v) a cache older than its Parquet is rebuilt -----------------------------


@contextlib.contextmanager
def _capture_warnings():
    """WARNING messages logged in this process inside the block.

    A CLI invocation resets loguru's handlers, so start capturing after it.
    """
    from loguru import logger

    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    try:
        yield seen
    finally:
        logger.remove(handler)


@pytest.fixture
def warnings_seen():
    with _capture_warnings() as seen:
        yield seen


def _stale_warnings(seen: list[str]) -> list[str]:
    return [m for m in seen if "is older than" in m]


def _refolded(frame: pl.DataFrame) -> pl.DataFrame:
    """Another fold of the same contig: other values and only 20 u columns."""
    u_cols = [f"u{u}" for u in range(1, 21)]
    return frame.select("position", "strand", *[pl.col(c).reverse() for c in u_cols])


def _rewrite_parquet(
    parquet_dir: Path, cache_dir: Path, chrom: str, frame: pl.DataFrame
) -> Path:
    """Overwrite one contig's Parquet, dated after its cache like a later re-fold.

    The mtime is set explicitly, 5 s past the cache's, so a coarse filesystem
    clock cannot give the two the same timestamp. That is also ahead of this
    machine's clock, as a Parquet written on a node whose clock runs fast.
    """
    path = parquet_dir / f"{chrom}.accessibility.parquet"
    frame.write_parquet(path)
    built = max(cache_paths(cache_dir, chrom, s)[0].stat().st_mtime_ns for s in "+-")
    os.utime(path, ns=(built + 5_000_000_000, built + 5_000_000_000))
    return path


def test_ensure_rebuilds_a_cache_older_than_its_parquet(
    parquet_dir: Path, frames, warnings_seen
) -> None:
    assert sorted(ensure_accessibility_cache(parquet_dir, parquet_dir)) == [
        "chrA",
        "chrB",
    ]
    refolded = _refolded(frames["chrA"])
    parquet = _rewrite_parquet(parquet_dir, parquet_dir, "chrA", refolded)

    assert ensure_accessibility_cache(parquet_dir, parquet_dir) == ["chrA"]
    assert _stale_warnings(warnings_seen) == [
        f"Accessibility cache for chrA in {parquet_dir} is older than {parquet}; "
        "rebuilding it"
    ]
    assert np.load(cache_paths(parquet_dir, "chrA", "+")[0]).shape[1] == 20
    # The rebuilt cache is dated no earlier than its Parquet, even though that
    # lies ahead of the clock: nothing more to do on the next run.
    assert ensure_accessibility_cache(parquet_dir, parquet_dir) == []
    assert len(_stale_warnings(warnings_seen)) == 1

    reference = GenomeAccessibilityService.from_frames(
        {"chrA": refolded, "chrB": frames["chrB"]}
    )
    _assert_same_answers(reference, GenomeAccessibilityService(parquet_dir))


@pytest.mark.parametrize(
    "layout", ["next_to_parquet", "in_cache_dir", "next_to_parquet_with_cache_dir"]
)
def test_service_rebuilds_a_cache_older_than_its_parquet(
    tmp_path: Path, parquet_dir: Path, frames, warnings_seen, layout: str
) -> None:
    """The lazy path (no ensure_accessibility_cache first) rebuilds it too."""
    stale_dir = tmp_path / "cache" if layout == "in_cache_dir" else parquet_dir
    ensure_accessibility_cache(parquet_dir, stale_dir)
    cache_dir = parquet_dir if layout == "next_to_parquet" else tmp_path / "cache"
    refolded = _refolded(frames["chrA"])
    _rewrite_parquet(parquet_dir, stale_dir, "chrA", refolded)

    service = GenomeAccessibilityService(parquet_dir, cache_dir=cache_dir)
    reference = GenomeAccessibilityService.from_frames(
        {"chrA": refolded, "chrB": frames["chrB"]}
    )
    _assert_same_answers(reference, service)
    stale = _stale_warnings(warnings_seen)
    assert len(stale) == 1 and "for chrA in" in stale[0], stale
    assert np.load(cache_paths(cache_dir, "chrA", "-")[0]).shape[1] == 20


# --- (iv) neither cache nor Parquet -> AccessibilityError ----------------------


def test_missing_cache_and_parquet_raise(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    service = GenomeAccessibilityService(empty)
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


@pytest.mark.parametrize("workers", [1, 2])
def test_unreadable_parquet_is_reported_by_name(
    parquet_dir: Path, workers: int
) -> None:
    """In-process and pooled builds wrap a failure the same way."""
    (parquet_dir / "chrX.accessibility.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(
        AccessibilityError,
        match="Failed to build accessibility cache for chrX.accessibility.parquet",
    ) as exc_info:
        ensure_accessibility_cache(parquet_dir, parquet_dir, workers=workers)
    assert exc_info.value.__cause__ is not None


def test_build_is_atomic(parquet_dir: Path) -> None:
    ensure_accessibility_cache(parquet_dir, parquet_dir)
    leftovers = [f for f in os.listdir(parquet_dir) if ".tmp" in f]
    assert leftovers == []


# --- the option reaches both public forms --------------------------------------


def _assert_looked_up(df: pl.DataFrame) -> None:
    """Guard: some opening energy is not the 10.0 out-of-bounds default."""
    assert (df["opening_energy"] != 10.0).any(), "every hit fell back to 10.0"


def _rounded_profiles() -> dict[str, pl.DataFrame]:
    """sioff.accessibility profiles rounded to 0.1, as the Parquet writer does."""
    import sioff

    profiles = sioff.accessibility(
        GENOME,
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

    from_frames = sioff.off_targets(risearch_file=PREDICTIONS, accessibility=profiles)
    _assert_looked_up(from_frames)

    cache_dir = tmp_path / "cache"
    single = sioff.off_targets(
        risearch_file=PREDICTIONS,
        accessibility_dir=acc_dir,
        accessibility_cache_dir=cache_dir,
    )
    assert isinstance(single, pl.DataFrame)
    assert single.sort(single.columns).equals(from_frames.sort(from_frames.columns))
    assert cache_paths(cache_dir, "chr1", "+")[0].exists()
    assert not list(acc_dir.glob("*.npy")), "cache written into the Parquet dir"

    # Directory form: the cache is built up front, before the pool starts.
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / PREDICTIONS.name).write_bytes(PREDICTIONS.read_bytes())
    cache_dir2 = tmp_path / "cache2"
    gen = sioff.off_targets(
        risearch_file=in_dir,
        accessibility_dir=acc_dir,
        accessibility_cache_dir=cache_dir2,
        n_workers=1,
    )
    first = next(gen)
    assert cache_paths(cache_dir2, "chr1", "-")[0].exists()
    frames = [first, *gen]
    assert len(frames) == 1
    batch = frames[0]
    common = [c for c in single.columns if c in batch.columns]
    assert batch.select(common).sort(common).equals(single.select(common).sort(common))

    # Default: no cache dir, so the cache is built next to the Parquet.
    default = sioff.off_targets(risearch_file=PREDICTIONS, accessibility_dir=acc_dir)
    assert isinstance(default, pl.DataFrame)
    assert default.sort(default.columns).equals(from_frames.sort(from_frames.columns))
    assert cache_paths(acc_dir, "chr1", "+")[0].exists()


# --- off_targets prepares the cache before scoring ---


def _parquet_dir(tmp_path, profiles, extra_contig=None):
    d = tmp_path / "acc"
    d.mkdir()
    for chrom, frame in profiles.items():
        frame.write_parquet(d / f"{chrom}.accessibility.parquet")
    if extra_contig:  # a contig no prediction hits
        next(iter(profiles.values())).write_parquet(
            d / f"{extra_contig}.accessibility.parquet"
        )
    return d


def _in_dir(tmp_path, predictions):
    d = tmp_path / "in"
    d.mkdir()
    (d / predictions.name).write_bytes(predictions.read_bytes())
    return d


@pytest.mark.parametrize("form", ["single", "directory"])
def test_off_targets_refuses_a_directory_without_profiles(tmp_path, form):
    import sioff

    empty = tmp_path / "empty"
    empty.mkdir()
    source = PREDICTIONS if form == "single" else _in_dir(tmp_path, PREDICTIONS)
    with pytest.raises(AccessibilityError, match="No accessibility profiles in"):
        out = sioff.off_targets(
            risearch_file=source,
            accessibility_dir=empty,
            accessibility_cache_dir=tmp_path / "cache",
            n_workers=1,
        )
        if not isinstance(out, pl.DataFrame):
            list(out)


def _record_calls(monkeypatch, targets, called: list[str]) -> None:
    """Spy on ``(class, method)`` pairs: record each call, then run the method."""
    for cls, name in targets:
        original = getattr(cls, name)

        def spy(*args, _original=original, _name=name, **kwargs):
            called.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cls, name, spy)


def test_directory_mode_checks_profiles_before_the_transcriptome(tmp_path, monkeypatch):
    """An empty -a fails before the GTF parse and self-hybridisation, and leaves
    no transcriptome IPC temp dir behind."""
    import tempfile

    import sioff.core.off_targets as core

    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    called: list[str] = []
    _record_calls(
        monkeypatch,
        [
            (core.AnnotationParser, "load_gtf"),
            (core.RIsearchService, "self_hybridization_emin_batch"),
        ],
        called,
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AccessibilityError, match="No accessibility profiles in"):
        list(
            core.compute_off_targets_directory(
                input_dir=_in_dir(tmp_path, PREDICTIONS),
                sirna_fasta=EXAMPLES / "sirnas.fa",
                gtf_file=EXAMPLES / "annotation.gtf",
                accessibility_dir=empty,
                n_workers=1,
            )
        )
    assert not list(scratch.glob("risearch_ipc_*")), "transcriptome IPC dir leaked"
    assert called == []


def test_directory_mode_leaves_no_ipc_dir_when_setup_fails(tmp_path, monkeypatch):
    """Any failure before the pool starts (here a bad --alpha) cleans up too."""
    import tempfile

    import sioff.core.off_targets as core

    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    with pytest.raises(ValueError):
        list(
            core.compute_off_targets_directory(
                input_dir=_in_dir(tmp_path, PREDICTIONS),
                gtf_file=EXAMPLES / "annotation.gtf",
                alpha="0.8,1.0",
                n_workers=1,
            )
        )
    assert not list(scratch.glob("risearch_ipc_*")), "transcriptome IPC dir leaked"


@pytest.mark.parametrize("source", ["risearch_file", "sirna_fasta"])
def test_single_file_checks_profiles_before_reading_predictions(
    tmp_path, monkeypatch, source
):
    """An empty -a fails before predictions are loaded, searched or intersected."""
    import sioff.core.off_targets as core

    called: list[str] = []
    _record_calls(
        monkeypatch,
        [
            (core.RIsearchParser, "load"),
            (core.RIsearchService, "validate_sirna_fasta"),
            (core.RIsearchService, "index_target"),
            (core.RIsearchService, "run_search"),
            (core.AnnotationParser, "load_gtf"),
        ],
        called,
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    predictions = (
        {"risearch_file": PREDICTIONS}
        if source == "risearch_file"
        else {
            "sirna_fasta": EXAMPLES / "sirnas.fa",
            "target_fasta": GENOME,
            "target_index": tmp_path / "missing.idx",
        }
    )
    with pytest.raises(AccessibilityError, match="No accessibility profiles in"):
        core.compute_off_targets_single(
            **predictions,
            gtf_file=EXAMPLES / "annotation.gtf",
            accessibility_dir=empty,
        )
    assert called == []


def test_single_file_with_empty_dir_reads_a_complete_cache_dir(tmp_path):
    import sioff

    profiles = _rounded_profiles()
    acc = _parquet_dir(tmp_path, profiles)
    cache = tmp_path / "cache"
    ensure_accessibility_cache(acc, cache)
    empty = tmp_path / "empty"
    empty.mkdir()
    got = sioff.off_targets(
        risearch_file=PREDICTIONS,
        accessibility_dir=empty,
        accessibility_cache_dir=cache,
    )
    ref = sioff.off_targets(risearch_file=PREDICTIONS, accessibility=profiles)
    _assert_looked_up(ref)
    assert got.sort(got.columns).equals(ref.sort(ref.columns))
    assert not any(empty.iterdir())


def test_single_file_mode_builds_every_contig_up_front(tmp_path):
    """Not lazily per hit contig: a contig without hits is cached too."""
    import sioff

    acc = _parquet_dir(tmp_path, _rounded_profiles(), extra_contig="chrNoHits")
    sioff.off_targets(risearch_file=PREDICTIONS, accessibility_dir=acc)
    for strand in "+-":
        assert cache_paths(acc, "chrNoHits", strand)[0].exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
@pytest.mark.parametrize("form", ["single", "directory"])
def test_read_only_parquet_dir_without_cache_dir_is_reported(tmp_path, form):
    import sioff

    acc = _parquet_dir(tmp_path, _rounded_profiles())
    source = PREDICTIONS if form == "single" else _in_dir(tmp_path, PREDICTIONS)
    acc.chmod(0o555)
    try:
        with pytest.raises(AccessibilityError, match="--accessibility-cache-dir"):
            out = sioff.off_targets(
                risearch_file=source, accessibility_dir=acc, n_workers=1
            )
            if not isinstance(out, pl.DataFrame):
                list(out)
    finally:
        acc.chmod(0o755)


KEYS = ["sirna_id", "chrom", "start", "end", "strand", "energy"]


def _fold_to_parquet(out: Path, window: int, span: int, unpaired: int) -> None:
    from typer.testing import CliRunner

    from sioff.cli import app

    result = CliRunner().invoke(
        app,
        [
            "accessibility",
            "-f",
            str(GENOME),
            "-o",
            str(out),
            "-W",
            str(window),
            "-L",
            str(span),
            "-u",
            str(unpaired),
            "--output-format",
            "parquet",
        ],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("form", ["single", "directory"])
def test_refolding_a_parquet_dir_rebuilds_its_cache(tmp_path, form):
    """Re-fold into the -a of an earlier run: the next run must not read the old cache."""
    import sioff

    acc = tmp_path / "acc"
    source = PREDICTIONS if form == "single" else _in_dir(tmp_path, PREDICTIONS)

    def run() -> pl.DataFrame:
        out = sioff.off_targets(
            risearch_file=source, accessibility_dir=acc, n_workers=1
        )
        return out if isinstance(out, pl.DataFrame) else pl.concat(list(out))

    _fold_to_parquet(acc, 40, 20, 10)
    first = run()
    assert cache_paths(acc, "chr1", "+")[0].exists()

    _fold_to_parquet(acc, 80, 40, 20)
    parquet = acc / "chr1.accessibility.parquet"
    built = max(cache_paths(acc, "chr1", s)[0].stat().st_mtime_ns for s in "+-")
    if parquet.stat().st_mtime_ns <= built:  # coarse filesystem clock
        os.utime(parquet, ns=(built + 5_000_000_000, built + 5_000_000_000))
    with _capture_warnings() as seen:
        second = run()

    reference = sioff.off_targets(
        risearch_file=PREDICTIONS, accessibility={"chr1": pl.read_parquet(parquet)}
    )
    _assert_looked_up(reference)
    common = [c for c in second.columns if c in reference.columns]
    ref = reference.select(common).sort(KEYS)
    assert (first.sort(KEYS)["opening_energy"] != ref["opening_energy"]).any(), (
        "the two folds must give different energies"
    )
    assert second.select(common).sort(KEYS).equals(ref)
    stale = _stale_warnings(seen)
    assert len(stale) == 1 and str(parquet) in stale[0], stale
