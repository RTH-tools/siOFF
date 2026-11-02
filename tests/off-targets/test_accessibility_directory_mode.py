"""Directory mode looks opening energies up in the memory-mapped cache.

Every worker memory-maps the uint8 ``.access.npy`` cache the parent built
before starting the pool (chr1: 30 GB dense per strand would be the cost of
loading it instead -- 46 workers on hg38 were all OOM-killed). The result must
be bit-for-bit what the dense lookup
(``GenomeAccessibilityService.annotate_opening_energy_vectorized``) produces,
including the corner cases:

- both strands, interaction lengths 1..40 (column clamped at u30);
- hits at position 1 and at the last position of the contig;
- hits running past the last position (out of bounds -> 10.0);
- positions missing from the Parquet (-> 25.5, the dense ``np.full`` default);
- a strand with no rows in the file (-> 10.0);
- a contig absent from the accessibility directory (-> 10.0).
"""

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from sioff.core.off_targets import (
    WorkerFailureError,
    compute_off_targets_directory,
)
from sioff.services.accessibility import (
    GenomeAccessibilityService,
    cache_paths,
    ensure_accessibility_cache,
)

N_U = 30
# Small row groups so a contig spans several of them and the row-group walk
# (statistics -> needed row groups only) is exercised, not just a single read.
ROW_GROUP_ROWS = 500
CHROM_LEN = {"chrA": 2600, "chrB": 900}
GAP = (1200, 1300)  # positions missing from chrA, both strands


def _write_accessibility(path: Path, positions: dict[str, np.ndarray], rng) -> None:
    """Write one contig in the exact ``sioff accessibility`` Parquet schema."""
    schema = pa.schema(
        [("position", pa.int32()), ("strand", pa.utf8())]
        + [(f"u{i}", pa.float32()) for i in range(1, N_U + 1)]
    )
    with pq.ParquetWriter(path, schema) as writer:
        for strand in ("+", "-"):
            pos = positions.get(strand)
            if pos is None:
                continue
            for lo in range(0, len(pos), ROW_GROUP_ROWS):
                p = pos[lo : lo + ROW_GROUP_ROWS]
                # Values already rounded to 0.1, plus the two sentinels.
                u = np.round(rng.uniform(0.0, 25.0, (len(p), N_U)) * 10.0) / 10.0
                u[rng.random(u.shape) < 0.02] = 25.5
                u[rng.random(u.shape) < 0.02] = 10.0
                u = u.astype(np.float32)
                arrays = [pa.array(p.astype(np.int32)), pa.array([strand] * len(p))] + [
                    pa.array(u[:, k]) for k in range(N_U)
                ]
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))


@pytest.fixture(scope="module")
def acc_dir(tmp_path_factory) -> Path:
    rng = np.random.default_rng(7)
    d = tmp_path_factory.mktemp("acc")
    full = np.arange(1, CHROM_LEN["chrA"] + 1)
    gapped = full[(full < GAP[0]) | (full > GAP[1])]
    _write_accessibility(
        d / "chrA.accessibility.parquet", {"+": gapped, "-": gapped}, rng
    )
    # chrB: '+' only, so '-' hits must fall back to 10.0 like the eager path.
    _write_accessibility(
        d / "chrB.accessibility.parquet",
        {"+": np.arange(1, CHROM_LEN["chrB"] + 1)},
        rng,
    )
    return d


def _synthetic_hits(rng, n_sirnas: int = 5, n_rows: int = 300) -> pl.DataFrame:
    frames = []
    for i in range(n_sirnas):
        chroms = rng.choice(["chrA", "chrA", "chrA", "chrB", "chrC"], n_rows)
        strands = rng.choice(["+", "-"], n_rows)
        lengths = rng.integers(1, 41, n_rows)  # 1..40 -> u30 clamp above 30
        starts = np.empty(n_rows, dtype=np.int64)
        for j, c in enumerate(chroms):
            # Up to 60 past the end so some hits run out of bounds.
            starts[j] = rng.integers(1, CHROM_LEN.get(str(c), 1000) + 60)
        ends = starts + lengths - 1
        # Deterministic boundary rows on top of the random ones.
        extra = [
            ("chrA", "+", 1, 1),
            ("chrA", "-", 1, 1),
            ("chrA", "+", 2581, 2600),  # ends on the last position
            ("chrA", "-", 2600, 2600),
            ("chrA", "+", 2582, 2601),  # one past the end -> OOB
            ("chrA", "-", 2601, 2620),
            ("chrA", "+", 1199, 1221),  # end inside the gap -> 25.5
            ("chrA", "-", 1250, 1270),  # start inside the gap -> 25.5
            ("chrA", "+", 100, 129),  # length 30 -> u30
            ("chrA", "+", 100, 130),  # length 31 -> clamped to u30
            ("chrA", "-", 100, 139),  # length 40 -> clamped to u30
            ("chrB", "-", 10, 30),  # strand missing from the file -> 10.0
            ("chrC", "+", 10, 30),  # contig missing from the dir -> 10.0
        ]
        frames.append(
            pl.DataFrame(
                {
                    "sirna_id": [f"siRNA{i}"] * (n_rows + len(extra)),
                    "chrom": list(chroms) + [e[0] for e in extra],
                    "start": list(starts) + [e[2] for e in extra],
                    "end": list(ends) + [e[3] for e in extra],
                    "strand": list(strands) + [e[1] for e in extra],
                    "energy": rng.uniform(-30.0, -8.0, n_rows + len(extra)),
                    "raw_e_min": [-30.0 - i] * (n_rows + len(extra)),
                },
                schema={
                    "sirna_id": pl.Utf8,
                    "chrom": pl.Utf8,
                    "start": pl.Int32,
                    "end": pl.Int32,
                    "strand": pl.Utf8,
                    "energy": pl.Float32,
                    "raw_e_min": pl.Float32,
                },
            )
        )
    return pl.concat(frames)


@pytest.fixture(scope="module")
def hits(tmp_path_factory) -> tuple[Path, pl.DataFrame]:
    rng = np.random.default_rng(11)
    df = _synthetic_hits(rng)
    d = tmp_path_factory.mktemp("hits")
    for (sid,), part in df.group_by("sirna_id"):
        part.write_parquet(d / f"risearch_{sid}.parquet")
    return d, df


# "energy" breaks ties: the random fixture can produce two hits with the same
# siRNA, contig, coordinates and strand (seed 11 does, once), and polars does
# not order tied rows deterministically, so without it the comparison below
# depended on which of the pair came first (failed once on CI, ubuntu 3.14).
SORT_KEYS = ["sirna_id", "chrom", "start", "end", "strand", "energy"]


def _run(
    hits_dir: Path,
    acc_dir: Path,
    n_workers: int,
    cache_dir: Path | None,
):
    gen = compute_off_targets_directory(
        input_dir=hits_dir,
        accessibility_dir=acc_dir,
        accessibility_cache_dir=cache_dir,
        alpha="0.8;1",
        gamma="0.9;1",
        theta="0.7",
        n_workers=n_workers,
    )
    frames, metas = [], {}
    for df, meta in gen:
        frames.append(df)
        metas.update(meta["z_per_sirna"])
    return pl.concat(frames).sort(SORT_KEYS), metas


def _dense_oracle(acc_dir: Path, all_hits: pl.DataFrame) -> pl.DataFrame:
    service = GenomeAccessibilityService.from_frames(
        {c: pl.read_parquet(acc_dir / f"{c}.accessibility.parquet") for c in CHROM_LEN}
    )
    return service.annotate_opening_energy_vectorized(all_hits).sort(SORT_KEYS)


def test_directory_mode_matches_the_dense_lookup(hits, acc_dir, tmp_path):
    """Independent oracle: the dense lookup on the concatenated hits."""
    hits_dir, all_hits = hits
    result, _ = _run(hits_dir, acc_dir, n_workers=3, cache_dir=tmp_path / "cache")
    expected = _dense_oracle(acc_dir, all_hits)
    assert result.height == all_hits.height
    np.testing.assert_array_equal(
        result["opening_energy"].to_numpy(), expected["opening_energy"].to_numpy()
    )

    # Guard that the fixture really exercised every branch.
    oe = result["opening_energy"]
    assert oe.dtype == pl.Float32
    assert (oe == 25.5).sum() > 0, "no position-missing rows"
    assert (oe == 10.0).sum() > 0, "no out-of-bounds/unknown rows"
    assert result.filter(pl.col("chrom") == "chrC")["opening_energy"].eq(10.0).all()
    assert (
        result.filter((pl.col("chrom") == "chrB") & (pl.col("strand") == "-"))[
            "opening_energy"
        ]
        .eq(10.0)
        .all()
    )
    assert oe.n_unique() > 20, "values look degenerate"


def _cache_files(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir() if ".access." in p.name)


def test_cache_goes_to_cache_dir_or_next_to_the_profiles(hits, acc_dir, tmp_path):
    """The cache is built in ``cache_dir``, or next to the Parquet without one."""
    hits_dir, all_hits = hits
    assert _cache_files(acc_dir) == []
    cache_dir = tmp_path / "cache"
    first, first_z = _run(hits_dir, acc_dir, n_workers=2, cache_dir=cache_dir)
    assert first.height == all_hits.height
    assert _cache_files(acc_dir) == [], "wrote into the accessibility dir"
    assert _cache_files(cache_dir) == sorted(
        cache_paths(cache_dir, c, s)[k].name
        for c in CHROM_LEN
        for s in "+-"
        for k in (0, 1)
    )

    # Default cache dir is the accessibility dir itself (on a private copy).
    acc_copy = tmp_path / "acc_copy"
    acc_copy.mkdir()
    for f in acc_dir.iterdir():
        (acc_copy / f.name).write_bytes(f.read_bytes())
    default, default_z = _run(hits_dir, acc_copy, n_workers=1, cache_dir=None)
    assert_frame_equal(default, first, check_exact=True)
    assert default_z == first_z
    assert _cache_files(acc_copy) == _cache_files(cache_dir)


def test_memmap_serves_a_cache_only_dir_without_parquet(hits, acc_dir, tmp_path):
    """A complete cache in --accessibility-cache-dir needs no Parquet at all."""
    hits_dir, all_hits = hits
    cache_only = tmp_path / "cache_only"
    assert ensure_accessibility_cache(acc_dir, cache_only, workers=2) == [
        "chrA",
        "chrB",
    ]
    no_parquet = tmp_path / "no_parquet"
    no_parquet.mkdir()
    result, _ = _run(hits_dir, no_parquet, n_workers=2, cache_dir=cache_only)
    expected = _dense_oracle(acc_dir, all_hits)
    np.testing.assert_array_equal(
        result["opening_energy"].to_numpy(), expected["opening_energy"].to_numpy()
    )
    assert not list(no_parquet.iterdir())


def test_failed_worker_raises_after_good_results(hits, acc_dir, tmp_path):
    """A worker failure must surface as an error, never as a silent skip."""
    hits_dir, _ = hits
    broken = tmp_path / "broken"
    broken.mkdir()
    for f in hits_dir.iterdir():
        (broken / f.name).write_bytes(f.read_bytes())
    (broken / "risearch_corrupt.parquet").write_bytes(b"this is not parquet")

    gen = compute_off_targets_directory(
        input_dir=broken,
        accessibility_dir=acc_dir,
        accessibility_cache_dir=tmp_path / "cache",
        alpha="0.8;1",
        n_workers=2,
    )
    frames = []
    with pytest.raises(WorkerFailureError, match=r"1/6 .*failed") as info:
        for df, _ in gen:
            frames.append(df)
    assert len(frames) == 5, "good siRNAs must still be yielded"
    assert "risearch_corrupt.parquet" in str(info.value)
    assert info.value.n_failed == 1 and info.value.n_total == 6
    assert info.value.failed_names == ["risearch_corrupt.parquet"]


def test_failed_worker_exits_nonzero_from_cli(hits, acc_dir, tmp_path, plain):
    from typer.testing import CliRunner

    from sioff.cli import app

    hits_dir, _ = hits
    broken = tmp_path / "broken"
    broken.mkdir()
    for f in hits_dir.iterdir():
        (broken / f.name).write_bytes(f.read_bytes())
    (broken / "risearch_corrupt.parquet").write_bytes(b"this is not parquet")

    result = CliRunner().invoke(
        app,
        [
            "off-targets",
            "-r",
            str(broken),
            "-a",
            str(acc_dir),
            "--accessibility-cache-dir",
            str(tmp_path / "cache"),
            "-j",
            "2",
            "-o",
            str(tmp_path / "out" / "output.tsv"),
        ],
    )
    assert result.exit_code != 0
    text = plain(result.output)
    assert "1/6" in text and "failed" in text and "risearch_corrupt.parquet" in text
    assert "No predictions to process" not in text
    # The five good siRNAs were still written before the non-zero exit.
    out = pl.read_csv(tmp_path / "out" / "output.tsv", separator="\t")
    assert out["sirna_id"].n_unique() == 5


def test_summary_only_returns_sums_without_rows(hits, acc_dir, tmp_path):
    """--summary-only: no per-site rows leave the workers, the sums are unchanged."""
    hits_dir, _ = hits
    cache = tmp_path / "cache"

    def run(summary_only):
        metas, n_rows, n_frames = {}, 0, 0
        for df, meta in compute_off_targets_directory(
            input_dir=hits_dir,
            accessibility_dir=acc_dir,
            accessibility_cache_dir=cache,
            alpha="0.8;1",
            gamma="0.9;1",
            theta="0.7",
            n_workers=2,
            summary_only=summary_only,
        ):
            n_frames += 1
            n_rows += df.height
            metas.update(meta["z_per_sirna"])
        return metas, n_rows, n_frames

    full, full_rows, full_frames = run(False)
    agg, agg_rows, agg_frames = run(True)
    assert full_rows > 0 and agg_rows == 0
    assert agg_frames == full_frames
    assert set(agg) == set(full)
    for sid, row in full.items():
        for k, v in row.items():
            if k != "sirna_id":
                assert agg[sid][k] == pytest.approx(v, rel=1e-6), (sid, k)
