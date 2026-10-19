"""``sioff accessibility`` writes the memmap cache directly (``--output-format npy``).

The npy output must be the very files `build_accessibility_cache` makes from
the Parquet of the same fold, so the ``memmap`` strategy answers the same
whichever way the cache came about, and ``auto`` must pick ``memmap`` for a
directory that holds only that cache.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sioff.cli import app
from sioff.services.accessibility import (
    AccessibilityError,
    _fold_full_chromosome,
    build_accessibility_cache,
    cache_paths,
)

DATA = Path(__file__).parent / "data"
GENOME = DATA / "genome.fa"
PREDICTIONS = DATA / "risearch_genome_example.out"
FOLD = ["-W", "40", "-L", "20", "-u", "10"]


def _assert_same_cache(a: Path, b: Path, chroms) -> None:
    for chrom in chroms:
        for strand in "+-":
            (npy_a, npz_a), (npy_b, npz_b) = (
                cache_paths(a, chrom, strand),
                cache_paths(b, chrom, strand),
            )
            assert npy_a.read_bytes() == npy_b.read_bytes(), (chrom, strand)
            with np.load(npz_a) as x, np.load(npz_b) as y:
                for key in ("index", "value"):
                    assert x[key].dtype == y[key].dtype
                    np.testing.assert_array_equal(x[key], y[key])


def _accessibility(out: Path, *extra: str) -> None:
    result = CliRunner().invoke(
        app, ["accessibility", "-f", str(GENOME), "-o", str(out), *FOLD, *extra]
    )
    assert result.exit_code == 0, result.output


@pytest.fixture(scope="module")
def dirs(tmp_path_factory) -> tuple[Path, Path, list[str]]:
    """``(npy_dir, parquet_dir, chroms)`` from the CLI, default and parquet format."""
    base = tmp_path_factory.mktemp("acc")
    npy_dir, parquet_dir = base / "npy", base / "parquet"
    _accessibility(npy_dir)
    _accessibility(parquet_dir, "--output-format", "parquet")
    chroms = sorted(
        p.name.removesuffix(".accessibility.parquet")
        for p in parquet_dir.glob("*.accessibility.parquet")
    )
    assert chroms
    return npy_dir, parquet_dir, chroms


def test_default_format_is_npy_and_parquet_is_opt_in(dirs) -> None:
    npy_dir, parquet_dir, chroms = dirs
    assert not list(npy_dir.glob("*.parquet"))
    assert sorted(f.name for f in npy_dir.iterdir()) == sorted(
        p.name for c in chroms for s in "+-" for p in cache_paths(npy_dir, c, s)
    )
    assert not [f for f in parquet_dir.iterdir() if ".access." in f.name]


def test_npy_output_equals_cache_built_from_parquet(dirs, tmp_path: Path) -> None:
    npy_dir, parquet_dir, chroms = dirs
    for chrom in chroms:
        build_accessibility_cache(
            parquet_dir / f"{chrom}.accessibility.parquet", tmp_path
        )
    _assert_same_cache(npy_dir, tmp_path, chroms)


def test_npy_output_equals_parquet_cache_across_chunks(tmp_path: Path) -> None:
    # Several fold chunks per strand, and GC hairpins whose opening energies
    # exceed 25.5 kcal/mol, so the exceptions sidecar is not empty.
    rng = np.random.default_rng(3)
    hairpin = "CGGGGCCCGGGGCGGGCCGAAAGGCCCGCCCCGGGCCCCG"
    seq = (
        "".join(rng.choice(list("ACGT"), 900))
        + hairpin * 3
        + "".join(rng.choice(list("ACGT"), 1_500))
    )
    args = (seq, "chrT", 80, 40, 30)
    parquet = tmp_path / "chrT.accessibility.parquet"
    _fold_full_chromosome(*args, str(parquet), 700)
    _fold_full_chromosome(*args, str(tmp_path / "npy"), 700, output_format="npy")
    build_accessibility_cache(parquet, tmp_path / "from_parquet")
    _assert_same_cache(tmp_path / "npy", tmp_path / "from_parquet", ["chrT"])
    for strand in "+-":
        with np.load(cache_paths(tmp_path / "npy", "chrT", strand)[1]) as side:
            assert side["index"].size > 0


def test_unknown_output_format_is_rejected(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "accessibility",
            "-f",
            str(GENOME),
            "-o",
            str(tmp_path),
            "--output-format",
            "bin",
        ],
    )
    assert result.exit_code == 1
    assert "--output-format" in result.output


def test_auto_reads_an_npy_directory_as_memmap(dirs, tmp_path: Path) -> None:
    import sioff

    npy_dir, parquet_dir, _ = dirs
    before = sorted(f.name for f in npy_dir.iterdir())
    reference = sioff.off_targets(
        risearch_file=PREDICTIONS, accessibility_dir=parquet_dir
    )
    single = sioff.off_targets(risearch_file=PREDICTIONS, accessibility_dir=npy_dir)
    assert isinstance(reference, pl.DataFrame) and isinstance(single, pl.DataFrame)
    assert single.sort(single.columns).equals(reference.sort(reference.columns))

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / PREDICTIONS.name).write_bytes(PREDICTIONS.read_bytes())
    frames = list(
        sioff.off_targets(risearch_file=in_dir, accessibility_dir=npy_dir, n_workers=1)
    )
    assert len(frames) == 1
    common = [c for c in single.columns if c in frames[0].columns]
    assert (
        frames[0].select(common).sort(common).equals(single.select(common).sort(common))
    )
    assert sorted(f.name for f in npy_dir.iterdir()) == before


@pytest.mark.parametrize("strategy", ["eager", "chrom-major"])
def test_parquet_strategies_refuse_an_npy_directory(
    dirs, tmp_path: Path, strategy: str
) -> None:
    import sioff

    npy_dir = dirs[0]
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / PREDICTIONS.name).write_bytes(PREDICTIONS.read_bytes())
    for source in (PREDICTIONS, in_dir):
        with pytest.raises(AccessibilityError, match="--output-format parquet"):
            out = sioff.off_targets(
                risearch_file=source,
                accessibility_dir=npy_dir,
                accessibility_strategy=strategy,
                n_workers=1,
            )
            if not isinstance(out, pl.DataFrame):
                list(out)
