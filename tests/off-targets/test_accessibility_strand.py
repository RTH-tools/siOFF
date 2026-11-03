"""Single-strand accessibility profiles (``--strand +``).

A transcriptome FASTA already holds each target in its sense orientation, so
the reverse complement need not be folded. A plus-only profile has no ``-``
rows; ``sioff accessibility`` writes no minus pair for it, and a per-record
npy fold carries an empty ``-`` cache, which is what
`build_accessibility_cache` makes from such a Parquet. A minus-strand hit
looked up in it keeps the default opening energy and logs a warning, unless
``sense_only`` drops it before the lookup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

import sioff
import sioff.services.accessibility as acc
from sioff.cli import app
from sioff.commands.accessibility import run as accessibility_run
from sioff.config import config_to_kwargs, load_config
from sioff.services.accessibility import (
    AccessibilityError,
    GenomeAccessibilityService,
    _fold_full_chromosome,
    build_accessibility_cache,
    cache_paths,
    packed_paths,
)

DATA = Path(__file__).parent / "data"
GENOME = DATA / "genome.fa"
FOLD = ["-W", "40", "-L", "20", "-u", "10"]
FOLD_KW = {"window_size": 40, "max_span": 20, "unpaired_prob": 10}
CHROM = "chrT"
N_U = 10


def _random_seq(n: int, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), n))


@pytest.fixture
def warnings_seen(monkeypatch):
    """WARNING messages logged during the test, with the warn-once memory reset."""
    from loguru import logger

    monkeypatch.setattr(acc, "_WARNED_MISSING_STRANDS", set())
    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    yield seen
    logger.remove(handler)


def _missing_strand_warnings(seen: list[str]) -> list[str]:
    return [m for m in seen if "no '-' strand" in m]


def _length_warnings(seen: list[str]) -> list[str]:
    return [m for m in seen if "'+' covers" in m]


# --------------------------------------------------------------------------
# Folding one strand
# --------------------------------------------------------------------------


def test_api_plus_only_returns_the_plus_rows_of_a_both_strand_fold() -> None:
    both = sioff.accessibility(GENOME, **FOLD_KW)
    plus = sioff.accessibility(GENOME, strand="+", **FOLD_KW)
    assert plus.keys() == both.keys()
    for chrom, df in plus.items():
        assert df["strand"].unique().to_list() == ["+"]
        assert df.equals(both[chrom].filter(pl.col("strand") == "+"))


def test_api_rejects_an_unknown_strand() -> None:
    with pytest.raises(AccessibilityError, match="strand"):
        sioff.accessibility(GENOME, strand="-", **FOLD_KW)


def test_plus_only_parquet_holds_no_minus_rows(tmp_path: Path) -> None:
    parquet = tmp_path / f"{CHROM}.accessibility.parquet"
    _fold_full_chromosome(
        _random_seq(400), CHROM, 40, 20, N_U, str(parquet), 150, strand="+"
    )
    df = pl.read_parquet(parquet)
    assert df["strand"].unique().to_list() == ["+"]
    assert df["position"].to_list() == list(range(1, 401))


def test_plus_only_npy_equals_cache_built_from_plus_only_parquet(
    tmp_path: Path,
) -> None:
    args = (_random_seq(1_500), CHROM, 80, 40, 30)
    parquet = tmp_path / f"{CHROM}.accessibility.parquet"
    _fold_full_chromosome(*args, str(parquet), 700, strand="+")
    _fold_full_chromosome(
        *args, str(tmp_path / "npy"), 700, output_format="npy", strand="+"
    )
    build_accessibility_cache(parquet, tmp_path / "from_parquet")
    for strand in "+-":
        (npy_a, npz_a), (npy_b, npz_b) = (
            cache_paths(tmp_path / "npy", CHROM, strand),
            cache_paths(tmp_path / "from_parquet", CHROM, strand),
        )
        assert npy_a.read_bytes() == npy_b.read_bytes(), strand
        with np.load(npz_a) as x, np.load(npz_b) as y:
            for key in ("index", "value"):
                assert x[key].dtype == y[key].dtype
                np.testing.assert_array_equal(x[key], y[key])
    minus = np.load(cache_paths(tmp_path / "npy", CHROM, "-")[0])
    assert minus.shape == (0, 30) and minus.dtype == np.uint8


def test_cli_strand_plus_writes_a_plus_only_profile(tmp_path: Path) -> None:
    runner = CliRunner()
    for fmt in ("npy", "parquet"):
        out = tmp_path / fmt
        result = runner.invoke(
            app,
            [
                "accessibility",
                "-f",
                str(GENOME),
                "-o",
                str(out),
                *FOLD,
                "--output-format",
                fmt,
                "--strand",
                "+",
            ],
        )
        assert result.exit_code == 0, result.output
    npy_dir = tmp_path / "npy"
    assert sorted(f.name for f in npy_dir.iterdir()) == sorted(
        p.name for p in packed_paths(npy_dir, "+")
    )
    assert np.load(packed_paths(npy_dir, "+")[0]).shape[0] > 0
    df = pl.read_parquet(tmp_path / "parquet" / "chr1.accessibility.parquet")
    assert df["strand"].unique().to_list() == ["+"]


def test_cli_rejects_an_unknown_strand(tmp_path: Path, plain) -> None:
    result = CliRunner().invoke(
        app,
        ["accessibility", "-f", str(GENOME), "-o", str(tmp_path), "--strand", "x"],
    )
    assert result.exit_code == 1
    assert "--strand" in plain(result.output)
    assert not any(tmp_path.iterdir())


def test_config_strand_reaches_the_command(tmp_path: Path) -> None:
    import inspect

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "command: accessibility\n"
        "accessibility:\n  fasta: genome.fa\n  output: out\n  strand: '+'\n"
    )
    kwargs = config_to_kwargs(load_config(config_file), "accessibility")
    assert kwargs["strand"] == "+"
    inspect.signature(accessibility_run).bind_partial(**kwargs)


# --------------------------------------------------------------------------
# Looking up a plus-only profile
# --------------------------------------------------------------------------


@pytest.fixture
def profiles(tmp_path: Path):
    """``(plus_only, both)`` dirs, each holding the Parquet of the same fold."""
    seq = _random_seq(300)
    dirs = {}
    for name, strand in (("plus_only", "+"), ("both", "both")):
        d = tmp_path / name
        d.mkdir()
        _fold_full_chromosome(
            seq,
            CHROM,
            40,
            20,
            N_U,
            str(d / f"{CHROM}.accessibility.parquet"),
            strand=strand,
        )
        dirs[name] = d
    return dirs["plus_only"], dirs["both"]


def _hits_frame(hits) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": [CHROM] * len(hits),
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


def _via_service(service, hits) -> np.ndarray:
    out = service.annotate_opening_energy_vectorized(_hits_frame(hits))
    return out["opening_energy"].to_numpy()


def _default_cache_dir(d: Path, hits, tmp: Path) -> np.ndarray:
    return _via_service(GenomeAccessibilityService(d), hits)


def _frames(d: Path, hits, tmp: Path) -> np.ndarray:
    frames = {CHROM: pl.read_parquet(d / f"{CHROM}.accessibility.parquet")}
    return _via_service(GenomeAccessibilityService.from_frames(frames), hits)


def _memmap(d: Path, hits, tmp: Path) -> np.ndarray:
    return _via_service(GenomeAccessibilityService(d, cache_dir=tmp / d.name), hits)


PATHS = {
    "default_cache_dir": _default_cache_dir,
    "from_frames": _frames,
    "memmap": _memmap,
}


@pytest.mark.parametrize("path", list(PATHS))
def test_minus_hit_on_plus_only_profile_warns_once_and_keeps_default(
    profiles, warnings_seen, tmp_path: Path, path: str
) -> None:
    plus_only, both = profiles
    hits = [("+", 50, 60), ("-", 50, 60)]
    reference = PATHS[path](both, hits, tmp_path / "ref")
    got = PATHS[path](plus_only, hits, tmp_path / "cache")
    assert got[0] == reference[0]
    assert got[1] == pytest.approx(10.0)
    PATHS[path](plus_only, hits, tmp_path / "cache")  # warned once per process
    missing = _missing_strand_warnings(warnings_seen)
    assert len(missing) == 1, warnings_seen
    assert "hits on that strand keep the default opening energy of 10.0" in missing[0]
    assert "--sense-only" in missing[0] and "--strand both" in missing[0]
    assert not _length_warnings(warnings_seen)


@pytest.mark.parametrize("path", list(PATHS))
def test_plus_hits_on_plus_only_profile_log_nothing(
    profiles, warnings_seen, tmp_path: Path, path: str
) -> None:
    PATHS[path](profiles[0], [("+", 50, 60), ("+", 3, 30)], tmp_path / "cache")
    assert not warnings_seen


def test_build_cache_from_plus_only_parquet_logs_no_length_warning(
    profiles, warnings_seen, tmp_path: Path
) -> None:
    build_accessibility_cache(
        profiles[0] / f"{CHROM}.accessibility.parquet", tmp_path / "cache"
    )
    assert not _length_warnings(warnings_seen)


@pytest.mark.parametrize("sense_only", [False, True])
def test_off_targets_warns_about_minus_hits_unless_sense_only(
    warnings_seen, sense_only: bool
) -> None:
    profiles = sioff.accessibility(GENOME, strand="+", **FOLD_KW)
    predictions = pl.DataFrame(
        {
            "sirna_id": ["siRNA_1", "siRNA_1"],
            "chrom": ["chr1", "chr1"],
            "start": [10, 30],
            "end": [30, 50],
            "strand": ["+", "-"],
            "energy": [-20.0, -18.0],
        }
    )
    df = sioff.off_targets(
        predictions=predictions, accessibility=profiles, sense_only=sense_only
    )
    assert isinstance(df, pl.DataFrame)
    assert bool(_missing_strand_warnings(warnings_seen)) is not sense_only
