"""Tests for the public Python API (``import sioff``).

These guard the Annotated-Typer refactor that makes the CLI command functions
directly callable from plain Python. The key regression is that parameter
defaults must be real values (e.g. ``None``, ``"1.0"``) rather than Typer
``OptionInfo`` / ``ArgumentInfo`` placeholder objects, so that calling the
functions with ordinary keyword arguments works without OptionInfo leakage.
"""

import inspect
from pathlib import Path

import polars as pl
import pytest

import sioff

DATA_DIR = Path(__file__).parent / "data"
RISEARCH_FILE = DATA_DIR / "risearch_siRNAID.out"
GENOME_FASTA = DATA_DIR / "genome.fa"


def test_public_api_exposes_callables() -> None:
    """``import sioff`` exposes off_targets, accessibility, index, search as callables."""
    for name in ("off_targets", "accessibility", "index", "search"):
        assert hasattr(sioff, name), f"sioff is missing public attribute {name!r}"
        assert callable(getattr(sioff, name)), f"sioff.{name} is not callable"


def test_off_targets_has_no_accessibility_strategy() -> None:
    with pytest.raises(TypeError, match="accessibility_strategy"):
        sioff.off_targets(risearch_file=RISEARCH_FILE, accessibility_strategy="memmap")


def test_off_targets_returns_dataframe_with_probabilities() -> None:
    """sioff.off_targets(...) on a real single-file fixture returns a non-empty
    Polars DataFrame containing the probability column.

    This is the load-bearing regression: if any parameter default were still a
    Typer OptionInfo object, the call would not reach the energy-only probability
    path (it would raise typer.Exit) and there would be no P_off_target column.

    Energy-only path (no transcriptome): the fixture's chrom values
    (transcript_1..5) intersect neither the GTF transcript ids nor chr1, so a
    transcriptome join would zero out the frame. Passing only the predictions
    file exercises the single-siRNA probability calculation directly.
    """
    assert RISEARCH_FILE.exists(), f"Fixture not found: {RISEARCH_FILE}"

    df = sioff.off_targets(risearch_file=RISEARCH_FILE)

    assert isinstance(df, pl.DataFrame), f"Expected pl.DataFrame, got {type(df)!r}"
    assert df.height > 0, "Expected a non-empty result DataFrame"
    assert "P_off_target" in df.columns, (
        f"Expected a P_off_target column; got columns: {df.columns}"
    )


def test_off_targets_signature_has_no_typer_placeholder_defaults() -> None:
    """No parameter default of sioff.off_targets is a Typer OptionInfo/ArgumentInfo.

    With the modern Annotated idiom, the typer.Option(...) metadata lives in the
    annotation, and the parameter default is the real value. This guards against
    regressing back to ``param = typer.Option(default, ...)``.
    """
    from typer.models import ArgumentInfo, OptionInfo

    sig = inspect.signature(sioff.off_targets)
    offenders = []
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            continue
        if isinstance(param.default, (OptionInfo, ArgumentInfo)):
            offenders.append(name)

    assert not offenders, (
        "These parameters still default to a Typer placeholder object "
        f"(OptionInfo/ArgumentInfo): {offenders}"
    )


def test_accessibility_returns_chrom_to_dataframe(tmp_path: Path) -> None:
    """sioff.accessibility(...) returns a dict mapping chromosome -> in-memory
    DataFrame (schema [position, strand, u1..u{u}]) and writes NO files.

    Uses small RNAplfold parameters (W=10, L=5, u=3) suited to the short
    (~71 nt) genome fixture, matching the values used in test_accessibility.py.
    """
    if not GENOME_FASTA.exists():
        pytest.skip(f"Genome FASTA fixture not found: {GENOME_FASTA}")

    files_before = set(tmp_path.iterdir())
    result = sioff.accessibility(
        genome=GENOME_FASTA,
        window_size=10,
        max_span=5,
        unpaired_prob=3,
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)!r}"
    assert result, "Expected at least one chromosome in the result mapping"
    for chrom, df in result.items():
        assert isinstance(chrom, str), f"Expected str chromosome key, got {chrom!r}"
        assert isinstance(df, pl.DataFrame), f"Expected pl.DataFrame, got {type(df)!r}"
        assert df.columns[:2] == ["position", "strand"], (
            f"Unexpected leading columns: {df.columns}"
        )
        assert {"u1", "u2", "u3"}.issubset(df.columns), (
            f"Expected u1..u3 columns; got {df.columns}"
        )
        assert set(df["strand"].unique()) <= {"+", "-"}
        assert df.height > 0

    # The in-memory API must not write anything to disk.
    assert set(tmp_path.iterdir()) == files_before, "accessibility() wrote files"


def test_off_targets_accepts_str_path() -> None:
    """Path parameters accept plain ``str`` (the README/scripting style), not only
    ``Path``. A direct Python call bypasses Typer's str->Path coercion, so the
    command body coerces str paths itself.
    """
    df = sioff.off_targets(risearch_file=str(RISEARCH_FILE))
    assert isinstance(df, pl.DataFrame), f"Expected pl.DataFrame, got {type(df)!r}"
    assert df.height > 0, "Expected a non-empty result DataFrame"


def test_off_targets_directory_mode_yields_dataframes(tmp_path: Path) -> None:
    """A *directory* of per-siRNA prediction files returns a generator yielding
    one in-memory DataFrame per siRNA — and writes NO files (the in-memory API).

    This is the contract change from the earlier re-export, which returned a
    summary dict and streamed to disk. File writing is now the CLI's job.
    """
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / RISEARCH_FILE.name).write_bytes(RISEARCH_FILE.read_bytes())

    files_before = set(tmp_path.rglob("*"))
    gen = sioff.off_targets(risearch_file=str(in_dir))

    import collections.abc

    assert isinstance(gen, collections.abc.Iterator), (
        f"Expected a generator/iterator, got {type(gen)!r}"
    )

    frames = list(gen)
    assert frames, "Expected at least one per-siRNA DataFrame"
    for df in frames:
        assert isinstance(df, pl.DataFrame), f"Expected pl.DataFrame, got {type(df)!r}"
        assert df.height > 0
        assert "P_off_target" in df.columns

    # The in-memory API must not write anything to disk.
    assert set(tmp_path.rglob("*")) == files_before, "directory mode wrote files"


def test_off_targets_no_input_raises_value_error() -> None:
    """Bad input raises a plain ValueError, not a Typer/Click Exit."""
    import typer

    with pytest.raises(ValueError):
        sioff.off_targets()

    # Defensive: the raised exception must not be a Typer/Click Exit.
    try:
        sioff.off_targets()
    except typer.Exit:  # pragma: no cover - should never hit
        pytest.fail("API leaked a typer.Exit instead of a plain exception")
    except ValueError:
        pass


def test_accessibility_missing_genome_raises(tmp_path: Path) -> None:
    """A missing genome raises FileNotFoundError (not typer.Exit)."""
    with pytest.raises(FileNotFoundError):
        sioff.accessibility(genome=tmp_path / "does_not_exist.fa")


# ---------------------------------------------------------------------------
# In-memory predictions: sioff.search(...) -> sioff.off_targets(predictions=...)
# ---------------------------------------------------------------------------


def test_off_targets_accepts_a_predictions_dataframe() -> None:
    """A DataFrame with the search schema is accepted in place of a file and
    gives the same result as loading that file from disk."""
    from sioff.services.risearch_parser import RIsearchParser

    predictions = RIsearchParser().load(RISEARCH_FILE)
    from_frame = sioff.off_targets(predictions=predictions)
    from_file = sioff.off_targets(risearch_file=RISEARCH_FILE)

    assert isinstance(from_frame, pl.DataFrame)
    assert from_frame.height == from_file.height > 0
    assert from_frame.columns == from_file.columns
    assert from_frame.sort(from_frame.columns).equals(from_file.sort(from_file.columns))


def test_off_targets_predictions_frame_may_carry_extra_columns_and_dtypes() -> None:
    """Only the six search columns matter; extra columns are ignored and
    compatible dtypes (Int64 coordinates, Float64 energy) are cast."""
    from sioff.services.risearch_parser import RIsearchParser

    predictions = (
        RIsearchParser()
        .load(RISEARCH_FILE)
        .cast({"start": pl.Int64, "end": pl.Int64, "energy": pl.Float64})
        .with_columns(pl.lit("x").alias("extra"))
    )
    df = sioff.off_targets(predictions=predictions)
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert "P_off_target" in df.columns


def test_off_targets_predictions_frame_missing_columns_raises() -> None:
    bad = pl.DataFrame({"sirna_id": ["a"], "chrom": ["c"], "energy": [-20.0]})
    with pytest.raises(ValueError, match="start"):
        sioff.off_targets(predictions=bad)


def test_off_targets_predictions_and_file_together_raise() -> None:
    """Two prediction sources are ambiguous, not silently merged or preferred."""
    from sioff.services.risearch_parser import RIsearchParser

    predictions = RIsearchParser().load(RISEARCH_FILE)
    with pytest.raises(ValueError, match="predictions"):
        sioff.off_targets(predictions=predictions, risearch_file=RISEARCH_FILE)


def test_off_targets_accepts_in_memory_accessibility(tmp_path: Path) -> None:
    """sioff.accessibility -> sioff.off_targets(accessibility=...) matches the
    Parquet-directory path built from the same profiles, and writes nothing."""
    profiles = sioff.accessibility(
        GENOME_FASTA, window_size=40, max_span=20, unpaired_prob=10
    )
    acc_dir = tmp_path / "acc"
    acc_dir.mkdir()
    for chrom, frame in profiles.items():
        frame.write_parquet(acc_dir / f"{chrom}.accessibility.parquet")

    genome_predictions = DATA_DIR / "risearch_genome_example.out"
    from_dir = sioff.off_targets(
        risearch_file=genome_predictions, accessibility_dir=acc_dir
    )
    files_before = set(tmp_path.rglob("*"))
    from_frames = sioff.off_targets(
        risearch_file=genome_predictions, accessibility=profiles
    )

    assert isinstance(from_frames, pl.DataFrame)
    assert "opening_energy" in from_frames.columns
    assert from_frames.sort(from_frames.columns).equals(from_dir.sort(from_dir.columns))
    assert set(tmp_path.rglob("*")) == files_before, "in-memory run wrote files"


def test_off_targets_accessibility_and_dir_together_raise(tmp_path: Path) -> None:
    profiles = {"chr1": pl.DataFrame({"position": [1], "strand": ["+"], "u1": [1.0]})}
    with pytest.raises(ValueError, match="accessibility"):
        sioff.off_targets(
            risearch_file=RISEARCH_FILE,
            accessibility=profiles,
            accessibility_dir=tmp_path,
        )


def test_off_targets_directory_mode_rejects_in_memory_accessibility(
    tmp_path: Path,
) -> None:
    """Directory mode streams per-siRNA files through worker processes that
    read profiles from disk; in-memory profiles are a single-frame feature."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / RISEARCH_FILE.name).write_bytes(RISEARCH_FILE.read_bytes())
    profiles = {"chr1": pl.DataFrame({"position": [1], "strand": ["+"], "u1": [1.0]})}
    with pytest.raises(ValueError, match="accessibility_dir"):
        sioff.off_targets(risearch_file=in_dir, accessibility=profiles)


def test_search_output_feeds_off_targets_in_memory(tmp_path: Path) -> None:
    """The advertised round trip: sioff.search -> sioff.off_targets, no files."""
    pytest.importorskip("risearch")

    idx = sioff.index(GENOME_FASTA, tmp_path / "genome.idx")
    hits = sioff.search(DATA_DIR / "sirnas.fa", idx, target=GENOME_FASTA)
    assert hits.height > 0

    files_before = set(tmp_path.rglob("*"))
    df = sioff.off_targets(predictions=hits)

    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert "P_off_target" in df.columns
    assert set(tmp_path.rglob("*")) == files_before, "in-memory run wrote files"
