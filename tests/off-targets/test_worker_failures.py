"""A run must fail loudly when worker tasks fail.

A run whose every worker was OOM-killed used to print "Warning: No predictions
to process" and exit 0. The engine now raises `WorkerFailureError` once every
task has been attempted (successful siRNAs are still yielded first), and the
CLI turns that into a non-zero exit with a summary. These runs have no
accessibility directory; runs with one are covered in
test_accessibility_directory_mode.py.
"""

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from sioff.cli import app
from sioff.core.off_targets import WorkerFailureError, compute_off_targets

DATA_DIR = Path(__file__).parent / "data"
RISEARCH_FILE = DATA_DIR / "risearch_siRNAID.out"

runner = CliRunner()


@pytest.fixture
def mixed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "in"
    d.mkdir()
    (d / RISEARCH_FILE.name).write_bytes(RISEARCH_FILE.read_bytes())
    (d / "broken.parquet").write_bytes(b"this is not a parquet file")
    return d


@pytest.fixture
def broken_dir(tmp_path: Path) -> Path:
    d = tmp_path / "in"
    d.mkdir()
    (d / "broken.parquet").write_bytes(b"this is not a parquet file")
    return d


def test_core_yields_successes_then_raises(mixed_dir: Path) -> None:
    gen = compute_off_targets(risearch_file=mixed_dir, n_workers=1)
    frames = []
    with pytest.raises(WorkerFailureError) as exc_info:
        for df, _meta in gen:
            frames.append(df)
    assert len(frames) == 1 and isinstance(frames[0], pl.DataFrame)
    err = exc_info.value
    assert err.n_failed == 1 and err.n_total == 2
    assert err.failed_names == ["broken.parquet"]
    assert "1/2" in str(err) and "first error: broken.parquet:" in str(err)


def test_core_raises_when_every_file_fails(broken_dir: Path) -> None:
    with pytest.raises(WorkerFailureError, match="1/1"):
        list(compute_off_targets(risearch_file=broken_dir, n_workers=1))


def test_cli_exits_non_zero_when_workers_fail(
    broken_dir: Path, tmp_path: Path, plain
) -> None:
    out = tmp_path / "out.tsv"
    result = runner.invoke(
        app, ["off-targets", "-r", str(broken_dir), "-j", "1", "-o", str(out)]
    )
    assert result.exit_code != 0
    text = plain(result.output)
    assert "1/1" in text and "broken.parquet" in text
    assert "No predictions to process" not in text


def test_cli_still_writes_successful_sirnas(
    mixed_dir: Path, tmp_path: Path, plain
) -> None:
    out = tmp_path / "out.tsv"
    result = runner.invoke(
        app, ["off-targets", "-r", str(mixed_dir), "-j", "1", "-o", str(out)]
    )
    assert result.exit_code != 0
    assert out.exists() and pl.read_csv(out, separator="\t").height > 0
    assert "1/2" in plain(result.output)


def test_cli_run_where_every_task_fails_leaves_no_stale_table(
    broken_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.tsv"
    out.write_text("a table from an earlier run\n")
    result = runner.invoke(
        app, ["off-targets", "-r", str(broken_dir), "-j", "1", "-o", str(out)]
    )
    assert result.exit_code == 1
    assert not out.exists()
