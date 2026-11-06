"""SIGTERM ends an off-targets run through its cleanup.

Slurm sends SIGTERM on a timeout or ``scancel``. Its default action kills the
process at once, so the engine's ``finally`` never ran and the input-sized
``sioff_split_*`` copy stayed in TMPDIR. The CLI turns SIGTERM into
``SystemExit(143)`` for the duration of the run.
"""

import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from typer.testing import CliRunner

from sioff.cli import app
from sioff.commands import off_targets as cmd

EX = Path(__file__).resolve().parents[2] / "examples" / "data"

# Score in this process (-j 1) with a task that waits, so the run is caught
# mid-scoring; the marker says the split parts exist.
_SCRIPT = """
import sys, time
from pathlib import Path
import sioff.core.off_targets as core

marker = Path(sys.argv[1])

def _waits(*args):
    marker.touch()
    time.sleep(120)

core._score_part = _waits
from sioff.cli import app
sys.argv = ["sioff", *sys.argv[2:]]
app()
"""


def _engine_dirs(root: Path) -> list[str]:
    return sorted(
        p.name for p in root.iterdir() if p.name.startswith(("sioff_", "risearch_"))
    )


def test_sigterm_mid_run_removes_the_temp_dirs(tmp_path):
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    marker = tmp_path / "scoring"
    argv = ["off-targets", "-r", str(EX / "predictions.out")]
    argv += ["-t", str(EX / "annotation.gtf"), "-j", "1"]
    argv += ["-o", str(tmp_path / "x.tsv")]
    proc = subprocess.Popen(
        [sys.executable, "-c", _SCRIPT, str(marker), *argv],
        env={**os.environ, "TMPDIR": str(tmp)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while not marker.exists():
            assert proc.poll() is None, proc.communicate()[1]
            assert time.monotonic() < deadline, "the run never started scoring"
            time.sleep(0.05)
        assert [n.split("_")[1] for n in _engine_dirs(tmp)] == ["ipc", "split"]
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 143, stderr
    assert _engine_dirs(tmp) == []


def test_the_handler_is_restored_and_workers_start_without_it():
    before = signal.getsignal(signal.SIGTERM)
    restore = cmd._exit_on_sigterm()
    try:
        assert signal.getsignal(signal.SIGTERM) != before
        with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
            in_worker = pool.submit(signal.getsignal, signal.SIGTERM).result()
        assert in_worker == signal.SIG_DFL
    finally:
        restore()
    assert signal.getsignal(signal.SIGTERM) == before


def test_a_cli_run_restores_the_handler():
    before = signal.getsignal(signal.SIGTERM)
    result = CliRunner().invoke(
        app, ["off-targets", "-r", str(EX / "predictions.out"), "-j", "1"]
    )
    assert result.exit_code == 0, result.output
    assert signal.getsignal(signal.SIGTERM) == before
