"""Spawned workers log at the level the parent's CLI chose, not loguru's DEBUG.

A spawned worker starts a fresh interpreter with loguru's default DEBUG sink on
stderr; the CLI's WARNING handler does not cross spawn. A plain run with
``-j 4`` printed 29 INFO/DEBUG lines for the 5 example siRNAs, tens of
thousands on a genome. The workers write to the real stderr, so these runs go
through a subprocess: CliRunner only swaps the parent's ``sys.stderr``.
"""

import subprocess
import sys
from pathlib import Path

from loguru import logger

from sioff.core.off_targets import compute_off_targets

EX = Path(__file__).resolve().parents[2] / "examples" / "data"


def _cli(tmp_path: Path, *global_opts: str) -> subprocess.CompletedProcess:
    argv = [
        *global_opts,
        "off-targets",
        "-r",
        str(EX / "predictions.out"),
        "-t",
        str(EX / "annotation.gtf"),
        "-a",
        str(EX / "accessibility"),
        "-o",
        str(tmp_path / "x.tsv"),
        "-j",
        "2",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "sioff.cli", *argv], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return proc


def test_workers_print_no_info_or_debug_lines(tmp_path):
    stderr = _cli(tmp_path).stderr
    assert "| INFO" not in stderr and "| DEBUG" not in stderr, stderr


def test_workers_print_debug_lines_under_verbose(tmp_path):
    # Only a worker intersects: the 5 siRNAs are scored in the pool.
    assert "Intersecting chrom=" in _cli(tmp_path, "-v").stderr


def test_an_in_process_run_keeps_the_parents_handlers():
    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(m.record["message"]), level="DEBUG")
    try:
        list(
            compute_off_targets(
                risearch_file=EX / "predictions.out",
                gtf_file=EX / "annotation.gtf",
                n_workers=1,
            )
        )
    finally:
        logger.remove(handler)
    assert any(m.startswith("Intersecting chrom=") for m in seen)
