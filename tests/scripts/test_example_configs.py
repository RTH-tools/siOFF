"""The shipped example configs must actually produce results.

`example_yaml/off-targets.example.yaml` is the front door: README points at it as
the first command to run. It previously exited 0 while writing a header-only
file, because the predictions fixture targeted `transcript_1..5` while the
annotation described `chr1` — disjoint chromosomes *and* disjoint coordinate
ranges. A run that succeeds and produces nothing looks like a working install,
so the failure was invisible.

These tests run the real config through the real CLI and assert it yields rows.
"""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "example_yaml"


@pytest.fixture
def off_targets_config(tmp_path: Path) -> Path:
    """The shipped config, rewritten into tmp_path so the run touches no repo file.

    Relative paths resolve against the config file's own directory, so moving the
    config means rewriting them to absolute paths. Done this way rather than by
    dropping a copy next to the original because the run also emits `.summary`
    files beside its output — an interrupted test would otherwise litter the repo.
    """
    import re

    text = (EXAMPLE_DIR / "off-targets.example.yaml").read_text()
    text = re.sub(
        r"(?m)^(\s*\w+:\s*)(\.\./[\w./-]+)\s*$",
        lambda m: f"{m.group(1)}{(EXAMPLE_DIR / m.group(2)).resolve()}",
        text,
    )
    # Redirect by matching the key rather than a literal path: a hardcoded path
    # silently stops matching when the config is edited, and the redirect then
    # no-ops into the repo instead of tmp_path.
    text, redirected = re.subn(
        r"(?m)^(\s*output:\s*)\./[\w./-]+\.tsv\s*$",
        lambda m: f"{m.group(1)}{tmp_path / 'out.tsv'}",
        text,
    )
    assert redirected == 1, (
        f"expected exactly one relative .tsv output key to redirect, found {redirected} — "
        "the example config's output key changed shape"
    )

    config = tmp_path / "off-targets.yaml"
    config.write_text(text)
    return config


def test_shipped_off_targets_example_produces_rows(off_targets_config, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "sioff.cli", "-c", str(off_targets_config)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    out = tmp_path / "out.tsv"
    assert out.exists(), "example config produced no output file"
    # The default strategy never builds the memory-mapped accessibility cache,
    # so the run must leave the shipped examples/ tree untouched.
    acc_dir = REPO_ROOT / "examples" / "data" / "accessibility"
    assert not [f for f in acc_dir.iterdir() if ".access." in f.name], (
        "the example run wrote accessibility cache files into the repo tree"
    )

    df = pl.read_csv(out, separator="\t")
    assert df.height > 0, (
        "the shipped example intersected nothing — the predictions fixture and the "
        "annotation no longer overlap"
    )


def test_shipped_example_output_carries_a_usable_probability_model(
    off_targets_config, tmp_path
):
    """Rows alone are not enough: the scientific columns must be populated.

    Guards the class of bug where annotation parses but expression does not, which
    zeroes ``W_i = Expression_i * exp(-dG/RT)`` and collapses every probability.
    """
    subprocess.run(
        [sys.executable, "-m", "sioff.cli", "-c", str(off_targets_config)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )

    df = pl.read_csv(tmp_path / "out.tsv", separator="\t")

    # Checked first so an empty result reports as such rather than as a TypeError
    # from aggregating an empty column.
    assert df.height > 0, "example produced a header-only file"

    for column in ("gene_id", "exp_value", "energy", "P_off_target"):
        assert column in df.columns, f"missing {column}"

    assert df["gene_id"].null_count() == 0
    assert df["exp_value"].min() > 0.0, "expression values all zero"
    assert df["energy"].max() < 0.0, "duplex energies should be negative"
    assert 0.0 < df["P_off_target"].max() <= 1.0


def test_example_configs_reference_files_that_exist():
    """Every relative path in the shipped configs must resolve."""
    import re

    missing: list[str] = []
    for config in sorted(EXAMPLE_DIR.glob("*.example.yaml")):
        for line in config.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            match = re.search(r":\s*(\.\./[\w./-]+)\s*$", line)
            if match and not (config.parent / match.group(1)).resolve().exists():
                missing.append(f"{config.name}: {match.group(1)}")

    assert not missing, f"example configs point at missing files: {missing}"
