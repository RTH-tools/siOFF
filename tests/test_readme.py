"""The README's runnable examples work, and its internal links resolve."""

import re
import runpy
import shutil
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from sioff.cli import app

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
EXAMPLES = ROOT / "examples" / "data"


def _section(heading: str) -> str:
    start = README.index(f"\n{heading}\n")
    end = README.find("\n## ", start + 1)
    return README[start : end if end != -1 else None]


@pytest.fixture
def clone_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory laid out like the root of a clone."""
    shutil.copytree(EXAMPLES, tmp_path / "examples" / "data")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_python_api_example_runs_on_example_data(clone_root: Path) -> None:
    blocks = re.findall(r"```python\n(.*?)```", _section("## Python API"), re.S)
    assert len(blocks) == 1
    script = clone_root / "readme_example.py"
    script.write_text(blocks[0])

    namespace = runpy.run_path(str(script))

    df = namespace["df"]
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert "P_off_target" in df.columns


def test_step1_commands_feed_off_targets(clone_root: Path) -> None:
    """Step 1's `sioff index` + `sioff search -o` output is readable by Step 3's `-r`."""
    runner = CliRunner()
    data = "examples/data"
    for args in (
        ["index", f"{data}/genome.fa", "-o", "genome.idx"],
        ["search", f"{data}/sirnas.fa", "genome.idx", "-t", f"{data}/genome.fa",
         "-o", "predictions.tsv"],
        ["off-targets", "-r", "predictions.tsv", "-t", f"{data}/annotation.gtf",
         "-a", f"{data}/accessibility", "-o", "out/off_targets.tsv"],
    ):  # fmt: skip
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (args, result.output)

    out = pl.read_csv(clone_root / "out" / "off_targets.tsv", separator="\t")
    assert out.height > 0
    assert set(out["sirna_id"]) <= {f"siRNA_{i}" for i in range(1, 6)}


def _slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces to dashes."""
    text = heading.strip().lower().replace("`", "")
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def test_readme_internal_links_resolve() -> None:
    prose = re.sub(r"```.*?```", "", README, flags=re.S)
    anchors = {_slug(h) for h in re.findall(r"^#+ (.+)$", prose, re.M)}
    links = set(re.findall(r"\]\(#([^)]+)\)", prose))
    assert links, "no internal links found"
    assert links <= anchors, sorted(links - anchors)
