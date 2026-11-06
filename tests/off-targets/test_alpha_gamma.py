"""alpha/gamma sweeps: alpha > gamma pairs are skipped and listed; a value no pair uses is an error."""

from pathlib import Path

import pytest
from loguru import logger
from typer.testing import CliRunner

from sioff._logging import setup_logging
from sioff.cli import app
from sioff.core.off_targets import (
    _build_alpha_gamma_pairs,
    _skipped_alpha_gamma_pairs,
    compute_off_targets,
)

runner = CliRunner()
EX = Path(__file__).resolve().parents[2] / "examples" / "data"

BENCH_A = "0.5;0.65;0.7;0.75;0.8;0.85;0.9;0.95;1"
BENCH_G = "0.55;0.65;0.7;0.75;0.8;0.85;0.9;0.95;1"


def test_benchmark_grid_keeps_45_pairs():
    pairs = _build_alpha_gamma_pairs(BENCH_A, BENCH_G)
    assert len(pairs) == 45  # includes (1.0, 1.0)


def test_example_yaml_sweep_runs():
    assert _build_alpha_gamma_pairs("0.8;0.9;1.0", "0.9;1.0") == [
        (1.0, 1.0),
        (0.8, 0.9),
        (0.8, 1.0),
        (0.9, 0.9),
        (0.9, 1.0),
    ]


def test_skipped_pairs_are_listed():
    assert _skipped_alpha_gamma_pairs("0.8;0.9", "0.85") == [(0.9, 0.85)]


def test_nothing_skipped_when_every_pair_is_kept():
    assert _skipped_alpha_gamma_pairs("0.8", "0.9;1.0") == []


def test_alpha_used_by_no_pair_is_an_error():
    with pytest.raises(ValueError, match="--alpha 0.95 pairs with no --gamma"):
        _build_alpha_gamma_pairs("0.95", "0.9")


def test_gamma_used_by_no_pair_is_an_error():
    with pytest.raises(ValueError, match="--gamma 0.5 pairs with no --alpha"):
        _build_alpha_gamma_pairs("0.8", "0.5;0.9")


def test_baseline_uses_no_value():
    with pytest.raises(ValueError, match="--alpha 1.0"):
        _build_alpha_gamma_pairs("1.0", "0.9")


def test_a_default_value_says_it_is_the_default():
    """--gamma 0.8 alone: the user never typed the 1.0 the error names."""
    with pytest.raises(ValueError) as info:
        _build_alpha_gamma_pairs("1.0", "0.8")
    assert str(info.value) == (
        "--alpha 1.0 pairs with no --gamma value >= it "
        "(1.0 is the default when --alpha is not given)"
    )


def test_a_typed_value_is_not_called_the_default():
    with pytest.raises(ValueError) as info:
        _build_alpha_gamma_pairs("0.95", "0.9")
    assert "default" not in str(info.value)


def _cli(*extra: str, global_opts: tuple[str, ...] = ()):
    return runner.invoke(
        app,
        [*global_opts, "off-targets", "-r", str(EX / "predictions.out")]
        + ["-t", str(EX / "annotation.gtf")]
        + ["--sirna-table", str(EX / "sirna_table.tsv")]
        + list(extra),
    )


@pytest.mark.parametrize("global_opts", [(), ("-v",)], ids=["plain", "verbose"])
def test_cli_prints_skipped_pairs_once(plain, global_opts):
    """Once in any form: the console line, or under sioff -v the engine's."""
    try:
        result = _cli(
            "--alpha", "0.8;0.9", "--gamma", "0.85;0.95", global_opts=global_opts
        )
    finally:
        setup_logging(False)  # leave no DEBUG handler on CliRunner's stream
    assert result.exit_code == 0, result.output
    out = plain(result.output)
    assert out.count("Skipped") == 1, out
    assert "(0.9, 0.85)" in out


def test_cli_names_the_default_it_pairs(plain):
    result = _cli("--gamma", "0.8")
    assert result.exit_code == 1
    assert "(1.0 is the default when --alpha is not given)" in " ".join(
        plain(result.output).split()
    )


def test_cli_prints_nothing_when_nothing_is_skipped(plain):
    result = _cli("--alpha", "0.8", "--gamma", "0.9")
    assert result.exit_code == 0, result.output
    assert "Skipped" not in plain(result.output)


def test_cli_rejects_a_value_no_pair_uses(plain):
    result = _cli("--alpha", "0.95", "--gamma", "0.9")
    assert result.exit_code == 1
    assert "--alpha 0.95 pairs with no --gamma value >= it" in plain(result.output)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_engine_logs_skipped_pairs_in_the_parent(n_workers):
    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(m.record["message"]), level="INFO")
    try:
        list(
            compute_off_targets(
                risearch_file=EX / "predictions.out",
                gtf_file=EX / "annotation.gtf",
                sirna_table=EX / "sirna_table.tsv",
                alpha="0.8;0.9",
                gamma="0.85;0.95",
                n_workers=n_workers,
            )
        )
    finally:
        logger.remove(handler)
    assert seen.count("Skipped alpha > gamma pairs: [(0.9, 0.85)]") == 1
