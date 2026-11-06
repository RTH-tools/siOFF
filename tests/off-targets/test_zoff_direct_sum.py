"""Zoff is the off-target rows' weight sum, not Z - W_on.

When the on-target dominates a siRNA's Z, Z - W_on cancels: in double
precision every digit is gone once W_on/Zoff passes about 1e15 (about 1e7
in single precision), and Zoff came out as 0.0 or a power of two. A -40
kcal/mol on-target against -17 kcal/mol off-targets is already there. The
off-target sums are now aggregated directly, next to W_on.
"""

import math
import re
from pathlib import Path

import polars as pl
import pytest

from sioff.core.off_targets import compute_off_targets
from sioff.services.probability import ProbabilityService

EX = Path(__file__).resolve().parents[2] / "examples" / "data"
AG = [(0.9, 1.0)]
THETA = [1.2]
SUFFIXES = {
    "": "alpha=1.0 and gamma=1.0",
    ":alpha=0.9,gamma=1.0": "alpha=0.9 and gamma=1.0",
    ":theta=1.2": "theta=1.2",
}
FIELDS = re.compile(r"(\w+): ([^;\s]+)")


def _summary_values(text: str) -> dict[str, dict[str, float]]:
    """``{preamble: {field: value}}`` for each ``# For ...;`` line."""
    out = {}
    for line in text.splitlines():
        if line.startswith("# For "):
            head, rest = line[len("# For ") :].split(";", 1)
            out[head] = {k: float(v) for k, v in FIELDS.findall(rest)}
    return out


def _frame() -> pl.DataFrame:
    """A -40 kcal/mol on-target (t_on) and three -17 kcal/mol off-targets."""
    return pl.DataFrame(
        {
            "sirna_id": ["si"] * 4,
            "transcript_id": ["t_on", "t1", "t2", "t3"],
            "energy": pl.Series([-40.0, -17.0, -17.0, -17.0], dtype=pl.Float32),
            "opening_energy": pl.Series([0.0, 0.5, 1.0, 0.0], dtype=pl.Float32),
            "exp_value": pl.Series([1.0, 2.0, 1.0, 3.0], dtype=pl.Float32),
        }
    )


def _score(on_target: str = "t_on", aggregate_only: bool = False):
    return ProbabilityService(None).calculate_probabilities_per_sirna(
        _frame(),
        alpha_gamma_pairs=AG,
        theta_values=THETA,
        on_target_map={"si": on_target},
        e_min={"si": -40.0},
        aggregate_only=aggregate_only,
    )


def _dominated(on_target: str = "t_on"):
    out, meta = _score(on_target)
    svc = ProbabilityService(None)
    summary = svc.format_legacy_summary(
        "si",
        meta["z_per_sirna"]["si"],
        meta["on_target_weights"]["si"],
        AG,
        THETA,
    )
    return out, meta, _summary_values(summary)


def _off_sum(out: pl.DataFrame, col: str) -> float:
    return math.fsum(out.filter(~pl.col("is_on_target"))[col].to_list())


@pytest.mark.parametrize("suffix", list(SUFFIXES))
def test_summary_zoff_is_the_off_target_weight_sum(suffix: str) -> None:
    out, _, summary = _dominated()
    line = summary[SUFFIXES[suffix]]

    for field, col in (
        ("Zoff", f"boltzmann_weight{suffix}"),
        ("Zoff_noacc", f"boltzmann_weight_noacc{suffix}"),
    ):
        expected = _off_sum(out, col)
        assert expected > 0
        assert line[field] != 0.0, (suffix, field)
        assert line[field] == pytest.approx(expected, rel=1e-12), (suffix, field)


@pytest.mark.parametrize("aggregate_only", [False, True])
def test_metadata_carries_the_off_target_sums(aggregate_only: bool) -> None:
    out, _ = _score()
    _, meta = _score(aggregate_only=aggregate_only)
    weights = meta["on_target_weights"]["si"]

    for suffix in SUFFIXES:
        for kind in ("", "_noacc"):
            got = weights[f"W_off{kind}{suffix}"]
            expected = _off_sum(out, f"boltzmann_weight{kind}{suffix}")
            assert got == pytest.approx(expected, rel=1e-12), (suffix, kind)


def test_without_an_on_target_row_zoff_is_z() -> None:
    _, _, summary = _dominated(on_target="t_none")

    for head in SUFFIXES.values():
        line = summary[head]
        assert line["Zoff"] == line["Z"], head
        assert line["Zoff_noacc"] == line["Z_noacc"], head


def test_without_an_on_target_map_zoff_is_z() -> None:
    svc = ProbabilityService(None)
    _, meta = svc.calculate_probabilities_per_sirna(_frame())
    summary = _summary_values(
        svc.format_legacy_summary("si", meta["z_per_sirna"]["si"], {}, [], [])
    )

    line = summary["alpha=1.0 and gamma=1.0"]
    assert line["Zoff"] == line["Z"]
    assert line["Zoff_noacc"] == line["Z_noacc"]


def test_example_data_zoff_agrees_with_z_minus_w_on_where_that_is_exact() -> None:
    svc = ProbabilityService(None)
    compared = skipped = 0
    for _, meta in compute_off_targets(
        risearch_file=EX / "predictions.out",
        gtf_file=EX / "annotation.gtf",
        accessibility_dir=EX / "accessibility",
        sirna_table=EX / "sirna_table.tsv",
        alpha="0.8",
        gamma="0.8",
        theta="1.2",
        n_workers=1,
    ):
        for sid, z_row in meta["z_per_sirna"].items():
            w = meta["on_target_weights"][sid]
            summary = _summary_values(
                svc.format_legacy_summary(sid, z_row, w, [(0.8, 0.8)], [1.2])
            )
            for suffix, head in (
                ("", "alpha=1.0 and gamma=1.0"),
                (":alpha=0.8,gamma=0.8", "alpha=0.8 and gamma=0.8"),
                (":theta=1.2", "theta=1.2"),
            ):
                for kind in ("", "_noacc"):
                    zoff = summary[head][f"Zoff{kind}"]
                    w_on = w[f"W_on{kind}{suffix}"]
                    assert zoff > 0, (sid, head, kind)
                    if w_on / zoff >= 1e6:
                        skipped += 1
                        continue
                    z = z_row[f"Z_sirna{kind}{suffix}"]
                    assert zoff == pytest.approx(z - w_on, rel=1e-9), (sid, head)
                    compared += 1

    # 5 siRNAs x 3 parameter sets x acc/noacc; some on-targets dominate.
    assert compared + skipped == 30 and compared >= 10 and skipped >= 1, (
        compared,
        skipped,
    )


@pytest.mark.parametrize("aggregate_only", [False, True])
def test_a_row_without_ids_counts_as_off_target(aggregate_only: bool) -> None:
    # A null transcript_id leaves is_on_target null for the mapped siRNA.
    df = _frame().with_columns(
        pl.Series("transcript_id", ["t_on", "t1", None, "t3"], dtype=pl.Utf8)
    )
    kw = dict(on_target_map={"si": "t_on"}, e_min={"si": -40.0})
    svc = ProbabilityService(None)
    out, _ = svc.calculate_probabilities_per_sirna(df, **kw)
    _, meta = svc.calculate_probabilities_per_sirna(
        df, aggregate_only=aggregate_only, **kw
    )

    assert out["is_on_target"].null_count() == 1
    off = out.filter(~pl.col("is_on_target").fill_null(False))
    w = meta["on_target_weights"]["si"]
    for kind in ("", "_noacc"):
        expected = math.fsum(off[f"boltzmann_weight{kind}"].to_list())
        assert w[f"W_off{kind}"] == pytest.approx(expected, rel=1e-12), kind
        z = meta["z_per_sirna"]["si"][f"Z_sirna{kind}"]
        assert w[f"W_on{kind}"] + w[f"W_off{kind}"] == pytest.approx(z, rel=1e-12)
    assert meta["n_on_target_rows"] == {"si": 1}
