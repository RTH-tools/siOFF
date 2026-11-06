"""Strong duplexes do not overflow the Boltzmann weights.

``energy`` is read as Float32, and the weights used to be computed in it too:
Float32 tops out at about 3.4e38, so at 37 °C (RT ~ 0.6163 kcal/mol) a hit
below about -54.7 kcal/mol, or a weaker one with a large ``exp_value``, gave
an infinite weight and an infinite Z. The strong hit's probability was then
NaN and every other hit of the siRNA got 0. The weights, Z and the on-target
sums are now Float64; the probabilities are written as Float32.
"""

import math
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sioff.cli import app
from sioff.core.off_targets import compute_off_targets
from sioff.services.probability import R, ProbabilityService

RT = R * (37.0 + 273.15)
SUMS = ("boltzmann_weight", "Z_sirna")
FLOATS = re.compile(r"(\w+): ([^;\s]+)")


def _frame(
    energies: list[float],
    exp_values: list[float] | None = None,
    sirna_id: str = "miR",
) -> pl.DataFrame:
    n = len(energies)
    return pl.DataFrame(
        {
            "sirna_id": [sirna_id] * n,
            "transcript_id": [f"t{i}" for i in range(n)],
            "energy": pl.Series(energies, dtype=pl.Float32),
            "opening_energy": pl.Series([0.0] * n, dtype=pl.Float32),
            "exp_value": pl.Series(exp_values or [1.0] * n, dtype=pl.Float32),
        }
    )


def _expected_p(energies: list[float], exp_values: list[float]) -> list[float]:
    """P from the Float32-rounded energies, in Python's double precision."""
    weights = [
        e * math.exp(-float(np.float32(dg)) / RT) for dg, e in zip(energies, exp_values)
    ]
    z = math.fsum(weights)
    return [w / z for w in weights]


def _assert_finite(df: pl.DataFrame) -> None:
    cols = [c for c in df.columns if c.startswith((*SUMS, "P_off_target"))]
    assert cols
    for c in cols:
        assert df[c].is_finite().all(), (c, df[c].to_list())


def _assert_z_is_the_weight_sum(df: pl.DataFrame) -> None:
    for c in df.columns:
        if not c.startswith("boltzmann_weight"):
            continue
        z_col = c.replace("boltzmann_weight", "Z_sirna", 1)
        for (sid,), part in df.group_by("sirna_id"):
            assert part[z_col].n_unique() == 1
            assert part[z_col][0] == pytest.approx(
                math.fsum(part[c].to_list()), rel=1e-12
            ), (sid, z_col)


def _assert_meta_finite(meta: dict) -> None:
    for group in ("z_per_sirna", "on_target_weights"):
        for sid, row in meta[group].items():
            for k, v in row.items():
                if k != "sirna_id":
                    assert math.isfinite(v), (group, sid, k, v)


def test_a_hit_beyond_the_float32_range_keeps_every_probability() -> None:
    energies = [-58.9, -20.0, -15.0]
    out, meta = ProbabilityService(None).calculate_probabilities_per_sirna(
        _frame(energies)
    )

    assert out["P_off_target"].to_list() == pytest.approx(
        _expected_p(energies, [1.0] * 3), rel=1e-6
    )
    assert out["P_off_target"].to_list() == pytest.approx(
        [1.0, 3.9e-28, 1.2e-31], rel=0.05
    )
    _assert_finite(out)
    _assert_z_is_the_weight_sum(out)
    _assert_meta_finite(meta)


def test_dtypes_weights_and_z_double_probabilities_single() -> None:
    out, _ = ProbabilityService(None).calculate_probabilities_per_sirna(
        _frame([-58.9, -20.0, -15.0]),
        alpha_gamma_pairs=[(0.8, 0.8)],
        theta_values=[1.2],
        e_min={"miR": -60.0},
    )
    for c in out.columns:
        if c.startswith(SUMS):
            assert out[c].dtype == pl.Float64, c
        elif c.startswith("P_off_target"):
            assert out[c].dtype == pl.Float32, c
    for c in ("energy", "dG_total", "opening_energy"):
        assert out[c].dtype == pl.Float32, c


def test_a_large_expression_does_not_overflow_the_weight() -> None:
    # 1e5 * exp(50 / RT) ~ 1.6e40: past Float32 although exp() alone fits.
    energies, exp_values = [-50.0, -20.0], [1e5, 1.0]
    out, meta = ProbabilityService(None).calculate_probabilities_per_sirna(
        _frame(energies, exp_values)
    )

    assert out["boltzmann_weight"][0] == pytest.approx(
        1e5 * math.exp(-float(np.float32(-50.0)) / RT), rel=1e-12
    )
    assert out["P_off_target"].to_list() == pytest.approx(
        _expected_p(energies, exp_values), rel=1e-6
    )
    _assert_finite(out)
    _assert_meta_finite(meta)


def test_many_strong_hits_do_not_overflow_z() -> None:
    # Each weight (~1e38) fits in Float32; 200 of them do not.
    out, meta = ProbabilityService(None).calculate_probabilities_per_sirna(
        _frame([-54.0] * 200)
    )

    assert out["boltzmann_weight"].max() < np.finfo(np.float32).max
    _assert_finite(out)
    _assert_z_is_the_weight_sum(out)
    _assert_meta_finite(meta)
    assert out["P_off_target"].to_list() == pytest.approx([1 / 200] * 200, rel=1e-6)


def test_every_alpha_gamma_and_theta_suffix_stays_finite() -> None:
    # (0.8, 1.0) clamps the strong hit to E_min itself, which overflows
    # Float32 on its own; theta=1.2 stretches it to -68.7 kcal/mol.
    df = pl.concat([_frame([-58.9, -20.0, -15.0]), _frame([-25.0, -12.0], None, "s")])
    out, meta = ProbabilityService(None).calculate_probabilities_per_sirna(
        df,
        alpha_gamma_pairs=[(0.8, 0.8), (0.8, 1.0)],
        theta_values=[1.2],
        e_min={"miR": -60.0, "s": -60.0},
    )

    p_cols = [c for c in out.columns if c.startswith("P_off_target")]
    assert len(p_cols) == 4
    _assert_finite(out)
    _assert_z_is_the_weight_sum(out)
    _assert_meta_finite(meta)
    sums = out.group_by("sirna_id").agg(pl.col(p_cols).cast(pl.Float64).sum())
    for c in p_cols:
        assert sums[c].to_list() == pytest.approx([1.0, 1.0], rel=1e-6), c


def test_aggregate_only_gives_the_same_finite_sums() -> None:
    kw = dict(
        alpha_gamma_pairs=[(0.8, 0.8), (0.8, 1.0)],
        theta_values=[1.2],
        on_target_map={"miR": "t0", "s": "t1"},
        e_min={"miR": -60.0, "s": -60.0},
    )
    df = pl.concat([_frame([-58.9, -20.0, -15.0]), _frame([-25.0, -12.0], None, "s")])
    svc = ProbabilityService(None)
    _, full = svc.calculate_probabilities_per_sirna(df, **kw)
    agg_df, agg = svc.calculate_probabilities_per_sirna(df, aggregate_only=True, **kw)

    assert agg_df.height == 0
    _assert_meta_finite(full)
    _assert_meta_finite(agg)
    for group in ("z_per_sirna", "on_target_weights"):
        assert agg[group].keys() == full[group].keys() == {"miR", "s"}
        for sid, row in full[group].items():
            assert row.keys() == agg[group][sid].keys()
            for k, v in row.items():
                if k != "sirna_id":
                    assert agg[group][sid][k] == pytest.approx(v, rel=1e-12), (sid, k)
    # The strong hit is miR's on-target: it carries all of Z.
    assert full["on_target_weights"]["miR"]["W_on"] == pytest.approx(
        full["z_per_sirna"]["miR"]["Z_sirna"], rel=1e-12
    )


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sirna_id": ["miR"] * 3,
            "chrom": ["chr1"] * 3,
            "start": [100, 200, 300],
            "end": [121, 221, 321],
            "strand": ["+"] * 3,
            "energy": [-58.9, -20.0, -15.0],
        }
    )


def test_engine_scores_a_hit_beyond_the_float32_range() -> None:
    chunks = list(compute_off_targets(predictions=_predictions(), n_workers=1))

    assert len(chunks) == 1
    df, meta = chunks[0]
    df = df.sort("energy")
    assert df["P_off_target"].dtype == pl.Float32
    assert df["P_off_target"].to_list() == pytest.approx(
        _expected_p([-58.9, -20.0, -15.0], [1.0] * 3), rel=1e-6
    )
    assert df["P_off_target"][0] == 1.0
    _assert_finite(df)
    _assert_meta_finite(meta)


def test_cli_table_and_summary_have_finite_numbers(tmp_path: Path) -> None:
    rows = _predictions().iter_rows()
    tsv = tmp_path / "predictions.out"
    tsv.write_text(
        "".join(
            f"{sid}\t1\t21\t{chrom}\t{s}\t{e}\t{strand}\t{energy}\n"
            for sid, chrom, s, e, strand, energy in rows
        )
    )
    out = tmp_path / "out" / "off_targets.tsv"
    result = CliRunner().invoke(
        app,
        ["off-targets", "-r", str(tsv), "--theta", "1.2", "-j", "1",
         "-o", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output

    table = pl.read_csv(out, separator="\t").sort("energy")
    for c in ("P_off_target", "P_off_target:theta=1.2"):
        assert table[c].is_finite().all(), (c, table[c].to_list())
        assert table[c][0] == pytest.approx(1.0, rel=1e-6), c

    summary = (out.parent / "miR.summary").read_text()
    values = [
        v
        for line in summary.splitlines()
        if line.startswith("# For ")
        for _, v in FLOATS.findall(line.split(";", 1)[1])
    ]
    assert len(values) == 2 * 6, summary
    for v in values:
        assert v.lower() not in {"inf", "-inf", "nan"}, summary
        assert math.isfinite(float(v)), summary


def test_cli_detailed_parquet_keeps_double_weights(tmp_path: Path) -> None:
    rows = _predictions().iter_rows()
    tsv = tmp_path / "predictions.out"
    tsv.write_text(
        "".join(
            f"{sid}\t1\t21\t{chrom}\t{s}\t{e}\t{strand}\t{energy}\n"
            for sid, chrom, s, e, strand, energy in rows
        )
    )
    out = tmp_path / "out" / "off_targets.parquet"
    result = CliRunner().invoke(
        app,
        ["off-targets", "-r", str(tsv), "--theta", "1.2", "--detailed-report",
         "--output-format", "parquet", "-j", "1", "-o", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output

    table = pl.read_parquet(out)
    sums = [c for c in table.columns if c.startswith(SUMS)]
    probs = [c for c in table.columns if c.startswith("P_off_target")]
    assert len(sums) == 8 and len(probs) == 2
    for c in sums:
        assert table[c].dtype == pl.Float64, c
        assert table[c].is_finite().all(), (c, table[c].to_list())
    for c in probs:
        assert table[c].dtype == pl.Float32, c
        assert table[c].is_finite().all(), (c, table[c].to_list())
    strong = table.filter(pl.col("energy") == pl.col("energy").min())
    assert strong["P_off_target"].item() == pytest.approx(1.0, rel=1e-6)
