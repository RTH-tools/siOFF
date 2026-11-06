"""The per-siRNA model's weights and probabilities follow the Boltzmann formula.

W_i = exp_value_i * exp(-(energy_i + opening_energy_i) / RT) and
P_i = W_i / sum(W): checked against a direct computation with no expression,
with a missing expression (weight 0) and with accessibility looked up through
the service. The file used to compare the deleted single-siRNA model with
the per-siRNA one; the formula is what is left to pin.
"""

from unittest.mock import MagicMock

import polars as pl
import pytest

from sioff.services.probability import RT, ProbabilityService


def _plain():
    return ProbabilityService(None), pl.DataFrame(
        {
            "sirna_id": ["s1"] * 3,
            "chrom": ["chr1"] * 3,
            "start": [100, 200, 300],
            "end": [120, 220, 320],
            "energy": [-10.0, -5.0, -7.5],
        }
    )


def _with_expression():
    svc, df = _plain()
    return svc, df.with_columns(pl.Series("exp_value", [100.0, None, 20.0]))


def _with_accessibility():
    acc = MagicMock()
    acc.annotate_opening_energy_vectorized.side_effect = lambda df: df.with_columns(
        pl.Series("opening_energy", [0.5, 1.5, 0.0]).cast(pl.Float32)
    )
    _, df = _plain()
    return ProbabilityService(acc), df.with_columns(
        pl.lit("+").alias("strand"), pl.Series("exp_value", [3.0, 4.0, 5.0])
    )


@pytest.mark.parametrize(
    "build", [_plain, _with_expression, _with_accessibility], ids=lambda f: f.__name__
)
def test_per_sirna_model_matches_the_boltzmann_formula(build):
    svc, df = build()
    per, _ = svc.calculate_probabilities_per_sirna(df)
    per = per.sort(["chrom", "start", "end"])

    open_e = per["opening_energy"] if "opening_energy" in per.columns else 0.0
    dg = per["energy"] + open_e
    expr = per["exp_value"].fill_null(0.0) if "exp_value" in per.columns else 1.0
    w = expr * (-dg / RT).exp()
    p = w / w.sum()

    assert per["dG_total"].to_list() == pytest.approx(dg.to_list(), rel=1e-6)
    assert per["P_off_target"].null_count() == 0
    assert per["boltzmann_weight"].null_count() == 0
    assert per["P_off_target"].to_list() == pytest.approx(p.to_list(), rel=1e-6)
    assert per["boltzmann_weight"].to_list() == pytest.approx(w.to_list(), rel=1e-6)
