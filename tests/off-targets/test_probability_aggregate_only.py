"""``aggregate_only`` computes the per-siRNA sums without the per-row columns.

``--summary-only`` writes the .summary files from the per-siRNA partition
functions (``z_per_sirna``) and on-target weights alone, yet the full path
added an energy, a dG, two Boltzmann-weight and a probability column per
alpha/gamma/theta set to every site x transcript row (385 columns for the
benchmark grid) and shipped the frame back to the parent: ~3 GB per heavy
siRNA. With ``aggregate_only`` the sums must be exactly what the full path
reports, and no per-row frame is returned.
"""

import numpy as np
import polars as pl
import pytest

from sioff.services.probability import ProbabilityService

AG = [(1.0, 1.0), (0.8, 0.9), (0.5, 0.55)]
THETA = [0.7, 0.95]


def _frame(seed: int = 3, n: int = 400) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    sirna = rng.choice(["s1", "s2", "s3"], n)
    energy = rng.uniform(-30.0, -10.0, n).astype(np.float32)
    return pl.DataFrame(
        {
            "sirna_id": sirna,
            "chrom": rng.choice(["chr1", "chr2"], n),
            "start": rng.integers(1, 10_000, n),
            "end": rng.integers(10_000, 20_000, n),
            "strand": rng.choice(["+", "-"], n),
            "energy": energy,
            "raw_e_min": np.float32(-31.0),
            "opening_energy": rng.uniform(0.0, 12.0, n).astype(np.float32),
            "exp_value": rng.uniform(0.0, 50.0, n).astype(np.float32),
            "gene_id": rng.choice(["g1", "g2", "g3", "g4"], n),
            "transcript_id": rng.choice(["t1", "t2", "t3", "t4", "t5"], n),
        }
    )


@pytest.mark.parametrize("on_target_map", [None, {"s1": "g1", "s2": "t3"}])
def test_aggregate_only_reports_the_same_sums(on_target_map):
    svc = ProbabilityService(None)
    kw = dict(
        alpha_gamma_pairs=AG,
        theta_values=THETA,
        on_target_map=on_target_map,
        precomputed_opening_energy=True,
    )
    full_df, full = svc.calculate_probabilities_per_sirna(_frame(), **kw)
    agg_df, agg = svc.calculate_probabilities_per_sirna(
        _frame(), aggregate_only=True, **kw
    )

    assert full_df.height == 400
    assert agg_df.height == 0
    assert agg["n_sirnas"] == full["n_sirnas"] == 3
    assert agg["n_rows"] == 400
    assert set(agg["z_per_sirna"]) == set(full["z_per_sirna"])
    for sid, row in full["z_per_sirna"].items():
        for k, v in row.items():
            if k != "sirna_id":
                assert agg["z_per_sirna"][sid][k] == pytest.approx(v, rel=1e-6), (
                    sid,
                    k,
                )
    assert agg["on_target_weights"].keys() == full["on_target_weights"].keys()
    for sid, w in full["on_target_weights"].items():
        for k, v in w.items():
            assert agg["on_target_weights"][sid][k] == pytest.approx(v, rel=1e-6), (
                sid,
                k,
            )
