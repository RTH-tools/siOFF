"""
Golden-file regression test for partition function (Z) values.

Runs ProbabilityService.calculate_probabilities_per_sirna() on the pre-validated
intersection results for 4 siRNAs and checks that Zoff_noacc matches the reference
derived from a freshly-run old Python-2.7 pipeline to within 0.2%.

Any regression in:
  - alpha/gamma clamping logic
  - theta scaling
  - raw_e_min handling
  - on-target detection / Zoff = Z - W_on computation
  - expression weighting formula

will cause this test to fail.
"""

import json
from pathlib import Path

import polars as pl
import pytest

from sioff.services.probability import ProbabilityService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "data"

# Parameters used when the reference was generated
ALPHA_GAMMA = [
    (0.5, 0.55),
    (0.5, 0.85),
    (0.5, 0.9),
    (0.5, 0.95),
    (0.5, 1.0),
    (0.85, 0.85),
    (0.85, 0.9),
    (0.85, 0.95),
    (0.85, 1.0),
    (0.9, 0.9),
    (0.9, 0.95),
    (0.9, 1.0),
    (0.95, 0.95),
    (0.95, 1.0),
]
THETA = [0.5, 0.9]

# ENSG00000177889 siRNA targets its own gene → its on-target hits live in the DF
ON_TARGET_MAP = {
    "ENSG00000177889-78-18537": "ENSG00000177889",
}

# Tolerance: 0.2% (2× the observed Python 2.7 vs Python 3 float rounding gap)
TOLERANCE = 0.002


@pytest.fixture(scope="module")
def regression_input() -> pl.DataFrame:
    path = FIXTURE_DIR / "regression_input.parquet"
    if not path.exists():
        pytest.skip(
            f"Fixture not found: {path}. Run create_regression_fixtures.py first."
        )
    return pl.read_parquet(path)


@pytest.fixture(scope="module")
def regression_reference() -> dict:
    path = FIXTURE_DIR / "regression_reference.json"
    if not path.exists():
        pytest.skip(
            f"Reference not found: {path}. Run create_regression_fixtures.py first."
        )
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def computed_meta(regression_input) -> dict:
    svc = ProbabilityService(None)
    _, meta = svc.calculate_probabilities_per_sirna(
        regression_input,
        alpha_gamma_pairs=ALPHA_GAMMA,
        theta_values=THETA,
        on_target_map=ON_TARGET_MAP,
        e_min=dict(
            regression_input.group_by("sirna_id")
            .agg(pl.col("raw_e_min").min())
            .iter_rows()
        ),
    )
    return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_zoff_noacc(meta: dict, sid: str, suffix: str = "") -> float:
    """Zoff_noacc as the .summary reports it: the off-target rows' own sum."""
    z_total = meta["z_per_sirna"].get(sid, {}).get(f"Z_sirna_noacc{suffix}", 0.0)
    w = meta["on_target_weights"].get(sid, {})
    return w.get(f"W_off_noacc{suffix}", z_total)


def param_key_to_suffix(param_key: str) -> str:
    if "alpha=" in param_key:
        parts = param_key.split(",")
        a = parts[0].replace("alpha=", "")
        g = parts[1].replace("gamma=", "")
        # alpha=1.0,gamma=1.0 is the base case — stored under the empty suffix
        if a == "1.0" and g == "1.0":
            return ""
        return f":alpha={a},gamma={g}"
    else:
        theta = param_key.replace("theta=", "")
        return f":theta={theta}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fixture_has_expected_sirnas(regression_input):
    ids = set(regression_input["sirna_id"].unique().to_list())
    expected = {
        "ENSG00000109991-51-19902",
        "ENSG00000137726-36-18816",
        "ENSG00000167941-35-19603",
        "ENSG00000177889-78-18537",
    }
    assert ids == expected


def test_fixture_row_counts(regression_input):
    counts = (
        regression_input.group_by("sirna_id").agg(pl.len().alias("n")).sort("sirna_id")
    )
    for row in counts.iter_rows(named=True):
        assert row["n"] > 10_000, f"{row['sirna_id']} has only {row['n']} rows"


@pytest.mark.parametrize(
    "sid",
    [
        "ENSG00000109991-51-19902",
        "ENSG00000137726-36-18816",
        "ENSG00000167941-35-19603",
        "ENSG00000177889-78-18537",
    ],
)
def test_zoff_noacc_base_case(sid, computed_meta, regression_reference):
    """Base case (alpha=1.0, gamma=1.0): Zoff_noacc must match reference within 0.2%."""
    ref_z = regression_reference[sid].get("alpha=1.0,gamma=1.0")
    if ref_z is None:
        pytest.skip(f"No base-case reference for {sid}")

    got = get_zoff_noacc(computed_meta, sid, "")
    ratio = got / ref_z
    assert abs(ratio - 1.0) < TOLERANCE, (
        f"{sid} base: got {got:.4e}, ref {ref_z:.4e}, ratio {ratio:.6f}"
    )


@pytest.mark.parametrize(
    "sid",
    [
        "ENSG00000109991-51-19902",
        "ENSG00000137726-36-18816",
        "ENSG00000167941-35-19603",
        "ENSG00000177889-78-18537",
    ],
)
def test_zoff_noacc_all_parameters(sid, computed_meta, regression_reference):
    """All alpha/gamma/theta combinations must match within 0.2%."""
    ref_params = regression_reference.get(sid, {})
    failures = []

    for param_key, ref_z in ref_params.items():
        if ref_z == 0.0:
            continue
        suffix = param_key_to_suffix(param_key)
        got = get_zoff_noacc(computed_meta, sid, suffix)
        ratio = got / ref_z
        if abs(ratio - 1.0) >= TOLERANCE:
            failures.append(
                f"  {param_key}: got {got:.4e}, ref {ref_z:.4e}, ratio {ratio:.6f}"
            )

    assert not failures, (
        f"{sid}: {len(failures)} parameter(s) out of tolerance:\n" + "\n".join(failures)
    )


def test_on_target_zoff_smaller_than_ztotal(computed_meta):
    """For ENSG00000177889 (has on-target), Zoff_noacc < Z_noacc_total."""
    sid = "ENSG00000177889-78-18537"
    z_row = computed_meta["z_per_sirna"].get(sid, {})
    z_total = z_row.get("Z_sirna_noacc", 0.0)
    zoff = get_zoff_noacc(computed_meta, sid, "")

    assert z_total > 0
    assert zoff < z_total, (
        "Zoff_noacc should be smaller than Z_total when on-target rows present"
    )
    # On-target weight must be substantial (gene is expressed and strongly bound)
    w_on = computed_meta["on_target_weights"].get(sid, {}).get("W_on_noacc", 0.0)
    assert w_on > 0, "Expected non-zero on-target weight for ENSG00000177889"


def test_no_on_target_for_other_sirnas(computed_meta):
    """siRNAs without on-target genes must have Zoff == Z_total."""
    for sid in [
        "ENSG00000109991-51-19902",
        "ENSG00000137726-36-18816",
        "ENSG00000167941-35-19603",
    ]:
        z_row = computed_meta["z_per_sirna"].get(sid, {})
        z_total = z_row.get("Z_sirna_noacc", 0.0)
        zoff = get_zoff_noacc(computed_meta, sid, "")
        assert abs(zoff / z_total - 1.0) < 1e-9, (
            f"{sid}: expected Zoff == Z_total (no on-target), but got ratio {zoff / z_total:.6f}"
        )
