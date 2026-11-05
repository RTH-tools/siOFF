"""
Unit tests for ProbabilityService.calculate_probabilities_per_sirna().

This is the production code path used by the off-targets command.
The existing test_probability*.py files only cover calculate_probabilities()
(single partition function, no parameter sweeps).

All expected values are computed analytically so the tests do not depend on
external data files.
"""

import math

import polars as pl
import pytest

from sioff.services.probability import ProbabilityService, RT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_df(**cols):
    """Build a minimal DataFrame accepted by calculate_probabilities_per_sirna."""
    base = {
        "sirna_id": ["si1"] * len(next(iter(cols.values()))),
        "gene_id": ["g_off"] * len(next(iter(cols.values()))),
        "transcript_id": ["t_off"] * len(next(iter(cols.values()))),
        "opening_energy": [0.0] * len(next(iter(cols.values()))),
    }
    base.update(cols)
    return pl.DataFrame(base)


def zoff_noacc_from_meta(meta, sid, suffix=""):
    """Extract Zoff_noacc for a siRNA from the metadata dict."""
    z_row = meta["z_per_sirna"].get(sid, {})
    w_on = meta["on_target_weights"].get(sid, {})
    z_total = z_row.get(f"Z_sirna_noacc{suffix}", 0.0)
    w_on_val = w_on.get(f"W_on_noacc{suffix}", 0.0)
    return z_total - w_on_val


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseCase:
    """alpha=1, gamma=1 — no clamping; Z_noacc = sum(exp_value * exp(-energy / RT))."""

    def test_z_noacc_two_rows(self):
        energy = [-20.0, -15.0]
        exp_val = [10.0, 5.0]

        df = make_df(energy=energy, exp_value=exp_val)
        svc = ProbabilityService(None)
        # Pass a dummy alpha/gamma to bypass the single-siRNA early-exit path
        # (which returns a different metadata structure).
        _, meta = svc.calculate_probabilities_per_sirna(
            df, alpha_gamma_pairs=[(0.85, 0.9)]
        )

        expected = 10.0 * math.exp(20.0 / RT) + 5.0 * math.exp(15.0 / RT)
        got = zoff_noacc_from_meta(meta, "si1")
        assert abs(got / expected - 1.0) < 1e-5, (
            f"got {got:.4e}, expected {expected:.4e}"
        )

    def test_multiple_sirnas_independent_z(self):
        """Each siRNA must have its own partition function."""
        df = pl.DataFrame(
            {
                "sirna_id": ["siA", "siA", "siB", "siB"],
                "energy": [-20.0, -15.0, -10.0, -5.0],
                "exp_value": [1.0, 1.0, 1.0, 1.0],
                "opening_energy": [0.0, 0.0, 0.0, 0.0],
                "gene_id": ["g1", "g1", "g2", "g2"],
                "transcript_id": ["t1", "t1", "t2", "t2"],
            }
        )
        svc = ProbabilityService(None)
        _, meta = svc.calculate_probabilities_per_sirna(df)

        za = zoff_noacc_from_meta(meta, "siA")
        zb = zoff_noacc_from_meta(meta, "siB")

        expected_a = math.exp(20 / RT) + math.exp(15 / RT)
        expected_b = math.exp(10 / RT) + math.exp(5 / RT)
        assert abs(za / expected_a - 1.0) < 1e-5
        assert abs(zb / expected_b - 1.0) < 1e-5
        # They must differ (siB is much smaller)
        assert zb < za


class TestAlphaGammaClamping:
    """Energy clamping: if energy < alpha * E_min → use gamma * E_min."""

    # E_min = -39.41 kcal/mol (self-hyb value from our validation runs)
    E_MIN = -39.41

    def _run(self, energies, exp_vals, alpha, gamma, emin=None):
        if emin is None:
            emin = self.E_MIN
        df = make_df(energy=energies, exp_value=exp_vals)
        svc = ProbabilityService(None)
        _, meta = svc.calculate_probabilities_per_sirna(
            df, alpha_gamma_pairs=[(alpha, gamma)], e_min={"si1": emin}
        )
        suffix = f":alpha={alpha},gamma={gamma}"
        return zoff_noacc_from_meta(meta, "si1", suffix)

    def test_no_clamping_when_energy_above_threshold(self):
        """energy=-20 > 0.85 * (-39.41) = -33.50 → NOT clamped."""
        got = self._run([-20.0], [1.0], alpha=0.85, gamma=0.9)
        expected = math.exp(20.0 / RT)
        assert abs(got / expected - 1.0) < 1e-5

    def test_clamping_when_energy_below_threshold(self):
        """energy=-35 < 0.85 * (-39.41) = -33.50 → clamped to 0.9 * (-39.41)."""
        alpha, gamma = 0.85, 0.9
        clamped_e = gamma * self.E_MIN  # = -35.469
        got = self._run([-35.0], [1.0], alpha=alpha, gamma=gamma)
        expected = math.exp(-clamped_e / RT)
        assert abs(got / expected - 1.0) < 1e-5

    def test_partial_clamping_mixed_rows(self):
        """One row clamped, one not."""
        alpha, gamma = 0.85, 0.9
        # Row 1: energy=-35 < threshold=-33.50 → clamped to gamma*E_min=-35.469
        # Row 2: energy=-20 > threshold=-33.50 → unchanged
        clamped_e = gamma * self.E_MIN
        got = self._run([-35.0, -20.0], [1.0, 1.0], alpha=alpha, gamma=gamma)
        expected = math.exp(-clamped_e / RT) + math.exp(20.0 / RT)
        assert abs(got / expected - 1.0) < 1e-5

    def test_boundary_energy_equals_threshold_not_clamped(self):
        """energy exactly at threshold (energy == alpha * E_min) is NOT clamped."""
        alpha, gamma = 0.9, 0.95
        threshold = alpha * self.E_MIN  # -35.469
        got = self._run([threshold], [1.0], alpha=alpha, gamma=gamma)
        # energy is NOT < threshold (it equals it), so no clamping
        expected = math.exp(-threshold / RT)
        assert abs(got / expected - 1.0) < 1e-5

    def test_raw_emin_anchors_clamping(self):
        """A passed-in E_min overrides energy.min() as the clamping anchor."""
        # If E_min = -35 (less negative than actual energy = -39)
        # then E_min used for clamping should be -35, not -39
        # At alpha=0.9: threshold with raw=-35 → 0.9*(-35) = -31.5
        #               threshold with raw=-39 → 0.9*(-39) = -35.1
        alpha, gamma = 0.9, 1.0
        raw_emin = -35.0
        energy = -32.0  # -32 < -31.5 (threshold with raw=-35) → clamped
        # -32 > -35.1 (threshold with raw=-39) → NOT clamped

        # With e_min given: should clamp
        df_with = make_df(energy=[energy], exp_value=[1.0])
        svc = ProbabilityService(None)
        _, meta = svc.calculate_probabilities_per_sirna(
            df_with, alpha_gamma_pairs=[(alpha, gamma)], e_min={"si1": raw_emin}
        )
        suffix = f":alpha={alpha},gamma={gamma}"
        z_with = zoff_noacc_from_meta(meta, "si1", suffix)
        clamped_e = gamma * raw_emin  # = -35.0
        expected_clamped = math.exp(-clamped_e / RT)
        assert abs(z_with / expected_clamped - 1.0) < 1e-5

        # Without e_min: E_min = energy.min() = -32, threshold = 0.9*(-32) = -28.8
        # -32 < -28.8 → would be clamped to gamma * (-32) = -32 (no change since gamma=1)
        df_without = pl.DataFrame(
            {
                "sirna_id": ["si1"],
                "energy": [energy],
                "exp_value": [1.0],
                "opening_energy": [0.0],
                "gene_id": ["g1"],
                "transcript_id": ["t1"],
            }
        )
        _, meta2 = svc.calculate_probabilities_per_sirna(
            df_without, alpha_gamma_pairs=[(alpha, gamma)]
        )
        z_without = zoff_noacc_from_meta(meta2, "si1", suffix)
        expected_unclamped = math.exp(-energy / RT)
        assert abs(z_without / expected_unclamped - 1.0) < 1e-5

        # The two results must differ
        assert abs(z_with - z_without) / max(z_with, z_without) > 0.01


class TestThetaScaling:
    """Theta scales energy linearly: assigned = theta*(energy+10) - 10."""

    def test_theta_scaling_z(self):
        theta = 0.9
        energy = -20.0
        assigned = theta * (energy + 10.0) - 10.0  # = 0.9*(-10) - 10 = -19.0

        df = make_df(energy=[energy], exp_value=[1.0])
        svc = ProbabilityService(None)
        _, meta = svc.calculate_probabilities_per_sirna(df, theta_values=[theta])

        suffix = f":theta={theta}"
        got = zoff_noacc_from_meta(meta, "si1", suffix)
        expected = math.exp(-assigned / RT)
        assert abs(got / expected - 1.0) < 1e-5

    def test_theta_05_scaling(self):
        theta = 0.5
        energy = -20.0
        assigned = theta * (energy + 10.0) - 10.0  # = 0.5*(-10) - 10 = -15.0

        df = make_df(energy=[energy], exp_value=[1.0])
        svc = ProbabilityService(None)
        _, meta = svc.calculate_probabilities_per_sirna(df, theta_values=[theta])

        suffix = f":theta={theta}"
        got = zoff_noacc_from_meta(meta, "si1", suffix)
        expected = math.exp(-assigned / RT)
        assert abs(got / expected - 1.0) < 1e-5

    def test_theta_1_equals_base_case(self):
        """theta=1.0 should produce same Z_noacc as the base case."""
        energy = -20.0
        df = make_df(energy=[energy], exp_value=[1.0])
        svc = ProbabilityService(None)
        _, meta = svc.calculate_probabilities_per_sirna(df, theta_values=[1.0])

        # base case: no suffix
        z_base = zoff_noacc_from_meta(meta, "si1", "")
        # theta=1 suffix
        z_theta1 = zoff_noacc_from_meta(meta, "si1", ":theta=1.0")
        assert abs(z_theta1 / z_base - 1.0) < 1e-5


class TestOnTargetInPredictionsMode:
    """On-target rows stay in Z but contribute to W_on; Zoff = Z - W_on."""

    def _build_df(self):
        return pl.DataFrame(
            {
                "sirna_id": ["si1", "si1"],
                "energy": [-25.0, -30.0],
                "exp_value": [100.0, 10.0],
                "opening_energy": [0.0, 0.0],
                "gene_id": ["g_off", "g_on"],
                "transcript_id": ["t_off", "t_on"],
            }
        )

    def test_zoff_excludes_on_target_weight(self):
        df = self._build_df()
        svc = ProbabilityService(None)
        _, meta = svc.calculate_probabilities_per_sirna(
            df, on_target_map={"si1": "g_on"}
        )

        w_off = 100.0 * math.exp(25.0 / RT)
        _w_on = 10.0 * math.exp(30.0 / RT)  # on-target weight — excluded from Zoff
        expected_zoff = w_off  # on-target excluded

        zoff = zoff_noacc_from_meta(meta, "si1")
        assert abs(zoff / expected_zoff - 1.0) < 1e-5

    def test_ztotal_includes_on_target(self):
        df = self._build_df()
        svc = ProbabilityService(None)
        _, meta = svc.calculate_probabilities_per_sirna(
            df, on_target_map={"si1": "g_on"}
        )
        z_row = meta["z_per_sirna"]["si1"]
        z_total = z_row.get("Z_sirna_noacc", 0.0)

        w_off = 100.0 * math.exp(25.0 / RT)
        w_on = 10.0 * math.exp(30.0 / RT)
        expected_total = w_off + w_on

        assert abs(z_total / expected_total - 1.0) < 1e-5

    def test_no_on_target_map_all_rows_in_zoff(self):
        df = self._build_df()
        svc = ProbabilityService(None)
        # Pass a dummy alpha/gamma to bypass the single-siRNA early-exit path.
        _, meta = svc.calculate_probabilities_per_sirna(
            df, alpha_gamma_pairs=[(0.85, 0.9)]
        )

        z_total = meta["z_per_sirna"]["si1"].get("Z_sirna_noacc", 0.0)
        zoff = zoff_noacc_from_meta(meta, "si1")
        # Without on_target_map, no rows are tagged → Zoff == Z_total
        assert abs(zoff / z_total - 1.0) < 1e-10


class TestEMinMapping:
    """E_min passed as a mapping anchors the clamp; metadata counts on-target rows."""

    @staticmethod
    def _clamped(e_min):
        df = pl.DataFrame(
            {
                "sirna_id": ["s1", "s1"],
                "energy": [-20.0, -15.0],
                "exp_value": [1.0, 1.0],
                "opening_energy": [0.0, 0.0],
            }
        )
        res, _ = ProbabilityService(None).calculate_probabilities_per_sirna(
            df, alpha_gamma_pairs=[(0.8, 0.8)], e_min=e_min
        )
        return res["energy:alpha=0.8,gamma=0.8"].to_list()

    def test_e_min_mapping_anchors_clamping(self):
        assert self._clamped({"s1": -20.0}) == pytest.approx([-16.0, -15.0])
        assert self._clamped({"s1": -30.0}) == pytest.approx([-20.0, -15.0])
        assert self._clamped(None) == pytest.approx([-16.0, -15.0])

    def test_missing_e_min_entry_is_named(self):
        df = pl.DataFrame(
            {"sirna_id": ["s1", "s2"], "energy": [-20.0, -15.0], "opening_energy": 0.0}
        )
        with pytest.raises(ValueError, match="s2"):
            ProbabilityService(None).calculate_probabilities_per_sirna(
                df, e_min={"s1": -20.0}
            )

    def test_metadata_counts_on_target_rows(self):
        df = pl.DataFrame(
            {
                "sirna_id": ["s1", "s1", "s2"],
                "transcript_id": ["t_on", "t_off", "t_off"],
                "energy": [-20.0, -15.0, -10.0],
                "opening_energy": 0.0,
            }
        )
        _, meta = ProbabilityService(None).calculate_probabilities_per_sirna(
            df, on_target_map={"s1": "t_on", "s2": "t_on"}
        )
        assert meta["n_on_target_rows"] == {"s1": 1, "s2": 0}
