import unittest
from unittest.mock import MagicMock
import polars as pl
import numpy as np
from sioff.services.probability import ProbabilityService, RT, R


class TestProbabilityService(unittest.TestCase):
    def test_calculate_simple(self):
        """Test calculation without accessibility."""
        service = ProbabilityService(None)

        df = pl.DataFrame({"sirna_id": ["s1"] * 3, "energy": [-10.0, -5.0, 0.0]})

        res, _ = service.calculate_probabilities_per_sirna(df)

        self.assertIn("dG_total", res.columns)
        self.assertIn("P_off_target", res.columns)

        pass

    def test_formula_correction(self):
        """Verify the formula correction."""
        service = ProbabilityService(None)
        df = pl.DataFrame({"sirna_id": ["s1"], "energy": [-10.0]})

        res, _ = service.calculate_probabilities_per_sirna(df)
        p = res["P_off_target"][0]

        self.assertGreater(p, 0.9)

    def test_with_accessibility(self):
        """Test with accessibility service mocked."""
        acc_service = MagicMock()
        # annotate_opening_energy_vectorized adds opening_energy column (0.5 for all rows)
        acc_service.annotate_opening_energy_vectorized.side_effect = lambda df: (
            df.with_columns(pl.lit(0.5).cast(pl.Float32).alias("opening_energy"))
        )

        service = ProbabilityService(acc_service)

        df = pl.DataFrame(
            {
                "sirna_id": ["s1"],
                "energy": [-10.0],
                "chrom": ["chr1"],
                "start": [10],
                "end": [12],
                "strand": ["+"],
            }
        )

        res, _ = service.calculate_probabilities_per_sirna(df)

        self.assertIn("opening_energy", res.columns)
        dG_open = res["opening_energy"][0]

        self.assertAlmostEqual(dG_open, 0.5, places=2)

    def test_temperature_default_matches_37(self):
        """ProbabilityService() default must be identical to temperature=37.0."""
        df = pl.DataFrame({"sirna_id": ["s1"] * 3, "energy": [-10.0, -5.0, -2.0]})

        svc_default = ProbabilityService(None)
        svc_37 = ProbabilityService(None, temperature=37.0)

        res_default, _ = svc_default.calculate_probabilities_per_sirna(df)
        res_37, _ = svc_37.calculate_probabilities_per_sirna(df)

        np.testing.assert_array_almost_equal(
            res_default["P_off_target"].to_numpy(),
            res_37["P_off_target"].to_numpy(),
            decimal=10,
            err_msg="Default temperature must match explicit temperature=37.0",
        )

    def test_temperature_alters_boltzmann_weights(self):
        """Higher temperature produces different (closer-to-uniform) probabilities."""
        df = pl.DataFrame({"sirna_id": ["s1"] * 3, "energy": [-10.0, -5.0, -2.0]})

        svc_37 = ProbabilityService(None, temperature=37.0)
        svc_80 = ProbabilityService(None, temperature=80.0)

        res_37, _ = svc_37.calculate_probabilities_per_sirna(df)
        res_80, _ = svc_80.calculate_probabilities_per_sirna(df)

        p_37 = res_37["P_off_target"].to_numpy()
        p_80 = res_80["P_off_target"].to_numpy()

        self.assertFalse(
            np.allclose(p_37, p_80),
            "Probabilities must differ between temperature=37 and temperature=80",
        )
        # At higher T, distribution is less peaked: leading probability drops
        self.assertGreater(
            p_37[0], p_80[0], "Highest-weight entry less dominant at higher T"
        )

    def test_temperature_alters_per_sirna_probabilities(self):
        """calculate_probabilities_per_sirna respects temperature."""
        df = pl.DataFrame(
            {
                "sirna_id": ["s1", "s1", "s2", "s2"],
                "energy": [-10.0, -5.0, -8.0, -3.0],
            }
        )

        svc_37 = ProbabilityService(None, temperature=37.0)
        svc_80 = ProbabilityService(None, temperature=80.0)

        res_37, _ = svc_37.calculate_probabilities_per_sirna(df)
        res_80, _ = svc_80.calculate_probabilities_per_sirna(df)

        p_37 = res_37["P_off_target"].to_numpy()
        p_80 = res_80["P_off_target"].to_numpy()

        self.assertFalse(
            np.allclose(p_37, p_80),
            "Per-siRNA probabilities must differ between temperature=37 and temperature=80",
        )

    def test_frame_without_sirna_id_is_rejected(self):
        service = ProbabilityService(None)
        df = pl.DataFrame({"energy": [-10.0, -5.0]})

        with self.assertRaisesRegex(ValueError, "sirna_id"):
            service.calculate_probabilities_per_sirna(df)

    def test_module_level_RT_still_exported(self):
        """Module-level RT constant is still importable for backward compat."""
        self.assertAlmostEqual(RT, R * 310.15, places=8)


if __name__ == "__main__":
    unittest.main()
