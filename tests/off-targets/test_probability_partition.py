import unittest
import polars as pl
import numpy as np
from sioff.services.probability import ProbabilityService, RT


class TestProbabilityPartition(unittest.TestCase):
    def setUp(self):
        self.service = ProbabilityService(None)

    def test_z_score_off_targets_only(self):
        """Test P(OT) calculation with only off-targets (Z = sum of off-targets)."""
        # Data: 2 off-targets
        # 1. dG = -10.0, Expr = 100
        # 2. dG = -5.0, Expr = 100
        df = pl.DataFrame(
            {
                "sirna_id": ["s1", "s1"],
                "energy": [-10.0, -5.0],
                "exp_value": [100.0, 100.0],
                "chrom": ["chr1", "chr1"],
                "start": [100, 200],
                "end": [120, 220],
            }
        )

        # Expected Weights
        # W1 = 100 * exp(-(-10)/RT) = 100 * exp(10/RT)
        # W2 = 100 * exp(-(-5)/RT) = 100 * exp(5/RT)
        w1 = 100.0 * np.exp(10.0 / RT)
        w2 = 100.0 * np.exp(5.0 / RT)
        z = w1 + w2

        expected_p1 = w1 / z
        expected_p2 = w2 / z

        res, _ = self.service.calculate_probabilities_per_sirna(df)

        p1 = res["P_off_target"][0]
        p2 = res["P_off_target"][1]

        # Assert sum is 1.0 (since Z covers all sites in this closed universe)
        self.assertAlmostEqual(p1 + p2, 1.0, places=4)
        self.assertAlmostEqual(p1, expected_p1, places=4)
        self.assertAlmostEqual(p2, expected_p2, places=4)


if __name__ == "__main__":
    unittest.main()
