"""Regression tests for strand- and circular-origin-invariant evaluation."""

from __future__ import annotations

import unittest

from skbio.sequence import DNA

from common import evaluate_theta_vs_target


class CircularEvaluatorTests(unittest.TestCase):
    TARGET = "AACCGTACGTTAGGCTATGC"

    def normalized(self, theta: str) -> dict:
        return evaluate_theta_vs_target(
            theta,
            self.TARGET,
            normalize_strand=True,
            normalize_circular_origin=True,
        )

    def test_rotated_complete_circle_scores_as_identical(self) -> None:
        theta = self.TARGET[7:] + self.TARGET[:7]
        self.assertEqual(self.normalized(theta)["f1"], 1.0)
        self.assertEqual(self.normalized(theta)["identity_pct"], 100.0)

    def test_reverse_complement_scores_as_identical(self) -> None:
        theta = str(DNA(self.TARGET).reverse_complement())
        self.assertEqual(self.normalized(theta)["f1"], 1.0)
        self.assertEqual(self.normalized(theta)["identity_pct"], 100.0)

    def test_reverse_complemented_rotation_scores_as_identical(self) -> None:
        rotated = self.TARGET[11:] + self.TARGET[:11]
        theta = str(DNA(rotated).reverse_complement())
        self.assertEqual(self.normalized(theta)["f1"], 1.0)
        self.assertEqual(self.normalized(theta)["identity_pct"], 100.0)

    def test_historical_default_remains_fixed_origin_and_strand(self) -> None:
        theta = str(DNA(self.TARGET).reverse_complement())
        self.assertLess(evaluate_theta_vs_target(theta, self.TARGET)["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
