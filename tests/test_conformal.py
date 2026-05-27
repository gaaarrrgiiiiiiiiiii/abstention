"""
tests/test_conformal.py
=======================
Unit tests for the ConformalPredictor.
Verifies coverage guarantee, calibration behaviour, and edge cases.
"""
import pytest
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from conformal import ConformalPredictor, apply_conformal_abstention


@pytest.fixture
def dummy_calibration_data():
    """Synthetic softmax probabilities for a 3-class model."""
    rng = np.random.default_rng(42)
    N = 1000
    # Skewed toward correct class (simulating a trained model)
    correct_probs = rng.beta(5, 1, size=N)           # concentrated near 1
    other_two     = rng.dirichlet([1, 1], size=N) * (1 - correct_probs[:, None])
    labels        = rng.integers(0, 3, size=N)

    smx = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        k = labels[i]
        smx[i, k] = correct_probs[i]
        others = [j for j in range(3) if j != k]
        smx[i, others[0]] = other_two[i, 0]
        smx[i, others[1]] = other_two[i, 1]

    return smx, labels


class TestConformalPredictor:

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    def test_marginal_coverage_guarantee(self, dummy_calibration_data, alpha):
        """
        The empirical coverage on a held-out test set must be >= 1 - alpha.
        (Finite-sample guarantee: 1 - alpha - 1/N ≤ coverage ≤ 1)
        """
        smx, labels = dummy_calibration_data
        n = len(labels)
        n_cal = n // 2

        cal_smx, test_smx = smx[:n_cal], smx[n_cal:]
        cal_lbl, test_lbl = labels[:n_cal], labels[n_cal:]

        cp = ConformalPredictor(alpha=alpha)
        cp.calibrate(cal_smx, cal_lbl)
        pred_sets = cp.predict_sets(test_smx)
        coverage, _ = cp.evaluate_coverage(pred_sets, test_lbl)

        # Finite-sample bound: coverage >= 1 - alpha (with 1/N slack)
        lower_bound = 1.0 - alpha - 1.0 / n_cal
        assert coverage >= lower_bound, (
            f"Coverage {coverage:.4f} < lower bound {lower_bound:.4f} for alpha={alpha}"
        )

    def test_q_hat_not_none_after_calibrate(self, dummy_calibration_data):
        smx, labels = dummy_calibration_data
        cp = ConformalPredictor(alpha=0.1)
        assert cp.q_hat is None
        cp.calibrate(smx, labels)
        assert cp.q_hat is not None

    def test_raises_if_not_calibrated(self):
        cp = ConformalPredictor(alpha=0.1)
        with pytest.raises(ValueError, match="not calibrated"):
            cp.predict_sets(np.ones((10, 3)))

    def test_prediction_sets_include_high_prob_class(self):
        """A class with probability 0.99 must always be in the prediction set."""
        smx = np.array([[0.99, 0.005, 0.005]], dtype=np.float32)
        labels = np.array([0])

        cp = ConformalPredictor(alpha=0.1)
        # Calibrate on same point (degenerate but valid for this check)
        cp.calibrate(smx, labels)
        pred_sets = cp.predict_sets(smx)
        assert 0 in pred_sets[0], "Class 0 with prob 0.99 must be in prediction set"

    def test_apply_conformal_abstention_wrapper(self, dummy_calibration_data):
        smx, labels = dummy_calibration_data
        n_cal = len(labels) // 2
        pred_sets, cp = apply_conformal_abstention(
            smx[:n_cal], labels[:n_cal], smx[n_cal:], alpha=0.1
        )
        assert len(pred_sets) == len(labels) - n_cal
        assert cp.q_hat is not None

    def test_coverage_near_one_with_alpha_zero(self, dummy_calibration_data):
        """Very small alpha should yield near-perfect coverage."""
        smx, labels = dummy_calibration_data
        n = len(labels) // 2
        cp = ConformalPredictor(alpha=0.001)
        cp.calibrate(smx[:n], labels[:n])
        pred_sets = cp.predict_sets(smx[n:])
        cov, _ = cp.evaluate_coverage(pred_sets, labels[n:])
        assert cov >= 0.99, f"Expected near-perfect coverage, got {cov:.4f}"
