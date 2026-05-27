"""
tests/test_uncertainty.py
=========================
Unit tests for UncertaintyAgent and DecisionAgent.
"""
import pytest
import torch
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.uncertainty_agent import UncertaintyAgent
from agents.decision_agent import DecisionAgent


class TestUncertaintyAgent:

    @pytest.fixture
    def agent(self):
        return UncertaintyAgent(use_llm=False)

    def test_verbalized_confidence_range(self, agent):
        """Heuristic verbalized confidence must be in [0, 1]."""
        for base_prob in [0.01, 0.1, 0.5, 0.9, 0.99]:
            for epi in [0.0, 0.1, 0.5]:
                conf = agent.get_verbalized_confidence({}, base_prob, epi)
                assert 0.0 <= conf <= 1.0, \
                    f"Confidence out of [0,1]: {conf} for base_prob={base_prob}, epi={epi}"

    def test_verbalized_confidence_not_constant(self, agent):
        """Confidence must vary with base_prob — no longer hardcoded to 0.5."""
        low_conf  = agent.get_verbalized_confidence({}, 0.51, 0.5)   # near decision boundary
        high_conf = agent.get_verbalized_confidence({}, 0.99, 0.01)  # far from boundary, low epi
        assert high_conf > low_conf, \
            f"Expected high_conf > low_conf, got {high_conf:.3f} vs {low_conf:.3f}"

    def test_fuse_uncertainty_range(self, agent):
        """Composite uncertainty must be in [0, 1]."""
        rng = np.random.default_rng(0)
        N = 50
        base_probs    = rng.dirichlet([1, 1], size=N).astype(np.float32)
        base_probs3   = np.hstack([base_probs, 1 - base_probs.sum(axis=1, keepdims=True)])
        epistemic_unc = rng.uniform(0, 0.5, size=(N,)).astype(np.float32)

        composite = agent.fuse_uncertainty(base_probs3, epistemic_unc, features_list=None)
        assert composite.shape == (N,), f"Shape mismatch: {composite.shape}"
        assert (composite >= 0).all() and (composite <= 1).all(), \
            f"Composite unc out of [0,1]: min={composite.min():.4f}, max={composite.max():.4f}"

    def test_fuse_uncertainty_monotone_with_epistemic(self, agent):
        """Higher epistemic uncertainty → higher composite uncertainty (all else equal)."""
        base_prob = np.array([[0.6, 0.4, 0.0]], dtype=np.float32)
        low_epi  = np.array([0.01], dtype=np.float32)
        high_epi = np.array([0.49], dtype=np.float32)

        c_low  = agent.fuse_uncertainty(base_prob, low_epi,  None)[0]
        c_high = agent.fuse_uncertainty(base_prob, high_epi, None)[0]
        assert c_high >= c_low, \
            f"Higher epistemic should raise composite unc: {c_high:.4f} vs {c_low:.4f}"


class TestDecisionAgent:

    @pytest.fixture
    def agent(self):
        return DecisionAgent(feature_dim=10, uncertainty_dim=3, device="cpu")

    def test_action_is_binary(self, agent):
        """Action must be 0 (predict) or 1 (abstain)."""
        feat = torch.randn(1, 10)
        unc  = torch.rand(1, 3)
        for _ in range(20):
            action = agent.select_action(feat, unc)
            assert action in {0, 1}, f"Invalid action: {action}"

    def test_log_probs_accumulated(self, agent):
        """select_action must store log_prob in saved_log_probs."""
        feat = torch.randn(1, 10)
        unc  = torch.rand(1, 3)
        agent.select_action(feat, unc)
        assert len(agent.saved_log_probs) == 1

    def test_policy_update_clears_memory(self, agent):
        """After update_policy, saved_log_probs and rewards must be empty."""
        feat = torch.randn(1, 10)
        unc  = torch.rand(1, 3)
        agent.select_action(feat, unc)
        agent.store_reward(1.0)
        agent.update_policy()
        assert len(agent.saved_log_probs) == 0
        assert len(agent.rewards) == 0

    def test_update_policy_returns_float(self, agent):
        feat = torch.randn(1, 10)
        unc  = torch.rand(1, 3)
        agent.select_action(feat, unc)
        agent.store_reward(0.5)
        loss = agent.update_policy()
        assert isinstance(loss, float), f"Expected float loss, got {type(loss)}"

    def test_no_crash_without_rewards(self, agent):
        """Calling update_policy with empty buffers must not crash."""
        agent.update_policy()  # should return None silently
