"""
tests/test_model.py
===================
Unit tests for the AbstentionModel and BaselineModel architectures.
Tests output shapes, softmax validity, and abstain class presence.
"""
import pytest
import torch
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from abstention_model import AbstentionModel
from baseline_model import BaselineModel


@pytest.fixture(params=[30, 64, 432])
def input_dim(request):
    return request.param


class TestAbstentionModel:

    def test_output_shape_3_classes(self, input_dim):
        model = AbstentionModel(input_dim=input_dim, dropout=0.0)
        model.eval()
        x = torch.randn(8, input_dim)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 3), f"Expected (8, 3) got {out.shape}"

    def test_softmax_sums_to_one(self, input_dim):
        model = AbstentionModel(input_dim=input_dim, dropout=0.0)
        model.eval()
        x = torch.randn(16, input_dim)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5), \
            f"Softmax rows don't sum to 1: {sums}"

    def test_abstain_class_exists(self, input_dim):
        """Output must have 3 logits including index 2 (abstain)."""
        model = AbstentionModel(input_dim=input_dim, dropout=0.0)
        x = torch.randn(4, input_dim)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == 3, "Abstain class (index 2) missing"

    def test_no_nan_outputs(self, input_dim):
        model = AbstentionModel(input_dim=input_dim, dropout=0.0)
        x = torch.randn(32, input_dim)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "NaN detected in model outputs"

    def test_dropout_enabled_in_train_mode(self):
        """
        Verify stochasticity in train mode (used by MC Dropout).
        BatchNorm normalises constant-input tensors to zero variance, so we
        use fresh random inputs each forward pass to expose dropout randomness.
        """
        model = AbstentionModel(input_dim=64, dropout=0.5)
        model.train()
        identical_count = 0
        n_trials = 10
        for _ in range(n_trials):
            x = torch.randn(32, 64)     # fresh random input each time
            with torch.no_grad():
                out = model(x)
            # With p=0.5 dropout, identical argmax results on fresh inputs is highly unlikely
            # We track the output for comparison on next pass
            x2 = x.clone()              # same input, different dropout mask
            with torch.no_grad():
                out2 = model(x2)
            if torch.allclose(out, out2, atol=1e-5):
                identical_count += 1
        # Allow at most 2 out of 10 identical (pure chance threshold)
        assert identical_count <= 2, \
            f"Dropout not active: {identical_count}/10 runs produced identical outputs"



class TestBaselineModel:

    def test_output_shape_2_classes(self):
        model = BaselineModel(input_dim=30, dropout=0.0)
        model.eval()
        x = torch.randn(8, 30)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 2), f"Expected (8, 2) got {out.shape}"

    def test_baseline_has_no_abstain_class(self):
        model = BaselineModel(input_dim=30, dropout=0.0)
        x = torch.randn(4, 30)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == 2, "Baseline should output 2 classes (0=legit, 1=fraud)"
