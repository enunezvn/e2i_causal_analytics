"""Tests for the advisory feature-ceiling / separability diagnostic node.

Confirms the node distinguishes a feature-bound (separability) ceiling from a
genuinely separable problem, and that it is a safe no-op (never raises, never
alters flow) on degenerate / non-binary / missing-data inputs.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.agents.ml_foundation.model_trainer.nodes.feature_ceiling_diagnostic import (
    feature_ceiling_diagnostic,
)


def _binary_state(X, y, problem_type="binary_classification"):
    return {
        "problem_type": problem_type,
        "X_train_preprocessed": X,
        "train_data": {"y": y},
    }


@pytest.mark.asyncio
class TestFeatureCeilingDiagnostic:
    async def test_feature_bound_when_signal_is_noise(self):
        """y independent of X (pure noise) at low prevalence -> feature_bound,
        PR-AUC lift ~1 (no skill)."""
        rng = np.random.default_rng(0)
        n = 3000
        X = rng.standard_normal((n, 20))
        y = (rng.random(n) < 0.05).astype(int)  # labels unrelated to X
        result = await feature_ceiling_diagnostic(_binary_state(X, y))
        assert result["feature_ceiling_computed"] is True
        assert result["feature_ceiling_label"] == "feature_bound"
        assert result["feature_ceiling_pr_auc_lift"] < 3.0
        assert "feature" in result["feature_ceiling_note"].lower()

    async def test_separable_when_classes_well_separated(self):
        """Well-separated classes at the SAME low prevalence -> separable,
        high AUC, with no imbalance handling applied."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=3000,
            n_features=20,
            n_informative=10,
            n_redundant=2,
            weights=[0.95, 0.05],
            class_sep=2.5,
            flip_y=0.0,
            random_state=0,
        )
        result = await feature_ceiling_diagnostic(_binary_state(X, y))
        assert result["feature_ceiling_computed"] is True
        assert result["feature_ceiling_label"] == "separable"
        assert result["feature_ceiling_auc"] >= 0.80

    async def test_skips_for_regression(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))
        y = rng.standard_normal(200)
        result = await feature_ceiling_diagnostic(_binary_state(X, y, problem_type="regression"))
        assert result["feature_ceiling_computed"] is False
        assert result["feature_ceiling_label"] == "not_computed"

    async def test_skips_when_no_features(self):
        result = await feature_ceiling_diagnostic(
            {"problem_type": "binary_classification", "train_data": {"y": np.array([0, 1, 0, 1])}}
        )
        assert result["feature_ceiling_computed"] is False

    async def test_skips_single_class(self):
        X = np.random.default_rng(2).standard_normal((100, 5))
        y = np.zeros(100, dtype=int)
        result = await feature_ceiling_diagnostic(_binary_state(X, y))
        assert result["feature_ceiling_computed"] is False

    async def test_skips_when_minority_below_two(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((100, 5))
        y = np.zeros(100, dtype=int)
        y[0] = 1  # exactly one minority sample
        result = await feature_ceiling_diagnostic(_binary_state(X, y))
        assert result["feature_ceiling_computed"] is False

    async def test_advisory_keys_present_and_does_not_raise(self):
        """The node must always return the advisory contract keys."""
        rng = np.random.default_rng(4)
        X = rng.standard_normal((1500, 10))
        y = (rng.random(1500) < 0.1).astype(int)
        result = await feature_ceiling_diagnostic(_binary_state(X, y))
        assert "feature_ceiling_computed" in result
        assert "feature_ceiling_label" in result
        assert "feature_ceiling_note" in result
