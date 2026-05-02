"""Unit tests for Brier custom objective (Phase 1 W4 day-1).

Reference: shard 17 Week 3 Day 1 of `.claude/plans/adaptive_criteria_v3_followup/`.
Acceptance: LightGBM trains with objective=brier_objective_lightgbm; AUC
within 0.02 of logloss-trained baseline on the same synthetic data.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

SEED = 42
N_SAMPLES = 500
N_FEATURES = 8


def _make_logistic_dgp(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Logistic data-generating process with mild signal."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    coefs = rng.standard_normal(N_FEATURES)
    logits = X @ coefs
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=N_SAMPLES) < probs).astype(int)
    return X, y


class _StubDataset:
    """Minimal LightGBM Dataset stand-in for the callback contract test."""

    def __init__(self, y: np.ndarray) -> None:
        self._y = y

    def get_label(self) -> np.ndarray:
        return self._y


class TestBrierObjectiveContract:
    """Contract tests independent of LightGBM (use stub dataset)."""

    def test_gradient_zero_when_pred_perfectly_matches_label(self):
        """At z → +∞ (p=1), if y=1 then gradient = 2(1-1)·1·0 = 0.
        At z → -∞ (p=0), if y=0 then gradient = 2(0-0)·0·1 = 0.
        Use moderate z=±10 for numerical stability.
        """
        from src.mlops.objectives.brier import brier_objective_lightgbm

        z = np.array([10.0, -10.0])
        y = np.array([1.0, 0.0])
        ds = _StubDataset(y)
        grad, hess = brier_objective_lightgbm(z, ds)
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)

    def test_hessian_is_positive_for_all_finite_z(self):
        """Newton-PD diagonal hessian = 2·[p(1-p)]² is always > 0 for p ∈ (0, 1)."""
        from src.mlops.objectives.brier import brier_objective_lightgbm

        z = np.linspace(-5.0, 5.0, 50)
        y = np.zeros_like(z)
        ds = _StubDataset(y)
        _, hess = brier_objective_lightgbm(z, ds)
        assert (hess > 0.0).all()

    def test_shapes_match_input(self):
        """Both gradient + hessian have shape (n,) matching input."""
        from src.mlops.objectives.brier import brier_objective_lightgbm

        z = np.zeros(7)
        y = np.zeros(7)
        ds = _StubDataset(y)
        grad, hess = brier_objective_lightgbm(z, ds)
        assert grad.shape == (7,)
        assert hess.shape == (7,)

    def test_gradient_sign_correct_when_pred_below_target(self):
        """At z=0 (p=0.5), if y=1 then gradient = 2·(0.5-1)·0.5·0.5 = -0.25 < 0.
        Negative gradient means LightGBM should INCREASE z (push p higher).
        """
        from src.mlops.objectives.brier import brier_objective_lightgbm

        z = np.array([0.0])
        y = np.array([1.0])
        ds = _StubDataset(y)
        grad, _ = brier_objective_lightgbm(z, ds)
        assert grad[0] < 0.0
        np.testing.assert_allclose(grad, [-0.25], atol=1e-6)

    def test_gradient_sign_correct_when_pred_above_target(self):
        """At z=0 (p=0.5), if y=0 then gradient = 2·(0.5-0)·0.5·0.5 = +0.25 > 0.
        Positive gradient means LightGBM should DECREASE z (push p lower).
        """
        from src.mlops.objectives.brier import brier_objective_lightgbm

        z = np.array([0.0])
        y = np.array([0.0])
        ds = _StubDataset(y)
        grad, _ = brier_objective_lightgbm(z, ds)
        assert grad[0] > 0.0
        np.testing.assert_allclose(grad, [0.25], atol=1e-6)

    def test_sigmoid_numerical_stability(self):
        """Large positive/negative z should not overflow."""
        from src.mlops.objectives.brier import brier_objective_lightgbm

        z = np.array([100.0, -100.0, 0.0])
        y = np.array([1.0, 0.0, 1.0])
        ds = _StubDataset(y)
        grad, hess = brier_objective_lightgbm(z, ds)
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()


class TestBrierObjectiveLightGBMSmoke:
    """End-to-end smoke: LightGBM with brier_objective should reach AUC
    within 0.02 of logloss-trained baseline on the same synthetic data
    (shard 17 Week 3 Day 1 acceptance criterion).
    """

    def test_lightgbm_trains_with_brier_objective_auc_within_tolerance(self):
        from lightgbm import LGBMClassifier

        from src.mlops.objectives.brier import brier_objective_lightgbm

        X, y = _make_logistic_dgp()
        # Both models share hyperparameters EXCEPT objective. Brier loss has
        # flatter gradients near p=0.5 than logloss, so it converges more
        # slowly; both models use the same n_estimators / learning_rate /
        # max_depth so the AUC comparison reflects objective choice, not
        # tuning differences.
        common_kwargs = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "random_state": SEED,
            "verbose": -1,
        }
        baseline = LGBMClassifier(**common_kwargs)
        baseline.fit(X, y)
        baseline_auc = roc_auc_score(y, baseline.predict_proba(X)[:, 1])

        # Brier custom objective:
        # NOTE: LGBMClassifier with `objective=callable` cannot compute class
        # probabilities (LightGBM warning: "Cannot compute class probabilities
        # or labels due to the usage of customized objective function.
        # Returning raw scores instead."). predict_proba returns 1D raw scores.
        # AUC is rank-invariant so we use the raw scores directly.
        brier_model = LGBMClassifier(
            objective=brier_objective_lightgbm,
            **common_kwargs,
        )
        brier_model.fit(X, y)
        brier_raw = np.asarray(brier_model.predict_proba(X))
        if brier_raw.ndim == 2:
            brier_scores = brier_raw[:, 1]
        else:
            brier_scores = brier_raw  # raw 1D logits
        brier_auc = roc_auc_score(y, brier_scores)

        # AUC tolerance per shard 17 Week 3 Day 1: <0.02 of logloss baseline.
        # Codex cycle 12 (2026-05-02) D-verdict: do NOT widen this tolerance
        # before fixing the Brier hessian. Pre-fix `2·p·(1-p)` (logloss
        # hessian) overestimated Brier curvature 4× at p=0.5 and starved
        # convergence; post-fix `2·[p(1-p)]²` (Newton-PD diagonal of Brier on
        # logit scale) restores convergence and the original 0.02 envelope.
        assert abs(brier_auc - baseline_auc) < 0.02, (
            f"brier_auc={brier_auc:.4f} vs baseline_auc={baseline_auc:.4f} "
            f"(delta={abs(brier_auc - baseline_auc):.4f}, expected <0.02)"
        )

    def test_brier_objective_actually_improves_brier_score(self):
        """The CORE acceptance: Brier-trained model must achieve lower Brier
        loss than the logloss-trained baseline (otherwise the custom objective
        is broken). AUC tolerance covers ranking; this covers calibration —
        the WHOLE POINT of switching to a proper-scoring-rule objective.

        For meaningful comparison we sigmoid-transform the Brier model's
        raw scores back to probabilities. The logloss baseline already
        outputs probabilities natively.
        """
        from lightgbm import LGBMClassifier

        from src.mlops.objectives.brier import _sigmoid, brier_objective_lightgbm

        X, y = _make_logistic_dgp()
        common_kwargs = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "random_state": SEED,
            "verbose": -1,
        }

        baseline = LGBMClassifier(**common_kwargs)
        baseline.fit(X, y)
        baseline_proba = baseline.predict_proba(X)[:, 1]
        baseline_brier = brier_score_loss(y, baseline_proba)

        brier_model = LGBMClassifier(
            objective=brier_objective_lightgbm,
            **common_kwargs,
        )
        brier_model.fit(X, y)
        brier_raw = np.asarray(brier_model.predict_proba(X))
        if brier_raw.ndim == 2:
            brier_scores = brier_raw[:, 1]
        else:
            brier_scores = brier_raw
        # Sigmoid the raw logits back to probability scale for Brier loss.
        brier_proba = _sigmoid(np.asarray(brier_scores, dtype=np.float64))
        brier_brier = brier_score_loss(y, brier_proba)

        # Brier-trained should have LOWER Brier loss than logloss baseline.
        # Tolerance 0.005 lets Brier win or be within noise of baseline; if
        # Brier is meaningfully WORSE on its own loss function, the
        # gradient/hessian formulas are wrong.
        assert brier_brier <= baseline_brier + 0.005, (
            f"brier_model_brier_loss={brier_brier:.4f} should be <= "
            f"baseline_brier_loss={baseline_brier:.4f} + 0.005 — "
            "Brier custom objective is not actually optimizing Brier loss"
        )
