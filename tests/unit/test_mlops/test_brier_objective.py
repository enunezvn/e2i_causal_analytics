"""Unit tests for Brier custom objective (Phase 1 W4 days 1+2).

Reference: shard 17 Week 3 Days 1 + 2 of
`.claude/plans/adaptive_criteria_v3_followup/`. Day 1 = LightGBM,
Day 2 = XGBoost. Both share the chain-rule logit-scale gradient
``2(p-y)·p(1-p)`` + Newton-PD-Brier hessian ``2·[p(1-p)]²`` per
shard 17 footnote ``[1]``.

Acceptance:
- LightGBM trains with ``objective=brier_objective_lightgbm``; AUC
  within 0.02 of logloss-trained baseline on the same synthetic data.
- XGBoost trains with ``objective=brier_objective_xgboost`` (same
  acceptance criterion; convergence requires more iterations because
  XGBoost's default ``reg_lambda=1.0`` + tree_method='hist' converge
  Brier-loss more slowly than LightGBM with the same math — see
  ``brier_objective_xgboost`` docstring).
"""

from __future__ import annotations

import numpy as np
import pytest
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
    """Minimal LightGBM Dataset / XGBoost DMatrix stand-in for contract tests.

    Both libraries' native APIs pass an object with ``.get_label()`` as the
    second argument; this stub satisfies that duck-typing contract.
    """

    def __init__(self, y: np.ndarray) -> None:
        self._y = y

    def get_label(self) -> np.ndarray:
        return self._y


# Parametrize contract tests over BOTH brier_objective_* symbols. Both share
# the same _brier_grad_hess_from_raw_scores core; this proves the public
# wrappers are equivalent on the same input (both LightGBM + XGBoost custom-
# objective conventions are identical: sklearn `(y_true, y_pred)`, native
# `(y_pred, dmatrix-or-dataset)`).
@pytest.fixture(params=["brier_objective_lightgbm", "brier_objective_xgboost"])
def objective_fn(request):
    """Return the named brier_objective_* callable."""
    from src.mlops.objectives import brier as brier_mod

    return getattr(brier_mod, request.param)


class TestBrierObjectiveContract:
    """Contract tests independent of LightGBM/XGBoost (use stub dataset).

    Parametrized over both ``brier_objective_lightgbm`` and
    ``brier_objective_xgboost`` to verify the public wrappers preserve the
    chain-rule logit-scale math contract identically. Both delegate to
    ``_brier_grad_hess_from_raw_scores``; this test class is the equivalence
    proof for cross-booster math correctness.
    """

    def test_gradient_zero_when_pred_perfectly_matches_label(self, objective_fn):
        """At z → +∞ (p=1), if y=1 then gradient = 2(1-1)·1·0 = 0.
        At z → -∞ (p=0), if y=0 then gradient = 2(0-0)·0·1 = 0.
        Use moderate z=±10 for numerical stability.
        """
        z = np.array([10.0, -10.0])
        y = np.array([1.0, 0.0])
        ds = _StubDataset(y)
        grad, _ = objective_fn(z, ds)
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)

    def test_hessian_is_positive_for_all_finite_z(self, objective_fn):
        """Newton-PD diagonal hessian = 2·[p(1-p)]² is always > 0 for p ∈ (0, 1)."""
        z = np.linspace(-5.0, 5.0, 50)
        y = np.zeros_like(z)
        ds = _StubDataset(y)
        _, hess = objective_fn(z, ds)
        assert (hess > 0.0).all()

    def test_shapes_match_input(self, objective_fn):
        """Both gradient + hessian have shape (n,) matching input."""
        z = np.zeros(7)
        y = np.zeros(7)
        ds = _StubDataset(y)
        grad, hess = objective_fn(z, ds)
        assert grad.shape == (7,)
        assert hess.shape == (7,)

    def test_gradient_sign_correct_when_pred_below_target(self, objective_fn):
        """At z=0 (p=0.5), if y=1 then gradient = 2·(0.5-1)·0.5·0.5 = -0.25 < 0.
        Negative gradient means the booster should INCREASE z (push p higher).
        """
        z = np.array([0.0])
        y = np.array([1.0])
        ds = _StubDataset(y)
        grad, _ = objective_fn(z, ds)
        assert grad[0] < 0.0
        np.testing.assert_allclose(grad, [-0.25], atol=1e-6)

    def test_gradient_sign_correct_when_pred_above_target(self, objective_fn):
        """At z=0 (p=0.5), if y=0 then gradient = 2·(0.5-0)·0.5·0.5 = +0.25 > 0.
        Positive gradient means the booster should DECREASE z (push p lower).
        """
        z = np.array([0.0])
        y = np.array([0.0])
        ds = _StubDataset(y)
        grad, _ = objective_fn(z, ds)
        assert grad[0] > 0.0
        np.testing.assert_allclose(grad, [0.25], atol=1e-6)

    def test_sigmoid_numerical_stability(self, objective_fn):
        """Large positive/negative z should not overflow."""
        z = np.array([100.0, -100.0, 0.0])
        y = np.array([1.0, 0.0, 1.0])
        ds = _StubDataset(y)
        grad, hess = objective_fn(z, ds)
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()


def test_lightgbm_and_xgboost_brier_callables_are_mathematically_identical():
    """Both `brier_objective_*` symbols must produce bit-identical output for
    the same inputs (they share `_brier_grad_hess_from_raw_scores` core).
    This test pins the equivalence contract: any future divergence is a bug.
    """
    from src.mlops.objectives.brier import (
        brier_objective_lightgbm,
        brier_objective_xgboost,
    )

    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(50)
    y = rng.integers(0, 2, size=50).astype(np.float64)

    # Sklearn-style call: (y_true, y_pred)
    g_l, h_l = brier_objective_lightgbm(y, z)
    g_x, h_x = brier_objective_xgboost(y, z)
    np.testing.assert_array_equal(g_l, g_x)
    np.testing.assert_array_equal(h_l, h_x)

    # Native-style call: (y_pred, dataset_or_dmatrix)
    ds = _StubDataset(y)
    g_l2, h_l2 = brier_objective_lightgbm(z, ds)
    g_x2, h_x2 = brier_objective_xgboost(z, ds)
    np.testing.assert_array_equal(g_l2, g_x2)
    np.testing.assert_array_equal(h_l2, h_x2)


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
            # n_jobs=1 caps LightGBM's internal OMP-thread count to 1; under
            # pytest-xdist `-n auto` (cycle-12 cosmetic follow-up) multiple
            # workers each spawning N threads exhausted droplet swap and
            # crashed gw3. Single-threaded keeps memory bounded; smoke runtime
            # is acceptable (~30s per smoke test).
            "n_jobs": 1,
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
            # n_jobs=1 caps LightGBM's internal OMP-thread count to 1; under
            # pytest-xdist `-n auto` (cycle-12 cosmetic follow-up) multiple
            # workers each spawning N threads exhausted droplet swap and
            # crashed gw3. Single-threaded keeps memory bounded; smoke runtime
            # is acceptable (~30s per smoke test).
            "n_jobs": 1,
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


class TestBrierObjectiveXGBoostSmoke:
    """End-to-end smoke: XGBoost with brier_objective_xgboost should reach AUC
    within 0.02 of logloss-trained baseline (shard 17 Week 3 Day 2 acceptance).

    Hyperparameter note: XGBoost smoke uses ``n_estimators=1000,
    learning_rate=0.10`` (LightGBM smoke uses ``n_estimators=300,
    learning_rate=0.05``). The dominant cause is **gradient-magnitude
    scaling**: Brier gradient ``2(p-y)·p(1-p)`` peaks at 0.125, logloss
    gradient ``p-y`` peaks at 0.5 — a 4× deficit per leaf step that
    requires ~4× more iterations to accumulate equivalent cumulative
    leaf score. ``reg_lambda`` is secondary; codex cycle-13 empirical
    sweep (2026-05-02) confirmed:
      - n_est=300, lr=0.05, λ=1: Brier delta = +0.0100 (FAILS ≤+0.005)
      - n_est=300, lr=0.05, λ=0: Brier delta = +0.0123 (still FAILS, slightly worse)
      - n_est=600, lr=0.10, λ=1: Brier delta = +0.0056 (slightly FAILS)
      - n_est=1000, lr=0.10, λ=1 [CHOSEN]: Brier delta = +0.0046 (PASSES)
    AUC delta is 0.0000 across all configs (XGBoost ranks identically with
    either objective; the gap is purely calibration / iteration budget).
    """

    def test_xgboost_trains_with_brier_objective_auc_within_tolerance(self):
        from xgboost import XGBClassifier

        from src.mlops.objectives.brier import brier_objective_xgboost

        X, y = _make_logistic_dgp()
        common_kwargs = {
            "n_estimators": 1000,
            "learning_rate": 0.10,
            "max_depth": 4,
            "random_state": SEED,
            "verbosity": 0,
            # n_jobs=1 caps XGBoost's internal OMP-thread count to 1 (cycle-12
            # cosmetic follow-up; mirrors LightGBM smoke).
            "n_jobs": 1,
        }
        baseline = XGBClassifier(**common_kwargs)
        baseline.fit(X, y)
        baseline_auc = roc_auc_score(y, baseline.predict_proba(X)[:, 1])

        # XGBClassifier with `objective=callable` returns 2-column predict_proba
        # where column 1 is sigmoid-applied (XGBoost applies the binary link
        # internally for predict_proba even with custom objective). Use the
        # column-1 probability directly.
        brier_model = XGBClassifier(
            objective=brier_objective_xgboost,
            **common_kwargs,
        )
        brier_model.fit(X, y)
        brier_proba = brier_model.predict_proba(X)[:, 1]
        brier_auc = roc_auc_score(y, brier_proba)

        # AUC tolerance per shard 17 Week 3 Day 2: <0.02 of logloss baseline.
        # Codex cycle 12 D-verdict applies verbatim — do NOT widen tolerance
        # to mask any math defects. The chain-rule + Newton-PD-Brier hessian
        # is the same as the LightGBM Day-1 path post-cycle-12 fix.
        assert abs(brier_auc - baseline_auc) < 0.02, (
            f"brier_auc={brier_auc:.4f} vs baseline_auc={baseline_auc:.4f} "
            f"(delta={abs(brier_auc - baseline_auc):.4f}, expected <0.02)"
        )

    def test_brier_objective_actually_improves_brier_score_xgboost(self):
        """Mirror of LightGBM smoke #2: Brier-trained XGBoost must achieve
        Brier loss within 0.005 of the logloss baseline (i.e., the custom
        objective is actually optimizing Brier, not diverging).

        Unlike LGBMClassifier (which returns raw scores from predict_proba
        when objective=callable), XGBClassifier applies sigmoid internally
        even with custom objective — predict_proba returns proper [0, 1]
        probabilities, so no sigmoid post-processing is needed.
        """
        from xgboost import XGBClassifier

        from src.mlops.objectives.brier import brier_objective_xgboost

        X, y = _make_logistic_dgp()
        common_kwargs = {
            "n_estimators": 1000,
            "learning_rate": 0.10,
            "max_depth": 4,
            "random_state": SEED,
            "verbosity": 0,
            "n_jobs": 1,
        }

        baseline = XGBClassifier(**common_kwargs)
        baseline.fit(X, y)
        baseline_proba = baseline.predict_proba(X)[:, 1]
        baseline_brier = brier_score_loss(y, baseline_proba)

        brier_model = XGBClassifier(
            objective=brier_objective_xgboost,
            **common_kwargs,
        )
        brier_model.fit(X, y)
        brier_proba = brier_model.predict_proba(X)[:, 1]
        brier_brier = brier_score_loss(y, brier_proba)

        # Brier-trained should be within 0.005 of logloss baseline. Larger
        # iteration budget (vs LightGBM smoke) compensates XGBoost's slower
        # convergence; if the gap is meaningfully wider, the gradient/hessian
        # formulas are wrong (regression to the cycle-12 LightGBM bug).
        assert brier_brier <= baseline_brier + 0.005, (
            f"brier_model_brier_loss={brier_brier:.4f} should be <= "
            f"baseline_brier_loss={baseline_brier:.4f} + 0.005 — "
            "Brier custom objective is not actually optimizing Brier loss "
            "(XGBoost path)"
        )
