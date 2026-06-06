"""Tests for the LogisticRegression solver/penalty reconciliation policy.

Issue #232 runtime follow-up: saga is the l1-safe floor, but lbfgs is far
faster for l2/None at identical AUC. ``reconcile_lr_solver`` picks the
fastest valid solver for the known penalty, while leaving non-LR / non-
managed-solver param dicts untouched.
"""

import pytest

from src.mlops.lr_solver_policy import (
    lr_solver_for_penalty,
    reconcile_lr_solver,
)


class TestLrSolverForPenalty:
    @pytest.mark.parametrize(
        "penalty,expected",
        [
            ("l1", "saga"),
            ("elasticnet", "saga"),
            ("L1", "saga"),  # case-insensitive
            ("ElasticNet", "saga"),
            ("l2", "lbfgs"),
            ("none", "lbfgs"),
            (None, "lbfgs"),
        ],
    )
    def test_solver_for_penalty(self, penalty, expected):
        assert lr_solver_for_penalty(penalty) == expected


class TestReconcileLrSolver:
    def test_l2_downgrades_saga_to_lbfgs(self):
        params = {"solver": "saga", "penalty": "l2", "C": 1.0}
        out = reconcile_lr_solver(params)
        assert out["solver"] == "lbfgs"
        assert out is params  # mutates in place

    def test_l1_keeps_saga(self):
        assert reconcile_lr_solver({"solver": "saga", "penalty": "l1"})["solver"] == "saga"

    def test_elasticnet_keeps_saga(self):
        assert reconcile_lr_solver({"solver": "saga", "penalty": "elasticnet"})["solver"] == "saga"

    def test_absent_penalty_defaults_to_lbfgs(self):
        # No penalty key -> sklearn LR default is l2 -> lbfgs is valid + fast.
        assert reconcile_lr_solver({"solver": "saga"})["solver"] == "lbfgs"

    def test_l1_with_lbfgs_is_corrected_to_saga(self):
        # Safety direction: an l1 penalty must never be left on lbfgs.
        assert reconcile_lr_solver({"solver": "lbfgs", "penalty": "l1"})["solver"] == "saga"

    def test_non_managed_solver_untouched(self):
        # liblinear (coefficient_sensitivity) must not be rewritten.
        params = {"solver": "liblinear", "penalty": "l2"}
        assert reconcile_lr_solver(params)["solver"] == "liblinear"

    def test_no_solver_is_noop(self):
        # Non-LR estimator params (no solver) pass through unchanged.
        params = {"max_depth": 5, "penalty": "l1"}
        assert reconcile_lr_solver(dict(params)) == params

    def test_empty_dict_is_noop(self):
        assert reconcile_lr_solver({}) == {}
