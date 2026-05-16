"""TDD coverage for issue #232 — LR + LR_Conformal share ``solver="saga"``.

Acceptance criteria (verbatim from the issue):

A1. Module-level ``_LR_FIXED_PARAMS = {"max_iter": 1000, "solver": "saga"}`` lives
    in ``hyperparameter_tuner.py`` and is imported by ``scripts/run_tier0_test.py``.
A2. HPO dispatcher routes BOTH ``LogisticRegression`` AND
    ``LogisticRegression_Conformal`` through ``_LR_FIXED_PARAMS``.
A3. Path 1 (tier-0 alt-train) builds with
    ``**_LR_FIXED_PARAMS, "class_weight": "balanced"``.

These tests are deliberately tight on shape and call-site so that the two
paths can't drift independently. See [[issue-#232]] for the failure mode
(``Solver lbfgs supports only 'l2' or None penalties, got l1 penalty.``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.agents.ml_foundation.model_trainer.nodes import hyperparameter_tuner
from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import (
    _LR_FIXED_PARAMS,
    _get_fixed_params,
)

# tests/unit/test_agents/test_ml_foundation/test_model_trainer/test_*.py
#    -> parents[5] = repo root
REPO_ROOT = Path(__file__).resolve().parents[5]
TIER0_SCRIPT = REPO_ROOT / "scripts" / "run_tier0_test.py"
assert TIER0_SCRIPT.exists(), f"tier0 script not found at {TIER0_SCRIPT}"


# ---------------------------------------------------------------------------
# A1 — module-level helper constant
# ---------------------------------------------------------------------------


class TestA1HelperConstantExists:
    """A1: ``_LR_FIXED_PARAMS`` is a module-level dict with saga+max_iter."""

    def test_helper_is_module_level_dict(self) -> None:
        assert isinstance(_LR_FIXED_PARAMS, dict)

    def test_helper_pins_solver_saga(self) -> None:
        assert _LR_FIXED_PARAMS["solver"] == "saga"

    def test_helper_pins_max_iter_1000(self) -> None:
        assert _LR_FIXED_PARAMS["max_iter"] == 1000

    def test_helper_keys_are_exactly_solver_and_max_iter(self) -> None:
        # Keep the contract narrow — adding new pinned kwargs is a separate
        # decision. Random_state is per-call (not module-level).
        assert set(_LR_FIXED_PARAMS) == {"solver", "max_iter"}

    def test_helper_attribute_on_module(self) -> None:
        # Belt-and-suspenders: importing it via attribute access must work
        # because run_tier0_test.py imports it that way.
        assert hasattr(hyperparameter_tuner, "_LR_FIXED_PARAMS")
        assert hyperparameter_tuner._LR_FIXED_PARAMS["solver"] == "saga"


# ---------------------------------------------------------------------------
# A2 — HPO dispatcher consumes the helper for BOTH LR + LR_Conformal
# ---------------------------------------------------------------------------


class TestA2DispatcherRoutesBothLRFamilies:
    """A2: ``_get_fixed_params`` returns the same saga/max_iter pinning for
    ``LogisticRegression`` AND ``LogisticRegression_Conformal``.
    """

    def test_logistic_regression_gets_saga(self) -> None:
        params = _get_fixed_params("LogisticRegression")
        assert params["solver"] == "saga"
        assert params["max_iter"] == 1000
        assert params["random_state"] == 42  # per-call default

    def test_logistic_regression_conformal_gets_saga(self) -> None:
        """The original bug: exact-string ``== "LogisticRegression"`` made
        ``_Conformal`` fall through, so HPO Trial 0/1/2 with penalty=l1 all
        returned -inf. After A2, both algorithms hit the same branch.
        """
        params = _get_fixed_params("LogisticRegression_Conformal")
        assert params["solver"] == "saga"
        assert params["max_iter"] == 1000
        assert params["random_state"] == 42

    def test_lr_and_lr_conformal_return_equivalent_lr_keys(self) -> None:
        """The LR-family pinning must be identical across the two names so
        the two paths cannot silently drift."""
        lr = _get_fixed_params("LogisticRegression")
        lr_conf = _get_fixed_params("LogisticRegression_Conformal")
        lr_keys = {k: lr[k] for k in ("solver", "max_iter")}
        lr_conf_keys = {k: lr_conf[k] for k in ("solver", "max_iter")}
        assert lr_keys == lr_conf_keys

    def test_lr_dispatcher_consumes_helper_constant(self) -> None:
        """Ensure the dispatcher pulls from ``_LR_FIXED_PARAMS`` (not a
        re-typed literal). Mutating the module attribute must change the
        dispatcher output — proving the dispatcher reads the helper.
        """
        sentinel = {"max_iter": 7777, "solver": "saga"}
        original = hyperparameter_tuner._LR_FIXED_PARAMS
        hyperparameter_tuner._LR_FIXED_PARAMS = sentinel
        try:
            params = _get_fixed_params("LogisticRegression")
            assert params["max_iter"] == 7777
            conf = _get_fixed_params("LogisticRegression_Conformal")
            assert conf["max_iter"] == 7777
        finally:
            hyperparameter_tuner._LR_FIXED_PARAMS = original

    def test_unknown_algorithm_still_returns_empty(self) -> None:
        # Regression guard: widening the LR branch must not accidentally
        # leak saga into unrelated algorithms.
        assert _get_fixed_params("UnknownAlgorithm") == {}

    def test_random_forest_unaffected(self) -> None:
        params = _get_fixed_params("RandomForest")
        assert "solver" not in params


# ---------------------------------------------------------------------------
# A3 — tier-0 alt-train consumes ``_LR_FIXED_PARAMS``
# ---------------------------------------------------------------------------


class TestA3Tier0AltTrainImportsHelper:
    """A3: ``scripts/run_tier0_test.py`` imports ``_LR_FIXED_PARAMS`` and uses
    it in the Step 5b alt-train builder so the alt-LR cannot be built with
    ``solver=lbfgs`` + ``penalty=l1``.
    """

    @pytest.fixture(scope="class")
    def tier0_source(self) -> str:
        return TIER0_SCRIPT.read_text()

    @pytest.fixture(scope="class")
    def tier0_ast(self, tier0_source: str) -> ast.AST:
        return ast.parse(tier0_source)

    def test_tier0_imports_lr_fixed_params(self, tier0_ast: ast.AST) -> None:
        """``from ...hyperparameter_tuner import _LR_FIXED_PARAMS`` must be
        present somewhere in the script."""
        found = False
        for node in ast.walk(tier0_ast):
            if isinstance(node, ast.ImportFrom):
                if node.module and "hyperparameter_tuner" in node.module:
                    for alias in node.names:
                        if alias.name == "_LR_FIXED_PARAMS":
                            found = True
                            break
        assert found, (
            "scripts/run_tier0_test.py must import _LR_FIXED_PARAMS from "
            "src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner"
        )

    def test_tier0_alt_train_references_helper(self, tier0_source: str) -> None:
        """Source must contain a use site for ``_LR_FIXED_PARAMS`` (typically
        a ``**_LR_FIXED_PARAMS`` spread inside the alt-train builder).
        """
        # At least two textual occurrences expected: the import and one
        # spread/use site. We assert >=2 so renaming/imports stay consistent.
        assert tier0_source.count("_LR_FIXED_PARAMS") >= 2, (
            "scripts/run_tier0_test.py must reference _LR_FIXED_PARAMS at "
            "least twice (import + spread into alt-train builder)"
        )

    def test_tier0_alt_train_handles_lr_conformal(self, tier0_source: str) -> None:
        """The alt-train branch must include ``LogisticRegression_Conformal``
        in the LR-family check so the conformal variant also gets the
        saga pinning when surfaced as an alt-candidate.
        """
        # Anchor on the LR-family literal used in the alt-train builder.
        # The fix expands the membership tuple from
        # ("RandomForest", "LogisticRegression")
        # to include "LogisticRegression_Conformal".
        assert "LogisticRegression_Conformal" in tier0_source, (
            "scripts/run_tier0_test.py must reference LogisticRegression_"
            "Conformal in the alt-train LR-family branch so the conformal "
            "variant inherits solver=saga"
        )


# ---------------------------------------------------------------------------
# Cross-cutting / smoke: the failing builder no longer crashes
# ---------------------------------------------------------------------------


class TestA4LRBuildsWithL1Penalty:
    """A4 surrogate: build LR-family with ``penalty="l1"`` using the merged
    fixed_params + a representative HPO sample. Pre-fix this raises
    ``ValueError: Solver lbfgs supports only 'l2' or None penalties``.
    """

    def test_logistic_regression_l1_constructs(self) -> None:
        from sklearn.linear_model import LogisticRegression

        fixed = _get_fixed_params("LogisticRegression")
        # Simulate Optuna trial picking penalty=l1.
        params = {**fixed, "C": 0.0746, "penalty": "l1"}
        # Construction + fit on tiny data: pre-fix this crashes during fit.
        import numpy as np

        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 4))
        y = (rng.random(40) > 0.5).astype(int)
        clf = LogisticRegression(**params)
        clf.fit(X, y)  # must not raise
        assert clf.coef_.shape == (1, 4)

    def test_logistic_regression_conformal_l1_constructs(self) -> None:
        """Same shape as the registry's ``LogisticRegression_Conformal`` HPO
        space: penalty=l1 must construct after the dispatcher fix.
        """
        from sklearn.linear_model import LogisticRegression

        fixed = _get_fixed_params("LogisticRegression_Conformal")
        params = {**fixed, "C": 0.0746, "penalty": "l1"}
        import numpy as np

        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 4))
        y = (rng.random(40) > 0.5).astype(int)
        clf = LogisticRegression(**params)
        clf.fit(X, y)
        assert clf.coef_.shape == (1, 4)
