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

    def test_tier0_alt_train_unpacks_helper_in_lr_family_branch(self, tier0_ast: ast.AST) -> None:
        """Non-vacuous AST check (codex MED-2): the alt-train builder must
        contain an ``if alt["name"] in (... LogisticRegression ...
        LogisticRegression_Conformal ...)`` body that assigns a dict
        literal containing ``**_LR_FIXED_PARAMS``. Pre-fix the unpack
        does not exist; under the fix this assertion holds exactly at the
        alt-train use site, so reverting just the unpack (not the import,
        not the literal) trips this test.
        """
        # Walk every ``if`` node whose Compare test mentions a Tuple of
        # str constants including both LR names. Inside that branch's body,
        # there must be at least one dict literal with a ``**_LR_FIXED_PARAMS``
        # keyword (i.e. a Dict node whose keys list contains None matched to
        # a Name value ``_LR_FIXED_PARAMS``).
        found = False
        for node in ast.walk(tier0_ast):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            # Look for a Compare node with `in` operator and Tuple right-hand
            # side containing both LR family names.
            if not isinstance(test, ast.Compare):
                continue
            if not (len(test.ops) == 1 and isinstance(test.ops[0], ast.In)):
                continue
            rhs = test.comparators[0] if test.comparators else None
            if not isinstance(rhs, ast.Tuple):
                continue
            tuple_strs = {
                elt.value
                for elt in rhs.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
            if not {"LogisticRegression", "LogisticRegression_Conformal"} <= tuple_strs:
                continue
            # Found the LR-family branch. Now scan its body for a dict
            # literal that double-stars _LR_FIXED_PARAMS.
            for body_node in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if not isinstance(body_node, ast.Dict):
                    continue
                # ``**X`` shows up as a (key=None, value=Name('X')) pair.
                for k, v in zip(body_node.keys, body_node.values, strict=True):
                    if k is None and isinstance(v, ast.Name) and v.id == "_LR_FIXED_PARAMS":
                        found = True
                        break
                if found:
                    break
            if found:
                break

        assert found, (
            "Did not find an ``if alt['name'] in (... 'LogisticRegression', "
            "'LogisticRegression_Conformal' ...)`` branch whose body unpacks "
            "``**_LR_FIXED_PARAMS`` into a dict literal in "
            "scripts/run_tier0_test.py. The alt-train must actually consume "
            "the helper inside that branch, not merely reference it elsewhere."
        )


# ---------------------------------------------------------------------------
# Defense-in-depth (codex MED-1): the final training constructor in
# ``model_trainer_node._filter_hyperparameters`` also merges _LR_FIXED_PARAMS
# so direct callers that bypass HPO can't smuggle penalty=l1 + lbfgs through.
# ---------------------------------------------------------------------------


class TestModelTrainerNodeFiltersLRFamily:
    """Defense-in-depth: ``_filter_hyperparameters`` injects ``solver=saga``
    and ``max_iter=1000`` for LR + LR_Conformal when not already set.
    """

    def _call_filter(self, algorithm_name: str, hyperparameters):
        from src.agents.ml_foundation.model_trainer.nodes import model_trainer_node

        return model_trainer_node._filter_hyperparameters(algorithm_name, hyperparameters)

    def test_lr_gets_saga_in_filter(self) -> None:
        filtered = self._call_filter("LogisticRegression", {"penalty": "l1", "C": 1.0})
        assert filtered["solver"] == "saga"
        assert filtered["max_iter"] == 1000

    def test_lr_conformal_gets_saga_in_filter(self) -> None:
        filtered = self._call_filter("LogisticRegression_Conformal", {"penalty": "l1", "C": 1.0})
        assert filtered["solver"] == "saga"
        assert filtered["max_iter"] == 1000

    def test_caller_override_wins(self) -> None:
        """If the caller has explicitly set ``solver`` or ``max_iter``, the
        helper must not overwrite it — preserves the original constructor
        contract.
        """
        filtered = self._call_filter(
            "LogisticRegression",
            {"penalty": "l2", "solver": "liblinear", "max_iter": 500},
        )
        assert filtered["solver"] == "liblinear"
        assert filtered["max_iter"] == 500


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


# ---------------------------------------------------------------------------
# Issue #273 — algorithm_registry.py's LR-family ``default_hyperparameters``
# MUST agree with ``_LR_FIXED_PARAMS`` (SSOT) for ``solver`` and ``max_iter``.
#
# Approach chosen: Option (B) — runtime tests + an in-source anchor check, NOT
# a module-level import of ``_LR_FIXED_PARAMS`` into ``algorithm_registry.py``.
# The direct import would create a circular dependency:
#
#     algorithm_registry
#       -> hyperparameter_tuner (target import)
#         -> model_trainer (parent package)
#           -> model_trainer.nodes.__init__
#             -> quality_remediation
#               -> algorithm_registry  (cycle back-edge)
#
# Empirically verified during this fix attempt: the module-level import trips
# ``ImportError: cannot import name 'REGULARIZATION_SEARCH_SPACE' from
# partially initialized module ... (most likely due to a circular import)``.
# Per the issue ACs, Option (B) is the accepted fallback.
#
# Failure mode: a future edit to the registry's LR defaults (e.g. flipping
# ``solver=saga`` to ``solver=lbfgs`` or dropping ``max_iter``) could silently
# re-introduce the lbfgs+l1 crash from #232. The runtime AC checks trip on
# divergence; the source-anchor check keeps the SSOT pointer in-file so a
# developer editing the registry sees the contract.
# ---------------------------------------------------------------------------


class TestIssue273RegistryDefaultsAgreeWithSSOT:
    """Issue #273: ``algorithm_registry.py`` LR-family ``default_hyperparameters``
    must contain ``solver`` and ``max_iter`` values that agree with
    ``_LR_FIXED_PARAMS`` (SSOT). Drift trips these tests.

    Four-pronged guard:

    1. Runtime invariant — the registry's runtime values must equal SSOT.
       Any divergent value (e.g. ``solver="liblinear"``) trips the test.
    2. Runtime negative invariant — no SSOT key may carry a *conflicting*
       value (guards against partial drift, e.g. ``solver`` synced but
       ``max_iter`` stale).
    3. End-to-end smoke — building a sklearn LogisticRegression from the
       registry's defaults with ``penalty=l1`` must not raise. Recreates the
       exact #232 crash if the registry ever reverts to ``solver=lbfgs``.
    4. Source anchor — registry source must mention ``_LR_FIXED_PARAMS`` and
       ``hyperparameter_tuner`` so editing developers find the contract.
    """

    # Path discovered at import time so the test fails-loud if the module moves.
    REGISTRY_PATH = (
        REPO_ROOT
        / "src"
        / "agents"
        / "ml_foundation"
        / "model_selector"
        / "nodes"
        / "algorithm_registry.py"
    )

    @pytest.fixture(scope="class")
    def registry_source(self) -> str:
        assert self.REGISTRY_PATH.exists(), (
            f"algorithm_registry.py not found at {self.REGISTRY_PATH}"
        )
        return self.REGISTRY_PATH.read_text()

    def test_registry_lr_defaults_agree_with_ssot(self) -> None:
        """Runtime invariant: the registry's LR ``default_hyperparameters``
        contains ``solver`` and ``max_iter`` with values matching SSOT.
        Any divergent value (e.g. ``solver="liblinear"``) trips this test.
        """
        from src.agents.ml_foundation.model_selector.nodes.algorithm_registry import (
            ALGORITHM_REGISTRY,
        )

        lr_defaults = ALGORITHM_REGISTRY["LogisticRegression"]["default_hyperparameters"]
        for key, value in _LR_FIXED_PARAMS.items():
            assert key in lr_defaults, (
                f"Registry LogisticRegression.default_hyperparameters is missing "
                f"SSOT key {key!r}. Expected {value!r} (per _LR_FIXED_PARAMS)."
            )
            assert lr_defaults[key] == value, (
                f"Registry LogisticRegression.default_hyperparameters[{key!r}] = "
                f"{lr_defaults[key]!r} diverges from SSOT _LR_FIXED_PARAMS[{key!r}] = "
                f"{value!r}. Sync the values, or refactor to remove the duplicate."
            )

    def test_registry_lr_defaults_has_no_conflicting_keys(self) -> None:
        """Negative invariant: the registry's LR ``default_hyperparameters``
        must not contain any SSOT key with a *conflicting* value. Guards
        against partial drift (e.g. ``solver`` synced but ``max_iter`` stale).
        """
        from src.agents.ml_foundation.model_selector.nodes.algorithm_registry import (
            ALGORITHM_REGISTRY,
        )

        lr_defaults = ALGORITHM_REGISTRY["LogisticRegression"]["default_hyperparameters"]
        conflicts = {
            key: (lr_defaults[key], _LR_FIXED_PARAMS[key])
            for key in _LR_FIXED_PARAMS
            if key in lr_defaults and lr_defaults[key] != _LR_FIXED_PARAMS[key]
        }
        assert not conflicts, (
            f"Registry LR default_hyperparameters has values that conflict "
            f"with _LR_FIXED_PARAMS (SSOT): {conflicts}. Sync the literals."
        )

    def test_registry_lr_solver_is_saga_safe_for_l1(self) -> None:
        """End-to-end smoke: build a LogisticRegression from the registry's
        default_hyperparameters with ``penalty=l1`` (overriding the default
        ``l2``) and confirm construction+fit does not raise. Pre-fix scenario:
        if the registry ever flips back to ``solver=lbfgs``, this trips with
        ``ValueError: Solver lbfgs supports only 'l2' or None penalties``.
        """
        from sklearn.linear_model import LogisticRegression

        from src.agents.ml_foundation.model_selector.nodes.algorithm_registry import (
            ALGORITHM_REGISTRY,
        )

        lr_defaults = dict(ALGORITHM_REGISTRY["LogisticRegression"]["default_hyperparameters"])
        # Simulate the failure scenario: HPO trial samples penalty=l1.
        lr_defaults["penalty"] = "l1"
        import numpy as np

        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 4))
        y = (rng.random(40) > 0.5).astype(int)
        clf = LogisticRegression(**lr_defaults)
        clf.fit(X, y)  # would raise if solver had reverted to lbfgs
        assert clf.coef_.shape == (1, 4)

    def test_registry_docstring_references_ssot_module(self, registry_source: str) -> None:
        """Anchor invariant: the registry source must mention
        ``_LR_FIXED_PARAMS`` and the SSOT module path, so a developer editing
        the LR literals sees the contract and runs this test. Pure text
        check — cheap and AST-free.
        """
        assert "_LR_FIXED_PARAMS" in registry_source, (
            "algorithm_registry.py must mention `_LR_FIXED_PARAMS` in its "
            "docstring or comments so the SSOT contract is auditable in-source."
        )
        assert "hyperparameter_tuner" in registry_source, (
            "algorithm_registry.py must mention `hyperparameter_tuner` so the "
            "SSOT module path is auditable in-source."
        )
