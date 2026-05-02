"""W3-lite Day 3 (shard 17 W3 row Day 3) — fold_random_state plumbing.

Acceptance per shard 17: "All split-touching nodes accept ``fold_random_state``
parameter." The three nodes named in the kickoff are:

  1. ``model_trainer/nodes/split_loader.py``  — must propagate the value
  2. ``model_trainer/nodes/hyperparameter_tuner.py`` — must resolve the value
     into ``_get_fixed_params(random_state=...)``
  3. ``model_trainer/nodes/model_trainer_node.py`` — must override the
     ``random_state`` baked into ``best_hyperparameters`` when a per-fold seed
     is set

Resolution precedence (per shard 21 §A audit table row "Hardcoded
``random_state=42`` sites"): ``state['fold_random_state']`` >
``state['random_state']`` > ``fallback`` (default 42).

The Day-3 slice intentionally stops at parameter acceptance + threading —
the orchestrator that *generates* per-fold seeds and the
``RepeatedStratifiedSplitter`` (shard 21 §A) are Day 4-5 work.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helper module contract: src/agents/ml_foundation/model_trainer/random_state.py
# ---------------------------------------------------------------------------


class TestResolveFoldRandomState:
    """Helper precedence: fold_random_state > random_state > fallback."""

    def test_returns_fold_random_state_when_set(self) -> None:
        from src.agents.ml_foundation.model_trainer.random_state import (
            resolve_fold_random_state,
        )

        assert resolve_fold_random_state({"fold_random_state": 99}) == 99

    def test_falls_back_to_state_random_state(self) -> None:
        from src.agents.ml_foundation.model_trainer.random_state import (
            resolve_fold_random_state,
        )

        assert resolve_fold_random_state({"random_state": 7}) == 7

    def test_falls_back_to_default_42(self) -> None:
        from src.agents.ml_foundation.model_trainer.random_state import (
            resolve_fold_random_state,
        )

        assert resolve_fold_random_state({}) == 42

    def test_explicit_fallback_overrides_default(self) -> None:
        from src.agents.ml_foundation.model_trainer.random_state import (
            resolve_fold_random_state,
        )

        assert resolve_fold_random_state({}, fallback=123) == 123

    def test_fold_random_state_wins_over_state_random_state(self) -> None:
        from src.agents.ml_foundation.model_trainer.random_state import (
            resolve_fold_random_state,
        )

        state = {"fold_random_state": 99, "random_state": 7}
        assert resolve_fold_random_state(state) == 99

    def test_zero_is_a_valid_seed(self) -> None:
        """Per shard 21 §A.3 _derive_seed: seeds may be any non-negative int.
        Treating 0 as 'unset' would corrupt fold 0 of any deterministic chain.
        """
        from src.agents.ml_foundation.model_trainer.random_state import (
            resolve_fold_random_state,
        )

        assert resolve_fold_random_state({"fold_random_state": 0}) == 0
        assert resolve_fold_random_state({"random_state": 0}) == 0


# ---------------------------------------------------------------------------
# split_loader: propagate fold_random_state through to downstream state
# ---------------------------------------------------------------------------


def _make_minimal_split_state(
    n_features: int = 3, n_per_split: int = 8, **extra: Any
) -> Dict[str, Any]:
    """Build a state dict with all four splits already materialized so
    load_splits takes its 'splits already in state' branch (no Feast/db).
    """
    rng = np.random.default_rng(0)

    def _block() -> Dict[str, Any]:
        X = pd.DataFrame(
            rng.standard_normal((n_per_split, n_features)),
            columns=[f"f{i}" for i in range(n_features)],
        )
        y = pd.Series((rng.random(n_per_split) > 0.5).astype(int))
        return {"X": X, "y": y, "row_count": n_per_split}

    state: Dict[str, Any] = {
        "train_data": _block(),
        "validation_data": _block(),
        "test_data": _block(),
        "holdout_data": _block(),
    }
    state.update(extra)
    return state


class TestSplitLoaderAcceptsFoldRandomState:
    def test_load_splits_propagates_fold_random_state(self) -> None:
        """When state carries ``fold_random_state``, the load_splits node
        echoes it in its return dict so downstream nodes can read it
        unambiguously (LangGraph state merge + explicit propagation)."""
        from src.agents.ml_foundation.model_trainer.nodes.split_loader import (
            load_splits,
        )

        state = _make_minimal_split_state(fold_random_state=99)
        out = asyncio.run(load_splits(state))

        assert "error" not in out, out
        assert out.get("fold_random_state") == 99

    def test_load_splits_omits_fold_random_state_when_not_set(self) -> None:
        """Backward-compat: without a per-fold seed the legacy single-split
        callers get the same return shape they always have. The key may be
        absent or explicitly None — assert it is not a non-None integer."""
        from src.agents.ml_foundation.model_trainer.nodes.split_loader import (
            load_splits,
        )

        state = _make_minimal_split_state()
        out = asyncio.run(load_splits(state))

        assert "error" not in out, out
        assert out.get("fold_random_state") is None


# ---------------------------------------------------------------------------
# hyperparameter_tuner._get_fixed_params: optional random_state kwarg
# ---------------------------------------------------------------------------


class TestGetFixedParamsAcceptsRandomState:
    @pytest.mark.parametrize(
        "algorithm_name",
        [
            "XGBoost",
            "LightGBM",
            "RandomForest",
            "LogisticRegression",
            "Ridge",
            "Lasso",
            "CausalForest",
            "LinearDML",
            "DRLearner",
            "SLearner",
            "TLearner",
            "XLearner",
        ],
    )
    def test_random_state_kwarg_overrides_default(self, algorithm_name: str) -> None:
        from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import (
            _get_fixed_params,
        )

        params = _get_fixed_params(algorithm_name, random_state=99)
        assert params.get("random_state") == 99, (algorithm_name, params)

    def test_default_remains_42_for_backward_compat(self) -> None:
        """Existing callers that don't pass random_state must still see 42."""
        from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import (
            _get_fixed_params,
        )

        params = _get_fixed_params("XGBoost")
        assert params.get("random_state") == 42

    def test_zero_seed_propagates(self) -> None:
        from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import (
            _get_fixed_params,
        )

        params = _get_fixed_params("LightGBM", random_state=0)
        assert params.get("random_state") == 0


# ---------------------------------------------------------------------------
# model_trainer_node.train_model: override best_hyperparameters[random_state]
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Sklearn-shape stub that records the ``random_state`` passed to __init__
    and pretends to fit. Lets us assert the value that flows into the model
    constructor without instantiating XGBoost/LightGBM."""

    captured_random_state: Any = None

    def __init__(self, **params: Any) -> None:  # noqa: D401 — stub
        type(self).captured_random_state = params.get("random_state")
        self._params = params

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "_RecordingModel":
        return self


class TestTrainModelHonorsFoldRandomState:
    def test_fold_random_state_overrides_best_hyperparameters_random_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Train flow: when ``state['fold_random_state']`` is set, the value
        passed to the model constructor's ``random_state`` is the fold seed,
        even though ``best_hyperparameters['random_state']`` was set to 42 by
        an earlier (non-fold-aware) HPO pass."""
        from src.agents.ml_foundation.model_trainer.nodes import model_trainer_node

        # Reset capture
        _RecordingModel.captured_random_state = None

        # Force the model class lookup to return our stub regardless of
        # algorithm_name. Patch both the dynamic helper and the optuna_optimizer
        # delegate path so the lookup never hits a real ML library.
        monkeypatch.setattr(
            model_trainer_node,
            "_get_model_class_dynamic",
            lambda algorithm_name, problem_type: _RecordingModel,
        )

        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((16, 3)).astype(np.float64)
        y_train = (rng.random(16) > 0.5).astype(int)
        X_val = rng.standard_normal((8, 3)).astype(np.float64)
        y_val = (rng.random(8) > 0.5).astype(int)

        state: Dict[str, Any] = {
            "algorithm_name": "XGBoost",
            "problem_type": "binary_classification",
            "best_hyperparameters": {"random_state": 42, "n_estimators": 5},
            "X_train_preprocessed": X_train,
            "X_validation_preprocessed": X_val,
            "train_data": {"X": pd.DataFrame(X_train), "y": pd.Series(y_train), "row_count": 16},
            "validation_data": {
                "X": pd.DataFrame(X_val),
                "y": pd.Series(y_val),
                "row_count": 8,
            },
            "feature_columns": ["f0", "f1", "f2"],
            "early_stopping": False,
            "fold_random_state": 99,
        }

        out = asyncio.run(model_trainer_node.train_model(state))

        assert out.get("training_status") == "completed", out
        assert _RecordingModel.captured_random_state == 99, (
            f"expected 99 (fold_random_state), got {_RecordingModel.captured_random_state}; "
            f"best_hyperparameters['random_state']=42 must be overridden by the fold seed"
        )

    def test_no_fold_random_state_preserves_best_hyperparameters_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backward-compat: when no fold_random_state is set, the
        best_hyperparameters value flows through unchanged (legacy path)."""
        from src.agents.ml_foundation.model_trainer.nodes import model_trainer_node

        _RecordingModel.captured_random_state = None

        monkeypatch.setattr(
            model_trainer_node,
            "_get_model_class_dynamic",
            lambda algorithm_name, problem_type: _RecordingModel,
        )

        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((16, 3)).astype(np.float64)
        y_train = (rng.random(16) > 0.5).astype(int)
        X_val = rng.standard_normal((8, 3)).astype(np.float64)
        y_val = (rng.random(8) > 0.5).astype(int)

        state: Dict[str, Any] = {
            "algorithm_name": "XGBoost",
            "problem_type": "binary_classification",
            "best_hyperparameters": {"random_state": 7, "n_estimators": 5},
            "X_train_preprocessed": X_train,
            "X_validation_preprocessed": X_val,
            "train_data": {"X": pd.DataFrame(X_train), "y": pd.Series(y_train), "row_count": 16},
            "validation_data": {
                "X": pd.DataFrame(X_val),
                "y": pd.Series(y_val),
                "row_count": 8,
            },
            "feature_columns": ["f0", "f1", "f2"],
            "early_stopping": False,
            # No fold_random_state.
        }

        out = asyncio.run(model_trainer_node.train_model(state))

        assert out.get("training_status") == "completed", out
        assert _RecordingModel.captured_random_state == 7
