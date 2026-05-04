"""Phase 1 W2 day-5 integration smoke (shard 17 W2 row Day 5).

Exercises the new W2 algorithms (NGBoost / NGBoost_Conformal / LightGBM_Monotone)
end-to-end through `train_model` → `evaluate_model` on a small synthetic dataset.
The acceptance criterion (per shard 17 + shard 19 §A/§B/§C) is "all 3 paths
complete without crashes; metrics emitted" — this is a smoke check, NOT a
performance benchmark.

Fixture sizing intentionally small (N=200, 5 features) per the W2-prep
memory-pressure codex callout (16 GB / 6 GB-swap droplet); a regression of
the constraint INJECTION + LOOKUP wiring is what we're verifying, not
generalization performance.

LightGBM_Monotone path: per cycle-10 D5a finding, `state["monotone_vector"]`
is not yet emitted by the data_preparer / synthetic_data_generator_v2; the
test exercises BOTH the with-vector path (constraints injected) and the
no-vector path (soft-degrade-with-warning per shard 19 §C.4).

NGBoost_Conformal path: predict_proba delegates to base NGBoost (per
mapie_wrapper amendment 2); the conformal contribution is the predict_sets
artifact (NOT consumed by v3 contract in Phase 1).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import evaluate_model
from src.agents.ml_foundation.model_trainer.nodes.model_trainer_node import train_model

SEED = 42
N_TRAIN = 100
N_VAL = 30
N_TEST = 20
N_FEATURES = 5


def _make_state(
    algorithm_name: str,
    best_hyperparameters: Dict[str, Any],
    model_candidate_meta: Dict[str, Any] | None = None,
    monotone_vector: list | None = None,
) -> Dict[str, Any]:
    """Build a minimal trainer state with deterministic synthetic data."""
    rng = np.random.default_rng(SEED)
    X_train = rng.standard_normal((N_TRAIN, N_FEATURES))
    y_train = rng.integers(0, 2, N_TRAIN)
    X_val = rng.standard_normal((N_VAL, N_FEATURES))
    y_val = rng.integers(0, 2, N_VAL)
    X_test = rng.standard_normal((N_TEST, N_FEATURES))
    y_test = rng.integers(0, 2, N_TEST)

    state: Dict[str, Any] = {
        "algorithm_name": algorithm_name,
        "best_hyperparameters": best_hyperparameters,
        "problem_type": "binary_classification",
        "X_train_preprocessed": X_train,
        "X_validation_preprocessed": X_val,
        "X_test_preprocessed": X_test,
        "train_data": {"y": y_train},
        "validation_data": {"y": y_val},
        "test_data": {"y": y_test},
        "success_criteria": {},
        "early_stopping": False,
        "early_stopping_patience": 10,
        "model_candidate": model_candidate_meta or {},
    }
    if monotone_vector is not None:
        state["monotone_vector"] = monotone_vector
    return state


@pytest.mark.integration
@pytest.mark.asyncio
async def test_w2_smoke_ngboost_end_to_end():
    """NGBoost trains end-to-end + evaluator skips post-hoc isotonic.

    NGBoost is calibration-native: the registry sets
    `skip_post_hoc_calibration=True` (W2 day-1 commit `94190e0`). The
    evaluator gate (W2 day-2 commit `3fb978f`) reads this flag and skips
    isotonic; the calibrated_ece alias is populated from native ECE
    (W2 day-2 follow-up `dc26a8d`).
    """
    state = _make_state(
        algorithm_name="NGBoost",
        best_hyperparameters={
            "n_estimators": 50,
            "learning_rate": 0.05,
            "minibatch_frac": 1.0,
            "col_sample": 1.0,
            "base_max_depth": 3,
            "base_min_samples_leaf": 5,
        },
        model_candidate_meta={
            "skip_post_hoc_calibration": True,
            "distribution_predictor": True,
        },
    )

    # Train
    train_result = await train_model(state)
    assert "error" not in train_result, f"NGBoost train failed: {train_result.get('error')}"
    assert train_result["trained_model"] is not None
    assert train_result["framework"] == "ngboost"

    # Evaluate (need to merge train_result back into state for evaluator)
    eval_state = {**state, **train_result}
    eval_result = await evaluate_model(eval_state)
    assert "error" not in eval_result, f"NGBoost eval failed: {eval_result.get('error')}"
    # Calibration-native skip path:
    assert eval_result["post_hoc_calibration"]["calibration_applied"] is False
    assert eval_result["post_hoc_calibration"]["skip_reason"] == "skip_post_hoc_calibration_flag"
    # Native ECE copied to calibrated_ece alias (cycle-8 fix `dc26a8d`):
    assert "calibrated_ece" in eval_result
    # Standard test_metrics emitted:
    assert "test_metrics" in eval_result
    test_metrics = eval_result["test_metrics"]
    assert "accuracy" in test_metrics


@pytest.mark.integration
@pytest.mark.asyncio
async def test_w2_smoke_ngboost_conformal_end_to_end():
    """NGBoost_Conformal trains end-to-end via the conformal factory.

    The factory at `optuna_optimizer.get_model_class` (W2 day-3 commit
    `e925b3f`) recognizes the `_Conformal` suffix, recurses to fetch
    NGBoost wrapper as the base class, and returns a closure factory that
    instantiates `MapieConformalBinaryClassifier`. The wrapper's
    `predict_proba` delegates to base NGBoost (amendment 2; mapie 0.8.6
    has no native predict_proba on MapieClassifier).
    """
    state = _make_state(
        algorithm_name="NGBoost_Conformal",
        best_hyperparameters={
            "n_estimators": 50,
            "learning_rate": 0.05,
            "method": "lac",
            "alpha": 0.10,
        },
        model_candidate_meta={
            "skip_post_hoc_calibration": True,
            "distribution_predictor": True,
            "conformal_wrapper": True,
        },
    )

    train_result = await train_model(state)
    assert "error" not in train_result, (
        f"NGBoost_Conformal train failed: {train_result.get('error')}"
    )
    assert train_result["trained_model"] is not None
    # The wrapper exposes predict_sets for future-work logging:
    assert hasattr(train_result["trained_model"], "predict_sets")

    eval_state = {**state, **train_result}
    eval_result = await evaluate_model(eval_state)
    assert "error" not in eval_result, f"NGBoost_Conformal eval failed: {eval_result.get('error')}"
    # Conformal also marks skip_post_hoc_calibration:
    assert eval_result["post_hoc_calibration"]["calibration_applied"] is False
    # Cycle-11 codex IMPORTANT fix: assert metrics-emitted half of the
    # acceptance criterion (was missing in the original day-5 commit).
    assert "test_metrics" in eval_result
    assert "accuracy" in eval_result.get("test_metrics", {})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_w2_smoke_lightgbm_monotone_with_vector():
    """LightGBM_Monotone with explicit monotone_vector → constraints injected.

    Tests the full chain: model_candidate.monotone_constraints_required=True
    + state["monotone_vector"] → train_model injects monotone_constraints
    into filtered_params (W2 day-4 commit `792de7c` per shard 19 §C.4) →
    LightGBM trains with the constraints in place.
    """
    state = _make_state(
        algorithm_name="LightGBM_Monotone",
        best_hyperparameters={
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
        model_candidate_meta={
            "monotone_constraints_required": True,
        },
        monotone_vector=[1, 0, -1, 0, 1],  # one int per N_FEATURES=5
    )

    train_result = await train_model(state)
    assert "error" not in train_result, (
        f"LightGBM_Monotone train failed: {train_result.get('error')}"
    )
    model = train_result["trained_model"]
    assert model is not None
    # Constraints reached LightGBM (W2 day-4 §C.4 + cycle-10 fix `33fe3e0` int cast):
    assert model.get_params().get("monotone_constraints") == [1, 0, -1, 0, 1]

    eval_state = {**state, **train_result}
    eval_result = await evaluate_model(eval_state)
    assert "error" not in eval_result, f"LightGBM_Monotone eval failed: {eval_result.get('error')}"
    # Standard isotonic path runs (no skip flag for monotone variants):
    assert (
        eval_result["post_hoc_calibration"].get("skip_reason") != "skip_post_hoc_calibration_flag"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_w2_smoke_lightgbm_monotone_without_vector_soft_degrades():
    """LightGBM_Monotone without monotone_vector → soft-degrade per §C.4.

    Per cycle-10 D5a: state["monotone_vector"] is not yet emitted by the
    data_preparer / synthetic_generator. This is the path the day-5 smoke
    will hit in production until the producer is wired. Acceptance: train
    completes (degraded to unconstrained), warning is logged, no crash.
    """
    state = _make_state(
        algorithm_name="LightGBM_Monotone",
        best_hyperparameters={
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
        model_candidate_meta={
            "monotone_constraints_required": True,
        },
        monotone_vector=None,  # explicit absence
    )

    train_result = await train_model(state)
    assert "error" not in train_result, (
        f"LightGBM_Monotone soft-degrade train failed: {train_result.get('error')}"
    )
    model = train_result["trained_model"]
    assert model is not None
    # No constraints applied (degraded to unconstrained):
    params = model.get_params()
    assert params.get("monotone_constraints") in (None, "None")
