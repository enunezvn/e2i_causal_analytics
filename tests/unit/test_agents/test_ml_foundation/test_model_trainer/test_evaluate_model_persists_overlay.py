"""Regression tests for the post-PR-30 wiring bug.

The v3 adaptive overlay in ``_apply_adaptive_criteria_overlay`` correctly
mutates the success_criteria dict, but those mutations are local-only.
They must propagate through 2 hops at unit-test level:

- Hop 1: ``evaluate_model`` must apply the overlay and include the
  overlaid dict in its return so it persists into LangGraph node state.
- Hop 2: ``ModelTrainerAgent.run`` must extract ``success_criteria`` from
  the LangGraph ``final_state`` and include it in the agent's output dict.

Without these, the runner's JSON artifact captures the validator's
pre-overlay stash, breaking downstream deployer signal + audit trail.
PR #30's unit tests passed because they exercised
``_apply_adaptive_criteria_overlay`` in isolation, never traversing the
chain that the integration suite exposed.

Hops 3 and 4 are covered by ``tests/integration/test_adaptive_criteria_e2e.py``
(runner JSON artifact) and the hop-4 unit tests in
``tests/unit/test_agents/test_tier_0/test_pipeline.py`` respectively.
"""

import asyncio
import math
from typing import Any, Dict

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import evaluate_model

from .conftest import (
    N_FEATURES,
    N_TEST_SAMPLES,
    N_TRAIN_SAMPLES,
    N_VAL_SAMPLES,
    RANDOM_STATE,
    MockBinaryClassifier,
)

# ---------------------------------------------------------------------------
# Hop 1 — evaluate_model return
# ---------------------------------------------------------------------------


def _build_stash_success_criteria(regime: str = "clean") -> Dict[str, Any]:
    """Validator-emitted stash dict (criteria_validator.py:316-332)."""
    return {
        "minimum_auc": 0.75,
        "minimum_precision": 0.70,
        "minimum_recall": 0.65,
        "minimum_f1": 0.70,
        "minimum_lift_over_baseline": 0.10,
        "experiment_id": "exp_test",
        "baseline_model": "stratified_dummy",
        "criteria_source": "adaptive",
        "_adaptive_inputs": {
            "n_samples": 900,
            "prevalence": 0.50,
            "feature_count": 8,
            "regime": regime,
        },
    }


def _build_evaluation_state(success_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """State shaped like what the LangGraph evaluate_model node receives.

    Uses the same MockBinaryClassifier and shapes as ``conftest.py``'s
    ``binary_classification_state`` fixture so existing test infrastructure
    is reused. Seed is fixed for reproducibility.
    """
    np.random.seed(RANDOM_STATE)
    return {
        "trained_model": MockBinaryClassifier(),
        "problem_type": "binary_classification",
        "X_train_preprocessed": np.random.rand(N_TRAIN_SAMPLES, N_FEATURES),
        "X_validation_preprocessed": np.random.rand(N_VAL_SAMPLES, N_FEATURES),
        "X_test_preprocessed": np.random.rand(N_TEST_SAMPLES, N_FEATURES),
        "train_data": {"y": np.random.randint(0, 2, N_TRAIN_SAMPLES)},
        "validation_data": {"y": np.random.randint(0, 2, N_VAL_SAMPLES)},
        "test_data": {"y": np.random.randint(0, 2, N_TEST_SAMPLES)},
        "success_criteria": success_criteria,
    }


def test_evaluate_model_returns_overlaid_success_criteria() -> None:
    """The returned dict must contain ``success_criteria`` reflecting
    the v3 overlay: regime-keyed AUC override, deprecated keys popped,
    v3 active gates inserted.

    FAILS on current main: ``evaluate_model`` returns
    ``{**metrics_result, **success_results, **suspicion_result}`` without
    a ``success_criteria`` key (evaluator.py:374-378). The local
    overlay rebinding inside ``_check_success_criteria`` (line 1686)
    is dropped on function exit.
    """
    state = _build_evaluation_state(_build_stash_success_criteria(regime="clean"))

    result = asyncio.run(evaluate_model(state))

    assert "success_criteria" in result, (
        "evaluate_model dropped success_criteria from its return — the "
        "overlaid v3 dict cannot reach state. This is the wiring bug "
        "from PR #30 that the E2E suite caught."
    )

    sc = result["success_criteria"]

    # v3-deprecated keys popped by overlay.
    assert "minimum_precision" not in sc
    assert "minimum_f1" not in sc

    # v3 active gates inserted by overlay.
    assert sc["minimum_net_benefit_at_p_t"] == pytest.approx(0.0, abs=1e-6)
    assert sc["minimum_mcc"] == pytest.approx(0.45, abs=1e-6)
    # Issue #866: evaluate_model threads the materialized split sizes into
    # the overlay, so the calibration caps reflect the TEST-split noise floor
    # (n_test=20 here → sqrt(1000/20) scale), not the fixed 0.15/0.30 floors.
    _cal_scale = math.sqrt(1000 / N_TEST_SAMPLES)
    assert sc["maximum_calibration_slope_deviation"] == pytest.approx(0.15 * _cal_scale, abs=1e-9)
    assert sc["maximum_calibration_intercept_magnitude"] == pytest.approx(
        0.30 * _cal_scale, abs=1e-9
    )

    # Regime-keyed AUC override (clean: max(0.75, baseline_auc + 0.20)).
    assert sc["minimum_auc"] >= 0.75


def test_evaluate_model_fixed_mode_passes_criteria_through() -> None:
    """Flag-OFF / no _adaptive_inputs: evaluate_model returns the input
    success_criteria unchanged (overlay no-op).

    FAILS on current main: same root cause — the return dict has no
    ``success_criteria`` key, so the overlay's no-op contract for fixed
    mode also cannot be observed by callers.
    """
    sc_fixed = {
        "minimum_auc": 0.75,
        "minimum_precision": 0.70,
        "minimum_recall": 0.65,
        "minimum_f1": 0.70,
        "minimum_lift_over_baseline": 0.10,
        "experiment_id": "exp_test",
        "baseline_model": "stratified_dummy",
        "criteria_source": "fixed",
    }
    state = _build_evaluation_state(sc_fixed)

    result = asyncio.run(evaluate_model(state))

    assert "success_criteria" in result
    out_sc = result["success_criteria"]
    # Fixed mode: dict pass-through with the original keys preserved.
    assert "minimum_auc" in out_sc
    assert "minimum_precision" in out_sc
    assert "minimum_f1" in out_sc
    assert out_sc["criteria_source"] == "fixed"
    # No v3 active gates inserted.
    assert "minimum_net_benefit_at_p_t" not in out_sc
    assert "minimum_mcc" not in out_sc


def test_evaluate_model_threads_split_sizes_into_overlay() -> None:
    """Issue #866: the overlay must receive the materialized train/val/test
    split sizes so the overfit cap scales with the splits the train→val AUC
    delta is actually measured on (n_train=100, n_val=30 here — far noisier
    than the stashed full-frame n_samples=900 implies).

    FAILS before the fix: the overlay recomputes from the stash alone, so
    ``maximum_train_val_delta`` stays at the fixed 0.03 floor.
    """
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        _SE_AUC_ANCHOR,
        _hanley_mcneil_se_auc,
    )

    state = _build_evaluation_state(_build_stash_success_criteria(regime="clean"))
    result = asyncio.run(evaluate_model(state))

    sc = result["success_criteria"]
    expected_cap = max(
        0.03,
        2.0
        * math.hypot(
            _hanley_mcneil_se_auc(_SE_AUC_ANCHOR, N_VAL_SAMPLES, 0.50),
            _hanley_mcneil_se_auc(_SE_AUC_ANCHOR, N_TRAIN_SAMPLES, 0.50),
        ),
    )
    assert sc["maximum_train_val_delta"] == pytest.approx(expected_cap, abs=1e-9)
    assert sc["maximum_train_val_delta"] > 0.03  # tiny splits ⇒ cap widened


# ---------------------------------------------------------------------------
# Hop 2 — agent extraction & return
# ---------------------------------------------------------------------------


def _minimal_trainer_input() -> Dict[str, Any]:
    """Smallest input_data dict that ``ModelTrainerAgent.run`` accepts.

    Required fields per ``agent.py:136-155``: model_candidate (with 4 sub-
    fields), qc_report, experiment_id. Other fields default safely.
    """
    return {
        "model_candidate": {
            "algorithm_name": "logistic_regression",
            "algorithm_class": "sklearn.linear_model.LogisticRegression",
            "hyperparameter_search_space": {},
            "default_hyperparameters": {},
        },
        "qc_report": {"qc_passed": True},
        "experiment_id": "exp_test",
        "success_criteria": {"minimum_auc": 0.75, "criteria_source": "adaptive"},
        "enable_hpo": False,
        "enable_mlflow": False,
        "enable_checkpointing": False,
    }


def _build_minimal_final_state(success_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Synthetic LangGraph final_state containing the overlaid criteria.

    Only fields that the agent extracts via ``final_state.get(...)`` need
    to be present; all gets have safe defaults so we set the bare minimum.
    """
    return {
        "training_run_id": "train_test",
        "model_id": "model_test",
        "trained_model": object(),
        "train_metrics": {},
        "validation_metrics": {},
        "test_metrics": {"baseline_test_auc": 0.50},
        "success_criteria_met": True,
        "success_criteria_results": {"minimum_auc": True},
        "success_criteria": success_criteria,  # ← Hop 1 contract: overlaid dict
        "mlflow_status": "not_logged",
    }


def test_agent_run_propagates_success_criteria_to_output(monkeypatch) -> None:
    """Regression test for hop 2 of the propagation chain.

    ``ModelTrainerAgent.run`` must extract ``success_criteria`` from the
    LangGraph compiled graph's ``final_state`` and include it in the
    output dict so the runner can copy it into ``state["success_criteria"]``.
    Without this, even a correctly-overlaid LangGraph node state never
    reaches the runner's JSON artifact.

    FAILS on current main: ``agent.py:287-288`` extracts only
    ``success_criteria_met`` and ``success_criteria_results`` from
    ``final_state``; the output dict at ``agent.py:407-409`` has no
    ``success_criteria`` key.
    """
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    expected_sc = {
        "minimum_auc": 0.95,  # regime override applied
        "minimum_recall": 0.65,
        "minimum_lift_over_baseline": 0.10,
        "minimum_net_benefit_at_p_t": 0.0,  # v3 active gate
        "minimum_mcc": 0.45,  # v3 active gate
        "maximum_calibration_slope_deviation": 0.15,
        "maximum_calibration_intercept_magnitude": 0.30,
        "criteria_source": "adaptive",
        # ``minimum_precision`` and ``minimum_f1`` correctly absent (popped by overlay).
    }

    fake_final_state = _build_minimal_final_state(success_criteria=expected_sc)

    agent = ModelTrainerAgent()

    async def _fake_ainvoke(initial_state, **kwargs):
        return fake_final_state

    # NOTE: the attribute is ``agent.graph`` (verified at agent.py:97;
    # ``self.graph = create_model_trainer_graph()``), NOT
    # ``agent.compiled_graph`` — that attribute does not exist.
    monkeypatch.setattr(agent.graph, "ainvoke", _fake_ainvoke)

    result = asyncio.run(agent.run(_minimal_trainer_input()))

    assert "success_criteria" in result, (
        "ModelTrainerAgent.run dropped success_criteria from its output dict — "
        'the overlaid v3 dict cannot reach the runner\'s state["success_criteria"].'
    )
    assert result["success_criteria"] == expected_sc
    # Sanity: the two derived fields still flow through.
    assert "success_criteria_met" in result
    assert "success_criteria_results" in result
