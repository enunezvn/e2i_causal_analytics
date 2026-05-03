"""Phase 1 W1 day-5 integration smoke (shard 17 W1 row Day 5 + shard 20 §J Day 5).

Exercises the full `evaluate_model` node with two configurations:

1. ``success_criteria`` with ``clinical_threshold_range.use_case = "diagnostic"``
   — verifies NB-area + DCA + anchor metrics + Brier decomposition all emit.

2. ``success_criteria = {}`` — verifies the legacy path (only
   ``net_benefit_grid`` emitted; no NB-area / DCA / anchor fields) is
   unchanged.

Acceptance per shard 20 §G #7 + #8 + #11 + #12 + #15.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import evaluate_model
from src.agents.ml_foundation.model_trainer.nodes.model_trainer_node import train_model

SEED = 42
N_TRAIN = 200
N_VAL = 60
N_TEST = 60
N_FEATURES = 5


def _make_state(success_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal trainer state with deterministic synthetic data + LogReg.

    Uses LogisticRegression which is calibration-friendly + fast at
    N=200, so the smoke test runs in seconds. We shape the labels to
    have ~27% prevalence (E[sigmoid(1.5·Z − 0.5)] for Z ~ N(0,1)) so
    the diagnostic τ-range [0.05, 0.30] sits in the NB-area sweet spot.
    Cycle-23 C-2: docstring corrected from "~30%" — the precise
    expectation under standard-normal X is ~0.27.
    """
    rng = np.random.default_rng(SEED)
    # Build a separable signal: y depends on the first feature.
    X_all = rng.standard_normal((N_TRAIN + N_VAL + N_TEST, N_FEATURES))
    logits = 1.5 * X_all[:, 0] - 0.5
    p_all = 1.0 / (1.0 + np.exp(-logits))
    y_all = (rng.uniform(size=len(X_all)) < p_all).astype(int)

    X_train = X_all[:N_TRAIN]
    y_train = y_all[:N_TRAIN]
    X_val = X_all[N_TRAIN : N_TRAIN + N_VAL]
    y_val = y_all[N_TRAIN : N_TRAIN + N_VAL]
    X_test = X_all[N_TRAIN + N_VAL :]
    y_test = y_all[N_TRAIN + N_VAL :]

    state: Dict[str, Any] = {
        "algorithm_name": "LogisticRegression",
        "best_hyperparameters": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 500,
        },
        "problem_type": "binary_classification",
        "X_train_preprocessed": X_train,
        "X_validation_preprocessed": X_val,
        "X_test_preprocessed": X_test,
        "train_data": {"y": y_train},
        "validation_data": {"y": y_val},
        "test_data": {"y": y_test},
        "success_criteria": success_criteria,
        "early_stopping": False,
        "early_stopping_patience": 10,
        "model_candidate": {},
    }
    return state


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_smoke_emits_nb_area_and_dca() -> None:
    """``clinical_threshold_range`` present → NB-area + DCA + anchor + Brier."""
    sc: Dict[str, Any] = {
        "dataset_disease": "breast_cancer_recurrence",
        "clinical_threshold_range": {"use_case": "diagnostic"},
    }
    state = _make_state(sc)

    train_result = await train_model(state)
    assert "error" not in train_result, (
        f"LogReg train failed: {train_result.get('error')}"
    )

    eval_state = {**state, **train_result}
    eval_result = await evaluate_model(eval_state)
    assert "error" not in eval_result, (
        f"LogReg eval failed: {eval_result.get('error')}"
    )
    test_metrics = eval_result["test_metrics"]

    # NB-area block (§G #7).
    assert "net_benefit_area" in test_metrics
    assert "net_benefit_area_treat_all" in test_metrics
    assert "net_benefit_area_relative_to_treat_all" in test_metrics
    assert "net_benefit_area_form" in test_metrics
    # Cycle-23 C-1: also pin the form value range — must be one of the
    # Q-W5-1 RESOLVED literals.
    assert test_metrics["net_benefit_area_form"] in {"raw", "relative_to_treat_all"}
    assert "tau_low" in test_metrics
    assert "tau_high" in test_metrics
    assert "n_grid_points" in test_metrics

    # DCA artifact (§G #7).
    dca = test_metrics["decision_curve_data"]
    assert isinstance(dca, dict)
    assert dca["n_grid_points"] == 21

    # Anchor block (§G #11 + #12).
    assert "primary_tau" in test_metrics
    assert test_metrics["primary_tau"] == pytest.approx(0.21)  # disease default
    assert "nb_anchor_secondary_gate_active" in test_metrics
    assert "nb_anchor_passes" in test_metrics
    assert "nb_anchor_vs_area_disagree" in test_metrics

    # Disease bounds beat use_case=diagnostic → tau_low == 0.10.
    assert test_metrics["tau_low"] == pytest.approx(0.10)
    assert test_metrics["tau_high"] == pytest.approx(0.30)

    # Brier decomposition (§G #8 + #15). The
    # ``compute_calibration_analysis`` return dict is attached at the
    # top-level ``calibration_analysis`` key of the evaluator result.
    cal_block = eval_result.get("calibration_analysis") or {}
    assert "brier_reliability" in cal_block, (
        f"Expected brier_reliability in calibration_analysis; "
        f"keys={sorted(cal_block.keys())}"
    )
    assert "brier_resolution" in cal_block
    assert "brier_uncertainty" in cal_block
    assert "brier_recombined" in cal_block
    assert "brier_decomposition_residual" in cal_block
    # Smoke-grade sign-convention bound. At N_test=60 with default 10-bin
    # calibration, the residual sits in the 1e-3 band; the tighter
    # 1e-4 / 1e-6 bounds are exercised by the deterministic-fixture
    # unit tests in `test_advanced_validation_brier.py`.
    assert cal_block["brier_decomposition_residual"] < 1e-2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_smoke_legacy_path_unchanged_when_no_schema() -> None:
    """No ``clinical_threshold_range`` → legacy ``net_benefit_grid`` only."""
    state = _make_state(success_criteria={})  # NO clinical_threshold_range key

    train_result = await train_model(state)
    assert "error" not in train_result

    eval_state = {**state, **train_result}
    eval_result = await evaluate_model(eval_state)
    assert "error" not in eval_result
    test_metrics = eval_result["test_metrics"]

    # Legacy emit MUST still be present + shape unchanged.
    # Cycle-23 I-1: the §F spec says "same `net_benefit_grid` it does
    # today." Pin both the 6-key shape and key naming so a refactor of
    # `_V3_NB_GRID_P_T_VALUES` cannot silently pass.
    nb_grid = test_metrics["net_benefit_grid"]
    assert isinstance(nb_grid, dict)
    assert len(nb_grid) == 6
    assert set(nb_grid.keys()) == {
        "p_t=0.05", "p_t=0.10", "p_t=0.20", "p_t=0.30", "p_t=0.40", "p_t=0.50",
    }
    for value in nb_grid.values():
        # NB values are floats (or NaN for boundary cases); never None.
        assert isinstance(value, float)

    # NEW Day-1/2 fields MUST be absent in legacy mode.
    assert "net_benefit_area" not in test_metrics
    assert "decision_curve_data" not in test_metrics
    assert "primary_tau" not in test_metrics
    assert "nb_anchor_secondary_gate_active" not in test_metrics
