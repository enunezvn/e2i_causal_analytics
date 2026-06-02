"""#633 — deploy the post-hoc calibrated model (TDD red-first).

The tier0 model_trainer historically DEPLOYS the raw ``trained_model``
(MLflow-logged, checkpointed, returned) while building a post-hoc
``calibrated_model`` used ONLY for diagnostics. #633's clean regime has an
under-confident AUC-champion whose RAW probabilities fail the v3 calibration
slope gate. The chosen fix: deploy the calibrated model when calibration was
actually applied, and judge ALL v3 calibration gates on the DEPLOYED model's
probabilities so the gate-prob source matches the deployed artifact.

These tests pin the new contract:

1. When post-hoc calibration is APPLIED, ``evaluate_model`` returns a
   ``deployed_model`` that is the calibrated estimator (NOT the raw one) and
   sets ``calibration_applied=True``.
2. When calibration is SKIPPED (skip flag / non-classifier / no val data),
   the deployed model stays the raw ``trained_model`` and
   ``calibration_applied`` is False — no regression of those paths.
3. The v3 slope/intercept gates are computed on the DEPLOYED model's test
   probabilities (calibrated when applied), so a well-calibrated calibrated
   model can clear a slope gate the raw model fails.
4. ``mlflow_logger`` logs the deployed model; ``checkpointer`` checkpoints
   the deployed model; ``ModelTrainerAgent.run`` returns the deployed model
   as ``trained_model`` — falling back to raw when ``deployed_model`` absent.
5. The calibrated estimator is joblib-serializable (deployable artifact).
"""

from __future__ import annotations

import asyncio
import io
from typing import Any, Dict

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

from src.agents.ml_foundation.model_trainer.nodes.evaluator import evaluate_model

SEED = 42


# ---------------------------------------------------------------------------
# Helpers: a REAL sklearn classifier so is_classifier()==True and post-hoc
# calibration actually runs (the conftest MockBinaryClassifier is not an
# sklearn classifier → calibration is skipped, which we cover separately).
# ---------------------------------------------------------------------------


def _under_confident_tree(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
    """A shallow tree on a learnable signal — discriminates but its raw
    leaf-frequency probabilities tend to be miscalibrated, the #633 shape."""
    return DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(X, y)


def _make_signal_data(n_train: int = 240, n_val: int = 120, n_test: int = 120, n_features: int = 5):
    rng = np.random.default_rng(SEED)

    def _xy(n: int):
        X = rng.normal(size=(n, n_features))
        logit = 1.3 * X[:, 0] + 0.8 * X[:, 1] - 0.5 * X[:, 2]
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.uniform(size=n) < p).astype(int)
        # guarantee both classes present with >= 30 each side for the
        # slope/intercept NaN guard (n_pos>=30 and n_neg>=30 on test).
        if y.sum() < 35:
            y[:35] = 1
        if (1 - y).sum() < 35:
            y[-35:] = 0
        return X, y

    return _xy(n_train), _xy(n_val), _xy(n_test)


def _build_state(
    *,
    model: Any,
    skip_calibration: bool = False,
    with_val: bool = True,
) -> Dict[str, Any]:
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _make_signal_data()
    state: Dict[str, Any] = {
        "trained_model": model,
        "problem_type": "binary_classification",
        "X_train_preprocessed": Xtr,
        "X_validation_preprocessed": Xva if with_val else None,
        "X_test_preprocessed": Xte,
        "train_data": {"y": ytr},
        "validation_data": {"y": yva if with_val else None},
        "test_data": {"y": yte},
        "success_criteria": {"minimum_auc": 0.5, "criteria_source": "fixed"},
        "calibration_method": "sigmoid",  # deterministic, low-n safe
    }
    if skip_calibration:
        state["model_candidate"] = {"skip_post_hoc_calibration": True}
    return state


# ---------------------------------------------------------------------------
# 1 + 3. Calibration APPLIED → deployed_model is calibrated; gate uses it.
# ---------------------------------------------------------------------------


def test_deployed_model_is_calibrated_when_calibration_applied() -> None:
    (Xtr, ytr), _, _ = _make_signal_data()
    model = _under_confident_tree(Xtr, ytr)
    state = _build_state(model=model)

    result = asyncio.run(evaluate_model(state))

    assert result.get("post_hoc_calibration", {}).get("calibration_applied") is True, (
        "test precondition: post-hoc calibration must actually run for this case"
    )
    assert "deployed_model" in result, (
        "evaluate_model must surface a deployed_model state key so downstream "
        "log/checkpoint/return persist the deployed artifact (#633)."
    )
    assert result["calibration_applied"] is True
    deployed = result["deployed_model"]
    assert isinstance(deployed, CalibratedClassifierCV), (
        "when calibration applied, the deployed model MUST be the calibrated "
        "estimator, not the raw trained_model (#633)."
    )
    assert deployed is not model


def test_v3_slope_gate_uses_deployed_calibrated_probs() -> None:
    """The emitted calibration_slope_deviation must match the DEPLOYED
    (calibrated) test probabilities, not the raw model's. We assert the
    deviation equals the slope-deviation recomputed on the calibrated probs."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _compute_calibration_slope_intercept,
        _positive_class_proba,
    )

    (Xtr, ytr), _, (Xte, yte) = _make_signal_data()
    model = _under_confident_tree(Xtr, ytr)
    state = _build_state(model=model)

    result = asyncio.run(evaluate_model(state))
    assert result["calibration_applied"] is True

    deployed = result["deployed_model"]
    cal_proba_pos = _positive_class_proba(deployed.predict_proba(Xte))
    slope_cal, intercept_cal = _compute_calibration_slope_intercept(np.asarray(yte), cal_proba_pos)

    test_metrics = result["test_metrics"]
    emitted_dev = test_metrics["calibration_slope_deviation"]
    emitted_mag = test_metrics["calibration_intercept_magnitude"]
    assert emitted_dev == pytest.approx(abs(slope_cal - 1.0), abs=1e-9), (
        "v3 slope-deviation gate must read the DEPLOYED (calibrated) probs."
    )
    assert emitted_mag == pytest.approx(abs(intercept_cal), abs=1e-9), (
        "v3 intercept-magnitude gate must read the DEPLOYED (calibrated) probs."
    )


def test_deployed_calibrated_model_is_joblib_serializable() -> None:
    (Xtr, ytr), _, (Xte, _) = _make_signal_data()
    model = _under_confident_tree(Xtr, ytr)
    state = _build_state(model=model)

    result = asyncio.run(evaluate_model(state))
    deployed = result["deployed_model"]

    buf = io.BytesIO()
    joblib.dump(deployed, buf)
    buf.seek(0)
    reloaded = joblib.load(buf)
    assert hasattr(reloaded, "predict")
    assert hasattr(reloaded, "predict_proba")
    assert reloaded.predict_proba(Xte[:3]).shape == (3, 2)


# ---------------------------------------------------------------------------
# 2. Calibration SKIPPED → deployed model stays raw (no regression).
# ---------------------------------------------------------------------------


def test_deployed_model_is_raw_when_calibration_skipped_by_flag() -> None:
    (Xtr, ytr), _, _ = _make_signal_data()
    model = _under_confident_tree(Xtr, ytr)
    state = _build_state(model=model, skip_calibration=True)

    result = asyncio.run(evaluate_model(state))

    assert result.get("post_hoc_calibration", {}).get("calibration_applied") is False
    assert result["calibration_applied"] is False
    assert result["deployed_model"] is model, (
        "skip_post_hoc_calibration → deployed model MUST remain the raw "
        "trained_model (calibration-native algos already pre-calibrated)."
    )


def test_deployed_model_is_raw_when_no_validation_data() -> None:
    (Xtr, ytr), _, _ = _make_signal_data()
    model = _under_confident_tree(Xtr, ytr)
    state = _build_state(model=model, with_val=False)

    result = asyncio.run(evaluate_model(state))

    # No val data → calibration block does not run → raw stays deployed.
    assert result["calibration_applied"] is False
    assert result["deployed_model"] is model


# ---------------------------------------------------------------------------
# 4. Downstream persistence reads deployed_model (mlflow / checkpoint / agent).
# ---------------------------------------------------------------------------


def test_checkpointer_saves_deployed_model_when_present(monkeypatch, tmp_path) -> None:
    from src.agents.ml_foundation.model_trainer.nodes import checkpointer as ckpt_mod

    raw = object()
    deployed = object()
    captured: Dict[str, Any] = {}

    def _fake_save_model(model, path, framework):
        captured["model"] = model
        return "deadbeef"

    monkeypatch.setattr(ckpt_mod, "_save_model", _fake_save_model)
    monkeypatch.setattr(ckpt_mod, "_prepare_metadata", lambda *a, **k: {})
    monkeypatch.setattr(ckpt_mod, "_save_metadata", lambda *a, **k: None)

    state = {
        "trained_model": raw,
        "deployed_model": deployed,
        "calibration_applied": True,
        "experiment_id": "exp",
        "algorithm_name": "decision_tree",
        "checkpoint_dir": str(tmp_path),
    }
    out = asyncio.run(ckpt_mod.save_checkpoint(state))
    assert out["checkpoint_status"] == "success"
    assert captured["model"] is deployed, (
        "checkpointer must persist the deployed (calibrated) model, not raw."
    )


def test_checkpointer_falls_back_to_raw_when_no_deployed(monkeypatch, tmp_path) -> None:
    from src.agents.ml_foundation.model_trainer.nodes import checkpointer as ckpt_mod

    raw = object()
    captured: Dict[str, Any] = {}
    monkeypatch.setattr(ckpt_mod, "_save_model", lambda m, p, f: captured.update(model=m) or "h")
    monkeypatch.setattr(ckpt_mod, "_prepare_metadata", lambda *a, **k: {})
    monkeypatch.setattr(ckpt_mod, "_save_metadata", lambda *a, **k: None)

    state = {
        "trained_model": raw,
        "experiment_id": "exp",
        "algorithm_name": "decision_tree",
        "checkpoint_dir": str(tmp_path),
    }
    out = asyncio.run(ckpt_mod.save_checkpoint(state))
    assert out["checkpoint_status"] == "success"
    assert captured["model"] is raw


def test_agent_run_returns_deployed_model(monkeypatch) -> None:
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    raw = object()
    deployed = object()
    final_state = {
        "training_run_id": "t",
        "model_id": "m",
        "trained_model": raw,
        "deployed_model": deployed,
        "calibration_applied": True,
        "train_metrics": {},
        "validation_metrics": {},
        "test_metrics": {"baseline_test_auc": 0.5},
        "success_criteria_met": True,
        "success_criteria_results": {"minimum_auc": True},
        "success_criteria": {"minimum_auc": 0.75},
        "mlflow_status": "not_logged",
    }

    agent = ModelTrainerAgent()

    async def _fake_ainvoke(initial_state, **kwargs):
        return final_state

    monkeypatch.setattr(agent.graph, "ainvoke", _fake_ainvoke)

    input_data = {
        "model_candidate": {
            "algorithm_name": "decision_tree",
            "algorithm_class": "sklearn.tree.DecisionTreeClassifier",
            "hyperparameter_search_space": {},
            "default_hyperparameters": {},
        },
        "qc_report": {"qc_passed": True},
        "experiment_id": "exp",
        "enable_hpo": False,
        "enable_mlflow": False,
        "enable_checkpointing": False,
    }
    result = asyncio.run(agent.run(input_data))
    assert result["trained_model"] is deployed, (
        "agent output's trained_model must be the deployed (calibrated) model "
        "when calibration applied (#633)."
    )
