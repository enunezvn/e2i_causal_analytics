"""#2248: the retrain path never engaged the adaptive success criteria.

``ADAPTIVE_CRITERIA`` defaults ON (owner decision 2026-06-02), but the adaptive scheme
needs three pre-eval inputs (``n_samples`` / ``prevalence`` / ``feature_count``) that
scope_definer cannot know — it runs before any data is loaded. ``scripts/run_tier0_test.py``
injects them itself; ``MLFoundationPipeline.run`` (the API / live-retrain origin) never did,
so every retrain fell back to the fixed Apr-26 thresholds (precision >= 0.70 / F1 >= 0.70)
and the owner-run retrain of ``initiation_kisqali_goldstd_lr_v1`` (job c4252b47,
2026-09-23) was refused on precision 0.627 / F1 0.688.

The pipeline now derives the inputs from the splits it hands the trainer (the frames the
model is actually trained and evaluated on) and stashes them where the validator would
have — ``success_criteria['_adaptive_inputs']`` — so the evaluator overlay applies the
adaptive thresholds at eval time.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _apply_adaptive_criteria_overlay,
    _check_success_criteria,
)
from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
    define_success_criteria,
)
from src.agents.tier_0.pipeline import (
    MLFoundationPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)
from src.agents.tier_0.split_handoff import (
    adaptive_inputs_from_splits,
    frames_to_trainer_splits,
)

TARGET = "treatment_initiated"


def _frame(labels: list, start: int) -> pd.DataFrame:
    n = len(labels)
    return pd.DataFrame(
        {
            "patient_id": [f"p{start + i}" for i in range(n)],
            "insurance_type": ["commercial", "medicare"] * (n // 2) + ["medicaid"] * (n % 2),
            "age_at_diagnosis": [50.0 + i for i in range(n)],
            "disease_severity": [float(i % 3) for i in range(n)],
            TARGET: labels,
        }
    )


# 20 rows in total, 7 positives -> prevalence 0.35; three features reach X.
FRAMES = {
    "train": _frame([1, 0, 0, 1, 0, 0, 1, 0], 0),
    "validation": _frame([1, 0, 0, 0, 1], 100),
    "test": _frame([0, 1, 0, 0], 200),
    "holdout": _frame([1, 0, 0], 300),
}


def _splits() -> Dict[str, Dict[str, Any]]:
    return frames_to_trainer_splits(FRAMES, TARGET, drop_columns=("patient_id",))


# ---------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inputs_are_measured_on_every_split_the_trainer_receives():
    inputs = adaptive_inputs_from_splits(_splits())
    assert inputs == {"n_samples": 20, "prevalence": pytest.approx(7 / 20), "feature_count": 3}
    assert isinstance(inputs["n_samples"], int) and isinstance(inputs["feature_count"], int)


@pytest.mark.unit
def test_an_empty_holdout_contributes_no_rows():
    frames = dict(FRAMES, holdout=None)
    inputs = adaptive_inputs_from_splits(
        frames_to_trainer_splits(frames, TARGET, drop_columns=("patient_id",))
    )
    assert inputs == {"n_samples": 17, "prevalence": pytest.approx(6 / 17), "feature_count": 3}


@pytest.mark.unit
def test_boolean_labels_count_as_binary():
    frames = {k: v.assign(**{TARGET: v[TARGET].astype(bool)}) for k, v in FRAMES.items()}
    inputs = adaptive_inputs_from_splits(
        frames_to_trainer_splits(frames, TARGET, drop_columns=("patient_id",))
    )
    assert inputs is not None and inputs["prevalence"] == pytest.approx(7 / 20)


@pytest.mark.unit
@pytest.mark.parametrize(
    "labels",
    [
        ["yes", "no", "no", "yes"],  # not 0/1 — a prevalence would be a guess
        [0.0, 0.5, 1.0, 0.25],  # continuous target
        [0, 0, 0, 0],  # one class: no prevalence to calibrate against
        [0, None, 1, 0],  # missing labels
    ],
)
def test_labels_that_are_not_a_clean_binary_target_yield_no_inputs(labels):
    frames = {k: _frame(labels, i * 100) for i, k in enumerate(FRAMES)}
    splits = frames_to_trainer_splits(frames, TARGET, drop_columns=("patient_id",))
    assert adaptive_inputs_from_splits(splits) is None


@pytest.mark.unit
def test_splits_without_frames_yield_no_inputs():
    assert adaptive_inputs_from_splits({"train_data": None, "validation_data": {}}) is None


# ---------------------------------------------------------------------------
# the pipeline stage (scope criteria are the REAL scope_definer node's output)
# ---------------------------------------------------------------------------


async def _scope_criteria(monkeypatch, flag: str) -> Dict[str, Any]:
    """What scope_definer returns on the retrain path: no pre-eval inputs in state."""
    monkeypatch.setenv("ADAPTIVE_CRITERIA", flag)
    out = await define_success_criteria(
        {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
        }
    )
    return out["success_criteria"]


def _result(success_criteria: Dict[str, Any]) -> PipelineResult:
    r = PipelineResult(
        pipeline_run_id="run", status="running", current_stage=PipelineStage.DATA_PREPARATION
    )
    r.experiment_id = "exp-2248"
    r.scope_spec = {
        "problem_type": "binary_classification",
        "prediction_target": TARGET,
        "entity_column": "patient_id",
    }
    r.success_criteria = success_criteria
    r.model_candidate = {"algorithm_name": "LogisticRegression"}
    r.prepared_frames = dict(FRAMES)
    return r


async def _trainer_criteria(result: PipelineResult) -> Dict[str, Any]:
    pipeline = MLFoundationPipeline(config=PipelineConfig(skip_mlflow=True, enable_hpo=False))
    captured: Dict[str, Any] = {}
    fake_trainer = MagicMock()

    async def _run(trainer_input):
        captured.update(trainer_input)
        return {"validation_metrics": {}, "success_criteria_met": False}

    fake_trainer.run = AsyncMock(side_effect=_run)
    with patch.object(pipeline, "_get_agent", return_value=fake_trainer):
        await pipeline._run_model_training(
            input_data={"data_source": "patient_journeys", "target_outcome": TARGET},
            result=result,
            obs_context=None,
        )
    return captured["success_criteria"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrain_hands_the_trainer_adaptive_inputs_measured_on_its_splits(monkeypatch):
    scope_criteria = await _scope_criteria(monkeypatch, "true")
    assert scope_criteria["criteria_source"] == "adaptive_fallback_to_fixed"  # the #2248 state
    assert "_adaptive_inputs" not in scope_criteria

    criteria = await _trainer_criteria(_result(scope_criteria))

    assert criteria["_adaptive_inputs"] == {
        "n_samples": 20,
        "prevalence": pytest.approx(7 / 20),
        "feature_count": 3,
        "regime": None,
        "deployment_intent": "clinical",
    }
    assert criteria["criteria_source"] == "adaptive"
    # the scope stage's own dict is not rewritten in place
    assert "_adaptive_inputs" not in scope_criteria
    assert scope_criteria["criteria_source"] == "adaptive_fallback_to_fixed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_scope_stamped_deployment_intent_is_carried_not_reset(monkeypatch):
    scope_criteria = await _scope_criteria(monkeypatch, "true")
    scope_criteria["deployment_intent"] = "commercial"
    criteria = await _trainer_criteria(_result(scope_criteria))
    assert criteria["_adaptive_inputs"]["deployment_intent"] == "commercial"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_fixed_scheme_opt_out_is_respected(monkeypatch):
    scope_criteria = await _scope_criteria(monkeypatch, "false")
    assert scope_criteria["criteria_source"] == "fixed"
    criteria = await _trainer_criteria(_result(scope_criteria))
    assert "_adaptive_inputs" not in criteria
    assert criteria["criteria_source"] == "fixed"
    assert criteria["minimum_precision"] == 0.70


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inputs_the_caller_already_supplied_are_not_overwritten(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_CRITERIA", "true")
    out = await define_success_criteria(
        {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "n_samples": 5000,
            "prevalence": 0.2,
            "feature_count": 40,
        }
    )
    stashed = dict(out["success_criteria"]["_adaptive_inputs"])
    criteria = await _trainer_criteria(_result(out["success_criteria"]))
    assert criteria["_adaptive_inputs"] == stashed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_non_binary_target_keeps_the_fixed_fallback(monkeypatch):
    scope_criteria = await _scope_criteria(monkeypatch, "true")
    result = _result(scope_criteria)
    result.prepared_frames = {
        k: v.assign(**{TARGET: ["yes", "no"] * (len(v) // 2) + ["no"] * (len(v) % 2)})
        for k, v in FRAMES.items()
    }
    criteria = await _trainer_criteria(result)
    assert "_adaptive_inputs" not in criteria
    assert criteria["criteria_source"] == "adaptive_fallback_to_fixed"


# ---------------------------------------------------------------------------
# end to end through the evaluator's real overlay + gate check, on the metrics the
# owner-run retrain actually measured (episodic_memories, model_trainer, 05:44Z
# 2026-09-23; split sizes from the worker log: 5294/1791/869/884, X = 15 columns)
# ---------------------------------------------------------------------------

JOB_C4252B47_TEST_METRICS: Dict[str, Any] = {
    "roc_auc": 0.8352780405760538,
    "precision": 0.6267029972752044,
    "recall": 0.7615894039735099,
    "f1_score": 0.6875934230194319,
    "mcc": 0.5012862002023821,
    "calibrated_ece": 0.0161936089035093,
    "calibration_slope_deviation": 0.30959412032497513,
    "calibration_intercept_magnitude": 0.15403918004334052,
    "baseline_test_auc": 0.5039711739490988,
    "minimum_lift_over_baseline": 0.33130686662695497,
    "train_val_auc_delta": 0.010009423627472924,
    "net_benefit_grid": {
        "p_t=0.05": 0.3179092726061413,
        "p_t=0.10": 0.28794271832246515,
        "p_t=0.20": 0.24050632911392406,
        "p_t=0.30": 0.19710669077757687,
        "p_t=0.40": 0.1595703874184887,
        "p_t=0.50": 0.12082853855005754,
    },
}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_job_c4252b47_is_judged_by_the_adaptive_gates_not_precision_and_f1(monkeypatch):
    scope_criteria = await _scope_criteria(monkeypatch, "true")
    fixed = _check_success_criteria(
        dict(JOB_C4252B47_TEST_METRICS), scope_criteria, "binary_classification"
    )
    assert fixed["success_criteria_results"]["minimum_precision"] is False  # what the job saw
    assert fixed["success_criteria_results"]["minimum_f1"] is False

    stashed = dict(scope_criteria)
    stashed["_adaptive_inputs"] = {
        "n_samples": 8838,
        "prevalence": 0.3475,
        "feature_count": 15,
        "regime": None,
        "deployment_intent": "clinical",
    }
    stashed["criteria_source"] = "adaptive"
    overlaid = _apply_adaptive_criteria_overlay(
        stashed, JOB_C4252B47_TEST_METRICS, n_train=5294, n_val=1791, n_test=869
    )
    results = _check_success_criteria(
        dict(JOB_C4252B47_TEST_METRICS), overlaid, "binary_classification"
    )["success_criteria_results"]

    assert "minimum_precision" not in results and "minimum_f1" not in results
    for gate in (
        "minimum_auc",
        "minimum_recall",
        "minimum_mcc",
        "minimum_net_benefit_at_p_t",
        "maximum_calibration_error",
        "maximum_calibration_intercept_magnitude",
        "maximum_train_val_delta",
        "minimum_lift_over_baseline",
    ):
        assert results[gate] is True, gate
    # The one gate the deployed (isotonic-calibrated) model still fails — a finding
    # about the calibrator, reported on #2248, not a threshold this change touches.
    assert results["maximum_calibration_slope_deviation"] is False
