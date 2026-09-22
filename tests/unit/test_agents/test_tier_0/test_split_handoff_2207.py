"""#2207 follow-up (codex r1 HIGH-1): MLFoundationPipeline hands data_preparer's frames to
the trainer in its ``{X, y, row_count}`` split contract.

The pipeline used to pass ``input_data.get("train_data")`` — the caller's pre-loaded
splits — and nothing else; on the retraining path (``execute_model_retraining`` ->
``run``) nothing pre-loads splits, so ``ModelTrainerAgent`` received four empty dicts
and ``split_loader`` failed on the first missing ``X`` before any training. The tier-0
harness never saw it because it builds the splits itself.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.agents.tier_0.pipeline import (
    MLFoundationPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)
from src.agents.tier_0.split_handoff import (
    SPLIT_KEYS,
    frames_to_trainer_splits,
    preloaded_splits,
)


def _frame(n: int, start: int = 0) -> pd.DataFrame:
    idx = range(start, start + n)
    return pd.DataFrame(
        {
            "hcp_id": [f"h{i}" for i in idx],
            "event_date": pd.to_datetime(["2026-01-01"] * n),
            "specialty": ["onc"] * n,
            "f1": [float(i) for i in idx],
            "treatment_initiated": [i % 2 for i in idx],
        },
        index=list(idx),
    )


FRAMES = {
    "train": _frame(6, 0),
    "validation": _frame(4, 100),
    "test": _frame(3, 200),
    "holdout": _frame(2, 300),
}


@pytest.mark.unit
def test_frames_become_x_y_row_count_with_target_entity_and_dates_removed():
    splits = frames_to_trainer_splits(
        FRAMES, "treatment_initiated", drop_columns=("hcp_id", "event_date")
    )
    assert set(splits) == set(SPLIT_KEYS)
    train = splits["train_data"]
    assert list(train["X"].columns) == ["specialty", "f1"]  # categoricals stay (trainer encodes)
    assert train["y"].tolist() == [0, 1, 0, 1, 0, 1]
    assert train["row_count"] == 6
    assert splits["holdout_data"]["row_count"] == 2
    # original indices are preserved (the trainer's leakage validator compares index sets)
    assert list(train["X"].index) == [0, 1, 2, 3, 4, 5]
    assert list(splits["validation_data"]["X"].index) == [100, 101, 102, 103]


@pytest.mark.unit
def test_missing_holdout_is_an_empty_split_but_missing_train_raises():
    frames = dict(FRAMES, holdout=None)
    splits = frames_to_trainer_splits(frames, "treatment_initiated")
    assert splits["holdout_data"]["row_count"] == 0 and splits["holdout_data"]["X"].empty
    with pytest.raises(ValueError, match="no train frame"):
        frames_to_trainer_splits(dict(FRAMES, train=None), "treatment_initiated")


@pytest.mark.unit
def test_unknown_or_absent_target_raises():
    with pytest.raises(ValueError, match="prediction_target is not set"):
        frames_to_trainer_splits(FRAMES, None)
    with pytest.raises(ValueError, match="not in the train frame"):
        frames_to_trainer_splits(FRAMES, "initiation_kisqali")


@pytest.mark.unit
def test_preloaded_splits_require_all_four():
    full = {k: {"X": 1} for k in SPLIT_KEYS}
    assert preloaded_splits(full) == full
    assert preloaded_splits({"train_data": {"X": 1}}) is None
    assert preloaded_splits({}) is None


# ---------------------------------------------------------------------------
# the pipeline stages
# ---------------------------------------------------------------------------


def _pipeline() -> MLFoundationPipeline:
    return MLFoundationPipeline(config=PipelineConfig(skip_mlflow=True, enable_hpo=False))


def _result() -> PipelineResult:
    r = PipelineResult(
        pipeline_run_id="run", status="running", current_stage=PipelineStage.DATA_PREPARATION
    )
    r.experiment_id = "exp-1"
    r.scope_spec = {
        "problem_type": "binary_classification",
        "prediction_target": "treatment_initiated",
        "entity_column": "hcp_id",
        "date_column": "event_date",
    }
    r.success_criteria = {"min_auc": 0.6}
    r.model_candidate = {"algorithm_name": "logistic_regression"}
    return r


@pytest.mark.unit
@pytest.mark.asyncio
async def test_data_prep_stage_stores_the_frames_on_the_result():
    pipeline = _pipeline()
    result = _result()
    fake_dp = MagicMock()
    fake_dp.run = AsyncMock(
        return_value={
            "qc_report": {"overall_score": 0.9},
            "baseline_metrics": {},
            "gate_passed": True,
            "train_df": FRAMES["train"],
            "validation_df": FRAMES["validation"],
            "test_df": FRAMES["test"],
            "holdout_df": None,
        }
    )
    with (
        patch.object(pipeline, "_get_agent", return_value=fake_dp),
        patch.object(pipeline.config, "enable_feast", False),
    ):
        await pipeline._run_data_preparation(
            input_data={"data_source": "patient_journeys"}, result=result, obs_context=None
        )
    assert result.prepared_frames is not None
    assert result.prepared_frames["train"] is FRAMES["train"]
    assert result.prepared_frames["holdout"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_training_stage_hands_prepared_frames_to_the_trainer_when_nothing_is_preloaded():
    pipeline = _pipeline()
    result = _result()
    result.prepared_frames = dict(FRAMES)
    captured: Dict[str, Any] = {}
    fake_trainer = MagicMock()

    async def _run(trainer_input):
        captured.update(trainer_input)
        return {"validation_metrics": {"roc_auc": 0.7}, "success_criteria_met": True}

    fake_trainer.run = AsyncMock(side_effect=_run)
    with patch.object(pipeline, "_get_agent", return_value=fake_trainer):
        await pipeline._run_model_training(
            input_data={"data_source": "patient_journeys", "target_outcome": "treatment_initiated"},
            result=result,
            obs_context=None,
        )
    for key in SPLIT_KEYS:
        assert set(captured[key]) >= {"X", "y", "row_count"}, key
    assert captured["train_data"]["row_count"] == 6
    assert "hcp_id" not in captured["train_data"]["X"].columns
    assert "treatment_initiated" not in captured["train_data"]["X"].columns
    assert captured["train_data"]["y"].name == "treatment_initiated"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_training_stage_keeps_the_callers_preloaded_splits():
    pipeline = _pipeline()
    result = _result()
    result.prepared_frames = dict(FRAMES)  # present, but the caller pre-loaded its own
    preloaded = {
        k: {"X": pd.DataFrame({"a": [1]}), "y": pd.Series([1]), "row_count": 1} for k in SPLIT_KEYS
    }
    captured: Dict[str, Any] = {}
    fake_trainer = MagicMock()

    async def _run(trainer_input):
        captured.update(trainer_input)
        return {"validation_metrics": {}, "success_criteria_met": False}

    fake_trainer.run = AsyncMock(side_effect=_run)
    with patch.object(pipeline, "_get_agent", return_value=fake_trainer):
        await pipeline._run_model_training(
            input_data={"data_source": "x", **preloaded}, result=result, obs_context=None
        )
    assert captured["train_data"] is preloaded["train_data"]
