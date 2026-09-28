"""Stage 4's primary_metric reads the trainer's real AUC key (Part of #2157).

Live, 2026-09-28 (retrain 073c38eb): MLflow logged primary_metric=0.8363, but the pipeline
logged ``Stage 4 complete: ... primary_metric=0.0000``. That same value went into the
model_trainer audit-chain entry. The evaluator emits ``test_metrics["roc_auc"]``
(model_trainer/nodes/evaluator.py), and the pipeline read only ``auc_roc``.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.tier_0.pipeline import (
    MLFoundationPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)

pytestmark = pytest.mark.unit


async def _audited_primary_metric(test_metrics: Dict[str, Any]) -> Any:
    pipeline = MLFoundationPipeline(config=PipelineConfig(skip_mlflow=True, enable_hpo=False))
    result = PipelineResult(
        pipeline_run_id="run", status="running", current_stage=PipelineStage.MODEL_TRAINING
    )
    result.experiment_id = "exp-2157"
    result.scope_spec = {"problem_type": "binary_classification"}
    result.model_candidate = {"algorithm_name": "LogisticRegression"}
    trainer = MagicMock()
    trainer.run = AsyncMock(
        return_value={
            "training_run_id": "train_x",
            "success_criteria_met": True,
            "validation_metrics": {"roc_auc": 0.8363},
            "test_metrics": test_metrics,
        }
    )
    audit = MagicMock()
    with (
        patch.object(pipeline, "_get_agent", return_value=trainer),
        patch.object(pipeline, "_record_audit_entry", audit),
    ):
        await pipeline._run_model_training({"problem_description": "p"}, result, None)
    (call,) = [c for c in audit.call_args_list if c.kwargs["agent_name"] == "model_trainer"]
    return call.kwargs["output_data"]["primary_metric"]


@pytest.mark.asyncio
async def test_the_evaluators_roc_auc_is_the_primary_metric():
    assert await _audited_primary_metric({"roc_auc": 0.8353, "f1": 0.69}) == 0.8353


@pytest.mark.asyncio
async def test_the_alias_key_still_resolves():
    assert await _audited_primary_metric({"auc_roc": 0.8353}) == 0.8353


@pytest.mark.asyncio
async def test_regression_metrics_still_resolve():
    assert await _audited_primary_metric({"rmse": 1.25, "r2": 0.4}) == 1.25
