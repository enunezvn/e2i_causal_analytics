"""#749/#772: model_trainer must populate the SEMANTIC graph (FalkorDB e2i_causal),
not only procedural memory. Its store_model_pattern hook (Model + Hyperparameters +
TRAINED_WITH/BELONGS_TO) was defined but never called from run(), so a full Tier 0 run
left e2i_causal unchanged. Pin that run() now invokes the semantic writer with the
mapped training data, and degrades gracefully.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

_OUTPUT = {
    "experiment_id": "exp-749",
    "training_run_id": "run-749",
    "algorithm_name": "logistic_regression",
    "test_metrics": {"auc_roc": 0.71, "f1_score": 0.5},
    "best_hyperparameters": {"C": 1.0},
    "success_criteria_met": True,
}


@pytest.mark.unit
def test_update_semantic_memory_invokes_model_pattern_writer():
    agent = ModelTrainerAgent()
    with patch("src.agents.ml_foundation.model_trainer.agent.ModelTrainerMemoryHooks") as HookCls:
        hook = HookCls.return_value
        hook.store_model_pattern = AsyncMock(return_value=True)
        asyncio.run(agent._update_semantic_memory(_OUTPUT))

    hook.store_model_pattern.assert_awaited_once()
    kwargs = hook.store_model_pattern.await_args.kwargs
    assert kwargs["experiment_id"] == "exp-749"
    assert kwargs["training_run_id"] == "run-749"
    assert kwargs["algorithm_name"] == "logistic_regression"
    assert kwargs["test_metrics"]["auc_roc"] == 0.71
    assert kwargs["best_hyperparameters"] == {"C": 1.0}
    assert kwargs["success_criteria_met"] is True


@pytest.mark.unit
def test_run_exposes_update_semantic_memory():
    agent = ModelTrainerAgent()
    assert hasattr(agent, "_update_semantic_memory"), (
        "model_trainer must expose _update_semantic_memory so run() populates e2i_causal"
    )


@pytest.mark.unit
def test_update_semantic_memory_degrades_gracefully_on_error():
    agent = ModelTrainerAgent()
    with patch(
        "src.agents.ml_foundation.model_trainer.agent.ModelTrainerMemoryHooks",
        side_effect=RuntimeError("falkordb unreachable"),
    ):
        asyncio.run(agent._update_semantic_memory(_OUTPUT))  # must not raise
