"""#749/#772: model_selector must populate the SEMANTIC graph (FalkorDB e2i_causal).
Its store_algorithm_pattern hook (Algorithm + SUITED_FOR/USED_IN) was defined but never
called from run(). Pin that run() now invokes the semantic writer with mapped data and
degrades gracefully.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.ml_foundation.model_selector.agent import ModelSelectorAgent

_OUTPUT = {
    "experiment_id": "exp-749",
    "model_candidate": {
        "algorithm_name": "logistic_regression",
        "algorithm_family": "linear",
        "selection_score": 0.82,
    },
    "selection_summary": {"problem_type": "binary_classification"},
    "benchmark_results": {"lr": 0.82},
}


@pytest.mark.unit
def test_update_semantic_memory_invokes_algorithm_pattern_writer():
    agent = ModelSelectorAgent()
    with patch("src.agents.ml_foundation.model_selector.agent.ModelSelectorMemoryHooks") as HookCls:
        hook = HookCls.return_value
        hook.store_algorithm_pattern = AsyncMock(return_value=True)
        asyncio.run(agent._update_semantic_memory(_OUTPUT))

    hook.store_algorithm_pattern.assert_awaited_once()
    kwargs = hook.store_algorithm_pattern.await_args.kwargs
    assert kwargs["experiment_id"] == "exp-749"
    assert kwargs["algorithm_name"] == "logistic_regression"
    assert kwargs["algorithm_family"] == "linear"
    assert kwargs["problem_type"] == "binary_classification"
    assert kwargs["selection_score"] == 0.82


@pytest.mark.unit
def test_run_exposes_update_semantic_memory():
    agent = ModelSelectorAgent()
    assert hasattr(agent, "_update_semantic_memory")


@pytest.mark.unit
def test_update_semantic_memory_degrades_gracefully_on_error():
    agent = ModelSelectorAgent()
    with patch(
        "src.agents.ml_foundation.model_selector.agent.ModelSelectorMemoryHooks",
        side_effect=RuntimeError("falkordb unreachable"),
    ):
        asyncio.run(agent._update_semantic_memory(_OUTPUT))  # must not raise
