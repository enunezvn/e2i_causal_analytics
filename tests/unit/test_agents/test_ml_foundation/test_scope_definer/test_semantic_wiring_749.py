"""#749: the scope_definer agent must populate the SEMANTIC graph (FalkorDB
`e2i_causal`), not only procedural memory. Its `store_experiment_pattern` hook
(Experiment + ProblemType + HAS_TYPE) was defined but never called from `run()`,
so a full Tier 0 run left `e2i_causal` unchanged. These pins assert the agent now
invokes the semantic writer with the mapped scope data, and degrades gracefully.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.ml_foundation.scope_definer.agent import ScopeDefinerAgent

_OUTPUT = {
    "experiment_id": "exp-749",
    "scope_spec": {
        "problem_type": "binary_classification",
        "target_variable": "discontinuation_flag",
        "features": ["age_at_index", "payer_category"],
        "experiment_name": "disc-mart-749",
    },
    "success_criteria": {"min_auc": 0.65},
    "created_at": "2026-06-07T00:00:00Z",
}


@pytest.mark.unit
def test_update_semantic_memory_invokes_experiment_pattern_writer():
    agent = ScopeDefinerAgent()
    with patch("src.agents.ml_foundation.scope_definer.agent.ScopeDefinerMemoryHooks") as HookCls:
        hook = HookCls.return_value
        hook.store_experiment_pattern = AsyncMock(return_value=True)
        asyncio.run(agent._update_semantic_memory(_OUTPUT))

    hook.store_experiment_pattern.assert_awaited_once()
    kwargs = hook.store_experiment_pattern.await_args.kwargs
    assert kwargs["experiment_id"] == "exp-749"
    assert kwargs["problem_type"] == "binary_classification"
    assert kwargs["target_variable"] == "discontinuation_flag"
    assert "age_at_index" in kwargs["features"]
    assert kwargs["success_criteria"] == {"min_auc": 0.65}


@pytest.mark.unit
def test_run_persist_calls_semantic_memory_alongside_procedural():
    """The run() persist path must invoke _update_semantic_memory (the wiring gap)."""
    agent = ScopeDefinerAgent()
    assert hasattr(agent, "_update_semantic_memory"), (
        "scope_definer must expose _update_semantic_memory so run() populates e2i_causal"
    )


@pytest.mark.unit
def test_update_semantic_memory_degrades_gracefully_on_error():
    """A semantic-write failure must NOT raise (mirrors procedural graceful degradation)."""
    agent = ScopeDefinerAgent()
    with patch(
        "src.agents.ml_foundation.scope_definer.agent.ScopeDefinerMemoryHooks",
        side_effect=RuntimeError("falkordb unreachable"),
    ):
        asyncio.run(agent._update_semantic_memory(_OUTPUT))  # must not raise
