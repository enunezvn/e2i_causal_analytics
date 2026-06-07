"""#749/#772: feature_analyzer ALREADY had a _update_semantic_memory, but it called the
non-existent semantic_memory.add_relationship(source=..., target=...) signature (never a
real method) and wrote NO typed Feature entities — so e2i_causal stayed unchanged. Pin
that it now drives the store_feature_importance_patterns hook (Feature + HAS_IMPORTANCE +
INTERACTS_WITH), maps state -> hook args, preserves its (updated, entries) return, and
degrades gracefully.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.ml_foundation.feature_analyzer.agent import FeatureAnalyzerAgent

_STATE = {
    "experiment_id": "exp-749",
    "global_importance_ranked": [("age_at_index", 0.5), ("payer_category", 0.3)],
    "top_interactions_raw": [("age_at_index", "payer_category", 0.12)],
}


@pytest.mark.unit
def test_update_semantic_memory_invokes_feature_importance_writer():
    agent = FeatureAnalyzerAgent()
    with patch(
        "src.agents.ml_foundation.feature_analyzer.agent.FeatureAnalyzerMemoryHooks"
    ) as HookCls:
        hook = HookCls.return_value
        hook.store_feature_importance_patterns = AsyncMock(return_value=True)
        updated, entries = asyncio.run(agent._update_semantic_memory(_STATE))

    hook.store_feature_importance_patterns.assert_awaited_once()
    kwargs = hook.store_feature_importance_patterns.await_args.kwargs
    assert kwargs["experiment_id"] == "exp-749"
    assert kwargs["global_importance"]["age_at_index"] == 0.5
    assert kwargs["global_importance"]["payer_category"] == 0.3
    assert kwargs["interactions"][0]["feature_1"] == "age_at_index"
    assert kwargs["interactions"][0]["feature_2"] == "payer_category"
    assert kwargs["interactions"][0]["interaction_strength"] == 0.12
    # preserves the (updated, entries) contract run() unpacks
    assert updated is True
    assert entries >= 2


@pytest.mark.unit
def test_run_exposes_update_semantic_memory():
    agent = FeatureAnalyzerAgent()
    assert hasattr(agent, "_update_semantic_memory")


@pytest.mark.unit
def test_update_semantic_memory_degrades_gracefully_on_error():
    agent = FeatureAnalyzerAgent()
    with patch(
        "src.agents.ml_foundation.feature_analyzer.agent.FeatureAnalyzerMemoryHooks",
        side_effect=RuntimeError("falkordb unreachable"),
    ):
        updated, entries = asyncio.run(agent._update_semantic_memory(_STATE))  # must not raise
    assert updated is False
    assert entries == 0
