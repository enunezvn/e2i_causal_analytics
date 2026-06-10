"""Unit tests (no DB) for the feedback_learner knowledge stores (#837).

Cover the CI-safe contracts: the build factory's fail-closed behavior (no client
=> empty dict => update_backend_wired stays False => update_effectiveness None,
the F15 honest path), and the store's None-value short-circuit which must NOT
touch the DB. The real persist + read-back path is proven faithfully in
tests/integration/test_feedback_learner_knowledge_stores_realdb.py.

REASON-BEFORE-RULES / anti-mocking: the fail-closed paths are driven with REAL
``None`` substrate (no client / empty value), not a patched stub that fakes
success. The "no DB touch" assertion uses a client whose ``.table`` raises, to
PROVE the short-circuit happens before any persistence call.
"""

from __future__ import annotations

import pytest

from src.agents.feedback_learner.graph import _finalize_training_signal
from src.agents.feedback_learner.knowledge_stores import (
    KNOWLEDGE_TYPES,
    SupabaseKnowledgeStore,
    build_knowledge_stores,
)
from src.agents.feedback_learner.nodes.knowledge_updater import KnowledgeUpdaterNode


def test_build_knowledge_stores_none_is_empty():
    """No client => empty dict. KnowledgeUpdaterNode then reports
    update_backend_wired=False (bool({}) is False), preserving the F15 honest
    path where update_effectiveness is None."""
    assert build_knowledge_stores(None) == {}


def test_build_knowledge_stores_builds_all_four_typed_stores():
    sentinel_client = object()
    stores = build_knowledge_stores(sentinel_client)
    assert set(stores) == set(KNOWLEDGE_TYPES) == {
        "baseline",
        "agent_config",
        "prompt",
        "threshold",
    }
    for kt, store in stores.items():
        assert isinstance(store, SupabaseKnowledgeStore)
        assert store._knowledge_type == kt


class _ExplodingClient:
    """A client whose .table() raises — proves a code path never touched the DB."""

    def table(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("DB must not be touched")


@pytest.mark.asyncio
async def test_update_with_none_value_returns_false_without_db_access():
    """A None/empty proposed_change is not a real recorded learning: update()
    must return False (not applied) WITHOUT any persistence call — so it can
    never be counted toward update_effectiveness."""
    store = SupabaseKnowledgeStore(_ExplodingClient(), "baseline")
    result = await store.update(key="agent_x", value=None, justification="n/a")
    assert result is False


@pytest.mark.asyncio
async def test_node_unwired_effectiveness_is_none():
    """With no stores wired (build_knowledge_stores(None)), the node reports
    update_backend_wired=False and the real finalize emits update_effectiveness
    None — never a misleading 0.0 (F15 contract preserved by #837)."""
    node = KnowledgeUpdaterNode(build_knowledge_stores(None))
    state = {
        "status": "running",
        "learning_recommendations": [
            {
                "recommendation_id": "R1",
                "category": "data_update",
                "affected_agents": ["explainer"],
                "proposed_change": "update baseline",
                "description": "d",
            }
        ],
    }
    out = await node.execute(state)  # type: ignore[arg-type]
    assert out["update_backend_wired"] is False
    assert out["applied_updates"] == []
    assert len(out["proposed_updates"]) == 1

    final = await _finalize_training_signal(out)  # type: ignore[arg-type]
    assert final["training_signal"].update_effectiveness is None
