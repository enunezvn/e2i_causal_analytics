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
    assert (
        set(stores)
        == set(KNOWLEDGE_TYPES)
        == {
            "baseline",
            "agent_config",
            "prompt",
            "threshold",
        }
    )
    for kt, store in stores.items():
        assert isinstance(store, SupabaseKnowledgeStore)
        assert store._knowledge_type == kt


class _DBTouched(BaseException):
    """Raised if a code path reaches the DB. A BaseException subclass so the
    store's ``except Exception`` cannot swallow it — proving the short-circuit
    happened BEFORE any persistence call (decisive; cf. the #825 pattern)."""


class _ExplodingClient:
    """A client whose .table() raises _DBTouched — proves no DB access."""

    def table(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise _DBTouched("DB must not be touched for a meaningless value")


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [None, "", "   ", {}, []])
async def test_update_with_meaningless_value_returns_false_without_db_access(empty):
    """A None / blank string / empty collection is not a real recorded learning:
    update() must return False (not applied) WITHOUT any persistence call — so an
    empty learning can never be counted toward update_effectiveness. _DBTouched
    (BaseException) propagates if the store reaches the client, making this
    decisive rather than a happens-to-return-False via swallowed exception."""
    store = SupabaseKnowledgeStore(_ExplodingClient(), "baseline")
    assert await store.update(key="agent_x", value=empty, justification="n/a") is False


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


@pytest.mark.asyncio
async def test_build_production_feedback_stores_fail_closed(monkeypatch):
    """The shared production builder used by the Celery task, the /feedback/learn
    route, and process_feedback_batch must FAIL CLOSED to (None, None) when the
    async Supabase client is unavailable — so every wired entry point runs the
    honest unwired path (update_effectiveness None), never a fabricated value."""
    import src.memory.services.factories as factories

    async def _raise():
        raise RuntimeError("SUPABASE_URL unset")

    monkeypatch.setattr(factories, "get_async_supabase_client", _raise)

    from src.agents.feedback_learner.agent import build_production_feedback_stores

    feedback_store, knowledge_stores = await build_production_feedback_stores()
    assert feedback_store is None
    assert knowledge_stores is None
