"""#788: causal_impact must actually WRITE memory (episodic with a real 1536-dim vector +
semantic CausalPath) from a successful run.

The Tier-1 analog of the Tier-0 #787 drift. The episodic path was broken at several
layers, each of which silently produced *no usable row*:

  L1 — no memory was contributed from a direct ``causal_impact`` run: the lone
       ``save_episodic_memory`` was never called from ``run()``.
  L2 — ``save_episodic_memory`` built ``EpisodicMemoryInput(importance=, context_summary=,
       action_taken=, outcome=, emotional_valence=)`` with kwargs that do NOT exist on the
       dataclass → TypeError, swallowed by the bare ``except`` → returns None.
  L3 — it called ``insert_episodic_memory(memory=, embedding=None)`` whose canonical path
       stores the row WITHOUT a vector (None filtered out) → semantic recall silently
       missed every causal_impact episodic.
  L4 — the ``causal_analysis_completed`` / ``causal_analysis`` event_types were absent from
       the ``memory_event_type`` enum → every insert was rejected (22P02), swallowed
       (fixed by migration 040 + proven by the faithful integration test).

These tests pin the fix: ``run()`` contributes to memory via the canonical
``contribute_to_memory`` path (auto-embeds 1536-dim + grows CausalPath), gated behind
``enable_memory`` and graceful, with a UUID-valid ``session_id``; and the
``save_episodic_memory`` contract method builds a valid input routed through the
auto-embedding path.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.causal_impact.agent import CausalImpactAgent

# The dir conftest autouse-neutralizes ``_contribute_to_memory`` so synthetic run() tests
# stay hermetic. Capture the REAL method here (at import, before the fixture runs) so the
# two tests that exercise its body can bypass the no-op.
_REAL_CONTRIBUTE = CausalImpactAgent._contribute_to_memory

_SYNTHETIC_INPUT = {
    "query": "what is the impact of hcp engagement on patient conversions?",
    "treatment_var": "hcp_engagement_level",
    "outcome_var": "patient_conversion_rate",
    "confounders": ["geographic_region"],
    "data_source": "synthetic",
    "interpretation_depth": "standard",
    "brand": "kisqali",
    "region": "northeast",
}


# ---------------------------------------------------------------------------
# L2 + L3: the save_episodic_memory CONTRACT method builds a VALID input and
# routes through the auto-embedding path (1536-dim lands), never the vectorless one.
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_episodic_memory_routes_through_auto_embed():
    agent = CausalImpactAgent()
    event = {
        "event_type": "causal_analysis",
        "description": "Causal analysis: hcp_engagement_level -> patient_conversion_rate",
        "importance": 0.8,
        "raw_content": {"ate": 0.12, "method": "ols"},
        "outcome": "completed",
    }
    session_id = str(uuid.uuid4())

    with (
        patch(
            "src.memory.episodic_memory.insert_episodic_memory_with_text",
            new=AsyncMock(return_value="mem-788"),
        ) as auto_embed,
        patch(
            "src.memory.episodic_memory.insert_episodic_memory",
            new=AsyncMock(return_value="VECTORLESS"),
        ) as vectorless,
    ):
        result = await agent.save_episodic_memory(event, session_id=session_id)

    # Returns the real id — proves no swallowed TypeError (L2) and no None.
    assert result == "mem-788"
    # Routed through the AUTO-EMBED path (the only path that lands a 1536-dim vector, L3).
    auto_embed.assert_awaited_once()
    # The deprecated embedding=None vectorless path must NOT be used.
    vectorless.assert_not_awaited()

    call = auto_embed.await_args
    memory = call.kwargs.get("memory") or call.args[0]
    assert memory.event_type == "causal_analysis"
    assert memory.description
    assert memory.importance_score == 0.8  # mapped to the REAL dataclass field name
    assert call.kwargs.get("session_id") == session_id


# ---------------------------------------------------------------------------
# _contribute_to_memory delegates to the canonical contribute_to_memory with the
# real output/state, brand/region, and a UUID-valid session_id (#787 column trap).
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_contribute_to_memory_uses_uuid_session_and_real_payload():
    agent = CausalImpactAgent()
    output = {
        "query_id": "q-abc123def456",  # NOT a uuid — must never be used as session_id
        "status": "completed",
        "ate_estimate": 0.123,
        "confidence": 0.71,
        "refutation_passed": True,
    }
    state = {
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "brand": "kisqali",
        "region": "northeast",
        "audit_workflow_id": "22222222-2222-2222-2222-222222222222",
    }

    with patch(
        "src.agents.causal_impact.memory_hooks.contribute_to_memory",
        new=AsyncMock(return_value={"episodic_stored": 1, "semantic_stored": 1}),
    ) as contrib:
        await _REAL_CONTRIBUTE(agent, output, state)

    contrib.assert_awaited_once()
    kwargs = contrib.await_args.kwargs
    assert kwargs["result"] is output
    assert kwargs["state"] is state
    assert kwargs["brand"] == "kisqali"
    assert kwargs["region"] == "northeast"
    # session_id MUST parse as a UUID (the column is uuid) — uses audit_workflow_id.
    uuid.UUID(kwargs["session_id"])
    assert kwargs["session_id"] == "22222222-2222-2222-2222-222222222222"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_contribute_to_memory_falls_back_to_fresh_uuid_when_no_workflow_id():
    agent = CausalImpactAgent()
    output = {"query_id": "q-xyz", "status": "completed", "ate_estimate": 0.1, "confidence": 0.6}
    state = {"treatment_var": "t", "outcome_var": "o"}  # no audit_workflow_id / session_id

    with patch(
        "src.agents.causal_impact.memory_hooks.contribute_to_memory",
        new=AsyncMock(return_value={}),
    ) as contrib:
        await _REAL_CONTRIBUTE(agent, output, state)

    contrib.assert_awaited_once()
    sid = contrib.await_args.kwargs["session_id"]
    uuid.UUID(sid)  # still a valid UUID (freshly generated)
    assert sid != "q-xyz"  # never the non-uuid query_id


# ---------------------------------------------------------------------------
# L1: run() must CONTRIBUTE to memory on a real (synthetic) success.
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_contribute_to_memory_forwards_raw_session_id_not_minted():
    """contribute_to_memory forwards the RAW session_id downstream; it no longer mints
    a fresh uuid for a non-uuid id (the #787/#788 mint is reversed). Coercion for the
    uuid-typed ``episodic_memories.session_id`` column now happens at the writer
    boundary (#1404) — a non-uuid becomes an honest NULL there and a composite
    ``{user}~{session}`` id recovers the real session uuid, never a fabricated one that
    destroys session linkage (#1403)."""
    from src.agents.causal_impact.memory_hooks import contribute_to_memory

    captured: dict = {}

    class _Hooks:
        async def cache_causal_analysis(self, *a, **k):
            return False

        async def store_causal_analysis(self, *, session_id, **k):
            captured["session_id"] = session_id
            return "mem-x"

        async def store_causal_path(self, *a, **k):
            return False

    result = {
        "status": "completed",
        "ate_estimate": 0.1,
        "confidence": 0.6,
        "refutation_passed": False,  # → store_causal_path skipped
    }
    state = {"treatment_var": "t", "outcome_var": "o"}

    await contribute_to_memory(
        result=result, state=state, memory_hooks=_Hooks(), session_id="q-not-a-uuid"
    )

    # Forwarded as-is — the writer coerces it to an honest NULL downstream; the hook
    # no longer replaces a real id with a random uuid (#1403).
    assert captured["session_id"] == "q-not-a-uuid"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_contributes_to_memory_on_success():
    agent = CausalImpactAgent()  # enable_memory defaults True

    with (
        patch.object(agent, "_get_mlflow_tracker", return_value=None),
        patch.object(agent, "_contribute_to_memory", new=AsyncMock(return_value=None)) as contrib,
    ):
        result = await agent.run(dict(_SYNTHETIC_INPUT))

    assert result["status"] == "completed"
    contrib.assert_awaited_once()  # the real run contributed memory


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_skips_memory_when_disabled():
    agent = CausalImpactAgent(enable_memory=False)

    with (
        patch.object(agent, "_get_mlflow_tracker", return_value=None),
        patch.object(agent, "_contribute_to_memory", new=AsyncMock(return_value=None)) as contrib,
    ):
        result = await agent.run(dict(_SYNTHETIC_INPUT))

    assert result["status"] == "completed"
    contrib.assert_not_awaited()  # gated off — no write, no pollution


# ---------------------------------------------------------------------------
# Graceful degradation: a failing memory write must never break the analysis.
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.asyncio
async def test_contribute_to_memory_degrades_gracefully():
    agent = CausalImpactAgent()
    output = {"query_id": "q-1", "status": "completed", "ate_estimate": 0.1, "confidence": 0.6}
    state = {"treatment_var": "t", "outcome_var": "o"}

    with patch(
        "src.agents.causal_impact.memory_hooks.contribute_to_memory",
        new=AsyncMock(side_effect=RuntimeError("supabase down")),
    ):
        await _REAL_CONTRIBUTE(agent, output, state)  # must NOT raise


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_succeeds_even_if_memory_contribution_raises():
    agent = CausalImpactAgent()

    with (
        patch.object(agent, "_get_mlflow_tracker", return_value=None),
        patch.object(
            agent, "_contribute_to_memory", new=AsyncMock(side_effect=RuntimeError("boom"))
        ),
    ):
        result = await agent.run(dict(_SYNTHETIC_INPUT))

    # run() owns graceful degradation around the memory write.
    assert result["status"] == "completed"
