"""#788 faithful integration: a causal_impact episodic write must land a REAL 1536-dim
vector on the row (not None), with zero swallowed errors.

This is the decisive, no-mocking proof the issue asks for: the REAL embedding service
(OpenAI text-embedding-ada-002 → 1536-dim) and the REAL Supabase ``episodic_memories``
table. It exercises BOTH episodic surfaces:

  1. ``CausalImpactAgent.save_episodic_memory`` — the contract method (#788 names it).
  2. ``contribute_to_memory`` → ``store_causal_analysis`` — the canonical path a real
     Tier-1 ``causal_impact`` / ``tool_composer`` run drives (the #788 verification and
     the mechanism #785 relies on), writing ``event_type='causal_analysis_completed'``.

Both previously failed at the ``memory_event_type`` enum (``causal_analysis*`` missing,
migration 040) and the ``memory_outcome_type`` enum (free-text outcome), errors the bare
``except`` swallowed. Gated behind ``E2I_RUN_REAL_EPISODIC=1`` (+ creds) so CI without
keys/services skips it; run faithfully with::

    dotenv run -- env E2I_RUN_REAL_EPISODIC=1 \
        python -m pytest tests/integration/test_causal_impact_episodic_embedding_788.py -q

Each test inserts a uniquely-marked row, reads the embedding back, asserts width 1536,
then deletes the row (non-polluting).
"""

import json
import os
import uuid

import pytest

_GATE = os.environ.get("E2I_RUN_REAL_EPISODIC") == "1"
_HAS_CREDS = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-embedding/real-supabase test; set E2I_RUN_REAL_EPISODIC=1 + creds",
    ),
]


def _embedding_len(raw) -> int:
    """pgvector reads back as a JSON-array string '[...]' (or a list); return its width."""
    if raw is None:
        return 0
    if isinstance(raw, list):
        return len(raw)
    if isinstance(raw, str):
        return len(json.loads(raw))
    return 0


@pytest.mark.asyncio
async def test_save_episodic_memory_lands_1536_dim_vector():
    """The contract method (#788 names it) must land a real 1536-dim vector."""
    from src.agents.causal_impact.agent import CausalImpactAgent
    from src.memory.episodic_memory import get_supabase_client

    agent = CausalImpactAgent()
    marker = f"788-contract-{uuid.uuid4()}"
    event = {
        "event_type": "causal_analysis",
        "description": f"{marker}: hcp_engagement_level -> patient_conversion_rate (ATE=0.12)",
        "importance": 0.7,
        "outcome": "completed",  # free-text status — must map to a VALID outcome enum
        "raw_content": {"marker": marker, "ate_estimate": 0.12, "method": "ols"},
    }
    session_id = str(uuid.uuid4())

    memory_id = await agent.save_episodic_memory(event, session_id=session_id)
    assert memory_id, "save_episodic_memory returned None — write path swallowed an error"

    client = get_supabase_client()
    try:
        resp = (
            client.table("episodic_memories")
            .select("memory_id, embedding, session_id, event_type, outcome_type")
            .eq("memory_id", memory_id)
            .execute()
        )
        rows = resp.data or []
        assert len(rows) == 1, f"expected exactly one row for {memory_id}, got {len(rows)}"
        row = rows[0]
        assert str(row["session_id"]) == session_id
        assert row["event_type"] == "causal_analysis"
        assert row["outcome_type"] == "success"  # "completed" mapped to a valid enum
        # The decisive assertion: a real 1536-dim vector landed (not None / not 384).
        assert _embedding_len(row["embedding"]) == 1536
    finally:
        client.table("episodic_memories").delete().eq("memory_id", memory_id).execute()


@pytest.mark.asyncio
async def test_contribute_to_memory_lands_causal_analysis_completed_episodic():
    """The canonical run path (#788 verification / #785 mechanism): contribute_to_memory
    → store_causal_analysis writes a ``causal_analysis_completed`` episodic with a real
    1536-dim vector and zero swallowed errors."""
    from src.agents.causal_impact.memory_hooks import contribute_to_memory
    from src.memory.episodic_memory import get_supabase_client

    session_id = str(uuid.uuid4())
    result = {
        "status": "completed",
        "ate_estimate": 0.153,
        "confidence": 0.72,
        "confidence_interval": (0.10, 0.21),
        "refutation_passed": True,
        # REVIEW band → episodic writes (partial_success) but store_causal_path is
        # skipped, keeping this test to a single, cleanly-removable episodic row.
        "gate_decision": "review",
        "needs_review": True,
        "model_used": "backdoor.linear_regression",
        "executive_summary": "Engagement raises conversion (788 self-test).",
    }
    state = {
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "confounders": ["geographic_region"],
        "query": "788 self-test canonical path",
        "brand": "kisqali",
    }

    counts = await contribute_to_memory(
        result=result, state=state, session_id=session_id, brand="kisqali", region="northeast"
    )
    assert counts["episodic_stored"] == 1, f"episodic not stored (counts={counts})"

    client = get_supabase_client()
    try:
        resp = (
            client.table("episodic_memories")
            .select("memory_id, embedding, session_id, event_type, outcome_type")
            .eq("session_id", session_id)
            .execute()
        )
        rows = resp.data or []
        assert len(rows) == 1, (
            f"expected one episodic row for session {session_id}, got {len(rows)}"
        )
        row = rows[0]
        assert row["event_type"] == "causal_analysis_completed"
        assert row["outcome_type"] == "partial_success"  # REVIEW band
        assert _embedding_len(row["embedding"]) == 1536  # real 1536-dim vector landed
    finally:
        client.table("episodic_memories").delete().eq("session_id", session_id).execute()
