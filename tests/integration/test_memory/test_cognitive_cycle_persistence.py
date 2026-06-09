"""Faithful producer test: the 4-phase CognitiveService persists a real
`cognitive_cycles` parent row (audit-F1 reversal, owner decision 2026-06-09).

NO MOCKS. This writes to the real Supabase `cognitive_cycles` table and reads the
row back, then deletes it (self-cleaning). It SKIPS when Supabase / the restored
table is unreachable (e.g. CI with no live DB), so it is a local faithfulness
guard, not a CI-gating unit test — consistent with the repo's real-DB test policy.

Why this exists: migration 032 dropped cognitive_cycles on a "no writer" rationale,
but the live workflow generates a `cycle_id` per query and threads it onto
episodic_memories / learning_signals while NEVER writing the parent row. Migration
042 restored the table and this change wires the producer; this test proves the
producer actually persists a real, readable cycle record.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.memory.cognitive_integration import CognitiveQueryInput, CognitiveService


def _supabase_cognitive_cycles_reachable() -> bool:
    """True only if a live Supabase has the restored cognitive_cycles table."""
    try:
        from src.memory.services.factories import get_supabase_client

        client = get_supabase_client()
        client.table("cognitive_cycles").select("cycle_id").limit(1).execute()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _supabase_cognitive_cycles_reachable(),
    reason="faithful test: requires a live Supabase with the restored cognitive_cycles table (no mocks)",
)


async def test_persist_cognitive_cycle_writes_real_readable_row() -> None:
    """_persist_cognitive_cycle upserts a real cognitive_cycles row from real
    4-phase results, readable back with the cycle's actual fields."""
    from src.memory.services.factories import get_supabase_client

    service = CognitiveService()
    cycle_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    cognitive_input = CognitiveQueryInput(
        query="Why did Kisqali TRx dip in the Northeast?",
        session_id=session_id,
        user_id="tdd-user",
        brand="Kisqali",
        region="northeast",
    )
    phase1_result = {
        "query_type": "causal",
        "entities": {"brands": ["Kisqali"], "regions": ["northeast"]},
    }
    phase2_result = {"evidence": [{"id": "e1"}, {"id": "e2"}], "hops_executed": 2}
    phase3_result = {
        "response": "TRx dipped due to access friction.",
        "confidence": 0.82,
        "agent_used": "causal_analyst",
    }

    try:
        await service._persist_cognitive_cycle(
            cycle_id=cycle_id,
            session_id=session_id,
            cognitive_input=cognitive_input,
            phase1_result=phase1_result,
            phase2_result=phase2_result,
            phase3_result=phase3_result,
            phases_completed=["summarizer", "investigator", "agent", "reflector_started"],
            status="completed",
            started_at=started_at,
            duration_ms=1234.5,
        )

        client = get_supabase_client()
        row = (
            client.table("cognitive_cycles").select("*").eq("cycle_id", cycle_id).single().execute()
        ).data

        assert row is not None, "producer did not persist a cognitive_cycles row"
        assert row["session_id"] == session_id
        assert row["user_query"] == cognitive_input.query
        assert row["detected_intent"] == "causal"
        assert row["status"] == "completed"
        assert row["synthesized_response"] == "TRx dipped due to access friction."
        assert abs(float(row["confidence_score"]) - 0.82) < 1e-6
        assert int(row["total_duration_ms"]) == 1234
        assert row["evidence_collected"] == 2
        assert row["hops_executed"] == 2
        assert "Kisqali" in (row["brands_discussed"] or [])
        # agent identity lives in agent_outputs JSONB (NOT the e2i_agent_name[] enum
        # column) so a free-form agent label can never trip a 22P02 enum error.
        assert row["agent_outputs"]["agent_used"] == "causal_analyst"
    finally:
        get_supabase_client().table("cognitive_cycles").delete().eq("cycle_id", cycle_id).execute()
