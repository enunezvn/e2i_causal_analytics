"""#873 faithful integration: heterogeneous_optimizer's episodic store must PERSIST.

Red-first proof for issue #873. Before the fix, ``store_cate_analysis`` built its
``EpisodicMemoryInput`` with ``outcome_type="cate_analysis_delivered"`` — not a value
of the DB ``memory_outcome_type`` enum (success / partial_success / failure / pending
/ escalated) — so every insert raised 22P02, the hook's ``except Exception`` swallowed
it into a ``logger.warning``, and het CATE analyses were silently NEVER written to
episodic memory (fail-open-by-logging).

This test drives the REAL store path (real Supabase ``episodic_memories`` table, real
embedding service — no mocks) and pins that the happy path actually lands a row:
``store_cate_analysis`` returns a memory_id AND the row exists with

  * ``event_type   = 'cate_analysis_completed'``  (the DOMAIN event — added to
    ``memory_event_type`` by database/memory/046; this is where the "CATE analysis
    delivered" signal lives, together with ``agent_name='heterogeneous_optimizer'``)
  * ``outcome_type = 'success'``                  (the generic STATE enum — #873
    decision (b): map, don't extend; mirrors the causal_impact #788 fix)

Each test inserts a uniquely-marked row and deletes it afterwards (non-polluting).
Gated like the other faithful real-DB tests; run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_het_episodic_outcome_873.py'
"""

import os
import uuid

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB episodic test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


@pytest.mark.asyncio
async def test_store_cate_analysis_persists_episodic_row():
    """The het episodic store path must return a memory_id and land a real row.

    Red before the #873 fix: the ``memory_outcome_type`` enum rejects
    ``'cate_analysis_delivered'`` (22P02), the hook swallows it and returns None.
    """
    from src.agents.heterogeneous_optimizer.memory_hooks import (
        HeterogeneousOptimizerMemoryHooks,
    )
    from src.memory.episodic_memory import get_supabase_client

    hooks = HeterogeneousOptimizerMemoryHooks()
    marker = f"873-het-{uuid.uuid4()}"
    session_id = str(uuid.uuid4())
    analysis_result = {
        "status": "completed",
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "overall_ate": 0.12,
        "heterogeneity_score": 0.34,
        "high_responders": [{"segment": "northeast_high_decile", "cate": 0.21}],
        "low_responders": [{"segment": "rural_low_decile", "cate": 0.02}],
        "marker": marker,
    }

    memory_id = await hooks.store_cate_analysis(
        session_id=session_id,
        analysis_result=analysis_result,
        brand="kisqali",
        region="northeast",
    )
    assert memory_id, (
        "store_cate_analysis returned None — the episodic write swallowed an error "
        "(#873: memory_outcome_type enum rejected the outcome value; see captured "
        "'Failed to store CATE analysis in episodic memory' warning above)"
    )

    client = get_supabase_client()
    try:
        resp = (
            client.table("episodic_memories")
            .select("memory_id, session_id, event_type, outcome_type, agent_name, description")
            .eq("memory_id", memory_id)
            .execute()
        )
        rows = resp.data or []
        assert len(rows) == 1, f"expected exactly one row for {memory_id}, got {len(rows)}"
        row = rows[0]
        assert str(row["session_id"]) == session_id
        # The DOMAIN signal: this is where "a CATE analysis was delivered" lives.
        assert row["event_type"] == "cate_analysis_completed"
        assert row["agent_name"] == "heterogeneous_optimizer"
        # The STATE signal: generic memory_outcome_type enum value (#873 decision (b)).
        assert row["outcome_type"] == "success"
        assert "CATE analysis" in (row["description"] or "")
    finally:
        client.table("episodic_memories").delete().eq("memory_id", memory_id).execute()
