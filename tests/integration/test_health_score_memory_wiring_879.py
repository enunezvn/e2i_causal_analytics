"""#879 faithful integration: health_score's AGENT RUN PATH must persist episodic memory.

Red-first proof for issue #879 (the #876 / PR #878 follow-up). #878 proved the
HOOK (``contribute_to_memory`` -> ``store_health_check``) persists when called
directly against the real DB — but the hook had ZERO callers in ``src/``: the
agent's production run path (``HealthScoreAgent.check_health``, the funnel used
by the ``/api/health-score`` route, ``quick_check``/``full_check`` and
``check_system_health``) never invoked it, so production wrote NO health_score
episodic rows. These tests drive the PRODUCTION run path — not the hook — and
assert the row actually lands in the live DB.

Determinism: a default-constructed agent has no health backends wired, so the
F1 fail-closed composer yields measured_count=0 -> grade F + the "No health
dimensions could be measured" critical issue + status="completed". That is
ALWAYS a significant event per ``_is_significant_health_event``, so the
episodic store must fire on every such run (outcome_type='success' per the
#876 mapping — the CHECK operation completed; system severity lives in
raw_content + importance_score).

Each test self-cleans (episodic rows by session_id; the working-memory cache
key best-effort). Gated like the other faithful real-DB tests; run with the
shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_health_score_memory_wiring_879.py'
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


def _health_score_rows(session_id: str) -> list:
    """Episodic rows the run-under-test produced, located by its session_id."""
    from src.memory.episodic_memory import get_supabase_client

    resp = (
        get_supabase_client()
        .table("episodic_memories")
        .select("memory_id, session_id, event_type, outcome_type, agent_name, description")
        .eq("agent_name", "health_score")
        .eq("session_id", session_id)
        .execute()
    )
    return resp.data or []


def _cleanup_rows(memory_ids: list) -> None:
    from src.memory.episodic_memory import get_supabase_client

    client = get_supabase_client()
    for memory_id in memory_ids:
        client.table("episodic_memories").delete().eq("memory_id", memory_id).execute()


async def _cleanup_cache(check_scope: str) -> None:
    """Best-effort: drop the working-memory cache key the run wrote so a
    grade-F quick-check from THIS TEST never shadows real dashboard reads on
    the shared box (TTL would expire it anyway)."""
    try:
        from src.agents.health_score.memory_hooks import get_health_score_memory_hooks

        await get_health_score_memory_hooks().invalidate_cache(check_scope)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_check_health_run_path_persists_episodic_row():
    """RED before #879: check_health never calls contribute_to_memory, so a
    deterministically-significant run leaves ZERO episodic rows."""
    from src.agents.health_score.agent import HealthScoreAgent

    session_id = str(uuid.uuid4())
    agent = HealthScoreAgent(enable_mlflow=False, enable_opik=False)

    output = await agent.check_health(scope="quick", session_id=session_id)

    rows = _health_score_rows(session_id)
    try:
        # The run itself must have completed and been significant (grade F,
        # nothing measured) — i.e. the store SHOULD have fired.
        assert output.status == "completed"
        assert output.health_grade == "F"
        assert output.critical_issues, "expected the unmeasured-dims critical issue"

        assert len(rows) == 1, (
            f"expected exactly 1 episodic row from the agent run path, got {len(rows)} — "
            "nothing in HealthScoreAgent.check_health invokes contribute_to_memory (#879)"
        )
        row = rows[0]
        assert row["session_id"] == session_id
        # DOMAIN signal: event_type + agent_name (#876 convention).
        assert row["event_type"] == "health_check_completed"
        # STATE signal: the CHECK completed -> 'success' (severity in raw_content).
        assert row["outcome_type"] == "success"
    finally:
        _cleanup_rows([r["memory_id"] for r in rows])
        await _cleanup_cache("quick")


@pytest.mark.asyncio
async def test_memory_store_failure_does_not_poison_run(monkeypatch):
    """046-trap: a FAILING episodic insert (fabricated failure, not a fabricated
    success) must not change the run's status/errors — the hook swallows it and
    the caller-side try/except guarantees the agent output is untouched."""
    from src.agents.health_score.agent import HealthScoreAgent

    async def _boom(*args, **kwargs):
        raise RuntimeError("fabricated episodic insert failure (046-trap probe)")

    # store_health_check imports this symbol inside the method body, so patching
    # the source module attribute intercepts the REAL store path mid-flight.
    monkeypatch.setattr("src.memory.episodic_memory.insert_episodic_memory_with_text", _boom)

    session_id = str(uuid.uuid4())
    agent = HealthScoreAgent(enable_mlflow=False, enable_opik=False)

    output = await agent.check_health(scope="quick", session_id=session_id)

    rows = _health_score_rows(session_id)
    try:
        # The run is poisoned by NOTHING: status stays completed, and no error
        # entry leaks from the memory path.
        assert output.status == "completed"
        assert output.health_grade == "F"
        assert all(
            "046-trap probe" not in str(e) and "memory" not in str(e).lower() for e in output.errors
        ), f"memory failure leaked into agent errors: {output.errors}"
        # And of course no row landed.
        assert rows == []
    finally:
        _cleanup_rows([r["memory_id"] for r in rows])
        await _cleanup_cache("quick")
