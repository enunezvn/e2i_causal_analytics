"""#883 deferred learning-signals family (PR #884/#886 "flagged, not fixed here").

Three deferred items, each proven red-first against the live docker DB:

Item 1 — double-encoded JSONB. ``record_learning_signal``
(src/memory/procedural_memory.py:523) ran ``json.dumps(signal.signal_details)``
before the supabase insert, so the JSONB column stored a JSON **string
scalar**, not an object (the exact ``raw_content`` double-encode B2's
``get_prior_cohorts`` had to ``json.loads`` around). Same writer family:
``insert_procedural_memory`` (:249) double-encoded ``tool_sequence`` — live DB
showed ALL 1566 procedural_memories rows with
``jsonb_typeof(tool_sequence) = 'string'``.

    RED (pre-fix, this suite on 59b4067a):
      test_record_learning_signal_signal_details_lands_as_object
        AssertionError: signal_details came back as <class 'str'>
      test_insert_procedural_memory_tool_sequence_lands_as_array
        AssertionError: tool_sequence came back as <class 'str'>

Item 2 — missing FK/CASCADE. The SSOT
(database/memory/001_agentic_memory_schema_v1.3.sql:502) declares
``cycle_id UUID REFERENCES cognitive_cycles(cycle_id) ON DELETE CASCADE`` but
the live table has NO cycle_id constraint at all (pg_constraint: only
hcp/patient/trigger FKs). Deleting a cognitive cycle stranded its signals.
Migration 072 adds the constraint (orphans verified 0 live, NULLed
defensively first).

    RED (pre-072): test_learning_signals_cycle_delete_cascades
      AssertionError: signal survived the cycle delete (orphan) — no FK CASCADE

Item 3 — dead rubric persistence path. ``graph.py`` plumbs
``db_client -> RubricNode`` but NO production build site injected one, and
nothing set ``rubric_evaluation_context``, so the (post-#886-correct) rubric
write could never fire. The production path is armed by injecting the shared
async client at the production build sites and deriving the evaluation
context from the run's REAL collected feedback (most recent item with both
query and response).

    RED (pre-fix): test_production_learning_cycle_lands_rubric_signal
      AssertionError: no rubric learning_signals row landed from the
      production /feedback/learn path — the rubric persistence path is dead

Run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_learning_signals_deferred_883.py'
"""

import os
import random
import uuid
from datetime import datetime, timedelta, timezone

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB learning_signals tests; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


def _sync_client():
    from src.memory.services.factories import get_supabase_client

    return get_supabase_client()


async def _fresh_async_client():
    """Reset the factories' cached ASYNC client (bound to the first caller's
    event loop; pytest-asyncio gives each test a fresh loop — the B2 trap)."""
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    from src.memory.services.factories import get_async_supabase_client

    return await get_async_supabase_client()


# ---------------------------------------------------------------------------
# Item 1 — JSONB shape round-trips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_learning_signal_signal_details_lands_as_object():
    """RED before fix: postgrest receives the pre-dumped string and stores a
    JSON string scalar -> supabase read-back returns ``str``. GREEN: dict."""
    from src.memory.procedural_memory import LearningSignalInput, record_learning_signal

    session_id = str(uuid.uuid4())
    details = {"domain_signal": "shape-probe-883", "nested": {"k": 1, "list": [1, 2]}}

    try:
        await record_learning_signal(
            LearningSignalInput(
                signal_type="rating",
                signal_value=0.9,
                signal_details=details,
            ),
            session_id=session_id,
        )

        rows = (
            _sync_client()
            .table("learning_signals")
            .select("signal_id, signal_details")
            .eq("session_id", session_id)
            .execute()
        ).data or []
        assert len(rows) == 1, "learning_signals row did not land"
        got = rows[0]["signal_details"]
        assert isinstance(got, dict), (
            f"signal_details came back as {type(got)} — json.dumps before a JSONB "
            "insert double-encodes (stored as a JSON string scalar, not an object)"
        )
        assert got == details, "signal_details object did not round-trip intact"
    finally:
        _sync_client().table("learning_signals").delete().eq("session_id", session_id).execute()


@pytest.mark.asyncio
async def test_insert_procedural_memory_tool_sequence_lands_as_array():
    """RED before fix: every procedural_memories row stored tool_sequence as a
    JSON string scalar (live: 1566/1566 'string'). GREEN: a real JSON array."""
    from src.memory.procedural_memory import ProceduralMemoryInput, insert_procedural_memory

    marker = f"defL-883-shape-{uuid.uuid4().hex[:8]}"
    seq = [{"tool": "probe_a", "params": {"x": 1}}, {"tool": "probe_b"}]
    rng = random.Random(42)
    embedding = [rng.uniform(-1, 1) for _ in range(1536)]

    procedure_id = await insert_procedural_memory(
        ProceduralMemoryInput(
            procedure_name=marker,
            procedure_type="optimization",
            tool_sequence=seq,
            trigger_pattern=f"shape probe {marker}",
        ),
        trigger_embedding=embedding,
        dedup_name_prefix=marker,
    )
    try:
        rows = (
            _sync_client()
            .table("procedural_memories")
            .select("procedure_id, tool_sequence")
            .eq("procedure_id", procedure_id)
            .execute()
        ).data or []
        assert len(rows) == 1, "procedural_memories row did not land"
        got = rows[0]["tool_sequence"]
        assert isinstance(got, list), (
            f"tool_sequence came back as {type(got)} — json.dumps before a JSONB "
            "insert double-encodes (stored as a JSON string scalar, not an array)"
        )
        assert got == seq, "tool_sequence array did not round-trip intact"
    finally:
        _sync_client().table("procedural_memories").delete().eq(
            "procedure_id", procedure_id
        ).execute()


@pytest.mark.asyncio
async def test_store_hpo_pattern_jsonb_columns_land_as_objects():
    """Same writer family (#883 deferred sweep): store_hpo_pattern dumped
    search_space / best_hyperparameters / feature_types into JSONB — live DB
    held 887/887 'string'-shaped best_hyperparameters rows before migration
    072 repaired them. GREEN: real objects land."""
    from src.mlops.hpo_pattern_memory import HPOPatternInput, store_hpo_pattern

    marker = f"defL-883-hpo-{uuid.uuid4().hex[:8]}"
    pattern_id = await store_hpo_pattern(
        HPOPatternInput(
            algorithm_name=f"XGBoost_{marker}",
            problem_type="binary_classification",
            search_space={"n_estimators": {"type": "int", "low": 50, "high": 200}},
            best_hyperparameters={"n_estimators": 150},
            best_value=0.92,
            optimization_metric="roc_auc",
            n_trials=10,
            n_completed=10,
            feature_types={"age": "numeric"},
        )
    )
    assert pattern_id, "store_hpo_pattern did not persist"
    client = _sync_client()
    try:
        rows = (
            client.table("ml_hpo_patterns")
            .select("search_space, best_hyperparameters, feature_types")
            .eq("pattern_id", pattern_id)
            .execute()
        ).data or []
        assert len(rows) == 1
        row = rows[0]
        for col in ("search_space", "best_hyperparameters", "feature_types"):
            assert isinstance(row[col], dict), (
                f"{col} came back as {type(row[col])} — double-encoded JSON string scalar"
            )
        assert row["best_hyperparameters"] == {"n_estimators": 150}
    finally:
        client.table("ml_hpo_patterns").delete().eq("pattern_id", pattern_id).execute()
        client.table("procedural_memories").delete().eq("procedure_id", pattern_id).execute()


# ---------------------------------------------------------------------------
# Item 2 — cycle_id FK ON DELETE CASCADE (migration 072)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_learning_signals_cycle_delete_cascades():
    """RED before migration 072: learning_signals has NO cycle_id constraint
    (SSOT 001 declares ON DELETE CASCADE) — the signal survives the cycle
    delete as an orphan. GREEN: deleting the cycle cascades to the signal."""
    from src.memory.procedural_memory import LearningSignalInput, record_learning_signal

    client = _sync_client()
    session_id = str(uuid.uuid4())
    cycle = (
        client.table("cognitive_cycles")
        .insert(
            {
                "session_id": session_id,
                "user_query": "defL-883 cascade probe",
                "status": "completed",
            }
        )
        .execute()
    ).data
    assert cycle, "could not seed a cognitive_cycles row"
    cycle_id = cycle[0]["cycle_id"]

    try:
        signal_id = await record_learning_signal(
            LearningSignalInput(signal_type="implicit_positive", signal_value=0.5),
            cycle_id=cycle_id,
            session_id=session_id,
        )

        # Sanity: the signal landed and references the cycle.
        landed = (
            client.table("learning_signals")
            .select("signal_id")
            .eq("signal_id", signal_id)
            .execute()
        ).data
        assert landed, "learning_signals row did not land before the delete"

        client.table("cognitive_cycles").delete().eq("cycle_id", cycle_id).execute()

        survivors = (
            client.table("learning_signals")
            .select("signal_id, cycle_id")
            .eq("signal_id", signal_id)
            .execute()
        ).data or []
        assert survivors == [], (
            "signal survived the cycle delete (orphan) — learning_signals.cycle_id "
            "has no FK ON DELETE CASCADE (SSOT 001:502 declares it; migration 072 adds it)"
        )
    finally:
        client.table("learning_signals").delete().eq("session_id", session_id).execute()
        client.table("cognitive_cycles").delete().eq("cycle_id", cycle_id).execute()


# ---------------------------------------------------------------------------
# Item 3 — production rubric persistence path (armed via db_client injection
# + context derived from the run's real feedback)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(300)  # the rubric judge may make one real LLM call
async def test_production_learning_cycle_lands_rubric_signal():
    """Faithful production-path proof: seed ONE real chatbot_message_feedback
    row, run the actual /feedback/learn execution path
    (``_execute_learning_cycle`` — the exact production build at
    feedback.py:1353), and require a rubric learning_signals row to land.

    RED before fix: the production build injects no ``db_client`` and nothing
    sets ``rubric_evaluation_context`` -> the rubric node skips -> zero rows.
    GREEN: the route's graph build passes the shared async client; the rubric
    node derives its evaluation context from the run's real collected
    feedback; one ``signal_type='rating'`` row lands with the rubric columns
    populated and ``signal_details`` a real JSONB object (Item 1 tie-in).
    """
    from src.api.routes.feedback import RunLearningRequest, _execute_learning_cycle

    await _fresh_async_client()  # re-bind the cached async client to THIS loop

    client = _sync_client()
    marker = f"defL-883-rubric-{uuid.uuid4().hex[:8]}"
    fb_session = f"{uuid.uuid4()}~itest"
    now = datetime.now(timezone.utc)

    # Real FK chain: user profile -> conversation -> assistant message -> feedback.
    fb_user = fb_session.split("~")[0]
    client.table("chatbot_user_profiles").insert(
        {"id": fb_user, "email": f"itest-{fb_user[:8]}@example.test"}
    ).execute()
    client.table("chatbot_conversations").insert(
        {"session_id": fb_session, "user_id": fb_user}
    ).execute()
    msg = (
        client.table("chatbot_messages")
        .insert(
            {
                "session_id": fb_session,
                "role": "assistant",
                "content": "TRx decline attributable to access changes in Q3.",
                "agent_name": "causal_impact",
            }
        )
        .execute()
    ).data
    assert msg, "could not seed chatbot_messages"
    seeded = (
        client.table("chatbot_message_feedback")
        .insert(
            {
                "message_id": msg[0]["id"],
                "session_id": fb_session,
                "rating": "thumbs_down",
                "comment": "effect size not quantified",
                "query_text": f"Why did Remibrutinib TRx drop in the northeast? ({marker})",
                "response_preview": "TRx decline attributable to access changes in Q3.",
                "agent_name": "causal_impact",
                "created_at": now.isoformat(),
            }
        )
        .execute()
    ).data
    assert seeded, "could not seed chatbot_message_feedback"
    fb_id = seeded[0]["id"]

    def _rubric_rows(require_marker: bool = True):
        rows = (
            client.table("learning_signals")
            .select(
                "signal_id, signal_type, signal_value, signal_details, "
                "rubric_scores, rubric_total, improvement_type, improvement_priority"
            )
            .eq("signal_type", "rating")
            .gte("created_at", (now - timedelta(minutes=1)).isoformat())
            .execute()
        ).data or []
        matched = [
            r
            for r in rows
            if isinstance(r.get("signal_details"), dict)
            and r["signal_details"].get("domain_signal") == "rubric_evaluation"
        ]
        if require_marker:
            matched = [
                r
                for r in matched
                if marker
                in str(r["signal_details"].get("context_summary", {}).get("user_query", ""))
            ]
        return matched

    try:
        response = await _execute_learning_cycle(
            RunLearningRequest(
                time_range_start=(now - timedelta(minutes=10)).isoformat(),
                time_range_end=(now + timedelta(minutes=5)).isoformat(),
                focus_agents=["causal_impact"],
            )
        )
        assert response.status.value != "failed", f"learning cycle failed: {response.errors}"

        landed = _rubric_rows()
        assert len(landed) == 1, (
            "no rubric learning_signals row landed from the production "
            "/feedback/learn path — the rubric persistence path is dead "
            "(no production build site injects db_client and nothing sets "
            "rubric_evaluation_context)"
        )
        row = landed[0]
        assert row["signal_type"] == "rating"
        assert row["rubric_total"] is not None, "rubric_total (ml/022 column) not populated"
        assert row["rubric_scores"], "rubric_scores (ml/022 column) not populated"
        assert row["improvement_type"] is not None
        assert row["improvement_priority"] is not None
        # Item 1 tie-in: the details column is a real JSONB object, not a
        # double-encoded string scalar.
        assert isinstance(row["signal_details"], dict)
        assert row["signal_details"]["source_agent"] == "feedback_learner"
        assert row["signal_details"]["context_summary"]["agents_used"] == ["causal_impact"]
    finally:
        # Clean EVERY rubric row this run deposited (not only marker-matched
        # ones) so a derivation regression can never leave residue.
        for r in _rubric_rows(require_marker=False):
            client.table("learning_signals").delete().eq("signal_id", r["signal_id"]).execute()
        client.table("chatbot_message_feedback").delete().eq("id", fb_id).execute()
        # conversation delete cascades the seeded message
        client.table("chatbot_conversations").delete().eq("session_id", fb_session).execute()
        client.table("chatbot_user_profiles").delete().eq("id", fb_user).execute()
        # The route persists the finalized training signal (batch_id NULL at
        # this layer) — clean what this run deposited (window-scoped, and the
        # suite holds the shared flock).
        client.table("dspy_agent_training_signals").delete().eq(
            "source_agent", "feedback_learner"
        ).gte("created_at", (now - timedelta(minutes=1)).isoformat()).execute()
