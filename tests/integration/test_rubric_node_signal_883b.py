"""#883 PR B §5 faithful integration: rubric_node learning_signals realignment.

``RubricNode._store_evaluation`` wrote ``signal_type="rubric_evaluation"`` —
not a ``learning_signal_type`` member ({thumbs_up, thumbs_down, correction,
rating, implicit_positive, implicit_negative}, verified live) — plus two
nonexistent columns (``source_agent``, ``context_summary``) into
``learning_signals``: guaranteed 22P02/PGRST204 on every call, swallowed at
the broad except. Unreachable today only because no production graph build
injects ``db_client`` — but the write path must land a row BEFORE any wiring
makes it reachable (the #873 lesson).

Fix convention (#876/#878, mirrored from the 883-A reflector/SignalCollector
remaps): map onto the EXISTING enum member ``rating`` (a rubric evaluation IS
a graded score; ``signal_value`` = the weighted score), keep the
purpose-built rubric columns (``rubric_scores``/``rubric_total`` from
database/ml/022 and the ``improvement_*`` enum columns — the schema was
designed FOR this payload; only the signal_type literal and two column names
drifted), and fold the domain label + source agent + context summary into
``signal_details``.

RED: zero rows land (insert rejected, swallowed). GREEN: one row with the
mapped enum value and the rubric fields in their designed columns.

Run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_rubric_node_signal_883b.py'
"""

import os
import uuid
from typing import Any, cast

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB learning_signals test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


def _signals_for_session(session_id: str) -> list:
    from src.memory.episodic_memory import get_supabase_client

    return (
        get_supabase_client()
        .table("learning_signals")
        .select(
            "signal_id, signal_type, signal_value, signal_details, rubric_scores, "
            "rubric_total, improvement_type, improvement_priority, improvement_details, "
            "session_id"
        )
        .eq("session_id", session_id)
        .execute()
    ).data or []


def _cleanup_signals(session_id: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("learning_signals").delete().eq("session_id", session_id).execute()


async def _fresh_async_client():
    """The factories module caches the ASYNC Supabase client bound to the
    first caller's event loop; pytest-asyncio gives each test a fresh loop,
    so a test running after other async-client tests would see a dead-loop
    client (insert fails -> swallowed -> a false RED). Reset the cache so
    this test binds its own."""
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    from src.memory.services.factories import get_async_supabase_client

    return await get_async_supabase_client()


@pytest.mark.asyncio
async def test_rubric_evaluation_lands_learning_signal_row():
    """RED before fix: signal_type 'rubric_evaluation' 22P02s (plus
    source_agent/context_summary PGRST204) -> swallowed -> zero rows.
    GREEN: one 'rating' row with the rubric fields in their designed columns
    and the domain label preserved in signal_details."""
    from src.agents.feedback_learner.evaluation import (
        CriterionScore,
        EvaluationContext,
        ImprovementDecision,
        RubricEvaluation,
    )
    from src.agents.feedback_learner.nodes.rubric_node import RubricNode

    session_id = str(uuid.uuid4())
    marker = f"883b-rubric-{uuid.uuid4().hex[:8]}"

    evaluation = RubricEvaluation(
        weighted_score=3.4,
        criterion_scores=[
            CriterionScore(
                criterion="causal_validity", score=3.0, reasoning="confounders partially addressed"
            ),
            CriterionScore(criterion="actionability", score=4.0, reasoning="clear next steps"),
        ],
        decision=ImprovementDecision.SUGGESTION,
        overall_analysis=f"solid causal chain, light on uncertainty ({marker})",
        pattern_flags=[],
        improvement_suggestion="quantify uncertainty in the effect estimate",
        evaluation_method="llm",
    )
    context = EvaluationContext(
        user_query=f"Why did Remibrutinib TRx drop in the northeast? ({marker})",
        final_response="TRx decline attributable to access changes.",
        session_id=session_id,
        agent_names=["causal_impact"],
        messages_evaluated=1,
    )

    # The evaluator is not exercised by the storage path under test.
    node = RubricNode(evaluator=cast(Any, object()), db_client=await _fresh_async_client())

    try:
        await node._store_evaluation(evaluation, context)

        rows = _signals_for_session(session_id)
        assert len(rows) == 1, (
            "no learning_signals row landed — _store_evaluation swallowed an error "
            "(#883 §5: signal_type 'rubric_evaluation' is not a learning_signal_type "
            "member, and source_agent/context_summary are not columns; see the "
            "captured 'Failed to store rubric evaluation' warning above)"
        )
        row = rows[0]
        # The EXISTING enum member, never an extension (#876 convention).
        assert row["signal_type"] == "rating"
        assert row["signal_value"] == pytest.approx(3.4)
        # The purpose-built rubric columns (database/ml/022).
        assert row["rubric_total"] == pytest.approx(3.4)
        assert set(row["rubric_scores"].keys()) == {"causal_validity", "actionability"}
        # The improvement_* enum columns match the node's decision logic.
        assert row["improvement_type"] == "prompt"  # lowest criterion: causal_validity
        assert row["improvement_priority"] == "medium"  # SUGGESTION
        assert row["improvement_details"]["decision"] == "suggestion"
        assert row["improvement_details"]["evaluation_method"] == "llm"
        # Domain label + displaced fields live in signal_details.
        details = row["signal_details"]
        assert details["domain_signal"] == "rubric_evaluation"
        assert details["source_agent"] == "feedback_learner"
        assert details["context_summary"]["agents_used"] == ["causal_impact"]
        assert marker in details["context_summary"]["user_query"]
    finally:
        _cleanup_signals(session_id)


@pytest.mark.asyncio
async def test_rubric_storage_with_non_uuid_session_does_not_explode():
    """learning_signals.session_id is uuid-typed; EvaluationContext.session_id
    is a free string. A non-UUID session must not kill the insert — it is
    preserved in signal_details instead and the row still lands."""
    from src.agents.feedback_learner.evaluation import (
        CriterionScore,
        EvaluationContext,
        ImprovementDecision,
        RubricEvaluation,
    )
    from src.agents.feedback_learner.nodes.rubric_node import RubricNode
    from src.memory.episodic_memory import get_supabase_client

    marker = f"883b-rubric-nonuuid-{uuid.uuid4().hex[:8]}"
    evaluation = RubricEvaluation(
        weighted_score=4.6,
        criterion_scores=[
            CriterionScore(criterion="evidence_chain", score=4.6, reasoning="well cited"),
        ],
        decision=ImprovementDecision.ACCEPTABLE,
        overall_analysis=f"acceptable ({marker})",
        evaluation_method="llm",
    )
    context = EvaluationContext(
        user_query=f"probe ({marker})",
        final_response="ok",
        session_id=f"chat-session-{marker}",  # NOT a UUID
        agent_names=[],
    )

    node = RubricNode(evaluator=cast(Any, object()), db_client=await _fresh_async_client())
    client = get_supabase_client()
    try:
        await node._store_evaluation(evaluation, context)

        rows = (
            client.table("learning_signals")
            .select("signal_id, signal_type, session_id, signal_details")
            .contains("signal_details", {"raw_session_id": f"chat-session-{marker}"})
            .execute()
        ).data or []
        assert len(rows) == 1, (
            "row did not land for a non-UUID session — the uuid-typed session_id "
            "column must not be fed a free-form string"
        )
        assert rows[0]["session_id"] is None
        assert rows[0]["signal_type"] == "rating"
    finally:
        client.table("learning_signals").delete().contains(
            "signal_details", {"raw_session_id": f"chat-session-{marker}"}
        ).execute()
