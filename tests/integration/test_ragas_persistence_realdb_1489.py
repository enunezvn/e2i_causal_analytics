"""#1489 deferral 6: the RAGAS persistence paths against the REAL database.

Why a real-DB pass and not just unit tests
------------------------------------------
Every failure this schema has actually had was invisible to a mocked writer.
#883: ``signal_type="rubric_evaluation"`` was not a ``learning_signal_type``
member, so every insert 22P02'd and was swallowed — with a mock client the
payload looked perfect. #883/#873: ``json.dumps`` before a supabase insert
double-encodes into a JSON **string scalar**, which a MagicMock records as a
correct-looking dict. Column CHECK constraints, enum membership, uuid typing
and JSONB shape only exist in Postgres. So the writers #1489 wired
(``retrieved_chunks``/``retrieval_scores`` on the live path,
``ragas_scores``/``ragas_weighted``/``combined_score`` and
``evaluation_results`` on the offline one) are exercised here for real.

This box IS prod
----------------
Every row created here carries ``TEST_MARKER`` and a minted UUID, and the
fixture deletes them on teardown whether the test passed or failed. The
``learning_signals`` rows deliberately do NOT set
``signal_details.domain_signal = 'dspy_signal'``, which is the exact predicate
``LearningSignalsFeedbackStore`` filters on — so even mid-test they are
invisible to the Tier-5 feedback learner and cannot enter a training set.

No LLM is called anywhere in this file. The RAGAS bundles are literals
standing in for a judge run that already happened, which is precisely the
shape the offline producer consumes.

Run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_ragas_persistence_realdb_1489.py'
"""

from __future__ import annotations

import os
import uuid

import pytest

from tests.integration._asyncio_compat import run_sync

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB RAGAS persistence test; set E2I_DB_INTEGRATION=1 + creds",
    ),
]

TEST_MARKER = "e2i-1489-integration"


def _sync_client():
    from src.memory.services.factories import get_supabase_client

    return get_supabase_client()


@pytest.fixture(autouse=True)
def fresh_async_client():
    """Drop the module-cached async client around every test.

    ``get_async_supabase_client`` memoises into ``factories._async_supabase_client``
    for connection reuse, and the ``httpx.AsyncClient`` underneath binds to the
    loop that created it. ``run_sync`` deliberately creates AND CLOSES a fresh
    loop per call (#220), so a cached client outliving its loop raises
    ``RuntimeError: Event loop is closed`` on the next test. Measured: without
    this fixture 4 of these 10 tests failed that way, and which 4 depended on
    collection order.
    """
    from src.memory.services import factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


def _with_async_client(fn):
    """Run ``fn(client)`` with a real async client on ONE loop.

    Every await for a given test has to happen inside a single ``run_sync``:
    building the client in one call and using it in the next hands the second
    call a connection pool whose loop is already closed.
    """

    async def _inner():
        from src.memory.services.factories import get_async_supabase_client

        return await fn(await get_async_supabase_client())

    return run_sync(_inner())


def _sweep_marked_rows() -> None:
    """Delete every row carrying TEST_MARKER, whoever wrote it.

    Tracking ids as they are returned is NOT sufficient, and that is measured
    rather than assumed: on the first run of this file four tests died inside
    ``run_sync`` with ``Event loop is closed`` AFTER their INSERT had already
    committed, so the id never reached the tracker and three rows survived
    teardown on the prod database. A sweep keyed on the marker cleans up a row
    whose id was never observed, and re-cleans anything an earlier crashed run
    left behind.

    Child before parent: ``evaluation_results.learning_signal_id`` is a real FK.
    """
    client = _sync_client()
    client.table("evaluation_results").delete().like("query", f"%{TEST_MARKER}%").execute()
    client.table("learning_signals").delete().like(
        "signal_details->>test_marker", f"%{TEST_MARKER}%"
    ).execute()
    client.table("learning_signals").delete().like(
        "signal_details->context_summary->>user_query", f"%{TEST_MARKER}%"
    ).execute()


@pytest.fixture
def owned_rows():
    """Delete everything this file writes, before AND after each test.

    Teardown runs even on failure: a red test must not be the reason a prod
    table keeps a junk row. The pre-test sweep means a previously crashed run
    cannot leak into this one's assertions (the trend-view test counts rows).
    """
    _sweep_marked_rows()

    class _Owned:
        marker = TEST_MARKER

        def signal(self, signal_id):
            return signal_id

        def evaluation(self, row):
            return row

    try:
        yield _Owned()
    finally:
        _sweep_marked_rows()


def _fetch_signal(signal_id: str) -> dict:
    rows = (
        _sync_client()
        .table("learning_signals")
        .select("*")
        .eq("signal_id", signal_id)
        .execute()
        .data
    )
    assert rows, f"learning_signals row {signal_id} did not land"
    return rows[0]


def _fetch_evaluation(evaluation_id: str) -> dict:
    rows = (
        _sync_client()
        .table("evaluation_results")
        .select("*")
        .eq("evaluation_id", evaluation_id)
        .execute()
        .data
    )
    assert rows, f"evaluation_results row {evaluation_id} did not land"
    return rows[0]


# ---------------------------------------------------------------------------
# The live path: record_learning_signal -> retrieved_chunks / retrieval_scores
# ---------------------------------------------------------------------------


def test_live_signal_writer_lands_chunks_as_real_jsonb(owned_rows):
    """The columns migration 022 created "for RAGAS evaluation" accept a write.

    Shape matters as much as value: #883 proved this writer family stored
    ``json.dumps``'d payloads as JSON **string scalars** (postgrest re-encodes),
    which reads back as ``str`` rather than ``list``/``dict``. Asserting the
    Python types through PostgREST is the check a mock cannot make.
    """
    from src.memory.procedural_memory import LearningSignalInput, record_learning_signal

    chunks = [
        {"content": "payer mix shifted to Tier 3", "source": "semantic", "hop": 1},
        {"content": "NRx held flat", "source": "episodic", "hop": 2},
    ]
    signal_id = owned_rows.signal(
        run_sync(
            record_learning_signal(
                signal=LearningSignalInput(
                    signal_type="rating",
                    signal_value=0.8,
                    signal_details={"test_marker": TEST_MARKER},
                    retrieved_chunks=chunks,
                    retrieval_scores=[0.91, 0.42],
                )
            )
        )
    )

    row = _fetch_signal(signal_id)
    assert isinstance(row["retrieved_chunks"], list), "stored as a JSON string scalar (#883)"
    assert row["retrieved_chunks"] == chunks
    assert row["retrieval_scores"] == [0.91, 0.42]
    assert isinstance(row["retrieved_chunks"][0], dict)


def test_live_signal_without_retrieval_keeps_the_schema_defaults(owned_rows):
    """None must not become NULL: ``retrieval_scores`` is read as an array and
    a NULL would break a consumer that a '[]' default would not."""
    from src.memory.procedural_memory import LearningSignalInput, record_learning_signal

    signal_id = owned_rows.signal(
        run_sync(
            record_learning_signal(
                signal=LearningSignalInput(
                    signal_type="rating",
                    signal_value=0.5,
                    signal_details={"test_marker": TEST_MARKER},
                )
            )
        )
    )

    row = _fetch_signal(signal_id)
    assert row["retrieved_chunks"] == []
    assert row["retrieval_scores"] == []


def test_a_zero_retrieval_turn_is_distinguishable_from_never_written(owned_rows):
    """The measured-empty case. Both rows read '[]' in the chunks column, so the
    distinction has to survive somewhere a reader can see it — here, the
    presence of the signal's own retrieval evidence in signal_details."""
    from src.memory.procedural_memory import LearningSignalInput, record_learning_signal

    signal_id = owned_rows.signal(
        run_sync(
            record_learning_signal(
                signal=LearningSignalInput(
                    signal_type="rating",
                    signal_value=0.5,
                    signal_details={"test_marker": TEST_MARKER, "evidence_count": 0},
                    retrieved_chunks=[],
                    retrieval_scores=[],
                )
            )
        )
    )

    row = _fetch_signal(signal_id)
    assert row["retrieved_chunks"] == []
    assert row["signal_details"]["evidence_count"] == 0


# ---------------------------------------------------------------------------
# The offline path: RubricNode + EvaluationResultsRepository
# ---------------------------------------------------------------------------


def _bundle(faithfulness=0.9, answer_relevancy=0.5):
    from src.agents.feedback_learner.ragas_scoring import RagasBundle

    return RagasBundle(
        scores={"faithfulness": faithfulness, "answer_relevancy": answer_relevancy},
        evaluation_model="gpt-4o",
    )


def _evaluation(weighted: float = 4.0):
    from src.agents.feedback_learner.evaluation.models import (
        CriterionScore,
        ImprovementDecision,
        RubricEvaluation,
    )

    return RubricEvaluation(
        weighted_score=weighted,
        criterion_scores=[
            CriterionScore(criterion="causal_validity", score=4.0, reasoning="r"),
            CriterionScore(criterion="actionability", score=4.0, reasoning="r"),
        ],
        decision=ImprovementDecision.ACCEPTABLE,
        overall_analysis=f"{TEST_MARKER} analysis",
    )


def _context(contexts=None):
    from src.agents.feedback_learner.evaluation.models import EvaluationContext

    return EvaluationContext(
        user_query=f"{TEST_MARKER}: why did TRx fall?",
        final_response=f"{TEST_MARKER}: access narrowed.",
        agent_names=["causal_impact"],
        session_id=str(uuid.uuid4()),
        retrieved_contexts=list(contexts or []),
    )


def _rubric_node(client):
    from unittest.mock import MagicMock

    from src.agents.feedback_learner.nodes.rubric_node import RubricNode

    # Real DB client, inert evaluator: the LLM judge is not what this file
    # tests and calling it would cost an Anthropic request per assertion.
    return RubricNode(evaluator=MagicMock(), db_client=client)


def _repo(client):
    from src.repositories.evaluation_results import get_evaluation_results_repository

    return get_evaluation_results_repository(supabase_client=client)


def test_rubric_row_lands_all_three_ragas_columns(owned_rows):
    """RED before #1489 in the sense that mattered: the seam existed but no
    caller passed a bundle, so these three columns were '{}'::jsonb / NULL on
    every one of 3,959 live rows."""
    from src.agents.feedback_learner.ragas_scoring import combined_score

    bundle = _bundle()
    signal_id = owned_rows.signal(
        _with_async_client(
            lambda client: _rubric_node(client)._store_evaluation(
                _evaluation(), _context(["ctx-one"]), ragas=bundle
            )
        )
    )
    assert signal_id, "the insert returned no signal_id — nothing landed"

    row = _fetch_signal(signal_id)
    assert row["ragas_scores"] == {"faithfulness": 0.9, "answer_relevancy": 0.5}
    assert row["ragas_weighted"] == pytest.approx(bundle.weighted)
    assert row["combined_score"] == pytest.approx(combined_score(bundle.weighted, 4.0))
    # The evidence the score was computed against, on the same row.
    assert row["retrieved_chunks"] == [{"content": "ctx-one"}]
    # Coverage survives the round trip: a NULL metric column cannot say whether
    # the judge failed (#1488) or was never asked (#1485).
    coverage = row["signal_details"]["ragas_coverage"]
    assert set(coverage["measured"]) == {"faithfulness", "answer_relevancy"}
    assert set(coverage["not_evaluated"]) == {
        "context_precision",
        "context_recall",
        "answer_correctness",
    }


def test_ragas_weighted_satisfies_the_column_check_constraint(owned_rows):
    """``ragas_weighted`` and ``combined_score`` are CHECK (>= 0 AND <= 1) and
    _store_evaluation swallows insert errors — a float that left the range
    would make the row silently vanish rather than fail. A perfect bundle is
    the value most likely to land on 1.0000000000000002."""
    perfect = _bundle(faithfulness=1.0, answer_relevancy=1.0)
    signal_id = owned_rows.signal(
        _with_async_client(
            lambda client: _rubric_node(client)._store_evaluation(
                _evaluation(weighted=5.0), _context(), ragas=perfect
            )
        )
    )
    assert signal_id, "a perfect bundle was rejected by a CHECK constraint"

    row = _fetch_signal(signal_id)
    assert row["ragas_weighted"] == pytest.approx(1.0)
    assert row["combined_score"] == pytest.approx(1.0)


def test_evaluation_results_row_lands_with_unmeasured_metrics_as_null(owned_rows):
    """The table had zero rows and zero Python writers until #1487; this is the
    first one written by the producer. An unmeasured metric must be SQL NULL,
    never 0.0 — ``v_ragas_performance_trends`` AVG()s these columns and skips
    NULLs, so a 0.0 would drag the average down as if it were a judgment."""
    row = owned_rows.evaluation(
        _with_async_client(
            lambda client: _repo(client).record_evaluation(
                query=f"{TEST_MARKER}: why did TRx fall?",
                response=f"{TEST_MARKER}: access narrowed.",
                ragas=_bundle(),
                retrieved_contexts=["ctx-one", "ctx-two"],
            )
        )
    )
    assert row, "record_evaluation returned no row"

    stored = _fetch_evaluation(row["evaluation_id"])
    assert stored["faithfulness"] == pytest.approx(0.9)
    assert stored["answer_relevancy"] == pytest.approx(0.5)
    for never_asked in ("context_precision", "context_recall", "answer_correctness"):
        assert stored[never_asked] is None, f"{never_asked} was written as a number"
    assert stored["retrieved_contexts"] == ["ctx-one", "ctx-two"]
    assert stored["context_count"] == 2
    assert stored["evaluation_model"] == "gpt-4o"
    # RAGAS-only row: no rubric half, so no rubric aggregate.
    assert stored["rubric_aggregate"] is None


def test_zero_retrieval_evaluation_stores_an_empty_context_list(owned_rows):
    """7 of 10 real replays retrieve nothing; the row must record that rather
    than refuse, or the honest measurement never reaches the table."""
    row = owned_rows.evaluation(
        _with_async_client(
            lambda client: _repo(client).record_evaluation(
                query=f"{TEST_MARKER}: unanswerable?",
                response=f"{TEST_MARKER}: I could not verify that.",
                ragas=_bundle(faithfulness=None, answer_relevancy=0.0),
                retrieved_contexts=[],
            )
        )
    )

    stored = _fetch_evaluation(row["evaluation_id"])
    assert stored["retrieved_contexts"] == []
    assert stored["context_count"] == 0
    assert stored["faithfulness"] is None
    assert stored["answer_relevancy"] == pytest.approx(0.0)


def test_both_halves_land_linked_by_the_real_foreign_key(owned_rows):
    """``evaluation_results.learning_signal_id`` is a real FK to
    ``learning_signals``. #1487 added the parameter for this and nothing ever
    passed it, so the constraint had never been exercised by a writer."""
    bundle = _bundle()
    context = _context(["ctx-one"])
    captured: dict = {}

    async def _both(client):
        signal_id = await _rubric_node(client)._store_evaluation(
            _evaluation(), context, ragas=bundle
        )
        captured["signal_id"] = signal_id
        assert signal_id, "the learning_signals half did not land"
        return await _repo(client).record_evaluation(
            query=context.user_query,
            response=context.final_response,
            ragas=bundle,
            rubric=_evaluation(),
            retrieved_contexts=list(context.retrieved_contexts),
            learning_signal_id=signal_id,
        )

    try:
        row = owned_rows.evaluation(_with_async_client(_both))
    finally:
        owned_rows.signal(captured.get("signal_id"))

    stored = _fetch_evaluation(row["evaluation_id"])
    assert stored["learning_signal_id"] == captured["signal_id"]
    assert stored["causal_validity"] == pytest.approx(4.0)
    assert stored["actionability"] == pytest.approx(4.0)
    # Criteria the rubric did not score have no value to report.
    assert stored["evidence_chain"] is None
    assert stored["rubric_aggregate"] == pytest.approx(4.0)


def test_a_nonexistent_learning_signal_fk_is_rejected_by_the_database(owned_rows):
    """Fail-loud all the way down: the repository does not swallow the insert
    error, so a broken link surfaces instead of writing an orphan row."""
    with pytest.raises(Exception) as exc:
        _with_async_client(
            lambda client: _repo(client).record_evaluation(
                query=f"{TEST_MARKER}: orphan",
                response=f"{TEST_MARKER}: orphan",
                ragas=_bundle(),
                learning_signal_id=str(uuid.uuid4()),
            )
        )
    message = str(exc.value).lower()
    assert "foreign key" in message or "23503" in message


def test_the_trend_view_now_returns_the_row_it_was_built_for(owned_rows):
    """``v_ragas_performance_trends`` was documented as "daily RAGAS metric
    trends" and returned nothing at all, because its only table had no writer.
    This is the end-to-end proof that the view is no longer structurally
    empty."""
    owned_rows.evaluation(
        _with_async_client(
            lambda client: _repo(client).record_evaluation(
                query=f"{TEST_MARKER}: trend probe",
                response=f"{TEST_MARKER}: trend probe answer",
                ragas=_bundle(),
                retrieved_contexts=["ctx"],
            )
        )
    )

    rows = _sync_client().table("v_ragas_performance_trends").select("*").execute().data
    assert rows, "the trend view is still empty after a row was written"
    today = rows[0]
    assert today["evaluation_count"] >= 1
    assert today["avg_faithfulness"] is not None
