"""#1489 deferral 1: RubricNode's public producer seam + retrieved_chunks.

#1487 left ``_store_evaluation(ragas=)`` reachable only from inside the node
and documented the hook-up as #1489 work. Two gaps close here:

1. ``EvaluationContext.retrieved_contexts`` (models.py:50) has existed with
   ZERO producers and ZERO consumers repo-wide. It is the natural source for
   ``learning_signals.retrieved_chunks`` — measured non-default on 0 of 3,959
   live rows — so a signal row can carry the answer, the scores AND the
   evidence those scores were computed against.
2. ``evaluate_and_store`` is the public seam an offline producer calls.
   Reaching into ``_store_evaluation`` from another module would couple a
   batch script to a private method; the node returning the ``signal_id``
   (added by #1487 "so a caller can link an evaluation_results row to it")
   only helps if there is a public way to get one.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.feedback_learner.evaluation.models import (
    CriterionScore,
    EvaluationContext,
    ImprovementDecision,
    RubricEvaluation,
)
from src.agents.feedback_learner.nodes.rubric_node import RubricNode
from src.agents.feedback_learner.ragas_scoring import RagasBundle


def _client(returned_row=None):
    client = MagicMock()
    execute = AsyncMock()
    execute.return_value = MagicMock(data=[returned_row] if returned_row is not None else None)
    client.table.return_value.insert.return_value.execute = execute
    return client


def _inserted(client) -> dict:
    return client.table.return_value.insert.call_args[0][0]


def _evaluation(weighted: float = 4.0) -> RubricEvaluation:
    return RubricEvaluation(
        weighted_score=weighted,
        criterion_scores=[CriterionScore(criterion="causal_validity", score=4.0, reasoning="r")],
        decision=ImprovementDecision.ACCEPTABLE,
        overall_analysis="analysis",
    )


def _context(contexts=None) -> EvaluationContext:
    return EvaluationContext(
        user_query="Why did TRx fall?",
        final_response="Access narrowed.",
        agent_names=["causal_impact"],
        retrieved_contexts=list(contexts or []),
    )


class TestRetrievedChunks:
    async def test_contexts_land_in_retrieved_chunks(self):
        """RED: the column had no producer on this path, so a signal row could
        carry a faithfulness score with no record of what it was faithful to."""
        client = _client({"signal_id": "sig-1"})
        node = RubricNode(evaluator=MagicMock(), db_client=client)

        await node._store_evaluation(_evaluation(), _context(["payer mix", "NRx flat"]))

        assert _inserted(client)["retrieved_chunks"] == [
            {"content": "payer mix"},
            {"content": "NRx flat"},
        ]

    async def test_chunk_shape_matches_the_live_producer(self):
        """One column, one shape: the cognitive Reflector writes dicts keyed
        ``content``, so a reader does not have to branch on which producer
        wrote the row."""
        client = _client({"signal_id": "sig-1"})
        node = RubricNode(evaluator=MagicMock(), db_client=client)

        await node._store_evaluation(_evaluation(), _context(["only"]))

        chunk = _inserted(client)["retrieved_chunks"][0]
        assert set(chunk) == {"content"}

    async def test_oversized_context_is_capped_and_marked(self):
        from src.rag.retrieved_chunks import MAX_CHUNK_CONTENT_CHARS

        client = _client({"signal_id": "sig-1"})
        node = RubricNode(evaluator=MagicMock(), db_client=client)

        await node._store_evaluation(
            _evaluation(), _context(["y" * (MAX_CHUNK_CONTENT_CHARS + 10)])
        )

        chunk = _inserted(client)["retrieved_chunks"][0]
        assert len(chunk["content"]) == MAX_CHUNK_CONTENT_CHARS
        assert chunk["truncated"] is True

    async def test_no_contexts_writes_no_column_at_all(self):
        """Absence stays absence: the column keeps its '[]' schema default
        rather than being overwritten with an empty list that would look like
        a measured zero-retrieval turn."""
        client = _client({"signal_id": "sig-1"})
        node = RubricNode(evaluator=MagicMock(), db_client=client)

        await node._store_evaluation(_evaluation(), _context())

        assert "retrieved_chunks" not in _inserted(client)


class TestEvaluateAndStore:
    async def test_runs_the_evaluator_and_returns_the_signal_id(self):
        client = _client({"signal_id": "sig-42"})
        evaluator = MagicMock()
        evaluator.evaluate = AsyncMock(return_value=_evaluation())
        node = RubricNode(evaluator=evaluator, db_client=client)

        signal_id = await node.evaluate_and_store(context=_context(["ctx"]))

        assert signal_id == "sig-42"
        evaluator.evaluate.assert_awaited_once()

    async def test_passes_the_bundle_through_to_the_ragas_columns(self):
        client = _client({"signal_id": "sig-42"})
        evaluator = MagicMock()
        evaluator.evaluate = AsyncMock(return_value=_evaluation())
        node = RubricNode(evaluator=evaluator, db_client=client)
        bundle = RagasBundle(scores={"faithfulness": 0.8, "answer_relevancy": 0.4})

        await node.evaluate_and_store(context=_context(["ctx"]), ragas=bundle)

        row = _inserted(client)
        assert row["ragas_scores"] == {"faithfulness": 0.8, "answer_relevancy": 0.4}
        assert row["retrieved_chunks"] == [{"content": "ctx"}]
        # Both halves are real, so the documented blend is written.
        assert row["combined_score"] is not None

    async def test_a_heuristic_rubric_is_refused_before_it_is_persisted(self):
        """#471: the fallback emits neutral 3.0 scores indistinguishable from
        judged 3.0s, and learning_signals has no column that could mark them
        for the combined_score the RAGAS half would then blend into."""
        client = _client({"signal_id": "sig-42"})
        evaluation = _evaluation()
        evaluation = evaluation.model_copy(update={"evaluation_method": "heuristic_fallback"})
        evaluator = MagicMock()
        evaluator.evaluate = AsyncMock(return_value=evaluation)
        node = RubricNode(evaluator=evaluator, db_client=client)
        bundle = RagasBundle(scores={"faithfulness": 0.8})

        with pytest.raises(ValueError, match="heuristic"):
            await node.evaluate_and_store(context=_context(), ragas=bundle)

        client.table.return_value.insert.assert_not_called()

    async def test_without_a_client_it_reports_no_signal_rather_than_pretending(self):
        evaluator = MagicMock()
        evaluator.evaluate = AsyncMock(return_value=_evaluation())
        node = RubricNode(evaluator=evaluator, db_client=None)

        assert await node.evaluate_and_store(context=_context()) is None
