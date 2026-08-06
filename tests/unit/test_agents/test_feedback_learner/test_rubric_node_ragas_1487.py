"""RubricNode._store_evaluation carries the RAGAS half too (#1487).

``database/ml/022`` added ``learning_signals.ragas_scores`` / ``ragas_weighted``
/ ``combined_score`` FOR this payload — #883 already remapped the rubric half
onto its purpose-built columns and left the RAGAS half at schema defaults
because nothing produced one. These tests pin both directions: with no bundle
the payload is byte-for-byte what it was (absence stays absence), and with one
the three columns land using the same blend the evaluation_results writer uses.
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
from src.agents.feedback_learner.ragas_scoring import RagasBundle, combined_score

RAGAS_COLUMNS = ("ragas_scores", "ragas_weighted", "combined_score")


def _client(returned_row=None, raises=False):
    client = MagicMock()
    execute = AsyncMock()
    if raises:
        execute.side_effect = RuntimeError("db down")
    else:
        execute.return_value = MagicMock(data=[returned_row] if returned_row is not None else None)
    client.table.return_value.insert.return_value.execute = execute
    return client


def _inserted(client) -> dict:
    return client.table.return_value.insert.call_args[0][0]


def _evaluation(weighted: float = 4.0) -> RubricEvaluation:
    return RubricEvaluation(
        weighted_score=weighted,
        criterion_scores=[
            CriterionScore(criterion="causal_validity", score=4.0, reasoning="r"),
            CriterionScore(criterion="actionability", score=4.0, reasoning="r"),
        ],
        decision=ImprovementDecision.ACCEPTABLE,
        overall_analysis="analysis",
    )


def _context() -> EvaluationContext:
    return EvaluationContext(
        user_query="Why did TRx fall?",
        final_response="Access narrowed.",
        agent_names=["causal_impact"],
    )


def _node(client) -> RubricNode:
    return RubricNode(evaluator=MagicMock(), db_client=client)


class TestWithoutABundle:
    async def test_ragas_columns_are_not_written_at_all(self):
        """Absence is represented by absence: the columns keep their schema
        defaults ('{}'::jsonb / NULL) rather than being set to a fabricated
        zero or an empty-but-present score."""
        client = _client({"signal_id": "sig-1"})
        await _node(client)._store_evaluation(_evaluation(), _context())
        row = _inserted(client)
        for column in RAGAS_COLUMNS:
            assert column not in row

    async def test_rubric_half_is_unchanged(self):
        client = _client({"signal_id": "sig-1"})
        await _node(client)._store_evaluation(_evaluation(), _context())
        row = _inserted(client)
        assert row["signal_type"] == "rating"
        assert row["rubric_total"] == 4.0
        assert row["signal_details"]["domain_signal"] == "rubric_evaluation"
        assert "ragas_coverage" not in row["signal_details"]


class TestWithABundle:
    async def test_measured_scores_land_in_ragas_scores(self):
        client = _client({"signal_id": "sig-1"})
        bundle = RagasBundle(scores={"faithfulness": 0.8, "answer_relevancy": 0.4})
        await _node(client)._store_evaluation(_evaluation(), _context(), ragas=bundle)
        assert _inserted(client)["ragas_scores"] == {
            "faithfulness": 0.8,
            "answer_relevancy": 0.4,
        }

    async def test_unmeasured_metric_is_absent_from_the_jsonb(self):
        """calculate_combined_score() reads these keys; a null-valued key and an
        absent key both COALESCE to 0 there, but only absence says the judge
        never scored it."""
        client = _client({"signal_id": "sig-1"})
        bundle = RagasBundle(
            scores={"faithfulness": 0.8, "answer_relevancy": None},
            unmeasured_metrics=["answer_relevancy"],
        )
        await _node(client)._store_evaluation(_evaluation(), _context(), ragas=bundle)
        assert _inserted(client)["ragas_scores"] == {"faithfulness": 0.8}

    async def test_ragas_weighted_is_the_shared_blend(self):
        client = _client({"signal_id": "sig-1"})
        bundle = RagasBundle(scores={"faithfulness": 0.8, "answer_relevancy": 0.4})
        await _node(client)._store_evaluation(_evaluation(), _context(), ragas=bundle)
        assert _inserted(client)["ragas_weighted"] == bundle.weighted

    async def test_combined_score_is_the_documented_blend(self):
        client = _client({"signal_id": "sig-1"})
        bundle = RagasBundle(scores={"faithfulness": 0.8, "answer_relevancy": 0.4})
        evaluation = _evaluation(weighted=4.0)
        await _node(client)._store_evaluation(evaluation, _context(), ragas=bundle)
        assert _inserted(client)["combined_score"] == combined_score(bundle.weighted, 4.0)

    async def test_coverage_is_recorded_in_signal_details(self):
        """learning_signals has no column for it, and a bare NULL cannot say
        whether the judge failed or was never asked."""
        client = _client({"signal_id": "sig-1"})
        bundle = RagasBundle(
            scores={"faithfulness": 0.8, "answer_relevancy": None},
            unmeasured_metrics=["answer_relevancy"],
            evaluation_model="gpt-4o",
        )
        await _node(client)._store_evaluation(_evaluation(), _context(), ragas=bundle)
        coverage = _inserted(client)["signal_details"]["ragas_coverage"]
        assert coverage["measured"] == ["faithfulness"]
        assert coverage["unmeasured"] == ["answer_relevancy"]
        assert coverage["evaluation_model"] == "gpt-4o"

    async def test_all_unmeasured_bundle_writes_no_score_at_all(self):
        """An all-NaN judge run must leave ragas_weighted and combined_score
        NULL rather than publishing a plausible number."""
        client = _client({"signal_id": "sig-1"})
        bundle = RagasBundle(
            scores={"faithfulness": None, "answer_relevancy": None},
            unmeasured_metrics=["faithfulness", "answer_relevancy"],
        )
        await _node(client)._store_evaluation(_evaluation(), _context(), ragas=bundle)
        row = _inserted(client)
        assert row["ragas_scores"] == {}
        assert row["ragas_weighted"] is None
        assert row["combined_score"] is None
        assert row["signal_details"]["ragas_coverage"]["measured"] == []

    async def test_partial_bundle_is_not_a_forty_percent_of_zero_blend(self):
        """The #1485 real-pipeline shape. calculate_combined_score() would score
        this row as if the three unmeasured metrics had been judged 0.0."""
        client = _client({"signal_id": "sig-1"})
        bundle = RagasBundle(scores={"faithfulness": 0.524, "answer_relevancy": 0.179})
        await _node(client)._store_evaluation(_evaluation(4.0), _context(), ragas=bundle)
        row = _inserted(client)
        sql_would_give = round((0.524 * 0.25 + 0.179 * 0.20) * 0.4 + 0.75 * 0.6, 4)
        assert row["combined_score"] != pytest.approx(sql_would_give)
        assert row["combined_score"] == combined_score(bundle.weighted, 4.0)


class TestSignalIdLinkage:
    async def test_returns_the_inserted_signal_id(self):
        """evaluation_results.learning_signal_id is a FK to this row; without
        the id back, the two halves cannot be linked."""
        client = _client({"signal_id": "11111111-1111-1111-1111-111111111111"})
        signal_id = await _node(client)._store_evaluation(_evaluation(), _context())
        assert signal_id == "11111111-1111-1111-1111-111111111111"

    async def test_returns_none_when_the_insert_returns_no_row(self):
        client = _client(None)
        assert await _node(client)._store_evaluation(_evaluation(), _context()) is None

    async def test_db_error_is_still_swallowed(self):
        """Unchanged: this runs inside the feedback-learner graph, where a
        failed signal write must not fail the whole cycle."""
        client = _client(raises=True)
        assert await _node(client)._store_evaluation(_evaluation(), _context()) is None

    async def test_no_client_returns_none(self):
        node = RubricNode(evaluator=MagicMock(), db_client=None)
        assert await node._store_evaluation(_evaluation(), _context()) is None
