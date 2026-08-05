"""EvaluationResultsRepository — the evaluation_results writer (#1487).

``evaluation_results`` (database/ml/022, :180) and the
``v_ragas_performance_trends`` view built on it had no Python writer at all.
These tests pin the row shape against the DDL's columns and CHECK constraints,
and pin the two things the table CANNOT express — a heuristic-scored row and an
unmeasured metric — to a refusal and a NULL respectively.
"""

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.feedback_learner.evaluation.models import (
    CriterionScore,
    ImprovementDecision,
    RubricEvaluation,
)
from src.agents.feedback_learner.ragas_scoring import RagasBundle
from src.repositories.evaluation_results import (
    EvaluationResultsRepository,
    get_evaluation_results_repository,
)

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "database"
    / "ml"
    / "022_self_improvement_tables.sql"
)


def _ddl_columns() -> set:
    sql = MIGRATION_PATH.read_text()
    start = sql.index("CREATE TABLE IF NOT EXISTS evaluation_results")
    body = sql[start : sql.index("\n);", start)]
    return {
        match.group(1)
        for line in body.splitlines()[1:]
        if (match := re.match(r"\s{4}(\w+)\s+\S", line))
    }


def _mock_client(returned_row=None, insert_raises=False):
    client = MagicMock()
    execute = AsyncMock()
    if insert_raises:
        execute.side_effect = RuntimeError("db down")
    else:
        execute.return_value = MagicMock(data=[returned_row or {"evaluation_id": "eval-1"}])
    client.table.return_value.insert.return_value.execute = execute
    return client


def _inserted(client) -> dict:
    return client.table.return_value.insert.call_args[0][0]


def _rubric(method: str = "llm", weighted: float = 4.0) -> RubricEvaluation:
    return RubricEvaluation(
        weighted_score=weighted,
        criterion_scores=[
            CriterionScore(criterion="causal_validity", score=4.0, reasoning="r"),
            CriterionScore(criterion="actionability", score=3.0, reasoning="r"),
            CriterionScore(criterion="evidence_chain", score=5.0, reasoning="r"),
            CriterionScore(criterion="regulatory_awareness", score=4.0, reasoning="r"),
            CriterionScore(criterion="uncertainty_communication", score=2.0, reasoning="r"),
        ],
        decision=ImprovementDecision.ACCEPTABLE,
        overall_analysis="analysis",
        evaluation_method=method,
    )


def _bundle(**overrides) -> RagasBundle:
    payload = {
        "scores": {
            "faithfulness": 0.80,
            "answer_relevancy": 0.40,
            "context_precision": 0.90,
            "context_recall": 0.70,
            "answer_correctness": 0.60,
        }
    }
    payload.update(overrides)
    return RagasBundle(**payload)


class TestRowShape:
    async def test_every_written_key_is_a_real_ddl_column(self):
        """A key that is not a column is a PGRST204 on every single write."""
        client = _mock_client()
        repo = EvaluationResultsRepository(client)
        await repo.record_evaluation(
            query="Why did TRx fall?",
            response="Because access narrowed.",
            ragas=_bundle(),
            rubric=_rubric(),
            retrieved_contexts=["chunk a", "chunk b"],
            ground_truth="access",
            learning_signal_id="11111111-1111-1111-1111-111111111111",
            cognitive_cycle_id="22222222-2222-2222-2222-222222222222",
        )
        client.table.assert_called_once_with("evaluation_results")
        assert set(_inserted(client)) <= _ddl_columns()

    async def test_ragas_metrics_land_in_their_own_columns(self):
        client = _mock_client()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q", response="a", ragas=_bundle()
        )
        row = _inserted(client)
        assert row["faithfulness"] == 0.80
        assert row["answer_relevancy"] == 0.40
        assert row["context_precision"] == 0.90
        assert row["context_recall"] == 0.70
        assert row["answer_correctness"] == 0.60

    async def test_ragas_aggregate_is_the_shared_blend(self):
        client = _mock_client()
        bundle = _bundle()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q", response="a", ragas=bundle
        )
        assert _inserted(client)["ragas_aggregate"] == bundle.weighted

    async def test_rubric_criteria_land_in_their_own_columns(self):
        client = _mock_client()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q", response="a", rubric=_rubric()
        )
        row = _inserted(client)
        assert row["causal_validity"] == 4.0
        assert row["actionability"] == 3.0
        assert row["evidence_chain"] == 5.0
        assert row["regulatory_awareness"] == 4.0
        assert row["uncertainty_communication"] == 2.0
        assert row["rubric_aggregate"] == 4.0

    async def test_context_count_is_derived_not_asserted(self):
        client = _mock_client()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q", response="a", ragas=_bundle(), retrieved_contexts=["x", "y", "z"]
        )
        row = _inserted(client)
        assert row["retrieved_contexts"] == ["x", "y", "z"]
        assert row["context_count"] == 3

    async def test_zero_retrieval_row_is_written_honestly(self):
        """#1485's binding constraint is retrieval: 10 of 15 turns retrieved
        nothing. Those rows must persist with an empty list and a 0 count, not
        be dropped."""
        client = _mock_client()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q", response="a", ragas=_bundle(), retrieved_contexts=[]
        )
        row = _inserted(client)
        assert row["retrieved_contexts"] == []
        assert row["context_count"] == 0

    async def test_returns_the_inserted_row(self):
        client = _mock_client(returned_row={"evaluation_id": "eval-42"})
        result = await EvaluationResultsRepository(client).record_evaluation(
            query="q", response="a", ragas=_bundle()
        )
        assert result == {"evaluation_id": "eval-42"}


class TestPartialBundles:
    async def test_unmeasured_metrics_are_null_never_zero(self):
        """The #1485 real-pipeline shape. A 0.0 here is indistinguishable from a
        judged zero and would drag v_ragas_performance_trends' average down."""
        client = _mock_client()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q",
            response="a",
            ragas=RagasBundle(scores={"faithfulness": 0.524, "answer_relevancy": 0.179}),
        )
        row = _inserted(client)
        assert row["faithfulness"] == 0.524
        assert row["context_precision"] is None
        assert row["context_recall"] is None
        assert row["answer_correctness"] is None

    async def test_all_unmeasured_bundle_writes_no_aggregate(self):
        client = _mock_client()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q",
            response="a",
            ragas=RagasBundle(
                scores={"faithfulness": None},
                unmeasured_metrics=["faithfulness"],
            ),
        )
        assert _inserted(client)["ragas_aggregate"] is None

    async def test_rubric_only_row_has_no_ragas_columns_set(self):
        client = _mock_client()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q", response="a", rubric=_rubric()
        )
        row = _inserted(client)
        assert row["ragas_aggregate"] is None
        assert row["faithfulness"] is None

    async def test_ragas_only_row_has_no_rubric_columns_set(self):
        client = _mock_client()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q", response="a", ragas=_bundle()
        )
        row = _inserted(client)
        assert row["rubric_aggregate"] is None
        assert row["causal_validity"] is None


class TestRefusals:
    async def test_row_with_neither_half_is_refused(self):
        """An evaluation_results row with no scores at all records nothing."""
        client = _mock_client()
        with pytest.raises(ValueError, match="neither"):
            await EvaluationResultsRepository(client).record_evaluation(query="q", response="a")

    async def test_heuristic_rubric_is_refused(self):
        """#471: the fallback emits neutral 3.0s indistinguishable from judged
        3.0s, and v_ragas_performance_trends averages rubric_aggregate too."""
        client = _mock_client()
        with pytest.raises(ValueError, match="heuristic"):
            await EvaluationResultsRepository(client).record_evaluation(
                query="q", response="a", rubric=_rubric(method="heuristic_fallback")
            )

    async def test_unknown_rubric_criterion_is_refused(self):
        """A criterion with no column cannot be persisted; dropping it silently
        would leave rubric_aggregate describing scores the row does not show."""
        rubric = _rubric()
        rubric = rubric.model_copy(
            update={
                "criterion_scores": [
                    *rubric.criterion_scores,
                    CriterionScore(criterion="brand_awareness", score=3.0, reasoning="r"),
                ]
            }
        )
        client = _mock_client()
        with pytest.raises(ValueError, match="brand_awareness"):
            await EvaluationResultsRepository(client).record_evaluation(
                query="q", response="a", rubric=rubric
            )

    async def test_empty_query_or_response_is_refused(self):
        """Both columns are NOT NULL, and an empty response was never judged."""
        client = _mock_client()
        with pytest.raises(ValueError, match="query"):
            await EvaluationResultsRepository(client).record_evaluation(
                query="  ", response="a", ragas=_bundle()
            )
        with pytest.raises(ValueError, match="response"):
            await EvaluationResultsRepository(client).record_evaluation(
                query="q", response="", ragas=_bundle()
            )

    async def test_insert_failure_raises_rather_than_fails_open(self):
        """This is a batch writer, not the chat path: a lost evaluation row is
        a silently shortened dataset, so it must never be swallowed."""
        client = _mock_client(insert_raises=True)
        with pytest.raises(RuntimeError, match="db down"):
            await EvaluationResultsRepository(client).record_evaluation(
                query="q", response="a", ragas=_bundle()
            )

    async def test_missing_client_raises(self):
        """Silently returning None would let a whole batch report success while
        writing nothing."""
        with pytest.raises(RuntimeError, match="client"):
            await EvaluationResultsRepository(None).record_evaluation(
                query="q", response="a", ragas=_bundle()
            )


class TestProvenance:
    async def test_evaluation_model_and_duration_are_persisted(self):
        client = _mock_client()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q",
            response="a",
            ragas=_bundle(evaluation_model="gpt-4o", evaluation_duration_ms=2650),
        )
        row = _inserted(client)
        assert row["evaluation_model"] == "gpt-4o"
        assert row["evaluation_duration_ms"] == 2650

    async def test_learning_signal_link_is_persisted(self):
        client = _mock_client()
        await EvaluationResultsRepository(client).record_evaluation(
            query="q",
            response="a",
            ragas=_bundle(),
            learning_signal_id="11111111-1111-1111-1111-111111111111",
        )
        assert _inserted(client)["learning_signal_id"] == "11111111-1111-1111-1111-111111111111"

    async def test_non_uuid_link_is_refused_not_silently_dropped(self):
        """learning_signal_id is a uuid FK: a free-form string 22P02s the whole
        insert, so the row would vanish."""
        client = _mock_client()
        with pytest.raises(ValueError, match="learning_signal_id"):
            await EvaluationResultsRepository(client).record_evaluation(
                query="q", response="a", ragas=_bundle(), learning_signal_id="sess-abc"
            )


class TestRepositoryWiring:
    def test_table_and_primary_key_match_the_ddl(self):
        repo = EvaluationResultsRepository(None)
        assert repo.table_name == "evaluation_results"
        assert repo.id_column == "evaluation_id"

    def test_factory_returns_a_wired_repository(self):
        client = MagicMock()
        assert get_evaluation_results_repository(client).client is client
