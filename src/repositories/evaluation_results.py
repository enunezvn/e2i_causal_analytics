"""Evaluation-results repository — the ``evaluation_results`` writer (#1487).

``database/ml/022_self_improvement_tables.sql`` created ``evaluation_results``
(:180) as the per-query-response detail table behind
``v_ragas_performance_trends`` (:431), and nothing in Python ever wrote to it.
This is that writer.

Fail-LOUD, unlike most repositories here
----------------------------------------
``ClassificationLogRepository.record_classification`` is deliberately fail-open
because it rides a live chat turn and must never delay or break one. This
writer is the opposite case: its producers are offline batch evaluations
(#1485's real-pipeline gate, #1489's loop close-out) that judge ~10-15 samples
per run at several seconds of gpt-4o time each. Swallowing a failed insert
there would silently shorten the dataset a threshold decision is then made
from, and ``MIN_REAL_PIPELINE_SAMPLES`` cannot see rows that were never
attempted. So every refusal and every insert error raises.

What the table cannot express, and what this writer does about it
-----------------------------------------------------------------
There is no column that could mark a row's scores heuristic rather than judged,
and ``v_ragas_performance_trends`` averages ``ragas_aggregate`` and
``rubric_aggregate`` with no way to filter. So heuristic-scored halves are
refused outright rather than persisted and caveated — the same call #1485's
``_ragas_heuristic_contamination_error`` makes one layer up, and the #471
lesson for the rubric's neutral-3.0 fallback.

An unmeasured metric persists as SQL NULL. That is lossless for the metric
itself but cannot distinguish "the judge tried and NaN'd" (#1488) from "this
producer never asks for that metric" (#1485). ``RagasBundle.coverage`` carries
that distinction; the caller that also writes a ``learning_signals`` row should
persist it there (``RubricNode._store_evaluation`` does), and migration 033
documents the limitation on the columns themselves.
"""

import logging
import uuid
from typing import Any, Dict, Optional, Sequence

from src.agents.feedback_learner.evaluation.models import RubricEvaluation
from src.agents.feedback_learner.ragas_scoring import RAGAS_METRIC_WEIGHTS, RagasBundle
from src.repositories.base import BaseRepository

logger = logging.getLogger(__name__)

# Rubric criteria that have a column in evaluation_results (database/ml/022
# :195-199). These ARE the five criteria in
# src/agents/feedback_learner/evaluation/criteria.py; a sixth added there
# without a migration has nowhere to land.
RUBRIC_CRITERION_COLUMNS = (
    "causal_validity",
    "actionability",
    "evidence_chain",
    "regulatory_awareness",
    "uncertainty_communication",
)

# src/agents/feedback_learner/evaluation/models.py: the rubric evaluator stamps
# this when its no-key / no-package / parse-failure path emits neutral 3.0
# scores. Note it is NOT the same string as RAGAS's "fallback_heuristic".
HEURISTIC_RUBRIC_METHOD = "heuristic_fallback"


def _require_uuid(value: Optional[str], field: str) -> Optional[str]:
    """Validate a uuid-typed FK before PostgREST 22P02s the whole insert."""
    if value is None:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{field}={value!r} is not a UUID; the column is uuid-typed and a "
            "free-form value would reject the entire row"
        ) from exc


class EvaluationResultsRepository(BaseRepository):
    """Writer for the ``evaluation_results`` RAGAS + rubric detail table."""

    table_name = "evaluation_results"
    model_class = None
    id_column = "evaluation_id"

    async def record_evaluation(
        self,
        *,
        query: str,
        response: str,
        ragas: Optional[RagasBundle] = None,
        rubric: Optional[RubricEvaluation] = None,
        retrieved_contexts: Optional[Sequence[str]] = None,
        ground_truth: Optional[str] = None,
        learning_signal_id: Optional[str] = None,
        cognitive_cycle_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Persist one judged query-response pair.

        Args:
            query: The user query that was answered. Required (NOT NULL).
            response: The answer that was judged. Required (NOT NULL).
            ragas: Judged RAGAS metrics, or None for a rubric-only row.
                Unmeasured metrics are written as NULL, never 0.0, and
                ``ragas_aggregate`` covers only what was measured.
            rubric: A completed rubric evaluation, or None for a RAGAS-only row.
            retrieved_contexts: The contexts the pipeline actually retrieved. An
                empty list is meaningful and is stored as such — a zero-retrieval
                turn is a real, and on the real pipeline a common, outcome.
            ground_truth: Reference answer, when the producer has one.
            learning_signal_id: FK to the ``learning_signals`` row carrying the
                same evaluation, when one was written.
            cognitive_cycle_id: FK to the originating cognitive cycle.

        Returns:
            The inserted row.

        Raises:
            ValueError: The row would carry no scores, a heuristic-scored half,
                a rubric criterion with no column, an empty query/response, or a
                non-UUID foreign key.
            RuntimeError: No client is wired.
            Exception: Whatever the insert raised — deliberately not swallowed.
        """
        if not self.client:
            raise RuntimeError(
                "EvaluationResultsRepository has no supabase client; returning "
                "quietly here would let a batch evaluation report success while "
                "persisting nothing"
            )
        if not query.strip():
            raise ValueError("query is empty; the column is NOT NULL")
        if not response.strip():
            raise ValueError(
                "response is empty; the column is NOT NULL and an empty answer was never judged"
            )
        # An all-unmeasured bundle is an OBJECT, not a score. Testing presence
        # alone let a row through whose twelve score columns were every one
        # NULL; v_ragas_performance_trends.evaluation_count is COUNT(*), so it
        # inflated the denominator a reader compares the averages against while
        # contributing to none of them. A rubric half rescues such a row — that
        # asymmetry is deliberate.
        if (ragas is None or ragas.weighted is None) and rubric is None:
            raise ValueError(
                "refusing a row with neither a RAGAS nor a rubric half — it would "
                "record a query-response pair as evaluated while carrying no score"
            )
        if rubric is not None and rubric.evaluation_method == HEURISTIC_RUBRIC_METHOD:
            raise ValueError(
                "refusing a heuristic-scored rubric (#471): the fallback emits "
                "neutral 3.0 scores that are indistinguishable from judged 3.0s, "
                "and v_ragas_performance_trends averages rubric_aggregate with no "
                "column to filter on"
            )

        contexts = list(retrieved_contexts or [])
        data: Dict[str, Any] = {
            "query": query,
            "response": response,
            "ground_truth": ground_truth,
            "retrieved_contexts": contexts,
            "context_count": len(contexts),
            "ragas_aggregate": ragas.weighted if ragas else None,
            "rubric_aggregate": rubric.weighted_score if rubric else None,
            "evaluation_model": ragas.evaluation_model if ragas else None,
            "evaluation_duration_ms": ragas.evaluation_duration_ms if ragas else None,
            "learning_signal_id": _require_uuid(learning_signal_id, "learning_signal_id"),
            "cognitive_cycle_id": _require_uuid(cognitive_cycle_id, "cognitive_cycle_id"),
        }

        measured = ragas.measured if ragas else {}
        for metric in RAGAS_METRIC_WEIGHTS:
            data[metric] = measured.get(metric)

        for column in RUBRIC_CRITERION_COLUMNS:
            data[column] = None
        if rubric is not None:
            for score in rubric.criterion_scores:
                if score.criterion not in RUBRIC_CRITERION_COLUMNS:
                    raise ValueError(
                        f"rubric criterion {score.criterion!r} has no column in "
                        "evaluation_results; persisting the row would leave "
                        "rubric_aggregate describing a score the row does not show. "
                        "Add the column in a migration first."
                    )
                data[score.criterion] = score.score

        result = await self.client.table(self.table_name).insert(data).execute()
        logger.debug(
            "Stored evaluation_results row (ragas=%s rubric=%s contexts=%d)",
            "yes" if ragas else "no",
            "yes" if rubric else "no",
            len(contexts),
        )
        return result.data[0] if result.data else None


def get_evaluation_results_repository(
    supabase_client=None,
) -> EvaluationResultsRepository:
    """Get an EvaluationResultsRepository instance."""
    return EvaluationResultsRepository(supabase_client)
