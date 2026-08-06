"""Land a real-pipeline RAGAS judge run in the self-improvement schema (#1489).

What was missing
----------------
#1487 built both writers — ``EvaluationResultsRepository.record_evaluation``
and the ``ragas=`` seam on ``RubricNode`` — and shipped with neither called.
Its own PR said so: "Nothing calls ``record_evaluation`` or passes ``ragas=``
yet — the producer hook-up … is #1489 work." Measured on the live DB
2026-08-06: ``evaluation_results`` 0 rows, ``learning_signals.ragas_scores``
non-default on 0 of 3,959 rows. This module is that hook-up.

Which producer, and why not the live turn
-----------------------------------------
#1487 names it: "RAGAS judging costs seconds of gpt-4o time per sample and
must never run inline (#1484); the producers are offline (#1485's batch
eval …)". So the input here is a judge run that ALREADY happened —
``scripts/run_real_pipeline_ragas.py`` hands over the block it just parsed.
Nothing in this module calls a RAGAS judge, and nothing it touches runs on a
chat turn.

The join is the provenance
--------------------------
``run_dspy_lane_ragas_judge.py`` emits per-sample rows carrying ``query_id``
and the scores and NOTHING ELSE — no query, no answer, no contexts. Those come
from the replay records the same run judged. A ``query_id`` that cannot be
joined, or that matches two records, is a provenance failure and raises: a
score persisted against a guessed record would attribute a judgment to an
answer that was never judged, and no column afterwards could tell.

Fail-loud, like the repository it feeds
---------------------------------------
Every refusal raises. A swallowed insert would silently shorten the dataset a
threshold decision is then made from, and ``MIN_REAL_PIPELINE_SAMPLES`` cannot
see rows that were never attempted. Failures are collected across all turns
before raising, so one bad row does not abandon the rest of an expensive judge
run and the error names every row that failed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agents.feedback_learner.ragas_scoring import RagasBundle

logger = logging.getLogger(__name__)

__all__ = [
    "JUDGED_METRICS",
    "JudgedTurn",
    "RagasPersistenceError",
    "judged_turns",
    "persist_judged_turns",
]

# The metrics the real-pipeline judge asks for. The other three RAGAS metrics
# need a ground-truth reference the replay deliberately does not fabricate, so
# they are ABSENT from the bundle (migration 033: "an absent key means the
# metric was not measured") rather than present-and-None, which would claim
# the judge tried them and failed.
JUDGED_METRICS: Tuple[str, ...] = ("faithfulness", "answer_relevancy")


class RagasPersistenceError(RuntimeError):
    """A judged run could not be persisted faithfully.

    Carries ``summary`` so a caller can report what DID land before the
    failure — a partial write must be visible, not inferred.
    """

    def __init__(self, message: str, summary: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.summary: Dict[str, Any] = summary or {}


@dataclass(frozen=True)
class JudgedTurn:
    """One replay turn and the RAGAS bundle the judge produced for it."""

    query_id: str
    query: str
    response: str
    contexts: Tuple[str, ...]
    bundle: RagasBundle
    conversation_id: Optional[str] = None


def _index_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise RagasPersistenceError(
                f"replay record has no usable query_id ({query_id!r}); the judge's "
                "per-sample rows can only be joined back to a record by that id"
            )
        if query_id in index:
            raise RagasPersistenceError(
                f"duplicate query_id {query_id!r} in the replay records — the join "
                "would attribute a judgment to an arbitrary one of two answers"
            )
        index[query_id] = record
    return index


def judged_turns(
    block: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    judge_model: Optional[str] = None,
) -> List[JudgedTurn]:
    """Join a judge block's per-sample scores back to what they judged.

    Args:
        block: The ``RESULTS_JSON`` payload from
            ``scripts/run_dspy_lane_ragas_judge.py``.
        records: The replay records that run was fed
            (``replay_golden_set.py --record-out``).
        judge_model: Judge model label, persisted for provenance.

    Raises:
        RagasPersistenceError: The block is malformed, a per-sample row cannot
            be joined to exactly one record, or a row was scored by the
            word-overlap fallback rather than by the judge.
    """
    rows = block.get("per_sample") if isinstance(block, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RagasPersistenceError(
            "judge block carries no per_sample rows — there is nothing whose "
            "provenance could be established, and an empty persist must not "
            "read as a successful one"
        )

    index = _index_records(records)
    turns: List[JudgedTurn] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RagasPersistenceError(f"per_sample row is {type(row).__name__}, expected object")
        query_id = row.get("query_id")
        record = index.get(query_id) if isinstance(query_id, str) else None
        if record is None:
            raise RagasPersistenceError(
                f"judged sample {query_id!r} has no replay record; the scores carry "
                "no query text of their own, so persisting one against a guessed "
                f"record would misattribute a judgment (records: {sorted(index)})"
            )

        # RagasBundle refuses any label containing "heuristic": a quota error
        # mid-run degrades a sample to word-overlap scoring while the process
        # still exits 0, and evaluation_results has no column that could mark
        # such a row. Re-raise as this module's error so the caller sees which
        # sample was contaminated.
        try:
            bundle = RagasBundle(
                scores={metric: row.get(metric) for metric in JUDGED_METRICS},
                evaluation_method=row.get("evaluation_method"),
                evaluation_model=judge_model,
            )
        except ValueError as exc:
            raise RagasPersistenceError(f"sample {query_id!r}: {exc}") from exc

        turns.append(
            JudgedTurn(
                query_id=query_id,
                query=str(record.get("query") or ""),
                response=str(record.get("response_text") or ""),
                # The replay's own contexts, never a reference stand-in: a
                # zero-retrieval turn recorded [] and that zero IS the
                # measurement (#1485).
                contexts=tuple(str(c) for c in (record.get("contexts") or [])),
                bundle=bundle,
                conversation_id=record.get("conversation_id"),
            )
        )
    return turns


async def persist_judged_turns(
    turns: Sequence[JudgedTurn],
    eval_repo: Any,
    rubric_node: Any = None,
) -> Dict[str, Any]:
    """Write judged turns to ``evaluation_results`` (+ ``learning_signals``).

    Args:
        turns: Output of :func:`judged_turns`.
        eval_repo: ``EvaluationResultsRepository``.
        rubric_node: When given, a ``RubricNode`` whose ``evaluate_and_store``
            is called FIRST for each turn, so the ``learning_signals`` row
            carries ``ragas_scores``/``ragas_weighted``/``combined_score`` and
            ``retrieved_chunks``, and the ``evaluation_results`` row links back
            to it. This runs the rubric judge (an Anthropic call per turn) —
            offline and opt-in, never on a chat turn.

    Returns:
        A summary: rows written per table, and the query_ids skipped.

    Raises:
        RagasPersistenceError: Any turn failed to persist. Raised after every
            turn has been attempted, naming all of them, with the partial
            counts on ``.summary``.
    """
    summary: Dict[str, Any] = {
        "turns": len(turns),
        "evaluation_results_written": 0,
        "learning_signals_written": 0,
        "skipped_unscored": [],
    }
    failures: List[str] = []

    for turn in turns:
        # record_evaluation refuses a row with no measured half — it would
        # inflate v_ragas_performance_trends.evaluation_count (a COUNT(*))
        # while contributing to none of the averages. Skip it HERE, counted by
        # id: dropping it silently would hide exactly the judge malfunction
        # #1488's fail-closed gate exists to surface, and that gate has
        # already run on this block by the time we are called.
        if turn.bundle.weighted is None:
            summary["skipped_unscored"].append(turn.query_id)
            logger.warning(
                "sample %s scored no RAGAS metric — not persisted (nothing to record)",
                turn.query_id,
            )
            continue

        try:
            signal_id: Optional[str] = None
            if rubric_node is not None:
                from src.agents.feedback_learner.evaluation import EvaluationContext

                signal_id = await rubric_node.evaluate_and_store(
                    context=EvaluationContext(
                        user_query=turn.query,
                        final_response=turn.response,
                        session_id=turn.conversation_id,
                        retrieved_contexts=list(turn.contexts),
                    ),
                    ragas=turn.bundle,
                )
                if signal_id:
                    summary["learning_signals_written"] += 1

            await eval_repo.record_evaluation(
                query=turn.query,
                response=turn.response,
                ragas=turn.bundle,
                retrieved_contexts=list(turn.contexts),
                learning_signal_id=signal_id,
            )
            summary["evaluation_results_written"] += 1
        except Exception as exc:  # noqa: BLE001 - collected and re-raised below
            failures.append(f"{turn.query_id}: {type(exc).__name__}: {exc}")

    if failures:
        raise RagasPersistenceError(
            "failed to persist "
            f"{len(failures)} of {len(turns)} judged turns: " + "; ".join(failures),
            summary=summary,
        )
    return summary
