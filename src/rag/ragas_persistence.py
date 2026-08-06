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

# Run-level bookkeeping a judge block may report. Their presence is what makes
# a block reconcilable against its own rows; a block carrying none of them
# asserts nothing that could be contradicted.
_AGGREGATE_CLAIM_KEYS: Tuple[str, ...] = (
    "n_faithfulness",
    "faithfulness",
    "answer_relevancy",
)


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

    # A block whose own n_samples disagrees with its row count is PARTIAL: the
    # judge was killed or its output was cut mid-run. The rows that did arrive
    # look perfectly valid one at a time, and evaluation_results has no run id
    # or run-status column that could mark them as coming from an incomplete
    # run — so they would enter v_ragas_performance_trends as if the run had
    # finished. Persisting a bad MEASUREMENT is the point of this module;
    # persisting a partial RUN as a whole one is not. Absence of n_samples is
    # not a truncation claim (a caller may assemble a block by hand), so the
    # guard keys on disagreement.
    claimed = block.get("n_samples")
    if isinstance(claimed, int) and not isinstance(claimed, bool) and claimed != len(rows):
        raise RagasPersistenceError(
            f"judge block claims n_samples={claimed} but carries {len(rows)} per_sample "
            "rows — the output is truncated or partially merged, and nothing in "
            "evaluation_results could mark its rows as coming from an incomplete run"
        )

    # The row count can match while the NUMBERS do not. A block whose reported
    # aggregates no longer describe its own rows is stale, hand-edited or
    # partially merged, and reconciling them is exactly what the gate's
    # _ragas_consistency_error does — reused rather than reimplemented so the
    # two cannot drift. It lives HERE rather than only in the driver because
    # judged_turns and persist_judged_turns are the documented public flow, and
    # a guard the public flow walks past is not a guard (codex iter-4).
    #
    # Only a block that MAKES aggregate claims is reconciled: a caller
    # assembling one by hand declares none, so there is nothing to contradict.
    # Same principle as the n_samples check above — key on disagreement, not on
    # presence.
    if any(key in block for key in _AGGREGATE_CLAIM_KEYS):
        from src.optimization.dspy_lane_ab import _ragas_consistency_error

        inconsistent = _ragas_consistency_error(block)
        if inconsistent:
            raise RagasPersistenceError(
                f"judge block is internally inconsistent ({inconsistent}); its aggregates "
                "no longer describe its own rows, so the run is untrustworthy as a whole "
                "rather than merely low-scoring"
            )

    index = _index_records(records)
    turns: List[JudgedTurn] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RagasPersistenceError(f"per_sample row is {type(row).__name__}, expected object")
        raw_id = row.get("query_id")
        record = index.get(raw_id) if isinstance(raw_id, str) else None
        if not isinstance(raw_id, str) or record is None:
            raise RagasPersistenceError(
                f"judged sample {raw_id!r} has no replay record; the scores carry "
                "no query text of their own, so persisting one against a guessed "
                f"record would misattribute a judgment (records: {sorted(index)})"
            )
        query_id: str = raw_id

        contexts = tuple(str(c) for c in (record.get("contexts") or []))

        # An ABSENT key is not a None one. Migration 033 keeps them apart —
        # absent means "this producer never asks for that metric", None means
        # "the judge tried and could not score it" (#1488) — and only the
        # second is a malfunction worth investigating. The real-pipeline judge
        # sets both keys unconditionally, so a missing one means this block did
        # not come from it, and quietly reading it as None would record a judge
        # failure that never happened.
        missing = [metric for metric in JUDGED_METRICS if metric not in row]
        if missing:
            raise RagasPersistenceError(
                f"sample {query_id!r} is missing judged metric key(s) {missing}; the "
                "real-pipeline judge always emits them, so this block is malformed or "
                "from another producer. Reading an absent key as None would record a "
                "judge malfunction that never happened"
            )
        scores: Dict[str, Optional[float]] = {metric: row[metric] for metric in JUDGED_METRICS}
        # Faithfulness measures the answer AGAINST ITS CONTEXTS. On a turn that
        # retrieved none, ragas still emits a number (NaN coerced to 0.0, and
        # sometimes 1.0 for a vacuous answer) and the judge script excludes
        # exactly those rows from its own aggregate: "on a run that retrieved
        # no evidence the score is an artifact ... so it averages only over
        # samples with contexts". evaluation_results.faithfulness has no such
        # filter and v_ragas_performance_trends AVG()s the column, so
        # persisting the artifact would reintroduce one layer down what the
        # judge guards against upstream. MEASURED on the real #1489 close-out
        # run (7 of 10 turns zero-context): the judge reports 1.000 over its 3
        # context-bearing rows; averaging every row gives 0.286 — the view
        # would understate faithfulness by 0.71 and look perfectly healthy.
        # None here is #1488's "attempted, not scorable", which persists as SQL
        # NULL — never as a judged 0.0 (migration 033).
        if not contexts:
            scores["faithfulness"] = None

        # RagasBundle refuses any label containing "heuristic": a quota error
        # mid-run degrades a sample to word-overlap scoring while the process
        # still exits 0, and evaluation_results has no column that could mark
        # such a row. Re-raise as this module's error so the caller sees which
        # sample was contaminated.
        try:
            bundle = RagasBundle(
                scores=scores,
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
                # measurement (#1485). The RECORD is the authority on what was
                # retrieved, not the judge block's n_contexts bookkeeping.
                contexts=contexts,
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
                # ``_store_evaluation`` catches every insert error and returns
                # None — deliberate on the graph path, silent here. Counting
                # only successes let a --persist-signals run exit 0 with
                # learning_signals_written == 0 while writing evaluation_results
                # rows whose learning_signal_id is NULL, which afterwards reads
                # exactly like a deliberate RAGAS-only row. The caller asked for
                # both halves linked; half of that is a failure, and the
                # evaluation_results row is NOT written for this turn.
                if not signal_id:
                    raise RuntimeError(
                        "rubric_node.evaluate_and_store persisted no learning_signals "
                        "row (the insert was swallowed); refusing to write an "
                        "evaluation_results row that would look RAGAS-only afterwards"
                    )
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
