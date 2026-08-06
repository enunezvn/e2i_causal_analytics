"""#1489 deferral 1: the offline producer that lands judged RAGAS in the DB.

#1487 built both writers — ``EvaluationResultsRepository.record_evaluation``
and the ``ragas=`` seam on ``RubricNode._store_evaluation`` — and shipped with
neither called ("Nothing calls record_evaluation or passes ragas= yet", PR
#1493). Measured on the live DB 2026-08-06: ``evaluation_results`` 0 rows,
``learning_signals.ragas_scores`` non-default on 0 of 3,959 rows.

The producer #1487 names is the offline batch eval, NOT the live turn: "RAGAS
judging costs seconds of gpt-4o time per sample and must never run inline
(#1484); the producers are offline (#1485's batch eval …), and hooking one up
is #1489." So this wires ``scripts/run_real_pipeline_ragas.py``: the judge run
that already happened hands its per-sample scores to the writers. No new
gpt-4o call, and nothing on a live chat turn.

The join is the provenance: the judge block's ``per_sample`` rows carry only
``query_id`` and the scores, so the query/response/contexts come from the
replay records the same run judged. A row that cannot be joined is a
provenance failure and raises rather than persisting a pair that might not be
the pair that was scored.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


def _record(query_id: str, contexts: Optional[List[str]] = None, **over: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "query_id": query_id,
        "query": f"question {query_id}",
        "response_text": f"answer {query_id}",
        "contexts": list(contexts or []),
        "conversation_id": f"goldset-replay-20260806-{query_id}",
        "error": None,
    }
    base.update(over)
    return base


def _block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"model": "real-pipeline:cognitive", "n_samples": len(rows), "per_sample": rows}


def _row(
    query_id: str, faith: Optional[float], rel: Optional[float], **over: Any
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "query_id": query_id,
        "n_contexts": 1,
        "faithfulness": faith,
        "answer_relevancy": rel,
        "evaluation_method": None,
    }
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# judged_turns: joining the judge block back to what it judged
# ---------------------------------------------------------------------------


class TestJudgedTurns:
    def test_joins_scores_to_the_replay_record_that_was_judged(self):
        from src.rag.ragas_persistence import judged_turns

        turns = judged_turns(
            _block([_row("q01", 0.9, 0.5)]),
            [_record("q01", ["payer mix shifted"])],
        )

        assert len(turns) == 1
        turn = turns[0]
        assert turn.query == "question q01"
        assert turn.response == "answer q01"
        assert turn.contexts == ("payer mix shifted",)
        assert turn.bundle.measured == {"faithfulness": 0.9, "answer_relevancy": 0.5}

    def test_reports_only_the_two_metrics_the_real_path_can_measure(self):
        """context_precision/recall/answer_correctness need a ground truth the
        replay refuses to fabricate, so they are ABSENT (never asked), not
        None (asked and failed) — migration 033 documents the difference."""
        from src.rag.ragas_persistence import judged_turns

        bundle = judged_turns(_block([_row("q01", 0.9, 0.5)]), [_record("q01")])[0].bundle
        coverage = bundle.coverage

        assert set(coverage["measured"]) == {"faithfulness", "answer_relevancy"}
        assert coverage["unmeasured"] == []
        assert set(coverage["not_evaluated"]) == {
            "context_precision",
            "context_recall",
            "answer_correctness",
        }

    def test_a_metric_the_judge_could_not_score_is_unmeasured_not_zero(self):
        """#1488's vocabulary: a NaN'd metric arrives as None and must persist
        as SQL NULL, never as a judged 0.0."""
        from src.rag.ragas_persistence import judged_turns

        bundle = judged_turns(_block([_row("q01", None, 0.5)]), [_record("q01")])[0].bundle

        assert bundle.measured == {"answer_relevancy": 0.5}
        assert bundle.coverage["unmeasured"] == ["faithfulness"]

    def test_zero_retrieval_turn_keeps_its_empty_contexts(self):
        """7 of 10 turns retrieve nothing. [] is the measurement; substituting
        anything would score the answer against evidence it never saw."""
        from src.rag.ragas_persistence import judged_turns

        turn = judged_turns(_block([_row("q01", None, 0.0, n_contexts=0)]), [_record("q01")])[0]
        assert turn.contexts == ()

    def test_unjoinable_per_sample_row_raises(self):
        """The scores carry no query text of their own; persisting one against
        a guessed record would attribute a judgment to the wrong answer."""
        from src.rag.ragas_persistence import RagasPersistenceError, judged_turns

        with pytest.raises(RagasPersistenceError, match="q99"):
            judged_turns(_block([_row("q99", 0.9, 0.5)]), [_record("q01")])

    def test_duplicate_query_ids_in_records_raise(self):
        from src.rag.ragas_persistence import RagasPersistenceError, judged_turns

        with pytest.raises(RagasPersistenceError, match="duplicate"):
            judged_turns(_block([_row("q01", 0.9, 0.5)]), [_record("q01"), _record("q01")])

    def test_heuristic_scored_row_raises_rather_than_persisting(self):
        """A quota error mid-run silently degrades a sample to word-overlap
        heuristics while the process still exits 0. evaluation_results has no
        column that could mark such a row and v_ragas_performance_trends would
        average it in as a judgment."""
        from src.rag.ragas_persistence import RagasPersistenceError, judged_turns

        with pytest.raises(RagasPersistenceError, match="heuristic"):
            judged_turns(
                _block([_row("q01", 0.9, 0.5, evaluation_method="fallback_heuristic")]),
                [_record("q01")],
            )

    def test_malformed_block_raises(self):
        from src.rag.ragas_persistence import RagasPersistenceError, judged_turns

        with pytest.raises(RagasPersistenceError):
            judged_turns({"n_samples": 1}, [_record("q01")])

    def test_provenance_fields_ride_along(self):
        from src.rag.ragas_persistence import judged_turns

        turn = judged_turns(
            _block([_row("q01", 0.9, 0.5)]),
            [_record("q01")],
            judge_model="gpt-4o",
        )[0]
        assert turn.bundle.evaluation_model == "gpt-4o"
        assert turn.conversation_id == "goldset-replay-20260806-q01"


# ---------------------------------------------------------------------------
# persist_judged_turns: the writers, finally called
# ---------------------------------------------------------------------------


def _repo() -> MagicMock:
    repo = MagicMock()
    repo.record_evaluation = AsyncMock(return_value={"evaluation_id": "eval-1"})
    return repo


class TestPersistJudgedTurns:
    async def test_writes_one_evaluation_results_row_per_judged_turn(self):
        from src.rag.ragas_persistence import judged_turns, persist_judged_turns

        repo = _repo()
        turns = judged_turns(
            _block([_row("q01", 0.9, 0.5), _row("q02", 0.2, 0.1)]),
            [_record("q01", ["ctx-a"]), _record("q02")],
        )
        summary = await persist_judged_turns(turns, eval_repo=repo)

        assert summary["evaluation_results_written"] == 2
        first = repo.record_evaluation.await_args_list[0].kwargs
        assert first["query"] == "question q01"
        assert first["response"] == "answer q01"
        assert first["retrieved_contexts"] == ["ctx-a"]
        assert first["ragas"].measured == {"faithfulness": 0.9, "answer_relevancy": 0.5}

    async def test_ragas_only_rows_carry_no_learning_signal_link(self):
        from src.rag.ragas_persistence import judged_turns, persist_judged_turns

        repo = _repo()
        turns = judged_turns(_block([_row("q01", 0.9, 0.5)]), [_record("q01")])
        await persist_judged_turns(turns, eval_repo=repo)

        assert repo.record_evaluation.await_args.kwargs["learning_signal_id"] is None

    async def test_fully_unmeasured_turn_is_skipped_and_counted_not_dropped(self):
        """record_evaluation refuses a row with no measured half (it would
        inflate v_ragas_performance_trends.evaluation_count, a COUNT(*), while
        contributing to none of the averages). Skipping it silently would hide
        exactly the judge malfunction #1488 exists to surface."""
        from src.rag.ragas_persistence import judged_turns, persist_judged_turns

        repo = _repo()
        turns = judged_turns(
            _block([_row("q01", 0.9, 0.5), _row("q02", None, None)]),
            [_record("q01"), _record("q02")],
        )
        summary = await persist_judged_turns(turns, eval_repo=repo)

        assert summary["evaluation_results_written"] == 1
        assert summary["skipped_unscored"] == ["q02"]

    async def test_a_write_failure_is_raised_not_swallowed(self):
        """The repository is deliberately fail-loud: a swallowed insert would
        silently shorten the dataset a threshold decision is made from."""
        from src.rag.ragas_persistence import (
            RagasPersistenceError,
            judged_turns,
            persist_judged_turns,
        )

        repo = _repo()
        repo.record_evaluation.side_effect = RuntimeError("db down")
        turns = judged_turns(_block([_row("q01", 0.9, 0.5)]), [_record("q01")])

        with pytest.raises(RagasPersistenceError, match="db down"):
            await persist_judged_turns(turns, eval_repo=repo)

    async def test_every_turn_is_attempted_before_failures_are_raised(self):
        """One bad row must not abandon the rest of an expensive judge run,
        and the raised error must name every row that failed."""
        from src.rag.ragas_persistence import (
            RagasPersistenceError,
            judged_turns,
            persist_judged_turns,
        )

        repo = _repo()
        repo.record_evaluation.side_effect = [RuntimeError("boom"), {"evaluation_id": "eval-2"}]
        turns = judged_turns(
            _block([_row("q01", 0.9, 0.5), _row("q02", 0.8, 0.4)]),
            [_record("q01"), _record("q02")],
        )

        with pytest.raises(RagasPersistenceError) as exc:
            await persist_judged_turns(turns, eval_repo=repo)

        assert repo.record_evaluation.await_count == 2
        assert "q01" in str(exc.value)
        assert exc.value.summary["evaluation_results_written"] == 1

    async def test_with_a_rubric_node_both_halves_land_and_are_linked(self):
        """The learning_signals row carries ragas_scores + retrieved_chunks and
        the evaluation_results row points back at it — the FK #1487 added for
        exactly this ("the caller that also writes a learning_signals row")."""
        from src.rag.ragas_persistence import judged_turns, persist_judged_turns

        repo = _repo()
        node = MagicMock()
        node.evaluate_and_store = AsyncMock(return_value="sig-42")
        turns = judged_turns(_block([_row("q01", 0.9, 0.5)]), [_record("q01", ["ctx-a"])])

        summary = await persist_judged_turns(turns, eval_repo=repo, rubric_node=node)

        assert summary["learning_signals_written"] == 1
        context = node.evaluate_and_store.await_args.kwargs["context"]
        assert context.user_query == "question q01"
        assert context.final_response == "answer q01"
        assert context.retrieved_contexts == ["ctx-a"]
        assert node.evaluate_and_store.await_args.kwargs["ragas"].measured["faithfulness"] == 0.9
        assert repo.record_evaluation.await_args.kwargs["learning_signal_id"] == "sig-42"
