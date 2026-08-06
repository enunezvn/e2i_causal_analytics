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

        # Context-bearing, so faithfulness is a real measurement here (a
        # zero-retrieval row's faithfulness is an artifact — see
        # test_faithfulness_on_a_zero_context_row_is_unmeasured_not_a_score).
        bundle = judged_turns(_block([_row("q01", 0.9, 0.5)]), [_record("q01", ["ctx"])])[0].bundle
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

    def test_faithfulness_on_a_zero_context_row_is_unmeasured_not_a_score(self):
        """A zero-retrieval row's faithfulness is an ARTIFACT, not a judgment.

        ``run_dspy_lane_ragas_judge.py`` says so and acts on it: "on a run that
        retrieved no evidence the score is an artifact (NaN->0), so it averages
        only over samples with contexts" — its aggregate uses ``with_ctx``.
        ``evaluation_results.faithfulness`` has no such filter and
        ``v_ragas_performance_trends`` AVG()s the column directly, so persisting
        the artifact reintroduces one layer down exactly what the judge guards
        against upstream.

        MEASURED on the real #1489 close-out run (n=10, 7 zero-context): the
        judge reports faithfulness 1.000 over its 3 context-bearing rows, while
        averaging every row including the artifacts gives 0.286 — the view
        would understate faithfulness by 0.71 while looking perfectly healthy.
        """
        from src.rag.ragas_persistence import judged_turns

        turn = judged_turns(_block([_row("q01", 0.0, 0.35, n_contexts=0)]), [_record("q01")])[0]

        assert turn.bundle.measured == {"answer_relevancy": 0.35}
        assert turn.bundle.coverage["unmeasured"] == ["faithfulness"]
        # answer_relevancy compares the answer to the QUERY and needs no
        # contexts, so it stays a real measurement on a zero-retrieval turn.
        assert turn.bundle.weighted == pytest.approx(0.35)

    def test_faithfulness_on_a_context_bearing_row_is_kept(self):
        from src.rag.ragas_persistence import judged_turns

        turn = judged_turns(
            _block([_row("q01", 0.61, 0.35, n_contexts=2)]), [_record("q01", ["ctx"])]
        )[0]
        assert turn.bundle.measured["faithfulness"] == pytest.approx(0.61)

    def test_a_missing_n_contexts_falls_back_to_the_record(self):
        """Provenance, not the judge's bookkeeping, decides whether the turn
        retrieved anything — an older block without ``n_contexts`` must not
        silently promote artifacts back into the column."""
        from src.rag.ragas_persistence import judged_turns

        row = _row("q01", 0.9, 0.35)
        row.pop("n_contexts")
        turn = judged_turns(_block([row]), [_record("q01")])[0]

        assert turn.contexts == ()
        assert turn.bundle.coverage["unmeasured"] == ["faithfulness"]

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

    def test_a_metric_key_the_judge_never_emitted_raises(self):
        """An ABSENT key is not the same as a None one, and this module claims
        to keep them apart (migration 033: absent = "this producer never asks",
        None = "the judge tried and failed"). ``row.get(metric)`` collapsed the
        first into the second, so a malformed or foreign block would persist
        ``ragas_coverage.unmeasured=['answer_relevancy']`` — a judge
        malfunction that never happened.

        The real-pipeline judge sets BOTH keys unconditionally
        (run_dspy_lane_ragas_judge.py), so a missing one means the block did
        not come from it. Fail loud, as this function does for every other
        provenance failure. (codex iter-1 MED, reproduced.)
        """
        from src.rag.ragas_persistence import RagasPersistenceError, judged_turns

        row = _row("q01", 0.9, 0.5)
        del row["answer_relevancy"]

        with pytest.raises(RagasPersistenceError, match="answer_relevancy"):
            judged_turns(_block([row]), [_record("q01", ["ctx"])])

    def test_an_explicitly_null_metric_is_still_accepted(self):
        """The guard above must not swallow #1488's real case: the judge
        emitted the key and its value was None because it could not score it."""
        from src.rag.ragas_persistence import judged_turns

        turn = judged_turns(_block([_row("q01", None, 0.5)]), [_record("q01", ["ctx"])])[0]
        assert turn.bundle.coverage["unmeasured"] == ["faithfulness"]

    def test_a_truncated_block_raises(self):
        """A block whose own ``n_samples`` exceeds its row count is PARTIAL —
        the judge was killed or the output was cut mid-run. The rows that did
        arrive look perfectly valid individually, and ``evaluation_results``
        has no run id or run status column that could mark them as coming from
        an incomplete run, so they would enter the trend view as if the run had
        finished. Persisting a bad MEASUREMENT is right (that is the
        regression signal); persisting a partial RUN as a whole one is not.
        (codex iter-3 HIGH, reproduced: a block claiming n_samples=10 with 5
        rows produced 5 turns.)
        """
        from src.rag.ragas_persistence import RagasPersistenceError, judged_turns

        block = _block([_row(f"q{i:02d}", 0.9, 0.5) for i in range(1, 6)])
        block["n_samples"] = 10

        with pytest.raises(RagasPersistenceError, match="n_samples"):
            judged_turns(block, [_record(f"q{i:02d}", ["ctx"]) for i in range(1, 11)])

    def test_a_stale_aggregate_block_raises(self):
        """A block whose reported aggregates no longer describe its own rows is
        stale, hand-edited or partially merged.

        The row count can match while the numbers do not, so the n_samples
        guard alone is not enough. This check lived only in the script's
        ``persist_run``, which the documented module flow
        (``judged_turns`` -> ``persist_judged_turns``, both exported) walked
        straight past — reproduced: a block reporting answer_relevancy 0.99
        over rows recomputing to 0.50 wrote 2 rows through the module.
        (codex iter-4 HIGH.)
        """
        from src.rag.ragas_persistence import RagasPersistenceError, judged_turns

        rows = [_row("q01", 0.5, 0.0), _row("q02", 0.5, 1.0)]
        block = _block(rows)
        block.update({"n_faithfulness": 2, "faithfulness": 0.5, "answer_relevancy": 0.99})

        with pytest.raises(RagasPersistenceError, match="answer_relevancy"):
            judged_turns(block, [_record("q01", ["c"]), _record("q02", ["c"])])

    def test_a_consistent_full_block_is_accepted(self):
        """The guard must not refuse a real judge block. Aggregates computed
        the way the judge computes them (faithfulness over context-bearing
        rows, answer_relevancy over all rows) reconcile and pass."""
        from src.rag.ragas_persistence import judged_turns

        rows = [_row("q01", 0.5, 0.0), _row("q02", 0.5, 1.0)]
        block = _block(rows)
        block.update({"n_faithfulness": 2, "faithfulness": 0.5, "answer_relevancy": 0.5})

        assert len(judged_turns(block, [_record("q01", ["c"]), _record("q02", ["c"])])) == 2

    def test_a_block_making_no_aggregate_claims_is_not_reconciled(self):
        """A caller assembling a block by hand declares no aggregates, so there
        is nothing to contradict. The guard keys on DISAGREEMENT, not on
        presence — the same principle as the n_samples check."""
        from src.rag.ragas_persistence import judged_turns

        block = _block([_row("q01", 0.9, 0.5)])
        assert "faithfulness" not in block
        assert len(judged_turns(block, [_record("q01", ["ctx"])])) == 1

    def test_a_non_integer_n_samples_raises(self):
        """A present-but-unusable count is a malformed run-level claim, not an
        absent one. Skipping it because it failed an isinstance check let a
        block claiming ``n_samples="10"`` with one row through — the very
        partial-run case the count guard exists to catch, wearing the wrong
        type. (codex iter-5 MED, reproduced.)
        """
        from src.rag.ragas_persistence import RagasPersistenceError, judged_turns

        block = _block([_row("q01", 0.9, 0.5)])
        block["n_samples"] = "10"

        with pytest.raises(RagasPersistenceError, match="n_samples"):
            judged_turns(block, [_record("q01", ["ctx"])])

    def test_a_block_without_n_samples_is_still_accepted(self):
        """``n_samples`` is the judge's own bookkeeping; a caller assembling a
        block by hand need not supply it, and absence is not a truncation
        claim. The guard must key on DISAGREEMENT, not on presence."""
        from src.rag.ragas_persistence import judged_turns

        block = _block([_row("q01", 0.9, 0.5)])
        del block["n_samples"]

        assert len(judged_turns(block, [_record("q01", ["ctx"])])) == 1

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

    async def test_a_vanished_learning_signal_fails_the_turn(self):
        """``--persist-signals`` must not exit 0 having written no signals.

        ``_store_evaluation`` catches every insert error and returns None
        (deliberately, for the graph path), so ``evaluate_and_store`` can come
        back empty on a real DB failure. Counting only successes let the run
        report success with ``learning_signals_written == 0`` while writing
        evaluation_results rows whose ``learning_signal_id`` is NULL — which
        afterwards is indistinguishable from a deliberate RAGAS-only row. That
        is precisely the "the writer exists and nothing lands" failure #1487
        was filed about and #1489 exists to close. (codex iter-1 HIGH,
        reproduced: summary came back
        {"evaluation_results_written": 1, "learning_signals_written": 0}.)
        """
        from src.rag.ragas_persistence import (
            RagasPersistenceError,
            judged_turns,
            persist_judged_turns,
        )

        repo = _repo()
        node = MagicMock()
        node.evaluate_and_store = AsyncMock(return_value=None)
        turns = judged_turns(_block([_row("q01", 0.9, 0.5)]), [_record("q01", ["ctx"])])

        with pytest.raises(RagasPersistenceError, match="q01"):
            await persist_judged_turns(turns, eval_repo=repo, rubric_node=node)

        # The unlinked evaluation_results row is NOT written: a half-persisted
        # turn must not leave a row that reads as a complete RAGAS-only one.
        repo.record_evaluation.assert_not_awaited()

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
