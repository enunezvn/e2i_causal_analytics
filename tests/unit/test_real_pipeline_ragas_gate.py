"""Contract tests for the real-pipeline RAGAS gate (#1485).

The fixture eval (``scripts/run_ragas_eval.py``) never invokes the RAG
pipeline: ``run_evaluation()`` is called with no ``rag_pipeline``, so every
sample keeps its hardcoded ``answer`` and a ``retrieved_contexts`` that is
byte-identical to its reference ``contexts``. These tests pin the REAL path
instead — replay records carrying genuinely generated answers and genuinely
retrieved contexts, judged by the frozen gpt-4o judge, gated fail-closed.

The single most load-bearing behaviour here is
``test_adapter_never_emits_reference_contexts``: ``RAGASEvaluator.
evaluate_sample`` (src/rag/evaluation.py:982) silently substitutes
``sample.contexts`` whenever ``retrieved_contexts`` is empty, so a replay that
retrieved NOTHING would otherwise be scored against curated reference
evidence and score well. That is precisely the tautology #1485 is about, and
it must be impossible to reintroduce through this adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.rag.real_pipeline_eval import (
    REAL_PIPELINE_THRESHOLDS,
    build_samples_from_replay,
    check_real_pipeline_gates,
    contexts_from_evidence,
    summarize_retrieval,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    query_id: str = "q01",
    query: str = "Why did Kisqali TRx move in the Northeast?",
    response_text: str = "Kisqali TRx rose 12% on oncologist engagement.",
    contexts: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """One replay record in the shape ``replay_golden_set.py --record-out`` writes."""
    return {
        "query_id": query_id,
        "query": query,
        "model": "cognitive",
        "conversation_id": f"goldset-replay-20260805-{query_id}",
        "response_text": response_text,
        "contexts": ["Northeast TRx up 12% QoQ."] if contexts is None else contexts,
        "evidence_count": 1 if contexts is None else len(contexts),
        "hop_count": 2,
        "answer_chars": len(response_text),
        "latency_s": 17.4,
        "error": error,
    }


def _block(
    per_sample: Optional[List[Dict[str, Any]]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """A judge-output block whose aggregates genuinely describe its rows.

    Mirrors ``scripts/run_dspy_lane_ragas_judge.py``: ``n_samples`` counts all
    rows, ``n_faithfulness`` counts context-bearing rows, ``faithfulness``
    averages over context-bearing rows only, ``answer_relevancy`` over all.
    """
    if per_sample is None:
        per_sample = [
            {"query_id": f"q{i:02d}", "n_contexts": 2, "faithfulness": 0.8, "answer_relevancy": 0.5}
            for i in range(1, 11)
        ]
    ctx_rows = [r for r in per_sample if r.get("n_contexts", 0) > 0]

    def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    block = {
        "model": "cognitive",
        "n_samples": len(per_sample),
        "n_faithfulness": len(ctx_rows),
        "faithfulness": _mean(ctx_rows, "faithfulness"),
        "answer_relevancy": _mean(per_sample, "answer_relevancy"),
        "per_sample": per_sample,
    }
    block.update(overrides)
    return block


# Thresholds used by the logic tests. Deliberately explicit rather than the
# module constant: these tests pin gate BEHAVIOUR, and must not start passing
# or failing because a measured baseline was recalibrated.
_TEST_THRESHOLDS = {"faithfulness": 0.60, "answer_relevancy": 0.30}


# ---------------------------------------------------------------------------
# contexts_from_evidence
# ---------------------------------------------------------------------------


def test_contexts_from_evidence_extracts_content_field():
    evidence = [{"content": "Northeast TRx up 12%.", "source": "agent_activities"}]
    assert contexts_from_evidence(evidence) == ["Northeast TRx up 12%."]


def test_contexts_from_evidence_handles_non_dict_and_missing_content():
    """Never drop evidence silently: a shape we don't recognise still becomes text."""
    assert contexts_from_evidence(["raw string"]) == ["raw string"]
    out = contexts_from_evidence([{"source": "kg", "score": 0.4}])
    assert len(out) == 1 and "kg" in out[0]


def test_contexts_from_evidence_on_empty_is_empty():
    assert contexts_from_evidence(None) == []
    assert contexts_from_evidence([]) == []


# ---------------------------------------------------------------------------
# build_samples_from_replay
# ---------------------------------------------------------------------------


def test_adapter_emits_judge_sample_shape():
    """Samples must construct as EvaluationSample — the judge does exactly this."""
    from src.rag.evaluation import EvaluationSample

    samples = build_samples_from_replay([_record()])
    assert len(samples) == 1
    sample = EvaluationSample(**samples[0])
    assert sample.query == "Why did Kisqali TRx move in the Northeast?"
    assert sample.answer == "Kisqali TRx rose 12% on oncologist engagement."
    assert sample.retrieved_contexts == ["Northeast TRx up 12% QoQ."]


def test_adapter_never_emits_reference_contexts():
    """THE fail-open guard (src/rag/evaluation.py:982).

    ``evaluate_sample`` substitutes ``sample.contexts`` for an empty
    ``retrieved_contexts``. If the adapter ever populated ``contexts`` with
    curated reference evidence, a zero-retrieval replay would be judged
    against a perfect reference and score high — reintroducing the exact
    tautology #1485 exists to remove.
    """
    from src.rag.evaluation import EvaluationSample

    zero_retrieval = _record(contexts=[])
    samples = build_samples_from_replay([zero_retrieval])
    assert len(samples) == 1
    assert not samples[0].get("contexts"), (
        "adapter emitted reference contexts; a zero-retrieval replay would be "
        "silently scored against them by evaluate_sample()"
    )
    sample = EvaluationSample(**samples[0])
    assert sample.contexts == []
    assert sample.retrieved_contexts == []


def test_adapter_drops_errored_and_empty_answer_records():
    """Judging an error string scores the failure mode, not the answer."""
    records = [
        _record(query_id="q01"),
        _record(query_id="q02", error="CognitiveSearchError: boom"),
        _record(query_id="q03", response_text=""),
    ]
    samples = build_samples_from_replay(records)
    assert [s["metadata"]["query_id"] for s in samples] == ["q01"]


def test_adapter_preserves_multiple_real_contexts_in_order():
    ctxs = ["ctx one", "ctx two", "ctx three"]
    samples = build_samples_from_replay([_record(contexts=ctxs)])
    assert samples[0]["retrieved_contexts"] == ctxs


def test_adapter_does_not_fabricate_ground_truth():
    """faithfulness/answer_relevancy do not use it; inventing one is fabrication."""
    samples = build_samples_from_replay([_record()])
    assert samples[0]["ground_truth"] == ""


# ---------------------------------------------------------------------------
# summarize_retrieval
# ---------------------------------------------------------------------------


def test_summarize_retrieval_reports_hit_rate():
    """Retrieval-hit rate is the headline #1485 number (3/10 on 2026-07-18)."""
    records = [
        _record(query_id="q01", contexts=["a"]),
        _record(query_id="q02", contexts=[]),
        _record(query_id="q03", contexts=[]),
        _record(query_id="q04", error="boom", contexts=[]),
    ]
    summary = summarize_retrieval(records)
    assert summary["n_records"] == 4
    assert summary["n_errors"] == 1
    assert summary["n_with_contexts"] == 1
    assert summary["retrieval_hit_rate"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# check_real_pipeline_gates — happy path and threshold behaviour
# ---------------------------------------------------------------------------


def test_gates_pass_when_metrics_clear_thresholds():
    passed, failures = check_real_pipeline_gates(_block(), thresholds=_TEST_THRESHOLDS)
    assert passed, failures
    assert failures == []


def test_gates_fail_when_answer_relevancy_below_threshold():
    rows = [
        {"query_id": f"q{i:02d}", "n_contexts": 2, "faithfulness": 0.8, "answer_relevancy": 0.10}
        for i in range(1, 11)
    ]
    passed, failures = check_real_pipeline_gates(_block(rows), thresholds=_TEST_THRESHOLDS)
    assert not passed
    assert any("answer_relevancy" in f for f in failures)


# ---------------------------------------------------------------------------
# check_real_pipeline_gates — fail-closed behaviour
# ---------------------------------------------------------------------------


def test_gates_fail_closed_on_missing_metric():
    """A missing metric must BLOCK, never silently skip its gate."""
    block = _block()
    del block["answer_relevancy"]
    passed, failures = check_real_pipeline_gates(block, thresholds=_TEST_THRESHOLDS)
    assert not passed
    assert any("answer_relevancy" in f for f in failures)


def test_gates_fail_closed_on_none_metric():
    passed, failures = check_real_pipeline_gates(
        _block(faithfulness=None), thresholds=_TEST_THRESHOLDS
    )
    assert not passed
    assert any("faithfulness" in f for f in failures)


def test_gates_fail_closed_on_empty_judge_output():
    passed, failures = check_real_pipeline_gates({}, thresholds=_TEST_THRESHOLDS)
    assert not passed
    assert failures


def test_gates_fail_closed_on_aggregate_per_sample_mismatch():
    """A hand-edited or stale aggregate that no longer describes its rows."""
    block = _block()
    block["answer_relevancy"] = 0.99
    passed, failures = check_real_pipeline_gates(block, thresholds=_TEST_THRESHOLDS)
    assert not passed
    assert any("recompute" in f or "consistency" in f.lower() for f in failures)


def test_gates_fail_closed_on_scoreless_rows():
    """A context-bearing row the judge never scored is a hole in the mean."""
    rows = [
        {"query_id": f"q{i:02d}", "n_contexts": 2, "faithfulness": 0.8, "answer_relevancy": 0.5}
        for i in range(1, 10)
    ]
    rows.append({"query_id": "q10", "n_contexts": 2, "faithfulness": None, "answer_relevancy": 0.5})
    passed, failures = check_real_pipeline_gates(_block(rows), thresholds=_TEST_THRESHOLDS)
    assert not passed
    assert any("scoreless" in f.lower() or "finite" in f.lower() for f in failures)


def test_gates_fail_closed_below_faithfulness_denominator_floor():
    """A faithfulness mean over 1-2 context-bearing replays is noise, not signal."""
    rows = [
        {"query_id": "q01", "n_contexts": 2, "faithfulness": 0.9, "answer_relevancy": 0.5},
        {"query_id": "q02", "n_contexts": 2, "faithfulness": 0.9, "answer_relevancy": 0.5},
    ] + [
        {"query_id": f"q{i:02d}", "n_contexts": 0, "faithfulness": None, "answer_relevancy": 0.5}
        for i in range(3, 11)
    ]
    passed, failures = check_real_pipeline_gates(_block(rows), thresholds=_TEST_THRESHOLDS)
    assert not passed
    assert any(
        "faithfulness" in f and ("denominator" in f or "n_faithfulness" in f) for f in failures
    )


def test_gates_fail_closed_below_minimum_sample_count():
    """#1489 fixes the cadence at n>=10; a 3-sample run cannot carry the verdict."""
    rows = [
        {"query_id": f"q{i:02d}", "n_contexts": 2, "faithfulness": 0.8, "answer_relevancy": 0.5}
        for i in range(1, 4)
    ]
    passed, failures = check_real_pipeline_gates(
        _block(rows), thresholds=_TEST_THRESHOLDS, min_samples=10
    )
    assert not passed
    assert any("n_samples" in f or "samples" in f for f in failures)


# ---------------------------------------------------------------------------
# Metric split (#1485 step 2)
# ---------------------------------------------------------------------------


def test_only_real_path_metrics_are_gated():
    """context_precision/recall need a ground truth the replay does not fabricate.

    They must never gate the real path, even when a judge block carries them.
    """
    block = _block(context_precision=0.0, context_recall=0.0)
    passed, failures = check_real_pipeline_gates(block, thresholds=_TEST_THRESHOLDS)
    assert passed, failures
    assert not any("context_" in f for f in failures)


def test_default_thresholds_cover_exactly_the_real_path_metrics():
    assert set(REAL_PIPELINE_THRESHOLDS) == {"faithfulness", "answer_relevancy"}
    for name, value in REAL_PIPELINE_THRESHOLDS.items():
        assert 0.0 <= value <= 1.0, f"{name} threshold out of range: {value}"


def test_context_thresholds_are_rejected_rather_than_silently_ignored():
    """Passing a context metric in is a caller error, not a no-op."""
    with pytest.raises(ValueError, match="context"):
        check_real_pipeline_gates(_block(), thresholds={"context_precision": 0.8})
