"""Shard 04: faithful signal->Example conversion (F6)."""

from __future__ import annotations

import inspect as _inspect
import json

import pytest

from src.agents.feedback_learner.dspy_integration import (
    DSPY_AVAILABLE,
    FeedbackLearnerOptimizer,
    FeedbackLearnerTrainingSignal,
)


def _rich_signal() -> FeedbackLearnerTrainingSignal:
    return FeedbackLearnerTrainingSignal(
        batch_id="b1",
        feedback_count=12,
        time_range_start="t0",
        time_range_end="t1",
        patterns_detected=2,
        recommendations_generated=2,
        updates_applied=1,
        recommendation_actionability=0.8,
        update_effectiveness=0.9,
        total_latency_ms=1200.0,
        rubric_weighted_score=4.5,
        feedback_batch=[{"feedback_id": "f1", "feedback_type": "rating", "user_feedback": 2}],
        patterns=[
            {
                "pattern_type": "accuracy_issue",
                "severity": "high",
                "affected_agents": ["causal_impact"],
                "root_cause_hypothesis": "gap",
            }
        ],
        recommendations=[{"category": "prompt_update", "expected_impact": "higher accuracy"}],
        applied_updates=[{"key": "prompt.causal_impact", "new_value": "..."}],
        learning_summary="Detected an accuracy issue and recommended a prompt update.",
    )


def test_to_dict_carries_feedback_batch_and_patterns():
    d = _rich_signal().to_dict()
    assert d["input_context"].get("feedback_batch")  # non-empty
    assert d["output"].get("patterns")  # non-empty list
    assert d["output"].get("recommendations")
    assert d["output"].get("learning_summary")


def _healthy_signal() -> FeedbackLearnerTrainingSignal:
    """A cycle that processed real feedback and correctly found no patterns.

    #1668: this is the class the old builder discarded (it scores near 0, and
    the builder re-applied a ``reward < 0.5`` floor of its own), which is why the
    trainset was 100% positive and could only teach over-reporting.
    """
    sig = _rich_signal()
    sig.patterns_detected = 0
    sig.patterns = []
    sig.recommendations = []
    sig.recommendations_generated = 0
    return sig


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy required")
def test_conversion_is_balanced_and_content_bearing():
    """#1668 replaces the old ``8 rich signals -> 8 examples`` contract.

    That assertion encoded the defect: 8 identical POSITIVE cycles produced 8
    positive examples and the suite called it healthy. The builder now balances
    the label classes, so a mixed pool yields ``2 * min(pos, neg)`` interleaved.
    """
    opt = FeedbackLearnerOptimizer(optimizer_type="miprov2")
    signals = [_rich_signal().to_dict() for _ in range(8)] + [
        _healthy_signal().to_dict() for _ in range(8)
    ]

    pat = opt._signals_to_examples(signals, "pattern")
    rec = opt._signals_to_examples(signals, "recommendation")
    summ = opt._signals_to_examples(signals, "summary")
    upd = opt._signals_to_examples(signals, "update")

    assert len(pat) == 16
    assert sum(1 for e in pat if e.patterns) == 8
    assert sum(1 for e in pat if not e.patterns) == 8
    assert json.loads(pat[0].feedback_batch)  # non-empty input now
    # recommendation requires a non-empty detected_patterns INPUT, so only the
    # 8 pattern-bearing cycles are candidates — and they are single-class.
    assert rec == []
    assert json.loads(pat[0].feedback_batch)
    # summary is intentionally skipped (#1668: the label is an f-string template)
    assert summ == []
    # update is intentionally skipped (no paired current_knowledge stored)
    assert upd == []


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy required")
def test_single_class_pool_yields_no_examples():
    """8 positives with no negatives is not a trainset — it is the #1668 bias."""
    opt = FeedbackLearnerOptimizer(optimizer_type="miprov2")
    assert opt._signals_to_examples([_rich_signal().to_dict() for _ in range(8)], "pattern") == []


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy required")
def test_degenerate_signal_is_skipped():
    """A signal with no feedback_batch and no patterns must not become an example."""
    opt = FeedbackLearnerOptimizer(optimizer_type="miprov2")
    empty = FeedbackLearnerTrainingSignal(
        batch_id="e",
        feedback_count=0,
        time_range_start="t0",
        time_range_end="t1",
        total_latency_ms=10.0,
    ).to_dict()
    empty["reward"] = 0.9  # force past the reward gate
    assert opt._signals_to_examples([empty], "pattern") == []


def test_gepa_compile_receives_valset():
    """GEPA must validate on the held-out valset, not the trainset (F6)."""
    src = _inspect.getsource(FeedbackLearnerOptimizer._optimize_with_gepa)
    assert "optimizer.compile(module, trainset=trainset, valset=valset)" in src


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy required")
def test_summary_metric_scores_summary_outputs_not_recommendation_fields():
    """MIPROv2 summary phase must use a summary-aware metric, not recommendation_metric."""
    import dspy

    opt = FeedbackLearnerOptimizer(optimizer_type="miprov2")
    good = dspy.Prediction(
        summary="A detailed multi-sentence executive summary of the cycle outcomes here.",
        key_insights=["a", "b"],
        next_steps=["x"],
    )
    empty = dspy.Prediction(summary="", key_insights=[], next_steps=[])
    assert opt.summary_metric(None, good) >= 0.9
    assert opt.summary_metric(None, empty) == 0.0
    # recommendation_metric would be signal-deaf for a summary prediction.
    assert opt.recommendation_metric(None, good) == 0.0


@pytest.mark.asyncio
async def test_finalize_with_applied_updates_carries_full_dicts_not_crash():
    """Regression: applied_updates in state are update_id STRINGS; finalize must
    carry the full applied update dicts (from proposed_updates), not dict(str)."""
    from src.agents.feedback_learner.graph import _finalize_training_signal

    state = {
        "batch_id": "b",
        "time_range_start": "t0",
        "time_range_end": "t1",
        "focus_agents": [],
        "detected_patterns": [],
        "learning_recommendations": [],
        "feedback_items": [],
        "proposed_updates": [
            {"update_id": "U1", "key": "prompt.x", "new_value": "v"},
            {"update_id": "U2", "key": "prompt.y", "new_value": "w"},
        ],
        "applied_updates": ["U1"],  # <- STRINGS, the real shape
        "learning_summary": "did a thing",
        "collection_latency_ms": 0,
        "analysis_latency_ms": 0,
        "extraction_latency_ms": 0,
        "update_latency_ms": 0,
        "total_latency_ms": 10,
        "status": "updating",
    }
    out = await _finalize_training_signal(state)  # must not raise
    sig = out["training_signal"]
    assert sig.applied_updates == [{"update_id": "U1", "key": "prompt.x", "new_value": "v"}]
    # serializes cleanly for the summary-phase example
    d = sig.to_dict()
    assert d["output"]["applied_updates"][0]["update_id"] == "U1"
