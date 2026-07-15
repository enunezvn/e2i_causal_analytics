"""
Tests for `_finalize_training_signal` in `src/agents/feedback_learner/graph.py`.

Closes #424 (F-015): hardcoded `pattern_accuracy = 0.85` was a training-signal
anchor to a fabricated quality value. After the fix, `pattern_accuracy` is
`Optional[float]` and is `None` when no ground-truth validation label is
available — even when patterns are present. Downstream `compute_reward()` must
handle `None` gracefully by skipping the term and redistributing its weight.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from src.agents.feedback_learner.dspy_integration import FeedbackLearnerTrainingSignal
from src.agents.feedback_learner.graph import _finalize_training_signal


def _make_state(
    patterns: List[Dict[str, Any]] | None = None,
    feedback_items: List[Dict[str, Any]] | None = None,
    recommendations: List[Dict[str, Any]] | None = None,
    applied_updates: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Minimal-state factory for testing _finalize_training_signal."""
    return {
        "batch_id": "test_batch",
        "feedback_items": feedback_items or [],
        "detected_patterns": patterns or [],
        "learning_recommendations": recommendations or [],
        "proposed_updates": [],
        "applied_updates": applied_updates or [],
        "time_range_start": "2026-01-01T00:00:00Z",
        "time_range_end": "2026-01-07T00:00:00Z",
        "focus_agents": [],
        "total_latency_ms": 1000,
        "collection_latency_ms": 100,
        "analysis_latency_ms": 200,
        "extraction_latency_ms": 200,
        "update_latency_ms": 500,
        "model_used": "deterministic",
    }


def _run(coro):
    """Sync helper for awaiting coroutines in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestPatternAccuracyOptional:
    """F-015: pattern_accuracy must be Optional[float] = None, not hardcoded 0.85."""

    def test_pattern_accuracy_is_none_when_patterns_present_but_unvalidated(self) -> None:
        """
        Core anti-mocking assertion: when patterns are detected but there is no
        ground-truth label, training_signal.pattern_accuracy MUST be None.

        Before fix: pattern_accuracy = 0.85 (fabricated)
        After fix: pattern_accuracy = None (honest 'unavailable')
        """
        state = _make_state(
            patterns=[{"type": "missing_data", "severity": "high"}],
            feedback_items=[{"id": "fb1", "rating": 3}],
        )
        result = _run(_finalize_training_signal(state))

        signal = result["training_signal"]
        assert signal.pattern_accuracy is None, (
            "Pattern accuracy must be None (not fabricated 0.85) when no ground-truth "
            "labels are available, even when patterns are detected. "
            f"Got: {signal.pattern_accuracy}"
        )

    def test_pattern_accuracy_is_none_when_no_patterns(self) -> None:
        """When no patterns are detected and no labels are available, accuracy is None.

        Note: 'no patterns' alone doesn't mean accuracy=0.0 (false negatives may
        be possible). The honest answer is None until labels exist.
        """
        state = _make_state(patterns=[], feedback_items=[{"id": "fb1"}])
        result = _run(_finalize_training_signal(state))

        signal = result["training_signal"]
        assert signal.pattern_accuracy is None

    def test_no_hardcoded_085_anywhere_in_training_signal(self) -> None:
        """Regression pin: the literal 0.85 must never reappear via this path."""
        state = _make_state(
            patterns=[{"x": "y"}, {"a": "b"}, {"c": "d"}],
            feedback_items=[{"id": "fb1"}],
        )
        result = _run(_finalize_training_signal(state))

        signal = result["training_signal"]
        assert signal.pattern_accuracy != 0.85
        # Even if it weren't None, a literal 0.85 anchor would be a fabricated
        # quality signal; this pin catches accidental reintroduction.


class TestComputeRewardWithNoneAccuracy:
    """`compute_reward()` must gracefully handle pattern_accuracy=None."""

    def test_compute_reward_does_not_crash_on_none_accuracy(self) -> None:
        """If pattern_accuracy is None, compute_reward() must not raise TypeError."""
        signal = FeedbackLearnerTrainingSignal(
            batch_id="reward_none_test",
            feedback_count=10,
            time_range_start="2026-01-01T00:00:00Z",
            time_range_end="2026-01-02T00:00:00Z",
            patterns_detected=2,
            pattern_accuracy=None,
            recommendation_actionability=0.5,
            update_effectiveness=0.3,
            total_latency_ms=5000.0,
        )
        # Must not raise
        reward = signal.compute_reward()
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0

    def test_compute_reward_skips_accuracy_term_when_none(self) -> None:
        """When pattern_accuracy is None, the reward should NOT depend on its value.

        Compare two signals identical except for pattern_accuracy=None vs 0.0.
        With proper skip-and-redistribute behavior, the None case gets a higher
        reward (because we don't penalize for a missing measurement).
        """
        common: Dict[str, Any] = {
            "batch_id": "cmp",
            "feedback_count": 10,
            "time_range_start": "2026-01-01T00:00:00Z",
            "time_range_end": "2026-01-02T00:00:00Z",
            "patterns_detected": 1,
            "recommendation_actionability": 0.8,
            "update_effectiveness": 0.7,
            "total_latency_ms": 5000.0,
        }
        signal_none = FeedbackLearnerTrainingSignal(**common, pattern_accuracy=None)
        signal_zero = FeedbackLearnerTrainingSignal(**common, pattern_accuracy=0.0)

        reward_none = signal_none.compute_reward()
        reward_zero = signal_zero.compute_reward()

        # When accuracy is None, the term is omitted (skip-and-redistribute) so
        # the remaining present terms (which are good here) should yield a higher
        # reward than when we explicitly include a 0.0 penalty.
        assert reward_none > reward_zero, (
            f"Reward with accuracy=None ({reward_none:.4f}) should exceed reward with "
            f"accuracy=0.0 ({reward_zero:.4f}). The None case must skip the term "
            "rather than penalize."
        )

    def test_to_dict_serializes_none_accuracy_honestly(self) -> None:
        """to_dict() must serialize None as None — not silently rewrite to 0.0."""
        signal = FeedbackLearnerTrainingSignal(
            batch_id="dict_none_test",
            feedback_count=5,
            time_range_start="2026-01-01T00:00:00Z",
            time_range_end="2026-01-02T00:00:00Z",
            pattern_accuracy=None,
        )
        out = signal.to_dict()
        assert out["quality_metrics"]["pattern_accuracy"] is None, (
            "to_dict() must preserve None for unavailable measurements; rewriting "
            "to 0.0 would re-introduce the same anchor problem at the persistence "
            "layer."
        )


class TestNoPlaceholderCommentInGraph:
    """Regression pin: the placeholder comment must not return."""

    def test_graph_source_does_not_contain_placeholder_marker(self) -> None:
        """
        The literal '0.85 if patterns else 0.0  # Placeholder' line is a smell.
        Pin its absence so a future revert is caught at test-time.
        """
        from pathlib import Path

        graph_path = (
            Path(__file__).resolve().parents[4] / "src" / "agents" / "feedback_learner" / "graph.py"
        )
        source = graph_path.read_text()
        # The old line was:
        #   pattern_accuracy = 0.85 if patterns else 0.0  # Placeholder - would be validated
        # We forbid the substring "0.85 if patterns" specifically — that's the
        # anchor we're retiring.
        assert "0.85 if patterns" not in source, (
            "Detected re-introduction of hardcoded pattern_accuracy=0.85 anchor. See #424 / F-015."
        )


class TestUpdateEffectivenessWithheldApply:
    """auto_apply=False makes applied_updates structurally empty — 0/N would be
    a fabricated 'ineffective'; the honest value is None (unmeasurable)."""

    def test_none_when_apply_withheld_despite_wired_backend(self) -> None:
        state = _make_state()
        state["update_backend_wired"] = True
        state["proposed_updates"] = [
            {
                "update_id": "U1",
                "knowledge_type": "baseline",
                "key": "agent1",
                "old_value": None,
                "new_value": "v",
                "justification": "j",
                "effective_date": "2026-07-15T00:00:00+00:00",
            }
        ]
        state["applied_updates"] = []
        # auto_apply absent -> withheld (fail-closed default)
        result = _run(_finalize_training_signal(state))
        assert result["training_signal"].update_effectiveness is None

    def test_measured_when_auto_apply_true(self) -> None:
        state = _make_state()
        state["update_backend_wired"] = True
        state["auto_apply"] = True
        state["proposed_updates"] = [
            {
                "update_id": "U1",
                "knowledge_type": "baseline",
                "key": "agent1",
                "old_value": None,
                "new_value": "v",
                "justification": "j",
                "effective_date": "2026-07-15T00:00:00+00:00",
            }
        ]
        state["applied_updates"] = ["U1"]
        result = _run(_finalize_training_signal(state))
        assert result["training_signal"].update_effectiveness == 1.0
