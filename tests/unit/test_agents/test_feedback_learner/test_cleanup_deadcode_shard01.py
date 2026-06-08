"""Shard 01: dead-code cleanup regression guards (F7, F8)."""

from __future__ import annotations

import inspect

from src.agents.feedback_learner import graph as fbl_graph
from src.agents.feedback_learner.dspy_integration import FeedbackLearnerTrainingSignal


def test_finalize_has_no_bare_compute_discard_expressions():
    """F8: the finalize node must not contain the two orphaned min(...) statements."""
    src = inspect.getsource(fbl_graph._finalize_training_signal)
    # The dead lines computed efficiency/coverage targets and threw them away.
    assert '5000 / max(state.get("total_latency_ms"' not in src
    assert (
        "min(len(patterns) / max(len(feedback_items), 1), 1.0) if feedback_items else 0.0"
        not in src
    )


def test_compute_reward_still_responsive_after_cleanup():
    """Reward math is intact: efficiency/coverage are recomputed inside compute_reward()."""
    fast = FeedbackLearnerTrainingSignal(
        batch_id="b",
        feedback_count=20,
        time_range_start="t0",
        time_range_end="t1",
        patterns_detected=5,
        recommendations_generated=4,
        updates_applied=3,
        recommendation_actionability=0.8,
        update_effectiveness=0.9,
        total_latency_ms=1000.0,
    )
    slow = FeedbackLearnerTrainingSignal(
        batch_id="b",
        feedback_count=20,
        time_range_start="t0",
        time_range_end="t1",
        patterns_detected=5,
        recommendations_generated=4,
        updates_applied=3,
        recommendation_actionability=0.8,
        update_effectiveness=0.9,
        total_latency_ms=60000.0,  # much slower -> lower efficiency -> lower reward
    )
    assert fast.compute_reward() > slow.compute_reward()


def test_store_signals_locally_has_no_bare_expression_and_honest_log():
    """F7: fallback must not contain the discarded bare `entry["signal"]` expression."""
    from src.agents import tier2_signal_router as router_mod

    src = inspect.getsource(router_mod.Tier2SignalRouter._store_signals_locally)
    lines = [ln.strip() for ln in src.splitlines()]
    # The bare, discarded expression must be gone.
    assert 'entry["signal"]' not in lines, "bare discarded expression still present"
    # The honest behavior must be stated (signals are DROPPED, not retained).
    assert "DROPPED" in src or "dropped" in src


def test_deliver_fallback_log_does_not_falsely_claim_retention():
    """F7: the caller's ImportError log must not claim signals are retained locally."""
    from src.agents import tier2_signal_router as router_mod

    deliver_src = inspect.getsource(router_mod.Tier2SignalRouter._deliver_to_feedback_learner)
    assert "stored locally for later retrieval" not in deliver_src
