"""Signal rows shaped like the real table, for optimizer-gate tests (#1668).

The gate used to count rows, so ``[{"reward": 0.9}] * 25`` was a faithful
stand-in for "25 signals the trigger will count". It no longer is: the gate
counts the EXAMPLES the trainset builder will produce for the best-supplied
phase — ``2 * min(positives, negatives)``, derived from the same
``classify_signal_for_phase`` the builder uses — so a row with no
``input_context``/``output`` is not a signal the optimizer can train on and must
not be counted as one. These helpers build rows through the real
``FeedbackLearnerTrainingSignal.to_dict()``, which is what the persistence path
writes, so a test asserting "the gate sees N" is asserting it about rows the
builder would genuinely accept.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agents.feedback_learner.dspy_integration import FeedbackLearnerTrainingSignal

FEEDBACK = [{"feedback_id": "f1", "feedback_type": "rating", "user_feedback": 2}]
PATTERNS = [
    {
        "pattern_type": "accuracy_issue",
        "severity": "high",
        "affected_agents": ["causal_impact"],
        "root_cause_hypothesis": "retrieval gap",
    }
]
RECS = [{"category": "prompt_update", "expected_impact": "higher accuracy"}]


def signal_row(
    *,
    tag: str,
    patterns: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    feedback: List[Dict[str, Any]] = FEEDBACK,
    reward: float = 0.0,
    created_at: str = "2026-08-17T00:00:00+00:00",
) -> Dict[str, Any]:
    """One persisted-signal row, as ``build_signal_record`` would shape it."""
    signal = FeedbackLearnerTrainingSignal(
        batch_id=tag,
        feedback_count=len(feedback),
        time_range_start="t0",
        time_range_end="t1",
        patterns_detected=len(patterns),
        recommendations_generated=len(recommendations),
        feedback_batch=list(feedback),
        patterns=list(patterns),
        recommendations=list(recommendations),
        learning_summary="Learning cycle complete. Processed 1 feedback items.",
        total_latency_ms=1200.0,
    )
    row = signal.to_dict()
    row["reward"] = reward
    row["created_at"] = created_at
    return row


def positive(tag: str, *, reward: float = 0.9, **kw: Any) -> Dict[str, Any]:
    """A defect cycle: feedback in, patterns out. The scarce class in prod."""
    return signal_row(tag=tag, patterns=PATTERNS, recommendations=RECS, reward=reward, **kw)


def negative(tag: str, *, reward: float = 0.0, **kw: Any) -> Dict[str, Any]:
    """A healthy cycle: feedback in, correctly no patterns out."""
    return signal_row(tag=tag, patterns=[], recommendations=[], reward=reward, **kw)


def degenerate(tag: str, *, reward: float = 0.0, **kw: Any) -> Dict[str, Any]:
    """No feedback at all — 148 of the 223 real rows on 2026-08-17. Neither
    class: the INPUT
    is empty too, so the example would say "given nothing, emit nothing"."""
    return signal_row(tag=tag, patterns=[], recommendations=[], feedback=[], reward=reward, **kw)


def balanced_pool(k: int, *, reward: float = 0.9) -> List[Dict[str, Any]]:
    """``k`` positives + ``k`` negatives — a pool that builds a ``2k``-EXAMPLE trainset.

    ``reward`` applies to the positives only; a negative that cleared a reward
    floor has never been observed in 223 real rows (2026-08-17) and would
    misrepresent the
    population (a correct abstention scores near zero by construction).
    """
    return [positive(f"p{i}", reward=reward) for i in range(k)] + [
        negative(f"n{i}") for i in range(k)
    ]
