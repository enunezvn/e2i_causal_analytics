"""#1668: the GEPA trainset must not be selected on defect yield.

``_signals_to_examples`` carried its OWN floor::

    if signal.get("reward", 0) < 0.5:  # only successful cycles
        continue

Two things are wrong with it.

1. **It makes the caller's ``min_reward`` inoperative.** The floor is applied a
   second time, downstream of the fetch, so ``run_feedback_learner_optimization(
   min_reward=0.0)`` — a real call shape, used by
   ``tests/integration/test_dspy_loop_e2e_bounded.py`` — still yields only the
   ``reward >= 0.5`` rows.

2. **It is exactly the sampling bias the issue is about.** For the pattern
   phase the label IS the patterns the cycle found, so selecting on reward
   selects on *having found patterns*. Measured over 220 real prod signals: the
   surviving trainset is 8 examples, **100% non-empty label**. The pattern
   detector would never once see a healthy batch labelled "no patterns", and
   under a gold-aware metric that trainset can only teach over-reporting.

Deleting the floor is NOT the fix either. Measured on the same 220 rows, the
unfiltered population is:

    feedback_batch  patterns   n
    absent          absent     148   <- degenerate: EMPTY INPUT, nothing to learn
    present         absent      57   <- the informative NEGATIVES
    present         present     15   <- the POSITIVES

So the informative pool is **72**, not 220, and it is 79.2% / 20.8%. Feeding it
raw would swap one bias for its mirror: under a symmetric metric a
never-report prompt would score 57/72 = 0.79 while a always-report prompt scores
15/72 = 0.21, and GEPA maximises the mean.

The builder therefore (a) requires the phase's INPUT to be non-empty, and
(b) balances the two label classes at ``k = min(n_pos, n_neg)`` and interleaves
them, so the 80/20 ``trainset``/``valset`` prefix split in ``_optimize_with_gepa``
stays balanced too. Balancing makes GEPA's mean score a *balanced* accuracy —
equal weight to a missed detection and to a false positive — which is the only
defensible target when the metric treats the two symmetrically.

No mocks: real signals through the real builder.
"""

from __future__ import annotations

import json

import pytest

from src.agents.feedback_learner.dspy_integration import (
    DSPY_AVAILABLE,
    FeedbackLearnerOptimizer,
    FeedbackLearnerTrainingSignal,
)

pytestmark = pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy required")

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


def _signal(*, tag: str, patterns, recommendations, feedback=FEEDBACK, reward=0.0) -> dict:
    sig = FeedbackLearnerTrainingSignal(
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
    row = sig.to_dict()
    row["reward"] = reward  # the stored value the old floor read
    return row


def _positive(tag: str, reward: float = 0.9) -> dict:
    return _signal(tag=tag, patterns=PATTERNS, recommendations=RECS, reward=reward)


def _negative(tag: str, reward: float = 0.0) -> dict:
    """A healthy batch: feedback WAS processed, correctly yielding no patterns."""
    return _signal(tag=tag, patterns=[], recommendations=[], reward=reward)


def _optimizer() -> FeedbackLearnerOptimizer:
    return FeedbackLearnerOptimizer(optimizer_type="gepa")


# --- defect 3a: the caller's reward floor must be the only reward floor -------


def test_low_reward_negatives_are_admitted():
    """RED before the fix: the hardcoded ``reward < 0.5`` drops all 5 negatives.

    A correct abstention on a healthy batch scores near 0 by construction, so
    the floor and the negative class are the same set.
    """
    signals = [_positive(f"p{i}") for i in range(5)] + [_negative(f"n{i}") for i in range(5)]
    examples = _optimizer()._signals_to_examples(signals, "pattern")
    empty_label = [e for e in examples if not e.patterns]
    assert len(empty_label) == 5, f"negatives dropped: {len(empty_label)} of 5 survived"


# --- defect 3b: the composition itself ---------------------------------------


def test_pattern_trainset_is_balanced_not_all_positive():
    """20 healthy cycles + 5 defect cycles -> 5 + 5, not 5 + 0 and not 5 + 20."""
    signals = [_positive(f"p{i}") for i in range(5)] + [_negative(f"n{i}") for i in range(20)]
    examples = _optimizer()._signals_to_examples(signals, "pattern")
    pos = [e for e in examples if e.patterns]
    neg = [e for e in examples if not e.patterns]
    assert (len(pos), len(neg)) == (5, 5)


def test_balanced_trainset_survives_the_80_20_prefix_split():
    """``_optimize_with_gepa`` splits by PREFIX; a class-ordered list would put
    every negative in the valset. Interleaving is what makes the split honest."""
    signals = [_positive(f"p{i}") for i in range(10)] + [_negative(f"n{i}") for i in range(10)]
    examples = _optimizer()._signals_to_examples(signals, "pattern")
    assert len(examples) == 20
    cut = int(len(examples) * 0.8)
    for part in (examples[:cut], examples[cut:]):
        pos = sum(1 for e in part if e.patterns)
        neg = sum(1 for e in part if not e.patterns)
        assert abs(pos - neg) <= 1, f"prefix split skewed: {pos} pos / {neg} neg"


def test_empty_input_signals_are_not_examples():
    """148 of 220 real rows collected NO feedback. 'given [], emit []' teaches
    nothing and is not a negative example — it is a degenerate one."""
    signals = [
        _signal(tag=f"e{i}", patterns=[], recommendations=[], feedback=[], reward=0.0)
        for i in range(10)
    ] + [_positive("p0"), _negative("n0")]
    examples = _optimizer()._signals_to_examples(signals, "pattern")
    assert len(examples) == 2
    for e in examples:
        assert json.loads(e.feedback_batch), "an example was built from an empty feedback batch"


# --- the honest-skip contract ------------------------------------------------


def test_single_class_pool_refuses_to_build_a_trainset():
    """All-positive is the #1668 defect; all-negative is its mirror. Neither is
    trainable, so the builder skips explicitly rather than appearing to work —
    the precedent set by the ``update`` phase."""
    assert (
        _optimizer()._signals_to_examples([_positive(f"p{i}") for i in range(8)], "pattern") == []
    )
    assert (
        _optimizer()._signals_to_examples([_negative(f"n{i}") for i in range(8)], "pattern") == []
    )


def test_summary_phase_is_an_explicit_skip():
    """``learning_summary`` is a deterministic f-string built by
    ``KnowledgeUpdaterNode._generate_summary`` — measured: all 220 stored values
    match that template, min length 135, so the metric's only discriminating
    term (length >= 40) saturates on every row. There is no gradient and no
    consumer loads ``feedback_learner_summary``."""
    signals = [_positive(f"p{i}") for i in range(5)] + [_negative(f"n{i}") for i in range(5)]
    assert _optimizer()._signals_to_examples(signals, "summary") == []


def test_update_phase_skip_is_preserved():
    signals = [_positive(f"p{i}") for i in range(5)]
    assert _optimizer()._signals_to_examples(signals, "update") == []


# --- no fabricated gold ------------------------------------------------------


def test_examples_carry_no_fabricated_gold_fields():
    """``confidence=0.8`` was a hardcoded constant on every example, including
    abstentions. Nothing persists a confidence, and the metric never scores it."""
    signals = [_positive(f"p{i}") for i in range(3)] + [_negative(f"n{i}") for i in range(3)]
    for e in _optimizer()._signals_to_examples(signals, "pattern"):
        assert "confidence" not in e, "fabricated confidence label present"


def test_recommendation_phase_requires_a_non_empty_input():
    """The signature's INPUT is ``detected_patterns``; with none there is nothing
    to condition on, whatever the recommendations field holds."""
    orphan = _signal(tag="o", patterns=[], recommendations=RECS, reward=0.9)
    with_patterns_no_recs = _signal(tag="w", patterns=PATTERNS, recommendations=[], reward=0.0)
    examples = _optimizer()._signals_to_examples(
        [orphan, _positive("p0"), with_patterns_no_recs], "recommendation"
    )
    assert len(examples) == 2  # balanced 1 + 1, the orphan excluded
    for e in examples:
        assert json.loads(e.detected_patterns)
