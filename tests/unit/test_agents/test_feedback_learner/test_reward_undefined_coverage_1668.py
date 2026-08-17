"""#1668: ``coverage`` must not anchor on a fabricated 0.0 when it is undefined.

``compute_reward`` derives coverage from ``patterns_detected / feedback_count``.
When ``feedback_count == 0`` that ratio is ``0/0`` — UNDEFINED, not zero — and
the ``else`` branch fabricated ``coverage_score = 0.0``.

This is the same defect class #1671 fixed for ``efficiency`` (``feedback_count /
total_latency_ms`` at ``total_latency_ms == 0``) and the convention this module
already applies twice: ``pattern_accuracy`` (F-015 / #424) and
``update_effectiveness`` (F15 / #837) are OMITTED and their weight redistributed
when unmeasurable, never anchored.

Measured over all 220 real ``dspy_agent_training_signals`` rows
(``source_agent='feedback_learner'``, prod, 2026-08-17), replayed through the
real function: **0 rows change reward and eligibility stays 8 -> 8**, because
every one of the 148 zero-feedback rows also has ``rubric=None`` and
``actionability=0.0``, so its remaining terms are all zero either way.

That null result needed a positive control, and it has one below: a zero-feedback
cycle WITH a rubric score does move (0.4 -> 0.5), which is what proves the
measurement is a measurement and not a blind probe. Such a row has never been
recorded, so the change is forward-looking — and a contentless cycle becoming
gate-eligible is separately handled by the trainset builder, which excludes
empty-``feedback_batch`` signals from the pattern phase outright.
"""

from __future__ import annotations

from src.agents.feedback_learner.dspy_integration import FeedbackLearnerTrainingSignal


def _signal(**kw) -> FeedbackLearnerTrainingSignal:
    base = {"batch_id": "b", "time_range_start": "t0", "time_range_end": "t1"}
    base.update(kw)
    return FeedbackLearnerTrainingSignal(**base)  # type: ignore[arg-type]


def test_zero_feedback_cycle_omits_coverage_instead_of_anchoring_zero():
    """RED before the fix: 0.4 (coverage anchored at 0.0 drags the mean down).

    Terms present: actionability 0.20 @ 0.0, rubric 0.20 @ 1.0 (a flawless 5.0).
    Efficiency is already omitted (#1671, total_latency_ms == 0). Coverage is
    ``0/0`` and must be omitted too, leaving ``(0.20*0 + 0.20*1) / 0.40 = 0.5``.
    """
    sig = _signal(feedback_count=0, patterns_detected=0, rubric_weighted_score=5.0)
    assert sig.compute_reward() == 0.5


def test_zero_feedback_cycle_with_midrange_rubric():
    """RED before the fix: 0.2. Rubric 3.0 -> 0.5 normalized -> 0.5*0.20/0.40."""
    sig = _signal(feedback_count=0, rubric_weighted_score=3.0)
    assert sig.compute_reward() == 0.25


def test_defined_coverage_is_untouched():
    """feedback_count > 0 leaves ``patterns/feedback`` measurable — keep scoring it."""
    sig = _signal(feedback_count=10, patterns_detected=1, rubric_weighted_score=5.0)
    # coverage = (1/10)/0.1 = 1.0 -> (0.20*0 + 0.10*1 + 0.20*1) / 0.50 = 0.6
    assert sig.compute_reward() == 0.6

    barren = _signal(feedback_count=10, patterns_detected=0, rubric_weighted_score=5.0)
    # coverage = 0.0 and that is a REAL measurement, not an anchor -> 0.20/0.50
    assert barren.compute_reward() == 0.4


def test_contentless_cycle_still_scores_zero():
    """The 148-row real population: no feedback, no rubric, nothing measurable."""
    sig = _signal(feedback_count=0)
    assert sig.compute_reward() == 0.0


def test_omission_cannot_be_faked_by_returning_one():
    """Guard against 'unmeasurable -> 1.0', the mirror-image fabrication."""
    sig = _signal(feedback_count=0, recommendation_actionability=0.0)
    assert sig.compute_reward() == 0.0
