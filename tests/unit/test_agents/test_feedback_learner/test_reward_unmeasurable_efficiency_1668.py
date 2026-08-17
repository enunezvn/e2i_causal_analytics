"""#1668: the reward's efficiency term must be a function of the CYCLE, not of
the clock.

``compute_reward`` derives ``efficiency_score`` from
``feedback_count / total_latency_ms``. ``total_latency_ms`` is the sum of four
node timers, each stamped as ``int((time.time() - start) * 1000)`` — so a cycle
whose nodes all finish in under a millisecond sums to **0**, and the division is
undefined. The original fallback (initial platform commit ``3e1c70cf4``, no
issue, no comment) fabricated an anchor for that case::

    efficiency_score = 1.0 if self.feedback_count == 0 else 0.5

That makes two cycles that did *identical* work score differently on nothing but
timer granularity. Measured against the 219 real ``dspy_agent_training_signals``
rows in prod on 2026-08-16:

===========================================  ====  ==============
zero-feedback cycle                          rows  stored reward
===========================================  ====  ==============
``total_latency_ms == 0`` -> efficiency 1.0    40  0.2 / 0.3
``total_latency_ms  > 0`` -> efficiency 0.0   108  0.0
===========================================  ====  ==============

All 148 collected nothing, detected nothing, recommended nothing and applied
nothing. The 0.3 spread between them is pure clock artifact.

The module already has a convention for a term it cannot measure, applied twice
and documented at length: **omit it and redistribute its weight, never anchor on
a fabricated value** (``pattern_accuracy``, F-015/#424; ``update_effectiveness``,
F15/#837/#838). The efficiency fallback is the one unmeasurable term that was
never brought onto that convention.

These tests pin the corrected behaviour against real stored rows. Eligibility is
NOT affected — every row this touches sits at <= 0.3, far below the optimizer's
``reward >= 0.5`` floor — which is asserted explicitly below, because changing
who can train would be a reward-semantics decision, not a defect fix.
"""

from __future__ import annotations

import pytest

from src.agents.feedback_learner.dspy_integration import FeedbackLearnerTrainingSignal


def _signal(**kw) -> FeedbackLearnerTrainingSignal:
    base = {
        "batch_id": "b",
        "feedback_count": 0,
        "time_range_start": "2026-08-16T00:00:00Z",
        "time_range_end": "2026-08-17T00:00:00Z",
    }
    base.update(kw)
    return FeedbackLearnerTrainingSignal(**base)  # type: ignore[arg-type]


# =============================================================================
# 1. The defect: the same cycle, two rewards, decided by the clock
# =============================================================================


def test_zero_feedback_cycles_score_identically_regardless_of_timer_granularity():
    """Two real prod cycles that did the same nothing must score the same.

    ``2026-06-12T02:22:42`` summed its four node timers to 0 ms and stored
    **0.3**. ``2026-06-11T08:48:45`` summed to 43 ms and stored **0.0**. Both
    collected 0 feedback items, detected 0 patterns, generated 0
    recommendations and applied 0 updates.
    """
    sub_millisecond = _signal(total_latency_ms=0.0)
    measured = _signal(total_latency_ms=43.0)

    assert sub_millisecond.compute_reward() == measured.compute_reward()
    # And the honest value is the measured one: nothing was learned.
    assert sub_millisecond.compute_reward() == 0.0


def test_unmeasurable_duration_omits_efficiency_rather_than_anchoring_it():
    """An unmeasurable term is dropped and its weight redistributed (F-015/F15).

    With a rubric present the weights are actionability .20 / efficiency .10 /
    coverage .10 / rubric .20 (pattern_accuracy and update_effectiveness are
    already omitted as unmeasurable). Dropping efficiency too leaves
    .20 + .10 + .20 = .50, so a cycle with actionability 0.4, coverage 0 and a
    4.0 rubric scores (.20*0.4 + .20*0.75) / .50.
    """
    signal = _signal(
        feedback_count=100,
        patterns_detected=0,
        recommendation_actionability=0.4,
        rubric_weighted_score=4.0,
        total_latency_ms=0.0,
    )
    expected = (0.20 * 0.4 + 0.20 * 0.75) / 0.50
    assert signal.compute_reward() == pytest.approx(round(expected, 4))


def test_sub_millisecond_cycle_with_feedback_is_not_scored_a_fabricated_half():
    """The ``else 0.5`` branch is an anchor with no measurement behind it.

    Real rows ``2026-06-10T23:46:41`` and ``2026-06-11T00:22:13``: 3 feedback
    items, 0 patterns, 0 recommendations, no rubric, timers summed to 0 ms —
    stored **0.15**, which is exactly ``0.15 * 0.5 / 0.50``: five-eighths of the
    row's whole score came from a number nobody measured.
    """
    signal = _signal(feedback_count=3, patterns_detected=0, total_latency_ms=0.0)
    assert signal.compute_reward() == 0.0


# =============================================================================
# 2. Guard: measured cycles and the eligibility floor are untouched
# =============================================================================


@pytest.mark.parametrize(
    "kwargs,stored_reward",
    [
        # 2026-08-13T03:55:18 — the highest-scoring pattern-free cycle in prod.
        (
            {
                "feedback_count": 74,
                "patterns_detected": 0,
                "recommendation_actionability": 0.0,
                "rubric_weighted_score": 3.92,
                "total_latency_ms": 69.0,
            },
            0.41,
        ),
        # 2026-08-08T07:09:02 — the newest of the 8 eligible signals.
        (
            {
                "feedback_count": 26,
                "patterns_detected": 2,
                "recommendation_actionability": 0.4,
                "rubric_weighted_score": 3.67,
                "total_latency_ms": 32.0,
            },
            0.6507,
        ),
        # 2026-08-16T10:01:13 — the freshest row at the time of filing.
        (
            {
                "feedback_count": 71,
                "patterns_detected": 0,
                "recommendation_actionability": 0.0,
                "rubric_weighted_score": 1.62,
                "total_latency_ms": 107.0,
            },
            0.2183,
        ),
    ],
)
def test_measured_cycles_replay_their_stored_reward_bit_for_bit(kwargs, stored_reward):
    """Any cycle with a measurable duration is untouched by this change.

    All three are real ``dspy_agent_training_signals`` rows read on 2026-08-16.
    """
    assert _signal(**kwargs).compute_reward() == stored_reward


def test_the_repaired_rows_stay_far_below_the_optimizer_floor():
    """This is a defect fix, not a change to who can train.

    Every row the fix touches has ``feedback_count == 0`` or 3, zero patterns
    and zero recommendations; the best of them stored 0.3. Lowering them to 0.0
    cannot move a signal across the 0.5 eligibility floor in either direction.

    The 0.5 is a literal, not a gate constant. #1668 removed the gate's reward
    floor entirely (it counts label classes now), so the number below describes
    the eligibility rule these rows were measured against at the time — a fact
    about the historical corpus, not a coupling to today's gate.
    """
    optimizer_floor_at_the_time = 0.5
    worst_case_before = 0.3  # measured max across the 42 affected prod rows
    assert worst_case_before < optimizer_floor_at_the_time
    for kwargs in (
        {"total_latency_ms": 0.0},
        {"total_latency_ms": 0.0, "update_effectiveness": 0.0},
        {"feedback_count": 3, "total_latency_ms": 0.0},
    ):
        assert _signal(**kwargs).compute_reward() < optimizer_floor_at_the_time


def test_a_perfectly_efficient_measured_cycle_still_earns_the_full_term():
    """Omission applies ONLY when the duration is unmeasurable, never to speed.

    A cycle with a real 1 ms duration and 100 items is clamped to efficiency
    1.0 exactly as before — the term is present, not dropped.
    """
    fast = _signal(
        feedback_count=100,
        patterns_detected=0,
        recommendation_actionability=0.0,
        rubric_weighted_score=5.0,
        total_latency_ms=1.0,
    )
    # The #1661 ceiling: a flawless pattern-free cycle lands exactly on 0.5.
    assert fast.compute_reward() == 0.5
