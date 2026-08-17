"""#1668: the GEPA feedback_learner metric must be scored AGAINST THE GOLD.

Before this suite, ``_score_patterns``/``_score_recommendations`` decided on the
PREDICTION alone::

    if not pred_patterns:
        return 0.0, "CRITICAL: no patterns detected (empty output)"

``gold`` was never consulted on that branch, and the ``gold_types`` fallback paid
the full overlap weight for any structured output. Measured truth table (both
phases, before):

    gold      pred      score
    EMPTY     EMPTY     0.00   <- a CORRECT ABSTENTION scored as catastrophic
    EMPTY     FULL      1.00   <- a FALSE POSITIVE scored PERFECT
    FULL      EMPTY     0.00       (correct: a missed detection)
    FULL      FULL      1.00       (correct)

Every deterministic detector in ``pattern_analyzer`` fires only on NEGATIVE
feedback (pooled rating < 3.0, > 5 corrections, > 3 outcome errors, per-agent
negative rate > 0.3), so "no patterns" is the correct answer on a healthy batch.
A metric that pays 1.00 for emitting patterns against an empty gold and 0.00 for
correctly emitting none can only teach unconditional over-reporting — and the
optimized ``feedback_learner_pattern`` artifact is loaded and used by the live
learning cycle (``PatternAnalyzerNode(prefer_optimized=True)``), so that is a
production behaviour change, not a lab curiosity.

Note the conflation this suite also pins: ``gold_types`` is empty BOTH when the
gold genuinely has no patterns (a non-empty prediction is then a FALSE POSITIVE)
and when the gold HAS patterns that carry no ``type``/``pattern_type`` key (the
overlap term is then simply unmeasurable). The two cases must be scored
differently — the second follows this codebase's omit-and-redistribute
convention (F-015/#424, F15/#837, efficiency/#1668), the first does not.

No mocks: real ``dspy.Example``/``dspy.Prediction``, real metric.
"""

from __future__ import annotations

import pytest

dspy = pytest.importorskip("dspy")

from src.optimization.gepa.metrics.feedback_learner_metric import (  # noqa: E402
    FeedbackLearnerGEPAMetric,
)

# Two well-formed patterns (type + severity) — what a healthy prediction looks like.
PATTERNS_FULL = [
    {"type": "low_rating", "severity": "high", "affected_agents": ["causal_impact"]},
    {"type": "correction_burst", "severity": "medium", "affected_agents": ["explainer"]},
]
# Gold patterns that carry NO type key — overlap is unmeasurable, not absent.
PATTERNS_UNTYPED = [{"description": "ratings slipped", "severity": "high"}]

RECS_FULL = [
    {"category": "prompt_update", "description": "tighten the ask head"},
    {"category": "data_update", "description": "refresh the corpus"},
]


@pytest.fixture
def metric() -> FeedbackLearnerGEPAMetric:
    return FeedbackLearnerGEPAMetric()


def _pattern_score(metric, gold_patterns, pred_patterns) -> float:
    gold = dspy.Example(
        feedback_batch="[]",
        agent_baselines="{}",
        historical_patterns="[]",
        patterns=gold_patterns,
    ).with_inputs("feedback_batch", "agent_baselines", "historical_patterns")
    pred = dspy.Prediction(patterns=pred_patterns, confidence=0.8, root_causes=[])
    return float(metric(gold, pred).score)


def _rec_score(metric, gold_recs, pred_recs) -> float:
    gold = dspy.Example(
        detected_patterns="[]",
        prior_learnings="[]",
        optimization_examples="[]",
        recommendations=gold_recs,
    ).with_inputs("detected_patterns", "prior_learnings", "optimization_examples")
    pred = dspy.Prediction(recommendations=pred_recs, implementation_order=[], risk_assessment="")
    return float(metric(gold, pred).score)


# --- the two rows that are WRONG today ---------------------------------------


def test_correct_abstention_on_empty_gold_scores_perfect(metric):
    """gold EMPTY + pred EMPTY is an exact match, not a catastrophe.

    RED before the fix: scores 0.0 ("CRITICAL: no patterns detected").
    """
    assert _pattern_score(metric, [], []) == 1.0


def test_false_positive_against_empty_gold_is_not_rewarded(metric):
    """gold EMPTY + pred FULL invents defects that were not there.

    RED before the fix: scores 1.00 — a perfect score for a false positive.
    0.0 is the maximal penalty a [0, 1] metric can express (``__call__``
    clamps, and GEPA averages), so a false positive and a missed detection sit
    at the same floor.
    """
    assert _pattern_score(metric, [], PATTERNS_FULL) == 0.0


# --- the two rows that are already right, pinned so the fix cannot break them -


def test_missed_detection_still_scores_zero(metric):
    assert _pattern_score(metric, PATTERNS_FULL, []) == 0.0


def test_matching_prediction_still_scores_perfect(metric):
    assert _pattern_score(metric, PATTERNS_FULL, PATTERNS_FULL) == 1.0


# --- the conflation ----------------------------------------------------------


def test_untyped_gold_keeps_structure_only_credit(metric):
    """Gold HAS patterns but none carry a type: overlap is unmeasurable.

    Omit-and-redistribute (this module's own convention for an unmeasurable
    term) — NOT the empty-gold false-positive branch.
    """
    assert _pattern_score(metric, PATTERNS_UNTYPED, PATTERNS_FULL) == 1.0


def test_untyped_gold_still_penalises_unstructured_prediction(metric):
    """Redistribution must not turn 'unmeasurable overlap' into free credit."""
    score = _pattern_score(metric, PATTERNS_UNTYPED, PATTERNS_UNTYPED)
    assert 0.0 < score < 1.0


def test_empty_gold_and_untyped_gold_are_scored_differently(metric):
    """The single fact the old ``if gold_types:`` branch could not represent."""
    assert _pattern_score(metric, [], PATTERNS_FULL) != _pattern_score(
        metric, PATTERNS_UNTYPED, PATTERNS_FULL
    )


# --- identical shape in the recommendation phase -----------------------------


def test_recommendation_phase_truth_table(metric):
    """RED before the fix on rows 1 and 2, exactly as for patterns."""
    assert _rec_score(metric, [], []) == 1.0
    assert _rec_score(metric, [], RECS_FULL) == 0.0
    assert _rec_score(metric, RECS_FULL, []) == 0.0
    assert _rec_score(metric, RECS_FULL, RECS_FULL) == 1.0


# --- the property the whole issue turns on -----------------------------------


def test_emitting_patterns_is_never_a_free_win(metric):
    """Over-reporting must not dominate abstention when the gold is empty.

    This is the mechanism by which training on today's trainset would make the
    platform worse: if ``score(EMPTY, FULL) > score(EMPTY, EMPTY)`` the optimizer
    is paid to invent findings.
    """
    assert _pattern_score(metric, [], PATTERNS_FULL) < _pattern_score(metric, [], [])
    assert _rec_score(metric, [], RECS_FULL) < _rec_score(metric, [], [])


# --- the hazard the gold-aware rewrite introduces, closed ---------------------


def test_unparseable_output_is_not_credited_as_an_abstention(metric):
    """Crediting an empty prediction 1.0 creates a new way to cheat: emit garbage.

    ``_as_list`` returned ``[]`` for a blob it could not parse, which was
    harmless while every empty prediction scored 0.0 anyway. It is not harmless
    now. Mirrors ``PatternAnalyzerNode``'s own ``pattern_parse_anomalies``
    guard: "0 patterns" after a parse failure is an anomaly, not a clean
    no-findings result.
    """
    for blob in ("not json at all", '{"patterns": 1}', "42"):
        assert _pattern_score(metric, [], blob) == 0.0, blob
        assert _rec_score(metric, [], blob) == 0.0, blob
    # a genuinely empty output is still a correct abstention
    assert _pattern_score(metric, [], "[]") == 1.0
    assert _pattern_score(metric, [], "") == 1.0
