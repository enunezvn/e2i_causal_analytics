"""Feedback Learner GEPA Metric for E2I Tier 5 Deep Agent.

This metric scores the Feedback Learner's DSPy modules for the three phases the
optimizer actually runs — pattern detection, recommendation generation, and
learning summary — against the gold example carried on each persisted training
signal (Shard 04). It is phase-aware: it inspects which output fields the
prediction carries and scores that phase.

IMPORTANT (dspy 3.1.0 contract): GEPA's valset evaluation runs the metric
through ``dspy.Evaluate``, which *sums* metric returns to compute the average.
A metric that returns a plain ``dict`` triggers ``int + dict`` there. dspy's
GEPA expects either a ``float`` or a ``dspy.Prediction(score=float, feedback=str)``
(``ScoreWithFeedback``). We return the latter so the same metric serves both the
reflective-feedback path and the scalar valset evaluation. (This was a latent
bug: the optimizer had never been invoked in production — audit F1 — so the
plain-dict return never reached the evaluation path.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from dspy import Example, Prediction

from src.optimization.gepa.metrics.base import DSPyTrace


def _parse_list(value: Any) -> tuple[list, bool]:
    """Coerce a model output to a list, reporting whether it was INTELLIGIBLE.

    #1668: the boolean matters now. Before the gold-aware rewrite an empty list
    and an unparseable blob both scored 0.0, so conflating them was harmless.
    Now an empty prediction against an empty gold is a *correct abstention*
    worth 1.0 — so "the model emitted nothing" and "the model emitted something
    we could not read" must not be the same answer, or a prompt that produces
    garbage would be paid for abstaining. Same shape as the analyzer's own
    ``pattern_parse_anomalies`` guard: "0 patterns" after a parse failure is an
    anomaly, not a clean no-findings result.

    Returns ``(items, intelligible)``.
    """
    if value is None:
        return [], True  # the field is genuinely absent
    if isinstance(value, list):
        return value, True
    if isinstance(value, str):
        if not value.strip():
            return [], True
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return [], False
        return (parsed, True) if isinstance(parsed, list) else ([], False)
    return [], False


def _as_list(value: Any) -> list:
    """Coerce a model output (list, JSON string, or None) to a list."""
    return _parse_list(value)[0]


def _types(items: list, *keys: str) -> set[str]:
    """Collect the values of the first present key for each dict item."""
    out: set[str] = set()
    for it in items:
        if isinstance(it, dict):
            for k in keys:
                v = it.get(k)
                if v:
                    out.add(str(v).lower())
                    break
    return out


@dataclass
class FeedbackLearnerGEPAMetric:
    """GEPA metric for the Tier-5 Feedback Learner agent.

    Scores the real output fields of whichever phase produced ``pred``:
    - pattern:        ``patterns`` (list[dict]) vs gold ``patterns``
    - recommendation: ``recommendations`` (list[dict]) vs gold ``recommendations``
    - summary:        ``summary`` (str) + ``key_insights``/``next_steps``

    Returns ``dspy.Prediction(score, feedback)`` per the dspy 3.1.0 GEPA contract.
    Never raises: any internal error degrades to a 0.0 score with feedback.
    """

    name: str = "feedback_learner_gepa"
    description: str = (
        "GEPA metric for Tier 5 Feedback Learner — scores pattern/recommendation/summary "
        "quality (structure + gold overlap) for the phase under optimization"
    )

    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> Prediction:
        """Compute a ScoreWithFeedback (dspy.Prediction) for GEPA."""
        try:
            score, feedback = self._score(gold, pred)
        except Exception as e:  # noqa: BLE001 - a metric must never abort optimization
            score, feedback = 0.0, f"Metric error (scored 0.0): {e}"
        score = max(0.0, min(1.0, float(score)))
        return Prediction(score=score, feedback=feedback)

    def _score(self, gold: Example, pred: Prediction) -> tuple[float, str]:
        # Disambiguate phase by the prediction's own output fields. The three
        # signatures' OUTPUT fields are disjoint (patterns / recommendations /
        # summary), so check the most distinctive collection outputs FIRST and
        # treat summary as the fallback — robust even if a stray `summary`
        # attribute is ever injected onto a pattern/recommendation prediction.
        if getattr(pred, "patterns", None) is not None:
            return self._score_patterns(gold, pred)
        if getattr(pred, "recommendations", None) is not None:
            return self._score_recommendations(gold, pred)
        if getattr(pred, "summary", None) is not None:
            return self._score_summary(gold, pred)
        return 0.0, "No recognized output fields (patterns/recommendations/summary) to score"

    # --- pattern phase -----------------------------------------------------
    def _score_patterns(self, gold: Example, pred: Prediction) -> tuple[float, str]:
        """Score a pattern prediction AGAINST THE GOLD (#1668).

        The previous implementation decided the empty-prediction case on the
        prediction alone (``if not pred_patterns: return 0.0``) and paid the
        full overlap weight for any structured output when the gold carried no
        type labels. Measured truth table, both phases:

            gold      pred      score
            EMPTY     EMPTY     0.00   <- a CORRECT ABSTENTION, scored fatal
            EMPTY     FULL      1.00   <- a FALSE POSITIVE, scored perfect
            FULL      EMPTY     0.00       (correct)
            FULL      FULL      1.00       (correct)

        Every deterministic detector in ``pattern_analyzer`` fires only on
        NEGATIVE feedback, so "no patterns" is the correct answer on a healthy
        batch — and the optimized artifact this metric selects is loaded by the
        live learning cycle (``PatternAnalyzerNode(prefer_optimized=True)``).
        A metric that pays more for inventing findings than for correctly
        finding none can only teach unconditional over-reporting.

        The empty-``gold_types`` branch also conflated two different situations,
        which is why the fix needs both halves:

        - the gold has NO patterns          -> a non-empty prediction is a FALSE
                                               POSITIVE and must not be paid
        - the gold HAS patterns but none of
          them carries a ``type`` key       -> the overlap term is simply
                                               UNMEASURABLE, so it is omitted and
                                               its weight redistributed, the
                                               convention this codebase already
                                               applies to ``pattern_accuracy``
                                               (#424), ``update_effectiveness``
                                               (#837) and ``efficiency`` (#1668)

        0.0 is the strongest penalty a ``[0, 1]`` metric can express — ``__call__``
        clamps and GEPA averages — so a false positive and a missed detection sit
        at the same floor rather than one being merely un-rewarded.
        """
        pred_patterns, intelligible = _parse_list(getattr(pred, "patterns", []))
        gold_patterns = _as_list(getattr(gold, "patterns", []))

        if not intelligible:
            return 0.0, (
                "UNPARSEABLE: the patterns output could not be read as a list — this is a "
                "parse failure, NOT an abstention"
            )

        if not pred_patterns:
            if not gold_patterns:
                return 1.0, "correct abstention: gold carries no patterns and none were emitted"
            return 0.0, f"MISSED: gold carries {len(gold_patterns)} pattern(s), none detected"

        if not gold_patterns:
            return 0.0, (
                f"FALSE POSITIVE: {len(pred_patterns)} pattern(s) emitted against a gold "
                "that carries none"
            )

        well_structured = sum(
            1
            for p in pred_patterns
            if isinstance(p, dict)
            and (p.get("type") or p.get("pattern_type"))
            and p.get("severity")
        )
        # (weight, score) pairs; the overlap term is omitted when unmeasurable.
        terms: list[tuple[float, float]] = [
            (0.3, 1.0),  # agrees with a non-empty gold that there IS something
            (0.3, well_structured / len(pred_patterns)),
        ]

        gold_types = _types(gold_patterns, "pattern_type", "type")
        pred_types = _types(pred_patterns, "pattern_type", "type")
        if gold_types:
            terms.append((0.4, len(gold_types & pred_types) / len(gold_types)))
            overlap_msg = f"{len(gold_types & pred_types)}/{len(gold_types)} gold types matched"
        else:
            overlap_msg = "gold carries no type labels (overlap term omitted, weight redistributed)"

        score = sum(w * s for w, s in terms) / sum(w for w, _ in terms)
        return score, (
            f"{len(pred_patterns)} patterns, {well_structured} well-structured, {overlap_msg}"
        )

    # --- recommendation phase ---------------------------------------------
    def _score_recommendations(self, gold: Example, pred: Prediction) -> tuple[float, str]:
        """Identical shape to :meth:`_score_patterns` — see its docstring (#1668)."""
        pred_recs, intelligible = _parse_list(getattr(pred, "recommendations", []))
        gold_recs = _as_list(getattr(gold, "recommendations", []))

        if not intelligible:
            return 0.0, (
                "UNPARSEABLE: the recommendations output could not be read as a list — this "
                "is a parse failure, NOT an abstention"
            )

        if not pred_recs:
            if not gold_recs:
                return 1.0, (
                    "correct abstention: gold carries no recommendations and none were generated"
                )
            return 0.0, f"MISSED: gold carries {len(gold_recs)} recommendation(s), none generated"

        if not gold_recs:
            return 0.0, (
                f"FALSE POSITIVE: {len(pred_recs)} recommendation(s) generated against a gold "
                "that carries none"
            )

        well_structured = sum(
            1
            for r in pred_recs
            if isinstance(r, dict)
            and r.get("category")
            and (r.get("description") or r.get("expected_impact"))
        )
        terms: list[tuple[float, float]] = [
            (0.3, 1.0),
            (0.3, well_structured / len(pred_recs)),
        ]

        gold_cats = _types(gold_recs, "category")
        pred_cats = _types(pred_recs, "category")
        if gold_cats:
            terms.append((0.4, len(gold_cats & pred_cats) / len(gold_cats)))
            overlap_msg = f"{len(gold_cats & pred_cats)}/{len(gold_cats)} gold categories matched"
        else:
            overlap_msg = (
                "gold carries no category labels (overlap term omitted, weight redistributed)"
            )

        score = sum(w * s for w, s in terms) / sum(w for w, _ in terms)
        return score, (
            f"{len(pred_recs)} recommendations, {well_structured} well-structured, {overlap_msg}"
        )

    # --- summary phase -----------------------------------------------------
    def _score_summary(self, gold: Example, pred: Prediction) -> tuple[float, str]:
        """Score a summary prediction against whatever gold actually exists (#1668).

        ``key_insights`` and ``next_steps`` are output fields of
        ``LearningSummarySignature`` that NO production node ever produces and
        nothing persists — ``_signals_to_examples`` used to hardcode them to
        ``[]`` on every gold. Paying 0.6 for emitting them against a gold that
        cannot carry them is the same "reward volume, never compare" defect the
        pattern phase had, so they are omitted (weight redistributed) unless the
        gold really carries them.

        What is left is a length term, which is why
        ``_signals_to_examples`` skips this phase outright for feedback_learner:
        all 220 stored ``learning_summary`` values are the deterministic f-string
        built by ``KnowledgeUpdaterNode._generate_summary`` (min length 135), so
        the term saturates on every row and there is no gradient to optimize.
        """
        summary = str(getattr(pred, "summary", "") or "").strip()
        gold_insights = _as_list(getattr(gold, "key_insights", []))
        gold_steps = _as_list(getattr(gold, "next_steps", []))
        key_insights = _as_list(getattr(pred, "key_insights", []))
        next_steps = _as_list(getattr(pred, "next_steps", []))

        terms: list[tuple[float, float]] = [
            (0.4, 1.0 if len(summary) >= 40 else (0.5 if summary else 0.0))
        ]
        omitted: list[str] = []
        if gold_insights:
            terms.append((0.3, 1.0 if key_insights else 0.0))
        else:
            omitted.append("key_insights")
        if gold_steps:
            terms.append((0.3, 1.0 if next_steps else 0.0))
        else:
            omitted.append("next_steps")

        score = sum(w * s for w, s in terms) / sum(w for w, _ in terms)
        omitted_msg = f", omitted (no gold): {'+'.join(omitted)}" if omitted else ""
        return score, (
            f"summary_len={len(summary)}, "
            f"insights={len(key_insights)}, next_steps={len(next_steps)}{omitted_msg}"
        )


__all__ = ["FeedbackLearnerGEPAMetric"]
