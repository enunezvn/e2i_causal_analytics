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


def _as_list(value: Any) -> list:
    """Coerce a model output (list, JSON string, or None) to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


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
            return self._score_summary(pred)
        return 0.0, "No recognized output fields (patterns/recommendations/summary) to score"

    # --- pattern phase -----------------------------------------------------
    def _score_patterns(self, gold: Example, pred: Prediction) -> tuple[float, str]:
        pred_patterns = _as_list(getattr(pred, "patterns", []))
        gold_patterns = _as_list(getattr(gold, "patterns", []))

        if not pred_patterns:
            return 0.0, "CRITICAL: no patterns detected (empty output)"

        non_empty = 0.3
        well_structured = sum(
            1
            for p in pred_patterns
            if isinstance(p, dict)
            and (p.get("type") or p.get("pattern_type"))
            and p.get("severity")
        )
        structure = 0.3 * (well_structured / len(pred_patterns))

        gold_types = _types(gold_patterns, "pattern_type", "type")
        pred_types = _types(pred_patterns, "pattern_type", "type")
        if gold_types:
            overlap = 0.4 * (len(gold_types & pred_types) / len(gold_types))
            overlap_msg = f"{len(gold_types & pred_types)}/{len(gold_types)} gold types matched"
        else:
            overlap = 0.4 if pred_types else 0.0  # no gold labels -> reward structured output
            overlap_msg = "no gold types (structure-only credit)"

        score = non_empty + structure + overlap
        return score, (
            f"{len(pred_patterns)} patterns, {well_structured} well-structured, {overlap_msg}"
        )

    # --- recommendation phase ---------------------------------------------
    def _score_recommendations(self, gold: Example, pred: Prediction) -> tuple[float, str]:
        pred_recs = _as_list(getattr(pred, "recommendations", []))
        gold_recs = _as_list(getattr(gold, "recommendations", []))

        if not pred_recs:
            return 0.0, "CRITICAL: no recommendations generated (empty output)"

        non_empty = 0.3
        well_structured = sum(
            1
            for r in pred_recs
            if isinstance(r, dict)
            and r.get("category")
            and (r.get("description") or r.get("expected_impact"))
        )
        structure = 0.3 * (well_structured / len(pred_recs))

        gold_cats = _types(gold_recs, "category")
        pred_cats = _types(pred_recs, "category")
        if gold_cats:
            overlap = 0.4 * (len(gold_cats & pred_cats) / len(gold_cats))
            overlap_msg = f"{len(gold_cats & pred_cats)}/{len(gold_cats)} gold categories matched"
        else:
            overlap = 0.4 if pred_cats else 0.0
            overlap_msg = "no gold categories (structure-only credit)"

        score = non_empty + structure + overlap
        return score, (
            f"{len(pred_recs)} recommendations, {well_structured} well-structured, {overlap_msg}"
        )

    # --- summary phase -----------------------------------------------------
    def _score_summary(self, pred: Prediction) -> tuple[float, str]:
        summary = str(getattr(pred, "summary", "") or "")
        key_insights = _as_list(getattr(pred, "key_insights", []))
        next_steps = _as_list(getattr(pred, "next_steps", []))

        length_score = 0.4 if len(summary.strip()) >= 40 else (0.2 if summary.strip() else 0.0)
        insights_score = 0.3 if key_insights else 0.0
        steps_score = 0.3 if next_steps else 0.0

        score = length_score + insights_score + steps_score
        return score, (
            f"summary_len={len(summary.strip())}, "
            f"insights={len(key_insights)}, next_steps={len(next_steps)}"
        )


__all__ = ["FeedbackLearnerGEPAMetric"]
