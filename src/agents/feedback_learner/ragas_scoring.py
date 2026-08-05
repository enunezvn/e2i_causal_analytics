"""RAGAS bundle and the combined-score blend — one source of truth (#1487).

``database/ml/022_self_improvement_tables.sql`` shipped the whole RAGAS half of
the self-improvement schema — ``learning_signals.ragas_scores`` /
``ragas_weighted`` / ``combined_score``, the ``evaluation_results`` table, the
``v_ragas_performance_trends`` view and the ``calculate_combined_score()``
function — with no Python writer. This module is the scoring half of wiring it
up: a validated bundle of judged RAGAS metrics plus the blend that turns it,
with a rubric score, into ``combined_score``.

It is deliberately pure (no I/O, no LLM calls). ``RAGAS judging costs many
seconds of gpt-4o calls`` — the live chat path already has a latency problem
(#1484) and must never do it inline. The realistic producers are offline: the
real-pipeline batch eval (#1485) and any future async scorer. So the public API
takes an explicit, already-judged bundle rather than computing one.

Building a bundle from the two real producers
---------------------------------------------
``src.rag.evaluation.EvaluationResult`` (#1488 semantics — an unmeasured metric
is ``None``, never ``0.0``, and the names it could not score are listed under
``metadata["unmeasured_metrics"]``)::

    RagasBundle(
        scores={
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        },
        unmeasured_metrics=result.metadata.get("unmeasured_metrics", ()),
        evaluation_method=result.metadata.get("evaluation_method"),
    )

A ``per_sample`` row from the real-pipeline judge (#1485), which reports only
faithfulness and answer_relevancy because the other three need a ground truth
the replay deliberately does not fabricate::

    RagasBundle(
        scores={
            "faithfulness": row["faithfulness"],
            "answer_relevancy": row["answer_relevancy"],
        },
        evaluation_method=row["evaluation_method"],
    )

Note the difference the bundle preserves: #1488's ``None`` means *the judge
tried and failed*; #1485's *absent key* means *this producer never asks for that
metric*. Both persist as SQL NULL, but only the first is a judge malfunction,
and ``coverage`` keeps them apart.

Where this deliberately diverges from ``calculate_combined_score()``
--------------------------------------------------------------------
The SQL function ``COALESCE``s a missing metric to 0 and a missing
``rubric_total`` to the bottom of the scale. On a partial bundle that silently
understates the score — a real-pipeline row measuring only faithfulness and
answer_relevancy could never exceed 0.45 no matter how good it was — and on a
rubric-only row it publishes 40%-of-nothing as if it were a measured blend.
Python instead renormalises over the weight that was actually measured, and
returns ``None`` when a half is missing entirely. The weights and the 0.4/0.6
split are identical, and ``tests/unit/test_agents/test_feedback_learner/
test_ragas_scoring.py`` parses them straight out of the migration to keep the
two from drifting. Do not call the SQL function on a partial row.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Per-metric weights, transcribed from calculate_combined_score() in
# database/ml/022_self_improvement_tables.sql. They sum to 1.0, so a COMPLETE
# bundle's weighted score is byte-identical to the SQL function's.
RAGAS_METRIC_WEIGHTS: Dict[str, float] = {
    "faithfulness": 0.25,
    "answer_relevancy": 0.20,
    "context_precision": 0.20,
    "context_recall": 0.20,
    "answer_correctness": 0.15,
}

# The documented blend: combined = (ragas * 0.4) + (rubric_normalised * 0.6),
# matching the SQL function's parameter defaults and the COMMENT ON COLUMN on
# learning_signals.combined_score.
RAGAS_BLEND_WEIGHT = 0.4
RUBRIC_BLEND_WEIGHT = 0.6

# The rubric's own scale; normalisation to 0-1 is (total - 1) / 4.
RUBRIC_SCALE_MIN = 1.0
RUBRIC_SCALE_MAX = 5.0

# ROUND(v::numeric, 4) in the SQL function.
_COMBINED_SCORE_QUANTUM = Decimal("0.0001")

# Same vocabulary as src/optimization/dspy_lane_ab.py and
# src/agents/feedback_learner/evaluation/models.py: a score the LLM judge never
# actually produced.
HEURISTIC_EVALUATION_METHOD = "fallback_heuristic"

__all__ = [
    "HEURISTIC_EVALUATION_METHOD",
    "RAGAS_BLEND_WEIGHT",
    "RAGAS_METRIC_WEIGHTS",
    "RUBRIC_BLEND_WEIGHT",
    "RUBRIC_SCALE_MAX",
    "RUBRIC_SCALE_MIN",
    "RagasBundle",
    "combined_score",
]


def _clamp_unit(value: float) -> float:
    """Keep a weighted average inside the 0-1 CHECK constraints.

    A mean of values in [0, 1] cannot mathematically leave [0, 1], but float
    accumulation can land on 1.0000000000000002 — which the column CHECK
    rejects, and ``RubricNode._store_evaluation`` swallows insert errors, so the
    row would silently vanish rather than fail loudly.
    """
    return min(1.0, max(0.0, value))


class RagasBundle(BaseModel):
    """One query-response pair's judged RAGAS metrics.

    Attributes:
        scores: Metric name to judged value in [0, 1]. ``None`` marks a metric
            the judge attempted but could not score (#1488); a metric simply
            absent from the mapping was never evaluated by this producer.
        unmeasured_metrics: Metric names the judge attempted but could not
            score, mirroring #1488's ``metadata["unmeasured_metrics"]``. A
            ``None`` value in ``scores`` means the same thing; supplying either
            (or both) is fine.
        evaluation_method: ``None`` for the judged path, per the #1485 judge
            script's convention. ``"fallback_heuristic"`` is refused.
        evaluation_model: Judge model label, persisted for provenance.
        evaluation_duration_ms: Judge wall time, persisted for provenance.
    """

    model_config = ConfigDict(frozen=True)

    scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    unmeasured_metrics: Sequence[str] = Field(default=())
    evaluation_method: Optional[str] = None
    evaluation_model: Optional[str] = None
    evaluation_duration_ms: Optional[int] = None

    @model_validator(mode="after")
    def _validate_bundle(self) -> "RagasBundle":
        named = set(self.scores) | set(self.unmeasured_metrics)
        unknown = sorted(named - set(RAGAS_METRIC_WEIGHTS))
        if unknown:
            raise ValueError(
                f"unknown RAGAS metric(s) {unknown}; a name outside "
                f"{sorted(RAGAS_METRIC_WEIGHTS)} would vanish from the weighted score "
                "instead of contributing to it"
            )

        for name, value in self.scores.items():
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError(
                    f"{name}={value!r} is not a measurement; #1488 converts a NaN'd "
                    "metric to None upstream, and a non-finite float cannot be "
                    "serialised to JSON or stored as a NULL"
                )
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name}={value!r} is outside the 0-1 range the column CHECK enforces"
                )

        contradictory = sorted(
            name
            for name in self.unmeasured_metrics
            if self.scores.get(name) is not None and name in self.scores
        )
        if contradictory:
            raise ValueError(
                f"{contradictory} listed as both measured and unmeasured — the bundle "
                "cannot say whether the judge scored them"
            )

        if self.evaluation_method == HEURISTIC_EVALUATION_METHOD:
            raise ValueError(
                "refusing a heuristic-scored bundle: word-overlap fallbacks are not "
                "RAGAS judgments, and evaluation_results has no column that could mark "
                "the row, so v_ragas_performance_trends would average them in as if a "
                "judge had produced them"
            )
        return self

    @property
    def measured(self) -> Dict[str, float]:
        """Metrics the judge actually scored, in weight order."""
        return {
            name: value
            for name in RAGAS_METRIC_WEIGHTS
            if (value := self.scores.get(name)) is not None
        }

    @property
    def weighted(self) -> Optional[float]:
        """Weighted aggregate over the MEASURED metrics, or None if there are none.

        Renormalised by the measured weight, so a complete bundle reproduces the
        SQL function exactly (the weights sum to 1) while a partial one reports
        the quality of what was judged rather than a COALESCE-to-zero penalty
        for what was not.
        """
        measured = self.measured
        if not measured:
            return None
        total_weight = sum(RAGAS_METRIC_WEIGHTS[name] for name in measured)
        weighted_sum = sum(RAGAS_METRIC_WEIGHTS[name] * value for name, value in measured.items())
        return _clamp_unit(weighted_sum / total_weight)

    @property
    def coverage(self) -> Dict[str, Any]:
        """What was judged, what failed, and what was never asked for.

        Persisted alongside the scores because ``evaluation_results`` stores an
        unmeasured metric as a bare NULL, which cannot distinguish a judge
        malfunction from a producer that does not report that metric at all.
        """
        measured = self.measured
        unmeasured = sorted(
            set(self.unmeasured_metrics)
            | {name for name, value in self.scores.items() if value is None}
        )
        return {
            "measured": list(measured),
            "unmeasured": unmeasured,
            "not_evaluated": sorted(
                set(RAGAS_METRIC_WEIGHTS) - set(measured) - set(unmeasured),
            ),
            "measured_weight": sum(RAGAS_METRIC_WEIGHTS[name] for name in measured),
        }

    def as_signal_scores(self) -> Dict[str, float]:
        """Payload for ``learning_signals.ragas_scores`` (JSONB).

        Only measured metrics appear: absence represents absence. The keys are
        the ones ``v_learning_signal_distribution`` and
        ``calculate_combined_score()`` read out of the JSONB.
        """
        return dict(self.measured)


def combined_score(
    ragas_weighted: Optional[float],
    rubric_total: Optional[float],
) -> Optional[float]:
    """Blend a RAGAS aggregate with a rubric total, or return None.

    ``None`` when either half is missing. The column documents a two-half score;
    publishing a rubric-only number there would be 40%-of-zero wearing the name
    of a measured blend, and a reader has no way to tell the difference after
    the fact.

    Args:
        ragas_weighted: :attr:`RagasBundle.weighted`, or None.
        rubric_total: Rubric score on its 1-5 scale, or None.

    Returns:
        The blend rounded to 4 decimals exactly as the SQL function rounds it,
        or None when a half is missing.

    Raises:
        ValueError: ``rubric_total`` is outside the 1-5 scale its normalisation
            is defined on (a 0 would yield a negative score the column CHECK
            rejects).
    """
    if ragas_weighted is None or rubric_total is None:
        return None
    if not RUBRIC_SCALE_MIN <= rubric_total <= RUBRIC_SCALE_MAX:
        raise ValueError(
            f"rubric_total={rubric_total!r} is outside the "
            f"{RUBRIC_SCALE_MIN}-{RUBRIC_SCALE_MAX} scale that "
            "(total - 1) / 4 normalises"
        )

    rubric_normalised = (rubric_total - RUBRIC_SCALE_MIN) / (RUBRIC_SCALE_MAX - RUBRIC_SCALE_MIN)
    blended = _clamp_unit(
        ragas_weighted * RAGAS_BLEND_WEIGHT + rubric_normalised * RUBRIC_BLEND_WEIGHT
    )
    # Postgres rounds the numeric CAST of the float, half-away-from-zero;
    # Python's round() rounds the binary value half-to-even and disagrees on
    # genuine ties (0.98435 -> 0.9843 vs 0.9844). Decimal(str(...)) reproduces
    # the same shortest-repr cast Postgres performs.
    return float(Decimal(str(blended)).quantize(_COMBINED_SCORE_QUANTUM, rounding=ROUND_HALF_UP))
