"""Result container + provenance labels for the twin effect engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PROVENANCE_SYNTHETIC = "synthetic_uplift_v1"
PROVENANCE_RWD = "rwd_uplift"
# Phase 2: the effect MAGNITUDE is estimated (region-standardized) from the
# brand's synthetic-gold cohort (business_metrics/per_hcp_rollup), so the ATE is
# brand- and intervention-differentiated rather than the flat synthetic uplift.
# Still synthetic-gold data (NOT real-world); the UI keeps the SYNTHETIC badge.
PROVENANCE_COHORT = "cohort_estimated_synthetic_gold_v1"

# The subgroup axes the engine reports, one per ``EffectHeterogeneity.by_*`` dimension
# (``digital_twin/models/simulation_models.py``). This is the vocabulary an estimator uses
# to declare what its scores resolve (#2054); a test pins it against those fields.
SUBGROUP_AXES: tuple[str, ...] = ("specialty", "decile", "region", "adoption_stage")


@dataclass(frozen=True)
class AxisProvenance:
    """Evidence and fallback contract for one declared heterogeneity axis.

    This metadata travels with the estimate so the API can explain where an axis came
    from and why a label may be absent.  ``fallback`` describes scoring only: a fallback
    value is never promoted into ``cate_by_axis`` as if it were a supported subgroup
    estimate.
    """

    basis: str
    source: str
    min_group_rows: int
    min_treated_rows: int | None = None
    min_control_rows: int | None = None
    fallback: str = "none"
    support_unit: str = "rows"
    estimand: str = "group_mean_cate"
    suppressed_groups: dict[str, str] = field(default_factory=dict)


@dataclass
class EffectEstimate:
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    att: float | None
    atc: float | None
    per_twin_uplift: np.ndarray
    auuc: float | None
    qini: float | None
    feature_importances: dict[str, float] | None
    n_train: int
    estimator_type: str
    data_provenance: str
    # Region scope of this estimate (#2023). Empty = the whole cohort, and ``cohort_*`` is
    # None because nothing was narrowed away. When regions ARE targeted, ``ate`` and its
    # interval are the effect ON those regions and ``cohort_*`` carries the cohort-wide
    # estimate they were narrowed from, so both numbers stay available to the caller.
    target_regions: list[str] = field(default_factory=list)
    cohort_ate: float | None = None
    cohort_ci_lower: float | None = None
    cohort_ci_upper: float | None = None
    # Which of ``SUBGROUP_AXES`` this estimator's scores actually RESOLVE, and the evidence
    # behind each (#2054). A key's PRESENCE is the declaration; both default to empty, so
    # an estimator that declares nothing resolves nothing and the engine reports ``{}``
    # rather than a group average that is only sampling noise in the twin draw.
    #   * key -> NON-EMPTY mapping: the estimator computed these group effects itself over
    #     its OWN evidence rows, and ``n_by_axis[axis]`` is that per-group row count. The
    #     engine reports these values as-is; they do not move with the twin count.
    #   * key -> EMPTY mapping: the axis is resolved by ``per_twin_uplift`` itself (a real
    #     per-twin score that varies WITHIN every group), so the engine groups the per-twin
    #     scores as before. ``n_by_axis`` is then the twin count and is left unset.
    cate_by_axis: dict[str, dict[str, float]] = field(default_factory=dict)
    n_by_axis: dict[str, dict[str, int]] = field(default_factory=dict)
    axis_provenance: dict[str, AxisProvenance] = field(default_factory=dict)

    def ci_width(self) -> float:
        return float(self.ate_ci_upper - self.ate_ci_lower)

    def uplift_summary(self) -> dict[str, float | int]:
        scores = np.asarray(self.per_twin_uplift, dtype=float).ravel()
        if scores.size == 0:
            return {"n": 0}
        return {
            "n": int(scores.size),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "p10": float(np.percentile(scores, 10)),
            "p90": float(np.percentile(scores, 90)),
        }
