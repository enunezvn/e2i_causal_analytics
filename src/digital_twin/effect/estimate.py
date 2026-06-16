"""Result container + provenance labels for the twin effect engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROVENANCE_SYNTHETIC = "synthetic_uplift_v1"
PROVENANCE_RWD = "rwd_uplift"
# Phase 2: the effect MAGNITUDE is estimated (region-standardized) from the
# brand's synthetic-gold cohort (business_metrics/per_hcp_rollup), so the ATE is
# brand- and intervention-differentiated rather than the flat synthetic uplift.
# Still synthetic-gold data (NOT real-world); the UI keeps the SYNTHETIC badge.
PROVENANCE_COHORT = "cohort_estimated_synthetic_gold_v1"


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
