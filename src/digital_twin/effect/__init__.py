"""Twin uplift effect engine — public exports.

The real causal effect engine that replaces the hardcoded INTERVENTION_EFFECTS
heuristic. Fail-closed: bad/insufficient data raises rather than fabricating an ATE.
"""

from src.digital_twin.effect.errors import EffectDataUnavailable
from src.digital_twin.effect.estimate import (
    PROVENANCE_COHORT,
    PROVENANCE_RWD,
    PROVENANCE_SYNTHETIC,
    EffectEstimate,
)
from src.digital_twin.effect.estimator import TwinEffectEstimator
from src.digital_twin.effect.heterogeneity import (
    SegmentEffect,
    segment_by_uplift_quantiles,
)
from src.digital_twin.effect.provider import (
    COHORT_ESTIMABLE_INTERVENTIONS,
    INTERVENTION_TREATMENT_MAP,
    CohortEffectDataProvider,
    EffectDataProvider,
    SyntheticEffectDataProvider,
    TrainingFrame,
    region_standardized_ate,
)
from src.digital_twin.effect.recommendation import (
    PolicyThresholds,
    Recommendation,
    RecommendationPolicy,
)

__all__ = [
    "EffectDataUnavailable",
    "EffectEstimate",
    "PROVENANCE_SYNTHETIC",
    "PROVENANCE_RWD",
    "PROVENANCE_COHORT",
    "TwinEffectEstimator",
    "SegmentEffect",
    "segment_by_uplift_quantiles",
    "EffectDataProvider",
    "SyntheticEffectDataProvider",
    "CohortEffectDataProvider",
    "INTERVENTION_TREATMENT_MAP",
    "COHORT_ESTIMABLE_INTERVENTIONS",
    "region_standardized_ate",
    "TrainingFrame",
    "PolicyThresholds",
    "Recommendation",
    "RecommendationPolicy",
]
