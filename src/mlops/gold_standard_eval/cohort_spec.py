"""CohortSpec — static descriptors for gold-standard evaluation cohorts.

Each CohortSpec records the ML experiment target name, the patient_journeys
brand partition, the ground-truth label column, the data grain, and a
leakage-safe seed covariate list.

``base_covariates`` is grounded in ``_PJ_COHORTS`` in
``src/services/cohort_resolution.py`` (the authoritative per-cohort
variable-set used at causal-inference time).  The final feature set for
model training is finalized empirically in a later task (FeatureBuilder /
EXPERIMENT lock); this spec is the starting seed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CohortSpec:
    name: str
    target: str  # ml_experiments.prediction_target
    brand: str  # patient_journeys.brand partition
    label_column: str  # ground-truth column in patient_journeys
    grain: str  # "patient" | "hcp"
    base_covariates: tuple[str, ...]  # leakage-safe seed features (from _PJ_COHORTS)


# Grounded in _PJ_COHORTS["initiation"] in src/services/cohort_resolution.py:
#   ("treatment_initiated", "treatment_arm", ["disease_severity", "academic_hcp", "geographic_region"])
# Post-outcome columns (days_to_treatment, discontinued_180d, persistent_180d, adherence_rate)
# and the label column itself are excluded to prevent leakage.
INITIATION = CohortSpec(
    name="initiation",
    target="csu_treatment_initiation",
    brand="Remibrutinib",
    label_column="treatment_initiated",
    grain="patient",
    base_covariates=(
        "disease_severity",
        "academic_hcp",
        "geographic_region",
    ),
)
