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
    brand: (
        str | None
    )  # patient_journeys.brand partition; None = all brands (no brand partition filter)
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

# Grounded in _PJ_COHORTS["persistence"]/["discontinuation"] in
# src/services/cohort_resolution.py. Both labels are 180-day post-index outcomes
# (each is the OTHER cohort's leakage column; both already in
# feature_builder.LEAKAGE_DENYLIST). brand=None: persistence is brand-agnostic in
# the synthetic DGP (pos rate ~0.55 across all 3 brands) so we train ALL brands;
# discontinued_180d == 1 - persistent_180d exactly in-data.
PERSISTENCE = CohortSpec(
    name="persistence",
    target="pnh_persistence",
    brand=None,
    label_column="persistent_180d",
    grain="patient",
    base_covariates=("disease_severity", "academic_hcp", "geographic_region"),
)

DISCONTINUATION = CohortSpec(
    name="discontinuation",
    target="pnh_discontinuation",
    brand=None,
    label_column="discontinued_180d",
    grain="patient",
    base_covariates=("disease_severity", "academic_hcp", "geographic_region"),
)
