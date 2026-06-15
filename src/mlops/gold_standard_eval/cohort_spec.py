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

# ---------------------------------------------------------------------------
# Per-brand patient cohort factory (P3 — 9 slots: cohort × brand)
# ---------------------------------------------------------------------------

BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
PATIENT_COHORTS = ("initiation", "persistence", "discontinuation")
_PATIENT_LABELS: dict[str, str] = {
    "initiation": "treatment_initiated",
    "persistence": "persistent_180d",
    "discontinuation": "discontinued_180d",
}
_BASE3 = ("disease_severity", "academic_hcp", "geographic_region")


def make_patient_spec(cohort: str, brand: str) -> CohortSpec:
    """Build a per-brand CohortSpec for a patient-grain cohort.

    cohort in PATIENT_COHORTS; brand in BRANDS.  target/name are uniform
    f"{cohort}_{brand.lower()}" so the 9 per-brand models register cleanly.
    """
    if cohort not in _PATIENT_LABELS:
        raise ValueError(f"unknown patient cohort {cohort!r}")
    if brand not in BRANDS:
        raise ValueError(f"unknown brand {brand!r}")
    key = f"{cohort}_{brand.lower()}"
    return CohortSpec(
        name=key,
        target=key,
        brand=brand,
        label_column=_PATIENT_LABELS[cohort],
        grain="patient",
        base_covariates=_BASE3,
    )


# ---------------------------------------------------------------------------
# Per-brand HCP-grain adoption cohort factory (HCP-T3 — the 4th cohort × 3 brands)
# ---------------------------------------------------------------------------
# Grain = (hcp_id, brand) on hcp_brand_adoption (migration 076). The label is
# ``adopted`` (0/1, from the shared leakage-safe _compute_adoption DGP); the
# predictive covariates live on hcp_profiles and are JOIN-embedded at load time
# (FeatureBuilder HCP path). consideration_date is the walk-forward temporal axis
# (aliased to journey_start_date at load) — a row attribute, never a feature.
HCP_ADOPTION_COHORT = "hcp_adoption"
_HCP_COVARIATES = (
    "peer_influence_score",
    "influence_network_size",
    "years_experience",
    "specialty",
    "geographic_region",
)


def make_hcp_spec(brand: str) -> CohortSpec:
    """Build a per-brand CohortSpec for the HCP-grain adoption cohort.

    brand in BRANDS.  target/name are uniform f"hcp_adoption_{brand.lower()}" so
    the 3 per-brand HCP models register cleanly alongside the 9 patient models.
    grain="hcp" routes FeatureBuilder to the hcp_brand_adoption + hcp_profiles
    load path (see FeatureBuilder.load_frame); base_covariates are the 5
    leakage-safe HCP attributes JOINed from hcp_profiles.
    """
    if brand not in BRANDS:
        raise ValueError(f"unknown brand {brand!r}")
    key = f"hcp_adoption_{brand.lower()}"
    return CohortSpec(
        name=key,
        target=key,
        brand=brand,
        label_column="adopted",
        grain="hcp",
        base_covariates=_HCP_COVARIATES,
    )


def goldstd_model_name(cohort: str, brand: str) -> str:
    """Return the canonical ml_model_registry model_name for a per-brand slot."""
    return f"{cohort}_{brand.lower()}_goldstd_lr_v1"


def goldstd_experiment_name(cohort: str, brand: str) -> str:
    """Return the canonical ml_experiments experiment_name for a per-brand slot."""
    return f"{cohort}_{brand.lower()}_goldstd_eval_v1"
