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
# T9 (2026-06-21): persistence/discontinuation depend on 7 leakage-safe covariates
# after the DGP enrichment (insurance access, age, comorbidity burden, prior-therapy
# lines added as prognostic drivers in cohort_outcomes.py).
# T11 (2026-06-22): initiation now ALSO uses the 7-covariate set — the same 4
# prognostic drivers were added to the treatment_initiated outcome equation
# (binary_outcome_with_cate via initiation_prognostic_offset), drawn ⊥ treatment_arm
# so ATE/CATE recovery is preserved. (geographic_region was never in the initiation
# eqn — region≈0 — so the lift comes from the 4 NEW drivers, not region.) The
# superseded pooled PERSISTENCE/DISCONTINUATION specs above stay at 3 (not retrained
# by run_patient_cohorts).
_BASE7 = _BASE3 + (
    "insurance_type",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
)
# COMM-ARMS Phase 1 (2026-07-19): persistence/discontinuation gain an 8th covariate,
# copay_support. copay enters the DISCONTINUATION logit, so it is genuine outcome
# signal; it is assigned pre-index and is NOT a leakage column, so the model may
# legitimately observe it. Withholding it made copay irreducible noise and dropped the
# faithful holdout AUC ~0.03 (Kisqali to within 0.0005 of the gate floor).
#
# initiation deliberately stays at _BASE7: copay does NOT enter the treatment_initiated
# equation (verified — that column is byte-identical pre/post copay), so adding it there
# would only feed the model a pure-noise feature.
#
# MUST stay in lockstep with MODEL_FEATURE_REFS in src/feature_store/model_feature_refs.py
# and the Field list in feature_repo/features/goldstd_cohort_features.py. A spec that
# declares more covariates than the refs fetch hands the serving bundle an incomplete
# vector (#576 null-trap → 503); test_model_feature_refs.py locks the three together.
_BASE8_COMMERCIAL = _BASE7 + ("copay_support",)
# COMM-ARMS Phase 2 (2026-07-19): persistence/discontinuation gain a 9th covariate,
# psp_enrolled. Same rationale as copay (line above): psp enters the discontinuation
# logit, is assigned pre-index, is not a leakage column, and letting the model observe
# it recovers ~+0.004 faithful holdout AUC (measured). initiation stays at _BASE7 (psp
# is not in the treatment_initiated equation).
_BASE9_COMMERCIAL = _BASE8_COMMERCIAL + ("psp_enrolled",)
# COMM-ARMS Phase 3 (2026-07-20): INITIATION gains rep_detailing_high + sample_dropped.
# This is the MIRROR IMAGE of copay/psp above: those two enter the discontinuation logit
# (persistence/discontinuation cohorts) and NOT the initiation equation, so they were
# added to _BASE9_COMMERCIAL but withheld from initiation. rep/sample are the opposite —
# they enter ONLY the treatment_initiated latent (initiation_outcomes), so they belong on
# initiation and are withheld from persistence/discontinuation (adding them there would
# feed those models a pure-noise feature). Same lockstep contract as copay/psp: a spec
# covariate the refs don't fetch => incomplete serving vector (#576 null-trap → 503).
_BASE9_INITIATION = _BASE7 + ("rep_detailing_high", "sample_dropped")
# COMM-ARMS Phase 4 (2026-07-20): INITIATION gains trigger_accepted — the NBA
# trigger-acceptance arm, which enters ONLY the treatment_initiated latent
# (initiation_outcomes), exactly like rep/sample above. Same lockstep contract:
# a spec covariate the refs don't fetch => incomplete serving vector (#576
# null-trap → 503); test_model_feature_refs.py locks spec/refs/Feast together.
_BASE10_INITIATION = _BASE9_INITIATION + ("trigger_accepted",)
_PATIENT_COVARIATES: dict[str, tuple[str, ...]] = {
    "initiation": _BASE10_INITIATION,
    "persistence": _BASE9_COMMERCIAL,
    "discontinuation": _BASE9_COMMERCIAL,
}


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
        base_covariates=_PATIENT_COVARIATES[cohort],
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
