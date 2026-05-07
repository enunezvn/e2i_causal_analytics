"""Optum patient_journeys feature manifest — Layer 1.3 audit (companion to CSU).

Catalogs every column emitted by ``scripts/convert_optum_rwd.py``'s
``_compute_features`` plus the journey-record assembler. The Optum surface
is much wider than CSU (~100 model-input columns vs CSU's ~14) because the
Optum cohort has labs, comorbidity codes, drug classes, and provider mix
features that CSU's vendor data lacks.

The manifest follows the same vocabulary as the CSU manifest:
- ``KnowableAt(reference="enrollment")`` — pre-prediction-time demographics
- ``KnowableAt(reference="index_date") + window_days=180`` — windowed event
  aggregations (lookback applied in ``_compute_features``)
- ``KnowableAt(reference="post_index")`` — journey metadata + targets;
  these MUST NOT be passed to a model.

Several feature families are expanded via helper functions:
- 8 comorbidity prefixes × 2 columns each (has_X + X_claim_count)
- 7 non-target drug classes × 4 columns each (ever/count/days/days_since_last)
- 8 lab panels × 3 columns each (tested/result_last/abnormal_flag)
"""

from __future__ import annotations

from src.data.feature_contract import FeatureContract, KnowableAt

OPTUM_LOOKBACK_DAYS = 180


# Mirror of convert_optum_rwd.py constants. Kept duplicated here (rather than
# imported) so the manifest is self-contained — if the converter changes,
# the audit test catches the drift.
COMORBIDITY_NAMES = (
    "atopic_dermatitis",
    "asthma",
    "allergic_rhinitis",
    "anxiety",
    "depression",
    "thyroid_autoimmune",
    "nsaid_hypersensitivity",
    "angioedema",
)

LAB_NAMES = (
    "ige_total",
    "eosinophil",
    "crp",
    "tpo_ab",
    "free_t4",
    "tsh",
    "ana",
    "cbc",
)

DRUG_CLASS_NAMES = (
    "h1_1g",
    "h1_2g",
    "h2",
    "ltra",
    "sys_steroid",
    "top_steroid",
    "immunosupp",
)


# =============================================================================
# Demographics — knowable at enrollment
# =============================================================================

_DEMO = [
    FeatureContract(
        name="age_at_index",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("age",),
    ),
    FeatureContract(
        name="age_group",
        knowable_at=KnowableAt(reference="enrollment"),
        source="derived",
        derivation_inputs=("age_at_index",),
    ),
    FeatureContract(
        name="gender",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("gdr_cd",),
    ),
    FeatureContract(
        name="zip5",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("zipcode_5",),
    ),
    FeatureContract(
        name="zip3",
        knowable_at=KnowableAt(reference="enrollment"),
        source="derived",
        derivation_inputs=("zip5",),
    ),
    FeatureContract(
        name="zip_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="derived",
        derivation_inputs=("zip5",),
    ),
    FeatureContract(
        name="geographic_region",
        knowable_at=KnowableAt(reference="enrollment"),
        source="derived",
        derivation_inputs=("zip5",),
    ),
    FeatureContract(
        name="insurance_product",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("bus",),
    ),
    FeatureContract(
        name="plan_type",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("product",),
    ),
    FeatureContract(
        name="urban_rural_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="derived",
        derivation_inputs=("zip3",),
    ),
    FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode_raw",),
    ),
]


# =============================================================================
# Disease characteristics — windowed lookback
# =============================================================================

_DISEASE = [
    FeatureContract(
        name="dx_l50_1_count",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="dx_l50_8_count",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="dx_l50_9_count",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="dx_total_csu",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("dx_l50_1_count", "dx_l50_8_count", "dx_l50_9_count"),
    ),
    FeatureContract(
        name="dx_angioedema_count",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="months_since_first_dx",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=(),
    ),
    FeatureContract(
        name="csu_chronicity",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=(),
    ),
]


# =============================================================================
# Comorbidities — 8 names × 2 columns each (windowed)
# =============================================================================

_COMORBIDITIES: list[FeatureContract] = []
for name in COMORBIDITY_NAMES:
    _COMORBIDITIES.append(
        FeatureContract(
            name=f"has_{name}",
            knowable_at=KnowableAt(reference="index_date"),
            source="diagnosis_events",
            derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
            aggregation="max",
            window_days=OPTUM_LOOKBACK_DAYS,
        )
    )
    _COMORBIDITIES.append(
        FeatureContract(
            name=f"{name}_claim_count",
            knowable_at=KnowableAt(reference="index_date"),
            source="diagnosis_events",
            derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
            aggregation="count",
            window_days=OPTUM_LOOKBACK_DAYS,
        )
    )

_COMORBIDITY_DERIVED = [
    FeatureContract(
        name="atopy_score",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("has_atopic_dermatitis", "has_asthma", "has_allergic_rhinitis"),
    ),
    FeatureContract(
        name="mental_health_flag",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("has_anxiety", "has_depression"),
    ),
    FeatureContract(
        name="elixhauser_score",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
        aggregation="sum",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="charlson_score",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
        aggregation="sum",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
]


# =============================================================================
# Healthcare utilization — windowed
# =============================================================================

_UTILIZATION = [
    FeatureContract(
        name="office_visits_total",
        knowable_at=KnowableAt(reference="index_date"),
        source="procedure_events",
        derivation_inputs=("proc_date", "proc_code"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="office_visits_allergist",
        knowable_at=KnowableAt(reference="index_date"),
        source="procedure_events",
        derivation_inputs=("proc_date", "proc_code", "npi"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="office_visits_dermatology",
        knowable_at=KnowableAt(reference="index_date"),
        source="procedure_events",
        derivation_inputs=("proc_date", "proc_code", "npi"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="office_visits_pcp",
        knowable_at=KnowableAt(reference="index_date"),
        source="procedure_events",
        derivation_inputs=("proc_date", "proc_code", "npi"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="ed_visits_total",
        knowable_at=KnowableAt(reference="index_date"),
        source="encounter_events",
        derivation_inputs=("admit_date", "tos_cd"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="ed_visits_urticaria_angio",
        knowable_at=KnowableAt(reference="index_date"),
        source="encounter_events",
        derivation_inputs=("admit_date", "tos_cd", "diag1"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="hospitalizations_total",
        knowable_at=KnowableAt(reference="index_date"),
        source="encounter_events",
        derivation_inputs=("admit_date",),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="unique_providers",
        knowable_at=KnowableAt(reference="index_date"),
        source="procedure_events",
        derivation_inputs=("proc_date", "npi"),
        aggregation="nunique",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
]


# =============================================================================
# Non-target drug classes — 7 classes × 4 columns each
# =============================================================================

_DRUG_CLASS: list[FeatureContract] = []
for cls in DRUG_CLASS_NAMES:
    _DRUG_CLASS.append(
        FeatureContract(
            name=f"{cls}_ever_filled",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=("medication_date", "drug_name"),
            aggregation="max",
            window_days=OPTUM_LOOKBACK_DAYS,
        )
    )
    _DRUG_CLASS.append(
        FeatureContract(
            name=f"{cls}_fill_count",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=("medication_date", "drug_name"),
            aggregation="count",
            window_days=OPTUM_LOOKBACK_DAYS,
        )
    )
    _DRUG_CLASS.append(
        FeatureContract(
            name=f"{cls}_days_supply_total",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=("medication_date", "drug_name", "days_sup"),
            aggregation="sum",
            window_days=OPTUM_LOOKBACK_DAYS,
        )
    )
    _DRUG_CLASS.append(
        FeatureContract(
            name=f"{cls}_days_since_last_fill",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=("medication_date", "drug_name"),
            aggregation="max",
            window_days=OPTUM_LOOKBACK_DAYS,
        )
    )


# =============================================================================
# Lab panels — 8 panels × 3 columns each
# =============================================================================

_LABS: list[FeatureContract] = []
for lab in LAB_NAMES:
    _LABS.append(
        FeatureContract(
            name=f"{lab}_tested",
            knowable_at=KnowableAt(reference="index_date"),
            source="lab_events",
            derivation_inputs=("fst_dt", "loinc_cd"),
            aggregation="max",
            window_days=OPTUM_LOOKBACK_DAYS,
        )
    )
    _LABS.append(
        FeatureContract(
            name=f"{lab}_result_last",
            knowable_at=KnowableAt(reference="index_date"),
            source="lab_events",
            derivation_inputs=("fst_dt", "loinc_cd", "result"),
            aggregation="max",
            window_days=OPTUM_LOOKBACK_DAYS,
        )
    )
    _LABS.append(
        FeatureContract(
            name=f"{lab}_abnormal_flag",
            knowable_at=KnowableAt(reference="index_date"),
            source="lab_events",
            derivation_inputs=("fst_dt", "loinc_cd", "abnl_cd"),
            aggregation="max",
            window_days=OPTUM_LOOKBACK_DAYS,
        )
    )


# =============================================================================
# Provider mix — windowed
# =============================================================================

_PROVIDER = [
    FeatureContract(
        name="specialist_concentration",
        knowable_at=KnowableAt(reference="index_date"),
        source="procedure_events",
        derivation_inputs=("proc_date", "npi"),
        aggregation="sum",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="primary_specialist_type",
        knowable_at=KnowableAt(reference="index_date"),
        source="procedure_events",
        derivation_inputs=("proc_date", "npi"),
        aggregation="max",
        window_days=OPTUM_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="saw_allergist_flag",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("primary_specialist_type",),
    ),
    FeatureContract(
        name="saw_dermatologist_flag",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("primary_specialist_type",),
    ),
]


# =============================================================================
# Index/lookback dates — at index_date by definition (NOT post-index)
# =============================================================================

_DATES = [
    FeatureContract(
        name="index_date",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=(),
    ),
    FeatureContract(
        name="lookback_start_date",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("index_date",),
    ),
]


# =============================================================================
# Post-index — FORBIDDEN as features (journey metadata + targets)
# =============================================================================

_POST_INDEX_FORBIDDEN = [
    FeatureContract(
        name="prediction_end_date",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("index_date",),
    ),
    FeatureContract(
        name="journey_start_date",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("index_date",),
    ),
    FeatureContract(
        name="journey_end_date",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("eligend",),
    ),
    FeatureContract(
        name="journey_duration_days",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("journey_start_date", "journey_end_date"),
    ),
    FeatureContract(
        name="journey_stage",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("treatment_initiated",),
    ),
    FeatureContract(
        name="journey_status",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("treatment_initiated", "discontinuation_flag"),
    ),
    # Targets
    FeatureContract(
        name="treatment_initiated",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("medication_date",),
    ),
    FeatureContract(
        name="initiated_biologic_180d",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("medication_date",),
    ),
    FeatureContract(
        name="discontinuation_flag",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("medication_date", "days_sup"),
    ),
    FeatureContract(
        name="discontinued_180d",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("medication_date", "days_sup"),
    ),
    FeatureContract(
        name="persistent_at_180d",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("medication_date", "days_sup"),
    ),
    # `brand` — same target-coupling pattern as CSU.
    FeatureContract(
        name="brand",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("treatment_initiated",),
    ),
]


OPTUM_FEATURES: list[FeatureContract] = (
    _DEMO
    + _DISEASE
    + _COMORBIDITIES
    + _COMORBIDITY_DERIVED
    + _UTILIZATION
    + _DRUG_CLASS
    + _LABS
    + _PROVIDER
    + _DATES
    + _POST_INDEX_FORBIDDEN
)


def optum_contract_for(name: str) -> FeatureContract | None:
    """Return the FeatureContract for a named Optum feature, or None if absent."""
    for c in OPTUM_FEATURES:
        if c.name == name:
            return c
    return None


OPTUM_SAFE_FEATURES: list[str] = [
    c.name for c in OPTUM_FEATURES if c.knowable_at.is_pre_or_at_index()
]

OPTUM_FORBIDDEN_AS_FEATURES: list[str] = [
    c.name for c in OPTUM_FEATURES if not c.knowable_at.is_pre_or_at_index()
]
