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
# KG entity-code lookups (Phase 2.9 Stage 2 PR-B)
# =============================================================================
#
# Each table maps a feature family name to its canonical KG entity codes:
# the most-specific source-vocab code (ICD10CM / RxNORM / LOINC) AND a UMLS
# CUI cross-walk. The KG querier resolves either to drug-disease evidence
# (Open Targets) or taxonomic relations (UMLS); UMLS CUI is the canonical
# joiner.
#
# Drug classes intentionally use UMLS class CUIs (not RxCUIs): RxNorm
# class-level membership is fuzzy at the active-ingredient level, but UMLS
# captures the pharmacologic class as a single concept that the
# KGQuerier.query_concept_relations call can navigate.

PRIMARY_DX_KG_CODES: tuple[tuple[str, str], ...] = (
    ("ICD10CM", "L50.9"),
    ("UMLS", "C0042109"),
)

DX_SPECIFIC_KG_CODES: dict[str, tuple[tuple[str, str], ...]] = {
    "dx_l50_1_count": (("ICD10CM", "L50.1"), ("UMLS", "C0042109")),
    "dx_l50_8_count": (("ICD10CM", "L50.8"), ("UMLS", "C0042109")),
    "dx_l50_9_count": (("ICD10CM", "L50.9"), ("UMLS", "C0042109")),
}

DX_ANGIOEDEMA_KG_CODES: tuple[tuple[str, str], ...] = (
    ("ICD10CM", "T78.3"),
    ("UMLS", "C0002994"),
)

COMORBIDITY_KG_CODES: dict[str, tuple[tuple[str, str], ...]] = {
    "atopic_dermatitis": (("ICD10CM", "L20.9"), ("UMLS", "C0011615")),
    "asthma": (("ICD10CM", "J45.909"), ("UMLS", "C0004096")),
    "allergic_rhinitis": (("ICD10CM", "J30.9"), ("UMLS", "C0018621")),
    "anxiety": (("ICD10CM", "F41.9"), ("UMLS", "C0003467")),
    "depression": (("ICD10CM", "F33.9"), ("UMLS", "C0011581")),
    # CUI C0856243 expected to map to autoimmune thyroiditis (E06.3).
    # PR-C cache builder will validate via UMLS UTS; if it's not the
    # right CUI the build fails loudly. Alternatives: C0040429
    # (Thyroiditis, Autoimmune) or C0677607 (Hashimoto Disease).
    "thyroid_autoimmune": (("ICD10CM", "E06.3"), ("UMLS", "C0856243")),
    # Plan included ICD10CM T88.7 ("Other adverse effects, NEC") but
    # we drop it: T88.7 maps to ALL drug ADRs and is too broad for
    # KG disambiguation. C2266824 narrows to NSAID-exacerbated
    # respiratory disease (AERD); broader NSAID-hypersensitivity CUIs
    # exist (e.g. C0338513 Drug Hypersensitivity) and cache builder
    # will surface mismatches at validation time.
    "nsaid_hypersensitivity": (("UMLS", "C2266824"),),
    "angioedema": (("ICD10CM", "T78.3"), ("UMLS", "C0002994")),
}

# Drug-class CUIs are intentionally pharmacologic-class concepts, not RxCUI
# ingredients (RxNorm class membership is fuzzy at the active-ingredient
# level). PR-C cache builder will re-resolve each CUI via UMLS UTS at
# build time so any mis-CUI surfaces as a build failure.
#  - C0001617 ("Adrenal Cortex Hormones") is the broad parent class —
#    used here for top_steroid because Open Targets evidence for topical
#    corticosteroids is not separately catalogued at the agent level.
#  - C2825472 is a more-specific NCI systemic-corticosteroid concept.
DRUG_CLASS_KG_CODES: dict[str, tuple[tuple[str, str], ...]] = {
    "h1_1g": (("UMLS", "C0066896"),),
    "h1_2g": (("UMLS", "C2718076"),),
    "h2": (("UMLS", "C0019613"),),
    "ltra": (("UMLS", "C0876129"),),
    "sys_steroid": (("UMLS", "C2825472"),),
    "top_steroid": (("UMLS", "C0001617"),),
    "immunosupp": (("UMLS", "C0021081"),),
}

LAB_KG_CODES: dict[str, tuple[tuple[str, str], ...]] = {
    "ige_total": (("LOINC", "6106-9"), ("UMLS", "C0922951")),
    "eosinophil": (("LOINC", "711-2"),),
    "crp": (("LOINC", "1988-5"),),
    "tpo_ab": (("LOINC", "9362-4"),),
    "free_t4": (("LOINC", "3024-7"),),
    "tsh": (("LOINC", "3016-3"),),
    "ana": (("LOINC", "5048-4"),),
    "cbc": (("LOINC", "58410-2"),),
}


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
        kg_entity_codes=PRIMARY_DX_KG_CODES,
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
        kg_entity_codes=DX_SPECIFIC_KG_CODES["dx_l50_1_count"],
    ),
    FeatureContract(
        name="dx_l50_8_count",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
        kg_entity_codes=DX_SPECIFIC_KG_CODES["dx_l50_8_count"],
    ),
    FeatureContract(
        name="dx_l50_9_count",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
        kg_entity_codes=DX_SPECIFIC_KG_CODES["dx_l50_9_count"],
    ),
    FeatureContract(
        name="dx_total_csu",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("dx_l50_1_count", "dx_l50_8_count", "dx_l50_9_count"),
        kg_entity_codes=(("UMLS", "C0042109"),),
    ),
    FeatureContract(
        name="dx_angioedema_count",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
        kg_entity_codes=DX_ANGIOEDEMA_KG_CODES,
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
    _kg_codes = COMORBIDITY_KG_CODES[name]
    _COMORBIDITIES.append(
        FeatureContract(
            name=f"has_{name}",
            knowable_at=KnowableAt(reference="index_date"),
            source="diagnosis_events",
            derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
            aggregation="max",
            window_days=OPTUM_LOOKBACK_DAYS,
            kg_entity_codes=_kg_codes,
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
            kg_entity_codes=_kg_codes,
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
    _drug_codes = DRUG_CLASS_KG_CODES[cls]
    _DRUG_CLASS.append(
        FeatureContract(
            name=f"{cls}_ever_filled",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=("medication_date", "drug_name"),
            aggregation="max",
            window_days=OPTUM_LOOKBACK_DAYS,
            kg_entity_codes=_drug_codes,
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
            kg_entity_codes=_drug_codes,
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
            kg_entity_codes=_drug_codes,
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
            kg_entity_codes=_drug_codes,
        )
    )


# =============================================================================
# Lab panels — 8 panels × 3 columns each
# =============================================================================

_LABS: list[FeatureContract] = []
for lab in LAB_NAMES:
    _lab_codes = LAB_KG_CODES[lab]
    _LABS.append(
        FeatureContract(
            name=f"{lab}_tested",
            knowable_at=KnowableAt(reference="index_date"),
            source="lab_events",
            derivation_inputs=("fst_dt", "loinc_cd"),
            aggregation="max",
            window_days=OPTUM_LOOKBACK_DAYS,
            kg_entity_codes=_lab_codes,
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
            kg_entity_codes=_lab_codes,
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
            kg_entity_codes=_lab_codes,
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
