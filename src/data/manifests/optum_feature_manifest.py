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

import dataclasses

from src.data.feature_contract import (
    CausalStructureAttestation,
    FeatureContract,
    KnowableAt,
)

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
#
# *** PR-C handoff note ***
# Several entries here use ``("UMLS", <CUI>)`` rather than a source-vocab
# code. ``EntityLinker.resolve(code, system)`` does NOT accept "UMLS" as a
# system (its `_UTS_SOURCE_BY_SYSTEM` only knows ICD10CM / RXNORM / LNC /
# CPT / HCPCS — it cross-walks source codes to CUIs, not CUIs to CUIs).
# PR-C's cache builder must special-case "UMLS" entries: skip the
# code-to-CUI cross-walk and call ``UMLSUTSClient.cui_lookup`` (or pass
# the CUI directly through to KGQuerier.query_drug_disease_edges /
# query_disease_hierarchy whose `cui` arguments accept bare CUIs). If
# PR-C uses EntityLinker.resolve indiscriminately, all 28 drug-class
# features + nsaid_hypersensitivity + IgE companion entries would
# silently degrade to no_signal.

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
    # Issue #156 item 6 — derived 8-vocabulary payer label
    # (commercial / commercial_exchange / medicare / medicare_advantage
    # / medicare_lis_dual / medicaid / cash / other). Computed at
    # cohort-build time by ``scripts.rwd_common.derive_payer_category``
    # from the raw demographics fields (bus / product / health_exch /
    # lis_dual). Knowable at enrollment because every input is a
    # static demographics field. Added to the manifest 2026-05-15 by
    # the Layer 1 manifest-coverage CI guard (Phase 1.5) which surfaced
    # the missing entry as the only real (non-allowlisted, non-audit)
    # uncovered column in the Optum journey output.
    FeatureContract(
        name="payer_category",
        knowable_at=KnowableAt(reference="enrollment"),
        source="derived",
        derivation_inputs=("bus", "product", "health_exch", "lis_dual"),
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
# v5 Gate B3 — engineered features (derived from pre-anchor inputs)
# =============================================================================
#
# Each engineered feature is computed by
# ``src.agents.ml_foundation.data_preparer.nodes.feature_engineering``
# (see ``OPTUM_ENGINEERED_FEATURES`` there) and inherits pre-anchor status
# from its declared ``derivation_inputs`` — every input listed below is
# itself declared pre-anchor in this manifest.
#
# Pre-spec: ``docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md``.
# Audit gate: Layer 1 traces the derivation chain (this declaration);
# Layer 3 (production adversarial probe) runs on the materialized column
# during ``adaptive_validity_check``. Both must pass before the feature
# is permitted to influence val_AUC.

_ENGINEERED_B3 = [
    FeatureContract(
        name="comorbidity_load_total",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=tuple(f"has_{c}" for c in COMORBIDITY_NAMES),
    ),
    FeatureContract(
        name="csu_dx_intensity",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("dx_total_csu", "months_since_first_dx"),
    ),
    FeatureContract(
        name="polypharmacy_breadth",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=tuple(f"{d}_ever_filled" for d in DRUG_CLASS_NAMES),
    ),
    FeatureContract(
        name="lab_workup_completeness",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=tuple(f"{lab}_tested" for lab in LAB_NAMES),
    ),
    FeatureContract(
        name="specialist_visit_interaction",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("office_visits_allergist", "office_visits_dermatology"),
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
    + _ENGINEERED_B3
    + _POST_INDEX_FORBIDDEN
)


# =============================================================================
# Causal-structure attestations — Layer-4 structural decider (Track-2B-v3 Phase 2)
# =============================================================================
#
# Every SAFE (pre/at-index) Optum feature is enriched — at the ``optum_contract_for``
# accessor below, which keeps the ``OPTUM_FEATURES`` registry statically traceable
# for the Layer-1 coverage guard — with the DAG fragment from
# which ``src.ml.causal_role_dgp.extractor.extract_role`` DERIVES its causal role
# relative to (T=biologic_initiation, Y=initiated_biologic_180d). Optum-initiation
# is a dx-anchored, treatment-naive PREDICTION cohort, so every legitimate
# pre-index feature is a confounder or instrument of the biologic-initiation
# decision (all ACCEPT) — none is outcome-derived. Verified against
# ``scripts/convert_optum_rwd.py``: feature windows are strictly pre-index
# ([index-180d, index-1d]) and biologic rows are stripped from the drug-class
# features, so no SAFE feature can encode the index biologic (leakage is
# structurally precluded, not merely absent). Per-feature literature grounding:
# ``docs/layer4/optum_initiation_attestation_research.md``.
#
# Post-index FORBIDDEN columns are NOT model inputs and are left un-attested
# (they fall to the manifest's forbidden gate). The decider stays DARK until an
# explicit, cohort-scoped ramp; these attestations are inert today.
#
# Role is DERIVED from edges, never declared. Two patterns:
#   confounder: feature->T, feature->Y, T->Y  (disease-severity / treatment-burden proxies)
#   instrument: feature->T, T->Y               (access/geography/payer/calendar drivers of T only)
_OPTUM_TREATMENT_NODE = "biologic_initiation"
_OPTUM_OUTCOME_NODE = "initiated_biologic_180d"

# SAFE features whose honest mechanism acts on the initiation DECISION only
# (no direct path to the outcome): geography/residence (specialist proximity &
# regional adoption), coverage/payer (formulary, prior-auth, step-therapy),
# specialist-access (the biologic-prescribing channel), and calendar anchors
# (temporal biologic adoption). Everything else SAFE is a confounder.
_OPTUM_INSTRUMENT_FEATURES: frozenset[str] = frozenset(
    {
        "zip5",
        "zip3",
        "zip_code",
        "geographic_region",
        "urban_rural_code",
        "insurance_product",
        "plan_type",
        "payer_category",
        "office_visits_allergist",
        "office_visits_dermatology",
        "specialist_concentration",
        "primary_specialist_type",
        "saw_allergist_flag",
        "saw_dermatologist_flag",
        "specialist_visit_interaction",
        "index_date",
        "lookback_start_date",
    }
)


def _optum_attestation(feature_node: str) -> CausalStructureAttestation:
    """Authored DAG fragment for a SAFE Optum feature (role derived, not declared).

    Instruments drive only the treatment decision (``feature->T``); confounders are
    common causes of the decision AND the outcome (``feature->T``, ``feature->Y``).
    Always includes the treatment-effect edge ``T->Y``.
    """
    t, y = _OPTUM_TREATMENT_NODE, _OPTUM_OUTCOME_NODE
    if feature_node in _OPTUM_INSTRUMENT_FEATURES:
        edges: tuple[tuple[str, str], ...] = ((feature_node, t), (t, y))
    else:
        edges = ((feature_node, t), (feature_node, y), (t, y))
    return CausalStructureAttestation(
        treatment_node=t,
        outcome_node=y,
        feature_node=feature_node,
        edges=edges,
    )


def optum_contract_for(name: str) -> FeatureContract | None:
    """Return the FeatureContract for a named Optum feature, or None if absent.

    SAFE (pre/at-index) features are ENRICHED here with their structural
    ``CausalStructureAttestation`` (Track-2B-v3 Phase 2). This accessor is the
    canonical lookup the Layer-4 structural decider reaches via
    ``lookup_feature_contract``, so attaching the attestation here keeps the
    ``OPTUM_FEATURES`` registry a purely declarative, statically-analyzable list
    that the Layer-1 manifest-coverage guard (``scripts/check_manifest_coverage.py``)
    can trace — a list-comprehension rebuild of the registry is an unsupported
    binding shape for that guard's AST tracer. Post-index FORBIDDEN columns are
    never enriched; the role is DERIVED from the authored edges, never declared.
    """
    for c in OPTUM_FEATURES:
        if c.name == name:
            if c.knowable_at.is_pre_or_at_index() and c.causal_structure is None:
                return dataclasses.replace(c, causal_structure=_optum_attestation(c.name))
            return c
    return None


OPTUM_SAFE_FEATURES: list[str] = [
    c.name for c in OPTUM_FEATURES if c.knowable_at.is_pre_or_at_index()
]

OPTUM_FORBIDDEN_AS_FEATURES: list[str] = [
    c.name for c in OPTUM_FEATURES if not c.knowable_at.is_pre_or_at_index()
]

# Targets across all three Optum cohorts (initiation / discontinuation /
# persistence). These are FORBIDDEN as features (post-index labels) but
# MUST NOT be dropped at cohort-build time so the downstream pipeline can
# extract the supervised signal via ``scope_spec.prediction_target``.
# Cohort-builder gates filter forbidden columns while explicitly
# preserving everything in this set.
#
# Maintenance: add to this set when introducing a new prediction target;
# the test ``test_targets_subset_of_forbidden`` enforces that every entry
# is also in ``OPTUM_FORBIDDEN_AS_FEATURES``.
OPTUM_TARGETS: frozenset[str] = frozenset(
    {
        "treatment_initiated",
        "initiated_biologic_180d",
        "discontinuation_flag",
        "discontinued_180d",
        "persistent_at_180d",
    }
)

# Forbidden columns that are NOT targets — the safe-to-drop set for the
# cohort-builder gate.
OPTUM_FORBIDDEN_NON_TARGET: list[str] = [
    f for f in OPTUM_FORBIDDEN_AS_FEATURES if f not in OPTUM_TARGETS
]
