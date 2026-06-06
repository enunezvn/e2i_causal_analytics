"""Tests for the Optum MART feature manifest.

The new Optum drop (``data/rwd/Optum_Parquet/Optum.parquet``) is an entity-stacked,
pre-engineered mart that shares only 4 of the 110 ``optum_feature_manifest`` SAFE
column names. It bypasses ``convert_optum_rwd.py``'s structural leakage gate entirely,
so THIS manifest is the authority for which mart columns are pre-index-admissible vs
post-index leakage for the INITIATION cohort.

Design + verified facts: ``.claude/plans/optum-initiation-adapter/IMPLEMENTATION-PLAN.md``.
Key load-bearing facts (measured against the live mart this session):
- ``index_biologic_brand != 'no_treatment'`` reproduces the initiation target EXACTLY
  (24,429 ever-treated) -> it is a deterministic target alias = post-index leakage.
- ``charlson_score`` / ``cci_*`` / ``elx_*`` correlate with PRE-index enrollment length
  but ~0 with POST-index follow-up (corr <= 0.07) -> disproof-cleared as pre-index.
"""

from src.data.manifests.optum_mart_feature_manifest import (
    MART_FORBIDDEN_AS_FEATURES,
    MART_SAFE_FEATURES,
    MART_TARGETS,
    optum_mart_contract_for,
)


def test_target_alias_index_biologic_brand_is_post_index_leakage():
    """index_biologic_brand reproduces the initiation target -> post-index/forbidden."""
    contract = optum_mart_contract_for("index_biologic_brand")
    assert contract is not None, "manifest must declare the target-alias column"
    assert not contract.knowable_at.is_pre_or_at_index()
    assert "index_biologic_brand" in MART_FORBIDDEN_AS_FEATURES
    assert "index_biologic_brand" not in MART_SAFE_FEATURES


def test_baseline_comorbidity_charlson_is_pre_index_safe():
    """charlson_score is disproof-cleared as pre-index (corr w/ post-index follow-up ~0)."""
    contract = optum_mart_contract_for("charlson_score")
    assert contract is not None
    assert contract.knowable_at.is_pre_or_at_index()
    assert "charlson_score" in MART_SAFE_FEATURES


def test_initiation_target_declared():
    """The initiation supervised label must be a declared target (forbidden-but-preserved)."""
    assert "initiated_biologic_180d" in MART_TARGETS
    assert "initiated_biologic_180d" in MART_FORBIDDEN_AS_FEATURES


# --- The approved pre-index admissible allow-list (owner-reviewed 2026-06-06) ---
# 7 demographics + 2 derived (geographic_region, enrollment_duration_days) + 55 baseline comorbidity.
_CCI = [
    "cci_mi",
    "cci_chf",
    "cci_pvd",
    "cci_cerebrovascular",
    "cci_dementia",
    "cci_chronic_pulmonary",
    "cci_rheumatic",
    "cci_peptic_ulcer",
    "cci_mild_liver",
    "cci_diabetes_no_complication",
    "cci_diabetes_complication",
    "cci_paraplegia",
    "cci_renal",
    "cci_malignancy",
    "cci_severe_liver",
    "cci_metastatic_cancer",
    "cci_hiv",
]
_ELX = [
    "elx_chf",
    "elx_cardiac_arrhythmia",
    "elx_valvular_disease",
    "elx_pulmonary_circulation",
    "elx_pvd",
    "elx_hypertension_uncomplicated",
    "elx_hypertension_complicated",
    "elx_paralysis",
    "elx_other_neurological",
    "elx_chronic_pulmonary",
    "elx_diabetes_uncomplicated",
    "elx_diabetes_complicated",
    "elx_hypothyroidism",
    "elx_renal_failure",
    "elx_liver_disease",
    "elx_peptic_ulcer",
    "elx_aids_hiv",
    "elx_lymphoma",
    "elx_metastatic_cancer",
    "elx_solid_tumor_no_metastasis",
    "elx_rheumatoid_collagen",
    "elx_coagulopathy",
    "elx_obesity",
    "elx_weight_loss",
    "elx_fluid_electrolyte",
    "elx_blood_loss_anemia",
    "elx_deficiency_anemia",
    "elx_alcohol_abuse",
    "elx_drug_abuse",
    "elx_psychoses",
    "elx_depression",
]
EXPECTED_SAFE = set(
    [
        "age_at_index",
        "gdr_cd",
        "payer_category",
        "payer_product",
        "payer_bus",
        "health_exchange_flag",
        "lis_dual_flag",
        "geographic_region",
        "enrollment_duration_days",  # derived
        "charlson_score",
        "charlson_risk_band",
        "elixhauser_van_walraven_score",
        "elixhauser_risk_band",
        "comorbidity_diag_distinct_count",
        "comorbidity_diag_claim_count",
        "high_comorbidity_burden_flag",
    ]
    + _CCI
    + _ELX
)


def test_safe_features_match_approved_allowlist_exactly():
    """The pre-index admissible set must be EXACTLY the owner-approved 64-column allow-list."""
    assert set(MART_SAFE_FEATURES) == EXPECTED_SAFE
    assert len(EXPECTED_SAFE) == 64


def test_all_comorbidity_flags_are_pre_index():
    for name in _CCI + _ELX:
        contract = optum_mart_contract_for(name)
        assert contract is not None, f"{name} must be declared"
        assert contract.knowable_at.is_pre_or_at_index(), f"{name} must be pre-index"


def test_dropped_and_filter_columns_are_not_features():
    """yrdob (collinear), raw zipcode_5 (PHI), record-counts (filter-only), post-index anchors -> not features."""
    for name in [
        "yrdob",
        "zipcode_5",
        "claim_record_count",
        "diagnosis_record_count",
        "procedure_record_count",
        "cost_record_count",
        "last_csu_dx_date",
        "elig_end_date",
        "data_quality_band",
        "data_quality_score",
    ]:
        assert name not in MART_SAFE_FEATURES, f"{name} must NOT be a model feature"


def test_proven_target_aliases_are_forbidden():
    """index_biologic_brand / treatment_start_date deterministically reproduce the label -> forbidden."""
    for name in ["index_biologic_brand", "treatment_start_date"]:
        contract = optum_mart_contract_for(name)
        assert contract is not None
        assert not contract.knowable_at.is_pre_or_at_index()
        assert name in MART_FORBIDDEN_AS_FEATURES


def test_manifest_registered_as_resolvable_source():
    """The mart manifest must be registered so Layer-5 declared-safe immunity engages.

    Registered as a DISTINCT source ('optum_mart') — the mart is the same cohort
    but a structurally different schema (4/110 column overlap with the converter
    output), so it cannot share the converter's 'optum' FeatureContract list.
    """
    from src.data.manifests import MANIFEST_SOURCES, lookup_feature_contract

    assert "optum_mart" in MANIFEST_SOURCES
    resolved = lookup_feature_contract("charlson_score", "optum_mart")
    assert resolved is not None
    assert resolved.knowable_at.is_pre_or_at_index()
    # A converter-output-only column is absent from the mart registry.
    assert lookup_feature_contract("dx_l50_x_count", "optum_mart") is None


# --- Multi-cohort extension (discontinuation / persistence; frame-shift) ---


def test_multicohort_targets_declared_post_index():
    """discontinued_180d / persistent_at_180d are post-index targets (forbidden-but-preserved)."""
    for t in ("discontinued_180d", "persistent_at_180d"):
        assert t in MART_TARGETS, f"{t} must be a declared target"
        c = optum_mart_contract_for(t)
        assert c is not None, f"{t} must be declared"
        assert not c.knowable_at.is_pre_or_at_index(), f"{t} must be post-index"
        assert t in MART_FORBIDDEN_AS_FEATURES
        assert t not in MART_SAFE_FEATURES


def test_disc_persist_derivation_aliases_are_forbidden():
    """The coverage/gap derivation inputs + precomputed outcome flags that
    (near-)deterministically reproduce the disc/persistence targets must be
    post-index forbidden aliases (defense-in-depth; never emitted by the adapter)."""
    for name in (
        "last_coverage_end",
        "last_observed_date",
        "max_internal_gap_days",
        "terminal_gap_days",
        "covered_days",
        "pdc",
        "discontinued_flag",
        "discontinued_90d_flag",
        "persistence_60d_flag",
        "maintained_flag",
        "adherent_flag",
    ):
        c = optum_mart_contract_for(name)
        assert c is not None, f"{name} must be declared as a forbidden alias"
        assert not c.knowable_at.is_pre_or_at_index(), f"{name} must be post-index"
        assert name in MART_FORBIDDEN_AS_FEATURES
        assert name not in MART_SAFE_FEATURES


def test_safe_allowlist_unchanged_at_64_across_cohorts():
    """Adding disc/persistence targets + aliases must NOT change the 64-feature
    safe set — the same baseline features are valid pre-index for all 3 cohorts."""
    assert len(MART_SAFE_FEATURES) == 64


def test_multicohort_targets_resolve_via_manifest_source():
    """Layer-5 resolution sees the new targets under the optum_mart source (lockstep)."""
    from src.data.manifests import lookup_feature_contract

    for t in ("discontinued_180d", "persistent_at_180d"):
        c = lookup_feature_contract(t, "optum_mart")
        assert c is not None and not c.knowable_at.is_pre_or_at_index()
