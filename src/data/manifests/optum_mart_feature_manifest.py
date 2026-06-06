"""Feature manifest for the Optum MART (entity-stacked, pre-engineered drop).

The new Optum drop ``data/rwd/Optum_Parquet/Optum.parquet`` is a denormalized,
entity-stacked mart (252 cols x 3.76M rows) that shares only 4 of the 110
``optum_feature_manifest`` SAFE column names. It bypasses
``scripts/convert_optum_rwd.py``'s structural leakage gate entirely, so THIS
manifest is the authority for which mart columns are pre-index-admissible vs
post-index leakage for the mart-sourced tier-0 cohorts (INITIATION plus the
treatment-anchored DISCONTINUATION / PERSISTENCE cohorts).

Scope: this manifest declares the columns the mart adapter
(``scripts/convert_optum_mart.py``) emits into ``patient_journeys`` — the
owner-approved 64-column pre-index allow-list (SAME across all three cohorts:
baseline features measured at the dx index are knowable at the later
treatment-start anchor too) plus the supervised target of each cohort plus their
proven target-aliases (declared forbidden for defense-in-depth). The post-index
leakage columns are NOT emitted by the adapter, so they need no contract here;
the adapter's positive-enumeration allow-list is the structural barrier and this
manifest is the runtime cross-check that also grants declared-safe immunity to
the sparse rare-event comorbidity flags (preventing the Tier-0 over-drop failure
mode).

Authoring rules (mirror ``optum_feature_manifest.py``):
- ``OPTUM_MART_FEATURES`` is a statically-declared literal list of
  ``FeatureContract`` objects so ``scripts/check_manifest_coverage.py``'s AST
  tracer can verify coverage (NO list-comprehension rebuild of the registry).
- ``knowable_at`` is HONEST: ``enrollment`` for static demographics,
  ``index_date`` for pre-index baseline comorbidity/utilization knowable at the
  diagnosis index, ``post_index`` for the target + treatment-derived aliases.
- ``aggregation=None``: every mart column is PRECOMPUTED upstream; we do not
  re-derive, so we declare WHEN a value is knowable, not how we aggregate it.

Owner-reviewed allow-list (2026-06-06): 7 demographics + 2 derived
(``geographic_region``, ``enrollment_duration_days``) + 55 baseline comorbidity.
Refinements applied: drop ``yrdob`` (collinear with age_at_index); derive
``geographic_region`` from ``zipcode_5`` and drop the raw 5-digit (PHI);
record-count columns are transparent quality-FILTER inputs, NOT model features;
``last_csu_dx_date`` / ``elig_end_date`` are observation-window ends used only
for censoring; risk bands kept alongside scores (owner choice).

Verified leakage facts (measured against the live mart, 2026-06-06):
- ``index_biologic_brand != 'no_treatment'`` reproduces the initiation target
  exactly -> deterministic target alias = post-index leakage.
- ``charlson_score`` / ``cci_*`` / ``elx_*`` correlate with PRE-index enrollment
  length but ~0 with POST-index follow-up (corr <= 0.07) -> disproof-cleared as
  pre-index (see ``.claude/plans/optum-initiation-adapter/IMPLEMENTATION-PLAN.md``).
"""

from src.data.feature_contract import FeatureContract, KnowableAt

_ENROLLMENT = KnowableAt(reference="enrollment")
_INDEX = KnowableAt(reference="index_date")
_POST = KnowableAt(reference="post_index")

# Statically-declared literal registry (AST-traceable; do NOT rebuild via comprehension).
OPTUM_MART_FEATURES: list[FeatureContract] = [
    # ===== PRE-INDEX demographics (knowable at enrollment) =====
    FeatureContract(name="age_at_index", knowable_at=_ENROLLMENT, source="mart_demographics"),
    FeatureContract(name="gdr_cd", knowable_at=_ENROLLMENT, source="mart_demographics"),
    FeatureContract(name="payer_category", knowable_at=_ENROLLMENT, source="mart_demographics"),
    FeatureContract(name="payer_product", knowable_at=_ENROLLMENT, source="mart_demographics"),
    FeatureContract(name="payer_bus", knowable_at=_ENROLLMENT, source="mart_demographics"),
    FeatureContract(
        name="health_exchange_flag", knowable_at=_ENROLLMENT, source="mart_demographics"
    ),
    FeatureContract(name="lis_dual_flag", knowable_at=_ENROLLMENT, source="mart_demographics"),
    # ===== PRE-INDEX derived (adapter-computed; raw source dropped) =====
    # geographic_region <- map_zipcode_to_region(zipcode_5); raw zipcode_5 dropped (PHI, 9043 unique).
    FeatureContract(name="geographic_region", knowable_at=_ENROLLMENT, source="mart_derived_geo"),
    # enrollment_duration_days <- index_date - elig_start_date (pre-index enrollment history).
    FeatureContract(
        name="enrollment_duration_days", knowable_at=_INDEX, source="mart_derived_enrollment"
    ),
    # ===== PRE-INDEX baseline comorbidity (knowable at index; disproof-cleared) =====
    FeatureContract(name="charlson_score", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="charlson_risk_band", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(
        name="elixhauser_van_walraven_score", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    FeatureContract(name="elixhauser_risk_band", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(
        name="comorbidity_diag_distinct_count", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    FeatureContract(
        name="comorbidity_diag_claim_count", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    FeatureContract(
        name="high_comorbidity_burden_flag", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    # Charlson component one-hots (17)
    FeatureContract(name="cci_mi", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_chf", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_pvd", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_cerebrovascular", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_dementia", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_chronic_pulmonary", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_rheumatic", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_peptic_ulcer", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_mild_liver", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(
        name="cci_diabetes_no_complication", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    FeatureContract(
        name="cci_diabetes_complication", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    FeatureContract(name="cci_paraplegia", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_renal", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_malignancy", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_severe_liver", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_metastatic_cancer", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="cci_hiv", knowable_at=_INDEX, source="mart_comorbidity"),
    # Elixhauser component one-hots (31)
    FeatureContract(name="elx_chf", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_cardiac_arrhythmia", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_valvular_disease", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(
        name="elx_pulmonary_circulation", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    FeatureContract(name="elx_pvd", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(
        name="elx_hypertension_uncomplicated", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    FeatureContract(
        name="elx_hypertension_complicated", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    FeatureContract(name="elx_paralysis", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_other_neurological", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_chronic_pulmonary", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(
        name="elx_diabetes_uncomplicated", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    FeatureContract(name="elx_diabetes_complicated", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_hypothyroidism", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_renal_failure", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_liver_disease", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_peptic_ulcer", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_aids_hiv", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_lymphoma", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_metastatic_cancer", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(
        name="elx_solid_tumor_no_metastasis", knowable_at=_INDEX, source="mart_comorbidity"
    ),
    FeatureContract(name="elx_rheumatoid_collagen", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_coagulopathy", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_obesity", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_weight_loss", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_fluid_electrolyte", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_blood_loss_anemia", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_deficiency_anemia", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_alcohol_abuse", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_drug_abuse", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_psychoses", knowable_at=_INDEX, source="mart_comorbidity"),
    FeatureContract(name="elx_depression", knowable_at=_INDEX, source="mart_comorbidity"),
    # ===== POST-INDEX supervised target (forbidden-as-feature, preserved at build) =====
    FeatureContract(name="initiated_biologic_180d", knowable_at=_POST, source="mart_target"),
    # ===== POST-INDEX proven target-aliases (declared forbidden for defense-in-depth) =====
    FeatureContract(name="index_biologic_brand", knowable_at=_POST, source="mart_treatment"),
    FeatureContract(name="treatment_start_date", knowable_at=_POST, source="mart_treatment"),
    # ===== POST-INDEX disc/persistence supervised targets (treatment-anchored) =====
    # Derived (Option B) from the coverage/gap columns below; forbidden-as-feature
    # but preserved at build (extracted per-run via prediction_target).
    FeatureContract(name="discontinued_180d", knowable_at=_POST, source="mart_target"),
    FeatureContract(name="persistent_at_180d", knowable_at=_POST, source="mart_target"),
    # ===== POST-INDEX disc/persistence proven aliases (defense-in-depth) =====
    # The coverage/gap derivation inputs + precomputed outcome flags that
    # (near-)deterministically reproduce discontinued_180d / persistent_at_180d
    # (discontinued_90d_flag agrees 98.2% with the derived target). The adapter
    # never emits these (positive enumeration), so they are declared forbidden
    # only as the runtime cross-check / leakage backstop for the treatment frame.
    FeatureContract(name="last_coverage_end", knowable_at=_POST, source="mart_outcome"),
    FeatureContract(name="last_observed_date", knowable_at=_POST, source="mart_outcome"),
    FeatureContract(name="max_internal_gap_days", knowable_at=_POST, source="mart_outcome"),
    FeatureContract(name="terminal_gap_days", knowable_at=_POST, source="mart_outcome"),
    FeatureContract(name="covered_days", knowable_at=_POST, source="mart_outcome"),
    FeatureContract(name="pdc", knowable_at=_POST, source="mart_outcome"),
    FeatureContract(name="discontinued_flag", knowable_at=_POST, source="mart_outcome"),
    FeatureContract(name="discontinued_90d_flag", knowable_at=_POST, source="mart_outcome"),
    FeatureContract(name="persistence_60d_flag", knowable_at=_POST, source="mart_outcome"),
    FeatureContract(name="maintained_flag", knowable_at=_POST, source="mart_outcome"),
    FeatureContract(name="adherent_flag", knowable_at=_POST, source="mart_outcome"),
]


def optum_mart_contract_for(name: str) -> FeatureContract | None:
    """Return the FeatureContract for a named mart column, or None if absent."""
    for contract in OPTUM_MART_FEATURES:
        if contract.name == name:
            return contract
    return None


MART_SAFE_FEATURES: list[str] = [
    c.name for c in OPTUM_MART_FEATURES if c.knowable_at.is_pre_or_at_index()
]

MART_FORBIDDEN_AS_FEATURES: list[str] = [
    c.name for c in OPTUM_MART_FEATURES if not c.knowable_at.is_pre_or_at_index()
]

# Supervised labels: forbidden-as-feature (post-index) but MUST be preserved at
# cohort-build so the pipeline can extract the target via prediction_target.
# One per tier-0 cohort: initiation + treatment-anchored discontinuation/persistence.
MART_TARGETS: frozenset[str] = frozenset(
    {"initiated_biologic_180d", "discontinued_180d", "persistent_at_180d"}
)

MART_FORBIDDEN_NON_TARGET: list[str] = [
    f for f in MART_FORBIDDEN_AS_FEATURES if f not in MART_TARGETS
]
