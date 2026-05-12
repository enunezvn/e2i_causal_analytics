"""CSU `patient_journeys` feature manifest — Layer 1.3 audit.

Catalogs every column emitted by ``scripts/convert_csu_rwd.py``'s
``_build_patient_journeys`` into a `FeatureContract`. The contract makes
each column's temporal validity claims auditable in code:

- **Pre-index** features (``knowable_at = index_date | enrollment``) are
  the only ones safe to use as model inputs.
- **Post-index** features (``knowable_at = post_index``) include
  journey-level metadata (``journey_status``, ``journey_duration_days``,
  ``journey_end_date``) and the prediction targets themselves. These
  must be excluded from feature surface; the contract documents that.

When the converter is changed (a new feature added, a derivation
reshuffled), this manifest must be updated in the same commit. The unit
tests in ``tests/unit/test_data/test_csu_feature_manifest.py`` enforce
chain consistency and surface coverage.

Reference: `.claude/state/leakage_compile_set_20260507.md` lists the 18
documented incidents that motivated this audit.

Window convention: 180 days is the canonical CSU lookback (the
``--lookback-days`` default in ``convert_csu_rwd.py``). When the
converter is run with a different lookback, the audit logic compares
``window_days`` ranges, not exact values.
"""

from __future__ import annotations

from src.data.feature_contract import FeatureContract, KnowableAt

CSU_LOOKBACK_DAYS = 180


# =============================================================================
# Demographics — knowable at enrollment (i.e., before index_date)
# =============================================================================

_DEMO_ENROLLMENT = [
    FeatureContract(
        name="age_continuous",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("age",),
    ),
    FeatureContract(
        name="age_group",
        knowable_at=KnowableAt(reference="enrollment"),
        source="derived",
        derivation_inputs=("age_continuous",),
    ),
    FeatureContract(
        name="gender",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("gdr_cd",),
    ),
    FeatureContract(
        name="zip_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("zipcode_5",),
    ),
    FeatureContract(
        name="geographic_region",
        knowable_at=KnowableAt(reference="enrollment"),
        source="derived",
        derivation_inputs=("zip_code",),
    ),
    FeatureContract(
        name="insurance_type",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("bus",),
    ),
    FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("ICD10CM", "L50.9"), ("UMLS", "C0042109")),
    ),
]


# =============================================================================
# Eligibility — knowable at index_date (windowed by lookback)
# =============================================================================

_ELIGIBILITY = [
    FeatureContract(
        name="eligibility_duration_days",
        knowable_at=KnowableAt(reference="index_date"),
        source="enrollment",
        derivation_inputs=("eligeff", "eligend"),
        # Note: `enrollment` is non-event; the converter clips to
        # [index_date - lookback, index_date) when masking is on, so the
        # FEATURE knowable_at is index_date even though the underlying
        # raw window is enrollment-typed.
    ),
]


# =============================================================================
# Windowed event aggregations — knowable at index_date (window_days REQUIRED)
# =============================================================================
#
# Backlog #17 (2026-05-12) — the medication-derived aggregates
# (``medication_claim_count``, ``days_on_therapy``, ``hcp_visits``,
# ``prior_treatments``, ``disease_severity``, ``engagement_score``) were
# moved to ``_POST_INDEX_FORBIDDEN`` below. Even when the converter
# applies the lookback window (``--lookback-days=180``), these features
# remain structurally target-equivalent because the CSU prediction
# target ``treatment_initiated`` is defined as "patient appears anywhere
# in the medication panel". Untreated patients are therefore absent
# from ``_med_by_pat`` and their windowed aggregates collapse to zero
# regardless of date filtering. Iter-5 (2026-05-09) empirical audit
# caught all six at z=14.13–69.18 on real CSU n=9607. Path 2 per
# backlog #17 — declare ``KnowableAt(reference="post_index")`` so
# Layer 1 catches deterministically. See ``docs/lineage/csu_field_audit.md``
# §3 rows 36–40, 43 for the canonical pre-existing audit; this manifest
# move reconciles the manifest declaration with that audit.
#
# ``procedure_claim_count`` and ``lab_claim_count`` retain
# ``KnowableAt(reference="index_date")``: they derive from independent
# event panels (``_proc_by_pat`` / ``_lab_by_pat``), and patients can
# have procedures or labs without being in the medication panel, so
# they are NOT structurally coupled to ``treatment_initiated``.

_WINDOWED_AGG = [
    FeatureContract(
        name="procedure_claim_count",
        knowable_at=KnowableAt(reference="index_date"),
        source="procedure_events",
        derivation_inputs=("proc_date",),
        aggregation="count",
        window_days=CSU_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="lab_claim_count",
        knowable_at=KnowableAt(reference="index_date"),
        source="lab_events",
        derivation_inputs=("fst_dt",),
        aggregation="count",
        window_days=CSU_LOOKBACK_DAYS,
    ),
]


# =============================================================================
# v5 Gate B3 — engineered features (derived from pre-anchor inputs)
# =============================================================================
#
# Each engineered feature is computed by
# ``src.agents.ml_foundation.data_preparer.nodes.feature_engineering``
# (see ``CSU_ENGINEERED_FEATURES`` there) and inherits pre-anchor status
# from its declared ``derivation_inputs`` — every input listed below is
# itself declared pre-anchor in this manifest.
#
# Pre-spec: ``docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md``.
# Audit gate: Layer 1 traces the derivation chain (this declaration);
# Layer 3 (production adversarial probe) runs on the materialized column
# during ``adaptive_validity_check``. Both must pass before the feature
# is permitted to influence val_AUC.

# Post-audit: claim_intensity_ratio dropped (z=40.83 amplified to 4.2x of
# max input z on CSU; the amplification heuristic flags ratio-style
# transforms that manufacture signal not present in either component as
# leakage-suspect). 4 candidates remain.
#
# Backlog #17 (2026-05-12): three of the four B3 engineered features
# derive from post-index inputs (``engagement_score`` / ``hcp_visits`` /
# ``prior_treatments`` / ``days_on_therapy`` / ``disease_severity``)
# and were moved to ``_POST_INDEX_FORBIDDEN`` below to preserve chain
# validity (a derived feature cannot be knowable earlier than any of
# its inputs). ``age_x_insurance_interaction`` remains here because
# its inputs (``age_continuous`` + ``insurance_type``) are both
# enrollment-knowable.
_ENGINEERED_B3 = [
    FeatureContract(
        name="age_x_insurance_interaction",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("age_continuous", "insurance_type"),
    ),
]


# =============================================================================
# Post-index — FORBIDDEN as features (journey metadata + targets)
# =============================================================================
# These columns appear in patient_journeys.json but MUST NOT be used as
# model inputs. The contract documents WHY; the test asserts the converter
# excludes them from the feature surface.

_POST_INDEX_FORBIDDEN = [
    # ------------------------------------------------------------------
    # Backlog #17 (2026-05-12) — medication-derived aggregates moved
    # here from ``_WINDOWED_AGG``. Even when the converter applies a
    # lookback window, untreated patients are absent from
    # ``CSUDataConverter._med_by_pat`` so all medication-derived
    # aggregates collapse to zero for them, making the features
    # structurally target-coupled (``treatment_initiated`` ≡
    # "patient in medication panel"). Iter-5 audit caught all six at
    # z=14.13–69.18 on real CSU n=9607 (`docs/results/...`).
    # ``contract_window_days`` is preserved at 180 to document the
    # converter's masking semantics even though the feature is now
    # declared post-index for Layer 1 catch purposes.
    # ------------------------------------------------------------------
    FeatureContract(
        name="medication_claim_count",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("medication_date",),
        aggregation="count",
        window_days=CSU_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="days_on_therapy",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("days_sup", "medication_date"),
        aggregation="sum",
        window_days=CSU_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="hcp_visits",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("npi", "medication_date"),
        aggregation="nunique",
        window_days=CSU_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="prior_treatments",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("brand_normalised", "medication_date"),
        aggregation="nunique",
        window_days=CSU_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="disease_severity",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("medication_date", "proc_code", "abnl_cd"),
        aggregation="sum",
        window_days=CSU_LOOKBACK_DAYS,
    ),
    FeatureContract(
        name="engagement_score",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=("npi", "medication_date", "fst_dt"),
        aggregation="sum",
        window_days=CSU_LOOKBACK_DAYS,
    ),
    # ------------------------------------------------------------------
    # Backlog #17 (2026-05-12) — B3 engineered features whose inputs
    # are now post-index. ``age_x_insurance_interaction`` remains in
    # ``_ENGINEERED_B3`` because its inputs are enrollment-knowable.
    # ------------------------------------------------------------------
    FeatureContract(
        name="engagement_per_visit",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("engagement_score", "hcp_visits"),
    ),
    FeatureContract(
        name="treatment_diversity_intensity",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("prior_treatments", "days_on_therapy"),
    ),
    FeatureContract(
        name="severity_engagement_product",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("disease_severity", "engagement_score"),
    ),
    # ------------------------------------------------------------------
    # Pre-existing post-index columns — journey metadata, brand, targets.
    # ------------------------------------------------------------------
    FeatureContract(
        name="journey_start_date",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("indexdt",),
    ),
    FeatureContract(
        name="journey_end_date",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("eligend", "treatment_initiated"),
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
        derivation_inputs=("treatment_initiated", "days_on_therapy"),
    ),
    FeatureContract(
        name="journey_status",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("treatment_initiated", "discontinuation_flag"),
    ),
    # Targets — knowable only after the prediction window resolves.
    FeatureContract(
        name="treatment_initiated",
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
    # `brand` is target-coupled in the converter:
    #     brand = "competitor" if treatment_initiated else None
    # Using it as a feature would re-encode the target perfectly. Forbidden.
    FeatureContract(
        name="brand",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
        derivation_inputs=("treatment_initiated",),
    ),
]


# Public registry: every CSU patient_journeys column is here.
CSU_FEATURES: list[FeatureContract] = (
    _DEMO_ENROLLMENT + _ELIGIBILITY + _WINDOWED_AGG + _ENGINEERED_B3 + _POST_INDEX_FORBIDDEN
)


def csu_contract_for(name: str) -> FeatureContract | None:
    """Return the FeatureContract for a named CSU feature, or None if absent.

    Useful in pipelines that want to retrieve declared metadata at runtime
    (e.g., the adaptive_validity_check node could surface the contract's
    declared aggregation when emitting a verdict).
    """
    for c in CSU_FEATURES:
        if c.name == name:
            return c
    return None


# Convenience views — useful for both tests and pipeline configuration.
CSU_SAFE_FEATURES: list[str] = [c.name for c in CSU_FEATURES if c.knowable_at.is_pre_or_at_index()]

CSU_FORBIDDEN_AS_FEATURES: list[str] = [
    c.name for c in CSU_FEATURES if not c.knowable_at.is_pre_or_at_index()
]

# Targets — features whose ``knowable_at`` is post-index AND whose role
# is "predict me". These are FORBIDDEN as features (since they are the
# label) but MUST NOT be dropped at cohort-build time, because the
# downstream pipeline reads ``scope_spec.prediction_target`` to extract
# the supervised signal. Cohort-builder gates filter forbidden columns
# but explicitly preserve everything in this set.
#
# Maintenance: add to this set when introducing a new prediction target;
# the test ``test_targets_subset_of_forbidden`` enforces that every entry
# is also in ``CSU_FORBIDDEN_AS_FEATURES``.
CSU_TARGETS: frozenset[str] = frozenset(
    {
        "treatment_initiated",
        "discontinuation_flag",
    }
)

# Forbidden columns that are NOT targets — the safe-to-drop set for the
# cohort-builder gate. Computed once at import time so the gate doesn't
# re-derive it per record.
CSU_FORBIDDEN_NON_TARGET: list[str] = [f for f in CSU_FORBIDDEN_AS_FEATURES if f not in CSU_TARGETS]
