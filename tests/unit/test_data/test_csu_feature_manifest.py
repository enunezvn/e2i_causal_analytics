"""Tests for the CSU patient_journeys feature manifest (Layer 1.3 audit).

Verifies that:
1. Every contract in CSU_FEATURES constructs cleanly (so no manifest entry
   silently violates the FeatureContract invariants).
2. The chain validates with no ContractChainViolations.
3. SAFE / FORBIDDEN views agree with the temporal-validity claims.
4. Every documented past leakage incident from the leakage compile set has
   a contract entry, and that entry correctly reflects whether the column
   is forbidden (post_index) vs. windowed (index_date + window_days).
"""

from __future__ import annotations

import pytest


def test_manifest_constructs_without_violation():
    """Importing the manifest must not raise. Each contract has been
    validated at construction time."""
    from src.data.manifests.csu_feature_manifest import CSU_FEATURES

    assert len(CSU_FEATURES) > 0


def test_manifest_unique_names():
    """No duplicate feature names — the manifest is a 1:1 registry."""
    from src.data.manifests.csu_feature_manifest import CSU_FEATURES

    names = [c.name for c in CSU_FEATURES]
    duplicates = [n for n in names if names.count(n) > 1]
    assert duplicates == [], f"Duplicate names in manifest: {set(duplicates)}"


def test_chain_validates_no_violations():
    """Chain consistency: every input that's also a manifest entry must
    have knowable_at <= the consuming feature's knowable_at."""
    from src.data.feature_contract import validate_contract_chain
    from src.data.manifests.csu_feature_manifest import CSU_FEATURES

    contracts = {c.name: c for c in CSU_FEATURES}
    violations = validate_contract_chain(contracts)
    assert violations == [], "Chain violations:\n  " + "\n  ".join(v.reason for v in violations)


def test_safe_view_excludes_forbidden_columns():
    """The SAFE/FORBIDDEN views must be disjoint and exhaustive."""
    from src.data.manifests.csu_feature_manifest import (
        CSU_FEATURES,
        CSU_FORBIDDEN_AS_FEATURES,
        CSU_SAFE_FEATURES,
    )

    safe = set(CSU_SAFE_FEATURES)
    forbidden = set(CSU_FORBIDDEN_AS_FEATURES)
    all_names = {c.name for c in CSU_FEATURES}

    assert safe & forbidden == set(), f"SAFE and FORBIDDEN overlap: {safe & forbidden}"
    assert safe | forbidden == all_names


def test_targets_are_subset_of_forbidden():
    """``CSU_TARGETS`` must be a subset of ``CSU_FORBIDDEN_AS_FEATURES``.

    Targets are post-index by definition (you cannot use the label as a
    feature) and therefore appear in the manifest's forbidden set. The
    cohort-builder gate uses ``CSU_FORBIDDEN_NON_TARGET`` which excludes
    ``CSU_TARGETS`` so the supervised signal survives the boundary
    filter.
    """
    from src.data.manifests.csu_feature_manifest import (
        CSU_FORBIDDEN_AS_FEATURES,
        CSU_TARGETS,
    )

    forbidden = set(CSU_FORBIDDEN_AS_FEATURES)
    extra_targets = CSU_TARGETS - forbidden
    assert not extra_targets, (
        f"CSU_TARGETS contains entries not in CSU_FORBIDDEN_AS_FEATURES: "
        f"{sorted(extra_targets)}. Either add them to the manifest as "
        f"post_index FeatureContracts, or remove them from CSU_TARGETS."
    )


def test_forbidden_non_target_is_complement():
    """``CSU_FORBIDDEN_NON_TARGET`` must equal ``FORBIDDEN_AS_FEATURES - TARGETS``."""
    from src.data.manifests.csu_feature_manifest import (
        CSU_FORBIDDEN_AS_FEATURES,
        CSU_FORBIDDEN_NON_TARGET,
        CSU_TARGETS,
    )

    expected = set(CSU_FORBIDDEN_AS_FEATURES) - CSU_TARGETS
    assert set(CSU_FORBIDDEN_NON_TARGET) == expected, (
        f"CSU_FORBIDDEN_NON_TARGET drift:\n"
        f"  expected (FORBIDDEN - TARGETS): {sorted(expected)}\n"
        f"  actual: {sorted(CSU_FORBIDDEN_NON_TARGET)}"
    )


def test_known_leak_incidents_are_covered_correctly():
    """Every documented leakage incident from the compile set must appear
    in the manifest, and the entry must correctly reflect whether it is
    forbidden (post_index target/journey-metadata) or windowed (legitimate
    pre-index when correctly derived)."""
    from src.data.manifests.csu_feature_manifest import (
        CSU_FORBIDDEN_AS_FEATURES,
        CSU_SAFE_FEATURES,
        csu_contract_for,
    )

    # Forbidden as features (post_index by construction).
    #
    # Backlog #17 (2026-05-12) — the medication-derived aggregates
    # (medication_claim_count, days_on_therapy, hcp_visits,
    # prior_treatments, disease_severity, engagement_score) were
    # reclassified from index_date to post_index because the CSU
    # prediction target `treatment_initiated` ≡ "patient in
    # medication panel", which makes these features structurally
    # target-coupled regardless of date filtering. The three B3
    # engineered features that depend on them (engagement_per_visit,
    # treatment_diversity_intensity, severity_engagement_product)
    # are post_index by chain. See manifest docstring + iter-5
    # empirical audit (2026-05-09).
    for name in [
        "journey_duration_days",
        "journey_status",
        "journey_end_date",
        "journey_start_date",
        "journey_stage",
        "treatment_initiated",
        "discontinuation_flag",
        # Backlog #17 — medication-derived aggregates
        "medication_claim_count",
        "days_on_therapy",
        "hcp_visits",
        "prior_treatments",
        "disease_severity",
        "engagement_score",
        # Backlog #17 — B3 engineered derived from the above
        "engagement_per_visit",
        "treatment_diversity_intensity",
        "severity_engagement_product",
    ]:
        c = csu_contract_for(name)
        assert c is not None, f"Manifest missing forbidden column: {name}"
        assert c.knowable_at.reference == "post_index", (
            f"{name} should be post_index; got {c.knowable_at}"
        )
        assert name in CSU_FORBIDDEN_AS_FEATURES

    # Documented past incidents that ARE legitimate when windowed.
    # ``procedure_claim_count`` and ``lab_claim_count`` retain
    # index_date status: they derive from independent event panels
    # (procedure / lab) and patients can have procedures or labs
    # without being in the medication panel — they are NOT
    # structurally coupled to ``treatment_initiated``.
    for name in [
        "lab_claim_count",
        "procedure_claim_count",
    ]:
        c = csu_contract_for(name)
        assert c is not None, f"Manifest missing windowed feature: {name}"
        assert c.knowable_at.reference == "index_date", (
            f"{name} should be index_date; got {c.knowable_at}"
        )
        assert c.window_days is not None and c.window_days > 0, (
            f"{name} must declare window_days; got {c.window_days}"
        )
        assert name in CSU_SAFE_FEATURES


def test_unwindowed_event_aggregations_would_violate_at_construction():
    """Sanity check: the FeatureContract layer correctly rejects a hand-rolled
    unwindowed event aggregation. If this test fails, the contract layer
    has regressed and the manifest's safety guarantees no longer hold."""
    from src.data.feature_contract import (
        ContractViolation,
        FeatureContract,
        KnowableAt,
    )

    with pytest.raises(ContractViolation):
        FeatureContract(
            name="unwindowed_disease_severity",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=("medication_date",),
            aggregation="sum",
            window_days=None,
        )


def test_csu_contract_for_returns_none_on_unknown():
    """csu_contract_for is null-safe for unknown names."""
    from src.data.manifests.csu_feature_manifest import csu_contract_for

    assert csu_contract_for("not_a_csu_feature") is None


# Columns the converter emits that are NOT model-input candidates: IDs,
# pipeline metadata, provenance, and placeholder fields that are always
# null/empty in the current data. These don't need contracts because they
# wouldn't be passed to a model anyway.
_NON_FEATURE_COLUMNS = {
    # IDs
    "patient_id",
    "patient_journey_id",
    "patient_hash",
    # Audit/provenance metadata
    "created_at",
    "updated_at",
    "ingestion_timestamp",
    "source_timestamp",
    "data_lag_hours",
    "data_split",
    "split_config_id",
    "data_quality_score",
    "data_source",
    "data_sources_matched",
    "source_match_confidence",
    "source_stacking_flag",
    "source_combination_method",
    # Placeholders (always None / empty list in current data)
    "risk_score",
    "comorbidities",
    "secondary_diagnosis_codes",
    "primary_diagnosis_desc",  # human-readable, redundant with primary_diagnosis_code
    "state",  # always None in current CSU data
}


def test_manifest_covers_all_csu_feature_columns():
    """Every column emitted by convert_csu_rwd.py that COULD be a model input
    must have a contract entry. Non-feature columns (IDs, metadata, empty
    placeholders) are exempt via _NON_FEATURE_COLUMNS.

    If the converter adds a new feature column, this test fails until the
    manifest is updated. That's the ratchet that makes the manifest stay
    in sync with the converter.
    """
    import json
    from pathlib import Path

    from src.data.manifests.csu_feature_manifest import CSU_FEATURES

    repo_root = Path(__file__).resolve().parents[3]
    journeys_path = repo_root / "data" / "rwd" / "csu" / "e2i_ml_v3_patient_journeys.json"
    if not journeys_path.exists():
        pytest.skip(f"CSU journeys file not present at {journeys_path}")

    records = json.loads(journeys_path.read_text())
    if not records:
        pytest.skip("CSU journeys file empty")

    actual_columns = set(records[0].keys())
    manifest_columns = {c.name for c in CSU_FEATURES}
    feature_candidates = actual_columns - _NON_FEATURE_COLUMNS

    uncovered = feature_candidates - manifest_columns
    assert uncovered == set(), (
        f"Manifest missing contracts for emitted feature columns: {sorted(uncovered)}. "
        f"Either add a FeatureContract or extend _NON_FEATURE_COLUMNS in this test "
        f"if the column is metadata."
    )

    # Reverse direction: every manifest entry should be a real column,
    # EXCEPT for columns the converter intentionally strips at write
    # time via ``_drop_forbidden_columns(CSU_FORBIDDEN_NON_TARGET)``.
    # Those columns are declared in the manifest so Layer 1 can catch
    # them when they DO appear (e.g., older artifacts, ad-hoc test
    # injection), but the converter's defense-in-depth gate omits them
    # from the on-disk JSON.
    #
    # Backlog #17 (2026-05-12): six medication-derived aggregates +
    # three B3 engineered features were moved to forbidden. If the
    # CSU JSON file in this checkout was written by an older converter
    # (before backlog #17), those columns may still be present —
    # tolerated, not required. If the file was written by the new
    # converter, those columns will be absent — also tolerated.
    #
    # Codex pass-1 M2 (2026-05-12): the exemption is narrowed to the
    # set ``CSU_INTENDED_DROP`` declared in
    # ``tests/unit/test_scripts/test_cohort_builder_forbidden_gate.py``
    # — i.e., the catalog of forbidden columns that the converter is
    # KNOWN to strip. Any future erroneous post_index manifest entry
    # would land in ``CSU_FORBIDDEN_NON_TARGET`` but NOT in
    # ``CSU_INTENDED_DROP``, so this assertion would fail loudly —
    # forcing the author to either (a) verify the converter actually
    # strips the column and add it to ``CSU_INTENDED_DROP``, or
    # (b) fix the bogus post_index declaration.
    from tests.unit.test_scripts.test_cohort_builder_forbidden_gate import (
        CSU_INTENDED_DROP,
    )

    extra = manifest_columns - actual_columns - set(CSU_INTENDED_DROP)
    assert extra == set(), (
        f"Manifest declares contracts for columns the converter doesn't emit: {sorted(extra)}. "
        f"This means the manifest has drifted from the converter. Resolutions:\n"
        f"  (a) Add the column to CSU_INTENDED_DROP in "
        f"tests/unit/test_scripts/test_cohort_builder_forbidden_gate.py "
        f"AND verify scripts/convert_csu_rwd.py's _drop_forbidden_columns "
        f"strips it (the companion test "
        f"test_csu_intended_drop_columns_all_in_non_target will then enforce "
        f"the manifest membership), OR\n"
        f"  (b) Remove the entry from the manifest if no longer applicable, OR\n"
        f"  (c) Add it to the converter so it appears on disk."
    )


def test_csu_primary_diagnosis_code_has_kg_entity_codes():
    """primary_diagnosis_code anchors the CSU cohort and must declare codes."""
    from src.data.manifests.csu_feature_manifest import CSU_FEATURES

    by_name = {c.name: c for c in CSU_FEATURES}
    assert by_name["primary_diagnosis_code"].kg_entity_codes, (
        "primary_diagnosis_code must declare kg_entity_codes"
    )


# Backlog #17 (2026-05-12) — pin the medication-derived aggregates +
# their B3-engineered dependents as ``knowable_at=post_index``. The
# CSU prediction target ``treatment_initiated`` is defined as "patient
# appears anywhere in the medication panel" (see
# ``scripts/convert_csu_rwd.py``); untreated patients are absent from
# ``_med_by_pat`` entirely, which means every medication-derived
# aggregate (count / sum / nunique) collapses to zero for them
# regardless of date windowing. The iter-5 empirical audit
# (2026-05-09) caught all six at Layer 3 z=14.13–69.18 even with
# ``--lookback-days=180`` applied — confirming the target-coupling is
# structural, not date-dependent. Reclassifying them to post_index
# moves the catch from Layer 3 (statistical, slower) to Layer 1
# (declarative, cheaper, deterministic).
_BACKLOG_17_POST_INDEX_FEATURES = [
    # Six medication-derived aggregates flagged by iter-5 audit.
    "medication_claim_count",
    "days_on_therapy",
    "hcp_visits",
    "prior_treatments",
    "disease_severity",
    "engagement_score",
    # Three B3-engineered features whose inputs are now post_index.
    # ``age_x_insurance_interaction`` (the fourth B3 feature) is NOT
    # listed here because its inputs (age_continuous + insurance_type)
    # are both enrollment-knowable.
    "engagement_per_visit",
    "treatment_diversity_intensity",
    "severity_engagement_product",
]


def test_backlog_17_med_aggregates_are_post_index():
    """The six medication-derived aggregates + three dependent B3
    engineered features MUST be declared post_index per backlog #17."""
    from src.data.manifests.csu_feature_manifest import (
        CSU_FORBIDDEN_AS_FEATURES,
        csu_contract_for,
    )

    for name in _BACKLOG_17_POST_INDEX_FEATURES:
        contract = csu_contract_for(name)
        assert contract is not None, f"Manifest missing contract for {name!r}"
        assert contract.knowable_at.reference == "post_index", (
            f"{name!r} must be declared knowable_at=post_index per backlog #17 "
            f"(medication-target-coupling); got knowable_at={contract.knowable_at}. "
            f"See manifest docstring + tests/integration/test_csu_val_auc_measurement.py"
            f"::test_csu_post_index_med_aggregates_dropped_via_layer_1."
        )
        assert name in CSU_FORBIDDEN_AS_FEATURES, (
            f"{name!r} must appear in CSU_FORBIDDEN_AS_FEATURES (the "
            f"computed view from knowable_at=post_index); got missing. "
            f"This likely means the convenience view fell out of sync."
        )


def test_backlog_17_procedure_and_lab_remain_pre_index():
    """``procedure_claim_count`` and ``lab_claim_count`` retain
    knowable_at=index_date status: they derive from independent event
    panels and patients can have procedures or labs without being in
    the medication panel — they are NOT structurally coupled to the
    target."""
    from src.data.manifests.csu_feature_manifest import (
        CSU_SAFE_FEATURES,
        csu_contract_for,
    )

    for name in ("procedure_claim_count", "lab_claim_count"):
        contract = csu_contract_for(name)
        assert contract is not None
        assert contract.knowable_at.is_pre_or_at_index(), (
            f"{name!r} must remain pre-or-at-index per backlog #17 audit "
            f"(independent event panels are NOT target-coupled); got "
            f"knowable_at={contract.knowable_at}"
        )
        assert name in CSU_SAFE_FEATURES
