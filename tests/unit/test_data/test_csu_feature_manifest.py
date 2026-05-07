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

    # Forbidden as features (post_index by construction)
    for name in [
        "journey_duration_days",
        "journey_status",
        "journey_end_date",
        "journey_start_date",
        "journey_stage",
        "treatment_initiated",
        "discontinuation_flag",
    ]:
        c = csu_contract_for(name)
        assert c is not None, f"Manifest missing forbidden column: {name}"
        assert c.knowable_at.reference == "post_index", (
            f"{name} should be post_index; got {c.knowable_at}"
        )
        assert name in CSU_FORBIDDEN_AS_FEATURES

    # Documented past incidents that ARE legitimate when windowed
    for name in [
        "engagement_score",
        "disease_severity",
        "days_on_therapy",
        "medication_claim_count",
        "lab_claim_count",
        "hcp_visits",
        "prior_treatments",
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

    # Reverse direction: every manifest entry should be a real column.
    extra = manifest_columns - actual_columns
    assert extra == set(), (
        f"Manifest declares contracts for columns the converter doesn't emit: {sorted(extra)}. "
        f"This means the manifest has drifted from the converter; either remove the entry "
        f"or fix the converter."
    )
