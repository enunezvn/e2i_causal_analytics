"""Tests for the Optum patient_journeys feature manifest (Layer 1.3 audit)."""

from __future__ import annotations

import pytest


def test_manifest_constructs_without_violation():
    from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES

    assert len(OPTUM_FEATURES) > 0


def test_manifest_unique_names():
    from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES

    names = [c.name for c in OPTUM_FEATURES]
    duplicates = [n for n in names if names.count(n) > 1]
    assert duplicates == [], f"Duplicate names: {set(duplicates)}"


def test_chain_validates_no_violations():
    from src.data.feature_contract import validate_contract_chain
    from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES

    contracts = {c.name: c for c in OPTUM_FEATURES}
    violations = validate_contract_chain(contracts)
    assert violations == [], "Chain violations:\n  " + "\n  ".join(v.reason for v in violations)


def test_safe_view_excludes_forbidden_columns():
    from src.data.manifests.optum_feature_manifest import (
        OPTUM_FEATURES,
        OPTUM_FORBIDDEN_AS_FEATURES,
        OPTUM_SAFE_FEATURES,
    )

    safe = set(OPTUM_SAFE_FEATURES)
    forbidden = set(OPTUM_FORBIDDEN_AS_FEATURES)
    all_names = {c.name for c in OPTUM_FEATURES}

    assert safe & forbidden == set()
    assert safe | forbidden == all_names


def test_documented_optum_targets_are_forbidden():
    """All Optum targets must be marked post_index/forbidden."""
    from src.data.manifests.optum_feature_manifest import (
        OPTUM_FORBIDDEN_AS_FEATURES,
        optum_contract_for,
    )

    for tgt in (
        "treatment_initiated",
        "initiated_biologic_180d",
        "discontinued_180d",
        "persistent_at_180d",
        "discontinuation_flag",
        "brand",
    ):
        c = optum_contract_for(tgt)
        assert c is not None, f"Manifest missing target: {tgt}"
        assert c.knowable_at.reference == "post_index"
        assert tgt in OPTUM_FORBIDDEN_AS_FEATURES


def test_drug_class_features_are_windowed():
    """All <cls>_fill_count features for non-target drug classes must declare
    window_days."""
    from src.data.manifests.optum_feature_manifest import (
        DRUG_CLASS_NAMES,
        optum_contract_for,
    )

    for cls in DRUG_CLASS_NAMES:
        c = optum_contract_for(f"{cls}_fill_count")
        assert c is not None, f"Missing contract for {cls}_fill_count"
        assert c.window_days is not None
        assert c.window_days > 0


def test_lab_features_are_windowed():
    from src.data.manifests.optum_feature_manifest import (
        LAB_NAMES,
        optum_contract_for,
    )

    for lab in LAB_NAMES:
        c = optum_contract_for(f"{lab}_tested")
        assert c is not None
        assert c.window_days is not None


_NON_FEATURE_OPTUM = {
    # IDs
    "patient_id",
    "patient_journey_id",
    "patient_hash",
    # Pipeline / audit metadata
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
    # Placeholders / always None
    "risk_score",
    "comorbidities",
    "secondary_diagnosis_codes",
    "primary_diagnosis_desc",
    "state",
}


@pytest.mark.parametrize("cohort", ["initiation", "discontinuation", "persistence"])
def test_manifest_covers_all_optum_feature_columns(cohort: str):
    """Every column in the Optum parquet for each cohort that COULD be a model
    input must have a contract in the manifest. Cohort-specific extras are
    targets that already live in OPTUM_FORBIDDEN_AS_FEATURES."""
    from pathlib import Path

    import pandas as pd

    from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES

    repo_root = Path(__file__).resolve().parents[3]
    pq_path = repo_root / "data" / "rwd" / "optum" / cohort / "e2i_ml_v3_patient_journeys.parquet"
    if not pq_path.exists():
        pytest.skip(f"Optum {cohort} parquet not present at {pq_path}")

    df = pd.read_parquet(pq_path)
    actual_columns = set(df.columns)
    manifest_columns = {c.name for c in OPTUM_FEATURES}
    feature_candidates = actual_columns - _NON_FEATURE_OPTUM

    uncovered = feature_candidates - manifest_columns
    assert uncovered == set(), (
        f"[{cohort}] Manifest missing contracts for emitted feature columns: "
        f"{sorted(uncovered)}. Either add a FeatureContract or extend "
        f"_NON_FEATURE_OPTUM if the column is metadata."
    )


def test_every_disease_specific_dx_feature_has_kg_entity_codes():
    """The 3 dx_l50_*_count features must declare their ICD-10 codes."""
    from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES

    by_name = {c.name: c for c in OPTUM_FEATURES}
    for fname in ("dx_l50_1_count", "dx_l50_8_count", "dx_l50_9_count"):
        assert by_name[fname].kg_entity_codes, (
            f"{fname} must declare kg_entity_codes for KG querying"
        )


def test_every_comorbidity_has_feature_has_kg_entity_codes():
    """All 8 has_<comorbidity> features must declare entity codes."""
    from src.data.manifests.optum_feature_manifest import (
        COMORBIDITY_NAMES,
        OPTUM_FEATURES,
    )

    by_name = {c.name: c for c in OPTUM_FEATURES}
    for name in COMORBIDITY_NAMES:
        feat_name = f"has_{name}"
        assert by_name[feat_name].kg_entity_codes, (
            f"{feat_name} must declare kg_entity_codes"
        )


def test_every_drug_class_ever_filled_feature_has_kg_entity_codes():
    """All 7 <drug_class>_ever_filled features must declare entity codes."""
    from src.data.manifests.optum_feature_manifest import (
        DRUG_CLASS_NAMES,
        OPTUM_FEATURES,
    )

    by_name = {c.name: c for c in OPTUM_FEATURES}
    for cls in DRUG_CLASS_NAMES:
        feat_name = f"{cls}_ever_filled"
        assert by_name[feat_name].kg_entity_codes, (
            f"{feat_name} must declare kg_entity_codes"
        )


def test_every_lab_tested_feature_has_kg_entity_codes():
    """All 8 <lab>_tested features must declare entity codes."""
    from src.data.manifests.optum_feature_manifest import (
        LAB_NAMES,
        OPTUM_FEATURES,
    )

    by_name = {c.name: c for c in OPTUM_FEATURES}
    for lab in LAB_NAMES:
        feat_name = f"{lab}_tested"
        assert by_name[feat_name].kg_entity_codes, (
            f"{feat_name} must declare kg_entity_codes"
        )


def test_primary_diagnosis_code_has_kg_entity_codes():
    """primary_diagnosis_code anchors the cohort and must declare codes."""
    from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES

    by_name = {c.name: c for c in OPTUM_FEATURES}
    assert by_name["primary_diagnosis_code"].kg_entity_codes, (
        "primary_diagnosis_code must declare kg_entity_codes"
    )
