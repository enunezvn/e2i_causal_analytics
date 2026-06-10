"""Contracts for the synthetic-CSU tier0 manifest (declared-safe by construction).

The synthetic generator constructs every declared feature BEFORE the outcome
draw (patient_generator.py confounders/covariates; hcp_adoption_artifact.py
exogenous centrality), so pre-index declarations are correct by construction —
without them the LLM/statistical leakage layers false-positively drop the
designed causal drivers (observed 2026-06-10: disease_severity dropped on
initiation; influence_network_size + peer_influence_score dropped on
hcp_adoption, collapsing AUC 0.78 -> 0.51).
"""

from src.data.manifests import MANIFEST_SOURCES, lookup_feature_contract
from src.data.manifests.synthetic_csu_feature_manifest import (
    SYNTHETIC_CSU_FEATURES,
    SYNTHETIC_CSU_FORBIDDEN_AS_FEATURES,
)

DECLARED_SAFE = [
    # patient grain — designed confounders/covariates, drawn pre-outcome
    "disease_severity",
    "academic_hcp",
    "age_at_diagnosis",
    "engagement_score",
    "segment_assignment",
    "treatment_arm",
    "insurance_type",
    "geographic_region",
    "primary_diagnosis_code",
    "urticaria_severity_uas7",
    "prior_antihistamine_therapy",
    # HCP grain — exogenous topology/profile attributes
    "influence_network_size",
    "peer_influence_score",
    "total_patient_volume",
    "years_experience",
    "specialty",
    "practice_type",
]

ANSWER_KEY = [
    "propensity_score",
    "treatment_effect_estimate",
    "days_to_treatment",
]


def test_source_registered():
    assert "synthetic_csu" in MANIFEST_SOURCES


def test_designed_features_are_declared_pre_index():
    for name in DECLARED_SAFE:
        contract = lookup_feature_contract(name, "synthetic_csu")
        assert contract is not None, f"{name}: no contract"
        assert contract.knowable_at.is_pre_or_at_index(), f"{name}: not pre-index"


def test_answer_key_columns_are_forbidden_not_safe():
    for name in ANSWER_KEY:
        contract = lookup_feature_contract(name, "synthetic_csu")
        assert contract is None or not contract.knowable_at.is_pre_or_at_index(), (
            f"{name}: DGP answer-key column must never be declared safe"
        )
        assert name in SYNTHETIC_CSU_FORBIDDEN_AS_FEATURES, f"{name}: missing from forbidden list"


def test_outcome_columns_not_declared_safe():
    for name in (
        "treatment_initiated",
        "discontinued_180d",
        "persistent_180d",
        "adopted_target_brand",
    ):
        contract = SYNTHETIC_CSU_FEATURES.get(name)
        assert contract is None or not contract.knowable_at.is_pre_or_at_index()


def test_unknown_feature_returns_none():
    assert lookup_feature_contract("nonexistent_col", "synthetic_csu") is None
