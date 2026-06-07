"""Leakage-contract tests for the optum_hcp (commercial-targeting) manifest.

Pins the admissibility invariants that make the HCP adoption-propensity cohort
honest: every adoption-DERIVED column is post-index forbidden, the admissible
practice-profile features are pre-index, and the safe/forbidden partitions do
not overlap. The converter (`convert_optum_hcp_adoption.py`) imports
`OPTUM_HCP_SAFE_FEATURES` as its emit allow-list, so this manifest is the single
source of truth for "what is a pre-adoption predictor vs a label leak."
"""

from __future__ import annotations

import pytest

from src.data.manifests.optum_hcp_feature_manifest import (
    OPTUM_HCP_FEATURES,
    OPTUM_HCP_FORBIDDEN_AS_FEATURES,
    OPTUM_HCP_SAFE_FEATURES,
    OPTUM_HCP_TARGETS,
    optum_hcp_contract_for,
)

# Columns the adoption target is computed from — must NEVER be model features.
_ADOPTION_DERIVED = (
    "adoption_status",
    "adoption_category",
    "adoption_category_method",
    "adopter_rank",
    "adopter_count",
    "adoption_cumulative_share",
    "days_to_first",
    "first_adoption_dt",
    "target_event_count",
    "target_patient_count",
    "distinct_target_code_count",
)


def test_target_is_post_index_and_registered() -> None:
    assert "adopted_target_brand" in OPTUM_HCP_TARGETS
    c = optum_hcp_contract_for("adopted_target_brand")
    assert c is not None and c.knowable_at.reference == "post_index"
    assert "adopted_target_brand" not in OPTUM_HCP_SAFE_FEATURES


@pytest.mark.parametrize("col", _ADOPTION_DERIVED)
def test_adoption_derived_columns_are_forbidden(col: str) -> None:
    c = optum_hcp_contract_for(col)
    assert c is not None, f"{col} must be declared (defense-in-depth)"
    assert not c.knowable_at.is_pre_or_at_index(), f"{col} must be post-index"
    assert col in OPTUM_HCP_FORBIDDEN_AS_FEATURES
    assert col not in OPTUM_HCP_SAFE_FEATURES


def test_safe_features_are_pre_index_and_disjoint_from_forbidden() -> None:
    assert OPTUM_HCP_SAFE_FEATURES, "safe feature list must be non-empty"
    assert not (set(OPTUM_HCP_SAFE_FEATURES) & set(OPTUM_HCP_FORBIDDEN_AS_FEATURES))
    for name in OPTUM_HCP_SAFE_FEATURES:
        c = optum_hcp_contract_for(name)
        assert c is not None and c.knowable_at.is_pre_or_at_index()


def test_admissible_features_are_network_volume_or_provider() -> None:
    # The deployable signal must come from claims practice profile, not adoption.
    sources = {optum_hcp_contract_for(n).source for n in OPTUM_HCP_SAFE_FEATURES}
    assert sources <= {"hcp_network", "hcp_volume", "hcp_provider"}
    # specialty_group (the dominant legitimate signal) must be admissible.
    assert "specialty_group" in OPTUM_HCP_SAFE_FEATURES


def test_contract_for_unknown_returns_none() -> None:
    assert optum_hcp_contract_for("definitely_not_a_column") is None


def test_no_duplicate_contracts() -> None:
    names = [c.name for c in OPTUM_HCP_FEATURES]
    assert len(names) == len(set(names))
