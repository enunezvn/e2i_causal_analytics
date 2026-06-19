"""Unit coverage for the HCP-grain causal dataset (hcp_adoption).

hcp_adoption is a JOIN dataset: hcp_brand_adoption (treatment_arm, adopted, brand)
JOIN hcp_profiles (peer_influence_score, influence_network_size -> centrality_z)
on hcp_id. These tests are CI-safe (no DB, no agent run); the live JOIN is covered
by a faithful check.
"""

import pytest


@pytest.mark.unit
def test_hcp_adoption_spec_registered():
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS, _CAUSAL_NUMERIC_COLUMNS

    assert "hcp_adoption" in _CAUSAL_DATASET_SPECS
    spec = _CAUSAL_DATASET_SPECS["hcp_adoption"]
    assert set(spec["treatment"]) == {"peer_influence_score", "treatment_arm"}
    assert spec["outcome"] == ["adopted"]
    assert spec["covariate"] == ["centrality_z"]
    # Every loadable column is numeric-coerced (the gate covers treatment+outcome+cov).
    numeric = _CAUSAL_NUMERIC_COLUMNS["hcp_adoption"]
    assert {"peer_influence_score", "treatment_arm", "adopted", "centrality_z"} <= numeric
