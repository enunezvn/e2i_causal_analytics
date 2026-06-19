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


import math
from unittest.mock import AsyncMock, patch

import pandas as pd
from fastapi import HTTPException

from src.api.routes import causal as causal_routes


def _fake_join_rows():
    # adoption rows (brand-filtered read) and profile rows (un-filtered read).
    adoption = [
        {"hcp_id": "h1", "treatment_arm": 1, "adopted": 1},
        {"hcp_id": "h2", "treatment_arm": 0, "adopted": 0},
        {"hcp_id": "h3", "treatment_arm": 1, "adopted": 1},
    ]
    profiles = [
        {"hcp_id": "h1", "peer_influence_score": 3.0, "influence_network_size": 25},
        {"hcp_id": "h2", "peer_influence_score": 1.0, "influence_network_size": 2},
        {"hcp_id": "h3", "peer_influence_score": 2.5, "influence_network_size": 14},
    ]
    return adoption, profiles


@pytest.mark.asyncio
async def test_hcp_join_frame_builds_treatment_outcome_and_centrality_z():
    adoption, profiles = _fake_join_rows()

    async def fake_paged(client, table, columns, brand):
        return adoption  # the brand-filtered adoption read

    async def fake_profiles(client):
        return profiles

    with (
        patch.object(causal_routes, "get_async_supabase_client", AsyncMock(return_value=object())),
        patch.object(causal_routes, "_te_paged_select", side_effect=fake_paged),
        patch.object(causal_routes, "_load_hcp_profile_centrality", side_effect=fake_profiles),
    ):
        df, select_cols = await causal_routes._load_agent_estimation_frame(
            dataset="hcp_adoption",
            treatment_var="treatment_arm",
            outcome_var="adopted",
            covariates=["centrality_z"],
            limit=1500,
            brand="Kisqali",
        )
    assert set(select_cols) == {"treatment_arm", "adopted", "centrality_z"}
    assert list(df.columns) == ["treatment_arm", "adopted", "centrality_z"]
    assert len(df) == 3
    # centrality_z = zscore(log1p(influence_network_size)) — h1 (size 25) is the highest.
    raw = [math.log1p(25), math.log1p(2), math.log1p(14)]
    mean = sum(raw) / 3
    std = (sum((x - mean) ** 2 for x in raw) / 3) ** 0.5
    assert df.loc[0, "centrality_z"] == pytest.approx((raw[0] - mean) / std, rel=1e-6)


@pytest.mark.asyncio
async def test_hcp_join_empty_backdoor_question_loads_just_treatment_outcome():
    adoption, profiles = _fake_join_rows()

    async def fake_paged(client, table, columns, brand):
        return adoption

    async def fake_profiles(client):
        return profiles

    with (
        patch.object(causal_routes, "get_async_supabase_client", AsyncMock(return_value=object())),
        patch.object(causal_routes, "_te_paged_select", side_effect=fake_paged),
        patch.object(causal_routes, "_load_hcp_profile_centrality", side_effect=fake_profiles),
    ):
        df, select_cols = await causal_routes._load_agent_estimation_frame(
            dataset="hcp_adoption",
            treatment_var="peer_influence_score",
            outcome_var="adopted",
            covariates=[],  # EXOGENOUS root: empty backdoor
            limit=1500,
            brand="Kisqali",
        )
    assert list(df.columns) == ["peer_influence_score", "adopted"]
    assert len(df) == 3


@pytest.mark.asyncio
async def test_hcp_join_rejects_disallowed_column():
    """The allowlist gate still applies on the JOIN path — an off-allowlist column 400s."""
    with patch.object(causal_routes, "get_async_supabase_client", AsyncMock(return_value=object())):
        with pytest.raises(HTTPException) as ei:
            await causal_routes._load_agent_estimation_frame(
                dataset="hcp_adoption",
                treatment_var="treatment_arm",
                outcome_var="adopted",
                covariates=["specialty"],  # not in the hcp_adoption allowlist
                limit=1500,
                brand="Kisqali",
            )
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_hcp_dataset_enumerates_only_hcp_questions():
    """The HCP dataset must NOT surface patient questions (treatment_arm ->
    persistent_180d) even though they share the causal_paths SSOT."""
    ssot = [
        {"treatment": "peer_influence_score", "outcome": "adopted", "brand": "Kisqali", "confounders": []},
        {"treatment": "treatment_arm", "outcome": "adopted", "brand": "Kisqali", "confounders": ["centrality_z"]},
        {"treatment": "treatment_arm", "outcome": "persistent_180d", "brand": "Kisqali",
         "confounders": ["disease_severity", "academic_hcp", "geographic_region"]},
    ]
    with patch.object(causal_routes, "_get_causal_path_repo") as mk:
        mk.return_value.get_distinct_questions = AsyncMock(return_value=ssot)
        qs = await causal_routes._discover_candidate_questions("hcp_adoption", brand="Kisqali")
    outcomes = {q.outcome for q in qs}
    assert outcomes == {"adopted"}
    treatments = {q.treatment for q in qs}
    assert treatments == {"peer_influence_score", "treatment_arm"}
    # The exogenous-root question keeps an EMPTY adjustment set.
    exo = next(q for q in qs if q.treatment == "peer_influence_score")
    assert exo.adjustment_set == []
    # The rep-engagement question keeps centrality_z (numeric, in the HCP allowlist).
    rep = next(q for q in qs if q.treatment == "treatment_arm")
    assert rep.adjustment_set == ["centrality_z"]


@pytest.mark.asyncio
async def test_patient_dataset_still_excludes_hcp_questions():
    """Symmetry: the patient dataset must NOT surface the adopted-outcome HCP rows."""
    ssot = [
        {"treatment": "treatment_arm", "outcome": "treatment_initiated", "brand": "Fabhalta",
         "confounders": ["disease_severity", "age_at_diagnosis"]},
        {"treatment": "peer_influence_score", "outcome": "adopted", "brand": "Fabhalta", "confounders": []},
    ]
    with patch.object(causal_routes, "_get_causal_path_repo") as mk:
        mk.return_value.get_distinct_questions = AsyncMock(return_value=ssot)
        qs = await causal_routes._discover_candidate_questions("patient_journeys", brand=None)
    assert {q.outcome for q in qs} == {"treatment_initiated"}
    assert all(q.treatment != "peer_influence_score" for q in qs)
