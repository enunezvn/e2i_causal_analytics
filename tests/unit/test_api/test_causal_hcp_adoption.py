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
        {
            "treatment": "peer_influence_score",
            "outcome": "adopted",
            "brand": "Kisqali",
            "confounders": [],
        },
        {
            "treatment": "treatment_arm",
            "outcome": "adopted",
            "brand": "Kisqali",
            "confounders": ["centrality_z"],
        },
        {
            "treatment": "treatment_arm",
            "outcome": "persistent_180d",
            "brand": "Kisqali",
            "confounders": ["disease_severity", "academic_hcp", "geographic_region"],
        },
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
        {
            "treatment": "treatment_arm",
            "outcome": "treatment_initiated",
            "brand": "Fabhalta",
            "confounders": ["disease_severity", "age_at_diagnosis"],
        },
        {
            "treatment": "peer_influence_score",
            "outcome": "adopted",
            "brand": "Fabhalta",
            "confounders": [],
        },
    ]
    with patch.object(causal_routes, "_get_causal_path_repo") as mk:
        mk.return_value.get_distinct_questions = AsyncMock(return_value=ssot)
        qs = await causal_routes._discover_candidate_questions("patient_journeys", brand=None)
    assert {q.outcome for q in qs} == {"treatment_initiated"}
    assert all(q.treatment != "peer_influence_score" for q in qs)


# ---------------------------------------------------------------------------
# P2 adversarial-review defect: JOIN datasets must not 500 on single-table reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_causal_variables_hcp_adoption_returns_curated_spec_no_db():
    """list_causal_variables for a JOIN dataset must NOT probe client.table(dataset)
    (which would PostgREST-404 on the non-physical 'hcp_adoption' name and raise
    APIError → HTTP 500). Instead it must return the curated spec lists directly
    with a 200, and the DB client must not be called at all.

    RED before the JOIN short-circuit is added to list_causal_variables.
    """
    from src.api.dependencies.auth import TEST_USER
    from src.api.routes.causal import list_causal_variables

    # A client whose .table() raises immediately — any call == test failure.
    class _NeverCallClient:
        def table(self, name: str):
            raise AssertionError(
                f"list_causal_variables should NOT call client.table('{name}') "
                "for a JOIN dataset — it would 500 in production."
            )

    import src.memory.services.factories as factories

    async def _fake_factory():
        return _NeverCallClient()

    original = factories.get_async_supabase_client
    factories.get_async_supabase_client = _fake_factory
    try:
        response = await list_causal_variables(dataset="hcp_adoption", user=TEST_USER)
    finally:
        factories.get_async_supabase_client = original

    # Must return the curated spec lists — no DB probe, no 500.
    assert set(response.treatment_candidates) == {"peer_influence_score", "treatment_arm"}
    assert response.outcome_candidates == ["adopted"]
    assert response.covariate_candidates == ["centrality_z"]
    # columns is the sorted union of all candidate lists for JOIN datasets
    # (no physical table to read column names from).
    assert set(response.columns) == {
        "peer_influence_score",
        "treatment_arm",
        "adopted",
        "centrality_z",
    }


@pytest.mark.asyncio
async def test_get_causal_estimation_data_hcp_adoption_returns_400_not_500():
    """get_causal_estimation_data for a JOIN dataset must fail-closed with an
    honest 400 (it cannot serve hcp_adoption via a single client.table() read),
    NOT a 500 from a PostgREST APIError on the non-existent table.

    RED before the JOIN guard is added to get_causal_estimation_data.
    """
    from src.api.dependencies.auth import TEST_USER
    from src.api.routes.causal import get_causal_estimation_data

    # A client whose .table() raises to simulate the production 500 path.
    class _NeverCallClient:
        def table(self, name: str):
            raise AssertionError(
                f"get_causal_estimation_data must NOT reach client.table('{name}') "
                "for a JOIN dataset — the JOIN guard must raise 400 before this."
            )

    import src.memory.services.factories as factories

    async def _fake_factory():
        return _NeverCallClient()

    original = factories.get_async_supabase_client
    factories.get_async_supabase_client = _fake_factory
    try:
        with pytest.raises(HTTPException) as exc_info:
            await get_causal_estimation_data(
                treatment_var="treatment_arm",
                outcome_var="adopted",
                dataset="hcp_adoption",
                covariates="centrality_z",
                limit=4000,
                user=TEST_USER,
            )
    finally:
        factories.get_async_supabase_client = original

    # Must be 400 (honest "use the agent path"), never 500.
    assert exc_info.value.status_code == 400
    assert "hcp_adoption" in exc_info.value.detail
    assert "JOIN" in exc_info.value.detail


# ---------------------------------------------------------------------------
# HIGH-1 review finding: /causal/variables must expose human-readable labels
# (parity with /segments/datasets which already has labels from Phase 0)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_causal_variables_patient_journeys_includes_labels():
    """list_causal_variables must return a 'labels' dict with human-readable
    names for every offered candidate column (parity with /segments/datasets).

    RED before the 'labels' field is added to CausalVariablesResponse and the
    handler is updated to populate it from _COLUMN_LABELS.
    """
    from src.api.dependencies.auth import TEST_USER
    from src.api.routes.causal import list_causal_variables

    # Fake probe row: every patient_journeys candidate column is "present".
    _pj_cols = [
        "treatment_arm",
        "treatment_initiated",
        "persistent_180d",
        "discontinued_180d",
        "adherent_180d",
        "low_gap_180d",
        "disease_severity",
        "engagement_score",
        "age_at_diagnosis",
        "academic_hcp",
        "geographic_region",
        "egfr",
        "proteinuria_g_day",
        "ldh_ratio",
        "urticaria_severity_uas7",
        "ecog_performance_status",
    ]
    fake_row = dict.fromkeys(_pj_cols)

    class _FakeQuery:
        def __init__(self, rows):
            self._rows = rows

        def select(self, *a, **kw):
            return self

        def limit(self, *a, **kw):
            return self

        async def execute(self):
            return type("_Result", (), {"data": self._rows})()

    class _FakeClient:
        def table(self, name: str) -> _FakeQuery:
            return _FakeQuery([fake_row])

    import src.memory.services.factories as factories

    original = factories.get_async_supabase_client

    async def _fake_factory():
        return _FakeClient()

    factories.get_async_supabase_client = _fake_factory
    try:
        resp = await list_causal_variables(dataset="patient_journeys", user=TEST_USER)
    finally:
        factories.get_async_supabase_client = original

    # labels present + human-readable for the offered candidates
    assert resp.labels.get("treatment_arm") == "Treatment arm"
    assert resp.labels.get("adherent_180d") == "Adherent at 180d"
    for col in resp.treatment_candidates + resp.outcome_candidates + resp.covariate_candidates:
        assert col in resp.labels, f"{col} has no display label"
