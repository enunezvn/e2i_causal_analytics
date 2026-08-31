# tests/unit/test_api/test_causal_nba_backdoor.py
"""#1872: acceptance_status -> conversion_flag regains its SSOT backdoor set.

Pre-fix, `nba_triggers` declared `covariate: []` (a stale pre-Phase-4 comment:
"no confounder is offered"), so both estimation paths intersected the
registry's modeled confounders (disease_severity + engagement_score,
treatment_arm.ARM_REGISTRY / causal_paths SSOT) down to NOTHING and reported
the naive difference — measured +0.0145 upward confounding bias on Kisqali
(prod DB, 2026-08-31). The confounders are patient_journeys columns, so the
estimation frame must ride the #1188 patient JOIN; the RCT edge
(control_group_flag, randomized) keeps its default-unadjusted behavior.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.api.routes import causal as causal_routes
from src.api.schemas.causal import AgentCausalAnalysisRequest


def _fake_trigger_rows():
    return [
        {
            "acceptance_status": "accepted",
            "conversion_flag": True,
            "patient_id": "pt_000001",
        },
        {
            "acceptance_status": "rejected",
            "conversion_flag": None,  # designed NULL -> fills to 0.0
            "patient_id": "pt_000002",
        },
        {
            "acceptance_status": "pending",
            "conversion_flag": False,
            "patient_id": "pt_000003",
        },
        # Orphan trigger: no matching patient row -> dropped in JOIN mode.
        {
            "acceptance_status": "accepted",
            "conversion_flag": True,
            "patient_id": "pt_999999",
        },
    ]


def _fake_patient_rows():
    return [
        {"patient_id": "pt_000001", "disease_severity": 7.5, "engagement_score": 6.0},
        {"patient_id": "pt_000002", "disease_severity": 2.5, "engagement_score": 4.5},
        {"patient_id": "pt_000003", "disease_severity": 5.0, "engagement_score": 5.0},
    ]


def _patched_reads(patient_rows=None):
    return (
        patch.object(causal_routes, "get_async_supabase_client", AsyncMock(return_value=object())),
        patch.object(
            causal_routes,
            "_load_trigger_question_rows",
            AsyncMock(return_value=_fake_trigger_rows()),
        ),
        patch.object(
            causal_routes,
            "_load_patient_baseline_rows",
            AsyncMock(
                return_value=patient_rows if patient_rows is not None else _fake_patient_rows()
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Loader: backdoor covariates ride the patient JOIN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_covariates_route_through_patient_join():
    """Non-empty covariates alone (no baselines) must activate the JOIN path and
    surface both confounders as real frame columns."""
    p1, p2, p3 = _patched_reads()
    with p1, p2, p3:
        df, select_cols = await causal_routes._load_agent_estimation_frame(
            dataset="nba_triggers",
            treatment_var="acceptance_status",
            outcome_var="conversion_flag",
            covariates=["disease_severity", "engagement_score"],
            limit=1500,
            brand=None,
        )
    # Orphan dropped; the 3 joined rows remain.
    assert len(df) == 3
    assert "disease_severity" in df.columns
    assert "engagement_score" in df.columns
    # Treatment/outcome coercion preserved: accepted -> 1.0, NULL outcome -> 0.0.
    assert sorted(df["acceptance_status"].tolist()) == [0.0, 0.0, 1.0]
    assert sorted(df["conversion_flag"].tolist()) == [0.0, 0.0, 1.0]
    assert set(select_cols) == {
        "acceptance_status",
        "conversion_flag",
        "disease_severity",
        "engagement_score",
    }


@pytest.mark.asyncio
async def test_join_drops_rows_missing_covariate_value():
    rows = _fake_patient_rows()
    rows[1] = {**rows[1], "engagement_score": None}
    p1, p2, p3 = _patched_reads(patient_rows=rows)
    with p1, p2, p3:
        df, _ = await causal_routes._load_agent_estimation_frame(
            dataset="nba_triggers",
            treatment_var="acceptance_status",
            outcome_var="conversion_flag",
            covariates=["disease_severity", "engagement_score"],
            limit=1500,
            brand=None,
        )
    # pt_000002 (missing engagement) and the orphan are dropped.
    assert len(df) == 2
    assert df["engagement_score"].notna().all()


@pytest.mark.asyncio
async def test_join_rejects_disallowed_covariate():
    """Only the curated backdoor pair may ride in through the covariate channel."""
    p1, p2, p3 = _patched_reads()
    with p1, p2, p3, pytest.raises(HTTPException) as exc:
        await causal_routes._load_agent_estimation_frame(
            dataset="nba_triggers",
            treatment_var="acceptance_status",
            outcome_var="conversion_flag",
            covariates=["adherence_rate"],
            limit=1500,
            brand=None,
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_covariates_and_baselines_dedupe_in_join():
    """disease_severity holds BOTH roles (backdoor covariate + curated baseline);
    requesting both must fetch/emit it once, alongside the disjoint columns."""
    rows = [{**r, "age_at_diagnosis": 50.0} for r in _fake_patient_rows()]
    p1, p2, p3 = _patched_reads(patient_rows=rows)
    with p1, p2, p3:
        df, select_cols = await causal_routes._load_agent_estimation_frame(
            dataset="nba_triggers",
            treatment_var="acceptance_status",
            outcome_var="conversion_flag",
            covariates=["disease_severity", "engagement_score"],
            limit=1500,
            brand=None,
            baseline_covariates=["disease_severity", "age_at_diagnosis"],
        )
    assert select_cols.count("disease_severity") == 1
    assert {"disease_severity", "engagement_score", "age_at_diagnosis"} <= set(select_cols)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# Submit endpoint: per-treatment default + role split
# ---------------------------------------------------------------------------


class _BG:
    def __init__(self):
        self.scheduled = []

    def add_task(self, fn, *args):
        self.scheduled.append((fn, args))


def _frame_and_cols(cols):
    import pandas as pd

    return pd.DataFrame({c: [1.0, 0.0] for c in cols}), list(cols)


@pytest.mark.asyncio
async def test_acceptance_edge_defaults_to_ssot_backdoor():
    """request.covariates=None on the OBSERVATIONAL edge must default to the
    curated backdoor pair, landing in the CONFOUNDERS channel."""
    captured: dict = {}

    async def _fake_task(analysis_id, request, df, covariates, data_source, baselines=None):
        captured["covariates"] = covariates
        captured["baselines"] = baselines

    loader = AsyncMock(
        return_value=_frame_and_cols(
            ["acceptance_status", "conversion_flag", "disease_severity", "engagement_score"]
        )
    )
    req = AgentCausalAnalysisRequest(
        treatment_var="acceptance_status",
        outcome_var="conversion_flag",
        dataset="nba_triggers",
        limit=1500,
    )
    bg = _BG()
    with (
        patch.object(causal_routes, "_load_agent_estimation_frame", loader),
        patch.object(causal_routes, "_run_agent_analysis_task", _fake_task),
        patch.object(causal_routes._agent_analysis_store, "set", AsyncMock()),
    ):
        await causal_routes.run_causal_agent_analysis(req, bg, user={"sub": "t"})
        for fn, args in bg.scheduled:
            await fn(*args)

    assert loader.await_args.kwargs["covariates"] == ["disease_severity", "engagement_score"]
    assert captured["covariates"] == ["disease_severity", "engagement_score"]
    assert not captured["baselines"]


@pytest.mark.asyncio
async def test_randomized_treatment_default_stays_unadjusted():
    """The RCT edge's default must remain the unadjusted #1188 design: the
    spec-level covariate OFFER never leaks into a randomized treatment."""
    loader = AsyncMock(return_value=_frame_and_cols(["control_group_flag", "action_taken"]))
    req = AgentCausalAnalysisRequest(
        treatment_var="control_group_flag",
        outcome_var="action_taken",
        dataset="nba_triggers",
        limit=1500,
    )
    with (
        patch.object(causal_routes, "_load_agent_estimation_frame", loader),
        patch.object(causal_routes, "_run_agent_analysis_task", AsyncMock()),
        patch.object(causal_routes._agent_analysis_store, "set", AsyncMock()),
    ):
        await causal_routes.run_causal_agent_analysis(req, _BG(), user={"sub": "t"})

    assert loader.await_args.kwargs["covariates"] == []


@pytest.mark.asyncio
async def test_role_split_covariate_wins_over_baseline():
    """With backdoor covariates AND adjust_baselines on the acceptance edge,
    disease_severity stays a CONFOUNDER (backdoor role wins); only the
    disjoint baselines ride the efficiency channel."""
    captured: dict = {}

    async def _fake_task(analysis_id, request, df, covariates, data_source, baselines=None):
        captured["covariates"] = covariates
        captured["baselines"] = baselines

    loader = AsyncMock(
        return_value=_frame_and_cols(
            [
                "acceptance_status",
                "conversion_flag",
                "disease_severity",
                "engagement_score",
                "age_at_diagnosis",
                "academic_hcp",
                "geographic_region=west",
            ]
        )
    )
    req = AgentCausalAnalysisRequest(
        treatment_var="acceptance_status",
        outcome_var="conversion_flag",
        dataset="nba_triggers",
        limit=1500,
        adjust_baselines=True,
    )
    bg = _BG()
    with (
        patch.object(causal_routes, "_load_agent_estimation_frame", loader),
        patch.object(causal_routes, "_run_agent_analysis_task", _fake_task),
        patch.object(causal_routes._agent_analysis_store, "set", AsyncMock()),
    ):
        await causal_routes.run_causal_agent_analysis(req, bg, user={"sub": "t"})
        for fn, args in bg.scheduled:
            await fn(*args)

    assert "disease_severity" in captured["covariates"]
    assert "engagement_score" in captured["covariates"]
    assert "disease_severity" not in captured["baselines"]
    assert {"age_at_diagnosis", "academic_hcp", "geographic_region=west"} <= set(
        captured["baselines"]
    )


# ---------------------------------------------------------------------------
# Raw estimation-data endpoint: honest 400 for patient-joined covariates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimation_data_rejects_patient_joined_covariates():
    """GET /causal/estimation-data reads ONE physical table; the patient-joined
    confounders are not triggers columns — an honest 400 pointing at the
    join-aware agent path, never a raw PostgREST 42703 500."""
    with pytest.raises(HTTPException) as exc:
        await causal_routes.get_causal_estimation_data(
            treatment_var="acceptance_status",
            outcome_var="conversion_flag",
            dataset="nba_triggers",
            covariates="disease_severity",
            limit=1500,
            user={"sub": "t"},
        )
    assert exc.value.status_code == 400
    assert "join" in str(exc.value.detail).lower()


# ---------------------------------------------------------------------------
# Propose-questions screening: per-treatment covariates (codex iter-1 MED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_questions_screens_rct_unadjusted():
    """The screening endpoint must mirror the submit endpoint's per-treatment
    default: the RANDOMIZED pair screens with NO covariates (single-table path,
    never join-dependent), while the observational acceptance pair screens with
    the SSOT backdoor set."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)

    async def _fake_loader(*, dataset, treatment_var, outcome_var, covariates, limit, **_kw):
        cols = [treatment_var, outcome_var, *covariates]
        return (
            pd.DataFrame({c: rng.normal(size=50) for c in cols}),
            list(cols),
        )

    loader = AsyncMock(side_effect=_fake_loader)
    with patch.object(causal_routes, "_load_agent_estimation_frame", loader):
        resp = await causal_routes.propose_causal_questions(
            dataset="nba_triggers", user={"sub": "t"}
        )

    by_treatment = {
        call.kwargs["treatment_var"]: call.kwargs["covariates"] for call in loader.await_args_list
    }
    assert by_treatment["control_group_flag"] == []
    assert by_treatment["acceptance_status"] == ["disease_severity", "engagement_score"]
    # Both treatments' pairs still make it into the ranked proposals.
    proposed_treatments = {c.treatment for c in resp.candidates}
    assert {"control_group_flag", "acceptance_status"} <= proposed_treatments
