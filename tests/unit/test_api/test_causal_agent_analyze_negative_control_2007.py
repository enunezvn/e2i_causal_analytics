# tests/unit/test_api/test_causal_agent_analyze_negative_control_2007.py
"""Lane G (#2007): the negative-control-outcome registry, the loader's
passthrough column and the agent state key the refutation node reads.

A negative-control outcome is an outcome the treatment cannot causally affect
that shares the treatment's confounders (Lipsitch, Tchetgen Tchetgen & Cohen
2010). The registry is DECLARED, never inferred from discovery, and lists ONLY
the (treatment -> control) pairs the 2026-09-11 omitted-confounder disproof
(``docs/demos/results/2026-09-11_negative_control_disproof/disproof.md``)
measured to respond on the synthetic generator at n = 1500 — declaring a
control that does not respond would PASS under confounding (false assurance).

Same stubbing seams as ``test_causal_randomized_flag.py`` (the task's graph +
store) and ``test_causal_geo_encoding.py`` (the loader's Supabase client).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.api.routes import causal as causal_routes
from src.api.routes.causal import (
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_NEGATIVE_CONTROL_OUTCOMES,
    _negative_control_outcome,
)
from src.api.schemas.causal import AgentCausalAnalysisRequest

pytestmark = pytest.mark.unit

# _load_agent_estimation_frame does a FUNCTION-LOCAL import of
# get_async_supabase_client, so patch the SOURCE module.
_CLIENT_FACTORY = "src.memory.services.factories.get_async_supabase_client"


class _FakeQuery:
    def __init__(self, rows, selected):
        self._rows = rows
        self._selected = selected

    def select(self, cols, *_a, **_k):
        self._selected.append(cols)
        return self

    def eq(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    async def execute(self):
        return type("R", (), {"data": self._rows})()


class _FakeClient:
    def __init__(self, rows):
        self._rows = rows
        self.selected: list = []

    def table(self, *_a, **_k):
        return _FakeQuery(self._rows, self.selected)


# ---------------------------------------------------------------------------
# (a) the resolver — fail-closed like _is_randomized_treatment
# ---------------------------------------------------------------------------


class TestNegativeControlOutcome:
    @pytest.mark.parametrize(
        ("treatment", "outcome", "expected"),
        [
            ("copay_support", "adherent_180d", "treatment_initiated"),
            ("psp_enrolled", "adherent_180d", "treatment_initiated"),
            ("rep_detailing_high", "treatment_initiated", "persistent_180d"),
        ],
    )
    def test_measured_responders_are_declared(self, treatment, outcome, expected):
        assert _negative_control_outcome("patient_journeys", treatment, outcome) == expected

    @pytest.mark.parametrize("treatment", ["sample_dropped", "trigger_accepted", "treatment_arm"])
    def test_arms_without_a_responding_control_are_none(self, treatment):
        # sample_dropped / trigger_accepted: no candidate control responds at
        # n = 1500 (disproof 0/3 each); treatment_arm has no structural null.
        assert _negative_control_outcome("patient_journeys", treatment, "adherent_180d") is None

    def test_control_equal_to_the_outcome_under_test_is_none(self):
        # psp_enrolled -> treatment_initiated IS the declared control; a control
        # cannot be the outcome under test.
        assert (
            _negative_control_outcome("patient_journeys", "psp_enrolled", "treatment_initiated")
            is None
        )

    def test_unknown_dataset_fails_closed(self):
        assert _negative_control_outcome("nonexistent", "copay_support", "adherent_180d") is None

    def test_datasets_without_a_registry_entry_are_none(self):
        assert _negative_control_outcome("hcp_adoption", "treatment_arm", "adopted") is None
        assert (
            _negative_control_outcome("nba_triggers", "acceptance_status", "conversion_flag")
            is None
        )

    def test_none_dataset_resolves_the_default_dataset_like_the_randomized_flag(self):
        assert (
            _negative_control_outcome(None, "copay_support", "adherent_180d")
            == "treatment_initiated"
        )

    def test_unlisted_mapped_column_is_never_fetched(self, monkeypatch):
        # Fail-closed: a registry value outside the spec's outcome list is
        # refused at the resolver (the loader would never see it).
        monkeypatch.setitem(
            causal_routes._CAUSAL_NEGATIVE_CONTROL_OUTCOMES,
            "patient_journeys",
            {"copay_support": "not_an_outcome_column"},
        )
        assert (
            _negative_control_outcome("patient_journeys", "copay_support", "adherent_180d") is None
        )


# ---------------------------------------------------------------------------
# (b) registry contract pin
# ---------------------------------------------------------------------------


def test_registry_values_are_spec_outcomes_and_keys_are_spec_treatments():
    assert set(_CAUSAL_NEGATIVE_CONTROL_OUTCOMES) == {"patient_journeys"}
    for dataset, mapping in _CAUSAL_NEGATIVE_CONTROL_OUTCOMES.items():
        spec = _CAUSAL_DATASET_SPECS[dataset]
        assert set(mapping) <= set(spec["treatment"]), mapping
        assert set(mapping.values()) <= set(spec["outcome"]), mapping
    # The disproof's exact responders — and NOTHING for the two arms where no
    # candidate control responded (a declared non-responder is false assurance).
    assert _CAUSAL_NEGATIVE_CONTROL_OUTCOMES["patient_journeys"] == {
        "copay_support": "treatment_initiated",
        "psp_enrolled": "treatment_initiated",
        "rep_detailing_high": "persistent_180d",
    }


# ---------------------------------------------------------------------------
# (c) the agent task + the submit endpoint
# ---------------------------------------------------------------------------


class _MemStore:
    def __init__(self) -> None:
        self._d: dict = {}

    async def get(self, key):
        return self._d.get(key)

    async def set(self, key, value):
        self._d[key] = value


def _capture_task_state(monkeypatch) -> dict:
    import src.agents.causal_impact.graph as graph_mod

    captured: dict = {}

    class _FakeGraph:
        async def ainvoke(self, state, **kwargs):
            captured.update(state)
            raise RuntimeError("stop after capture")

    monkeypatch.setattr(graph_mod, "create_causal_impact_graph", lambda: _FakeGraph())
    monkeypatch.setattr(causal_routes, "_agent_analysis_store", _MemStore())
    return captured


@pytest.mark.asyncio
async def test_task_threads_negative_control_outcome_and_splits_the_column(monkeypatch):
    captured = _capture_task_state(monkeypatch)
    df = pd.DataFrame(
        {
            "copay_support": [0.0, 1.0, 1.0],
            "adherent_180d": [0.0, 1.0, 0.0],
            "insurance_access_score": [0.2, 0.9, 0.5],
            "treatment_initiated": [1.0, None, 0.0],
        }
    )
    req = AgentCausalAnalysisRequest(
        treatment_var="copay_support",
        outcome_var="adherent_180d",
        dataset="patient_journeys",
    )
    await causal_routes._run_agent_analysis_task(
        "aid-nc", req, df, ["insurance_access_score"], "live"
    )
    assert captured.get("negative_control_outcome") == "treatment_initiated"
    # The adjustment channels are untouched by the passthrough column.
    assert captured["confounders"] == ["insurance_access_score"]
    assert captured["modeled_confounders"] == ["insurance_access_score"]
    # The control rides in its OWN data_cache entry: never a column of the
    # estimation frame (guided discovery tiers every non-question frame column
    # as a covariate; the estimator's no-backdoor fallback adjusts on every
    # other column), index-aligned so the node fits on the identical rows.
    est = captured["data_cache"]["estimation_data"]
    nc = captured["data_cache"]["negative_control_data"]
    assert "treatment_initiated" not in est.columns
    assert list(est.columns) == ["copay_support", "adherent_180d", "insurance_access_score"]
    assert list(nc.columns) == ["treatment_initiated"]
    assert nc.index.equals(est.index)
    assert nc["treatment_initiated"].isna().sum() == 1  # NULLs kept; the node drops its own


@pytest.mark.asyncio
async def test_task_sets_negative_control_outcome_none_when_undeclared(monkeypatch):
    captured = _capture_task_state(monkeypatch)
    df = pd.DataFrame({"treatment_arm": [0, 1], "persistent_180d": [0, 1]})
    req = AgentCausalAnalysisRequest(
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        dataset="patient_journeys",
    )
    await causal_routes._run_agent_analysis_task("aid-none", req, df, [], "live")
    assert "negative_control_outcome" in captured
    assert captured["negative_control_outcome"] is None
    assert "negative_control_data" not in captured["data_cache"]
    assert list(captured["data_cache"]["estimation_data"].columns) == [
        "treatment_arm",
        "persistent_180d",
    ]


@pytest.mark.asyncio
async def test_task_without_the_column_in_the_frame_still_declares_the_key(monkeypatch):
    """A declared control whose column did not make it into the frame (e.g. a
    caller that bypassed the submit endpoint): the key is still set so the
    node reports ``negative_control_column_missing`` itself; nothing is
    fabricated and the estimation frame is passed through untouched."""
    captured = _capture_task_state(monkeypatch)
    df = pd.DataFrame({"copay_support": [0.0, 1.0], "adherent_180d": [0.0, 1.0]})
    req = AgentCausalAnalysisRequest(
        treatment_var="copay_support",
        outcome_var="adherent_180d",
        dataset="patient_journeys",
    )
    await causal_routes._run_agent_analysis_task("aid-miss", req, df, [], "live")
    assert captured["negative_control_outcome"] == "treatment_initiated"
    assert "negative_control_data" not in captured["data_cache"]
    assert captured["data_cache"]["estimation_data"] is df


def test_negative_control_outcome_is_a_declared_state_channel():
    from src.agents.causal_impact.state import CausalImpactState

    assert "negative_control_outcome" in CausalImpactState.__annotations__


class _BG:
    def __init__(self):
        self.scheduled: list = []

    def add_task(self, fn, *args):
        self.scheduled.append((fn, args))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("treatment", "outcome", "expected_passthrough"),
    [
        ("copay_support", "adherent_180d", ["treatment_initiated"]),
        ("treatment_arm", "persistent_180d", None),
    ],
)
async def test_submit_passes_the_control_as_a_loader_passthrough_column(
    treatment, outcome, expected_passthrough
):
    frame = pd.DataFrame(
        {treatment: [1.0, 0.0], outcome: [1.0, 0.0], "disease_severity": [2.0, 1.0]}
    )
    if expected_passthrough:
        frame[expected_passthrough[0]] = [1.0, 0.0]
    expanded_cols = [treatment, outcome, "disease_severity"]
    req = AgentCausalAnalysisRequest(
        treatment_var=treatment,
        outcome_var=outcome,
        dataset="patient_journeys",
        covariates=["disease_severity"],
        limit=1500,
    )
    loader = AsyncMock(return_value=(frame, expanded_cols))
    captured: dict = {}

    async def _fake_task(
        analysis_id, request, df, covariates, data_source, baseline_covariates=None
    ):
        captured["covariates"] = covariates

    bg = _BG()
    with (
        patch.object(causal_routes, "_load_agent_estimation_frame", loader),
        patch.object(causal_routes, "_run_agent_analysis_task", _fake_task),
        patch.object(causal_routes._agent_analysis_store, "set", AsyncMock()),
    ):
        await causal_routes.run_causal_agent_analysis(req, bg, user={"sub": "t"})
        for fn, args in bg.scheduled:
            await fn(*args)

    assert loader.await_count == 1
    assert loader.await_args.kwargs["passthrough_columns"] == expected_passthrough
    # The passthrough column never reaches the adjustment set.
    assert captured["covariates"] == ["disease_severity"]


# ---------------------------------------------------------------------------
# (d) loader semantics on a stubbed client
# ---------------------------------------------------------------------------


def _pj_rows():
    return [
        {
            "copay_support": 1,
            "adherent_180d": 1,
            "insurance_access_score": 0.9,
            "treatment_initiated": "1",
        },
        {
            "copay_support": 0,
            "adherent_180d": 0,
            "insurance_access_score": 0.2,
            "treatment_initiated": None,
        },
        {
            "copay_support": None,
            "adherent_180d": 1,
            "insurance_access_score": 0.5,
            "treatment_initiated": 1,
        },
        {
            "copay_support": 1,
            "adherent_180d": 0,
            "insurance_access_score": 0.7,
            "treatment_initiated": 0,
        },
    ]


@pytest.mark.asyncio
async def test_loader_passthrough_is_fetched_coerced_kept_and_out_of_the_adjustment_set():
    client = _FakeClient(_pj_rows())
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=client)):
        df, expanded_cols = await causal_routes._load_agent_estimation_frame(
            dataset="patient_journeys",
            treatment_var="copay_support",
            outcome_var="adherent_180d",
            covariates=["insurance_access_score"],
            limit=1500,
            passthrough_columns=["treatment_initiated"],
        )
    # Fetched in the same select as the question columns.
    assert client.selected == [
        "copay_support,adherent_180d,insurance_access_score,treatment_initiated"
    ]
    # In the frame, numeric-coerced ("1" -> 1.0) ...
    assert "treatment_initiated" in df.columns
    assert df["treatment_initiated"].dtype == float
    # ... a NULL treatment drops its row, a NULL passthrough does NOT.
    assert len(df) == 3
    assert df["treatment_initiated"].isna().sum() == 1
    assert df["copay_support"].isna().sum() == 0
    # ... and NEVER in the adjustment set the caller passes as confounders.
    assert expanded_cols == ["copay_support", "adherent_180d", "insurance_access_score"]


@pytest.mark.asyncio
async def test_loader_keeps_an_all_null_passthrough_column():
    rows = [dict(r, treatment_initiated=None) for r in _pj_rows()]
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(rows))):
        df, expanded_cols = await causal_routes._load_agent_estimation_frame(
            dataset="patient_journeys",
            treatment_var="copay_support",
            outcome_var="adherent_180d",
            covariates=["insurance_access_score"],
            limit=1500,
            passthrough_columns=["treatment_initiated"],
        )
    assert "treatment_initiated" in df.columns
    assert bool(df["treatment_initiated"].isna().all())
    assert len(df) == 3
    assert "treatment_initiated" not in expanded_cols


@pytest.mark.asyncio
async def test_loader_still_drops_an_all_null_covariate_beside_a_passthrough():
    rows = [dict(r, insurance_access_score=None) for r in _pj_rows()]
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(rows))):
        df, expanded_cols = await causal_routes._load_agent_estimation_frame(
            dataset="patient_journeys",
            treatment_var="copay_support",
            outcome_var="adherent_180d",
            covariates=["insurance_access_score"],
            limit=1500,
            passthrough_columns=["treatment_initiated"],
        )
    assert "insurance_access_score" not in df.columns
    assert "treatment_initiated" in df.columns
    assert expanded_cols == ["copay_support", "adherent_180d"]


@pytest.mark.asyncio
async def test_loader_never_one_hot_expands_a_passthrough_column():
    rows = [
        {
            "treatment_arm": 1,
            "persistent_180d": 1,
            "disease_severity": 2,
            "geographic_region": "south",
        },
        {
            "treatment_arm": 0,
            "persistent_180d": 0,
            "disease_severity": 1,
            "geographic_region": "west",
        },
        {
            "treatment_arm": 1,
            "persistent_180d": 1,
            "disease_severity": 3,
            "geographic_region": "midwest",
        },
    ]
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(rows))):
        df, expanded_cols = await causal_routes._load_agent_estimation_frame(
            dataset="patient_journeys",
            treatment_var="treatment_arm",
            outcome_var="persistent_180d",
            covariates=["disease_severity"],
            limit=1500,
            passthrough_columns=["geographic_region"],
        )
    assert "geographic_region" in df.columns
    assert not [c for c in df.columns if c.startswith("geographic_region=")]
    assert expanded_cols == ["treatment_arm", "persistent_180d", "disease_severity"]


@pytest.mark.asyncio
async def test_loader_without_passthrough_is_unchanged():
    client = _FakeClient(_pj_rows())
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=client)):
        df, expanded_cols = await causal_routes._load_agent_estimation_frame(
            dataset="patient_journeys",
            treatment_var="copay_support",
            outcome_var="adherent_180d",
            covariates=["insurance_access_score"],
            limit=1500,
        )
    assert client.selected == ["copay_support,adherent_180d,insurance_access_score"]
    assert "treatment_initiated" not in df.columns
    assert expanded_cols == ["copay_support", "adherent_180d", "insurance_access_score"]


@pytest.mark.asyncio
async def test_loader_passthrough_is_not_subject_to_the_covariate_role_guard():
    """treatment_initiated holds NO covariate role (test_causal_covariate_roles
    rejects it as a covariate); as a passthrough it is accepted."""
    client = _FakeClient(_pj_rows())
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=client)):
        df, _ = await causal_routes._load_agent_estimation_frame(
            dataset="patient_journeys",
            treatment_var="copay_support",
            outcome_var="adherent_180d",
            covariates=[],
            limit=1500,
            passthrough_columns=["treatment_initiated"],
        )
    assert "treatment_initiated" in df.columns


# ---------------------------------------------------------------------------
# (e) fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loader_rejects_a_passthrough_column_outside_the_allowlist():
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient([]))):
        with pytest.raises(causal_routes.HTTPException) as ei:
            await causal_routes._load_agent_estimation_frame(
                dataset="patient_journeys",
                treatment_var="copay_support",
                outcome_var="adherent_180d",
                covariates=[],
                limit=10,
                passthrough_columns=["totally_made_up_col"],
            )
    assert ei.value.status_code == 400
    assert "not permitted" in str(ei.value.detail)
    assert "totally_made_up_col" in str(ei.value.detail)


@pytest.mark.asyncio
async def test_loader_refuses_a_passthrough_on_the_join_datasets():
    """The JOIN loaders (hcp_adoption; nba_triggers with covariates) do not
    carry a passthrough column. No registry entry exists for them today; a
    future entry must extend the JOIN loader, never be silently dropped (the
    node would then report column_missing for a column the route promised)."""
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient([]))):
        with pytest.raises(causal_routes.HTTPException) as ei:
            await causal_routes._load_agent_estimation_frame(
                dataset="hcp_adoption",
                treatment_var="treatment_arm",
                outcome_var="adopted",
                covariates=["centrality_z"],
                limit=10,
                passthrough_columns=["adopted"],
            )
    assert ei.value.status_code == 400
    assert "passthrough" in str(ei.value.detail)


# ---------------------------------------------------------------------------
# (f) end-to-end shape contract for T4: loader -> task split
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loader_then_task_gives_index_aligned_frames_for_the_node(monkeypatch):
    """The refutation node aligns the control by INDEX
    (``negative_control_data.loc[refutation_data.index]`` after the #1419
    subsample, which is ``frame.iloc[...]`` and keeps the original labels).
    Pins: both frames are built from the same records BEFORE the split, so
    every row-drop the loader applies (a NULL treatment here) applies to both
    identically; a NULL control drops nothing (stays NaN); the estimation
    frame never carries the control column."""
    captured = _capture_task_state(monkeypatch)
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(_pj_rows()))):
        df, expanded_cols = await causal_routes._load_agent_estimation_frame(
            dataset="patient_journeys",
            treatment_var="copay_support",
            outcome_var="adherent_180d",
            covariates=["insurance_access_score"],
            limit=1500,
            passthrough_columns=["treatment_initiated"],
        )
    req = AgentCausalAnalysisRequest(
        treatment_var="copay_support",
        outcome_var="adherent_180d",
        dataset="patient_journeys",
    )
    confounders = [c for c in expanded_cols if c not in ("copay_support", "adherent_180d")]
    await causal_routes._run_agent_analysis_task("aid-e2e", req, df, confounders, "live")

    est = captured["data_cache"]["estimation_data"]
    nc = captured["data_cache"]["negative_control_data"]
    # Same rows: the NULL-treatment row is absent from BOTH; the NULL control
    # row is present in BOTH (NaN in the control frame).
    assert len(est) == len(nc) == 3
    assert nc.index.equals(est.index)
    assert nc["treatment_initiated"].isna().sum() == 1
    assert "treatment_initiated" not in est.columns
    assert list(nc.columns) == ["treatment_initiated"]
    assert captured["negative_control_outcome"] == "treatment_initiated"
    assert "treatment_initiated" not in captured["confounders"]
    assert "treatment_initiated" not in captured["modeled_confounders"]
    # T4's alignment on a positional subsample of the estimation frame.
    subsample = est.iloc[[2, 0]]
    aligned = nc.loc[subsample.index]
    assert aligned.index.equals(subsample.index)
    assert len(aligned) == 2
