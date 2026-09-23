"""Lane A: the REAL Optum causal dataset ``optum_biologic_persistence`` in the
registry, the loader's coercion/one-hot on its shape, the covariate role gate,
and the per-dataset ``auto_discover`` default (spec 2026-09-22 §3A.3-4).

Fake-client seams as in test_causal_agent_analyze_negative_control_2007.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.routes.causal import agent as causal_routes
from src.api.routes.causal.datasets import (
    _ALL_CLINICAL_COVARIATES,
    _CAUSAL_BRAND_COLUMN,
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_DISCOVERY_DEFAULT_OFF,
    _CAUSAL_NEGATIVE_CONTROL_OUTCOMES,
    _CAUSAL_NUMERIC_COLUMNS,
    _CAUSAL_PHYSICAL_TABLE,
    _JOIN_DATASETS,
    _brand_scoped_covariates,
    _default_auto_discover,
    _negative_control_outcome,
    apply_dataset_provenance_filter,
    serves_synthetic_rows,
)
from src.api.routes.causal.loaders import _load_agent_estimation_frame
from src.api.schemas.causal import AgentCausalAnalysisRequest
from src.data.manifests import MART_SAFE_FEATURES
from src.repositories.provenance import PROVENANCE_TAGGED_TABLES

pytestmark = pytest.mark.unit

DATASET = "optum_biologic_persistence"
TABLE = "optum_biologic_persistence_causal"
TREATMENT = "treatment_dupixent"
OUTCOMES = (
    "persistent_at_180d_g28",
    "discontinued_180d",
    "biologic_switch_180d_flag",
    "persistent_at_180d",
)
CATEGORICALS = {
    "gdr_cd",
    "payer_category",
    "payer_product",
    "payer_bus",
    "charlson_risk_band",
    "elixhauser_risk_band",
    "geographic_region",
}
_CLIENT_FACTORY = "src.memory.services.factories.get_async_supabase_client"


class _GuardQuery:
    def __init__(self):
        self.eqs = []

    def eq(self, col, value):
        self.eqs.append((col, value))
        return self


def test_real_backed_dataset_ignores_the_deployment_flag(monkeypatch):
    """Live verify 2026-09-22 (docs/demos/results/2026-09-22_lane_c_live_verify):
    the deployed e2i_api sets E2I_INCLUDE_SYNTHETIC=true, under which the Optum
    dataset followed the deployment rule -- no provenance predicate on its read
    and ``serves_synthetic_rows`` True, so /causal/agent-analyze would have
    labelled the real 15,209-row cohort ``data_source="synthetic"``. The
    program defines this dataset as real-only (spec §5: is_synthetic=false on
    every row; the real-mode filter applies), so its read predicate and its
    label must not depend on the showcase flag."""
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    assert apply_dataset_provenance_filter(_GuardQuery(), DATASET).eqs == [("is_synthetic", False)]
    assert serves_synthetic_rows(DATASET) is False
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    assert apply_dataset_provenance_filter(_GuardQuery(), DATASET).eqs == [("is_synthetic", False)]
    assert serves_synthetic_rows(DATASET) is False
    # A dataset outside both sets keeps the deployment-wide behaviour.
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    assert apply_dataset_provenance_filter(_GuardQuery(), "patient_journeys").eqs == []
    assert serves_synthetic_rows("patient_journeys") is True


# ---------------------------------------------------------------------------
# registry pins
# ---------------------------------------------------------------------------


def test_spec_treatment_outcomes_and_covariates():
    spec = _CAUSAL_DATASET_SPECS[DATASET]
    assert spec["treatment"] == [TREATMENT]
    assert spec["outcome"] == list(OUTCOMES)  # primary first
    assert spec["covariate"] == list(MART_SAFE_FEATURES)
    assert len(spec["covariate"]) == 64
    assert (
        "randomized_treatment" not in spec
    )  # observational: keeps the unmeasured-confounding gate
    assert "baseline_covariate" not in spec


def test_physical_table_brand_column_and_single_table_path():
    assert _CAUSAL_PHYSICAL_TABLE[DATASET] == TABLE
    assert _CAUSAL_BRAND_COLUMN[DATASET] == "index_biologic_brand"
    assert DATASET not in _JOIN_DATASETS
    assert TABLE in PROVENANCE_TAGGED_TABLES


def test_every_baseline_feature_has_exactly_one_coercion_role():
    numeric = _CAUSAL_NUMERIC_COLUMNS[DATASET]
    categorical = _CAUSAL_CATEGORICAL_COLUMNS[DATASET]
    assert categorical == CATEGORICALS
    assert not (numeric & categorical)
    assert set(MART_SAFE_FEATURES) == (numeric | categorical) - {TREATMENT, *OUTCOMES}
    assert {TREATMENT, *OUTCOMES} <= numeric


def test_no_negative_control_is_declared_until_measured_on_this_source():
    assert DATASET not in _CAUSAL_NEGATIVE_CONTROL_OUTCOMES
    assert _negative_control_outcome(DATASET, TREATMENT, OUTCOMES[0]) is None


def test_brand_scoping_passes_every_baseline_feature_through():
    # none of the 64 names is a synthetic clinical biomarker, so brand=None keeps all
    assert not (set(MART_SAFE_FEATURES) & _ALL_CLINICAL_COVARIATES)
    assert _brand_scoped_covariates(list(MART_SAFE_FEATURES), None) == list(MART_SAFE_FEATURES)


# ---------------------------------------------------------------------------
# auto_discover default
# ---------------------------------------------------------------------------


def test_discovery_is_off_by_default_only_for_the_claims_frame_datasets():
    # Exact pin: the real cohort and Lane C's pre-wired csu_escalation_causal
    # (same 64-feature claims-frame shape, measured singular) and nothing else.
    assert _CAUSAL_DISCOVERY_DEFAULT_OFF == frozenset({DATASET, "csu_escalation_causal"})
    assert _default_auto_discover(DATASET) is False
    assert _default_auto_discover("patient_journeys") is True
    assert _default_auto_discover(None) is True


# ---------------------------------------------------------------------------
# loader: coercion + one-hot on this dataset's shape (fake client)
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, rows, selected):
        self._rows, self._selected = rows, selected

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
        self._rows, self.selected, self.tables = rows, [], []

    def table(self, name, *_a, **_k):
        self.tables.append(name)
        return _FakeQuery(self._rows, self.selected)


def _rows():
    return [
        {
            TREATMENT: 0,
            "persistent_at_180d_g28": 1,
            "age_at_index": 61,
            "payer_category": "commercial",
            "gdr_cd": "F",
            "geographic_region": None,
        },
        {
            TREATMENT: 1,
            "persistent_at_180d_g28": 0,
            "age_at_index": 34,
            "payer_category": "medicare",
            "gdr_cd": "M",
            "geographic_region": "south",
        },
        {
            TREATMENT: 1,
            "persistent_at_180d_g28": 1,
            "age_at_index": 45,
            "payer_category": "medicare_lis_dual",
            "gdr_cd": "F",
            "geographic_region": "west",
        },
    ]


@pytest.mark.asyncio
async def test_loader_reads_the_physical_table_and_one_hots_the_categoricals(monkeypatch):
    client = _FakeClient(_rows())
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    frame, cols = await _load_agent_estimation_frame(
        dataset=DATASET,
        treatment_var=TREATMENT,
        outcome_var="persistent_at_180d_g28",
        covariates=["age_at_index", "payer_category", "gdr_cd", "geographic_region"],
        limit=10,
    )
    assert client.tables == [TABLE]
    assert frame[TREATMENT].tolist() == [0.0, 1.0, 1.0]
    assert frame["age_at_index"].dtype.kind == "f"
    assert "payer_category" not in frame.columns and "payer_category=medicare" in frame.columns
    assert "gdr_cd=M" in cols and "geographic_region=west" in cols
    # Row 0's geographic_region is NULL: it must NOT silently collapse into the
    # drop_first reference level ("south") — it gets its own __missing__ dummy.
    assert "geographic_region=__missing__" in cols
    assert frame["geographic_region=__missing__"].tolist() == [1.0, 0.0, 0.0]
    assert set(cols) == {
        TREATMENT,
        "persistent_at_180d_g28",
        "age_at_index",
        "payer_category=medicare",
        "payer_category=medicare_lis_dual",
        "gdr_cd=M",
        "geographic_region=west",
        "geographic_region=__missing__",
    }


@pytest.mark.asyncio
async def test_loader_rejects_an_outcome_in_the_covariate_slot(monkeypatch):
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(_rows())))
    with pytest.raises(HTTPException) as exc:
        await _load_agent_estimation_frame(
            dataset=DATASET,
            treatment_var=TREATMENT,
            outcome_var="persistent_at_180d_g28",
            covariates=["discontinued_180d"],
            limit=10,
        )
    assert exc.value.status_code == 400 and "discontinued_180d" in exc.value.detail


@pytest.mark.asyncio
async def test_loader_refuses_a_constant_treatment(monkeypatch):
    """The brand dropdown offers index_biologic_brand because the table has it,
    but scoping to one brand makes treatment_dupixent constant — refused at
    load (400), never a silent finite-but-meaningless DoWhy estimate."""
    rows = [dict(r, **{TREATMENT: 1}) for r in _rows()]
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(rows)))
    with pytest.raises(HTTPException) as exc:
        await _load_agent_estimation_frame(
            dataset=DATASET,
            treatment_var=TREATMENT,
            outcome_var="persistent_at_180d_g28",
            covariates=["age_at_index"],
            limit=10,
        )
    assert exc.value.status_code == 400
    assert TREATMENT in exc.value.detail
    assert "constant" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_loader_still_loads_the_two_arm_cohort(monkeypatch):
    """The existing two-arm fake (unchanged from _rows()) must still load fine
    — the constant-treatment guard must not false-positive on a real contrast."""
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(_rows())))
    frame, cols = await _load_agent_estimation_frame(
        dataset=DATASET,
        treatment_var=TREATMENT,
        outcome_var="persistent_at_180d_g28",
        covariates=["age_at_index"],
        limit=10,
    )
    assert frame[TREATMENT].tolist() == [0.0, 1.0, 1.0]
    assert TREATMENT in cols


# ---------------------------------------------------------------------------
# submit endpoint: auto_discover resolution
# ---------------------------------------------------------------------------


class _MemStore:
    def __init__(self) -> None:
        self._d: dict = {}

    async def get(self, key):
        return self._d.get(key)

    async def set(self, key, value):
        self._d[key] = value


def _stub_submit(monkeypatch):
    df = pd.DataFrame(
        {
            TREATMENT: [0.0, 1.0, 1.0],
            "persistent_at_180d_g28": [1.0, 0.0, 1.0],
            "age_at_index": [61.0, 34.0, 45.0],
        }
    )
    monkeypatch.setattr(
        causal_routes,
        "_load_agent_estimation_frame",
        AsyncMock(return_value=(df, [TREATMENT, "persistent_at_180d_g28", "age_at_index"])),
    )
    monkeypatch.setattr(causal_routes, "_agent_analysis_store", _MemStore())


async def _submit(monkeypatch, **request_kwargs):
    _stub_submit(monkeypatch)
    tasks = BackgroundTasks()
    req = AgentCausalAnalysisRequest(**request_kwargs)
    resp = await causal_routes.run_causal_agent_analysis(req, tasks, user={"role": "analyst"})
    task_request = tasks.tasks[0].args[1]
    return resp, task_request


@pytest.mark.asyncio
async def test_submit_defaults_discovery_off_for_the_real_dataset(monkeypatch):
    resp, task_request = await _submit(
        monkeypatch,
        treatment_var=TREATMENT,
        outcome_var="persistent_at_180d_g28",
        dataset=DATASET,
        covariates=["age_at_index"],
    )
    assert task_request.auto_discover is False
    assert any("discovery" in w.lower() and DATASET in w for w in resp.warnings)


@pytest.mark.asyncio
async def test_submit_honors_an_explicit_opt_in(monkeypatch):
    _, task_request = await _submit(
        monkeypatch,
        treatment_var=TREATMENT,
        outcome_var="persistent_at_180d_g28",
        dataset=DATASET,
        covariates=["age_at_index"],
        auto_discover=True,
    )
    assert task_request.auto_discover is True


@pytest.mark.asyncio
async def test_submit_keeps_the_schema_default_for_synthetic_datasets(monkeypatch):
    _, task_request = await _submit(
        monkeypatch,
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        dataset="patient_journeys",
        covariates=["disease_severity"],
    )
    assert task_request.auto_discover is True


def test_discovery_leaderboard_uses_the_per_dataset_default():
    # A source pin (the leaderboard job is not unit-runnable): the hard-coded
    # ``auto_discover=True`` must be gone from discovery.py.
    import inspect

    from src.api.routes.causal import discovery

    src = inspect.getsource(discovery)
    assert "auto_discover=True" not in src
    assert "auto_discover=_default_auto_discover(dataset)" in src


def test_spec_declares_the_optum_mart_feature_manifest():
    """Lane B item 1 (owner GO 2026-09-23): the real dataset names the manifest
    its feature-role panel and structural-author review are built under
    (``scripts/measure_feature_role_panel.py --manifest-source optum_mart``). Two
    consumers arm on it in src/api/routes/causal/agent.py: the structural-prior
    lookup (an APPROVED review is matched by (T, Y, brand, manifest) — with no
    approved row it stays silent) and the request-panel check (a panel built
    under another manifest is refused with a 400). Lane C's
    ``csu_escalation_causal`` and the synthetic datasets declare nothing."""
    from src.data.manifests import MANIFEST_SOURCES

    spec = _CAUSAL_DATASET_SPECS[DATASET]
    assert spec["feature_manifest_source"] == "optum_mart"
    assert "optum_mart" in MANIFEST_SOURCES  # a registered manifest key, not a label
    for other in ("csu_escalation_causal", "patient_journeys", "hcp_adoption", "nba_triggers"):
        assert "feature_manifest_source" not in _CAUSAL_DATASET_SPECS[other], other
