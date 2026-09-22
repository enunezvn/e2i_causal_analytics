"""Lane C (spec 2026-09-22 §3C.2): the pre-wired ``csu_escalation_causal``
dataset in the registry, the production loader on its shape (coercion,
one-hot with the ``__missing__`` level, the real-mode provenance filter, the
constant-treatment refusal) and the per-dataset ``auto_discover`` default.

Fake-client seams as in test_causal_agent_analyze_negative_control_2007.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.routes.causal import agent as causal_routes
from src.api.routes.causal import catalog, discovery
from src.api.routes.causal import datasets as datasets_mod
from src.api.routes.causal.datasets import (
    _ALL_CLINICAL_COVARIATES,
    _CAUSAL_BRAND_COLUMN,
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_DISCOVERY_DEFAULT_OFF,
    _CAUSAL_NEGATIVE_CONTROL_OUTCOMES,
    _CAUSAL_NUMERIC_COLUMNS,
    _CAUSAL_PHYSICAL_TABLE,
    _CAUSAL_SYNTHETIC_BACKED,
    _JOIN_DATASETS,
    _brand_scoped_covariates,
    _default_auto_discover,
    _is_randomized_treatment,
    _list_dataset_brands,
    _negative_control_outcome,
    apply_dataset_provenance_filter,
    serves_synthetic_rows,
)
from src.api.routes.causal.loaders import _load_agent_estimation_frame
from src.api.schemas.causal import AgentCausalAnalysisRequest
from src.data.manifests import MART_SAFE_FEATURES
from src.ml.synthetic.generators.csu_escalation_causal import (
    CONTRACT_COLUMNS,
    generate_csu_escalation_cohort,
)
from src.repositories.provenance import PROVENANCE_TAGGED_TABLES

pytestmark = pytest.mark.unit

DATASET = "csu_escalation_causal"
TABLE = "csu_escalation_causal"
TREATMENT = "treatment_remibrutinib"
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


# ---------------------------------------------------------------------------
# registry pins
# ---------------------------------------------------------------------------


def test_spec_treatment_outcomes_and_covariates():
    spec = _CAUSAL_DATASET_SPECS[DATASET]
    assert spec["treatment"] == [TREATMENT]
    assert spec["outcome"] == list(OUTCOMES)  # primary first
    assert spec["covariate"] == list(MART_SAFE_FEATURES)
    assert len(spec["covariate"]) == 64
    assert "randomized_treatment" not in spec  # observational
    assert "baseline_covariate" not in spec
    assert _is_randomized_treatment(DATASET, TREATMENT) is False


def test_physical_table_brand_column_and_single_table_path():
    # The dataset key IS the table name (migration 149): no remap needed.
    assert _CAUSAL_PHYSICAL_TABLE.get(DATASET, DATASET) == TABLE
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


def test_registry_columns_exist_in_the_synthetic_backing_contract():
    """Every column the registry can ask the table for is a column the backing
    (and migration 149) carries -- a registry/backing drift fails here, not as
    a PostgREST 42703 at run time."""
    spec = _CAUSAL_DATASET_SPECS[DATASET]
    requested = {*spec["treatment"], *spec["outcome"], *spec["covariate"], "index_biologic_brand"}
    assert requested <= set(CONTRACT_COLUMNS)


def test_no_negative_control_is_declared_until_measured_on_this_source():
    assert DATASET not in _CAUSAL_NEGATIVE_CONTROL_OUTCOMES
    assert _negative_control_outcome(DATASET, TREATMENT, OUTCOMES[0]) is None


def test_brand_scoping_passes_every_baseline_feature_through():
    assert not (set(MART_SAFE_FEATURES) & _ALL_CLINICAL_COVARIATES)
    assert _brand_scoped_covariates(list(MART_SAFE_FEATURES), None) == list(MART_SAFE_FEATURES)


# ---------------------------------------------------------------------------
# auto_discover default
# ---------------------------------------------------------------------------


def test_discovery_is_off_by_default_for_the_claims_contract_dataset():
    assert DATASET in _CAUSAL_DISCOVERY_DEFAULT_OFF
    assert _default_auto_discover(DATASET) is False
    assert _default_auto_discover("patient_journeys") is True
    assert _default_auto_discover(None) is True


# ---------------------------------------------------------------------------
# loader: the production loader on the backing's rows (fake client)
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, rows, log):
        self._rows, self._log = rows, log

    def select(self, cols, *_a, **_k):
        self._log.append(("select", cols))
        return self

    def eq(self, col, value, *_a, **_k):
        self._log.append(("eq", col, value))
        if col == "is_synthetic":
            self._rows = [r for r in self._rows if bool(r.get("is_synthetic")) is bool(value)]
        else:
            self._rows = [r for r in self._rows if r.get(col) == value]
        return self

    def limit(self, n, *_a, **_k):
        self._log.append(("limit", n))
        self._rows = self._rows[:n]
        return self

    async def execute(self):
        return type("R", (), {"data": self._rows})()


class _FakeClient:
    """Rows behave like the table: ``eq`` filters them, including the
    provenance predicate -- so a real-mode read of an all-synthetic backing
    returns nothing, exactly as PostgREST would."""

    def __init__(self, rows):
        self._rows, self.log, self.tables = rows, [], []

    def table(self, name, *_a, **_k):
        self.tables.append(name)
        return _FakeQuery(list(self._rows), self.log)


def _backing_rows(n: int = 300):
    frame, _ = generate_csu_escalation_cohort(n=n, seed=11)
    frame = frame.astype(object).where(frame.notna(), None)
    return frame.to_dict(orient="records")


def _covariates():
    return ["age_at_index", "charlson_score", "payer_category", "gdr_cd", "geographic_region"]


def _planted_truth_run(monkeypatch):
    """The ONLY switch that reads the synthetic backing: the planted-truth
    module seam, which no deployment's environment can set. The
    deployment-wide showcase flag is unset here so the tests prove the seam
    alone unlocks the rows."""
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.setattr(datasets_mod, "PLANTED_TRUTH_RUN", True)


def _deployed_flag_only(monkeypatch):
    """The deployed e2i_api container's environment (docker inspect,
    2026-09-22: E2I_INCLUDE_SYNTHETIC=true) with the seam at its shipped
    value (False)."""
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    monkeypatch.setattr(datasets_mod, "PLANTED_TRUTH_RUN", False)


@pytest.mark.asyncio
async def test_real_mode_returns_no_rows_from_the_synthetic_backing(monkeypatch):
    """Spec §3C.2: the backing rows are is_synthetic=true, so the real-mode
    provenance filter yields nothing -- a 503, never a synthetic estimate
    served as real."""
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    client = _FakeClient(_backing_rows())
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    with pytest.raises(HTTPException) as exc:
        await _load_agent_estimation_frame(
            dataset=DATASET,
            treatment_var=TREATMENT,
            outcome_var=OUTCOMES[0],
            covariates=_covariates(),
            limit=1000,
        )
    assert exc.value.status_code == 503
    assert client.tables == [TABLE]
    assert ("eq", "is_synthetic", False) in client.log


@pytest.mark.asyncio
async def test_deployed_include_synthetic_flag_does_not_unlock_the_synthetic_backing(monkeypatch):
    """Verifier MED-B (2026-09-22): the deployed e2i_api sets
    E2I_INCLUDE_SYNTHETIC=true, which makes apply_provenance_filter a no-op
    for every reader -- so on the deployment the flag alone would have served
    the planted rows as if real once loaded. The dataset-level guard applies
    the real-mode predicate REGARDLESS of that flag: 503, never a synthetic
    estimate served as real."""
    _deployed_flag_only(monkeypatch)
    assert DATASET in _CAUSAL_SYNTHETIC_BACKED
    client = _FakeClient(_backing_rows())
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    with pytest.raises(HTTPException) as exc:
        await _load_agent_estimation_frame(
            dataset=DATASET,
            treatment_var=TREATMENT,
            outcome_var=OUTCOMES[0],
            covariates=_covariates(),
            limit=1000,
        )
    assert exc.value.status_code == 503
    assert client.tables == [TABLE]
    assert ("eq", "is_synthetic", False) in client.log
    assert serves_synthetic_rows(DATASET) is False


class _GuardQuery:
    def __init__(self):
        self.eqs: list = []

    def eq(self, col, value):
        self.eqs.append((col, value))
        return self


def test_dataset_provenance_guard_ignores_the_deployment_flag_for_the_backing(monkeypatch):
    """The guard's truth table: for the synthetic-backed dataset only the
    planted-truth opt-in reads the rows; every other dataset keeps the
    deployment-wide behaviour of apply_provenance_filter."""
    _deployed_flag_only(monkeypatch)
    assert apply_dataset_provenance_filter(_GuardQuery(), DATASET).eqs == [("is_synthetic", False)]
    # The real-backed Optum dataset is pinned real-only too (live verify
    # 2026-09-22): the deployment rule applies to datasets in neither set.
    assert apply_dataset_provenance_filter(_GuardQuery(), "optum_biologic_persistence").eqs == [
        ("is_synthetic", False)
    ]
    assert serves_synthetic_rows("optum_biologic_persistence") is False
    assert apply_dataset_provenance_filter(_GuardQuery(), "patient_journeys").eqs == []
    assert serves_synthetic_rows("patient_journeys") is True
    _planted_truth_run(monkeypatch)
    # Planted mode reads ONLY the planted rows -- never an unfiltered mixture.
    assert apply_dataset_provenance_filter(_GuardQuery(), DATASET).eqs == [("is_synthetic", True)]
    assert serves_synthetic_rows(DATASET) is True
    # Strict real-data instance: the predicate for everyone, seam shipped.
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.setattr(datasets_mod, "PLANTED_TRUTH_RUN", False)
    assert datasets_mod.PLANTED_TRUTH_RUN is False  # the shipped value
    assert apply_dataset_provenance_filter(_GuardQuery(), DATASET).eqs == [("is_synthetic", False)]
    assert apply_dataset_provenance_filter(_GuardQuery(), "optum_biologic_persistence").eqs == [
        ("is_synthetic", False)
    ]
    assert serves_synthetic_rows(DATASET) is False


@pytest.mark.asyncio
async def test_brand_dropdown_hides_the_synthetic_backing_from_the_deployed_flag(monkeypatch):
    _deployed_flag_only(monkeypatch)
    client = _FakeClient(_backing_rows())
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    assert await _list_dataset_brands(DATASET) == []
    assert ("eq", "is_synthetic", False) in client.log
    _planted_truth_run(monkeypatch)
    client = _FakeClient(_backing_rows())
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    assert await _list_dataset_brands(DATASET) == ["DUPIXENT", "RHAPSIDO", "XOLAIR"]
    assert ("eq", "is_synthetic", True) in client.log


@pytest.mark.asyncio
async def test_variables_probe_hides_the_synthetic_backing_from_the_deployed_flag(monkeypatch):
    """codex r1 MED: /causal/variables probes one row to learn the live
    columns; that probe must not pull a planted row into the API process on
    the deployed instance either."""
    _deployed_flag_only(monkeypatch)
    client = _FakeClient(_backing_rows())
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    resp = await catalog.list_causal_variables(
        dataset=DATASET, brand=None, user={"role": "analyst"}
    )
    assert client.tables == [TABLE]
    assert ("eq", "is_synthetic", False) in client.log
    # The route still offers the registry's DECLARED variables (it needs no
    # row for that); what it must never do is fetch a planted row to learn them.
    assert resp.dataset == DATASET


def test_every_route_labels_data_source_by_the_dataset_rule():
    """codex r1 HIGH: the discovery leaderboard's agent results labelled
    data_source by the deployment-wide flag, inverting the label both ways
    for the synthetic-backed dataset. Every route that stamps data_source
    must go through serves_synthetic_rows(dataset)."""
    import inspect

    for module in (causal_routes, discovery, catalog):
        source = inspect.getsource(module)
        assert "deployment_includes_synthetic(" not in source, module.__name__
    assert "serves_synthetic_rows(dataset)" in inspect.getsource(discovery)
    assert "serves_synthetic_rows(request.dataset)" in inspect.getsource(causal_routes)


@pytest.mark.asyncio
async def test_estimation_data_route_hides_the_synthetic_backing_from_the_deployed_flag(
    monkeypatch,
):
    _deployed_flag_only(monkeypatch)
    client = _FakeClient(_backing_rows())
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    with pytest.raises(HTTPException) as exc:
        await catalog.get_causal_estimation_data(
            treatment_var=TREATMENT,
            outcome_var=OUTCOMES[0],
            dataset=DATASET,
            covariates="age_at_index",
            limit=1000,
            user={"role": "analyst"},
        )
    assert exc.value.status_code == 503
    assert ("eq", "is_synthetic", False) in client.log


@pytest.mark.asyncio
async def test_planted_truth_run_loads_coerces_and_one_hots_the_backing(monkeypatch):
    _planted_truth_run(monkeypatch)
    client = _FakeClient(_backing_rows())
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    frame, cols = await _load_agent_estimation_frame(
        dataset=DATASET,
        treatment_var=TREATMENT,
        outcome_var=OUTCOMES[0],
        covariates=_covariates(),
        limit=1000,
    )
    assert client.tables == [TABLE]
    assert ("eq", "is_synthetic", True) in client.log  # only the planted rows
    assert len(frame) == 300
    assert set(frame[TREATMENT].unique()) == {0.0, 1.0}
    assert frame["age_at_index"].dtype.kind == "f" and frame["charlson_score"].dtype.kind == "f"
    assert "payer_category" not in frame.columns and "payer_category=medicare" in cols
    assert "gdr_cd=M" in cols
    # The backing plants NULL regions: they get their own dummy, never the
    # drop_first reference level.
    assert "geographic_region=__missing__" in cols
    assert frame["geographic_region=__missing__"].sum() > 0
    assert TREATMENT in cols and OUTCOMES[0] in cols
    assert all(frame[c].dtype.kind == "f" for c in cols)


@pytest.mark.asyncio
async def test_loader_rejects_an_outcome_in_the_covariate_slot(monkeypatch):
    _planted_truth_run(monkeypatch)
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(_backing_rows())))
    with pytest.raises(HTTPException) as exc:
        await _load_agent_estimation_frame(
            dataset=DATASET,
            treatment_var=TREATMENT,
            outcome_var=OUTCOMES[0],
            covariates=["discontinued_180d"],
            limit=100,
        )
    assert exc.value.status_code == 400 and "discontinued_180d" in exc.value.detail


@pytest.mark.asyncio
async def test_brand_scope_makes_the_treatment_constant_and_is_refused(monkeypatch):
    """The brand filter IS the treatment label: brand=RHAPSIDO leaves one arm
    -> 400 at load, never a finite-but-meaningless estimate."""
    _planted_truth_run(monkeypatch)
    client = _FakeClient(_backing_rows())
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    with pytest.raises(HTTPException) as exc:
        await _load_agent_estimation_frame(
            dataset=DATASET,
            treatment_var=TREATMENT,
            outcome_var=OUTCOMES[0],
            covariates=["age_at_index"],
            limit=1000,
            brand="RHAPSIDO",
        )
    assert exc.value.status_code == 400
    assert TREATMENT in exc.value.detail and "constant" in exc.value.detail.lower()
    assert ("eq", "index_biologic_brand", "RHAPSIDO") in client.log


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
            OUTCOMES[0]: [1.0, 0.0, 1.0],
            "age_at_index": [61.0, 34.0, 45.0],
        }
    )
    monkeypatch.setattr(
        causal_routes,
        "_load_agent_estimation_frame",
        AsyncMock(return_value=(df, [TREATMENT, OUTCOMES[0], "age_at_index"])),
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
async def test_submit_defaults_discovery_off_for_the_dataset(monkeypatch):
    resp, task_request = await _submit(
        monkeypatch,
        treatment_var=TREATMENT,
        outcome_var=OUTCOMES[0],
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
        outcome_var=OUTCOMES[0],
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
    import inspect

    from src.api.routes.causal import discovery

    src = inspect.getsource(discovery)
    assert "auto_discover=True" not in src
    assert "auto_discover=_default_auto_discover(dataset)" in src
