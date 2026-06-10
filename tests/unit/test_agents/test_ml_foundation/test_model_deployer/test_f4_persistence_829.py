"""F4 follow-up (#829): wire model_deployer persistence — produce a real
``model_registry_id`` (write ``ml_model_registry``) + inject the async Supabase
client so a real deployment writes ``ml_model_registry`` / ``ml_deployments``
rows, and ``db_persisted=True`` ONLY when a row is confirmed in the DB.

These are the CI-safe (no DB, no live MLflow) unit tests covering:

* the ``ModelDeployerState.model_registry_id`` channel field — without it the
  LangGraph ``StateGraph(ModelDeployerState)`` reducer (``extra="ignore"``)
  silently drops the register node's output so it never reaches
  ``_store_to_database``;
* the FAIL-CLOSED contract of the registry-persistence helper — no client / no
  resolvable experiment / no training run => returns ``None`` and writes
  nothing (never fabricates an ``algorithm`` or a registry id);
* the pure mapping helpers (run-id parse, validation-metrics → registry metrics);
* the register_model node CONTRACT: ``model_registry_id`` is ALWAYS present in
  the node output (``None`` when not persisted), and a SIMULATED MLflow
  registration (``simulated://``) never produces a registry id.

REASON-BEFORE-RULES / anti-mocking: NO mocks. The fail-closed paths are driven
with real ``None`` substrate (no client) — a faithful "missing backend"
condition — not a patched stub. The end-to-end real-DB write is proven in
``tests/integration/test_model_deployer_persistence_integration.py``.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    _metrics_to_registry_dict,
    _parse_mlflow_run_id,
    _persist_model_registry_row,
    register_model,
)
from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

# ---------------------------------------------------------------------------
# State channel field
# ---------------------------------------------------------------------------


def test_model_registry_id_is_a_declared_state_field():
    """``model_registry_id`` must be a declared ``ModelDeployerState`` field so
    the register node's output survives the ``extra="ignore"`` channel reducer
    and reaches ``_store_to_database``. Undeclared => silently dropped."""
    state = ModelDeployerState(audit_workflow_id=uuid4(), model_registry_id="reg-abc-123")
    assert state.model_registry_id == "reg-abc-123"


def test_model_registry_id_defaults_to_none():
    state = ModelDeployerState(audit_workflow_id=uuid4())
    assert state.model_registry_id is None


# ---------------------------------------------------------------------------
# Pure mapping helpers
# ---------------------------------------------------------------------------


def test_parse_mlflow_run_id_from_runs_uri():
    assert _parse_mlflow_run_id("runs:/abc123def/model") == "abc123def"
    assert _parse_mlflow_run_id("runs:/abc123def/nested/path") == "abc123def"


def test_parse_mlflow_run_id_returns_none_for_non_runs_uri():
    # MLflow 3.x ``models:/`` URIs carry no run id => None (caller falls back to
    # get_best_run); empty / malformed => None.
    assert _parse_mlflow_run_id("models:/m-deadbeef") is None
    assert _parse_mlflow_run_id("") is None
    assert _parse_mlflow_run_id(None) is None


def test_metrics_to_registry_dict_maps_canonical_keys():
    """validation_metrics (MetricsSchema field name ``auc_roc``) must map onto
    the registry's ``auc`` / ``pr_auc`` / ``brier_score`` / ``calibration_slope``
    keys that ``MLModelRegistryRepository.register_model`` reads."""
    from src.agents.ml_foundation.model_trainer.schemas import MetricsSchema

    vm = MetricsSchema(roc_auc=0.81, pr_auc=0.42, brier_score=0.15, calibration_slope=0.97)
    out = _metrics_to_registry_dict(vm)
    assert out["auc"] == pytest.approx(0.81)
    assert out["pr_auc"] == pytest.approx(0.42)
    assert out["brier_score"] == pytest.approx(0.15)
    assert out["calibration_slope"] == pytest.approx(0.97)


def test_metrics_to_registry_dict_accepts_plain_dict_and_none():
    out = _metrics_to_registry_dict({"auc_roc": 0.7})
    assert out["auc"] == pytest.approx(0.7)
    # None metrics -> all keys present and None (registry columns are nullable).
    out_none = _metrics_to_registry_dict(None)
    assert out_none == {"auc": None, "pr_auc": None, "brier_score": None, "calibration_slope": None}


# ---------------------------------------------------------------------------
# FAIL-CLOSED contract of the persistence helper (no mocks: real None substrate)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_registry_row_fails_closed_without_client():
    """No Supabase client => returns None, writes nothing, fabricates nothing."""
    result = await _persist_model_registry_row(
        None,
        experiment_id_str="exp_anything",
        model_uri="runs:/abc/model",
        registered_model_name="m",
        model_version=1,
        validation_metrics=None,
    )
    assert result is None


# ---------------------------------------------------------------------------
# register_model node CONTRACT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_model_always_emits_model_registry_id_key():
    """The node output must ALWAYS carry ``model_registry_id`` (None when not
    persisted) so ``_store_to_database`` has a definite signal."""
    result = await register_model(
        {
            "model_uri": "simulated://model",
            "deployment_name": "f4_829_contract",
            "experiment_id": "exp_does_not_exist",
        }
    )
    assert "model_registry_id" in result


@pytest.mark.asyncio
async def test_simulated_registration_never_produces_registry_id():
    """A SIMULATED MLflow registration (invalid URI => real connector fails)
    must NOT write ml_model_registry — fail closed, no fabricated id."""
    result = await register_model(
        {
            "model_uri": "simulated://model",
            "deployment_name": "f4_829_sim",
            "experiment_id": "exp_does_not_exist",
        }
    )
    assert result.get("registration_successful") is False
    assert result.get("model_registry_id") is None
