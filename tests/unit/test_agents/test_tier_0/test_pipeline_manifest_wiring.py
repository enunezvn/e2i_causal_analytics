"""Phase B: MLFoundationPipeline is the live-ready manifest origin.

The pipeline (used by the API and the live retraining trigger — NOT the
step-runner scripts) must resolve the cohort manifest from its inputs and
thread it through BOTH downstream consumers:

  - scope_definer (via scope_input → scope_spec → data_preparer), so Layer-1
    manifest contracts engage; and
  - model_deployer (via deployer_input["scope_spec"]), so the regulatory
    deployment manifest can read feature_manifest_source at promotion time.

Pre-fix: _run_scope_definition never set feature_manifest_source, and
deployer_input omitted scope_spec entirely — so the manifest machinery (incl.
PR #544's declared-safe FDR honor) was inert on the programmatic pipeline path.

Mirrors the D5.0 structural-decider wiring tests (_StopAfterCapture).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.tier_0.pipeline import (
    MLFoundationPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)


class _StopAfterCapture(Exception):
    """Stop the stage once the downstream agent's input is captured."""


def _result() -> PipelineResult:
    r = PipelineResult(
        pipeline_run_id="pipe_test",
        status="running",
        current_stage=PipelineStage.SCOPE_DEFINITION,
        experiment_id="exp_test",
    )
    r.scope_spec = {"experiment_id": "exp_test"}
    return r


@pytest.mark.asyncio
async def test_scope_definition_autoresolves_manifest_from_data_source() -> None:
    pipeline = MLFoundationPipeline(config=PipelineConfig(enable_feast=False))
    captured: dict = {}

    async def _capture_run(scope_input):
        captured.update(scope_input)
        raise _StopAfterCapture()

    mock = MagicMock()
    mock.run = _capture_run
    pipeline._get_agent = MagicMock(return_value=mock)

    input_data = {
        "problem_description": "p",
        "business_objective": "b",
        "target_outcome": "t",
        "data_source": "data/rwd/optum/initiation",
    }
    try:
        await pipeline._run_scope_definition(input_data, _result(), {})
    except _StopAfterCapture:
        pass

    assert captured.get("feature_manifest_source") == "optum"


@pytest.mark.asyncio
async def test_scope_definition_honors_explicit_manifest_override() -> None:
    pipeline = MLFoundationPipeline(config=PipelineConfig(enable_feast=False))
    captured: dict = {}

    async def _capture_run(scope_input):
        captured.update(scope_input)
        raise _StopAfterCapture()

    mock = MagicMock()
    mock.run = _capture_run
    pipeline._get_agent = MagicMock(return_value=mock)

    # data_source is a bare table name (no path segment) → explicit override is
    # the live-trigger mechanism.
    input_data = {
        "problem_description": "p",
        "business_objective": "b",
        "target_outcome": "t",
        "data_source": "ml_patient_journeys",
        "feature_manifest_source": "optum",
    }
    try:
        await pipeline._run_scope_definition(input_data, _result(), {})
    except _StopAfterCapture:
        pass

    assert captured.get("feature_manifest_source") == "optum"


@pytest.mark.asyncio
async def test_scope_definition_unknown_source_stays_unset() -> None:
    pipeline = MLFoundationPipeline(config=PipelineConfig(enable_feast=False))
    captured: dict = {}

    async def _capture_run(scope_input):
        captured.update(scope_input)
        raise _StopAfterCapture()

    mock = MagicMock()
    mock.run = _capture_run
    pipeline._get_agent = MagicMock(return_value=mock)

    input_data = {
        "problem_description": "p",
        "business_objective": "b",
        "target_outcome": "t",
        "data_source": "some_unregistered_table",
    }
    try:
        await pipeline._run_scope_definition(input_data, _result(), {})
    except _StopAfterCapture:
        pass

    assert captured.get("feature_manifest_source") is None


@pytest.mark.asyncio
async def test_model_deployment_threads_scope_spec_to_deployer() -> None:
    pipeline = MLFoundationPipeline(config=PipelineConfig(enable_feast=False))
    result = _result()
    result.scope_spec = {"experiment_id": "exp_test", "feature_manifest_source": "optum"}
    result.training_result = {
        "success_criteria_met": True,
        "validation_metrics": {"roc_auc": 0.66},
        "model_artifact_uri": "mlflow://model",
    }

    captured: dict = {}

    async def _capture_run(deployer_input):
        captured.update(deployer_input)
        raise _StopAfterCapture()

    mock = MagicMock()
    mock.run = _capture_run
    pipeline._get_agent = MagicMock(return_value=mock)

    input_data = {"data_source": "data/rwd/optum/initiation", "target_environment": "staging"}
    try:
        await pipeline._run_model_deployment(input_data, result, {})
    except _StopAfterCapture:
        pass

    assert "scope_spec" in captured, "deployer_input must carry scope_spec"
    assert captured["scope_spec"].get("feature_manifest_source") == "optum"
