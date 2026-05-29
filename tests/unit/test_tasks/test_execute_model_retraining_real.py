"""Phase D: execute_model_retraining runs the REAL MLFoundationPipeline.

Replaces the Phase-14 simulation stub (`performance_after = 0.85  # Simulated`)
with a real pipeline invocation on a committed cohort. The load-bearing
anti-mocking property: a fake metric is NEVER written — if the pipeline does not
produce a real validation metric (QC-gate halt, exception, missing cohort
identity), the job is marked FAILED and complete_retraining is NOT called.

Covers:
  - _cohort_input_from_training_config: builds the pipeline input from the
    retraining contract; fails loud when cohort identity is missing.
  - _extract_validation_auc: pulls the real validation AUC from the result.
  - _execute_real_retraining: success writes the real metric; every failure
    mode fails closed (status=failed, no complete_retraining, no 0.85).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tasks.drift_monitoring_tasks import (
    _cohort_input_from_training_config,
    _execute_real_retraining,
    _extract_validation_auc,
)

_FULL_COHORT = {
    "data_source": "data/rwd/optum/initiation",
    "target_outcome": "initiated_biologic_180d",
    "brand": "competitor",
    "feature_manifest_source": "optum",
    "problem_description": "Predict biologic initiation",
    "business_objective": "Optimize engagement",
}


# --------------------------------------------------------------------------- #
# _cohort_input_from_training_config
# --------------------------------------------------------------------------- #
def test_cohort_input_maps_all_fields() -> None:
    inp = _cohort_input_from_training_config(dict(_FULL_COHORT))
    assert inp["data_source"] == "data/rwd/optum/initiation"
    assert inp["target_outcome"] == "initiated_biologic_180d"
    assert inp["brand"] == "competitor"
    assert inp["feature_manifest_source"] == "optum"
    # The pipeline requires problem_description + business_objective.
    assert inp["problem_description"]
    assert inp["business_objective"]


def test_cohort_input_missing_data_source_fails_loud() -> None:
    cfg = dict(_FULL_COHORT)
    del cfg["data_source"]
    with pytest.raises(ValueError, match="data_source"):
        _cohort_input_from_training_config(cfg)


def test_cohort_input_missing_target_outcome_fails_loud() -> None:
    cfg = dict(_FULL_COHORT)
    del cfg["target_outcome"]
    with pytest.raises(ValueError, match="target_outcome"):
        _cohort_input_from_training_config(cfg)


# --------------------------------------------------------------------------- #
# _extract_validation_auc
# --------------------------------------------------------------------------- #
def test_extract_auc_reads_roc_auc() -> None:
    result = SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {"roc_auc": 0.661}},
    )
    assert _extract_validation_auc(result) == 0.661


def test_extract_auc_reads_auc_roc_alias() -> None:
    result = SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {"auc_roc": 0.72}},
    )
    assert _extract_validation_auc(result) == 0.72


def test_extract_auc_returns_none_when_absent() -> None:
    assert (
        _extract_validation_auc(SimpleNamespace(status="completed", training_result=None)) is None
    )
    assert (
        _extract_validation_auc(
            SimpleNamespace(status="completed", training_result={"validation_metrics": {}})
        )
        is None
    )


# --------------------------------------------------------------------------- #
# _execute_real_retraining
# --------------------------------------------------------------------------- #
def _patch_pipeline(result: Any) -> Any:
    """Patch MLFoundationPipeline so .run returns the given result."""
    mock_pipeline = MagicMock()
    mock_pipeline.run = AsyncMock(return_value=result)
    mock_cls = MagicMock(return_value=mock_pipeline)
    return patch("src.agents.tier_0.pipeline.MLFoundationPipeline", mock_cls), mock_pipeline


@pytest.mark.asyncio
async def test_real_retraining_success_writes_real_metric() -> None:
    result = SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {"roc_auc": 0.661}, "success_criteria_met": True},
        deployment_result={"model_version": "v2"},
    )
    pipe_patch, mock_pipeline = _patch_pipeline(result)

    repo = MagicMock()
    repo.update = AsyncMock()
    service = MagicMock()
    service.complete_retraining = AsyncMock()

    with (
        pipe_patch,
        patch(
            "src.repositories.drift_monitoring.RetrainingHistoryRepository",
            return_value=repo,
        ),
        patch(
            "src.services.retraining_trigger.get_retraining_trigger_service",
            return_value=service,
        ),
    ):
        out = await _execute_real_retraining("rt-1", "v1", "v2", dict(_FULL_COHORT))

    assert out["status"] == "completed"
    assert out["performance_after"] == 0.661
    # Real pipeline invoked with the cohort-derived input.
    called_input = mock_pipeline.run.call_args.args[0]
    assert called_input["data_source"] == "data/rwd/optum/initiation"
    assert called_input["feature_manifest_source"] == "optum"
    # Real metric persisted (not a fake).
    service.complete_retraining.assert_awaited_once()
    assert service.complete_retraining.await_args.kwargs["performance_after"] == 0.661


@pytest.mark.asyncio
async def test_real_retraining_pipeline_failure_fails_closed() -> None:
    """QC-gate halt (status=failed) → mark failed, NEVER write a metric."""
    result = SimpleNamespace(status="failed", training_result=None, deployment_result=None)
    pipe_patch, _ = _patch_pipeline(result)

    repo = MagicMock()
    repo.update = AsyncMock()
    service = MagicMock()
    service.complete_retraining = AsyncMock()

    with (
        pipe_patch,
        patch("src.repositories.drift_monitoring.RetrainingHistoryRepository", return_value=repo),
        patch(
            "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
        ),
    ):
        out = await _execute_real_retraining("rt-2", "v1", "v2", dict(_FULL_COHORT))

    assert out["status"] == "failed"
    # Anti-mocking: no fake metric written, complete_retraining never called.
    service.complete_retraining.assert_not_awaited()
    # The job was marked failed.
    statuses = [c.args[1].get("status") for c in repo.update.await_args_list if len(c.args) > 1]
    statuses += [c.kwargs.get("updates", {}).get("status") for c in repo.update.await_args_list]
    assert "failed" in [s for s in statuses if s] or any(
        "failed" in str(c) for c in repo.update.await_args_list
    )


@pytest.mark.asyncio
async def test_real_retraining_completed_but_no_metric_fails_closed() -> None:
    """Pipeline 'completed' but produced no validation AUC → cannot certify →
    fail closed rather than write a placeholder."""
    result = SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {}},
        deployment_result=None,
    )
    pipe_patch, _ = _patch_pipeline(result)

    repo = MagicMock()
    repo.update = AsyncMock()
    service = MagicMock()
    service.complete_retraining = AsyncMock()

    with (
        pipe_patch,
        patch("src.repositories.drift_monitoring.RetrainingHistoryRepository", return_value=repo),
        patch(
            "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
        ),
    ):
        out = await _execute_real_retraining("rt-3", "v1", "v2", dict(_FULL_COHORT))

    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_retraining_completed_but_criteria_not_met_fails_closed() -> None:
    """codex HIGH-1: the pipeline sets status='completed' even when it SKIPS
    deployment because success criteria were not met (pipeline.py:553-560). A
    trained-but-not-promotable run must NOT be recorded as a successful retrain —
    fail closed, write no metric."""
    result = SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {"roc_auc": 0.61}, "success_criteria_met": False},
        deployment_result=None,
    )
    pipe_patch, _ = _patch_pipeline(result)

    repo = MagicMock()
    repo.update = AsyncMock()
    service = MagicMock()
    service.complete_retraining = AsyncMock()

    with (
        pipe_patch,
        patch("src.repositories.drift_monitoring.RetrainingHistoryRepository", return_value=repo),
        patch(
            "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
        ),
    ):
        out = await _execute_real_retraining("rt-criteria", "v1", "v2", dict(_FULL_COHORT))

    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_retraining_missing_cohort_fails_closed() -> None:
    """Training config without a data_source can't retrain → fail closed,
    pipeline never invoked, no metric written."""
    pipe_patch, mock_pipeline = _patch_pipeline(SimpleNamespace(status="completed"))

    repo = MagicMock()
    repo.update = AsyncMock()
    service = MagicMock()
    service.complete_retraining = AsyncMock()

    cfg = {"epochs": 100}  # legacy stub-style config, no cohort identity
    with (
        pipe_patch,
        patch("src.repositories.drift_monitoring.RetrainingHistoryRepository", return_value=repo),
        patch(
            "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
        ),
    ):
        out = await _execute_real_retraining("rt-4", "v1", "v2", cfg)

    assert out["status"] == "failed"
    mock_pipeline.run.assert_not_awaited()
    service.complete_retraining.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_retraining_never_writes_simulated_085() -> None:
    """Regression guard against the removed `performance_after = 0.85` stub:
    a real run reports the pipeline's metric, and a failure reports none."""
    result = SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {"roc_auc": 0.638}, "success_criteria_met": True},
        deployment_result=None,
    )
    pipe_patch, _ = _patch_pipeline(result)
    repo = MagicMock()
    repo.update = AsyncMock()
    service = MagicMock()
    service.complete_retraining = AsyncMock()
    with (
        pipe_patch,
        patch("src.repositories.drift_monitoring.RetrainingHistoryRepository", return_value=repo),
        patch(
            "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
        ),
    ):
        out = await _execute_real_retraining("rt-5", "v1", "v2", dict(_FULL_COHORT))
    assert out["performance_after"] == 0.638
    assert out["performance_after"] != 0.85
