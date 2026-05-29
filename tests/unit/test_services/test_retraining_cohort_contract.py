"""Phase D (D1): the retraining trigger contract carries cohort identity.

A live retrain needs to know which committed cohort to retrain on. trigger_retraining
must thread a ``cohort`` dict (data_source / target_outcome / brand /
feature_manifest_source) into ``training_config`` so it reaches
``execute_model_retraining`` (and from there ``MLFoundationPipeline.run``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.retraining_trigger import RetrainingTriggerService, TriggerReason


@pytest.mark.asyncio
async def test_trigger_retraining_threads_cohort_into_training_config() -> None:
    service = RetrainingTriggerService()

    record = SimpleNamespace(id="job-1", created_at=datetime.now(timezone.utc))
    captured: dict = {}

    drift_repo = MagicMock()
    drift_repo.get_latest_drift_status = AsyncMock(return_value=[])
    retrain_repo = MagicMock()

    async def _capture_trigger(**kwargs):
        captured.update(kwargs)
        return record

    retrain_repo.trigger_retraining = _capture_trigger

    tracker = MagicMock()
    tracker.get_performance_trend = AsyncMock(side_effect=Exception("no perf"))

    cohort = {
        "data_source": "data/rwd/optum/initiation",
        "target_outcome": "initiated_biologic_180d",
        "brand": "competitor",
        "feature_manifest_source": "optum",
    }

    with (
        patch("src.repositories.drift_monitoring.DriftHistoryRepository", return_value=drift_repo),
        patch(
            "src.repositories.drift_monitoring.RetrainingHistoryRepository",
            return_value=retrain_repo,
        ),
        patch("src.services.performance_tracking.get_performance_tracker", return_value=tracker),
        patch("src.tasks.drift_monitoring_tasks.execute_model_retraining") as mock_task,
    ):
        mock_task.delay = MagicMock(return_value=MagicMock(id="task-1"))
        await service.trigger_retraining(
            model_version="optum_v1",
            reason=TriggerReason.MANUAL,
            cohort=cohort,
        )

    # The persisted training_config carries the cohort identity...
    tc = captured["training_config"]
    assert tc["data_source"] == "data/rwd/optum/initiation"
    assert tc["target_outcome"] == "initiated_biologic_180d"
    assert tc["feature_manifest_source"] == "optum"
    # ...and the queued task receives the same training_config.
    queued_tc = mock_task.delay.call_args.kwargs["training_config"]
    assert queued_tc["data_source"] == "data/rwd/optum/initiation"
    assert queued_tc["feature_manifest_source"] == "optum"
