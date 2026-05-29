"""#548: trigger_retraining queues the now-existing execute_twin_retraining task.

Before the fix, ``execute_twin_retraining`` did not exist, the import raised
ImportError, and the service logged the MISLEADING "Celery tasks not available"
(Celery IS available — only this one task was missing) while silently no-oping.

After the fix, the import succeeds and the queueing path runs. The honest guard
that remains is for genuine celery-unavailability with a truthful message; the
false "tasks not available" diagnostic for the existing task is gone.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.digital_twin.retraining_service import (
    TwinRetrainingService,
    TwinTriggerReason,
)


@pytest.mark.asyncio
async def test_trigger_retraining_queues_real_task(caplog) -> None:
    """trigger_retraining imports + queues the real execute_twin_retraining
    task — the import no longer fails, so the misleading 'Celery tasks not
    available' branch is NOT taken for this task."""
    service = TwinRetrainingService()
    model_id = uuid4()

    fake_async_result = MagicMock()
    fake_async_result.id = "celery-task-id-123"
    mock_task = MagicMock()
    mock_task.delay = MagicMock(return_value=fake_async_result)

    with (
        patch(
            "src.tasks.ab_testing_tasks.execute_twin_retraining",
            mock_task,
        ),
        caplog.at_level(logging.INFO),
    ):
        job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

    # The real task was queued with the job's identity + contract.
    mock_task.delay.assert_called_once()
    kwargs = mock_task.delay.call_args.kwargs
    assert kwargs["retraining_job_id"] == job.job_id
    assert kwargs["model_id"] == str(model_id)
    assert kwargs["training_config"] == job.training_config

    # The misleading diagnostic must NOT appear for the now-existing task.
    assert "Celery tasks not available" not in caplog.text
    assert "Queued retraining task: celery-task-id-123" in caplog.text


@pytest.mark.asyncio
async def test_trigger_retraining_honest_when_celery_unavailable(caplog) -> None:
    """If queueing genuinely fails (broker down), the job is still created and a
    TRUTHFUL message is logged — not the misleading 'tasks not available'."""
    service = TwinRetrainingService()
    model_id = uuid4()

    mock_task = MagicMock()
    mock_task.delay = MagicMock(side_effect=OSError("broker unreachable"))

    with (
        patch("src.tasks.ab_testing_tasks.execute_twin_retraining", mock_task),
        caplog.at_level(logging.ERROR),
    ):
        job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

    assert job is not None
    assert "Celery tasks not available" not in caplog.text
    assert "Failed to queue retraining task" in caplog.text
