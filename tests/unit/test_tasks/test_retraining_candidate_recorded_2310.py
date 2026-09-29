"""#2310 (codex r2 on PR #2315): a completion is what was RECORDED, not what was attempted.

``complete_retraining`` and the history ``update`` return None when no row was written. The
task must not report a completed retrain whose deployment link or completion never landed.

Fixture pattern of test_retraining_execute_requires_candidate_2242.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.unit.test_tasks.test_retraining_candidate_contract_2310 import _result
from tests.unit.test_tasks.test_retraining_execute_requires_candidate_2242 import (
    NEW_VERSION,
    _db,
)

pytestmark = pytest.mark.unit


async def _run(db, tc, completion: Any):
    from src.tasks.drift_monitoring_tasks import _execute_real_retraining

    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=_result())
    service = MagicMock()
    service.complete_retraining = AsyncMock(return_value=completion)
    with (
        patch("src.agents.tier_0.pipeline.MLFoundationPipeline", MagicMock(return_value=pipeline)),
        patch(
            "src.repositories.drift_monitoring.get_drift_monitoring_client",
            AsyncMock(return_value=db),
        ),
        patch(
            "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
        ),
    ):
        return await _execute_real_retraining("rt-1", "m", NEW_VERSION, tc)


@pytest.mark.asyncio
async def test_the_control_completes_when_both_writes_land():
    db, tc = _db(with_candidate=True)
    out = await _run(db, tc, completion=MagicMock(name="recorded_job"))
    assert out["status"] == "completed"


@pytest.mark.asyncio
async def test_a_completion_that_recorded_nothing_is_not_a_completion():
    db, tc = _db(with_candidate=True)
    out = await _run(db, tc, completion=None)
    assert out["status"] == "failed"
    assert "completion was not recorded" in out["error"]


@pytest.mark.asyncio
async def test_a_deployment_link_that_updated_no_history_row_is_not_a_completion():
    from src.repositories.drift_monitoring import RetrainingHistoryRepository

    db, tc = _db(with_candidate=True)
    real_update = RetrainingHistoryRepository.update

    async def _update(self, record_id, updates):
        if "deployment_id" in updates:
            return None  # the history row matched nothing
        return await real_update(self, record_id, updates)

    service_completion = MagicMock(name="recorded_job")
    with patch.object(RetrainingHistoryRepository, "update", _update):
        out = await _run(db, tc, completion=service_completion)
    assert out["status"] == "failed"
    assert "could not be linked to the job" in out["error"]
