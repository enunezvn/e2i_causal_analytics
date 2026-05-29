"""Repository-level tests for #546 provenance persistence.

``RetrainingHistoryRepository.complete_retraining`` gained an optional
``mlflow_run_id`` provenance pointer. When supplied it must be merged into the
record's ``training_config`` WITHOUT clobbering existing config; when absent the
update payload must be unchanged (non-breaking for the automated path, which
passes no run id). These tests patch the repository's own ``get_by_id``/``update``
so the real merge logic in ``complete_retraining`` is exercised.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.repositories.drift_monitoring import (
    RetrainingHistoryRecord,
    RetrainingHistoryRepository,
)


@pytest.mark.asyncio
async def test_complete_with_provenance_merges_into_training_config():
    """An mlflow_run_id is merged into training_config, preserving existing keys."""
    repo = RetrainingHistoryRepository()
    existing = RetrainingHistoryRecord(
        id="rec-1",
        old_model_version="v1",
        new_model_version="v2",
        trigger_reason="manual",
        drift_score_before=0.0,
        performance_before=0.80,
        training_config={"data_source": "optum_batch_42", "target_outcome": "y"},
        status="training",
    )

    captured: dict = {}

    async def _fake_update(record_id, updates):
        captured["record_id"] = record_id
        captured["updates"] = updates
        return existing  # value irrelevant to this assertion

    with (
        patch.object(repo, "get_by_id", AsyncMock(return_value=existing)),
        patch.object(repo, "update", side_effect=_fake_update),
    ):
        await repo.complete_retraining(
            "rec-1", performance_after=0.91, success=True, mlflow_run_id="run-xyz"
        )

    updates = captured["updates"]
    assert updates["status"] == "completed"
    assert updates["performance_after"] == 0.91
    # provenance merged in...
    assert updates["training_config"]["mlflow_run_id"] == "run-xyz"
    # ...without losing the existing cohort identity.
    assert updates["training_config"]["data_source"] == "optum_batch_42"
    assert updates["training_config"]["target_outcome"] == "y"


@pytest.mark.asyncio
async def test_complete_without_provenance_leaves_training_config_untouched():
    """No mlflow_run_id -> the update payload carries no training_config (the
    automated path passes no run id and must not be perturbed)."""
    repo = RetrainingHistoryRepository()

    captured: dict = {}

    async def _fake_update(record_id, updates):
        captured["updates"] = updates
        return None

    with (
        patch.object(repo, "get_by_id", AsyncMock()) as mock_get,
        patch.object(repo, "update", side_effect=_fake_update),
    ):
        await repo.complete_retraining("rec-2", performance_after=0.0, success=False)

    updates = captured["updates"]
    assert updates["status"] == "failed"
    assert "training_config" not in updates
    # No need to read the existing record when there's no provenance to merge.
    mock_get.assert_not_called()


@pytest.mark.asyncio
async def test_complete_with_provenance_missing_record_returns_none():
    """If the record vanished before completion, return None rather than persist
    a fabricated training_config against a non-existent row."""
    repo = RetrainingHistoryRepository()

    with (
        patch.object(repo, "get_by_id", AsyncMock(return_value=None)),
        patch.object(repo, "update", AsyncMock()) as mock_update,
    ):
        result = await repo.complete_retraining(
            "missing", performance_after=0.9, success=True, mlflow_run_id="run-1"
        )

    assert result is None
    mock_update.assert_not_called()


def test_record_created_at_default_is_tz_aware():
    """Guard: created_at default is timezone-aware (sanity for the model used)."""
    rec = RetrainingHistoryRecord(
        old_model_version="v1",
        new_model_version="v2",
        trigger_reason="manual",
        drift_score_before=0.0,
        performance_before=0.8,
    )
    assert rec.created_at.tzinfo is not None
    assert rec.created_at <= datetime.now(timezone.utc)
