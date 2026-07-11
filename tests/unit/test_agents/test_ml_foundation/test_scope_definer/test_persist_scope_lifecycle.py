"""Scope persistence lifecycle (migration 102).

ml_experiments.status was designed with a lifecycle
(draft/running/completed/stopped/archived) but no writer ever set it: the
repository omitted status from inserts, every scope_definer row inherited the
DB default 'running', and nothing ever transitioned it — so 692 lineage rows
(18 distinct scope names; each Tier-0 pipeline run blind-inserted a duplicate)
were counted as running A/B experiments by the experiment monitor and the
running-count endpoints.

These tests pin the fixed contract of ``_persist_scope_spec``:
- a scope-definition record is complete at write time → status='completed'
- re-runs refresh the existing row (get-or-refresh by name), never duplicate
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.agents.ml_foundation.scope_definer.agent import ScopeDefinerAgent
from src.repositories.ml_experiment import MLExperiment

_OUTPUT = {
    "experiment_id": "exp-abc123",
    "experiment_name": "Kisqali - Predict prescribing",
    "scope_spec": {
        "target_variable": "prescribing",
        "problem_description": "Predict prescribing behavior",
        "brand": "Kisqali",
        "region": "all",
    },
    "success_criteria": {"minimum_auc": 0.75},
}


def _repo(existing: MLExperiment | None) -> AsyncMock:
    repo = AsyncMock()
    repo.get_by_name.return_value = existing
    return repo


@pytest.mark.unit
@pytest.mark.asyncio
async def test_new_scope_created_as_completed():
    """First persistence of a scope creates the row with status='completed'
    (never the DB default 'running' — the record's work is done at insert)."""
    repo = _repo(existing=None)
    with patch(
        "src.agents.ml_foundation.scope_definer.agent._get_experiment_repository",
        new=AsyncMock(return_value=repo),
    ):
        await ScopeDefinerAgent()._persist_scope_spec(_OUTPUT)

    repo.create_experiment.assert_awaited_once()
    kwargs = repo.create_experiment.await_args.kwargs
    assert kwargs["status"] == "completed"
    assert kwargs["name"] == "Kisqali - Predict prescribing"
    assert kwargs["created_by"] == "scope_definer"
    repo.update.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_existing_scope_refreshed_not_duplicated():
    """Re-running the pipeline for a known scope refreshes the existing row
    (update, status='completed') instead of inserting a duplicate — the
    pre-fix behavior accumulated 252 copies of one scope name."""
    existing = MLExperiment(
        id=uuid4(),
        experiment_name="Kisqali - Predict prescribing",
        prediction_target="prescribing",
        created_by="scope_definer",
        status="running",
    )
    repo = _repo(existing=existing)
    with patch(
        "src.agents.ml_foundation.scope_definer.agent._get_experiment_repository",
        new=AsyncMock(return_value=repo),
    ):
        await ScopeDefinerAgent()._persist_scope_spec(_OUTPUT)

    repo.create_experiment.assert_not_awaited()
    repo.update.assert_awaited_once()
    row_id, updates = repo.update.await_args.args
    assert row_id == str(existing.id)
    assert updates["status"] == "completed"
    assert updates["minimum_auc"] == 0.75


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_repository_degrades_gracefully():
    """Repository unavailable → skip persistence without raising (unchanged
    graceful-degradation contract)."""
    with patch(
        "src.agents.ml_foundation.scope_definer.agent._get_experiment_repository",
        new=AsyncMock(return_value=None),
    ):
        await ScopeDefinerAgent()._persist_scope_spec(_OUTPUT)
