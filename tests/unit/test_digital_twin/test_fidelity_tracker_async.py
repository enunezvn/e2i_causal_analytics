"""
Async persistence tests for FidelityTracker (H7 + H7b).

These tests pin the fix for issue #705 Lane 1:

- ``record_prediction`` / ``validate`` / ``get_simulation_record`` /
  ``_find_record_by_simulation`` are ``async`` and ``await`` the (async)
  repository coroutines, so persistence actually happens instead of being a
  silently-dropped, never-awaited coroutine.
- ``validate`` calls the REAL repository method ``update_fidelity_validation``
  (with individual fields), not the non-existent ``update_fidelity_record``.

Run with ``-W error::RuntimeWarning`` to prove no coroutine is left unawaited.
"""

import warnings
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.digital_twin.fidelity_tracker import FidelityTracker
from src.digital_twin.models.simulation_models import (
    FidelityRecord,
    InterventionConfig,
    SimulationRecommendation,
    SimulationResult,
    SimulationStatus,
)


def _make_result() -> SimulationResult:
    return SimulationResult(
        model_id=uuid4(),
        intervention_config=InterventionConfig(
            intervention_type="email_campaign",
            channel="email",
        ),
        twin_count=1000,
        simulated_ate=0.10,
        simulated_ci_lower=0.06,
        simulated_ci_upper=0.14,
        simulated_std_error=0.02,
        status=SimulationStatus.COMPLETED,
        recommendation=SimulationRecommendation.DEPLOY,
        recommendation_rationale="Effect is significant",
        simulation_confidence=0.85,
        execution_time_ms=150,
    )


def _async_repo() -> AsyncMock:
    """A repo stub whose persistence methods are async (like the real one)."""
    repo = AsyncMock()
    repo.save_fidelity_record = AsyncMock(return_value=uuid4())
    repo.update_fidelity_validation = AsyncMock(return_value={"updated": True})
    repo.get_fidelity_by_simulation = AsyncMock(return_value=None)
    # The composite does NOT have update_fidelity_record; make sure the tracker
    # never reaches for it.
    if hasattr(repo, "update_fidelity_record"):
        del repo.update_fidelity_record
    return repo


class TestRecordPredictionAsync:
    async def test_record_prediction_awaits_save(self):
        """record_prediction awaits the async save_fidelity_record exactly once."""
        repo = _async_repo()
        tracker = FidelityTracker(repository=repo)
        result = _make_result()

        record = await tracker.record_prediction(result)

        assert isinstance(record, FidelityRecord)
        repo.save_fidelity_record.assert_awaited_once_with(record)

    async def test_record_prediction_no_unawaited_coroutine(self):
        """No 'coroutine was never awaited' RuntimeWarning (H7 silent drop)."""
        repo = _async_repo()
        tracker = FidelityTracker(repository=repo)
        result = _make_result()

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            await tracker.record_prediction(result)

        repo.save_fidelity_record.assert_awaited_once()


class TestValidateAsync:
    async def test_validate_calls_update_fidelity_validation(self):
        """validate awaits update_fidelity_validation (NOT update_fidelity_record)."""
        repo = _async_repo()
        tracker = FidelityTracker(repository=repo)
        result = _make_result()

        record = await tracker.record_prediction(result)
        experiment_id = uuid4()

        validated = await tracker.validate(
            simulation_id=result.simulation_id,
            actual_ate=0.09,
            actual_ci=(0.05, 0.13),
            actual_sample_size=800,
            actual_experiment_id=experiment_id,
            notes="done",
            validated_by="analyst@company.com",
        )

        repo.update_fidelity_validation.assert_awaited_once()
        args, kwargs = repo.update_fidelity_validation.call_args
        # tracking_id passed positionally (mirrors the repo signature)
        assert args[0] == record.tracking_id
        assert kwargs["actual_ate"] == 0.09
        assert kwargs["actual_ci_lower"] == 0.05
        assert kwargs["actual_ci_upper"] == 0.13
        assert kwargs["actual_sample_size"] == 800
        assert kwargs["actual_experiment_id"] == experiment_id
        assert kwargs["validation_notes"] == "done"
        assert kwargs["validated_by"] == "analyst@company.com"
        assert validated.actual_ate == 0.09

    async def test_validate_no_unawaited_coroutine(self):
        """No un-awaited coroutine warning on the validate path."""
        repo = _async_repo()
        tracker = FidelityTracker(repository=repo)
        result = _make_result()

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            await tracker.record_prediction(result)
            await tracker.validate(simulation_id=result.simulation_id, actual_ate=0.10)

        repo.update_fidelity_validation.assert_awaited_once()


class TestReadPathAsync:
    async def test_get_simulation_record_awaits_repo(self):
        """get_simulation_record awaits get_fidelity_by_simulation on cache-miss."""
        sim_id = uuid4()
        stored = FidelityRecord(
            simulation_id=sim_id,
            simulated_ate=0.10,
            simulated_ci_lower=0.06,
            simulated_ci_upper=0.14,
        )
        repo = _async_repo()
        repo.get_fidelity_by_simulation = AsyncMock(return_value=stored)
        tracker = FidelityTracker(repository=repo)  # empty in-memory cache

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            found = await tracker.get_simulation_record(sim_id)

        assert found is stored
        repo.get_fidelity_by_simulation.assert_awaited_once_with(sim_id)

    async def test_find_record_returns_in_memory_without_repo_call(self):
        """In-memory hit short-circuits before any repo call."""
        repo = _async_repo()
        tracker = FidelityTracker(repository=repo)
        result = _make_result()
        record = await tracker.record_prediction(result)

        found = await tracker.get_simulation_record(result.simulation_id)

        assert found is record
        repo.get_fidelity_by_simulation.assert_not_awaited()
