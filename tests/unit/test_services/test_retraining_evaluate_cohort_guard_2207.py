"""#2207 (codex round 1, HIGH): the scheduled evaluation path must not enqueue a
retraining job it cannot describe.

``evaluate_and_trigger_retraining`` — the helper behind the ``evaluate_retraining_need``
Celery task, which the daily ``retraining-evaluation-daily`` beat fans out to — used to
call ``trigger_retraining()`` with NO cohort contract when a decision needed no
approval. ``execute_model_retraining`` requires ``data_source`` + ``target_outcome``
(``_cohort_input_from_training_config`` fails loud), so every such trigger wrote a
``pending`` ml_retraining_history row and queued a job that could only fail closed.
No persisted model record carries a committed cohort identity (ml_model_registry has
no data_source column; ml_experiments has prediction_target/brand only), so the auto
path cannot invent one. It now refuses to trigger without a contract and says so in
its result; a caller that has the contract passes it through.

Scope: this guard covers ONLY the scheduled helper. The API trigger route calls
``RetrainingTriggerService.trigger_retraining`` directly and, by Phase-D design, keeps
``data_source`` / ``target_outcome`` optional (tests/api/test_monitoring_endpoints.py
expects an empty-cohort request to return a pending job) — an incomplete API trigger is
accepted and then fails closed at execution. Tightening that boundary is an owner call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.retraining_trigger import (
    RetrainingDecision,
    TriggerReason,
    evaluate_and_trigger_retraining,
    has_cohort_contract,
)


def _auto_approvable_decision() -> RetrainingDecision:
    return RetrainingDecision(
        should_retrain=True,
        reason=TriggerReason.DATA_DRIFT,
        confidence=0.9,
        drift_score=0.95,
        performance_score=0.7,
        requires_approval=False,
    )


@pytest.mark.asyncio
async def test_no_cohort_contract_means_no_job_is_enqueued():
    service = MagicMock()
    service.evaluate_retraining_need = AsyncMock(return_value=_auto_approvable_decision())
    service.trigger_retraining = AsyncMock()
    with patch(
        "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
    ):
        result = await evaluate_and_trigger_retraining("model_v1", auto_approve=True)

    service.trigger_retraining.assert_not_awaited()
    assert result["should_retrain"] is True
    assert result["retraining_triggered"] is False
    assert result["retraining_blocked_reason"] == "no_cohort_contract"
    assert "job_id" not in result


@pytest.mark.asyncio
async def test_a_partial_cohort_contract_is_not_enough():
    service = MagicMock()
    service.evaluate_retraining_need = AsyncMock(return_value=_auto_approvable_decision())
    service.trigger_retraining = AsyncMock()
    with patch(
        "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
    ):
        result = await evaluate_and_trigger_retraining(
            "model_v1", auto_approve=True, cohort={"data_source": "cohort_x"}
        )
    service.trigger_retraining.assert_not_awaited()
    assert result["retraining_triggered"] is False
    assert result["retraining_blocked_reason"] == "no_cohort_contract"


@pytest.mark.asyncio
async def test_with_a_full_cohort_contract_the_job_is_triggered_with_it():
    service = MagicMock()
    service.evaluate_retraining_need = AsyncMock(return_value=_auto_approvable_decision())
    job = MagicMock(job_id="job-1", new_model_version="model_v1_retrained_x")
    service.trigger_retraining = AsyncMock(return_value=job)
    cohort = {"data_source": "cohort_x", "target_outcome": "persistent_180d", "brand": "Kisqali"}
    with patch(
        "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
    ):
        result = await evaluate_and_trigger_retraining("model_v1", auto_approve=True, cohort=cohort)

    service.trigger_retraining.assert_awaited_once()
    assert service.trigger_retraining.await_args.kwargs["cohort"] == cohort
    assert result["retraining_triggered"] is True
    assert result["job_id"] == "job-1"
    assert "retraining_blocked_reason" not in result


@pytest.mark.asyncio
async def test_a_decision_that_needs_approval_is_still_not_triggered():
    decision = _auto_approvable_decision()
    decision.requires_approval = True
    service = MagicMock()
    service.evaluate_retraining_need = AsyncMock(return_value=decision)
    service.trigger_retraining = AsyncMock()
    cohort = {"data_source": "cohort_x", "target_outcome": "persistent_180d"}
    with patch(
        "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
    ):
        result = await evaluate_and_trigger_retraining(
            "model_v1", auto_approve=False, cohort=cohort
        )
    service.trigger_retraining.assert_not_awaited()
    assert result["retraining_triggered"] is False
    assert result["requires_approval"] is True


def test_has_cohort_contract_handles_none_empty_partial_and_complete():
    assert has_cohort_contract(None) is False
    assert has_cohort_contract({}) is False
    assert has_cohort_contract({"data_source": "cohort_x"}) is False
    assert has_cohort_contract({"data_source": "cohort_x", "target_outcome": ""}) is False
    assert has_cohort_contract({"data_source": "cohort_x", "target_outcome": "y"}) is True


# ---------------------------------------------------------------------------
# #2207 follow-up (2026-09-22): the sweep now supplies the registry row's contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_registry_contract_with_a_file_source_dict_is_triggered_with_it():
    """A JSON-encoded file source decodes back to the loader's dict shape."""
    service = MagicMock()
    service.evaluate_retraining_need = AsyncMock(return_value=_auto_approvable_decision())
    job = MagicMock(job_id="job-2", new_model_version="model_v1_retrained_y")
    service.trigger_retraining = AsyncMock(return_value=job)
    cohort = {
        "data_source": {"type": "file_dir", "path": "data/rwd/optum/initiation"},
        "target_outcome": "initiated_biologic_180d",
    }
    with patch(
        "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
    ):
        result = await evaluate_and_trigger_retraining("model_v1", auto_approve=True, cohort=cohort)
    assert service.trigger_retraining.await_args.kwargs["cohort"] == cohort
    assert result["retraining_triggered"] is True


def test_has_cohort_contract_accepts_a_dict_data_source():
    assert has_cohort_contract({"data_source": {"type": "files"}, "target_outcome": "y"}) is True
    assert has_cohort_contract({"data_source": {}, "target_outcome": "y"}) is False
