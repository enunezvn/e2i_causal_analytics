"""Gap G7: the data_preparer DQ DB-row must persist the real leakage verdict.

The agent computes a canonical ``leakage_detected`` bool (leakage_detector.py)
and persists it to MLflow, but ``_persist_qc_report`` built the
ml_data_quality_reports row WITHOUT it — so DataQualityReportRepository.
store_result defaulted the column to False and a stored row understated leakage
risk as always-False (and check_data_quality_gate, which reads it, would never
see a detected leak). These tests pin that the verdict is forwarded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.ml_foundation.data_preparer import DataPreparerAgent

_AGENT_MOD = "src.agents.ml_foundation.data_preparer.agent._get_dq_repository"


def _minimal_qc_report() -> dict:
    return {
        "experiment_id": "exp_g7_1",
        "status": "passed",
        "overall_score": 0.91,
        "expectation_results": [],
        "failed_expectations": [],
    }


@pytest.mark.asyncio
async def test_persist_qc_report_persists_leakage_detected_true() -> None:
    agent = DataPreparerAgent()
    mock_repo = AsyncMock()
    with patch(_AGENT_MOD, return_value=mock_repo):
        await agent._persist_qc_report(
            _minimal_qc_report(), "patient_journeys", leakage_detected=True
        )
    mock_repo.store_result.assert_awaited_once()
    db_record = mock_repo.store_result.await_args.args[0]
    assert db_record["leakage_detected"] is True, (
        "a detected leak must be persisted to the DQ row, not dropped to the "
        "store_result default of False (gap G7)."
    )


@pytest.mark.asyncio
async def test_persist_qc_report_persists_leakage_detected_false() -> None:
    agent = DataPreparerAgent()
    mock_repo = AsyncMock()
    with patch(_AGENT_MOD, return_value=mock_repo):
        await agent._persist_qc_report(
            _minimal_qc_report(), "patient_journeys", leakage_detected=False
        )
    db_record = mock_repo.store_result.await_args.args[0]
    assert db_record["leakage_detected"] is False


@pytest.mark.asyncio
async def test_persist_qc_report_leakage_defaults_false_when_unspecified() -> None:
    """Backwards-compatible default: callers that don't pass a verdict persist
    a conservative False rather than erroring."""
    agent = DataPreparerAgent()
    mock_repo = AsyncMock()
    with patch(_AGENT_MOD, return_value=mock_repo):
        await agent._persist_qc_report(_minimal_qc_report(), "patient_journeys")
    db_record = mock_repo.store_result.await_args.args[0]
    assert db_record.get("leakage_detected") is False
