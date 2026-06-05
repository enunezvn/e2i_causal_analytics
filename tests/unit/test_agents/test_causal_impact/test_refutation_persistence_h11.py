"""H11 — refutation node must not report an ephemeral (degraded) validation-outcome
write as a durable success.

The persistence-logging branch is extracted into
RefutationNode._log_validation_outcome_signal so it is testable without driving the
slow real-refuter path. On a DEGRADED write it must NOT emit the confident
"Logged validation outcome" INFO line, must log a WARNING, and must return None so a
non-durable signal is not propagated as a persisted id into agent state.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.causal_engine import StoreResult
from src.causal_engine.validation_outcome import ValidationOutcome, ValidationOutcomeType


def _outcome() -> ValidationOutcome:
    return ValidationOutcome(
        outcome_id="vo-h11",
        estimate_id="e-1",
        outcome_type=ValidationOutcomeType.PASSED,
        treatment_variable="t",
        outcome_variable="y",
        brand="Kisqali",
        sample_size=500,
        effect_size=0.5,
        gate_decision="proceed",
        confidence_score=0.8,
        tests_passed=4,
        tests_failed=0,
        tests_total=4,
        failure_patterns=[],
        raw_suite={},
        agent_context={},
        dag_hash="d-1",
        timestamp="2026-06-05T00:00:00Z",
    )


class TestPersistenceSignalH11:
    @pytest.mark.asyncio
    async def test_durable_write_returns_id_and_logs_info(self, caplog):
        node = RefutationNode()
        durable = StoreResult(
            outcome_id="vo-h11", persisted=True, degraded=False, backend="supabase"
        )
        with patch(
            "src.agents.causal_impact.nodes.refutation.log_validation_outcome_with_status",
            new=AsyncMock(return_value=durable),
        ):
            with caplog.at_level(logging.INFO):
                outcome_id = await node._log_validation_outcome_signal(_outcome())
        assert outcome_id == "vo-h11"
        assert any("Logged validation outcome" in r.getMessage() for r in caplog.records), (
            "a durable write should keep the confident INFO line"
        )

    @pytest.mark.asyncio
    async def test_degraded_write_returns_none_and_warns(self, caplog):
        node = RefutationNode()
        degraded = StoreResult(
            outcome_id="vo-h11", persisted=False, degraded=True, backend="memory_fallback"
        )
        with patch(
            "src.agents.causal_impact.nodes.refutation.log_validation_outcome_with_status",
            new=AsyncMock(return_value=degraded),
        ):
            with caplog.at_level(logging.INFO):
                outcome_id = await node._log_validation_outcome_signal(_outcome())
        # fail-closed: a non-durable signal must NOT propagate as a persisted id
        assert outcome_id is None
        # must NOT claim success
        assert not any("Logged validation outcome" in r.getMessage() for r in caplog.records), (
            "a degraded write must NOT emit the confident success line"
        )
        # must warn (alertable)
        assert any(
            r.levelno == logging.WARNING and "degraded" in r.getMessage().lower()
            for r in caplog.records
        ), "a degraded write must log a WARNING"

    @pytest.mark.asyncio
    async def test_exception_returns_none_and_warns(self, caplog):
        node = RefutationNode()
        with patch(
            "src.agents.causal_impact.nodes.refutation.log_validation_outcome_with_status",
            new=AsyncMock(side_effect=RuntimeError("store boom")),
        ):
            with caplog.at_level(logging.WARNING):
                outcome_id = await node._log_validation_outcome_signal(_outcome())
        assert outcome_id is None
        assert any(
            r.levelno == logging.WARNING and "Feedback Learner" in r.getMessage()
            for r in caplog.records
        )
