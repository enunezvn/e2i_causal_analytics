"""P10 — validation outcome persistence durability + zero-ATE (H10, H11).

- H10: effect_size==0.0 was dropped to NULL by Python truthiness, corrupting the
  Feedback-Learner signal (a placebo refutation / genuine null finding is ~0).
- H11: a Supabase failure silently fell back to an EPHEMERAL in-memory store and
  returned a success id, indistinguishable from a durable write.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.causal_engine.validation_outcome import ValidationOutcome, ValidationOutcomeType
from src.causal_engine.validation_outcome_store import (
    InMemoryValidationOutcomeStore,
    StoreResult,
    SupabaseValidationOutcomeStore,
    log_validation_outcome_with_status,
    reset_validation_outcome_store,
)


def _outcome(effect_size, outcome_id="o-1"):
    return ValidationOutcome(
        outcome_id=outcome_id,
        estimate_id="e-1",
        outcome_type=ValidationOutcomeType.PASSED,
        treatment_variable="t",
        outcome_variable="y",
        brand="Kisqali",
        sample_size=500,
        effect_size=effect_size,
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


def _mock_client(*, insert_raises=False, returns_data=True):
    client = MagicMock()
    table = MagicMock()
    exec_mock = table.insert.return_value.execute
    if insert_raises:
        exec_mock.side_effect = Exception("RLS denied")
    else:
        result = MagicMock()
        result.data = [{"outcome_id": "o-1"}] if returns_data else []
        exec_mock.return_value = result
    client.table.return_value = table
    return client


class TestZeroEffectSizeNotDropped:
    def test_effect_size_zero_round_trips(self):
        store = SupabaseValidationOutcomeStore()
        row = store._outcome_to_row(_outcome(0.0))
        assert row["effect_size"] == 0.0, "a 0.0 ATE must persist as 0.0, not NULL"
        assert row["effect_size"] is not None

    def test_effect_size_none_stays_none(self):
        store = SupabaseValidationOutcomeStore()
        row = store._outcome_to_row(_outcome(None))
        assert row["effect_size"] is None

    def test_effect_size_negative_zero_round_trips(self):
        store = SupabaseValidationOutcomeStore()
        row = store._outcome_to_row(_outcome(-0.0))
        assert row["effect_size"] == 0.0


class TestDurabilitySignal:
    @pytest.mark.asyncio
    async def test_durable_write_reports_persisted(self):
        store = SupabaseValidationOutcomeStore()
        store._get_client = lambda: _mock_client(returns_data=True)
        result = await store.store_with_status(_outcome(0.5))
        assert result.persisted is True
        assert result.degraded is False
        assert result.backend == "supabase"

    @pytest.mark.asyncio
    async def test_missing_client_reports_degraded(self):
        store = SupabaseValidationOutcomeStore()
        store._get_client = lambda: None
        result = await store.store_with_status(_outcome(0.5))
        assert result.persisted is False, "a non-durable in-memory write must NOT report success"
        assert result.degraded is True
        assert result.backend == "memory_fallback"

    @pytest.mark.asyncio
    async def test_insert_failure_reports_degraded(self):
        store = SupabaseValidationOutcomeStore()
        store._get_client = lambda: _mock_client(insert_raises=True)
        result = await store.store_with_status(_outcome(0.5))
        assert result.persisted is False
        assert result.degraded is True

    @pytest.mark.asyncio
    async def test_insert_returns_no_data_reports_degraded(self):
        store = SupabaseValidationOutcomeStore()
        store._get_client = lambda: _mock_client(returns_data=False)
        result = await store.store_with_status(_outcome(0.5))
        assert result.persisted is False
        assert result.degraded is True

    @pytest.mark.asyncio
    async def test_store_str_is_backward_compatible(self):
        store = SupabaseValidationOutcomeStore()
        store._get_client = lambda: _mock_client(returns_data=True)
        oid = await store.store(_outcome(0.5, outcome_id="o-bc"))
        assert oid == "o-bc"

    @pytest.mark.asyncio
    async def test_in_memory_store_reports_persisted(self):
        store = InMemoryValidationOutcomeStore()
        result = await store.store_with_status(_outcome(0.5))
        assert result.persisted is True
        assert result.degraded is False
        assert result.backend == "memory"


class TestLogValidationOutcomeWithStatus:
    """H11: the convenience fn must surface the durable-vs-degraded signal."""

    @pytest.mark.asyncio
    async def test_returns_store_result_persisted_for_durable_write(self):
        reset_validation_outcome_store()
        store = InMemoryValidationOutcomeStore()
        with patch(
            "src.causal_engine.validation_outcome_store.get_validation_outcome_store",
            return_value=store,
        ):
            result = await log_validation_outcome_with_status(_outcome(0.5, outcome_id="o-dur"))
        assert isinstance(result, StoreResult)
        assert result.outcome_id == "o-dur"
        assert result.persisted is True
        assert result.degraded is False
        reset_validation_outcome_store()

    @pytest.mark.asyncio
    async def test_returns_degraded_when_supabase_falls_back(self):
        reset_validation_outcome_store()
        store = SupabaseValidationOutcomeStore()
        store._get_client = lambda: None  # force the ephemeral in-memory fallback
        with patch(
            "src.causal_engine.validation_outcome_store.get_validation_outcome_store",
            return_value=store,
        ):
            result = await log_validation_outcome_with_status(_outcome(0.5, outcome_id="o-deg"))
        assert result.persisted is False, "a non-durable fallback must NOT report success"
        assert result.degraded is True
        assert result.backend == "memory_fallback"
        reset_validation_outcome_store()
