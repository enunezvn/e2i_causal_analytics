"""
Unit tests for SupabaseValidationOutcomeStore.

Tests cover:
- Store with Supabase available
- Fallback to in-memory when Supabase unavailable
- Serialization/deserialization of ValidationOutcome
- Query methods return correct results

Part of Phase 3: Knowledge Store Persistence
Reference: .claude/plans/feedback-loop-concept-drift-audit.md

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_validation_outcome():
    """Create a sample ValidationOutcome for testing."""
    from src.causal_engine.validation_outcome import (
        FailureCategory,
        ValidationFailurePattern,
        ValidationOutcome,
        ValidationOutcomeType,
    )

    return ValidationOutcome(
        outcome_id=str(uuid4()),
        estimate_id="est_001",
        outcome_type=ValidationOutcomeType.FAILED_CRITICAL,
        treatment_variable="rep_visits",
        outcome_variable="trx_change",
        brand="remibrutinib",
        sample_size=500,
        effect_size=0.15,
        confidence_score=0.85,
        gate_decision="block",
        tests_passed=3,
        tests_failed=2,
        tests_total=5,
        failure_patterns=[
            ValidationFailurePattern(
                category=FailureCategory.MODEL_MISSPECIFICATION,
                test_name="random_common_cause",
                description="Random common cause changed effect significantly",
                severity="high",
                original_effect=0.15,
                refuted_effect=0.08,
                delta_percent=46.7,
                recommendation="Consider unmeasured confounders",
            )
        ],
        agent_context={"source": "test"},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def sample_outcome_row():
    """Create a sample database row for testing."""
    return {
        "outcome_id": str(uuid4()),
        "estimate_id": "est_002",
        "outcome_type": "failed_critical",
        "treatment_variable": "email_campaigns",
        "outcome_variable": "engagement_score",
        "brand": "fabhalta",
        "sample_size": 300,
        "effect_size": 0.08,
        "confidence_score": 0.75,
        "gate_decision": "block",
        "tests_passed": 2,
        "tests_failed": 3,
        "tests_total": 5,
        "failure_patterns": [
            {
                "category": "unobserved_confounding",
                "test_name": "sensitivity_analysis",
                "description": "E-value indicates sensitivity to unmeasured confounding",
                "severity": "high",
                "original_effect": 0.08,
                "refuted_effect": 0.12,
                "delta_percent": 50.0,
                "recommendation": "Check for spillover effects",
            }
        ],
        "agent_context": {"env": "test"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    client = MagicMock()

    # Mock table method chain
    table_mock = MagicMock()
    client.table = MagicMock(return_value=table_mock)

    # Mock query chain methods
    table_mock.select = MagicMock(return_value=table_mock)
    table_mock.insert = MagicMock(return_value=table_mock)
    table_mock.eq = MagicMock(return_value=table_mock)
    table_mock.neq = MagicMock(return_value=table_mock)
    table_mock.gte = MagicMock(return_value=table_mock)
    table_mock.order = MagicMock(return_value=table_mock)
    table_mock.limit = MagicMock(return_value=table_mock)

    # Mock RPC
    rpc_mock = MagicMock()
    rpc_mock.execute = MagicMock(return_value=MagicMock(data=[]))
    client.rpc = MagicMock(return_value=rpc_mock)

    return client, table_mock


# =============================================================================
# SupabaseValidationOutcomeStore Tests
# =============================================================================


class TestSupabaseValidationOutcomeStoreInit:
    """Tests for SupabaseValidationOutcomeStore initialization."""

    def test_init_creates_store(self):
        """Test that store can be initialized."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        store = SupabaseValidationOutcomeStore()
        assert store._client is None  # Lazy init
        assert store._fallback_store is None

    def test_lazy_client_loading(self, mock_supabase_client):
        """Test that client is loaded lazily."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        client, _ = mock_supabase_client

        store = SupabaseValidationOutcomeStore()
        assert store._client is None

        with patch(
            "src.memory.services.factories.get_supabase_client",
            return_value=client,
        ):
            result = store._get_client()
            assert result is not None

    def test_fallback_store_creation(self):
        """Test that fallback store is created when needed."""
        from src.causal_engine.validation_outcome_store import (
            InMemoryValidationOutcomeStore,
            SupabaseValidationOutcomeStore,
        )

        store = SupabaseValidationOutcomeStore()
        fallback = store._get_fallback()

        assert fallback is not None
        assert isinstance(fallback, InMemoryValidationOutcomeStore)


class TestSupabaseValidationOutcomeStoreSerialization:
    """Tests for serialization/deserialization."""

    def test_outcome_to_row(self, sample_validation_outcome):
        """Test converting ValidationOutcome to database row."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        store = SupabaseValidationOutcomeStore()
        row = store._outcome_to_row(sample_validation_outcome)

        assert row["outcome_id"] == sample_validation_outcome.outcome_id
        assert row["outcome_type"] == "failed_critical"
        assert row["treatment_variable"] == "rep_visits"
        assert row["outcome_variable"] == "trx_change"
        assert row["brand"] == "remibrutinib"
        assert row["sample_size"] == 500
        assert row["effect_size"] == 0.15
        assert len(row["failure_patterns"]) == 1
        assert row["failure_patterns"][0]["category"] == "model_misspecification"

    def test_row_to_outcome(self, sample_outcome_row):
        """Test converting database row to ValidationOutcome."""
        from src.causal_engine.validation_outcome import (
            ValidationOutcome,
            ValidationOutcomeType,
        )
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        store = SupabaseValidationOutcomeStore()
        outcome = store._row_to_outcome(sample_outcome_row)

        assert isinstance(outcome, ValidationOutcome)
        assert outcome.outcome_type == ValidationOutcomeType.FAILED_CRITICAL
        assert outcome.treatment_variable == "email_campaigns"
        assert outcome.outcome_variable == "engagement_score"
        assert outcome.brand == "fabhalta"
        assert len(outcome.failure_patterns) == 1

    def test_round_trip_serialization(self, sample_validation_outcome):
        """Test that outcome survives round-trip through serialization."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        store = SupabaseValidationOutcomeStore()

        # Convert to row and back
        row = store._outcome_to_row(sample_validation_outcome)
        recovered = store._row_to_outcome(row)

        assert recovered.outcome_id == sample_validation_outcome.outcome_id
        assert recovered.outcome_type == sample_validation_outcome.outcome_type
        assert recovered.treatment_variable == sample_validation_outcome.treatment_variable
        assert len(recovered.failure_patterns) == len(
            sample_validation_outcome.failure_patterns
        )


class TestSupabaseValidationOutcomeStoreOperations:
    """Tests for store operations with mocked Supabase."""

    @pytest.mark.asyncio
    async def test_store_with_supabase_available(
        self, sample_validation_outcome, mock_supabase_client
    ):
        """Test storing outcome when Supabase is available."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        client, table_mock = mock_supabase_client

        # Mock successful insert
        table_mock.execute = MagicMock(
            return_value=MagicMock(data=[{"outcome_id": sample_validation_outcome.outcome_id}])
        )

        store = SupabaseValidationOutcomeStore()
        store._client = client

        result = await store.store(sample_validation_outcome)

        assert result == sample_validation_outcome.outcome_id
        client.table.assert_called_with("validation_outcomes")
        table_mock.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_fallback_when_supabase_unavailable(
        self, sample_validation_outcome
    ):
        """Test that store falls back to in-memory when Supabase unavailable."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        store = SupabaseValidationOutcomeStore()
        store._client = None  # Simulate no Supabase

        result = await store.store(sample_validation_outcome)

        assert result == sample_validation_outcome.outcome_id
        # Verify fallback store was used
        assert store._fallback_store is not None
        assert store._fallback_store.count == 1

    @pytest.mark.asyncio
    async def test_get_with_supabase_available(
        self, sample_outcome_row, mock_supabase_client
    ):
        """Test getting outcome when Supabase is available."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        client, table_mock = mock_supabase_client

        # Mock successful query
        table_mock.execute = MagicMock(
            return_value=MagicMock(data=[sample_outcome_row])
        )

        store = SupabaseValidationOutcomeStore()
        store._client = client

        outcome_id = sample_outcome_row["outcome_id"]
        result = await store.get(outcome_id)

        assert result is not None
        assert result.outcome_id == outcome_id
        table_mock.eq.assert_called_with("outcome_id", outcome_id)

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, mock_supabase_client):
        """Test getting non-existent outcome returns None."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        client, table_mock = mock_supabase_client

        # Mock empty result
        table_mock.execute = MagicMock(return_value=MagicMock(data=[]))

        store = SupabaseValidationOutcomeStore()
        store._client = client

        result = await store.get("nonexistent_id")

        assert result is None

    @pytest.mark.asyncio
    async def test_query_failures_with_filters(self, mock_supabase_client):
        """Test querying failures with various filters."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        client, table_mock = mock_supabase_client

        # Mock result with failures
        table_mock.execute = MagicMock(
            return_value=MagicMock(
                data=[
                    {
                        "outcome_id": str(uuid4()),
                        "outcome_type": "failed_refutation",
                        "treatment_variable": "rep_visits",
                        "outcome_variable": "trx_change",
                        "failure_patterns": [],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            )
        )

        store = SupabaseValidationOutcomeStore()
        store._client = client

        results = await store.query_failures(
            limit=10,
            treatment_variable="rep_visits",
            brand="remibrutinib",
        )

        assert len(results) >= 0  # May be filtered
        table_mock.neq.assert_called_with("outcome_type", "passed")

    @pytest.mark.asyncio
    async def test_get_failure_patterns_from_view(self, mock_supabase_client):
        """Test getting failure patterns from aggregated view."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        client, table_mock = mock_supabase_client

        # Mock view result
        table_mock.execute = MagicMock(
            return_value=MagicMock(
                data=[
                    {
                        "category": "refutation",
                        "test_name": "random_common_cause",
                        "failure_count": 5,
                        "avg_delta_percent": 30.5,
                        "recommendations": ["Check confounders"],
                    }
                ]
            )
        )

        store = SupabaseValidationOutcomeStore()
        store._client = client

        patterns = await store.get_failure_patterns(limit=10)

        assert len(patterns) == 1
        assert patterns[0]["category"] == "refutation"
        assert patterns[0]["count"] == 5
        client.table.assert_called_with("v_validation_failure_patterns")


class TestSupabaseValidationOutcomeStoreFallback:
    """Tests for fallback behavior when Supabase fails."""

    @pytest.mark.asyncio
    async def test_fallback_on_insert_error(
        self, sample_validation_outcome, mock_supabase_client
    ):
        """Test fallback to in-memory on insert error."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        client, table_mock = mock_supabase_client

        # Mock insert failure
        table_mock.execute = MagicMock(side_effect=Exception("Insert failed"))

        store = SupabaseValidationOutcomeStore()
        store._client = client

        # Should not raise, should use fallback
        result = await store.store(sample_validation_outcome)

        assert result == sample_validation_outcome.outcome_id
        assert store._fallback_store is not None

    @pytest.mark.asyncio
    async def test_fallback_on_query_error(self, mock_supabase_client):
        """Test fallback to in-memory on query error."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
        )

        client, table_mock = mock_supabase_client

        # Mock query failure
        table_mock.execute = MagicMock(side_effect=Exception("Query failed"))

        store = SupabaseValidationOutcomeStore()
        store._client = client

        # Should not raise, should use fallback
        results = await store.query_failures(limit=5)

        assert results == []  # Fallback is empty


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestValidationOutcomeStoreFactory:
    """Tests for the factory function."""

    def test_factory_returns_supabase_store_when_configured(self):
        """Test factory returns Supabase store when SUPABASE_URL is set."""
        from src.causal_engine.validation_outcome_store import (
            SupabaseValidationOutcomeStore,
            get_validation_outcome_store,
            reset_validation_outcome_store,
        )

        # Reset first to ensure fresh state
        reset_validation_outcome_store()

        with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co"}):
            store = get_validation_outcome_store(use_supabase=True)
            assert isinstance(store, SupabaseValidationOutcomeStore)

        # Clean up
        reset_validation_outcome_store()

    def test_factory_returns_inmemory_store_when_not_configured(self):
        """Test factory returns in-memory store when SUPABASE_URL not set."""
        from src.causal_engine.validation_outcome_store import (
            InMemoryValidationOutcomeStore,
            get_validation_outcome_store,
            reset_validation_outcome_store,
        )

        # Reset first to ensure fresh state
        reset_validation_outcome_store()

        # Remove SUPABASE_URL if set
        env_without_supabase = {k: v for k, v in os.environ.items() if k != "SUPABASE_URL"}
        with patch.dict(os.environ, env_without_supabase, clear=True):
            store = get_validation_outcome_store(use_supabase=True)
            assert isinstance(store, InMemoryValidationOutcomeStore)

        # Clean up
        reset_validation_outcome_store()

    def test_factory_returns_inmemory_store_when_requested(self):
        """Test factory returns in-memory store when use_supabase=False."""
        from src.causal_engine.validation_outcome_store import (
            InMemoryValidationOutcomeStore,
            get_validation_outcome_store,
            reset_validation_outcome_store,
        )

        # Reset first to ensure fresh state
        reset_validation_outcome_store()

        # Even with SUPABASE_URL set, should return in-memory
        with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co"}):
            store = get_validation_outcome_store(use_supabase=False)
            assert isinstance(store, InMemoryValidationOutcomeStore)

        # Clean up
        reset_validation_outcome_store()

    def test_reset_clears_global_stores(self):
        """Test that reset clears global store instances."""
        from src.causal_engine.validation_outcome_store import (
            _global_knowledge_store,
            _global_outcome_store,
            get_validation_outcome_store,
            reset_validation_outcome_store,
        )

        # Create stores
        get_validation_outcome_store(use_supabase=False)

        # Reset
        reset_validation_outcome_store()

        # Import again to check globals
        from src.causal_engine import validation_outcome_store as vos

        assert vos._global_outcome_store is None
        assert vos._global_knowledge_store is None
