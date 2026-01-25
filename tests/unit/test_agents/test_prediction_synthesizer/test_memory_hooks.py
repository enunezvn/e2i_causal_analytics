"""
Tests for Prediction Synthesizer Memory Hooks.

Tests the memory integration for the Prediction Synthesizer agent including:
- Context retrieval from working/episodic memory
- Prediction caching in working memory
- Model performance tracking
- Graceful degradation when memory systems unavailable
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.prediction_synthesizer.memory_hooks import (
    PredictionSynthesizerMemoryHooks,
    PredictionMemoryContext,
    PredictionRecord,
    ModelPredictionRecord,
    contribute_to_memory,
    get_prediction_synthesizer_memory_hooks,
    reset_memory_hooks,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def memory_hooks():
    """Create a fresh memory hooks instance."""
    reset_memory_hooks()
    return PredictionSynthesizerMemoryHooks()


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def mock_working_memory(mock_redis):
    """Create mock working memory with Redis client."""
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    wm.get_client = AsyncMock(return_value=mock_redis)
    return wm


@pytest.fixture
def sample_prediction_result():
    """Sample prediction result for testing."""
    return {
        "ensemble_prediction": {
            "point_estimate": 0.72,
            "prediction_interval_lower": 0.58,
            "prediction_interval_upper": 0.86,
            "confidence": 0.85,
            "model_agreement": 0.91,
            "ensemble_method": "weighted",
        },
        "individual_predictions": [
            {
                "model_id": "xgboost_churn",
                "model_type": "xgboost",
                "prediction": 0.75,
                "confidence": 0.82,
                "latency_ms": 45,
            },
            {
                "model_id": "rf_churn",
                "model_type": "random_forest",
                "prediction": 0.69,
                "confidence": 0.78,
                "latency_ms": 52,
            },
        ],
        "models_succeeded": 2,
        "models_failed": 0,
        "total_latency_ms": 150,
        "status": "completed",
    }


@pytest.fixture
def sample_state():
    """Sample state for testing."""
    return {
        "entity_id": "hcp_12345",
        "entity_type": "hcp",
        "prediction_target": "churn",
        "time_horizon": "30d",
        "features": {"engagement_score": 0.7, "tenure_months": 24},
        "status": "completed",
        "query": "What is the churn probability for this HCP?",
    }


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


class TestMemoryHooksInitialization:
    """Tests for memory hooks initialization."""

    def test_init_creates_instance(self, memory_hooks):
        """Test that initialization creates a valid instance."""
        assert memory_hooks is not None
        assert memory_hooks._working_memory is None  # Lazy-loaded

    def test_singleton_returns_same_instance(self):
        """Test that singleton returns the same instance."""
        reset_memory_hooks()
        hooks1 = get_prediction_synthesizer_memory_hooks()
        hooks2 = get_prediction_synthesizer_memory_hooks()
        assert hooks1 is hooks2

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton."""
        hooks1 = get_prediction_synthesizer_memory_hooks()
        reset_memory_hooks()
        hooks2 = get_prediction_synthesizer_memory_hooks()
        assert hooks1 is not hooks2

    def test_ttl_values(self, memory_hooks):
        """Test that TTL values are correctly set."""
        assert memory_hooks.CACHE_TTL_SECONDS == 86400  # 24 hours
        assert memory_hooks.PREDICTION_CACHE_TTL == 3600  # 1 hour


# ============================================================================
# CONTEXT RETRIEVAL TESTS
# ============================================================================


class TestGetContext:
    """Tests for context retrieval."""

    @pytest.mark.asyncio
    async def test_get_context_returns_context_object(self, memory_hooks):
        """Test that get_context returns a PredictionMemoryContext."""
        context = await memory_hooks.get_context(
            session_id="test-session",
            entity_id="hcp_123",
            entity_type="hcp",
            prediction_target="churn",
        )

        assert isinstance(context, PredictionMemoryContext)
        assert context.session_id == "test-session"
        assert isinstance(context.retrieval_timestamp, datetime)

    @pytest.mark.asyncio
    async def test_get_context_with_time_horizon(self, memory_hooks):
        """Test context retrieval with time horizon."""
        context = await memory_hooks.get_context(
            session_id="test-session",
            entity_id="hcp_123",
            entity_type="hcp",
            prediction_target="conversion",
            time_horizon="90d",
        )

        assert context.session_id == "test-session"
        assert isinstance(context.working_memory, list)
        assert isinstance(context.episodic_context, list)
        assert isinstance(context.cached_predictions, list)
        assert isinstance(context.model_performance, dict)

    @pytest.mark.asyncio
    async def test_get_context_graceful_degradation(self, memory_hooks):
        """Test that context retrieval handles missing memory gracefully."""
        # Patch the lazy-loading import to fail, simulating unavailable memory
        with patch(
            "src.agents.prediction_synthesizer.memory_hooks.PredictionSynthesizerMemoryHooks.working_memory",
            new_callable=lambda: property(lambda self: None),
        ):
            context = await memory_hooks.get_context(
                session_id="test-session",
                entity_id="hcp_123",
                entity_type="hcp",
                prediction_target="churn",
            )

            # Should return empty context, not raise
            assert context.working_memory == []
            assert context.cached_predictions == []
            assert context.episodic_context == []
            assert context.model_performance == {}


class TestWorkingMemoryContext:
    """Tests for working memory context retrieval."""

    @pytest.mark.asyncio
    async def test_get_working_memory_context_success(
        self, memory_hooks, mock_working_memory
    ):
        """Test successful working memory context retrieval."""
        mock_working_memory.get_messages.return_value = [
            {"role": "user", "content": "Predict churn for HCP 123"},
            {"role": "assistant", "content": "The churn probability is 0.72"},
        ]

        memory_hooks._working_memory = mock_working_memory

        messages = await memory_hooks._get_working_memory_context("test-session")

        assert len(messages) == 2
        mock_working_memory.get_messages.assert_called_once_with(
            "test-session", limit=10
        )

    @pytest.mark.asyncio
    async def test_get_working_memory_context_unavailable(self, memory_hooks):
        """Test handling when working memory is unavailable."""
        memory_hooks._working_memory = None

        messages = await memory_hooks._get_working_memory_context("test-session")

        assert messages == []


class TestCachedPredictions:
    """Tests for cached prediction retrieval."""

    @pytest.mark.asyncio
    async def test_get_cached_predictions_found(
        self, memory_hooks, mock_working_memory, mock_redis
    ):
        """Test retrieving cached predictions."""
        cached_data = {"point_estimate": 0.72, "confidence": 0.85}
        mock_redis.get.return_value = json.dumps(cached_data)
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks._get_cached_predictions(
            entity_id="hcp_123",
            entity_type="hcp",
            prediction_target="churn",
        )

        assert len(result) == 1
        assert result[0]["point_estimate"] == 0.72
        mock_redis.get.assert_called_once_with(
            "prediction_synthesizer:entity:hcp:hcp_123:churn"
        )

    @pytest.mark.asyncio
    async def test_get_cached_predictions_not_found(
        self, memory_hooks, mock_working_memory, mock_redis
    ):
        """Test when no cached predictions exist."""
        mock_redis.get.return_value = None
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks._get_cached_predictions(
            entity_id="hcp_123",
            entity_type="hcp",
            prediction_target="churn",
        )

        assert result == []


class TestModelPerformance:
    """Tests for model performance history retrieval."""

    @pytest.mark.asyncio
    async def test_get_model_performance_found(
        self, memory_hooks, mock_working_memory, mock_redis
    ):
        """Test retrieving model performance history."""
        performance_data = {
            "xgboost_churn": {"accuracy": 0.85, "calibration_error": 0.05},
            "rf_churn": {"accuracy": 0.82, "calibration_error": 0.08},
        }
        mock_redis.get.return_value = json.dumps(performance_data)
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks._get_model_performance_history("churn")

        assert "xgboost_churn" in result
        assert result["xgboost_churn"]["accuracy"] == 0.85

    @pytest.mark.asyncio
    async def test_get_model_performance_empty(
        self, memory_hooks, mock_working_memory, mock_redis
    ):
        """Test when no model performance history exists."""
        mock_redis.get.return_value = None
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks._get_model_performance_history("churn")

        assert result == {}


# ============================================================================
# CACHING TESTS
# ============================================================================


class TestCachePrediction:
    """Tests for prediction caching."""

    @pytest.mark.asyncio
    async def test_cache_prediction_success(
        self, memory_hooks, mock_working_memory, mock_redis, sample_prediction_result
    ):
        """Test successful prediction caching."""
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks.cache_prediction(
            session_id="test-session",
            entity_id="hcp_123",
            entity_type="hcp",
            prediction_target="churn",
            prediction_result=sample_prediction_result,
        )

        assert result is True
        # Should be called twice: once for entity key, once for session key
        assert mock_redis.setex.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_prediction_entity_key_format(
        self, memory_hooks, mock_working_memory, mock_redis, sample_prediction_result
    ):
        """Test entity key format in caching."""
        memory_hooks._working_memory = mock_working_memory

        await memory_hooks.cache_prediction(
            session_id="test-session",
            entity_id="hcp_123",
            entity_type="hcp",
            prediction_target="churn",
            prediction_result=sample_prediction_result,
        )

        # Check first call (entity key)
        first_call = mock_redis.setex.call_args_list[0]
        assert first_call[0][0] == "prediction_synthesizer:entity:hcp:hcp_123:churn"
        assert first_call[0][1] == 3600  # PREDICTION_CACHE_TTL

    @pytest.mark.asyncio
    async def test_cache_prediction_no_working_memory(
        self, memory_hooks, sample_prediction_result
    ):
        """Test caching when working memory unavailable."""
        # Patch the property to return None, simulating unavailable memory
        with patch(
            "src.agents.prediction_synthesizer.memory_hooks.PredictionSynthesizerMemoryHooks.working_memory",
            new_callable=lambda: property(lambda self: None),
        ):
            result = await memory_hooks.cache_prediction(
                session_id="test-session",
                entity_id="hcp_123",
                entity_type="hcp",
                prediction_target="churn",
                prediction_result=sample_prediction_result,
            )

            assert result is False


class TestUpdateModelPerformance:
    """Tests for model performance update."""

    @pytest.mark.asyncio
    async def test_update_model_performance_new(
        self, memory_hooks, mock_working_memory, mock_redis
    ):
        """Test updating model performance for new model."""
        mock_redis.get.return_value = None  # No existing data
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks.update_model_performance(
            prediction_target="churn",
            model_id="xgboost_churn",
            accuracy=0.85,
            calibration_error=0.05,
        )

        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_model_performance_existing(
        self, memory_hooks, mock_working_memory, mock_redis
    ):
        """Test updating existing model performance."""
        existing_data = {"rf_churn": {"accuracy": 0.82, "calibration_error": 0.08}}
        mock_redis.get.return_value = json.dumps(existing_data)
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks.update_model_performance(
            prediction_target="churn",
            model_id="xgboost_churn",
            accuracy=0.85,
            calibration_error=0.05,
        )

        assert result is True
        # Verify the merged data was stored
        call_args = mock_redis.setex.call_args
        stored_data = json.loads(call_args[0][2])
        assert "xgboost_churn" in stored_data
        assert "rf_churn" in stored_data


# ============================================================================
# EPISODIC MEMORY TESTS
# ============================================================================


class TestStorePrediction:
    """Tests for prediction storage in episodic memory."""

    @pytest.mark.asyncio
    async def test_store_prediction_graceful_failure(
        self, memory_hooks, sample_prediction_result, sample_state
    ):
        """Test that storage handles missing episodic memory gracefully."""
        memory_id = await memory_hooks.store_prediction(
            session_id="test-session",
            result=sample_prediction_result,
            state=sample_state,
        )

        # Should return None, not raise
        assert memory_id is None


class TestGetCalibrationData:
    """Tests for calibration data retrieval."""

    @pytest.mark.asyncio
    async def test_get_calibration_data_graceful_failure(self, memory_hooks):
        """Test that calibration data retrieval handles errors gracefully."""
        results = await memory_hooks.get_calibration_data(
            prediction_target="churn",
            entity_type="hcp",
        )

        assert results == []


class TestGetSimilarPredictions:
    """Tests for similar prediction retrieval."""

    @pytest.mark.asyncio
    async def test_get_similar_predictions_graceful_failure(self, memory_hooks):
        """Test that similar prediction retrieval handles errors gracefully."""
        results = await memory_hooks.get_similar_predictions(
            entity_type="hcp",
            prediction_target="churn",
            features={"engagement_score": 0.7},
        )

        assert results == []


# ============================================================================
# CONTRIBUTE TO MEMORY TESTS
# ============================================================================


class TestContributeToMemory:
    """Tests for the contribute_to_memory function."""

    @pytest.mark.asyncio
    async def test_contribute_skips_failed_prediction(self, sample_prediction_result):
        """Test that failed predictions are skipped."""
        state = {"status": "failed"}

        counts = await contribute_to_memory(
            result=sample_prediction_result,
            state=state,
        )

        assert counts["episodic_stored"] == 0
        assert counts["working_cached"] == 0

    @pytest.mark.asyncio
    async def test_contribute_with_successful_prediction(
        self, sample_prediction_result, sample_state
    ):
        """Test contribution with successful prediction."""
        counts = await contribute_to_memory(
            result=sample_prediction_result,
            state=sample_state,
        )

        # Without real memory systems, counts should be 0
        # but the function should complete without error
        assert isinstance(counts, dict)
        assert "episodic_stored" in counts
        assert "working_cached" in counts

    @pytest.mark.asyncio
    async def test_contribute_with_custom_hooks(
        self, sample_prediction_result, sample_state, mock_working_memory, mock_redis
    ):
        """Test contribution with custom memory hooks."""
        hooks = PredictionSynthesizerMemoryHooks()
        hooks._working_memory = mock_working_memory

        counts = await contribute_to_memory(
            result=sample_prediction_result,
            state=sample_state,
            memory_hooks=hooks,
            session_id="custom-session",
        )

        # Should have cached in working memory
        assert counts["working_cached"] == 1

    @pytest.mark.asyncio
    async def test_contribute_generates_session_id(
        self, sample_prediction_result, sample_state
    ):
        """Test that session ID is generated if not provided."""
        counts = await contribute_to_memory(
            result=sample_prediction_result,
            state=sample_state,
            session_id=None,
        )

        # Should complete without error
        assert isinstance(counts, dict)

    @pytest.mark.asyncio
    async def test_contribute_skips_caching_without_entity_info(
        self, sample_prediction_result
    ):
        """Test that caching is skipped without entity info."""
        state = {
            "status": "completed",
            # Missing entity_id, entity_type, prediction_target
        }

        counts = await contribute_to_memory(
            result=sample_prediction_result,
            state=state,
        )

        # Caching should be skipped, but function should complete
        assert counts["working_cached"] == 0


# ============================================================================
# DATA STRUCTURE TESTS
# ============================================================================


class TestDataStructures:
    """Tests for data structure validation."""

    def test_prediction_memory_context_creation(self):
        """Test PredictionMemoryContext creation."""
        context = PredictionMemoryContext(
            session_id="test-session",
            working_memory=[{"role": "user", "content": "test"}],
            episodic_context=[{"id": "mem-1"}],
            cached_predictions=[{"point_estimate": 0.72}],
            model_performance={"xgboost": {"accuracy": 0.85}},
        )

        assert context.session_id == "test-session"
        assert len(context.working_memory) == 1
        assert len(context.episodic_context) == 1
        assert len(context.cached_predictions) == 1
        assert "xgboost" in context.model_performance
        assert isinstance(context.retrieval_timestamp, datetime)

    def test_prediction_record_creation(self):
        """Test PredictionRecord creation."""
        record = PredictionRecord(
            session_id="session-123",
            entity_id="hcp_456",
            entity_type="hcp",
            prediction_target="churn",
            point_estimate=0.72,
            confidence=0.85,
            model_agreement=0.91,
            ensemble_method="weighted",
            models_succeeded=3,
            models_failed=0,
            time_horizon="30d",
        )

        assert record.session_id == "session-123"
        assert record.point_estimate == 0.72
        assert record.ensemble_method == "weighted"
        assert isinstance(record.timestamp, datetime)

    def test_model_prediction_record_creation(self):
        """Test ModelPredictionRecord creation."""
        record = ModelPredictionRecord(
            model_id="xgboost_churn",
            model_type="xgboost",
            prediction=0.75,
            confidence=0.82,
            latency_ms=45,
            features_used=["engagement_score", "tenure_months"],
        )

        assert record.model_id == "xgboost_churn"
        assert record.prediction == 0.75
        assert len(record.features_used) == 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestMemoryHooksIntegration:
    """Integration tests for memory hooks with the agent."""

    @pytest.mark.asyncio
    async def test_full_workflow_without_memory(self):
        """Test full workflow when memory systems unavailable."""
        hooks = PredictionSynthesizerMemoryHooks()

        # Patch the property to return None, simulating unavailable memory
        with patch(
            "src.agents.prediction_synthesizer.memory_hooks.PredictionSynthesizerMemoryHooks.working_memory",
            new_callable=lambda: property(lambda self: None),
        ):
            # Get context
            context = await hooks.get_context(
                session_id="integration-test",
                entity_id="hcp_123",
                entity_type="hcp",
                prediction_target="churn",
            )

            assert context.session_id == "integration-test"

            # Cache prediction (should fail gracefully)
            cached = await hooks.cache_prediction(
                session_id="integration-test",
                entity_id="hcp_123",
                entity_type="hcp",
                prediction_target="churn",
                prediction_result={"point_estimate": 0.72},
            )

            assert cached is False  # No working memory available

            # Store prediction (should fail gracefully)
            memory_id = await hooks.store_prediction(
                session_id="integration-test",
                result={"ensemble_prediction": {"point_estimate": 0.72}},
                state={"status": "completed", "entity_id": "hcp_123"},
            )

            assert memory_id is None

    @pytest.mark.asyncio
    async def test_cache_key_patterns(self, memory_hooks, mock_working_memory, mock_redis):
        """Test that cache keys follow expected patterns."""
        memory_hooks._working_memory = mock_working_memory

        # Cache a prediction
        await memory_hooks.cache_prediction(
            session_id="test-session",
            entity_id="territory_northeast",
            entity_type="territory",
            prediction_target="conversion",
            prediction_result={"point_estimate": 0.65},
        )

        # Verify key patterns
        calls = mock_redis.setex.call_args_list

        # First call should be entity key
        entity_key = calls[0][0][0]
        assert entity_key == "prediction_synthesizer:entity:territory:territory_northeast:conversion"

        # Second call should be session key
        session_key = calls[1][0][0]
        assert session_key == "prediction_synthesizer:session:test-session"
