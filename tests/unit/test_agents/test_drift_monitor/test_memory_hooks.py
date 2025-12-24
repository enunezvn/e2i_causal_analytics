"""Tests for Drift Monitor Memory Hooks.

Tests the 4-Memory Architecture integration for drift detection:
- Working Memory: Caching drift results
- Episodic Memory: Storing detection history
- Semantic Memory: Storing drift patterns
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.drift_monitor.memory_hooks import (
    DriftDetectionContext,
    DriftDetectionRecord,
    DriftPatternRecord,
    DriftMonitorMemoryHooks,
    contribute_to_memory,
    get_drift_monitor_memory_hooks,
    reset_memory_hooks,
)


# ============================================================================
# Test Data Structures
# ============================================================================


class TestDriftDetectionContext:
    """Test DriftDetectionContext data structure."""

    def test_create_context(self):
        """Test context creation with defaults."""
        context = DriftDetectionContext(session_id="test-session-123")

        assert context.session_id == "test-session-123"
        assert context.working_memory == []
        assert context.episodic_context == []
        assert context.semantic_context == {}
        assert isinstance(context.retrieval_timestamp, datetime)

    def test_context_with_data(self):
        """Test context creation with populated data."""
        context = DriftDetectionContext(
            session_id="test-session",
            working_memory=[{"cached": True}],
            episodic_context=[{"prior_detection": "high_drift"}],
            semantic_context={"patterns": [{"feature": "f1", "drift_type": "data"}]},
        )

        assert len(context.working_memory) == 1
        assert len(context.episodic_context) == 1
        assert len(context.semantic_context.get("patterns", [])) == 1

    def test_context_to_dict(self):
        """Test context serialization."""
        context = DriftDetectionContext(session_id="test-session")
        result = context.to_dict()

        assert result["session_id"] == "test-session"
        assert "retrieval_timestamp" in result
        assert isinstance(result["retrieval_timestamp"], str)


class TestDriftDetectionRecord:
    """Test DriftDetectionRecord data structure."""

    def test_create_record(self):
        """Test record creation."""
        record = DriftDetectionRecord(
            record_id="rec-123",
            session_id="sess-456",
            timestamp=datetime.now(timezone.utc),
            query="Check for drift",
            model_id="model_v1",
            features_monitored=["f1", "f2", "f3"],
            time_window="7d",
            brand="Remibrutinib",
            overall_drift_score=0.45,
            features_with_drift=["f1", "f3"],
            data_drift_count=2,
            model_drift_count=0,
            concept_drift_count=1,
            alert_count=1,
            max_severity="medium",
            drift_summary="Moderate drift detected in 2 features",
            recommended_actions=["Retrain model", "Investigate f1"],
            detection_latency_ms=250,
            warnings=[],
        )

        assert record.record_id == "rec-123"
        assert record.overall_drift_score == 0.45
        assert len(record.features_with_drift) == 2
        assert record.max_severity == "medium"

    def test_record_to_dict(self):
        """Test record serialization."""
        record = DriftDetectionRecord(
            record_id="rec-123",
            session_id="sess-456",
            timestamp=datetime.now(timezone.utc),
            query="Check drift",
            model_id=None,
            features_monitored=["f1"],
            time_window="7d",
            brand=None,
            overall_drift_score=0.0,
            features_with_drift=[],
            data_drift_count=0,
            model_drift_count=0,
            concept_drift_count=0,
            alert_count=0,
            max_severity="none",
            drift_summary="No drift detected",
            recommended_actions=[],
            detection_latency_ms=100,
            warnings=[],
        )

        result = record.to_dict()

        assert result["record_id"] == "rec-123"
        assert result["session_id"] == "sess-456"
        assert isinstance(result["timestamp"], str)
        assert result["overall_drift_score"] == 0.0


class TestDriftPatternRecord:
    """Test DriftPatternRecord data structure."""

    def test_create_pattern(self):
        """Test pattern creation."""
        pattern = DriftPatternRecord(
            pattern_id="pat-123",
            timestamp=datetime.now(timezone.utc),
            feature_name="conversion_rate",
            drift_type="data",
            model_id="model_v1",
            severity="high",
            test_statistic=2.45,
            p_value=0.01,
            psi_score=0.32,
            baseline_period="2024-01-01/2024-01-07",
            current_period="2024-01-08/2024-01-14",
            brand="Remibrutinib",
            co_drifting_features=["engagement_score", "reach"],
            related_models=["model_v1", "model_v2"],
        )

        assert pattern.pattern_id == "pat-123"
        assert pattern.drift_type == "data"
        assert pattern.severity == "high"
        assert len(pattern.co_drifting_features) == 2

    def test_pattern_to_dict(self):
        """Test pattern serialization."""
        pattern = DriftPatternRecord(
            pattern_id="pat-123",
            timestamp=datetime.now(timezone.utc),
            feature_name="f1",
            drift_type="model",
            model_id=None,
            severity="low",
            test_statistic=1.2,
            p_value=0.08,
            psi_score=None,
            baseline_period="baseline",
            current_period="current",
            brand=None,
            co_drifting_features=[],
            related_models=[],
        )

        result = pattern.to_dict()

        assert result["pattern_id"] == "pat-123"
        assert result["drift_type"] == "model"
        assert isinstance(result["timestamp"], str)


# ============================================================================
# Test Memory Hooks Class
# ============================================================================


class TestDriftMonitorMemoryHooks:
    """Test DriftMonitorMemoryHooks class."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_hooks()

    def test_create_hooks(self):
        """Test hooks creation."""
        hooks = DriftMonitorMemoryHooks()

        assert hooks is not None
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None
        assert hooks._supabase_client is None

    def test_cache_ttl_constant(self):
        """Test cache TTL is set correctly."""
        assert DriftMonitorMemoryHooks.CACHE_TTL_SECONDS == 86400

    @pytest.mark.asyncio
    async def test_get_context_no_memory_clients(self):
        """Test get_context when no memory clients available."""
        hooks = DriftMonitorMemoryHooks()

        context = await hooks.get_context(
            session_id="test-session",
            query="Check drift",
            features=["f1", "f2"],
        )

        assert context.session_id == "test-session"
        assert context.working_memory == []
        assert context.episodic_context == []
        assert context.semantic_context == {"patterns": [], "feature_clusters": [], "drift_history": {}}

    @pytest.mark.asyncio
    async def test_cache_drift_result_no_client(self):
        """Test caching returns False when no client."""
        hooks = DriftMonitorMemoryHooks()

        result = await hooks.cache_drift_result(
            session_id="test-session",
            result={"overall_drift_score": 0.5},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_cached_drift_result_no_client(self):
        """Test cached retrieval returns None when no client."""
        hooks = DriftMonitorMemoryHooks()

        result = await hooks.get_cached_drift_result("test-session")

        assert result is None

    @pytest.mark.asyncio
    async def test_store_drift_detection_no_client(self):
        """Test storage returns None when no client."""
        hooks = DriftMonitorMemoryHooks()

        result = await hooks.store_drift_detection(
            session_id="test-session",
            result={"overall_drift_score": 0.5, "features_with_drift": []},
            state={"query": "test", "features_to_monitor": ["f1"]},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_store_drift_pattern_no_client(self):
        """Test pattern storage returns False when no client."""
        hooks = DriftMonitorMemoryHooks()

        result = await hooks.store_drift_pattern(
            feature="f1",
            drift_type="data",
            severity="high",
            result={"data_drift_results": []},
            state={"model_id": None},
        )

        assert result is False


class TestDriftMonitorMemoryHooksWithMocks:
    """Test memory hooks with mocked clients."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_hooks()

    @pytest.mark.asyncio
    async def test_cache_drift_result_success(self):
        """Test successful caching with mocked Redis."""
        hooks = DriftMonitorMemoryHooks()

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        hooks._working_memory = mock_redis

        result = await hooks.cache_drift_result(
            session_id="test-session",
            result={"overall_drift_score": 0.5, "features_with_drift": ["f1"]},
            features=["f1", "f2", "f3"],
            model_id="model_v1",
        )

        assert result is True
        # Session cache + 3 feature caches
        assert mock_redis.set.call_count >= 4

    @pytest.mark.asyncio
    async def test_get_cached_drift_result_success(self):
        """Test successful cache retrieval with mocked Redis."""
        hooks = DriftMonitorMemoryHooks()

        cached_data = {
            "result": {"overall_drift_score": 0.5},
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=cached_data)
        hooks._working_memory = mock_redis

        result = await hooks.get_cached_drift_result("test-session")

        assert result == {"overall_drift_score": 0.5}
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_cache_success(self):
        """Test cache invalidation with mocked Redis."""
        hooks = DriftMonitorMemoryHooks()

        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(return_value=True)
        hooks._working_memory = mock_redis

        result = await hooks.invalidate_cache(
            session_id="test-session",
            feature="f1",
            model_id="model_v1",
        )

        assert result is True
        assert mock_redis.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_store_drift_detection_success(self):
        """Test successful episodic storage with mocked Supabase."""
        hooks = DriftMonitorMemoryHooks()

        mock_response = MagicMock()
        mock_response.data = [{"id": 1}]
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = mock_response
        mock_supabase = MagicMock()
        mock_supabase.table.return_value = mock_table
        hooks._supabase_client = mock_supabase

        result = await hooks.store_drift_detection(
            session_id="test-session",
            result={
                "overall_drift_score": 0.6,
                "features_with_drift": ["f1"],
                "alerts": [{"severity": "high"}],
                "data_drift_results": [{"feature": "f1"}],
                "model_drift_results": [],
                "concept_drift_results": [],
                "drift_summary": "High drift detected",
                "recommended_actions": ["Retrain"],
                "detection_latency_ms": 150,
                "warnings": [],
            },
            state={
                "query": "Check drift",
                "model_id": "model_v1",
                "features_to_monitor": ["f1", "f2"],
                "time_window": "7d",
                "brand": "Remibrutinib",
            },
        )

        assert result is not None
        mock_supabase.table.assert_called_with("agent_activities")

    @pytest.mark.asyncio
    async def test_store_drift_pattern_success(self):
        """Test successful semantic storage with mocked FalkorDB."""
        hooks = DriftMonitorMemoryHooks()

        mock_falkor = AsyncMock()
        mock_falkor.query = AsyncMock(return_value=[])
        hooks._semantic_memory = mock_falkor

        result = await hooks.store_drift_pattern(
            feature="conversion_rate",
            drift_type="data",
            severity="high",
            result={
                "features_with_drift": ["conversion_rate", "engagement"],
                "data_drift_results": [
                    {
                        "feature": "conversion_rate",
                        "test_statistic": 2.5,
                        "p_value": 0.01,
                    }
                ],
            },
            state={"model_id": "model_v1", "brand": "Remibrutinib"},
        )

        assert result is True
        # Should have multiple Cypher queries: Feature node, DriftPattern node, relationship, co-drift, model link
        assert mock_falkor.query.call_count >= 4

    @pytest.mark.asyncio
    async def test_store_all_drift_patterns(self):
        """Test storing all drift patterns."""
        hooks = DriftMonitorMemoryHooks()

        mock_falkor = AsyncMock()
        mock_falkor.query = AsyncMock(return_value=[])
        hooks._semantic_memory = mock_falkor

        result_data = {
            "features_with_drift": ["f1", "f2"],
            "data_drift_results": [
                {"feature": "f1", "drift_detected": True, "severity": "high", "test_statistic": 2.0, "p_value": 0.01},
                {"feature": "f2", "drift_detected": True, "severity": "medium", "test_statistic": 1.5, "p_value": 0.03},
            ],
            "model_drift_results": [],
            "concept_drift_results": [],
        }
        state = {"model_id": None, "brand": None}

        count = await hooks.store_all_drift_patterns(result_data, state)

        assert count == 2

    @pytest.mark.asyncio
    async def test_get_working_memory_context(self):
        """Test working memory context retrieval."""
        hooks = DriftMonitorMemoryHooks()

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(
            side_effect=[
                {"result": "session_cache"},
                {"has_drift": True},
                {"has_drift": False},
            ]
        )
        hooks._working_memory = mock_redis

        context = await hooks._get_working_memory_context(
            session_id="test-session",
            features=["f1", "f2"],
        )

        assert len(context) == 3

    @pytest.mark.asyncio
    async def test_get_episodic_context(self):
        """Test episodic context retrieval."""
        hooks = DriftMonitorMemoryHooks()

        mock_response = MagicMock()
        mock_response.data = [
            {
                "metadata": {
                    "features_monitored": ["f1", "f2"],
                    "overall_drift_score": 0.5,
                }
            }
        ]
        mock_query = MagicMock()
        mock_query.eq.return_value = mock_query
        mock_query.contains.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.execute.return_value = mock_response

        mock_supabase = MagicMock()
        mock_supabase.table.return_value.select.return_value = mock_query
        hooks._supabase_client = mock_supabase

        context = await hooks._get_episodic_context(
            features=["f1", "f3"],
            model_id="model_v1",
            brand="Remibrutinib",
            max_records=5,
        )

        assert len(context) == 1


# ============================================================================
# Test Contribute to Memory Function
# ============================================================================


class TestContributeToMemory:
    """Test contribute_to_memory function."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_hooks()

    @pytest.mark.asyncio
    async def test_contribute_no_drift(self):
        """Test contribution when no drift detected."""
        result = {
            "overall_drift_score": 0.0,
            "features_with_drift": [],
            "alerts": [],
            "data_drift_results": [],
            "model_drift_results": [],
            "concept_drift_results": [],
            "drift_summary": "No drift",
            "recommended_actions": [],
            "detection_latency_ms": 100,
            "warnings": [],
        }
        state = {
            "query": "Check drift",
            "features_to_monitor": ["f1", "f2"],
            "model_id": None,
            "time_window": "7d",
            "brand": None,
        }

        counts = await contribute_to_memory(result, state, "test-session")

        # Without real clients, all counts should be 0
        assert counts["working"] == 0
        assert counts["episodic"] == 0
        assert counts["semantic"] == 0

    @pytest.mark.asyncio
    async def test_contribute_with_mocked_clients(self):
        """Test contribution with mocked memory clients."""
        reset_memory_hooks()

        # Get and configure singleton
        hooks = get_drift_monitor_memory_hooks()

        # Mock working memory
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        hooks._working_memory = mock_redis

        # Mock supabase for episodic
        mock_response = MagicMock()
        mock_response.data = [{"id": 1}]
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = mock_response
        mock_supabase = MagicMock()
        mock_supabase.table.return_value = mock_table
        hooks._supabase_client = mock_supabase

        # Mock semantic memory
        mock_falkor = AsyncMock()
        mock_falkor.query = AsyncMock(return_value=[])
        hooks._semantic_memory = mock_falkor

        result = {
            "overall_drift_score": 0.7,
            "features_with_drift": ["f1", "f2"],
            "alerts": [{"severity": "high"}],
            "data_drift_results": [
                {"feature": "f1", "drift_detected": True, "severity": "high", "test_statistic": 2.0, "p_value": 0.01},
                {"feature": "f2", "drift_detected": True, "severity": "medium", "test_statistic": 1.5, "p_value": 0.03},
            ],
            "model_drift_results": [],
            "concept_drift_results": [],
            "drift_summary": "High drift detected",
            "recommended_actions": ["Retrain"],
            "detection_latency_ms": 150,
            "warnings": [],
        }
        state = {
            "query": "Check drift",
            "features_to_monitor": ["f1", "f2"],
            "model_id": "model_v1",
            "time_window": "7d",
            "brand": "Remibrutinib",
        }

        counts = await contribute_to_memory(result, state, "test-session")

        assert counts["working"] > 0
        assert counts["episodic"] == 1
        assert counts["semantic"] == 2


# ============================================================================
# Test Singleton Access
# ============================================================================


class TestSingletonAccess:
    """Test singleton access pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_hooks()

    def test_get_singleton(self):
        """Test getting singleton instance."""
        hooks1 = get_drift_monitor_memory_hooks()
        hooks2 = get_drift_monitor_memory_hooks()

        assert hooks1 is hooks2

    def test_reset_singleton(self):
        """Test resetting singleton."""
        hooks1 = get_drift_monitor_memory_hooks()
        reset_memory_hooks()
        hooks2 = get_drift_monitor_memory_hooks()

        assert hooks1 is not hooks2
