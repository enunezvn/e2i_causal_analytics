"""
Integration tests for E2I Observability Connector.

Tests end-to-end observability flows including:
- Opik SDK span emission
- Database write and query round-trip
- Metrics computation from real data
- Batch processing under load
- Circuit breaker behavior
- Cross-agent context propagation

These tests require real connections to Opik and/or Supabase.
Use pytest markers to skip when services are unavailable.
"""

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone, UTC
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Import observability components
# =============================================================================

from src.agents.ml_foundation.observability_connector import (
    AgentNameEnum,
    AgentTierEnum,
    BatchConfig,
    BatchProcessor,
    CacheBackend,
    CacheConfig,
    MetricsCache,
    ObservabilitySpan,
    SpanStatusEnum,
    create_span,
    get_batch_processor,
    get_metrics_cache,
    reset_batch_processor,
    reset_metrics_cache,
)
from src.agents.ml_foundation.observability_connector.self_monitor import (
    AsyncLatencyContext,
    HealthStatus,
    LatencyContext,
    MetricType,
    SelfMonitor,
    SelfMonitorConfig,
    get_self_monitor,
    reset_self_monitor,
)
from src.mlops.opik_connector import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
    OpikConnector,
    get_opik_connector,
    reset_opik_connector,
)

# =============================================================================
# Test Configuration
# =============================================================================

# Check for required environment variables
HAS_OPIK_API_KEY = bool(os.getenv("OPIK_API_KEY"))
HAS_SUPABASE_URL = bool(os.getenv("SUPABASE_URL"))
HAS_SUPABASE_KEY = bool(os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY"))
HAS_REDIS_URL = bool(os.getenv("REDIS_URL"))

# Pytest markers for conditional skipping
requires_opik = pytest.mark.skipif(
    not HAS_OPIK_API_KEY,
    reason="OPIK_API_KEY environment variable not set",
)
requires_supabase = pytest.mark.skipif(
    not (HAS_SUPABASE_URL and HAS_SUPABASE_KEY),
    reason="SUPABASE_URL and SUPABASE_KEY environment variables not set",
)
requires_redis = pytest.mark.skipif(
    not HAS_REDIS_URL,
    reason="REDIS_URL environment variable not set",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trace_id() -> str:
    """Generate a unique trace ID for test isolation."""
    return f"test-trace-{uuid.uuid4().hex[:16]}"


@pytest.fixture
def span_id() -> str:
    """Generate a unique span ID."""
    return f"test-span-{uuid.uuid4().hex[:16]}"


@pytest.fixture
def sample_span(trace_id: str, span_id: str) -> ObservabilitySpan:
    """Create a sample span for testing.

    Uses OBSERVABILITY_CONNECTOR agent which exists in the database agent_name_enum.
    """
    return ObservabilitySpan(
        trace_id=trace_id,
        span_id=span_id,
        agent_name=AgentNameEnum.OBSERVABILITY_CONNECTOR,
        agent_tier=AgentTierEnum.ML_FOUNDATION,
        operation_type="test_operation",
        started_at=datetime.utcnow(),  # Use naive datetime for complete() compatibility
        status=SpanStatusEnum.SUCCESS,
        attributes={"test": True, "integration": True},
    )


@pytest.fixture
def multiple_spans(trace_id: str) -> list[ObservabilitySpan]:
    """Create multiple spans for batch testing.

    Uses Tier 0 agents which exist in the database agent_name_enum.
    """
    spans = []
    # Use Tier 0 agents that exist in the database agent_name_enum
    tiers = [AgentTierEnum.ML_FOUNDATION, AgentTierEnum.ML_FOUNDATION]
    agents = [AgentNameEnum.OBSERVABILITY_CONNECTOR, AgentNameEnum.DATA_PREPARER]

    for i in range(10):
        span = ObservabilitySpan(
            trace_id=trace_id,
            span_id=f"test-span-{i}-{uuid.uuid4().hex[:8]}",
            agent_name=agents[i % 2],
            agent_tier=tiers[i % 2],
            operation_type=f"test_operation_{i}",
            started_at=datetime.utcnow(),  # Use naive datetime for complete() compatibility
            status=SpanStatusEnum.SUCCESS if i % 3 != 0 else SpanStatusEnum.ERROR,
            duration_ms=50 + i * 10,
            attributes={"index": i, "test": True},
        )
        spans.append(span)
    return spans


@pytest.fixture(autouse=True)
async def reset_singletons():
    """Reset all singletons before and after each test."""
    reset_opik_connector()
    await reset_batch_processor()
    await reset_metrics_cache()
    reset_self_monitor()
    yield
    reset_opik_connector()
    await reset_batch_processor()
    await reset_metrics_cache()
    reset_self_monitor()


# =============================================================================
# Test Classes
# =============================================================================


class TestOpikIntegration:
    """Integration tests for Opik SDK connectivity."""

    @requires_opik
    @pytest.mark.asyncio
    async def test_opik_health_check(self):
        """Test Opik API health check."""
        connector = get_opik_connector()

        # Health check should succeed with valid API key
        is_healthy = await connector.health_check()
        assert is_healthy is True

    @requires_opik
    @pytest.mark.asyncio
    async def test_emit_single_span_to_opik(self, sample_span: ObservabilitySpan):
        """Test emitting a single span to Opik."""
        connector = get_opik_connector()

        # Complete the span
        sample_span.complete()

        # Emit to Opik
        success = await connector.emit_span(sample_span)
        assert success is True

    @requires_opik
    @pytest.mark.asyncio
    async def test_emit_multiple_spans_to_opik(self, multiple_spans: list[ObservabilitySpan]):
        """Test emitting multiple spans to Opik."""
        connector = get_opik_connector()

        # Complete all spans
        for span in multiple_spans:
            span.complete()

        # Emit batch
        results = await connector.emit_spans_batch(multiple_spans)

        # All should succeed
        assert len(results) == len(multiple_spans)
        assert all(r is True for r in results)

    @requires_opik
    @pytest.mark.asyncio
    async def test_trace_context_manager(self, trace_id: str):
        """Test the trace context manager for automatic span creation."""
        connector = get_opik_connector()

        async with connector.trace_agent(
            agent_name=AgentNameEnum.ORCHESTRATOR,
            operation_type="integration_test_trace",
            trace_id=trace_id,
        ) as span:
            # Simulate some work
            await asyncio.sleep(0.01)
            span.attributes["test_value"] = "hello"

        # Span should be completed and emitted
        assert span.ended_at is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 10  # At least 10ms


class TestDatabaseIntegration:
    """Integration tests for Supabase database operations."""

    @pytest.fixture
    def supabase_repo(self):
        """Create repository with Supabase client."""
        from src.repositories import get_supabase_client
        from src.repositories.observability_span import ObservabilitySpanRepository

        client = get_supabase_client()
        return ObservabilitySpanRepository(supabase_client=client)

    @requires_supabase
    @pytest.mark.asyncio
    async def test_insert_and_query_span(self, sample_span: ObservabilitySpan, supabase_repo):
        """Test inserting a span and querying it back."""
        # Complete the span
        sample_span.complete()

        # Insert
        result = await supabase_repo.insert_span(sample_span)
        assert result is not None, "Insert should return the inserted span"

        # Query back
        spans = await supabase_repo.get_spans_by_trace_id(sample_span.trace_id)
        assert len(spans) >= 1

        found = next((s for s in spans if s.span_id == sample_span.span_id), None)
        assert found is not None
        assert found.agent_name == sample_span.agent_name

    @requires_supabase
    @pytest.mark.asyncio
    async def test_batch_insert_spans(self, multiple_spans: list[ObservabilitySpan], supabase_repo):
        """Test batch inserting multiple spans."""
        # Complete all spans
        for span in multiple_spans:
            span.complete()

        # Batch insert - returns dict with success/failure info
        results = await supabase_repo.insert_spans_batch(multiple_spans)

        assert isinstance(results, dict)
        assert results["success"], f"Batch insert failed: {results}"
        assert results["inserted_count"] == len(multiple_spans)

        # Query back
        trace_id = multiple_spans[0].trace_id
        spans = await supabase_repo.get_spans_by_trace_id(trace_id)
        assert len(spans) >= len(multiple_spans)

    @requires_supabase
    @pytest.mark.asyncio
    async def test_get_latency_stats(self, multiple_spans: list[ObservabilitySpan], supabase_repo):
        """Test latency statistics from database view."""
        # Insert test spans first
        for span in multiple_spans:
            span.complete()
        await supabase_repo.insert_spans_batch(multiple_spans)

        # Get latency stats (API: agent_name=None returns all agents)
        stats = await supabase_repo.get_latency_stats(agent_name=None)
        assert stats is not None
        assert isinstance(stats, list)

    @requires_supabase
    @pytest.mark.asyncio
    async def test_get_spans_by_agent(self, multiple_spans: list[ObservabilitySpan], supabase_repo):
        """Test querying spans by agent name."""
        # Insert test spans
        for span in multiple_spans:
            span.complete()
        await supabase_repo.insert_spans_batch(multiple_spans)

        # Query by agent (API: agent_name: str, hours: int = 24)
        # Use observability_connector which exists in the database agent_name_enum
        spans = await supabase_repo.get_spans_by_agent(
            agent_name="observability_connector",
            hours=1,
        )
        assert isinstance(spans, list)


class TestMetricsComputation:
    """Integration tests for metrics aggregation from real data."""

    @pytest.fixture
    def supabase_repo(self):
        """Create repository with Supabase client."""
        from src.repositories import get_supabase_client
        from src.repositories.observability_span import ObservabilitySpanRepository

        client = get_supabase_client()
        return ObservabilitySpanRepository(supabase_client=client)

    @requires_supabase
    @pytest.mark.asyncio
    async def test_aggregate_metrics_from_database(self, multiple_spans: list[ObservabilitySpan], supabase_repo):
        """Test computing metrics from database spans."""
        from src.agents.ml_foundation.observability_connector.nodes.metrics_aggregator import (
            aggregate_metrics,
        )

        # Insert test data
        for span in multiple_spans:
            span.complete()
        await supabase_repo.insert_spans_batch(multiple_spans)

        # Create state dict (aggregate_metrics takes Dict, not ObservabilityConnectorState)
        state: dict[str, Any] = {
            "query_type": "get_quality_metrics",
            "time_window": "1h",
        }

        # Aggregate metrics
        result = await aggregate_metrics(state)

        # Should return dict with metrics keys
        assert isinstance(result, dict)
        # Result should have at least aggregated_metrics key
        assert "aggregated_metrics" in result or "latency_by_agent" in result or "error_rates" in result

    @pytest.mark.asyncio
    async def test_metrics_cache_integration(self):
        """Test metrics caching with in-memory backend."""
        cache = get_metrics_cache(
            config=CacheConfig(
                backend=CacheBackend.MEMORY,
            )
        )

        # Store metrics
        test_metrics = {
            "total_spans": 100,
            "error_rate": 0.05,
            "avg_latency_ms": 150.0,
        }

        success = await cache.set_metrics(
            window="1h",
            agent=None,
            metrics=test_metrics,
        )
        assert success is True

        # Retrieve
        cached = await cache.get_metrics(window="1h", agent=None)
        assert cached is not None
        assert cached["total_spans"] == 100

    @requires_redis
    @pytest.mark.asyncio
    async def test_metrics_cache_redis_integration(self):
        """Test metrics caching with Redis backend."""
        cache = get_metrics_cache(
            config=CacheConfig(
                backend=CacheBackend.REDIS,
            )
        )

        # Store metrics
        test_metrics = {"test": True, "value": 42}

        success = await cache.set_metrics(
            window="1h",
            agent="test_agent",
            metrics=test_metrics,
        )
        assert success is True

        # Retrieve
        cached = await cache.get_metrics(window="1h", agent="test_agent")
        assert cached is not None
        assert cached["value"] == 42

        # Invalidate
        await cache.invalidate(agent="test_agent")
        cached_after = await cache.get_metrics(window="1h", agent="test_agent")
        assert cached_after is None


class TestBatchProcessing:
    """Integration tests for batch processing under load."""

    @pytest.mark.asyncio
    async def test_batch_processor_accumulates_spans(self, multiple_spans: list[ObservabilitySpan]):
        """Test that batch processor accumulates spans correctly."""
        processor = BatchProcessor(
            config=BatchConfig(
                max_batch_size=50,  # Won't auto-flush
                max_wait_seconds=60.0,
            )
        )

        # Add spans as dicts
        for span in multiple_spans:
            span.complete()
            await processor.add_span(span.model_dump(mode="json"))

        # Check buffer
        status = processor.get_status()
        assert status["buffer_size"] == len(multiple_spans)

    @pytest.mark.asyncio
    async def test_batch_processor_auto_flush_on_size(self):
        """Test automatic flush when buffer reaches max size."""
        flush_results: list[dict] = []

        def capture_flush(result: dict) -> None:
            flush_results.append(result)

        processor = BatchProcessor(
            config=BatchConfig(
                max_batch_size=5,  # Small size for quick flush
                max_wait_seconds=60.0,
            ),
            on_flush_complete=capture_flush,
        )

        # Add spans to trigger flush (as dicts)
        for i in range(6):
            span = create_span(
                trace_id=f"trace-{uuid.uuid4().hex[:8]}",
                span_id=f"span-{uuid.uuid4().hex[:8]}",
                agent_name=AgentNameEnum.ORCHESTRATOR,
                agent_tier=AgentTierEnum.COORDINATION,
                operation_type=f"op_{i}",
            )
            span.complete()
            await processor.add_span(span.model_dump(mode="json"))

        # Should have flushed when reaching 5 spans
        # Buffer should have 1 remaining span
        status = processor.get_status()
        assert status["buffer_size"] == 1
        assert status["metrics"]["total_spans_processed"] == 5

    @pytest.mark.asyncio
    async def test_batch_processor_periodic_flush(self):
        """Test periodic flush based on time."""
        flush_results: list[dict] = []

        def on_flush(result: dict) -> None:
            flush_results.append(result)

        processor = BatchProcessor(
            config=BatchConfig(
                max_batch_size=100,
                max_wait_seconds=0.5,  # Short wait for testing
            ),
            on_flush_complete=on_flush,
        )

        # Start background flush
        await processor.start()

        # Add a span as dict
        span = create_span(
            trace_id=f"trace-{uuid.uuid4().hex[:8]}",
            span_id=f"span-{uuid.uuid4().hex[:8]}",
            agent_name=AgentNameEnum.ORCHESTRATOR,
            agent_tier=AgentTierEnum.COORDINATION,
            operation_type="periodic_test",
        )
        span.complete()
        await processor.add_span(span.model_dump(mode="json"))

        # Wait for periodic flush
        try:
            await asyncio.sleep(0.8)  # Wait longer than max_wait_seconds
            # Verify flush happened by checking stats
            status = processor.get_status()
            assert status["metrics"]["total_spans_processed"] >= 1
        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_batch_processor_high_throughput(self):
        """Test batch processor under high throughput."""
        flush_count = 0

        def count_flush(result: dict) -> None:
            nonlocal flush_count
            flush_count += 1

        processor = BatchProcessor(
            config=BatchConfig(
                max_batch_size=20,
                max_wait_seconds=0.1,
            ),
            on_flush_complete=count_flush,
        )
        await processor.start()

        # Add 100 spans rapidly (as dicts)
        try:
            for i in range(100):
                span = create_span(
                    trace_id=f"trace-{uuid.uuid4().hex[:8]}",
                    span_id=f"span-{uuid.uuid4().hex[:8]}",
                    agent_name=AgentNameEnum.ORCHESTRATOR,
                    agent_tier=AgentTierEnum.COORDINATION,
                    operation_type=f"throughput_test_{i}",
                )
                span.complete()
                await processor.add_span(span.model_dump(mode="json"))

            # Wait for all flushes
            await asyncio.sleep(0.5)
            await processor.flush()

            # Should have processed all spans
            status = processor.get_status()
            assert status["metrics"]["total_spans_processed"] == 100
            assert flush_count >= 5  # At least 5 batches of 20
        finally:
            await processor.stop()


class TestCircuitBreaker:
    """Integration tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after consecutive failures."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            reset_timeout_seconds=0.5,
            half_open_max_calls=1,
        )
        circuit = CircuitBreaker(config)

        # Record failures
        for _ in range(3):
            circuit.record_failure()

        # Circuit should be open
        assert circuit.state == CircuitState.OPEN
        assert not circuit.allow_request()

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker transitions to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.1,  # Short for testing
            half_open_max_calls=1,
        )
        circuit = CircuitBreaker(config)

        # Open the circuit
        circuit.record_failure()
        circuit.record_failure()
        assert circuit.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should allow request (half-open)
        assert circuit.allow_request()
        assert circuit.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_on_success(self):
        """Test circuit breaker closes after success in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.1,
            half_open_max_calls=1,
            success_threshold=1,
        )
        circuit = CircuitBreaker(config)

        # Open the circuit
        circuit.record_failure()
        circuit.record_failure()

        # Wait for half-open
        await asyncio.sleep(0.15)
        circuit.allow_request()

        # Record success
        circuit.record_success()

        # Should be closed
        assert circuit.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_reopens_on_half_open_failure(self):
        """Test circuit breaker reopens if failure in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.1,
            half_open_max_calls=1,
        )
        circuit = CircuitBreaker(config)

        # Open the circuit
        circuit.record_failure()
        circuit.record_failure()

        # Wait for half-open
        await asyncio.sleep(0.15)
        circuit.allow_request()

        # Fail in half-open
        circuit.record_failure()

        # Should be open again
        assert circuit.state == CircuitState.OPEN

    @requires_opik
    @pytest.mark.asyncio
    async def test_opik_connector_with_circuit_breaker(self):
        """Test Opik connector uses circuit breaker correctly."""
        connector = get_opik_connector()

        # Get circuit breaker metrics
        metrics = connector.get_circuit_breaker_metrics()
        assert metrics is not None
        assert metrics.state == CircuitState.CLOSED


class TestCrossAgentContextPropagation:
    """Integration tests for context propagation across agents."""

    @pytest.mark.asyncio
    async def test_full_trace_propagation(self, trace_id: str):
        """Test full trace propagation across multiple spans."""
        spans: list[ObservabilitySpan] = []

        # Parent span (Orchestrator)
        parent_span = create_span(
            trace_id=trace_id,
            span_id=f"parent-{uuid.uuid4().hex[:8]}",
            agent_name=AgentNameEnum.ORCHESTRATOR,
            agent_tier=AgentTierEnum.COORDINATION,
            operation_type="orchestrate_query",
        )
        spans.append(parent_span)

        # Child span 1 (Gap Analyzer)
        child1_span = create_span(
            trace_id=trace_id,
            span_id=f"child1-{uuid.uuid4().hex[:8]}",
            agent_name=AgentNameEnum.GAP_ANALYZER,
            agent_tier=AgentTierEnum.CAUSAL_ANALYTICS,
            operation_type="analyze_gaps",
            parent_span_id=parent_span.span_id,
        )
        spans.append(child1_span)

        # Child span 2 (Causal Impact)
        child2_span = create_span(
            trace_id=trace_id,
            span_id=f"child2-{uuid.uuid4().hex[:8]}",
            agent_name=AgentNameEnum.CAUSAL_IMPACT,
            agent_tier=AgentTierEnum.CAUSAL_ANALYTICS,
            operation_type="compute_impact",
            parent_span_id=parent_span.span_id,
        )
        spans.append(child2_span)

        # Grandchild span
        grandchild_span = create_span(
            trace_id=trace_id,
            span_id=f"grandchild-{uuid.uuid4().hex[:8]}",
            agent_name=AgentNameEnum.GAP_ANALYZER,
            agent_tier=AgentTierEnum.CAUSAL_ANALYTICS,
            operation_type="compute_roi",
            parent_span_id=child1_span.span_id,
        )
        spans.append(grandchild_span)

        # Complete all
        for span in spans:
            span.complete()

        # Verify trace structure
        assert all(s.trace_id == trace_id for s in spans)
        assert child1_span.parent_span_id == parent_span.span_id
        assert child2_span.parent_span_id == parent_span.span_id
        assert grandchild_span.parent_span_id == child1_span.span_id

    def test_span_hierarchy_with_attributes(self, trace_id: str):
        """Test span hierarchy preserves attributes through propagation."""
        parent = create_span(
            trace_id=trace_id,
            span_id=f"parent-{uuid.uuid4().hex[:8]}",
            agent_name=AgentNameEnum.ORCHESTRATOR,
            agent_tier=AgentTierEnum.COORDINATION,
            operation_type="query",
        )
        parent.attributes["query"] = "test query"
        parent.attributes["user_id"] = "user123"

        child = create_span(
            trace_id=trace_id,
            span_id=f"child-{uuid.uuid4().hex[:8]}",
            agent_name=AgentNameEnum.GAP_ANALYZER,
            agent_tier=AgentTierEnum.CAUSAL_ANALYTICS,
            operation_type="analyze",
            parent_span_id=parent.span_id,
        )
        child.attributes["parent_query"] = parent.attributes.get("query")

        # Verify attributes propagation
        assert child.attributes["parent_query"] == "test query"
        assert child.parent_span_id == parent.span_id


class TestSelfMonitoringIntegration:
    """Integration tests for self-monitoring functionality."""

    def test_latency_tracking_integration(self):
        """Test latency tracking with real timing."""
        monitor = get_self_monitor()

        # Track some operations with real delays
        for _ in range(5):
            with LatencyContext(monitor, MetricType.SPAN_EMISSION):
                time.sleep(0.01)  # 10ms

        # Get health status
        health = monitor.get_health_status()

        # Should have stats for span emission
        assert MetricType.SPAN_EMISSION in health.components
        component = health.components[MetricType.SPAN_EMISSION]
        assert component.latency_stats.count == 5
        assert component.latency_stats.avg_ms >= 10  # At least 10ms average

    @pytest.mark.asyncio
    async def test_async_latency_tracking(self):
        """Test async latency tracking with real timing."""
        monitor = get_self_monitor()

        # Track async operations
        for _ in range(3):
            async with AsyncLatencyContext(monitor, MetricType.OPIK_API):
                await asyncio.sleep(0.02)  # 20ms

        # Get health status
        health = monitor.get_health_status()

        assert MetricType.OPIK_API in health.components
        component = health.components[MetricType.OPIK_API]
        assert component.latency_stats.count == 3
        assert component.latency_stats.avg_ms >= 20

    def test_error_tracking_affects_health(self):
        """Test that errors affect health status."""
        config = SelfMonitorConfig(
            window_size=10,
            min_samples_for_stats=3,
        )
        monitor = get_self_monitor(config=config, force_new=True)

        # Record some successes
        for _ in range(5):
            monitor.record_database_latency(50.0)

        # Record errors (more than 10% error rate)
        for _ in range(3):
            monitor.record_database_error("Test error")

        # Get health
        health = monitor.get_health_status()
        component = health.components[MetricType.DATABASE_WRITE]

        # Should have errors recorded
        assert component.error_count >= 3

    @pytest.mark.asyncio
    async def test_health_span_emission(self):
        """Test that health spans are emitted correctly."""
        config = SelfMonitorConfig(
            emit_health_spans=True,
            health_emission_interval_seconds=0.1,
        )
        monitor = get_self_monitor(config=config, force_new=True)

        # Record some data
        monitor.record_span_emission_latency(50.0)
        monitor.record_database_latency(30.0)

        # Emit health span
        health = await monitor.emit_health_span_now()

        assert health is not None
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.UNKNOWN]


class TestEndToEndFlow:
    """End-to-end integration tests combining all components."""

    @requires_opik
    @requires_supabase
    @pytest.mark.asyncio
    async def test_full_observability_pipeline(self, trace_id: str):
        """Test complete observability pipeline: create → emit → store → query."""
        from src.repositories.observability_span import ObservabilitySpanRepository

        repo = ObservabilitySpanRepository()
        connector = get_opik_connector()
        monitor = get_self_monitor()

        # Create spans with self-monitoring
        spans: list[ObservabilitySpan] = []

        async with AsyncLatencyContext(monitor, MetricType.SPAN_EMISSION):
            for i in range(3):
                span = create_span(
                    trace_id=trace_id,
                    span_id=f"e2e-{i}-{uuid.uuid4().hex[:8]}",
                    agent_name=AgentNameEnum.ORCHESTRATOR,
                    agent_tier=AgentTierEnum.COORDINATION,
                    operation_type=f"e2e_test_{i}",
                )
                span.complete()
                spans.append(span)

        # Emit to Opik
        async with AsyncLatencyContext(monitor, MetricType.OPIK_API):
            opik_results = await connector.emit_spans_batch(spans)

        # Store in database
        async with AsyncLatencyContext(monitor, MetricType.DATABASE_WRITE):
            db_results = await repo.insert_spans_batch(spans)

        # Query back
        stored_spans = await repo.get_spans_by_trace_id(trace_id)

        # Verify
        assert all(r is True for r in opik_results)
        assert len(db_results) == 3
        assert len(stored_spans) >= 3

        # Check health
        health = monitor.get_health_status()
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_services(self, sample_span: ObservabilitySpan):
        """Test that system degrades gracefully without external services."""
        # Mock unavailable services
        with patch.dict(os.environ, {"OPIK_API_KEY": ""}, clear=False):
            reset_opik_connector()

            processor = get_batch_processor(
                config=BatchConfig(max_batch_size=10, max_wait_seconds=1.0)
            )

            # Should still accept spans
            sample_span.complete()
            await processor.add_span(sample_span.model_dump(mode="json"))

            status = processor.get_status()
            assert status["buffer_size"] == 1

            # Flush should not crash
            await processor.flush()


# =============================================================================
# Performance Tests (Load Testing)
# =============================================================================


class TestLoadPerformance:
    """Load and performance tests for observability system."""

    @pytest.mark.asyncio
    async def test_100_concurrent_span_emissions(self):
        """Test handling 100 concurrent span emissions."""
        monitor = get_self_monitor()

        async def emit_span(index: int) -> ObservabilitySpan:
            span = create_span(
                trace_id=f"load-test-{uuid.uuid4().hex[:8]}",
                span_id=f"span-{index}-{uuid.uuid4().hex[:8]}",
                agent_name=AgentNameEnum.ORCHESTRATOR,
                agent_tier=AgentTierEnum.COORDINATION,
                operation_type=f"load_test_{index}",
            )
            # Simulate some processing
            await asyncio.sleep(0.001)
            span.complete()
            monitor.record_span_emission_latency(span.duration_ms or 1.0)
            return span

        # Run 100 concurrent emissions
        start_time = time.time()
        tasks = [emit_span(i) for i in range(100)]
        spans = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # Verify
        assert len(spans) == 100
        assert all(s.ended_at is not None for s in spans)

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0, f"100 emissions took {elapsed:.2f}s, expected < 5s"

    @pytest.mark.asyncio
    async def test_batch_1000_spans(self):
        """Test batching 1000 spans."""
        batch_count = 0

        def count_batch(result: dict) -> None:
            nonlocal batch_count
            batch_count += 1

        processor = BatchProcessor(
            config=BatchConfig(max_batch_size=100, max_wait_seconds=0.5),
            on_flush_complete=count_batch,
        )
        await processor.start()

        try:
            # Add 1000 spans
            start_time = time.time()
            for i in range(1000):
                span = create_span(
                    trace_id=f"batch-test-{uuid.uuid4().hex[:8]}",
                    span_id=f"span-{i}-{uuid.uuid4().hex[:8]}",
                    agent_name=AgentNameEnum.ORCHESTRATOR,
                    agent_tier=AgentTierEnum.COORDINATION,
                    operation_type=f"batch_test_{i}",
                )
                span.complete()
                await processor.add_span(span.model_dump(mode="json"))

            # Final flush
            await processor.flush()
            elapsed = time.time() - start_time

            # Verify
            status = processor.get_status()
            assert status["metrics"]["total_spans_processed"] == 1000
            assert batch_count == 10  # 1000 / 100 = 10 batches

            # Should complete quickly
            assert elapsed < 10.0, f"1000 spans took {elapsed:.2f}s"
        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_cache_performance_1000_operations(self):
        """Test cache performance with 1000 operations."""
        cache = get_metrics_cache(
            config=CacheConfig(
                backend=CacheBackend.MEMORY,
            )
        )

        start_time = time.time()

        # Write 1000 entries
        for i in range(1000):
            await cache.set_metrics(
                window="1h",
                agent=f"agent_{i % 50}",
                metrics={"index": i, "value": i * 2},
            )

        write_time = time.time() - start_time

        # Read 1000 entries
        start_time = time.time()
        hits = 0
        for i in range(1000):
            result = await cache.get_metrics(window="1h", agent=f"agent_{i % 50}")
            if result is not None:
                hits += 1

        read_time = time.time() - start_time

        # Performance assertions
        assert write_time < 1.0, f"1000 writes took {write_time:.2f}s"
        assert read_time < 0.5, f"1000 reads took {read_time:.2f}s"

        # Should have hits (last write for each agent preserved)
        assert hits >= 50, f"Expected at least 50 hits, got {hits}"
