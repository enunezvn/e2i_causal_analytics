"""Tests for metrics_aggregator node (aggregate_metrics)."""

import pytest

from src.agents.ml_foundation.observability_connector.nodes.metrics_aggregator import (
    aggregate_metrics,
)


class TestAggregateMetrics:
    """Test aggregate_metrics node."""

    @pytest.mark.asyncio
    async def test_aggregate_metrics_success(self):
        """Test successful metrics aggregation."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        assert result["quality_metrics_computed"] is True
        assert "latency_by_agent" in result
        assert "latency_by_tier" in result
        assert "error_rate_by_agent" in result
        assert "error_rate_by_tier" in result
        assert "token_usage_by_agent" in result
        assert "overall_success_rate" in result
        assert "overall_p95_latency_ms" in result
        assert "overall_p99_latency_ms" in result
        assert "total_spans_analyzed" in result
        assert "quality_score" in result
        assert "fallback_invocation_rate" in result
        assert "status_distribution" in result

    @pytest.mark.asyncio
    async def test_aggregate_metrics_latency_by_agent(self):
        """Test latency metrics by agent."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        latency_by_agent = result["latency_by_agent"]

        # Check that we have stats for agents
        assert len(latency_by_agent) > 0

        # Check that each agent has p50, p95, p99, avg
        for _agent_name, stats in latency_by_agent.items():
            assert "p50" in stats
            assert "p95" in stats
            assert "p99" in stats
            assert "avg" in stats
            assert stats["p50"] > 0
            assert stats["p95"] >= stats["p50"]
            assert stats["p99"] >= stats["p95"]

    @pytest.mark.asyncio
    async def test_aggregate_metrics_latency_by_tier(self):
        """Test latency metrics by tier."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        latency_by_tier = result["latency_by_tier"]

        # Check that we have tier 0 (all mock agents are tier 0)
        assert "0" in latency_by_tier

        # Check stats structure
        tier0_stats = latency_by_tier["0"]
        assert "p50" in tier0_stats
        assert "p95" in tier0_stats
        assert "p99" in tier0_stats
        assert "avg" in tier0_stats

    @pytest.mark.asyncio
    async def test_aggregate_metrics_error_rates(self):
        """Test error rate computation."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        error_rate_by_agent = result["error_rate_by_agent"]
        error_rate_by_tier = result["error_rate_by_tier"]

        # Check that error rates are between 0 and 1
        for _agent_name, error_rate in error_rate_by_agent.items():
            assert 0.0 <= error_rate <= 1.0

        for _tier, error_rate in error_rate_by_tier.items():
            assert 0.0 <= error_rate <= 1.0

    @pytest.mark.asyncio
    async def test_aggregate_metrics_token_usage(self):
        """Test token usage computation for Hybrid agents."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        token_usage_by_agent = result["token_usage_by_agent"]

        # Check that feature_analyzer (Hybrid agent) has token usage
        if "feature_analyzer" in token_usage_by_agent:
            usage = token_usage_by_agent["feature_analyzer"]
            assert "input" in usage
            assert "output" in usage
            assert "total" in usage
            assert usage["total"] == usage["input"] + usage["output"]

    @pytest.mark.asyncio
    async def test_aggregate_metrics_overall_success_rate(self):
        """Test overall success rate computation."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        overall_success_rate = result["overall_success_rate"]

        # Success rate should be between 0 and 1
        assert 0.0 <= overall_success_rate <= 1.0

        # Should be high for mock data (low error rate)
        assert overall_success_rate > 0.9

    @pytest.mark.asyncio
    async def test_aggregate_metrics_percentiles(self):
        """Test percentile computation."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        p95 = result["overall_p95_latency_ms"]
        p99 = result["overall_p99_latency_ms"]

        # Percentiles should be positive
        assert p95 > 0
        assert p99 > 0

        # P99 should be >= P95
        assert p99 >= p95

    @pytest.mark.asyncio
    async def test_aggregate_metrics_quality_score(self):
        """Test quality score computation."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        quality_score = result["quality_score"]

        # Quality score should be between 0 and 1
        assert 0.0 <= quality_score <= 1.0

        # Should be high for mock data (low errors, reasonable latency)
        assert quality_score > 0.5

    @pytest.mark.asyncio
    async def test_aggregate_metrics_fallback_rate(self):
        """Test fallback invocation rate."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        fallback_rate = result["fallback_invocation_rate"]

        # Fallback rate should be between 0 and 1
        assert 0.0 <= fallback_rate <= 1.0

        # Should be low for mock data
        assert fallback_rate < 0.1

    @pytest.mark.asyncio
    async def test_aggregate_metrics_status_distribution(self):
        """Test status distribution."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        status_dist = result["status_distribution"]

        # Should have ok, error, timeout
        assert "ok" in status_dist
        assert "error" in status_dist
        assert "timeout" in status_dist

        # Total should match total_spans_analyzed
        total = status_dist["ok"] + status_dist["error"] + status_dist["timeout"]
        assert total == result["total_spans_analyzed"]

        # Most should be ok
        assert status_dist["ok"] > status_dist["error"]
        assert status_dist["ok"] > status_dist["timeout"]

    @pytest.mark.asyncio
    async def test_aggregate_metrics_total_spans(self):
        """Test total spans analyzed."""
        state = {"time_window": "24h"}

        result = await aggregate_metrics(state)

        total_spans = result["total_spans_analyzed"]

        # Should have analyzed many spans (mock data has 1000)
        assert total_spans > 0

    @pytest.mark.asyncio
    async def test_aggregate_metrics_with_agent_filter(self):
        """Test metrics aggregation with agent filter."""
        state = {"time_window": "24h", "agent_name_filter": "scope_definer"}

        result = await aggregate_metrics(state)

        # Metrics should still be computed
        assert result["quality_metrics_computed"] is True

    @pytest.mark.asyncio
    async def test_aggregate_metrics_with_trace_filter(self):
        """Test metrics aggregation with trace filter."""
        state = {"time_window": "24h", "trace_id_filter": "trace_123"}

        result = await aggregate_metrics(state)

        # Metrics should still be computed
        assert result["quality_metrics_computed"] is True

    @pytest.mark.asyncio
    async def test_aggregate_metrics_different_time_windows(self):
        """Test metrics with different time windows."""
        for window in ["1h", "24h", "7d"]:
            state = {"time_window": window}

            result = await aggregate_metrics(state)

            assert result["quality_metrics_computed"] is True
            assert result["total_spans_analyzed"] > 0
