"""
Tests for Pipeline Health Node
"""

import pytest

from src.agents.health_score.nodes.pipeline_health import PipelineHealthNode


class TestPipelineHealthNode:
    """Tests for PipelineHealthNode"""

    @pytest.mark.asyncio
    async def test_healthy_pipelines(self, mock_pipeline_store, initial_state):
        """Test all healthy pipelines"""
        node = PipelineHealthNode(pipeline_store=mock_pipeline_store)
        result = await node.execute(initial_state)

        assert result["pipeline_health_score"] == 1.0
        assert len(result["pipeline_statuses"]) == 2
        for status in result["pipeline_statuses"]:
            assert status["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_stale_and_failed_pipelines(self, stale_pipeline_store, initial_state):
        """Test with stale and failed pipelines"""
        node = PipelineHealthNode(pipeline_store=stale_pipeline_store)
        result = await node.execute(initial_state)

        # 1 healthy + 1 stale (0.5) + 1 failed (0) = 1.5/3 = 0.5
        assert result["pipeline_health_score"] == 0.5

        # Check statuses
        status_map = {p["pipeline_name"]: p["status"] for p in result["pipeline_statuses"]}
        assert status_map["healthy_pipeline"] == "healthy"
        assert status_map["stale_pipeline"] == "stale"
        assert status_map["failed_pipeline"] == "failed"

    @pytest.mark.asyncio
    async def test_skips_for_non_pipeline_scope(self, mock_pipeline_store, initial_state):
        """Test that non-pipeline scope skips pipeline check"""
        initial_state["check_scope"] = "models"
        node = PipelineHealthNode(pipeline_store=mock_pipeline_store)
        result = await node.execute(initial_state)

        assert result["pipeline_health_score"] == 1.0
        assert result["pipeline_statuses"] == []

    @pytest.mark.asyncio
    async def test_includes_for_pipeline_scope(self, mock_pipeline_store, initial_state):
        """Test that pipelines scope includes pipeline check"""
        initial_state["check_scope"] = "pipelines"
        node = PipelineHealthNode(pipeline_store=mock_pipeline_store)
        result = await node.execute(initial_state)

        assert len(result["pipeline_statuses"]) == 2

    @pytest.mark.asyncio
    async def test_no_store_returns_healthy(self, initial_state):
        """Test that no store returns healthy by default"""
        node = PipelineHealthNode(pipeline_store=None)
        result = await node.execute(initial_state)

        assert result["pipeline_health_score"] == 1.0
        assert result["pipeline_statuses"] == []

    @pytest.mark.asyncio
    async def test_accumulates_latency(self, mock_pipeline_store, initial_state):
        """Test that latency is accumulated"""
        initial_state["check_latency_ms"] = 100
        node = PipelineHealthNode(pipeline_store=mock_pipeline_store)
        result = await node.execute(initial_state)

        assert result["check_latency_ms"] >= 100


class TestPipelineFreshness:
    """Tests for freshness-based status determination"""

    @pytest.mark.asyncio
    async def test_custom_freshness_thresholds(self, mock_pipeline_store, initial_state):
        """Test custom freshness thresholds"""
        node = PipelineHealthNode(
            pipeline_store=mock_pipeline_store,
            max_freshness_hours=48.0,
            stale_threshold_hours=24.0,
        )
        assert node.max_freshness_hours == 48.0
        assert node.stale_threshold_hours == 24.0


class TestPipelineStatusFields:
    """Tests for pipeline status field population"""

    @pytest.mark.asyncio
    async def test_all_fields_populated(self, mock_pipeline_store, initial_state):
        """Test that all status fields are populated"""
        node = PipelineHealthNode(pipeline_store=mock_pipeline_store)
        result = await node.execute(initial_state)

        status = result["pipeline_statuses"][0]
        assert status["pipeline_name"] is not None
        assert status["last_run"] is not None
        assert status["last_success"] is not None
        assert status["rows_processed"] >= 0
        assert status["freshness_hours"] >= 0 or status["freshness_hours"] == -1
        assert status["status"] in ["healthy", "stale", "failed"]
