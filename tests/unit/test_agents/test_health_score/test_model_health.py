"""
Tests for Model Health Node
"""

import pytest

from src.agents.health_score.nodes.model_health import ModelHealthNode


class TestModelHealthNode:
    """Tests for ModelHealthNode"""

    @pytest.mark.asyncio
    async def test_healthy_models(self, mock_metrics_store, initial_state):
        """Test all healthy models"""
        node = ModelHealthNode(metrics_store=mock_metrics_store)
        result = await node.execute(initial_state)

        assert result["model_health_score"] == 1.0
        assert len(result["model_metrics"]) == 2
        for metrics in result["model_metrics"]:
            assert metrics["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_degraded_models(self, degraded_metrics_store, initial_state):
        """Test with degraded/unhealthy models"""
        node = ModelHealthNode(metrics_store=degraded_metrics_store)
        result = await node.execute(initial_state)

        # 1 healthy + 1 degraded (0.5) + 1 unhealthy (0) = 1.5/3 = 0.5
        assert result["model_health_score"] == 0.5

        # Check statuses
        status_map = {m["model_id"]: m["status"] for m in result["model_metrics"]}
        assert status_map["healthy_model"] == "healthy"
        assert status_map["degraded_model"] == "degraded"
        assert status_map["unhealthy_model"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_skips_for_non_model_scope(self, mock_metrics_store, initial_state):
        """Test that non-model scope skips model check"""
        initial_state["check_scope"] = "pipelines"
        node = ModelHealthNode(metrics_store=mock_metrics_store)
        result = await node.execute(initial_state)

        assert result["model_health_score"] == 1.0
        assert result["model_metrics"] == []

    @pytest.mark.asyncio
    async def test_includes_for_model_scope(self, mock_metrics_store, initial_state):
        """Test that models scope includes model check"""
        initial_state["check_scope"] = "models"
        node = ModelHealthNode(metrics_store=mock_metrics_store)
        result = await node.execute(initial_state)

        assert len(result["model_metrics"]) == 2

    @pytest.mark.asyncio
    async def test_no_store_returns_healthy(self, initial_state):
        """Test that no store returns healthy by default"""
        node = ModelHealthNode(metrics_store=None)
        result = await node.execute(initial_state)

        assert result["model_health_score"] == 1.0
        assert result["model_metrics"] == []

    @pytest.mark.asyncio
    async def test_accumulates_latency(self, mock_metrics_store, initial_state):
        """Test that latency is accumulated"""
        initial_state["check_latency_ms"] = 100
        node = ModelHealthNode(metrics_store=mock_metrics_store)
        result = await node.execute(initial_state)

        assert result["check_latency_ms"] >= 100


class TestModelHealthThresholds:
    """Tests for threshold-based status determination"""

    @pytest.mark.asyncio
    async def test_low_accuracy_triggers_issue(self, initial_state):
        """Test that low accuracy triggers degraded status"""
        from tests.unit.test_agents.test_health_score.conftest import MockMetricsStore

        store = MockMetricsStore(models=["low_accuracy_model"])
        store.set_model_metrics(
            "low_accuracy_model",
            {
                "accuracy": 0.65,  # Below 0.7 threshold
                "auc_roc": 0.85,
                "latency_p99": 200,
                "prediction_count": 500,
                "error_rate": 0.01,
            },
        )

        node = ModelHealthNode(metrics_store=store)
        result = await node.execute(initial_state)

        assert result["model_metrics"][0]["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_multiple_issues_trigger_unhealthy(self, initial_state):
        """Test that multiple issues trigger unhealthy status"""
        from tests.unit.test_agents.test_health_score.conftest import MockMetricsStore

        store = MockMetricsStore(models=["multi_issue_model"])
        store.set_model_metrics(
            "multi_issue_model",
            {
                "accuracy": 0.60,  # Below threshold
                "auc_roc": 0.55,  # Below threshold
                "latency_p99": 2000,  # Above threshold
                "prediction_count": 500,
                "error_rate": 0.01,
            },
        )

        node = ModelHealthNode(metrics_store=store)
        result = await node.execute(initial_state)

        assert result["model_metrics"][0]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_high_error_rate_triggers_issue(self, initial_state):
        """Test that high error rate triggers degraded status"""
        from tests.unit.test_agents.test_health_score.conftest import MockMetricsStore

        store = MockMetricsStore(models=["high_error_model"])
        store.set_model_metrics(
            "high_error_model",
            {
                "accuracy": 0.85,
                "auc_roc": 0.9,
                "latency_p99": 200,
                "prediction_count": 500,
                "error_rate": 0.1,  # Above 0.05 threshold
            },
        )

        node = ModelHealthNode(metrics_store=store)
        result = await node.execute(initial_state)

        assert result["model_metrics"][0]["status"] == "degraded"


class TestModelMetricsFields:
    """Tests for model metrics field population"""

    @pytest.mark.asyncio
    async def test_all_fields_populated(self, mock_metrics_store, initial_state):
        """Test that all metric fields are populated"""
        node = ModelHealthNode(metrics_store=mock_metrics_store)
        result = await node.execute(initial_state)

        metrics = result["model_metrics"][0]
        assert metrics["model_id"] is not None
        assert metrics["accuracy"] is not None
        assert metrics["precision"] is not None
        assert metrics["recall"] is not None
        assert metrics["f1_score"] is not None
        assert metrics["auc_roc"] is not None
        assert metrics["prediction_latency_p50_ms"] is not None
        assert metrics["prediction_latency_p99_ms"] is not None
        assert metrics["predictions_last_24h"] >= 0
        assert metrics["error_rate"] >= 0
        assert metrics["status"] in ["healthy", "degraded", "unhealthy"]
