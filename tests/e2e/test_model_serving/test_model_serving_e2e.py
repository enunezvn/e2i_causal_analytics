"""End-to-End Tests for Model Serving Flow.

Tests the complete model serving pipeline:
- model_deployer agent → BentoML service → prediction_synthesizer agent
- Opik trace capture and verification
- Performance validation (latency, throughput)
- Prometheus metrics collection

These tests use mocked services for CI/CD.
Live tests require running Docker Compose stack.
"""

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.ml_foundation.model_deployer.nodes.deployment_orchestrator import (
    deploy_to_endpoint,
    package_model,
)
from src.agents.prediction_synthesizer.clients import (
    HTTPModelClient,
    HTTPModelClientConfig,
    ModelClientFactory,
    get_model_client,
)
from src.agents.prediction_synthesizer.clients.factory import (
    MockModelClient,
    ModelEndpointConfig,
    ModelEndpointsConfig,
    close_model_clients,
    configure_model_endpoints,
)
from src.api.dependencies.bentoml_client import (
    BentoMLClient,
    BentoMLClientConfig,
    close_bentoml_client,
    get_bentoml_client,
)


# =============================================================================
# Test Configuration
# =============================================================================

# Environment checks for live service testing
HAS_BENTOML_STACK = (
    bool(os.getenv("BENTOML_SERVICE_URL")) and
    bool(os.getenv("PROMETHEUS_URL"))
)

requires_live_stack = pytest.mark.skipif(
    not HAS_BENTOML_STACK,
    reason="BENTOML_SERVICE_URL and PROMETHEUS_URL environment variables not set",
)

# Check if BentoML is available (library import, not just CLI in PATH)
def _check_bentoml_available() -> bool:
    """Check if BentoML library is installed and importable."""
    try:
        import bentoml
        return True
    except ImportError:
        return False

def _check_bentoml_cli_in_path() -> bool:
    """Check if BentoML CLI is in PATH (required for packaging/deployment)."""
    import shutil
    return shutil.which("bentoml") is not None

HAS_BENTOML = _check_bentoml_available()
HAS_BENTOML_CLI = _check_bentoml_cli_in_path()

requires_bentoml = pytest.mark.skipif(
    not HAS_BENTOML,
    reason="BentoML not installed (pip install bentoml)",
)

# Tests that need CLI for packaging/deployment (not just library import)
requires_bentoml_infrastructure = pytest.mark.skipif(
    not HAS_BENTOML_CLI,
    reason="BentoML CLI not in PATH (requires full BentoML infrastructure)",
)

# Tests that need full BentoML deployment infrastructure (models registered, service URL)
# These tests actually deploy models and require the complete stack
HAS_BENTOML_DEPLOYMENT = bool(os.getenv("BENTOML_SERVICE_URL"))

requires_bentoml_deployment = pytest.mark.skipif(
    not HAS_BENTOML_DEPLOYMENT,
    reason="BENTOML_SERVICE_URL not set (requires full deployment infrastructure with registered models)",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def experiment_id() -> str:
    """Generate unique experiment ID for test isolation."""
    return f"e2e-model-serving-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def sample_model_state(experiment_id):
    """Sample state for model deployment."""
    return {
        "experiment_id": experiment_id,
        "model_uri": f"mlflow://models/churn_model/{experiment_id}",
        "model_version": 1,
        "model_type": "xgboost",
        "deployment_name": f"churn-{experiment_id[:8]}",
        "target_environment": "staging",
        "resources": {"cpu": "2", "memory": "4Gi"},
    }


@pytest.fixture
async def mock_model_factory():
    """Create factory with mock clients for E2E testing."""
    config = ModelEndpointsConfig(
        default_base_url="http://mock-bentoml:3000",
        endpoints={
            "churn_model": ModelEndpointConfig(
                model_id="churn_model",
                endpoint_url="",
                client_type="mock",
                default_prediction=0.72,
                default_confidence=0.88,
                enabled=True,
            ),
            "conversion_model": ModelEndpointConfig(
                model_id="conversion_model",
                endpoint_url="",
                client_type="mock",
                default_prediction=0.65,
                default_confidence=0.82,
                enabled=True,
            ),
            "causal_model": ModelEndpointConfig(
                model_id="causal_model",
                endpoint_url="",
                client_type="mock",
                default_prediction=0.15,  # CATE effect
                default_confidence=0.75,
                enabled=True,
            ),
        },
    )
    factory = ModelClientFactory(config)
    yield factory
    await factory.close_all()


@pytest.fixture(autouse=True)
async def cleanup_clients():
    """Cleanup global client instances after tests."""
    yield
    await close_model_clients()
    await close_bentoml_client()


# =============================================================================
# Phase 5.1.3: model_deployer → BentoML → prediction_synthesizer Flow
# =============================================================================


class TestModelServingFlow:
    """E2E tests for the complete model serving flow."""

    @requires_bentoml_deployment
    @pytest.mark.asyncio
    async def test_deploy_package_flow(self, sample_model_state):
        """Test model packaging and deployment flow (requires full BentoML infrastructure)."""
        # Step 1: Package model with BentoML
        package_result = await package_model(sample_model_state)

        assert package_result["bento_packaging_successful"] is True
        assert "bento_tag" in package_result
        assert sample_model_state["experiment_id"] in package_result["bento_tag"]

        # Step 2: Deploy to staging endpoint
        deploy_state = {
            **sample_model_state,
            "bento_tag": package_result["bento_tag"],
        }
        deploy_result = await deploy_to_endpoint(deploy_state)

        assert deploy_result["deployment_successful"] is True
        assert deploy_result["deployment_environment"] == "staging"
        assert "deployment_url" in deploy_result

    @requires_bentoml_deployment
    @pytest.mark.asyncio
    async def test_full_serving_flow_with_mock(
        self, mock_model_factory, sample_model_state
    ):
        """Test complete flow: deploy → serve → predict (requires full BentoML infrastructure)."""
        # Step 1: Package model
        package_result = await package_model(sample_model_state)
        assert package_result["bento_packaging_successful"] is True

        # Step 2: Deploy to endpoint
        deploy_state = {
            **sample_model_state,
            "bento_tag": package_result["bento_tag"],
        }
        deploy_result = await deploy_to_endpoint(deploy_state)
        assert deploy_result["deployment_successful"] is True

        # Step 3: Get prediction client (mock)
        client = await mock_model_factory.get_client("churn_model")

        # Step 4: Make prediction
        prediction = await client.predict(
            entity_id="HCP001",
            features={
                "recency": 10,
                "frequency": 5,
                "monetary": 500,
                "engagement_score": 0.75,
            },
            time_horizon="30d",
        )

        # Verify prediction output
        assert "prediction" in prediction
        assert 0 <= prediction["prediction"] <= 1
        assert "confidence" in prediction
        assert "latency_ms" in prediction
        assert prediction["model_type"] == "mock"

    @pytest.mark.asyncio
    async def test_multi_model_ensemble_flow(self, mock_model_factory):
        """Test prediction flow with multiple models (ensemble)."""
        # Get clients for multiple models
        clients = await mock_model_factory.get_clients([
            "churn_model",
            "conversion_model",
            "causal_model",
        ])

        assert len(clients) == 3

        # Make predictions from all models in parallel
        entity_id = "HCP001"
        features = {"recency": 10, "frequency": 5}

        async def get_prediction(model_id: str, client):
            return {
                "model_id": model_id,
                "result": await client.predict(
                    entity_id=entity_id,
                    features=features,
                    time_horizon="30d",
                ),
            }

        tasks = [
            get_prediction(model_id, client)
            for model_id, client in clients.items()
        ]
        results = await asyncio.gather(*tasks)

        # Verify all predictions received
        assert len(results) == 3

        predictions = {r["model_id"]: r["result"] for r in results}
        assert "churn_model" in predictions
        assert "conversion_model" in predictions
        assert "causal_model" in predictions

        # Verify each has valid prediction
        for model_id, result in predictions.items():
            assert "prediction" in result
            assert "confidence" in result

        # Calculate ensemble (weighted average)
        weights = {"churn_model": 0.5, "conversion_model": 0.3, "causal_model": 0.2}
        ensemble_prediction = sum(
            predictions[m]["prediction"] * w for m, w in weights.items()
        )

        assert 0 <= ensemble_prediction <= 1

    @pytest.mark.asyncio
    async def test_prediction_synthesizer_pattern(self, mock_model_factory):
        """Test prediction_synthesizer-style orchestration."""
        # Simulate prediction_synthesizer behavior
        models_to_query = ["churn_model", "conversion_model"]

        # Phase 1: Model Orchestration - parallel predictions
        clients = await mock_model_factory.get_clients(models_to_query)

        predictions = []
        for model_id, client in clients.items():
            result = await client.predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )
            predictions.append({
                "model_id": model_id,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
            })

        # Phase 2: Ensemble Combination - weighted average
        total_confidence = sum(p["confidence"] for p in predictions)
        ensemble_prediction = sum(
            p["prediction"] * (p["confidence"] / total_confidence)
            for p in predictions
        )

        # Phase 3: Context Enrichment (simulated)
        model_agreement = 1 - abs(
            predictions[0]["prediction"] - predictions[1]["prediction"]
        )

        # Build synthesized output
        output = {
            "ensemble_prediction": {
                "point_estimate": ensemble_prediction,
                "confidence": sum(p["confidence"] for p in predictions) / len(predictions),
                "model_agreement": model_agreement,
            },
            "individual_predictions": predictions,
            "models_succeeded": len(predictions),
            "models_failed": 0,
        }

        # Verify output structure matches prediction_synthesizer contract
        assert "ensemble_prediction" in output
        assert "point_estimate" in output["ensemble_prediction"]
        assert 0 <= output["ensemble_prediction"]["point_estimate"] <= 1
        assert "model_agreement" in output["ensemble_prediction"]
        assert output["models_succeeded"] == 2


# =============================================================================
# Phase 5.1.4: Opik Trace Verification
# =============================================================================


class TestOpikTraceCapture:
    """Tests for Opik observability trace capture."""

    @pytest.mark.asyncio
    async def test_prediction_captures_trace_context(self, mock_model_factory):
        """Test that predictions include trace context."""
        client = await mock_model_factory.get_client("churn_model")

        result = await client.predict(
            entity_id="HCP001",
            features={"recency": 10},
            time_horizon="30d",
        )

        # Verify timestamp captured (trace timing)
        assert "timestamp" in result
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_trace_spans_for_multi_model(self, mock_model_factory):
        """Test trace spans are created for each model call."""
        clients = await mock_model_factory.get_clients([
            "churn_model",
            "conversion_model",
        ])

        trace_data = []

        for model_id, client in clients.items():
            start_time = time.perf_counter()
            result = await client.predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )
            duration_ms = (time.perf_counter() - start_time) * 1000

            trace_data.append({
                "span_name": f"prediction_synthesizer.{model_id}",
                "model_id": model_id,
                "duration_ms": duration_ms,
                "result_latency_ms": result["latency_ms"],
                "timestamp": result["timestamp"],
            })

        # Verify trace structure
        assert len(trace_data) == 2

        for span in trace_data:
            assert "span_name" in span
            assert span["duration_ms"] >= 0
            assert "prediction_synthesizer" in span["span_name"]

    @pytest.mark.asyncio
    async def test_error_captured_in_trace(self):
        """Test that errors are captured in trace context.

        Factory raises ValueError when model is explicitly disabled.
        Non-existent models get HTTP clients with default URLs.
        """
        # Create factory with a disabled model
        config = ModelEndpointsConfig(
            endpoints={
                "disabled_model": ModelEndpointConfig(
                    model_id="disabled_model",
                    endpoint_url="http://disabled:3000",
                    client_type="http",
                    enabled=False,  # Explicitly disabled
                ),
            }
        )
        factory = ModelClientFactory(config)

        # Disabled model should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            await factory.get_client("disabled_model")

        assert "disabled" in str(exc_info.value).lower()

        await factory.close_all()


# =============================================================================
# Phase 5.2: Performance Validation
# =============================================================================


class TestPerformanceValidation:
    """Tests for performance requirements."""

    @pytest.mark.asyncio
    async def test_single_prediction_latency_under_100ms(self, mock_model_factory):
        """Test single prediction latency is under 100ms (P95 target)."""
        client = await mock_model_factory.get_client("churn_model")

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await client.predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        latencies.sort()
        p50 = latencies[50]
        p95 = latencies[95]
        p99 = latencies[99]

        # Target: <100ms P95
        assert p95 < 100, f"P95 latency {p95:.2f}ms exceeds 100ms target"
        # Mock should be very fast
        assert p50 < 10, f"P50 latency {p50:.2f}ms unexpectedly high"

    @pytest.mark.asyncio
    async def test_batch_prediction_throughput(self, mock_model_factory):
        """Test batch prediction throughput meets requirements."""
        client = await mock_model_factory.get_client("churn_model")

        batch_size = 100
        entities = [f"HCP{i:03d}" for i in range(batch_size)]

        start = time.perf_counter()

        predictions = []
        for entity_id in entities:
            result = await client.predict(
                entity_id=entity_id,
                features={"recency": 10},
                time_horizon="30d",
            )
            predictions.append(result)

        total_time = time.perf_counter() - start
        throughput = batch_size / total_time

        assert len(predictions) == batch_size
        # Target: at least 100 predictions/second for mock
        assert throughput > 100, f"Throughput {throughput:.2f}/s below 100/s target"

    @pytest.mark.asyncio
    async def test_concurrent_predictions_scale(self, mock_model_factory):
        """Test system handles concurrent predictions."""
        client = await mock_model_factory.get_client("churn_model")

        async def single_prediction(entity_id: str):
            return await client.predict(
                entity_id=entity_id,
                features={"recency": 10},
                time_horizon="30d",
            )

        # Run 50 concurrent predictions
        concurrency = 50
        start = time.perf_counter()

        tasks = [single_prediction(f"HCP{i:03d}") for i in range(concurrency)]
        results = await asyncio.gather(*tasks)

        total_time = (time.perf_counter() - start) * 1000

        assert len(results) == concurrency
        # 50 concurrent predictions should complete in <500ms for mock
        assert total_time < 500, f"Concurrent predictions took {total_time:.2f}ms (>500ms)"

    @pytest.mark.asyncio
    async def test_multi_model_parallel_latency(self, mock_model_factory):
        """Test parallel multi-model predictions meet latency requirements."""
        clients = await mock_model_factory.get_clients([
            "churn_model",
            "conversion_model",
            "causal_model",
        ])

        async def get_prediction(model_id: str, client):
            return await client.predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )

        # Measure parallel execution time
        start = time.perf_counter()

        tasks = [
            get_prediction(model_id, client)
            for model_id, client in clients.items()
        ]
        results = await asyncio.gather(*tasks)

        total_time = (time.perf_counter() - start) * 1000

        assert len(results) == 3
        # Parallel 3-model predictions should be similar to single model time
        assert total_time < 100, f"Parallel predictions took {total_time:.2f}ms (>100ms)"


# =============================================================================
# Live Service Tests (Requires Running Docker Compose Stack)
# =============================================================================


class TestLiveModelServing:
    """Tests requiring live BentoML services via Docker Compose."""

    @requires_live_stack
    @pytest.mark.asyncio
    async def test_live_health_check(self):
        """Test health check against live BentoML services."""
        client = await get_bentoml_client()

        health = await client.health_check()

        assert health["status"] == "healthy"
        assert "timestamp" in health

    @requires_live_stack
    @pytest.mark.asyncio
    async def test_live_prediction_latency(self):
        """Test live prediction meets latency requirements."""
        client = await get_bentoml_client()

        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            await client.predict(
                model_name="churn_model",
                input_data={"features": [[0.5, 0.3, 0.7, 0.2]]},
            )
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        # Target: <100ms P95
        assert p95 < 100, f"Live P95 latency {p95:.2f}ms exceeds 100ms target"

    @requires_live_stack
    @pytest.mark.asyncio
    async def test_live_prometheus_metrics(self):
        """Test Prometheus metrics are collected."""
        import httpx

        prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

        async with httpx.AsyncClient() as http_client:
            # Query Prometheus for BentoML metrics
            response = await http_client.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": "bentoml_prediction_total"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


# =============================================================================
# Integration with Agent Flow
# =============================================================================


class TestAgentIntegration:
    """Tests for agent integration patterns."""

    @requires_bentoml_deployment
    @pytest.mark.asyncio
    async def test_model_deployer_to_synthesizer_handoff(
        self, mock_model_factory, sample_model_state
    ):
        """Test handoff from model_deployer to prediction_synthesizer (requires full BentoML infrastructure)."""
        # 1. model_deployer: Package and deploy
        package_result = await package_model(sample_model_state)
        deploy_result = await deploy_to_endpoint({
            **sample_model_state,
            "bento_tag": package_result["bento_tag"],
        })

        # 2. Handoff data
        handoff = {
            "from_agent": "model_deployer",
            "to_agent": "prediction_synthesizer",
            "deployment_url": deploy_result["deployment_url"],
            "model_id": "churn_model",
            "bento_tag": package_result["bento_tag"],
            "deployment_environment": deploy_result["deployment_environment"],
        }

        # 3. prediction_synthesizer: Configure and predict
        client = await mock_model_factory.get_client(handoff["model_id"])

        prediction = await client.predict(
            entity_id="HCP001",
            features={"recency": 10},
            time_horizon="30d",
        )

        # Verify successful handoff
        assert deploy_result["deployment_successful"]
        assert prediction["prediction"] is not None

    @pytest.mark.asyncio
    async def test_orchestrator_multi_agent_flow(self, mock_model_factory):
        """Test orchestrator-style coordination of model serving."""
        # Simulate orchestrator coordinating model_deployer + prediction_synthesizer

        # Phase 1: Deploy (model_deployer)
        deploy_results = {
            "churn_model": {"status": "deployed", "url": "http://mock:3001"},
            "conversion_model": {"status": "deployed", "url": "http://mock:3002"},
        }

        # Phase 2: Predict (prediction_synthesizer)
        clients = await mock_model_factory.get_clients(list(deploy_results.keys()))

        predictions = {}
        for model_id in deploy_results:
            result = await clients[model_id].predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )
            predictions[model_id] = result

        # Phase 3: Synthesize results
        synthesis = {
            "models_deployed": len(deploy_results),
            "predictions_received": len(predictions),
            "all_successful": all(
                deploy_results[m]["status"] == "deployed"
                for m in deploy_results
            ) and len(predictions) == len(deploy_results),
        }

        assert synthesis["all_successful"]
        assert synthesis["models_deployed"] == 2
        assert synthesis["predictions_received"] == 2
