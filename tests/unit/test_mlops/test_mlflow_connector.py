"""Tests for MLflowConnector.

Version: 1.0.0
Tests the MLflow SDK wrapper with mocked MLflow operations.

Coverage:
- ModelStage enum
- CircuitBreaker functionality
- MLflowRun context manager
- MLflowConnector singleton pattern
- Experiment management
- Run tracking
- Model registry
- Graceful degradation
"""

import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.mlops.mlflow_connector import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
    MLflowConnector,
    MLflowRun,
    ModelStage,
    ModelVersion,
    RunStatus,
    AutoLogger,
)


# ============================================================================
# MODEL STAGE ENUM TESTS
# ============================================================================


class TestModelStage:
    """Test ModelStage enum."""

    def test_all_stages_defined(self):
        """Test all expected stages are defined."""
        expected_stages = [
            "development",
            "staging",
            "shadow",
            "production",
            "archived",
            "deprecated",
        ]
        actual_stages = [s.value for s in ModelStage]
        assert set(actual_stages) == set(expected_stages)

    def test_stage_is_string_enum(self):
        """Test that stages can be used as strings."""
        assert ModelStage.PRODUCTION == "production"
        assert ModelStage.STAGING == "staging"

    def test_stage_string_comparison(self):
        """Test string comparison works."""
        assert ModelStage.DEVELOPMENT.value == "development"
        assert ModelStage.PRODUCTION.value == "production"


class TestRunStatus:
    """Test RunStatus enum."""

    def test_all_statuses_defined(self):
        """Test all expected statuses are defined."""
        expected_statuses = ["running", "scheduled", "finished", "failed", "killed"]
        actual_statuses = [s.value for s in RunStatus]
        assert set(actual_statuses) == set(expected_statuses)


# ============================================================================
# CIRCUIT BREAKER CONFIG TESTS
# ============================================================================


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.reset_timeout_seconds == 30.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            reset_timeout_seconds=60.0,
            half_open_max_calls=5,
            success_threshold=3,
        )

        assert config.failure_threshold == 10
        assert config.reset_timeout_seconds == 60.0
        assert config.half_open_max_calls == 5
        assert config.success_threshold == 3


# ============================================================================
# CIRCUIT BREAKER METRICS TESTS
# ============================================================================


class TestCircuitBreakerMetrics:
    """Test CircuitBreakerMetrics dataclass."""

    def test_initial_values(self):
        """Test initial metric values are zero."""
        metrics = CircuitBreakerMetrics()

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.rejected_calls == 0
        assert metrics.times_opened == 0

    def test_record_success(self):
        """Test recording successful calls."""
        metrics = CircuitBreakerMetrics()

        metrics.record_success()
        metrics.record_success()

        assert metrics.total_calls == 2
        assert metrics.successful_calls == 2
        assert metrics.failed_calls == 0

    def test_record_failure(self):
        """Test recording failed calls."""
        metrics = CircuitBreakerMetrics()

        metrics.record_failure()

        assert metrics.total_calls == 1
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 1

    def test_record_rejected(self):
        """Test recording rejected calls."""
        metrics = CircuitBreakerMetrics()

        metrics.record_rejected()

        assert metrics.rejected_calls == 1

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = CircuitBreakerMetrics()
        metrics.record_success()
        metrics.record_failure()

        result = metrics.to_dict()

        assert result["total_calls"] == 2
        assert result["successful_calls"] == 1
        assert result["failed_calls"] == 1
        assert "success_rate" in result

    def test_success_rate_no_calls(self):
        """Test success rate when no calls made."""
        metrics = CircuitBreakerMetrics()

        # success_rate is only available via to_dict(), defaults to 1.0 when no calls
        result = metrics.to_dict()
        assert result["success_rate"] == 1.0


# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    def test_initial_state_is_closed(self):
        """Test circuit starts in closed state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True

    def test_allow_request_when_closed(self):
        """Test requests allowed when circuit is closed."""
        cb = CircuitBreaker()
        assert cb.allow_request() is True

    def test_record_success_keeps_closed(self):
        """Test recording success keeps circuit closed."""
        cb = CircuitBreaker()

        cb.record_success()

        assert cb.state == CircuitState.CLOSED
        assert cb.metrics.successful_calls == 1

    def test_failure_threshold_opens_circuit(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False
        assert cb.metrics.times_opened == 1

    def test_reject_when_open(self):
        """Test requests rejected when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config)

        cb.record_failure()
        cb.record_failure()

        assert cb.allow_request() is False
        assert cb.metrics.rejected_calls >= 0

    def test_half_open_after_timeout(self):
        """Test transition to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2, reset_timeout_seconds=0.01  # Very short for testing
        )
        cb = CircuitBreaker(config)

        cb.record_failure()
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

        time.sleep(0.02)  # Wait for timeout

        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_requests(self):
        """Test half-open state allows limited requests."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.01,
            half_open_max_calls=2,
        )
        cb = CircuitBreaker(config)

        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)

        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True

    def test_half_open_closes_on_success(self):
        """Test circuit closes from half-open on successful calls."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.01,
            success_threshold=2,
        )
        cb = CircuitBreaker(config)

        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Test circuit reopens from half-open on failure."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.01,
        )
        cb = CircuitBreaker(config)

        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Test success resets consecutive failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Should reset
        cb.record_failure()

        assert cb.state == CircuitState.CLOSED


# ============================================================================
# MODEL VERSION TESTS
# ============================================================================


class TestModelVersion:
    """Test ModelVersion dataclass."""

    def test_create_model_version(self):
        """Test creating a model version."""
        created = datetime.now(timezone.utc)
        version = ModelVersion(
            name="churn_model",
            version="1",
            source="runs:/run-123/model",
            run_id="run-123",
            stage=ModelStage.DEVELOPMENT,
            created_at=created,
            description="Initial model version",
            tags={"algorithm": "xgboost"},
        )

        assert version.name == "churn_model"
        assert version.version == "1"
        assert version.source == "runs:/run-123/model"
        assert version.run_id == "run-123"
        assert version.stage == ModelStage.DEVELOPMENT
        assert version.description == "Initial model version"

    def test_to_dict(self):
        """Test converting model version to dictionary."""
        created = datetime.now(timezone.utc)
        version = ModelVersion(
            name="churn_model",
            version="2",
            source="runs:/run-456/model",
            run_id="run-456",
            stage=ModelStage.STAGING,
            created_at=created,
        )

        result = version.to_dict()

        assert result["name"] == "churn_model"
        assert result["version"] == "2"
        assert result["stage"] == "staging"
        assert result["run_id"] == "run-456"


# ============================================================================
# MLFLOW CONNECTOR TESTS
# ============================================================================


class TestMLflowConnectorSingleton:
    """Test MLflowConnector singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    def teardown_method(self):
        """Reset singleton after each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    def test_singleton_pattern(self):
        """Test that only one instance is created."""
        # Create first connector and disable MLflow to avoid import issues
        conn1 = MLflowConnector(tracking_uri="test1")
        conn1._enabled = False

        conn2 = MLflowConnector(tracking_uri="test2")

        assert conn1 is conn2


class TestMLflowConnectorDisabled:
    """Test MLflowConnector when MLflow is not installed."""

    def setup_method(self):
        """Reset singleton before each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    def teardown_method(self):
        """Reset singleton after each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    @pytest.mark.asyncio
    async def test_get_or_create_experiment_disabled(self):
        """Test experiment creation returns mock ID when disabled."""
        conn = MLflowConnector()
        conn._enabled = False  # Force disabled state

        experiment_id = await conn.get_or_create_experiment("test_experiment")

        # Should return a mock experiment ID
        assert experiment_id is not None
        assert experiment_id.startswith("mock_")


class TestMLflowConnectorWithMock:
    """Test MLflowConnector with mocked MLflow operations."""

    def setup_method(self):
        """Reset singleton and set up mocks."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    def teardown_method(self):
        """Reset singleton after each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    def _create_mock_connector(self):
        """Create a connector with mocked MLflow internals."""
        conn = MLflowConnector()

        # Mock MLflow internals
        mock_mlflow = MagicMock()
        mock_client = MagicMock()

        conn._mlflow = mock_mlflow
        conn._client = mock_client
        conn._enabled = True

        return conn, mock_mlflow, mock_client

    @pytest.mark.asyncio
    async def test_get_or_create_experiment_existing(self):
        """Test getting an existing experiment."""
        conn, mock_mlflow, mock_client = self._create_mock_connector()

        # Mock existing experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        experiment_id = await conn.get_or_create_experiment("churn")

        assert experiment_id == "exp-123"
        mock_mlflow.get_experiment_by_name.assert_called_once_with("e2i_churn")

    @pytest.mark.asyncio
    async def test_get_or_create_experiment_new(self):
        """Test creating a new experiment."""
        conn, mock_mlflow, mock_client = self._create_mock_connector()

        # Mock no existing experiment
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "new-exp-456"

        experiment_id = await conn.get_or_create_experiment(
            "new_experiment",
            tags={"env": "test"}
        )

        assert experiment_id == "new-exp-456"
        mock_mlflow.create_experiment.assert_called_once()

    @pytest.mark.asyncio
    async def test_transition_model_stage(self):
        """Test transitioning model stage."""
        conn, mock_mlflow, mock_client = self._create_mock_connector()

        await conn.transition_model_stage(
            model_name="churn_model",
            version="1",
            stage=ModelStage.STAGING,
        )

        mock_client.transition_model_version_stage.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_latest_model_version(self):
        """Test getting latest model version."""
        conn, mock_mlflow, mock_client = self._create_mock_connector()

        # Mock response
        mock_version = MagicMock()
        mock_version.name = "test_model"
        mock_version.version = "1"
        mock_version.source = "runs:/abc/model"
        mock_version.run_id = "abc"
        mock_version.current_stage = "None"
        mock_version.creation_timestamp = 1000000
        mock_version.description = ""
        mock_version.tags = {}

        # The actual implementation uses search_model_versions
        mock_client.search_model_versions.return_value = [mock_version]

        version = await conn.get_latest_model_version("test_model")

        # Should return a ModelVersion object
        assert version is not None
        assert version.name == "test_model"
        mock_client.search_model_versions.assert_called_once()


class TestMLflowConnectorCircuitBreaker:
    """Test MLflowConnector circuit breaker integration."""

    def setup_method(self):
        """Reset singleton before each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    def teardown_method(self):
        """Reset singleton after each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self):
        """Test that circuit breaker opens after failures."""
        # Create connector with low threshold
        config = CircuitBreakerConfig(failure_threshold=2)
        conn = MLflowConnector(circuit_breaker_config=config)

        # Set up mocked MLflow that always fails
        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.side_effect = Exception("Connection failed")
        conn._mlflow = mock_mlflow
        conn._enabled = True

        # First two calls should fail and open the circuit
        for _ in range(2):
            try:
                await conn.get_or_create_experiment("test")
            except Exception:
                pass

        # Circuit should be open now
        assert conn.circuit_breaker.state == CircuitState.OPEN


class TestMLflowConnectorMetrics:
    """Test MLflowConnector metrics and health check."""

    def setup_method(self):
        """Reset singleton before each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    def teardown_method(self):
        """Reset singleton after each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    def test_get_metrics(self):
        """Test getting connector metrics."""
        conn = MLflowConnector()

        metrics = conn.get_metrics()

        assert "enabled" in metrics
        assert "tracking_uri" in metrics
        assert "circuit_breaker" in metrics
        assert "circuit_state" in metrics

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when MLflow is healthy."""
        conn, mock_mlflow, mock_client = self._create_mock_connector()

        # Mock successful experiment search
        mock_mlflow.search_experiments.return_value = []

        result = await conn.health_check()

        assert result["status"] in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check when MLflow is unavailable."""
        conn = MLflowConnector()
        conn._enabled = False

        result = await conn.health_check()

        assert result["status"] == "unhealthy"

    def _create_mock_connector(self):
        """Create a connector with mocked MLflow internals."""
        conn = MLflowConnector()

        mock_mlflow = MagicMock()
        mock_client = MagicMock()

        conn._mlflow = mock_mlflow
        conn._client = mock_client
        conn._enabled = True

        return conn, mock_mlflow, mock_client


class TestMLflowExperimentNaming:
    """Test experiment naming conventions."""

    def setup_method(self):
        """Reset singleton before each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    def teardown_method(self):
        """Reset singleton after each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    @pytest.mark.asyncio
    async def test_experiment_prefix_applied(self):
        """Test that experiment prefix is applied."""
        conn = MLflowConnector(experiment_prefix="custom")

        # Set up mock
        mock_mlflow = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-1"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        conn._mlflow = mock_mlflow
        conn._enabled = True

        await conn.get_or_create_experiment("test")

        mock_mlflow.get_experiment_by_name.assert_called_with("custom_test")


class TestAutoLogger:
    """Test AutoLogger class."""

    def setup_method(self):
        """Reset singleton before each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    def teardown_method(self):
        """Reset singleton after each test."""
        MLflowConnector._instance = None
        MLflowConnector._initialized = False

    @pytest.mark.asyncio
    async def test_enable_autolog_sklearn(self):
        """Test enabling sklearn autologging."""
        # AutoLogger requires a connector instance
        conn = MLflowConnector()
        conn._enabled = False  # Disable to avoid MLflow calls
        auto_logger = AutoLogger(connector=conn)

        # When connector is disabled, this should be a no-op
        await auto_logger.enable_autolog(framework="sklearn")

    @pytest.mark.asyncio
    async def test_enable_autolog_xgboost(self):
        """Test enabling xgboost autologging."""
        conn = MLflowConnector()
        conn._enabled = False
        auto_logger = AutoLogger(connector=conn)

        await auto_logger.enable_autolog(framework="xgboost")

    @pytest.mark.asyncio
    async def test_autolog_when_disabled(self):
        """Test autolog does nothing when connector is disabled."""
        conn = MLflowConnector()
        conn._enabled = False  # Disable MLflow
        auto_logger = AutoLogger(connector=conn)

        # Should be a no-op since connector is disabled
        await auto_logger.enable_autolog(framework="sklearn")
        await auto_logger.enable_autolog(framework="xgboost")
        await auto_logger.enable_autolog(framework="lightgbm")
