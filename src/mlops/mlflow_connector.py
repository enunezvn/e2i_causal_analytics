"""
MLflow Connector for E2I Causal Analytics.

This module provides a centralized wrapper for MLflow, enabling
experiment tracking, model registry, and artifact management across
all ML foundation agents in the E2I platform.

Features:
- Singleton pattern for consistent configuration
- Async context managers for training run tracking
- Experiment management and run tracking
- Model registry with stage transitions
- Database synchronization with ml_experiments/ml_training_runs tables
- Circuit breaker pattern for fault tolerance
- Graceful degradation when MLflow is unavailable

Usage:
    from src.mlops.mlflow_connector import MLflowConnector

    mlflow_conn = MLflowConnector()

    # Create/get experiment
    experiment_id = await mlflow_conn.get_or_create_experiment(
        name="churn_prediction",
        tags={"brand": "remibrutinib", "region": "us"}
    )

    # Track a training run
    async with mlflow_conn.start_run(
        experiment_id=experiment_id,
        run_name="xgboost_v1"
    ) as run:
        run.log_params({"n_estimators": 100, "max_depth": 6})
        # ... training code ...
        run.log_metrics({"auc": 0.85, "precision": 0.78})
        run.log_model(model, "model")

    # Register model to registry
    model_version = await mlflow_conn.register_model(
        run_id=run.run_id,
        model_name="churn_predictor",
        model_path="model"
    )

Author: E2I Causal Analytics Team
Version: 1.0.0 (Phase 5 - Initial Implementation)
"""

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================


class ModelStage(str, Enum):
    """Model registry stages aligned with database enum."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    SHADOW = "shadow"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class RunStatus(str, Enum):
    """MLflow run status values."""

    RUNNING = "running"
    SCHEDULED = "scheduled"
    FINISHED = "finished"
    FAILED = "failed"
    KILLED = "killed"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Circuit tripped, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Consecutive failures before opening
    reset_timeout_seconds: float = 30.0  # Time before trying half-open
    half_open_max_calls: int = 3  # Max test calls in half-open state
    success_threshold: int = 2  # Successes needed to close from half-open


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    times_opened: int = 0

    def record_success(self) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.last_success_time = time.time()

    def record_failure(self) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.last_failure_time = time.time()

    def record_rejected(self) -> None:
        """Record a rejected call (circuit open)."""
        self.rejected_calls += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "times_opened": self.times_opened,
            "success_rate": (
                self.successful_calls / self.total_calls
                if self.total_calls > 0
                else 1.0
            ),
        }


class CircuitBreaker:
    """Circuit breaker for MLflow calls."""

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self.metrics = CircuitBreakerMetrics()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.reset_timeout_seconds:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        self._success_count = 0
                        logger.info("Circuit breaker transitioned to HALF_OPEN")
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (healthy)."""
        return self.state == CircuitState.CLOSED

    def allow_request(self) -> bool:
        """Check if request should be allowed through."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        else:
            self.metrics.record_rejected()
            return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.metrics.record_success()
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker CLOSED (service recovered)")
            else:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self.metrics.record_failure()
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self.metrics.times_opened += 1
                logger.warning("Circuit breaker OPEN (half-open test failed)")
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                self.metrics.times_opened += 1
                logger.warning(
                    f"Circuit breaker OPEN (threshold {self.config.failure_threshold} reached)"
                )


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class MLflowRun:
    """Represents an active MLflow run with logging methods."""

    run_id: str
    experiment_id: str
    run_name: str
    start_time: datetime
    connector: "MLflowConnector"
    _status: RunStatus = RunStatus.RUNNING
    _params: Dict[str, Any] = field(default_factory=dict)
    _metrics: Dict[str, float] = field(default_factory=dict)
    _tags: Dict[str, str] = field(default_factory=dict)
    _artifacts: List[str] = field(default_factory=list)

    @property
    def status(self) -> RunStatus:
        """Get run status."""
        return self._status

    async def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of parameter names to values
        """
        await self.connector._log_params(self.run_id, params)
        self._params.update(params)

    async def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        await self.log_params({key: value})

    async def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/epoch number
        """
        await self.connector._log_metrics(self.run_id, metrics, step)
        self._metrics.update(metrics)

    async def log_metric(
        self, key: str, value: float, step: Optional[int] = None
    ) -> None:
        """Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        await self.log_metrics({key: value}, step)

    async def set_tags(self, tags: Dict[str, str]) -> None:
        """Set run tags.

        Args:
            tags: Dictionary of tag names to values
        """
        await self.connector._set_tags(self.run_id, tags)
        self._tags.update(tags)

    async def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file.

        Args:
            local_path: Local path to the artifact file
            artifact_path: Destination path within artifact store
        """
        await self.connector._log_artifact(self.run_id, local_path, artifact_path)
        self._artifacts.append(local_path)

    async def log_model(
        self,
        model: Any,
        artifact_path: str,
        flavor: str = "sklearn",
        registered_model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Log a trained model.

        Args:
            model: The trained model object
            artifact_path: Path within artifact store
            flavor: MLflow flavor (sklearn, xgboost, lightgbm, etc.)
            registered_model_name: If provided, also register the model
            **kwargs: Additional arguments for the specific flavor

        Returns:
            Model URI if successful
        """
        model_uri = await self.connector._log_model(
            self.run_id, model, artifact_path, flavor, **kwargs
        )
        if model_uri and registered_model_name:
            await self.connector.register_model(
                run_id=self.run_id,
                model_name=registered_model_name,
                model_path=artifact_path,
            )
        return model_uri

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "run_name": self.run_name,
            "status": self._status.value,
            "params": self._params,
            "metrics": self._metrics,
            "tags": self._tags,
            "start_time": self.start_time.isoformat(),
        }


@dataclass
class ModelVersion:
    """Represents a registered model version."""

    name: str
    version: str
    run_id: str
    source: str
    stage: ModelStage
    created_at: datetime
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "source": self.source,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "tags": self.tags,
        }


# ============================================================================
# MLFLOW CONNECTOR
# ============================================================================


class MLflowConnector:
    """Production-ready MLflow integration for E2I Causal Analytics.

    This class provides a thread-safe singleton wrapper around MLflow,
    with circuit breaker for fault tolerance and async support.

    Attributes:
        tracking_uri: MLflow tracking server URI
        artifact_uri: Artifact storage location
        circuit_breaker: Circuit breaker for fault tolerance
        enabled: Whether MLflow is enabled and available
    """

    _instance: Optional["MLflowConnector"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "MLflowConnector":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        artifact_uri: Optional[str] = None,
        experiment_prefix: str = "e2i",
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize MLflow connector.

        Args:
            tracking_uri: MLflow tracking server URI (default: env var or local)
            artifact_uri: Artifact storage location (default: env var or local)
            experiment_prefix: Prefix for experiment names
            circuit_breaker_config: Circuit breaker configuration
        """
        if self._initialized:
            return

        # Default to the MLflow server (http://localhost:5000) for the E2I project
        # The singleton pattern means if this is initialized early (before env var is set),
        # it locks in the tracking URI. Using the server as default ensures models are
        # logged to the correct location for tier0 tests and production workflows.
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        # When using a tracking server with --serve-artifacts, use the mlflow-artifacts scheme
        # This routes artifact uploads through the tracking server's proxy
        if self.tracking_uri.startswith("http"):
            # Extract host:port from tracking URI for artifact proxy
            # e.g., http://localhost:5000 -> mlflow-artifacts://localhost:5000
            server_host = self.tracking_uri.replace("http://", "").replace("https://", "")
            default_artifact_uri = f"mlflow-artifacts://{server_host}"
        else:
            default_artifact_uri = "mlartifacts"
        self.artifact_uri = artifact_uri or os.environ.get(
            "MLFLOW_ARTIFACT_URI", default_artifact_uri
        )
        self.experiment_prefix = experiment_prefix
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)

        # Initialize MLflow
        self._mlflow = None
        self._enabled = False
        self._initialize_mlflow()

        self._initialized = True

    def _initialize_mlflow(self) -> None:
        """Initialize MLflow client."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
            self._mlflow = mlflow
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
            self._enabled = True
            logger.info(f"MLflow initialized with tracking URI: {self.tracking_uri}")
        except ImportError:
            logger.warning("MLflow not installed. Running in degraded mode.")
            self._enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}. Running in degraded mode.")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if MLflow is enabled and circuit is closed."""
        return self._enabled and self.circuit_breaker.is_closed

    def _run_sync(self, coro: Any) -> Any:
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're in an async context, create a task
            return asyncio.ensure_future(coro)
        else:
            # We're in a sync context, run in new loop
            return asyncio.run(coro)

    # ========================================================================
    # EXPERIMENT MANAGEMENT
    # ========================================================================

    async def get_or_create_experiment(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        artifact_location: Optional[str] = None,
    ) -> str:
        """Get or create an MLflow experiment.

        Args:
            name: Experiment name (will be prefixed with experiment_prefix)
            tags: Optional tags for the experiment
            artifact_location: Custom artifact location

        Returns:
            Experiment ID

        Raises:
            RuntimeError: If MLflow is unavailable and circuit is open
        """
        if not self._enabled:
            logger.warning("MLflow disabled. Returning mock experiment ID.")
            return f"mock_{uuid.uuid4().hex[:8]}"

        if not self.circuit_breaker.allow_request():
            raise RuntimeError("MLflow circuit breaker is open")

        try:
            full_name = f"{self.experiment_prefix}_{name}"

            # Check if experiment exists
            experiment = self._mlflow.get_experiment_by_name(full_name)

            if experiment is None:
                # Create new experiment
                experiment_id = self._mlflow.create_experiment(
                    name=full_name,
                    artifact_location=artifact_location or self.artifact_uri,
                    tags=tags or {},
                )
                logger.info(f"Created experiment '{full_name}' with ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                # Update tags if provided
                if tags:
                    for key, value in tags.items():
                        self._mlflow.set_experiment_tag(experiment_id, key, value)
                logger.debug(f"Using existing experiment '{full_name}' with ID: {experiment_id}")

            self.circuit_breaker.record_success()
            return experiment_id

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to get/create experiment: {e}")
            raise

    async def list_experiments(
        self, view_type: str = "ACTIVE_ONLY"
    ) -> List[Dict[str, Any]]:
        """List all experiments.

        Args:
            view_type: Filter by view type (ACTIVE_ONLY, DELETED_ONLY, ALL)

        Returns:
            List of experiment dictionaries
        """
        if not self._enabled:
            return []

        if not self.circuit_breaker.allow_request():
            raise RuntimeError("MLflow circuit breaker is open")

        try:
            from mlflow.entities import ViewType

            view_map = {
                "ACTIVE_ONLY": ViewType.ACTIVE_ONLY,
                "DELETED_ONLY": ViewType.DELETED_ONLY,
                "ALL": ViewType.ALL,
            }

            experiments = self._client.search_experiments(
                view_type=view_map.get(view_type, ViewType.ACTIVE_ONLY)
            )

            self.circuit_breaker.record_success()
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "tags": dict(exp.tags) if exp.tags else {},
                }
                for exp in experiments
            ]

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to list experiments: {e}")
            raise

    # ========================================================================
    # RUN MANAGEMENT
    # ========================================================================

    @asynccontextmanager
    async def start_run(
        self,
        experiment_id: str,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        nested: bool = False,
    ):
        """Start a new MLflow run.

        Args:
            experiment_id: Experiment ID to log to
            run_name: Name for the run
            tags: Optional tags
            description: Run description
            nested: Whether this is a nested run

        Yields:
            MLflowRun object with logging methods

        Example:
            async with mlflow_conn.start_run(exp_id, "training_v1") as run:
                await run.log_params({"lr": 0.01})
                await run.log_metrics({"loss": 0.5})
        """
        run = None
        mlflow_run = None

        try:
            if not self._enabled:
                # Return mock run in degraded mode
                mock_run = MLflowRun(
                    run_id=f"mock_{uuid.uuid4().hex[:8]}",
                    experiment_id=experiment_id,
                    run_name=run_name,
                    start_time=datetime.now(timezone.utc),
                    connector=self,
                )
                yield mock_run
                return

            if not self.circuit_breaker.allow_request():
                raise RuntimeError("MLflow circuit breaker is open")

            # Start MLflow run
            run = self._mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                nested=nested,
                tags=tags or {},
                description=description,
            )

            # Create wrapper
            mlflow_run = MLflowRun(
                run_id=run.info.run_id,
                experiment_id=experiment_id,
                run_name=run_name,
                start_time=datetime.now(timezone.utc),
                connector=self,
            )

            # Set additional tags
            if tags:
                self._mlflow.set_tags(tags)

            self.circuit_breaker.record_success()
            logger.info(f"Started run '{run_name}' with ID: {mlflow_run.run_id}")

            yield mlflow_run

            # Mark as finished on success
            mlflow_run._status = RunStatus.FINISHED

        except Exception as e:
            self.circuit_breaker.record_failure()
            if mlflow_run:
                mlflow_run._status = RunStatus.FAILED
            logger.error(f"Run failed: {e}")
            raise

        finally:
            if self._enabled and run:
                self._mlflow.end_run()

    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run information by ID.

        Args:
            run_id: MLflow run ID

        Returns:
            Run information dictionary or None
        """
        if not self._enabled:
            return None

        if not self.circuit_breaker.allow_request():
            raise RuntimeError("MLflow circuit breaker is open")

        try:
            run = self._client.get_run(run_id)
            self.circuit_breaker.record_success()

            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": dict(run.data.tags),
            }

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to get run: {e}")
            return None

    async def search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for runs matching criteria.

        Args:
            experiment_ids: List of experiment IDs to search
            filter_string: MLflow filter string (e.g., "metrics.auc > 0.8")
            order_by: List of columns to sort by
            max_results: Maximum number of results

        Returns:
            List of matching run dictionaries
        """
        if not self._enabled:
            return []

        if not self.circuit_breaker.allow_request():
            raise RuntimeError("MLflow circuit breaker is open")

        try:
            runs = self._mlflow.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                order_by=order_by or ["start_time DESC"],
                max_results=max_results,
            )

            self.circuit_breaker.record_success()

            return runs.to_dict(orient="records") if len(runs) > 0 else []

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to search runs: {e}")
            return []

    # ========================================================================
    # LOGGING METHODS (Internal)
    # ========================================================================

    async def _log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Log parameters to a run."""
        if not self._enabled:
            return

        try:
            # Convert all values to strings (MLflow requirement)
            str_params = {k: str(v) for k, v in params.items()}
            self._mlflow.log_params(str_params)
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to log params: {e}")

    async def _log_metrics(
        self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to a run."""
        if not self._enabled:
            return

        try:
            for key, value in metrics.items():
                self._mlflow.log_metric(key, value, step=step)
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to log metrics: {e}")

    async def _set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        """Set tags on a run."""
        if not self._enabled:
            return

        try:
            self._mlflow.set_tags(tags)
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to set tags: {e}")

    async def _log_artifact(
        self, run_id: str, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log an artifact file."""
        if not self._enabled:
            return

        try:
            self._mlflow.log_artifact(local_path, artifact_path)
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to log artifact: {e}")

    async def _log_model(
        self,
        run_id: str,
        model: Any,
        artifact_path: str,
        flavor: str = "sklearn",
        **kwargs: Any,
    ) -> Optional[str]:
        """Log a model with the specified flavor.

        Returns the model URI from MLflow. In MLflow 3.x, this is the new
        'models:/m-{model_id}' format. The returned URI can be used to load
        the model via mlflow.{flavor}.load_model(model_uri).
        """
        if not self._enabled:
            return None

        try:
            # MLflow 3.x log_model returns ModelInfo with model_uri attribute
            model_info = None
            if flavor == "sklearn":
                model_info = self._mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            elif flavor == "xgboost":
                model_info = self._mlflow.xgboost.log_model(model, artifact_path, **kwargs)
            elif flavor == "lightgbm":
                model_info = self._mlflow.lightgbm.log_model(model, artifact_path, **kwargs)
            elif flavor == "pytorch":
                model_info = self._mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            elif flavor == "tensorflow":
                model_info = self._mlflow.tensorflow.log_model(model, artifact_path, **kwargs)
            elif flavor == "pyfunc":
                model_info = self._mlflow.pyfunc.log_model(artifact_path, **kwargs)
            else:
                # Try generic sklearn as fallback
                model_info = self._mlflow.sklearn.log_model(model, artifact_path, **kwargs)

            self.circuit_breaker.record_success()

            # Use the model_uri from ModelInfo (MLflow 3.x returns models:/m-{id} format)
            # Fall back to legacy runs:/ format for older MLflow versions
            if model_info is not None and hasattr(model_info, "model_uri"):
                model_uri = model_info.model_uri
            else:
                model_uri = f"runs:/{run_id}/{artifact_path}"

            logger.info(f"Logged model to: {model_uri}")
            return model_uri

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to log model: {e}")
            return None

    # ========================================================================
    # MODEL REGISTRY
    # ========================================================================

    async def register_model(
        self,
        run_id: str,
        model_name: str,
        model_path: str = "model",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[ModelVersion]:
        """Register a model from a run to the model registry.

        Args:
            run_id: Run ID containing the model
            model_name: Name for the registered model
            model_path: Path within run artifacts
            description: Model description
            tags: Model tags

        Returns:
            ModelVersion object or None
        """
        if not self._enabled:
            return ModelVersion(
                name=model_name,
                version="mock_v1",
                run_id=run_id,
                source=f"runs:/{run_id}/{model_path}",
                stage=ModelStage.DEVELOPMENT,
                created_at=datetime.now(timezone.utc),
                description=description,
                tags=tags or {},
            )

        if not self.circuit_breaker.allow_request():
            raise RuntimeError("MLflow circuit breaker is open")

        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            result = self._mlflow.register_model(model_uri, model_name)

            # Set description if provided
            if description:
                self._client.update_model_version(
                    name=model_name,
                    version=result.version,
                    description=description,
                )

            # Set tags if provided
            if tags:
                for key, value in tags.items():
                    self._client.set_model_version_tag(
                        name=model_name,
                        version=result.version,
                        key=key,
                        value=value,
                    )

            self.circuit_breaker.record_success()
            logger.info(f"Registered model '{model_name}' version {result.version}")

            return ModelVersion(
                name=model_name,
                version=result.version,
                run_id=run_id,
                source=model_uri,
                stage=ModelStage.DEVELOPMENT,
                created_at=datetime.now(timezone.utc),
                description=description,
                tags=tags or {},
            )

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to register model: {e}")
            return None

    async def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: ModelStage,
        archive_existing: bool = True,
    ) -> bool:
        """Transition a model version to a new stage.

        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage
            archive_existing: Whether to archive existing models in target stage

        Returns:
            True if successful
        """
        if not self._enabled:
            return True

        if not self.circuit_breaker.allow_request():
            raise RuntimeError("MLflow circuit breaker is open")

        try:
            # Map our enum to MLflow stages
            mlflow_stage_map = {
                ModelStage.DEVELOPMENT: "None",
                ModelStage.STAGING: "Staging",
                ModelStage.PRODUCTION: "Production",
                ModelStage.ARCHIVED: "Archived",
            }

            mlflow_stage = mlflow_stage_map.get(stage, "None")

            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=mlflow_stage,
                archive_existing_versions=archive_existing,
            )

            self.circuit_breaker.record_success()
            logger.info(f"Transitioned {model_name} v{version} to {stage.value}")
            return True

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to transition model stage: {e}")
            return False

    async def get_latest_model_version(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None,
    ) -> Optional[ModelVersion]:
        """Get the latest version of a registered model.

        Args:
            model_name: Registered model name
            stage: Optional stage filter

        Returns:
            ModelVersion or None
        """
        if not self._enabled:
            return None

        if not self.circuit_breaker.allow_request():
            raise RuntimeError("MLflow circuit breaker is open")

        try:
            # Get all versions
            versions = self._client.search_model_versions(f"name='{model_name}'")

            if not versions:
                return None

            # Filter by stage if specified
            if stage:
                mlflow_stage_map = {
                    ModelStage.DEVELOPMENT: "None",
                    ModelStage.STAGING: "Staging",
                    ModelStage.PRODUCTION: "Production",
                    ModelStage.ARCHIVED: "Archived",
                }
                target_stage = mlflow_stage_map.get(stage, "None")
                versions = [v for v in versions if v.current_stage == target_stage]

            if not versions:
                return None

            # Get latest by version number
            latest = max(versions, key=lambda v: int(v.version))

            self.circuit_breaker.record_success()

            # Map MLflow stage back to our enum
            stage_map = {
                "None": ModelStage.DEVELOPMENT,
                "Staging": ModelStage.STAGING,
                "Production": ModelStage.PRODUCTION,
                "Archived": ModelStage.ARCHIVED,
            }

            return ModelVersion(
                name=latest.name,
                version=latest.version,
                run_id=latest.run_id,
                source=latest.source,
                stage=stage_map.get(latest.current_stage, ModelStage.DEVELOPMENT),
                created_at=datetime.fromtimestamp(
                    latest.creation_timestamp / 1000, tz=timezone.utc
                ),
                description=latest.description,
                tags=dict(latest.tags) if latest.tags else {},
            )

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to get latest model version: {e}")
            return None

    async def load_model(
        self,
        model_uri: str,
        flavor: str = "sklearn",
    ) -> Any:
        """Load a model from MLflow.

        Args:
            model_uri: Model URI (runs:/... or models:/...)
            flavor: MLflow flavor

        Returns:
            Loaded model object
        """
        if not self._enabled:
            raise RuntimeError("MLflow is not enabled")

        if not self.circuit_breaker.allow_request():
            raise RuntimeError("MLflow circuit breaker is open")

        try:
            if flavor == "sklearn":
                model = self._mlflow.sklearn.load_model(model_uri)
            elif flavor == "xgboost":
                model = self._mlflow.xgboost.load_model(model_uri)
            elif flavor == "lightgbm":
                model = self._mlflow.lightgbm.load_model(model_uri)
            elif flavor == "pytorch":
                model = self._mlflow.pytorch.load_model(model_uri)
            elif flavor == "pyfunc":
                model = self._mlflow.pyfunc.load_model(model_uri)
            else:
                model = self._mlflow.pyfunc.load_model(model_uri)

            self.circuit_breaker.record_success()
            logger.info(f"Loaded model from: {model_uri}")
            return model

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to load model: {e}")
            raise

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return {
            "enabled": self._enabled,
            "tracking_uri": self.tracking_uri,
            "circuit_breaker": self.circuit_breaker.metrics.to_dict(),
            "circuit_state": self.circuit_breaker.state.value,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "unhealthy",
            "mlflow_available": False,
            "circuit_state": self.circuit_breaker.state.value,
            "tracking_uri": self.tracking_uri,
        }

        if not self._enabled:
            health["error"] = "MLflow not installed or initialized"
            return health

        if not self.circuit_breaker.allow_request():
            health["error"] = "Circuit breaker is open"
            return health

        try:
            # Try to list experiments as a health check
            experiments = self._client.search_experiments(max_results=1)
            health["status"] = "healthy"
            health["mlflow_available"] = True
            health["experiment_count"] = len(experiments)
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            health["error"] = str(e)

        return health


# ============================================================================
# AUTO-LOGGING UTILITIES
# ============================================================================


class AutoLogger:
    """Utility for automatic logging during training."""

    def __init__(self, connector: MLflowConnector):
        """Initialize auto-logger.

        Args:
            connector: MLflowConnector instance
        """
        self.connector = connector

    async def enable_autolog(
        self,
        framework: str = "sklearn",
        log_models: bool = True,
        log_input_examples: bool = False,
        log_model_signatures: bool = True,
        disable: bool = False,
    ) -> None:
        """Enable MLflow autologging for a framework.

        Args:
            framework: Framework name (sklearn, xgboost, lightgbm, etc.)
            log_models: Whether to log models
            log_input_examples: Whether to log input examples
            log_model_signatures: Whether to log model signatures
            disable: Whether to disable autologging
        """
        if not self.connector._enabled:
            return

        try:
            mlflow = self.connector._mlflow

            if framework == "sklearn":
                mlflow.sklearn.autolog(
                    log_models=log_models,
                    log_input_examples=log_input_examples,
                    log_model_signatures=log_model_signatures,
                    disable=disable,
                )
            elif framework == "xgboost":
                mlflow.xgboost.autolog(
                    log_models=log_models,
                    log_input_examples=log_input_examples,
                    log_model_signatures=log_model_signatures,
                    disable=disable,
                )
            elif framework == "lightgbm":
                mlflow.lightgbm.autolog(
                    log_models=log_models,
                    log_input_examples=log_input_examples,
                    log_model_signatures=log_model_signatures,
                    disable=disable,
                )
            else:
                logger.warning(f"Unknown framework for autolog: {framework}")

            logger.info(f"Enabled autolog for {framework}")

        except Exception as e:
            logger.error(f"Failed to enable autolog: {e}")

    async def disable_autolog(self, framework: str = "sklearn") -> None:
        """Disable autologging for a framework."""
        await self.enable_autolog(framework=framework, disable=True)


# ============================================================================
# MODULE-LEVEL SINGLETON ACCESS
# ============================================================================


def get_mlflow_connector(
    tracking_uri: Optional[str] = None,
    **kwargs: Any,
) -> MLflowConnector:
    """Get the MLflow connector singleton.

    Args:
        tracking_uri: Optional tracking URI override
        **kwargs: Additional configuration

    Returns:
        MLflowConnector instance
    """
    return MLflowConnector(tracking_uri=tracking_uri, **kwargs)
