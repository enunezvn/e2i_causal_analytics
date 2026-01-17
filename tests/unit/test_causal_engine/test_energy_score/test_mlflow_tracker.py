"""
Tests for src/causal_engine/energy_score/mlflow_tracker.py

Covers:
- ExperimentContext dataclass
- EnergyScoreMLflowTracker class
  - __init__
  - _check_mlflow
  - start_selection_run context manager
  - log_selection_result
  - _log_to_mlflow
  - _log_to_database
  - get_selection_comparison
  - get_estimator_performance
- create_tracker convenience function
"""

import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from uuid import uuid4

import numpy as np
import pytest

from src.causal_engine.energy_score.mlflow_tracker import (
    ExperimentContext,
    EnergyScoreMLflowTracker,
    create_tracker,
)
from src.causal_engine.energy_score.estimator_selector import (
    EstimatorResult,
    EstimatorType,
    SelectionResult,
    SelectionStrategy,
)
from src.causal_engine.energy_score.score_calculator import EnergyScoreResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_energy_score_result() -> EnergyScoreResult:
    """Create a sample EnergyScoreResult for testing."""
    return EnergyScoreResult(
        estimator_name="causal_forest",
        energy_score=0.42,
        treatment_balance_score=0.35,
        outcome_fit_score=0.45,
        propensity_calibration=0.40,
        n_samples=100,
        n_treated=45,
        n_control=55,
        computation_time_ms=150.5,
        ci_lower=0.38,
        ci_upper=0.46,
        bootstrap_std=0.02,
        details={"method": "bootstrap", "n_bootstrap": 100},
    )


@pytest.fixture
def sample_estimator_result(sample_energy_score_result) -> EstimatorResult:
    """Create a sample EstimatorResult for testing."""
    return EstimatorResult(
        estimator_type=EstimatorType.CAUSAL_FOREST,
        success=True,
        ate=0.25,
        cate=np.array([0.2, 0.25, 0.3]),
        ate_std=0.05,
        ate_ci_lower=0.15,
        ate_ci_upper=0.35,
        energy_score_result=sample_energy_score_result,
        propensity_scores=np.array([0.4, 0.5, 0.6]),
        estimation_time_ms=500.0,
    )


@pytest.fixture
def failed_estimator_result() -> EstimatorResult:
    """Create a failed EstimatorResult for testing."""
    return EstimatorResult(
        estimator_type=EstimatorType.LINEAR_DML,
        success=False,
        error_message="Convergence failed",
        error_type="ConvergenceError",
        estimation_time_ms=100.0,
    )


@pytest.fixture
def sample_selection_result(
    sample_estimator_result, failed_estimator_result
) -> SelectionResult:
    """Create a sample SelectionResult for testing."""
    return SelectionResult(
        selected=sample_estimator_result,
        selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
        all_results=[sample_estimator_result, failed_estimator_result],
        selection_reason="Lowest energy score among: causal_forest=0.4200",
        total_time_ms=650.0,
        energy_scores={"causal_forest": 0.42},
        energy_score_gap=0.0,
    )


# =============================================================================
# ExperimentContext Tests
# =============================================================================


class TestExperimentContext:
    """Tests for ExperimentContext dataclass."""

    def test_create_with_required_fields(self):
        """Test creating context with required fields only."""
        context = ExperimentContext(
            experiment_id="exp-123",
            run_id="run-456",
            experiment_name="test_experiment",
            started_at=datetime(2024, 1, 15, 10, 30),
        )

        assert context.experiment_id == "exp-123"
        assert context.run_id == "run-456"
        assert context.experiment_name == "test_experiment"
        assert context.started_at == datetime(2024, 1, 15, 10, 30)
        assert context.brand is None
        assert context.region is None
        assert context.kpi_name is None

    def test_create_with_all_fields(self):
        """Test creating context with all fields."""
        context = ExperimentContext(
            experiment_id="exp-123",
            run_id="run-456",
            experiment_name="causal_analysis",
            started_at=datetime(2024, 1, 15, 10, 30),
            brand="Remibrutinib",
            region="Northeast",
            kpi_name="TRx_Growth",
        )

        assert context.brand == "Remibrutinib"
        assert context.region == "Northeast"
        assert context.kpi_name == "TRx_Growth"


# =============================================================================
# EnergyScoreMLflowTracker Initialization Tests
# =============================================================================


class TestEnergyScoreMLflowTrackerInit:
    """Tests for EnergyScoreMLflowTracker initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "DATABASE_URL": "postgresql://user:pass@host/db"
        }):
            with patch(
                "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
                return_value=True
            ):
                tracker = EnergyScoreMLflowTracker()

        assert tracker.tracking_uri == "http://mlflow:5000"
        assert tracker.experiment_prefix == "e2i_causal"
        assert tracker.enable_db_logging is True
        assert tracker.db_connection_string == "postgresql://user:pass@host/db"
        assert tracker._current_context is None

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=True
        ):
            tracker = EnergyScoreMLflowTracker(
                tracking_uri="http://custom:9000",
                experiment_prefix="custom_prefix",
                enable_db_logging=False,
                db_connection_string="postgresql://custom@host/db",
            )

        assert tracker.tracking_uri == "http://custom:9000"
        assert tracker.experiment_prefix == "custom_prefix"
        assert tracker.enable_db_logging is False
        assert tracker.db_connection_string == "postgresql://custom@host/db"

    def test_init_without_env_vars(self):
        """Test initialization when env vars are not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove env vars if they exist
            for key in ["MLFLOW_TRACKING_URI", "DATABASE_URL"]:
                os.environ.pop(key, None)

            with patch(
                "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
                return_value=False
            ):
                tracker = EnergyScoreMLflowTracker()

        assert tracker.tracking_uri == "http://localhost:5000"
        assert tracker.db_connection_string is None


# =============================================================================
# _check_mlflow Tests
# =============================================================================


class TestCheckMlflow:
    """Tests for _check_mlflow method."""

    def test_check_mlflow_available(self):
        """Test when MLflow is available."""
        # Create tracker with mocked mlflow import
        with patch.dict("sys.modules", {"mlflow": MagicMock()}):
            tracker = EnergyScoreMLflowTracker.__new__(EnergyScoreMLflowTracker)
            tracker.tracking_uri = "http://localhost:5000"

            result = tracker._check_mlflow()

            assert result is True

    def test_check_mlflow_not_available(self):
        """Test when MLflow import fails."""
        # Simulate mlflow not being installed
        with patch.dict("sys.modules", {"mlflow": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'mlflow'")):
                tracker = EnergyScoreMLflowTracker.__new__(EnergyScoreMLflowTracker)
                tracker.tracking_uri = "http://localhost:5000"

                result = tracker._check_mlflow()

                assert result is False


# =============================================================================
# start_selection_run Tests
# =============================================================================


class TestStartSelectionRun:
    """Tests for start_selection_run context manager."""

    def test_start_selection_run_without_mlflow(self):
        """Test context manager when MLflow is not available."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker()

        with tracker.start_selection_run("test_experiment") as ctx:
            assert ctx is not None
            assert ctx.experiment_name == "test_experiment"
            assert isinstance(ctx.experiment_id, str)
            assert isinstance(ctx.run_id, str)
            assert tracker._current_context is ctx

        # Context cleared after exit
        assert tracker._current_context is None

    def test_start_selection_run_without_mlflow_with_e2i_context(self):
        """Test context manager with E2I-specific context."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker()

        with tracker.start_selection_run(
            experiment_name="brand_analysis",
            run_name="run_001",
            brand="Kisqali",
            region="West",
            kpi_name="NRx_Volume",
        ) as ctx:
            assert ctx.brand == "Kisqali"
            assert ctx.region == "West"
            assert ctx.kpi_name == "NRx_Volume"

    def test_start_selection_run_with_mlflow(self):
        """Test context manager when MLflow is available."""
        mock_mlflow = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_run = MagicMock()
        mock_run.info.run_id = "run-456"
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_run

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=True
        ):
            tracker = EnergyScoreMLflowTracker()
            tracker._mlflow_available = True

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            with tracker.start_selection_run(
                experiment_name="causal_test",
                brand="Fabhalta",
                tags={"custom_tag": "value"},
            ) as ctx:
                assert ctx.experiment_id == "exp-123"
                assert ctx.run_id == "run-456"
                assert ctx.brand == "Fabhalta"

    def test_start_selection_run_creates_experiment_if_not_exists(self):
        """Test that experiment is created if it doesn't exist."""
        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "new-exp-id"

        mock_run = MagicMock()
        mock_run.info.run_id = "run-789"
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_run

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=True
        ):
            tracker = EnergyScoreMLflowTracker()
            tracker._mlflow_available = True

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            with tracker.start_selection_run("new_experiment") as ctx:
                assert ctx.experiment_id == "new-exp-id"

        # Verify create_experiment was called
        mock_mlflow.create_experiment.assert_called_once()

    def test_start_selection_run_clears_context_on_exception(self):
        """Test that context is cleared even if exception occurs."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker()

        with pytest.raises(ValueError):
            with tracker.start_selection_run("test") as ctx:
                assert tracker._current_context is not None
                raise ValueError("Test error")

        # Context should be cleared even after exception
        assert tracker._current_context is None


# =============================================================================
# log_selection_result Tests
# =============================================================================


class TestLogSelectionResult:
    """Tests for log_selection_result method."""

    def test_log_selection_result_no_context(self, sample_selection_result):
        """Test logging when no context is active."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(enable_db_logging=False)

        # No exception should be raised
        tracker.log_selection_result(sample_selection_result)

    def test_log_selection_result_with_context_no_mlflow(self, sample_selection_result):
        """Test logging with context but no MLflow."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(enable_db_logging=False)

        with tracker.start_selection_run("test"):
            tracker.log_selection_result(sample_selection_result)

    def test_log_selection_result_calls_mlflow_logging(self, sample_selection_result):
        """Test that MLflow logging is called when available."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=True
        ):
            tracker = EnergyScoreMLflowTracker(enable_db_logging=False)

        tracker._current_context = ExperimentContext(
            experiment_id="exp-1",
            run_id="run-1",
            experiment_name="test",
            started_at=datetime.now(),
        )

        with patch.object(tracker, "_log_to_mlflow") as mock_log:
            tracker.log_selection_result(sample_selection_result)
            mock_log.assert_called_once_with(sample_selection_result, None)

    def test_log_selection_result_calls_db_logging(self, sample_selection_result):
        """Test that database logging is called when enabled."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(
                enable_db_logging=True,
                db_connection_string="postgresql://test@host/db"
            )

        tracker._current_context = ExperimentContext(
            experiment_id="exp-1",
            run_id="run-1",
            experiment_name="test",
            started_at=datetime.now(),
        )

        with patch.object(tracker, "_log_to_database") as mock_log:
            tracker.log_selection_result(sample_selection_result)
            mock_log.assert_called_once_with(sample_selection_result, "exp-1")

    def test_log_selection_result_with_additional_params(self, sample_selection_result):
        """Test logging with additional parameters."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=True
        ):
            tracker = EnergyScoreMLflowTracker(enable_db_logging=False)

        tracker._current_context = ExperimentContext(
            experiment_id="exp-1",
            run_id="run-1",
            experiment_name="test",
            started_at=datetime.now(),
        )

        additional_params = {"custom_param": "custom_value"}

        with patch.object(tracker, "_log_to_mlflow") as mock_log:
            tracker.log_selection_result(sample_selection_result, additional_params)
            mock_log.assert_called_once_with(sample_selection_result, additional_params)


# =============================================================================
# _log_to_mlflow Tests
# =============================================================================


class TestLogToMlflow:
    """Tests for _log_to_mlflow method."""

    def test_log_to_mlflow_logs_metrics(self, sample_selection_result):
        """Test that metrics are logged correctly."""
        mock_mlflow = MagicMock()

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=True
        ):
            tracker = EnergyScoreMLflowTracker()

        import sys
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_file.name = "/tmp/test.json"
                mock_temp.return_value = mock_file

                with patch("src.causal_engine.energy_score.mlflow_tracker.os.unlink"):
                    tracker._log_to_mlflow(sample_selection_result, None)

        # Verify metrics were logged
        mock_mlflow.log_metric.assert_any_call("selected_energy_score", 0.42)
        mock_mlflow.log_metric.assert_any_call("selected_ate", 0.25)
        mock_mlflow.log_metric.assert_any_call("selected_ate_std", 0.05)
        mock_mlflow.log_metric.assert_any_call("n_estimators_evaluated", 2)
        mock_mlflow.log_metric.assert_any_call("n_estimators_succeeded", 1)
        mock_mlflow.log_metric.assert_any_call("energy_score_gap", 0.0)
        mock_mlflow.log_metric.assert_any_call("total_selection_time_ms", 650.0)

    def test_log_to_mlflow_logs_params(self, sample_selection_result):
        """Test that parameters are logged correctly."""
        mock_mlflow = MagicMock()

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=True
        ):
            tracker = EnergyScoreMLflowTracker()

        import sys
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_file.name = "/tmp/test.json"
                mock_temp.return_value = mock_file

                with patch("src.causal_engine.energy_score.mlflow_tracker.os.unlink"):
                    tracker._log_to_mlflow(sample_selection_result, None)

        mock_mlflow.log_param.assert_any_call("selected_estimator", "causal_forest")
        mock_mlflow.log_param.assert_any_call("selection_strategy", "best_energy")

    def test_log_to_mlflow_with_additional_params(self, sample_selection_result):
        """Test that additional params are logged."""
        mock_mlflow = MagicMock()

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=True
        ):
            tracker = EnergyScoreMLflowTracker()

        additional_params = {"brand": "Kisqali", "region": "East"}

        import sys
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_file.name = "/tmp/test.json"
                mock_temp.return_value = mock_file

                with patch("src.causal_engine.energy_score.mlflow_tracker.os.unlink"):
                    tracker._log_to_mlflow(sample_selection_result, additional_params)

        mock_mlflow.log_param.assert_any_call("brand", "Kisqali")
        mock_mlflow.log_param.assert_any_call("region", "East")


# =============================================================================
# _log_to_database Tests
# =============================================================================


class TestLogToDatabase:
    """Tests for _log_to_database method."""

    def test_log_to_database_no_connection_string(self, sample_selection_result):
        """Test when no database connection string is provided."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(
                enable_db_logging=True,
                db_connection_string=None,
            )

        # Should not raise, just log warning
        tracker._log_to_database(sample_selection_result, "exp-123")

    def test_log_to_database_success(self, sample_selection_result):
        """Test successful database logging."""
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        # Mock Json class from extras
        mock_psycopg2_extras = MagicMock()
        mock_psycopg2_extras.Json = MagicMock(side_effect=lambda x: x)

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(
                enable_db_logging=True,
                db_connection_string="postgresql://test@host/db",
            )

        import sys
        with patch.dict(sys.modules, {
            "psycopg2": mock_psycopg2,
            "psycopg2.extras": mock_psycopg2_extras,
        }):
            tracker._log_to_database(sample_selection_result, "exp-123")

        # Verify connection and cursor were used
        mock_psycopg2.connect.assert_called_once_with("postgresql://test@host/db")
        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_log_to_database_handles_exception(self, sample_selection_result):
        """Test that database exceptions are handled gracefully."""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.side_effect = Exception("Connection failed")

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(
                enable_db_logging=True,
                db_connection_string="postgresql://test@host/db",
            )

        import sys
        with patch.dict(sys.modules, {"psycopg2": mock_psycopg2}):
            # Should not raise, just log error
            tracker._log_to_database(sample_selection_result, "exp-123")


# =============================================================================
# get_selection_comparison Tests
# =============================================================================


class TestGetSelectionComparison:
    """Tests for get_selection_comparison method."""

    def test_get_selection_comparison_no_connection(self):
        """Test when no database connection is available."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(db_connection_string=None)

        result = tracker.get_selection_comparison()

        assert result == {}

    def test_get_selection_comparison_success(self):
        """Test successful comparison query."""
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (100, 80, 20, 75.0, 0.15)

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(
                db_connection_string="postgresql://test@host/db"
            )

        # Mock the import by patching sys.modules before the method is called
        import sys
        with patch.dict(sys.modules, {"psycopg2": mock_psycopg2}):
            result = tracker.get_selection_comparison(days=30)

        assert result == {
            "total_experiments": 100,
            "same_selection": 80,
            "different_selection": 20,
            "pct_improved": 75.0,
            "avg_energy_improvement": 0.15,
        }

    def test_get_selection_comparison_no_data(self):
        """Test when query returns no data."""
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(
                db_connection_string="postgresql://test@host/db"
            )

        import sys
        with patch.dict(sys.modules, {"psycopg2": mock_psycopg2}):
            result = tracker.get_selection_comparison()

        assert result == {}

    def test_get_selection_comparison_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.side_effect = Exception("Query failed")

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(
                db_connection_string="postgresql://test@host/db"
            )

        import sys
        with patch.dict(sys.modules, {"psycopg2": mock_psycopg2}):
            result = tracker.get_selection_comparison()

        assert result == {}


# =============================================================================
# get_estimator_performance Tests
# =============================================================================


class TestGetEstimatorPerformance:
    """Tests for get_estimator_performance method."""

    def test_get_estimator_performance_no_connection(self):
        """Test when no database connection is available."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(db_connection_string=None)

        result = tracker.get_estimator_performance()

        assert result == []

    def test_get_estimator_performance_success(self):
        """Test successful performance query."""
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.description = [
            ("estimator_type",), ("avg_energy_score",), ("success_rate",)
        ]
        mock_cursor.fetchall.return_value = [
            ("causal_forest", 0.42, 0.95),
            ("linear_dml", 0.38, 0.90),
        ]

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(
                db_connection_string="postgresql://test@host/db"
            )

        import sys
        with patch.dict(sys.modules, {"psycopg2": mock_psycopg2}):
            result = tracker.get_estimator_performance()

        assert len(result) == 2
        assert result[0] == {
            "estimator_type": "causal_forest",
            "avg_energy_score": 0.42,
            "success_rate": 0.95,
        }
        assert result[1] == {
            "estimator_type": "linear_dml",
            "avg_energy_score": 0.38,
            "success_rate": 0.90,
        }

    def test_get_estimator_performance_empty_result(self):
        """Test when query returns no data."""
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.description = [("estimator_type",), ("avg_energy_score",)]
        mock_cursor.fetchall.return_value = []

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(
                db_connection_string="postgresql://test@host/db"
            )

        import sys
        with patch.dict(sys.modules, {"psycopg2": mock_psycopg2}):
            result = tracker.get_estimator_performance()

        assert result == []

    def test_get_estimator_performance_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.side_effect = Exception("Query failed")

        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(
                db_connection_string="postgresql://test@host/db"
            )

        import sys
        with patch.dict(sys.modules, {"psycopg2": mock_psycopg2}):
            result = tracker.get_estimator_performance()

        assert result == []


# =============================================================================
# create_tracker Tests
# =============================================================================


class TestCreateTracker:
    """Tests for create_tracker convenience function."""

    def test_create_tracker_with_defaults(self):
        """Test creating tracker with default values."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = create_tracker()

        assert isinstance(tracker, EnergyScoreMLflowTracker)

    def test_create_tracker_with_kwargs(self):
        """Test creating tracker with custom arguments."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = create_tracker(
                tracking_uri="http://custom:5000",
                experiment_prefix="test_prefix",
                enable_db_logging=False,
            )

        assert tracker.tracking_uri == "http://custom:5000"
        assert tracker.experiment_prefix == "test_prefix"
        assert tracker.enable_db_logging is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestMlflowTrackerIntegration:
    """Integration tests for the mlflow tracker."""

    def test_full_workflow_without_mlflow(self, sample_selection_result):
        """Test full workflow when MLflow is not available."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(enable_db_logging=False)

        with tracker.start_selection_run(
            experiment_name="integration_test",
            brand="Remibrutinib",
            region="Northeast",
            kpi_name="TRx_Growth",
        ) as ctx:
            assert ctx.experiment_name == "integration_test"
            assert ctx.brand == "Remibrutinib"

            tracker.log_selection_result(sample_selection_result)

        # Context cleared after exit
        assert tracker._current_context is None

    def test_multiple_runs_in_sequence(self, sample_selection_result):
        """Test multiple runs in sequence."""
        with patch(
            "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
            return_value=False
        ):
            tracker = EnergyScoreMLflowTracker(enable_db_logging=False)

        # First run
        with tracker.start_selection_run("run_1") as ctx1:
            assert ctx1.experiment_name == "run_1"
            tracker.log_selection_result(sample_selection_result)

        # Second run
        with tracker.start_selection_run("run_2") as ctx2:
            assert ctx2.experiment_name == "run_2"
            tracker.log_selection_result(sample_selection_result)

        assert tracker._current_context is None
