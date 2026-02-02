"""
Unit Tests for Feedback Loop Celery Tasks.

Tests cover:
- run_feedback_loop_short_window task
- run_feedback_loop_medium_window task
- run_feedback_loop_long_window task
- analyze_concept_drift_from_truth task
- run_full_feedback_loop task

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    client = MagicMock()

    # Mock RPC call for run_feedback_loop
    rpc_result = MagicMock()
    rpc_result.data = [
        {
            "run_id": "run-123",
            "predictions_labeled": 50,
            "predictions_skipped": 5,
        }
    ]
    rpc_execute = AsyncMock(return_value=rpc_result)
    rpc_mock = MagicMock(return_value=MagicMock(execute=rpc_execute))
    client.rpc = rpc_mock

    # Mock table queries
    table_result = MagicMock()
    table_result.data = []
    table_execute = AsyncMock(return_value=table_result)
    select_mock = MagicMock(
        return_value=MagicMock(
            eq=MagicMock(return_value=MagicMock(execute=table_execute)),
            execute=table_execute,
        )
    )
    table_mock = MagicMock(return_value=MagicMock(select=select_mock))
    client.table = table_mock

    return client


@pytest.fixture
def mock_feedback_loop_result():
    """Mock feedback loop RPC result."""
    return {
        "run_id": "run-456",
        "predictions_labeled": 100,
        "predictions_skipped": 10,
        "error": None,
    }


@pytest.fixture
def mock_drift_alerts_data():
    """Mock v_drift_alerts view data."""
    return [
        {
            "prediction_type": "churn",
            "accuracy_status": "OK",
            "calibration_status": "OK",
            "accuracy_drop": 0.02,
            "calibration_error": 0.05,
            "baseline_accuracy": 0.85,
            "current_accuracy": 0.83,
            "predictions_count": 500,
        },
        {
            "prediction_type": "trigger",
            "accuracy_status": "ALERT",
            "calibration_status": "OK",
            "accuracy_drop": 0.08,
            "calibration_error": 0.03,
            "baseline_accuracy": 0.90,
            "current_accuracy": 0.82,
            "predictions_count": 1000,
        },
    ]


@pytest.fixture
def mock_concept_drift_metrics():
    """Mock v_concept_drift_metrics view data."""
    return [
        {
            "prediction_type": "churn",
            "week_start": "2025-12-23",
            "total_predictions": 500,
            "labeled_predictions": 450,
            "accuracy": 0.83,
            "calibration_error": 0.05,
        },
    ]


# =============================================================================
# SHORT WINDOW FEEDBACK LOOP TESTS
# =============================================================================


class TestRunFeedbackLoopShortWindow:
    """Tests for run_feedback_loop_short_window Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_short_window

        assert run_feedback_loop_short_window is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_short_window

        assert hasattr(run_feedback_loop_short_window, "delay")
        assert hasattr(run_feedback_loop_short_window, "apply_async")

    def test_task_has_correct_name(self):
        """Test that task has correct Celery name."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_short_window

        assert run_feedback_loop_short_window.name == "src.tasks.run_feedback_loop_short_window"

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_processes_short_window_types(self, mock_get_client, mock_supabase_client):
        """Test that task processes trigger and next_best_action types."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_short_window

        mock_get_client.side_effect = AsyncMock(return_value=mock_supabase_client)

        result = run_feedback_loop_short_window()

        assert result is not None
        assert isinstance(result, dict)
        assert result.get("window") == "short"
        # Check that it's processing the right types
        prediction_types = result.get("prediction_types", [])
        assert "trigger" in prediction_types or "next_best_action" in prediction_types

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_accepts_custom_types(self, mock_get_client, mock_supabase_client):
        """Test that task accepts custom prediction types override."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_short_window

        mock_get_client.side_effect = AsyncMock(return_value=mock_supabase_client)

        custom_types = ["custom_type_1", "custom_type_2"]
        result = run_feedback_loop_short_window(prediction_types=custom_types)

        assert result is not None
        assert result.get("prediction_types") == custom_types

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_handles_no_client(self, mock_get_client):
        """Test graceful handling when no database client."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_short_window

        mock_get_client.side_effect = AsyncMock(return_value=None)

        result = run_feedback_loop_short_window()

        assert result is not None
        assert result.get("status") == "skipped"
        assert "No database client" in result.get("reason", "")


# =============================================================================
# MEDIUM WINDOW FEEDBACK LOOP TESTS
# =============================================================================


class TestRunFeedbackLoopMediumWindow:
    """Tests for run_feedback_loop_medium_window Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_medium_window

        assert run_feedback_loop_medium_window is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_medium_window

        assert hasattr(run_feedback_loop_medium_window, "delay")

    def test_task_has_correct_name(self):
        """Test that task has correct Celery name."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_medium_window

        assert run_feedback_loop_medium_window.name == "src.tasks.run_feedback_loop_medium_window"

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_processes_churn(self, mock_get_client, mock_supabase_client):
        """Test that task processes churn prediction type."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_medium_window

        mock_get_client.side_effect = AsyncMock(return_value=mock_supabase_client)

        result = run_feedback_loop_medium_window()

        assert result is not None
        assert result.get("window") == "medium"
        prediction_types = result.get("prediction_types", [])
        # Config uses hcp_churn or churn - check for either
        assert any("churn" in pt.lower() for pt in prediction_types)


# =============================================================================
# LONG WINDOW FEEDBACK LOOP TESTS
# =============================================================================


class TestRunFeedbackLoopLongWindow:
    """Tests for run_feedback_loop_long_window Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_long_window

        assert run_feedback_loop_long_window is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_long_window

        assert hasattr(run_feedback_loop_long_window, "delay")

    def test_task_has_correct_name(self):
        """Test that task has correct Celery name."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_long_window

        assert run_feedback_loop_long_window.name == "src.tasks.run_feedback_loop_long_window"

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_processes_long_window_types(self, mock_get_client, mock_supabase_client):
        """Test that task processes market_share_impact and risk types."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_long_window

        mock_get_client.side_effect = AsyncMock(return_value=mock_supabase_client)

        result = run_feedback_loop_long_window()

        assert result is not None
        assert result.get("window") == "long"
        prediction_types = result.get("prediction_types", [])
        assert "market_share_impact" in prediction_types or "risk" in prediction_types


# =============================================================================
# CONCEPT DRIFT ANALYSIS TESTS
# =============================================================================


@pytest.mark.xdist_group(name="concept_drift_integration")
class TestAnalyzeConceptDriftFromTruth:
    """Tests for analyze_concept_drift_from_truth Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.feedback_loop_tasks import analyze_concept_drift_from_truth

        assert analyze_concept_drift_from_truth is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.feedback_loop_tasks import analyze_concept_drift_from_truth

        assert hasattr(analyze_concept_drift_from_truth, "delay")

    def test_task_has_correct_name(self):
        """Test that task has correct Celery name."""
        from src.tasks.feedback_loop_tasks import analyze_concept_drift_from_truth

        assert analyze_concept_drift_from_truth.name == "src.tasks.analyze_concept_drift_from_truth"

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_queries_drift_alerts(self, mock_get_client, mock_drift_alerts_data):
        """Test that task queries v_drift_alerts view."""
        from src.tasks.feedback_loop_tasks import analyze_concept_drift_from_truth

        # Setup mock client
        mock_client = MagicMock()

        # Mock v_drift_alerts table
        alerts_result = MagicMock()
        alerts_result.data = mock_drift_alerts_data
        alerts_execute = AsyncMock(return_value=alerts_result)

        # Mock v_concept_drift_metrics table
        metrics_result = MagicMock()
        metrics_result.data = []
        metrics_execute = AsyncMock(return_value=metrics_result)

        def mock_table(table_name):
            mock_select = MagicMock()
            if table_name == "v_drift_alerts":
                mock_select.return_value = MagicMock(
                    eq=MagicMock(return_value=MagicMock(execute=alerts_execute)),
                    execute=alerts_execute,
                )
            else:
                mock_select.return_value = MagicMock(execute=metrics_execute)
            return MagicMock(select=mock_select)

        mock_client.table = mock_table
        mock_get_client.side_effect = AsyncMock(return_value=mock_client)

        result = analyze_concept_drift_from_truth()

        assert result is not None
        assert result.get("status") == "completed"
        assert "drift_results" in result

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_detects_accuracy_alert(self, mock_get_client, mock_drift_alerts_data):
        """Test that task detects accuracy degradation alerts."""
        from src.tasks.feedback_loop_tasks import analyze_concept_drift_from_truth

        mock_client = MagicMock()

        alerts_result = MagicMock()
        alerts_result.data = mock_drift_alerts_data
        alerts_execute = AsyncMock(return_value=alerts_result)

        metrics_result = MagicMock()
        metrics_result.data = []
        metrics_execute = AsyncMock(return_value=metrics_result)

        def mock_table(table_name):
            mock_select = MagicMock()
            if table_name == "v_drift_alerts":
                mock_select.return_value = MagicMock(
                    eq=MagicMock(return_value=MagicMock(execute=alerts_execute)),
                    execute=alerts_execute,
                )
            else:
                mock_select.return_value = MagicMock(execute=metrics_execute)
            return MagicMock(select=mock_select)

        mock_client.table = mock_table
        mock_get_client.side_effect = AsyncMock(return_value=mock_client)

        result = analyze_concept_drift_from_truth()

        # Should detect alert for "trigger" type (accuracy_status == "ALERT")
        assert result.get("alerts_triggered", 0) >= 1
        alerts = result.get("alerts", [])
        assert any(a.get("type") == "accuracy_degradation" for a in alerts)

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_accepts_prediction_type_filter(self, mock_get_client):
        """Test that task accepts prediction_type filter."""
        from src.tasks.feedback_loop_tasks import analyze_concept_drift_from_truth

        mock_client = MagicMock()

        alerts_result = MagicMock()
        alerts_result.data = []
        alerts_execute = AsyncMock(return_value=alerts_result)

        metrics_result = MagicMock()
        metrics_result.data = []
        AsyncMock(return_value=metrics_result)

        mock_select = MagicMock()
        mock_select.return_value = MagicMock(
            eq=MagicMock(return_value=MagicMock(execute=alerts_execute)),
            execute=alerts_execute,
        )

        mock_client.table = MagicMock(return_value=MagicMock(select=mock_select))
        mock_get_client.side_effect = AsyncMock(return_value=mock_client)

        result = analyze_concept_drift_from_truth(prediction_type="churn")

        assert result is not None
        assert result.get("prediction_type") == "churn"

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_uses_custom_windows(self, mock_get_client):
        """Test that task accepts custom baseline and current windows."""
        from src.tasks.feedback_loop_tasks import analyze_concept_drift_from_truth

        mock_client = MagicMock()

        alerts_result = MagicMock()
        alerts_result.data = []
        alerts_execute = AsyncMock(return_value=alerts_result)

        metrics_result = MagicMock()
        metrics_result.data = []
        AsyncMock(return_value=metrics_result)

        mock_select = MagicMock()
        mock_select.return_value = MagicMock(
            eq=MagicMock(return_value=MagicMock(execute=alerts_execute)),
            execute=alerts_execute,
        )

        mock_client.table = MagicMock(return_value=MagicMock(select=mock_select))
        mock_get_client.side_effect = AsyncMock(return_value=mock_client)

        result = analyze_concept_drift_from_truth(
            baseline_days=120,
            current_days=14,
        )

        assert result is not None
        assert result.get("baseline_days") == 120
        assert result.get("current_days") == 14

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_handles_no_client_drift(self, mock_get_client):
        """Test graceful handling when no database client."""
        from src.tasks.feedback_loop_tasks import analyze_concept_drift_from_truth

        mock_get_client.side_effect = AsyncMock(return_value=None)

        result = analyze_concept_drift_from_truth()

        assert result is not None
        assert result.get("status") == "skipped"


# =============================================================================
# FULL FEEDBACK LOOP TESTS
# =============================================================================


class TestRunFullFeedbackLoop:
    """Tests for run_full_feedback_loop Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.feedback_loop_tasks import run_full_feedback_loop

        assert run_full_feedback_loop is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.feedback_loop_tasks import run_full_feedback_loop

        assert hasattr(run_full_feedback_loop, "delay")

    def test_task_has_correct_name(self):
        """Test that task has correct Celery name."""
        from src.tasks.feedback_loop_tasks import run_full_feedback_loop

        assert run_full_feedback_loop.name == "src.tasks.run_full_feedback_loop"

    @patch("src.tasks.feedback_loop_tasks.analyze_concept_drift_from_truth")
    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_runs_all_windows(self, mock_get_client, mock_drift_task, mock_supabase_client):
        """Test that task runs feedback loop for all prediction types."""
        from src.tasks.feedback_loop_tasks import run_full_feedback_loop

        mock_get_client.side_effect = AsyncMock(return_value=mock_supabase_client)
        mock_drift_task.delay = MagicMock(return_value=MagicMock(id="drift-task-123"))

        result = run_full_feedback_loop()

        assert result is not None
        assert result.get("window") == "full"

    @patch("src.tasks.feedback_loop_tasks.analyze_concept_drift_from_truth")
    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_triggers_drift_analysis(
        self, mock_get_client, mock_drift_task, mock_supabase_client
    ):
        """Test that task triggers drift analysis when include_drift_analysis=True."""
        from src.tasks.feedback_loop_tasks import run_full_feedback_loop

        mock_get_client.side_effect = AsyncMock(return_value=mock_supabase_client)
        mock_drift_task.delay = MagicMock(return_value=MagicMock(id="drift-task-123"))

        result = run_full_feedback_loop(include_drift_analysis=True)

        assert result is not None
        assert result.get("drift_analysis") is not None
        mock_drift_task.delay.assert_called_once()

    @patch("src.tasks.feedback_loop_tasks.analyze_concept_drift_from_truth")
    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_skips_drift_analysis_when_disabled(
        self, mock_get_client, mock_drift_task, mock_supabase_client
    ):
        """Test that task skips drift analysis when include_drift_analysis=False."""
        from src.tasks.feedback_loop_tasks import run_full_feedback_loop

        mock_get_client.side_effect = AsyncMock(return_value=mock_supabase_client)

        result = run_full_feedback_loop(include_drift_analysis=False)

        assert result is not None
        assert result.get("drift_analysis") is None
        mock_drift_task.delay.assert_not_called()


# =============================================================================
# TASK CONFIGURATION TESTS
# =============================================================================


class TestTaskConfiguration:
    """Tests for task configuration and scheduling."""

    def test_all_tasks_exported(self):
        """Test that all tasks are exported from module."""
        from src.tasks import (
            analyze_concept_drift_from_truth,
            run_feedback_loop_long_window,
            run_feedback_loop_medium_window,
            run_feedback_loop_short_window,
            run_full_feedback_loop,
        )

        assert callable(run_feedback_loop_short_window)
        assert callable(run_feedback_loop_medium_window)
        assert callable(run_feedback_loop_long_window)
        assert callable(analyze_concept_drift_from_truth)
        assert callable(run_full_feedback_loop)

    def test_tasks_have_names(self):
        """Test that tasks have proper names."""
        from src.tasks.feedback_loop_tasks import (
            analyze_concept_drift_from_truth,
            run_feedback_loop_long_window,
            run_feedback_loop_medium_window,
            run_feedback_loop_short_window,
            run_full_feedback_loop,
        )

        assert run_feedback_loop_short_window.name.startswith("src.tasks.")
        assert run_feedback_loop_medium_window.name.startswith("src.tasks.")
        assert run_feedback_loop_long_window.name.startswith("src.tasks.")
        assert analyze_concept_drift_from_truth.name.startswith("src.tasks.")
        assert run_full_feedback_loop.name.startswith("src.tasks.")

    def test_beat_schedule_has_feedback_loop_tasks(self):
        """Test that Celery beat schedule includes feedback loop tasks."""
        from src.workers.celery_app import celery_app

        beat_schedule = celery_app.conf.beat_schedule

        # Verify feedback loop tasks are scheduled
        assert "feedback-loop-short-window" in beat_schedule
        assert "feedback-loop-medium-window" in beat_schedule
        assert "feedback-loop-long-window" in beat_schedule
        assert "feedback-loop-drift-analysis" in beat_schedule

    def test_beat_schedule_correct_intervals(self):
        """Test that beat schedule has correct intervals."""
        from src.workers.celery_app import celery_app

        beat_schedule = celery_app.conf.beat_schedule

        # Short window: 4 hours = 14400 seconds
        assert beat_schedule["feedback-loop-short-window"]["schedule"] == 14400.0

        # Medium window: 24 hours = 86400 seconds
        assert beat_schedule["feedback-loop-medium-window"]["schedule"] == 86400.0

        # Long window: 7 days = 604800 seconds
        assert beat_schedule["feedback-loop-long-window"]["schedule"] == 604800.0

    def test_task_routes_analytics_queue(self):
        """Test that feedback loop tasks are routed to analytics queue."""
        from src.workers.celery_app import celery_app

        task_routes = celery_app.conf.task_routes

        # Wildcard routes for feedback loop tasks
        assert "src.tasks.run_feedback_loop_*" in task_routes
        assert task_routes["src.tasks.run_feedback_loop_*"]["queue"] == "analytics"

        assert "src.tasks.analyze_concept_drift_*" in task_routes
        assert task_routes["src.tasks.analyze_concept_drift_*"]["queue"] == "analytics"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestTaskErrorHandling:
    """Tests for task error handling."""

    @patch("src.memory.services.factories.get_supabase_client")
    def test_short_window_handles_rpc_error(self, mock_get_client):
        """Test short-window task handles RPC errors gracefully."""
        from src.tasks.feedback_loop_tasks import run_feedback_loop_short_window

        mock_client = MagicMock()
        mock_rpc_execute = AsyncMock(side_effect=Exception("RPC call failed"))
        mock_client.rpc = MagicMock(return_value=MagicMock(execute=mock_rpc_execute))
        mock_get_client.side_effect = AsyncMock(return_value=mock_client)

        result = run_feedback_loop_short_window()

        # Should return result with errors, not crash
        assert result is not None
        assert len(result.get("errors", [])) > 0 or result.get("status") == "partial"

    @patch("src.memory.services.factories.get_supabase_client")
    def test_drift_analysis_handles_query_error(self, mock_get_client):
        """Test drift analysis handles query errors gracefully."""
        from src.tasks.feedback_loop_tasks import analyze_concept_drift_from_truth

        mock_client = MagicMock()
        mock_execute = AsyncMock(side_effect=Exception("Query failed"))
        mock_client.table = MagicMock(
            return_value=MagicMock(
                select=MagicMock(
                    return_value=MagicMock(
                        eq=MagicMock(return_value=MagicMock(execute=mock_execute)),
                        execute=mock_execute,
                    )
                )
            )
        )
        mock_get_client.side_effect = AsyncMock(return_value=mock_client)

        result = analyze_concept_drift_from_truth()

        assert result is not None
        assert result.get("status") == "failed"
        assert "error" in result


# =============================================================================
# CONFIGURATION LOADING TESTS
# =============================================================================


class TestConfigurationLoading:
    """Tests for configuration loading."""

    def test_load_config_returns_dict(self):
        """Test that load_config returns a dictionary."""
        from src.tasks.feedback_loop_tasks import load_config

        config = load_config()

        assert isinstance(config, dict)
        assert "feedback_loop" in config

    def test_load_config_has_schedule_settings(self):
        """Test that config has schedule settings."""
        from src.tasks.feedback_loop_tasks import load_config

        config = load_config()
        schedule = config.get("feedback_loop", {}).get("schedule", {})

        assert "short_window_types" in schedule
        assert "medium_window_types" in schedule
        assert "long_window_types" in schedule

    def test_load_config_has_processing_settings(self):
        """Test that config has processing settings."""
        from src.tasks.feedback_loop_tasks import load_config

        config = load_config()
        processing = config.get("feedback_loop", {}).get("processing", {})

        assert "batch_size" in processing
        assert "min_confidence_threshold" in processing

    def test_load_config_has_drift_integration(self):
        """Test that config has drift integration settings."""
        from src.tasks.feedback_loop_tasks import load_config

        config = load_config()
        drift = config.get("drift_integration", {})

        # Config may have concept_drift at top level or nested under drift_monitor_agent
        concept_drift = drift.get("concept_drift") or drift.get("drift_monitor_agent", {}).get(
            "concept_drift"
        )
        assert concept_drift is not None, "concept_drift settings not found"
        assert "comparison_windows" in concept_drift
        assert "alert_thresholds" in concept_drift


# =============================================================================
# ASYNC HELPER TESTS
# =============================================================================


class TestAsyncHelper:
    """Tests for run_async helper function."""

    def test_run_async_executes_coroutine(self):
        """Test that run_async executes a coroutine."""
        from src.tasks.feedback_loop_tasks import run_async

        async def sample_coro():
            return {"status": "success"}

        result = run_async(sample_coro())

        assert result == {"status": "success"}

    def test_run_async_handles_exception(self):
        """Test that run_async propagates exceptions."""
        from src.tasks.feedback_loop_tasks import run_async

        async def failing_coro():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_async(failing_coro())
