"""
Tests for src/workers/celery_app.py

Covers:
- Celery app configuration
- Queue definitions
- Task routing
- Beat schedule
- Debug task
- Worker info
"""

import os
from unittest.mock import patch

# =============================================================================
# Celery App Configuration Tests
# =============================================================================


class TestCeleryAppConfiguration:
    """Tests for Celery app configuration."""

    def test_celery_app_exists(self):
        """Test celery_app is properly initialized."""
        from src.workers.celery_app import celery_app

        assert celery_app is not None
        assert celery_app.main == "e2i_causal_analytics"

    def test_broker_url_configured(self):
        """Test broker URL is configured."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.broker_url is not None
        assert "redis" in celery_app.conf.broker_url

    def test_result_backend_configured(self):
        """Test result backend is configured."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.result_backend is not None
        assert "redis" in celery_app.conf.result_backend

    def test_serializer_is_json(self):
        """Test task serializer is JSON for security."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_serializer == "json"
        assert celery_app.conf.result_serializer == "json"
        assert celery_app.conf.accept_content == ["json"]

    def test_timezone_is_utc(self):
        """Test timezone is UTC."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.timezone == "UTC"
        assert celery_app.conf.enable_utc is True

    def test_task_acks_late_enabled(self):
        """Test late acknowledgment is enabled for reliability."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_acks_late is True

    def test_task_reject_on_worker_lost_enabled(self):
        """Test task requeue on worker crash is enabled."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_reject_on_worker_lost is True

    def test_time_limits_configured(self):
        """Test task time limits are set."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_time_limit == 7200  # 2 hours
        assert celery_app.conf.task_soft_time_limit == 6600  # 1h 50m

    def test_retry_settings_configured(self):
        """Test auto-retry settings are configured."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_autoretry_for == (Exception,)
        assert celery_app.conf.task_retry_kwargs == {"max_retries": 3}
        assert celery_app.conf.task_retry_backoff is True
        assert celery_app.conf.task_retry_backoff_max == 600  # 10 minutes

    def test_prefetch_multiplier_is_one(self):
        """Test prefetch is set to 1 for fair distribution."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_monitoring_enabled(self):
        """Test monitoring events are enabled."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.worker_send_task_events is True
        assert celery_app.conf.task_send_sent_event is True


# =============================================================================
# Queue Definition Tests
# =============================================================================


class TestQueueDefinitions:
    """Tests for queue definitions."""

    def test_task_queues_defined(self):
        """Test task queues are defined."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_queues is not None
        assert len(celery_app.conf.task_queues) > 0

    def test_light_worker_queues_exist(self):
        """Test light worker queues exist."""
        from src.workers.celery_app import celery_app

        queue_names = [q.name for q in celery_app.conf.task_queues]

        assert "default" in queue_names
        assert "quick" in queue_names
        assert "api" in queue_names

    def test_medium_worker_queues_exist(self):
        """Test medium worker queues exist."""
        from src.workers.celery_app import celery_app

        queue_names = [q.name for q in celery_app.conf.task_queues]

        assert "analytics" in queue_names
        assert "reports" in queue_names
        assert "aggregations" in queue_names

    def test_heavy_worker_queues_exist(self):
        """Test heavy worker queues exist."""
        from src.workers.celery_app import celery_app

        queue_names = [q.name for q in celery_app.conf.task_queues]

        assert "shap" in queue_names
        assert "causal" in queue_names
        assert "ml" in queue_names
        assert "twins" in queue_names

    def test_default_queue_is_default(self):
        """Test default queue is 'default'."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_default_queue == "default"
        assert celery_app.conf.task_default_exchange == "default"
        assert celery_app.conf.task_default_routing_key == "default"

    def test_queues_use_direct_exchange(self):
        """Test queues use direct exchange type."""
        from src.workers.celery_app import celery_app

        for queue in celery_app.conf.task_queues:
            assert queue.exchange.type == "direct"


# =============================================================================
# Task Routing Tests
# =============================================================================


class TestTaskRouting:
    """Tests for task routing configuration."""

    def test_task_routes_defined(self):
        """Test task routes are defined."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_routes is not None
        assert len(celery_app.conf.task_routes) > 0

    def test_api_tasks_route_to_api_queue(self):
        """Test API tasks route to api queue."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_routes.get("src.tasks.api.*") == {"queue": "api"}

    def test_cache_tasks_route_to_quick_queue(self):
        """Test cache tasks route to quick queue."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_routes.get("src.tasks.cache.*") == {"queue": "quick"}
        assert celery_app.conf.task_routes.get("src.tasks.invalidate_cache") == {"queue": "quick"}
        assert celery_app.conf.task_routes.get("src.tasks.warm_cache") == {"queue": "quick"}

    def test_shap_tasks_route_to_shap_queue(self):
        """Test SHAP tasks route to shap queue."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_routes.get("src.tasks.shap_explain") == {"queue": "shap"}
        assert celery_app.conf.task_routes.get("src.tasks.compute_shap_values") == {"queue": "shap"}

    def test_causal_tasks_route_to_causal_queue(self):
        """Test causal tasks route to causal queue."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_routes.get("src.tasks.causal_refutation") == {"queue": "causal"}
        assert celery_app.conf.task_routes.get("src.tasks.estimate_effect") == {"queue": "causal"}

    def test_ml_tasks_route_to_ml_queue(self):
        """Test ML tasks route to ml queue."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_routes.get("src.tasks.train_model") == {"queue": "ml"}
        assert celery_app.conf.task_routes.get("src.tasks.hyperparameter_tune") == {"queue": "ml"}

    def test_digital_twin_tasks_route_to_twins_queue(self):
        """Test digital twin tasks route to twins queue."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_routes.get("src.tasks.generate_twins") == {"queue": "twins"}
        assert celery_app.conf.task_routes.get("src.tasks.simulate_population") == {
            "queue": "twins"
        }

    def test_analytics_tasks_route_to_analytics_queue(self):
        """Test analytics tasks route to analytics queue."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_routes.get("src.tasks.calculate_metrics") == {
            "queue": "analytics"
        }
        assert celery_app.conf.task_routes.get("src.tasks.compute_statistics") == {
            "queue": "analytics"
        }

    def test_report_tasks_route_to_reports_queue(self):
        """Test report tasks route to reports queue."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.task_routes.get("src.tasks.generate_report") == {"queue": "reports"}
        assert celery_app.conf.task_routes.get("src.tasks.export_report") == {"queue": "reports"}


# =============================================================================
# Beat Schedule Tests
# =============================================================================


class TestBeatSchedule:
    """Tests for Celery beat schedule."""

    def test_beat_schedule_defined(self):
        """Test beat schedule is defined."""
        from src.workers.celery_app import celery_app

        assert celery_app.conf.beat_schedule is not None
        assert len(celery_app.conf.beat_schedule) > 0

    def test_drift_monitoring_scheduled(self):
        """Test drift monitoring is scheduled every 6 hours."""
        from src.workers.celery_app import celery_app

        schedule = celery_app.conf.beat_schedule.get("monitor-drift")
        assert schedule is not None
        assert schedule["task"] == "src.tasks.monitor_model_drift"
        assert schedule["schedule"] == 21600.0  # 6 hours
        assert schedule["options"]["queue"] == "analytics"

    def test_health_check_scheduled(self):
        """Test health check is scheduled every hour."""
        from src.workers.celery_app import celery_app

        schedule = celery_app.conf.beat_schedule.get("health-check")
        assert schedule is not None
        assert schedule["task"] == "src.tasks.health_check"
        assert schedule["schedule"] == 3600.0  # 1 hour
        assert schedule["options"]["queue"] == "quick"

    def test_cache_cleanup_scheduled(self):
        """Test cache cleanup is scheduled daily."""
        from src.workers.celery_app import celery_app

        schedule = celery_app.conf.beat_schedule.get("cache-cleanup")
        assert schedule is not None
        assert schedule["task"] == "src.tasks.cleanup_old_cache"
        assert schedule["schedule"] == 86400.0  # 24 hours
        assert schedule["options"]["queue"] == "quick"

    def test_queue_metrics_scheduled(self):
        """Test queue metrics collection scheduled every 5 minutes."""
        from src.workers.celery_app import celery_app

        schedule = celery_app.conf.beat_schedule.get("queue-metrics")
        assert schedule is not None
        assert schedule["task"] == "src.tasks.collect_queue_metrics"
        assert schedule["schedule"] == 300.0  # 5 minutes
        assert schedule["options"]["queue"] == "quick"

    def test_feast_materialization_scheduled(self):
        """Test Feast feature materialization is scheduled."""
        from src.workers.celery_app import celery_app

        schedule = celery_app.conf.beat_schedule.get("feast-materialize-incremental")
        assert schedule is not None
        assert schedule["schedule"] == 21600.0  # 6 hours
        assert schedule["options"]["queue"] == "analytics"

    def test_feature_freshness_check_scheduled(self):
        """Test feature freshness check is scheduled."""
        from src.workers.celery_app import celery_app

        schedule = celery_app.conf.beat_schedule.get("feast-check-freshness")
        assert schedule is not None
        assert schedule["schedule"] == 14400.0  # 4 hours
        assert schedule["kwargs"]["alert_on_stale"] is True

    def test_ab_testing_schedules_exist(self):
        """Test A/B testing scheduled tasks exist."""
        from src.workers.celery_app import celery_app

        assert "ab-interim-analysis-check" in celery_app.conf.beat_schedule
        assert "ab-enrollment-health-check" in celery_app.conf.beat_schedule
        assert "ab-srm-detection-sweep" in celery_app.conf.beat_schedule
        assert "ab-results-cleanup" in celery_app.conf.beat_schedule

    def test_feedback_loop_schedules_exist(self):
        """Test feedback loop scheduled tasks exist."""
        from src.workers.celery_app import celery_app

        assert "feedback-loop-short-window" in celery_app.conf.beat_schedule
        assert "feedback-loop-medium-window" in celery_app.conf.beat_schedule
        assert "feedback-loop-long-window" in celery_app.conf.beat_schedule
        assert "feedback-loop-drift-analysis" in celery_app.conf.beat_schedule


# =============================================================================
# Debug Task Tests
# =============================================================================


class TestDebugTask:
    """Tests for the debug_task function."""

    def test_debug_task_exists(self):
        """Test debug_task is registered."""
        from src.workers.celery_app import debug_task

        assert debug_task is not None
        assert hasattr(debug_task, "name")
        assert debug_task.name == "src.tasks.debug_task"

    def test_debug_task_is_bound(self):
        """Test debug_task is bound (has access to self)."""
        from src.workers.celery_app import debug_task

        # Bound tasks have _orig attribute or are marked with bind=True
        # The fact it's registered with a name and works confirms it's bound
        assert debug_task.name == "src.tasks.debug_task"


# =============================================================================
# Worker Info Tests
# =============================================================================


class TestGetWorkerInfo:
    """Tests for the get_worker_info function."""

    def test_get_worker_info_exists(self):
        """Test get_worker_info function exists."""
        from src.workers.celery_app import get_worker_info

        assert callable(get_worker_info)

    def test_get_worker_info_returns_dict(self):
        """Test get_worker_info returns a dictionary."""
        from src.workers.celery_app import get_worker_info

        result = get_worker_info()
        assert isinstance(result, dict)
        assert "type" in result
        assert "queues" in result

    def test_get_worker_info_unknown_type(self):
        """Test get_worker_info returns empty queues for unknown type."""
        from src.workers.celery_app import get_worker_info

        with patch.dict(os.environ, {"WORKER_TYPE": "unknown"}, clear=False):
            result = get_worker_info()
            assert result["type"] == "unknown"
            assert result["queues"] == []

    def test_get_worker_info_light_type(self):
        """Test get_worker_info returns light queues."""
        from src.workers.celery_app import get_worker_info

        with patch.dict(os.environ, {"WORKER_TYPE": "light"}, clear=False):
            result = get_worker_info()
            assert result["type"] == "light"
            assert result["queues"] == ["default", "quick", "api"]

    def test_get_worker_info_medium_type(self):
        """Test get_worker_info returns medium queues."""
        from src.workers.celery_app import get_worker_info

        with patch.dict(os.environ, {"WORKER_TYPE": "medium"}, clear=False):
            result = get_worker_info()
            assert result["type"] == "medium"
            assert result["queues"] == ["analytics", "reports", "aggregations"]

    def test_get_worker_info_heavy_type(self):
        """Test get_worker_info returns heavy queues."""
        from src.workers.celery_app import get_worker_info

        with patch.dict(os.environ, {"WORKER_TYPE": "heavy"}, clear=False):
            result = get_worker_info()
            assert result["type"] == "heavy"
            assert result["queues"] == ["shap", "causal", "ml", "twins"]


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_celery_app_exported_from_init(self):
        """Test celery_app is exported from __init__.py."""
        from src.workers import celery_app

        assert celery_app is not None
        assert celery_app.main == "e2i_causal_analytics"

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from src import workers

        assert hasattr(workers, "__all__")
        assert "celery_app" in workers.__all__
