"""Fixtures for drift_monitor agent tests.

Provides:
- Mock data connector to prevent real Supabase queries
- MLflow tracker mock to prevent real MLflow artifact logging
- MLflow cleanup to prevent "run already active" errors between tests
"""

import os
from unittest.mock import patch

import pytest

# Force mock connector for all drift monitor tests
os.environ["DRIFT_MONITOR_CONNECTOR"] = "mock"


@pytest.fixture(autouse=True)
def mock_mlflow_tracker():
    """Patch MLflow tracker so no real MLflow calls are made.

    Prevents 'mlflow-artifacts URI requires http tracking URI' errors
    when MLflow is configured with a local file:// backend.
    """
    with patch(
        "src.agents.drift_monitor.agent.DriftMonitorAgent._get_mlflow_tracker",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def cleanup_mlflow_runs():
    """End any active MLflow runs before and after each test."""
    _end_all_mlflow_runs()
    yield
    _end_all_mlflow_runs()


def _end_all_mlflow_runs():
    """End all active MLflow runs."""
    try:
        import mlflow

        for _ in range(10):
            if mlflow.active_run() is not None:
                mlflow.end_run()
            else:
                break
    except (ImportError, Exception):
        pass
