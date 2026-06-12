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


import pytest as _pytest_883b


@_pytest_883b.fixture(autouse=True)
def _hermetic_memory_clients_883b(monkeypatch):
    """Keep unit tests off the live memory backends (#883 PR B).

    Before #883 the agent_activities insert in this agent's memory hooks was
    schema-broken (PGRST204 swallowed -> None), so this suite was de-facto
    hermetic even on a creds-configured dev box. Now that the write actually
    LANDS, an unmocked hook call would insert REAL rows into the live DB
    mid-unit-run (the 883-A lesson). Force the lazy client factories to fail
    init — the hooks degrade to their honest "no client" paths. Tests that
    exercise a specific client behavior set ``hooks._supabase_client`` /
    ``hooks._working_memory`` / ``hooks._semantic_memory`` directly, which
    bypasses the factories entirely.
    """

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "hermetic unit-test memory layer (#883): set the hook's private "
            "client attribute directly if the test needs a specific behavior"
        )

    import src.repositories as repositories

    monkeypatch.setattr(repositories, "get_supabase_client", _unavailable)
    import src.memory.working_memory as working_memory

    monkeypatch.setattr(working_memory, "get_working_memory", _unavailable)
    import src.memory.semantic_memory as semantic_memory

    monkeypatch.setattr(semantic_memory, "get_semantic_memory", _unavailable)
