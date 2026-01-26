"""Fixtures for experiment_designer agent tests.

Provides:
- MLflow cleanup to prevent "run already active" errors between tests
- Environment variable setup for dspy imports
- Skip marker for tests requiring real LLM API key
"""

import os

import pytest

# Note: Real API key should be loaded from .env by root conftest.py (load_dotenv)
# Only set a placeholder if no key exists at all (allows imports to succeed)
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "test-key-for-import-only"


def _has_real_api_key() -> bool:
    """Check if a real (non-test) API key is available."""
    key = os.environ.get("OPENAI_API_KEY", "")
    # Test keys are placeholders
    return bool(key) and not key.startswith("test-key") and not key.startswith("sk-test")


# Register custom marker
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_llm_api: mark test as requiring real LLM API key"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with requires_llm_api if no real API key is available."""
    if _has_real_api_key():
        return  # Real key available, run all tests

    skip_marker = pytest.mark.skip(
        reason="Requires real OPENAI_API_KEY (test key provided for import only)"
    )

    # Test classes that call real LLM APIs (integration tests)
    llm_test_classes = {
        "TestExperimentDesignerAgent",
        "TestExperimentDesignerOutput",
        "TestAsyncExecution",
        "TestEdgeCases",
        "TestEndToEndWorkflows",
    }

    for item in items:
        # Check for explicit marker
        if "requires_llm_api" in [m.name for m in item.iter_markers()]:
            item.add_marker(skip_marker)
            continue

        # Skip tests in classes that call real LLM
        for cls_name in llm_test_classes:
            if cls_name in item.nodeid:
                item.add_marker(skip_marker)
                break


@pytest.fixture(autouse=True)
def cleanup_mlflow_runs():
    """End any active MLflow runs before and after each test.

    This prevents "Run with UUID ... is already active" errors when
    tests fail to properly clean up their MLflow runs.
    """
    # Cleanup before test
    _end_all_mlflow_runs()

    yield

    # Cleanup after test
    _end_all_mlflow_runs()


def _end_all_mlflow_runs():
    """End all active MLflow runs."""
    try:
        import mlflow

        # End any active runs (may be nested)
        for _ in range(10):  # Max nesting depth
            if mlflow.active_run() is not None:
                mlflow.end_run()
            else:
                break
    except ImportError:
        # MLflow not installed
        pass
    except Exception:
        # Ignore any errors during cleanup
        pass
