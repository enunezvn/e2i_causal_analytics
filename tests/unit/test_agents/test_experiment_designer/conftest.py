"""Fixtures for experiment_designer agent tests.

Provides:
- Mock LLM factory to prevent real API calls in unit tests
- MLflow cleanup to prevent "run already active" errors between tests
- Environment variable setup for dspy imports
"""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

# Note: Real API key should be loaded from .env by root conftest.py (load_dotenv)
# Only set a placeholder if no key exists at all (allows imports to succeed)
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "test-key-for-import-only"


# ---------------------------------------------------------------------------
# Mock LLM response used by all experiment_designer tests
# ---------------------------------------------------------------------------

MOCK_DESIGN_JSON = json.dumps(
    {
        "refined_hypothesis": "Increasing visit frequency causes higher engagement",
        "treatment_definition": {
            "name": "increased_visits",
            "description": "Increased visit frequency",
            "implementation_details": "Weekly visits instead of bi-weekly",
            "target_population": "All HCPs",
            "dosage_or_intensity": "2x frequency",
            "duration": "12 weeks",
            "delivery_mechanism": "In-person rep visits",
        },
        "outcome_definition": {
            "name": "engagement_score",
            "metric_type": "continuous",
            "measurement_method": "CRM engagement index",
            "measurement_frequency": "weekly",
            "baseline_value": 50.0,
            "expected_effect_size": 0.25,
            "minimum_detectable_effect": 0.15,
            "is_primary": True,
        },
        "design_type": "RCT",
        "design_rationale": "RCT is appropriate for this intervention study with randomization at the individual HCP level",
        "randomization_unit": "individual",
        "randomization_method": "stratified",
        "stratification_vars": ["territory", "specialty"],
        "blocking_variables": ["region"],
        "causal_assumptions": ["No unmeasured confounding", "SUTVA holds"],
        "anticipated_confounders": [
            {"name": "territory_size", "how_addressed": "Stratified randomization"},
            {"name": "baseline_engagement", "how_addressed": "Covariate adjustment"},
        ],
    }
)


class MockLLMResponse:
    """Mock LLM response with .content and .response_metadata."""

    def __init__(self, content: str = MOCK_DESIGN_JSON):
        self.content = content
        self.response_metadata = {
            "usage": {"input_tokens": 500, "output_tokens": 300}
        }


def make_mock_llm(content: str = MOCK_DESIGN_JSON) -> AsyncMock:
    """Create a mock LLM with ainvoke returning a proper response object."""
    mock = AsyncMock()
    mock.ainvoke.return_value = MockLLMResponse(content)
    return mock


# Register custom marker
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_llm_api: mark test as requiring real LLM API key"
    )


@pytest.fixture(autouse=True)
def mock_llm_factory():
    """Patch LLM factory functions so no real API calls are made.

    This applies to ALL tests in the experiment_designer directory.
    Patches:
      - design_reasoning: get_chat_llm, get_fast_llm, get_llm_provider
      - validity_audit: _get_validity_llm (force MockValidityLLM fallback)

    Tests that need real API access should be run manually with
    @pytest.mark.requires_llm_api.
    """
    from src.agents.experiment_designer.nodes.validity_audit import (
        MockValidityLLM,
    )

    with (
        patch(
            "src.agents.experiment_designer.nodes.design_reasoning.get_chat_llm",
            return_value=make_mock_llm(),
        ),
        patch(
            "src.agents.experiment_designer.nodes.design_reasoning.get_fast_llm",
            return_value=make_mock_llm(),
        ),
        patch(
            "src.agents.experiment_designer.nodes.design_reasoning.get_llm_provider",
            return_value="openai",
        ),
        patch(
            "src.agents.experiment_designer.nodes.validity_audit._get_validity_llm",
            return_value=(MockValidityLLM(), "mock-validity-llm", False),
        ),
        patch(
            "src.agents.experiment_designer.agent.ExperimentDesignerAgent._get_mlflow_tracker",
            return_value=None,
        ),
    ):
        yield


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
