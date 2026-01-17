"""
Pytest fixtures for API endpoint tests.

This module configures the testing environment for API tests:
1. Enables testing mode to bypass JWT authentication
2. Provides common fixtures for mocking dependencies
3. Ensures proper cleanup of app state between tests

Author: E2I Causal Analytics Team
"""

import os

import pytest
from fastapi.testclient import TestClient

# =============================================================================
# TESTING MODE CONFIGURATION
# =============================================================================

# Enable testing mode BEFORE importing the app
# This must be done before any imports that load the auth module
os.environ["E2I_TESTING_MODE"] = "1"

# Now import the app (auth module will see TESTING_MODE=True)
from src.api.main import app  # noqa: E402


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def test_client():
    """Create a test client for the FastAPI app.

    This fixture is module-scoped to avoid re-creating the client
    for every test, which improves performance.
    """
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_dependency_overrides():
    """Clean up dependency overrides after each test.

    This ensures tests don't leak mock configurations to other tests.
    """
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    """Provide mock authentication headers for tests.

    In testing mode, these aren't validated but can be used
    to verify header handling logic.
    """
    return {"Authorization": "Bearer test-token"}
