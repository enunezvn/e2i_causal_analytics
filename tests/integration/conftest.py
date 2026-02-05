"""
Pytest fixtures for integration tests.

This module configures the testing environment for integration tests:
1. Enables testing mode to bypass JWT authentication
2. Provides common fixtures for mocking dependencies
3. Ensures proper cleanup of app state between tests

Author: E2I Causal Analytics Team
"""

import os
from pathlib import Path
from unittest.mock import patch

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


# =============================================================================
# SKILLS FIXTURES
# =============================================================================

# Use the same fixtures as unit tests for skill loading
FIXTURES_SKILLS_DIR = Path(__file__).parent.parent / "unit" / "test_skills" / "fixtures" / "skills"


@pytest.fixture(autouse=True)
def _use_fixture_skills():
    """Redirect SkillLoader to use test fixtures instead of .claude/skills/.

    This is required because .claude/skills/ is gitignored and not available in CI.
    The fixture skills directory contains all the skill files needed for testing.
    """
    with patch(
        "src.skills.loader.SkillLoader.DEFAULT_BASE_PATH",
        str(FIXTURES_SKILLS_DIR),
    ):
        # Reset the module-level cached instances so they pick up the patched path
        with patch("src.skills.loader._default_loader", None):
            with patch("src.skills.matcher._default_matcher", None):
                yield
