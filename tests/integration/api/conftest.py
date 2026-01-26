"""
Conftest for API integration tests.

Loads environment variables from .env file before running tests.
This is needed because pytest-env isn't installed on the droplet.

Author: E2I Causal Analytics Team
"""

import os

# Set testing mode BEFORE any src imports to bypass JWT auth
os.environ["E2I_TESTING_MODE"] = "1"

from pathlib import Path

import pytest


def _load_dotenv_if_available():
    """Load .env file if dotenv is available."""
    try:
        from dotenv import load_dotenv

        # Find project root (where .env is located)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        env_path = project_root / ".env"

        if env_path.exists():
            load_dotenv(env_path, override=True)
            return True
    except ImportError:
        pass
    return False


# Load environment at module level (before tests run)
_dotenv_loaded = _load_dotenv_if_available()


@pytest.fixture(scope="module")
def ensure_environment():
    """Ensure environment variables are loaded for integration tests.

    Note: This fixture is NOT autouse to avoid skipping unrelated tests.
    Integration tests should explicitly use this fixture or the integration_test marker.
    """
    if not _dotenv_loaded:
        pytest.skip("dotenv not available - cannot load .env for integration tests")

    # Verify critical env vars are set
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        pytest.skip("Supabase not configured - skipping integration tests")
