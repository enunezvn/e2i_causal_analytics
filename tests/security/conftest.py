"""Pytest fixtures for security tests."""

import pytest


@pytest.fixture(autouse=True)
def isolate_environment():
    """Ensure tests don't affect real environment."""
    # Tests use patch.dict which handles cleanup automatically
    yield
