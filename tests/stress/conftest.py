"""Pytest configuration for stress tests.

Stress tests are:
- NOT run by default in CI
- Run with: pytest tests/stress/ -m stress
- Require more memory and time than unit tests
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "stress: mark test as stress test (not run by default)",
    )
    config.addinivalue_line(
        "markers",
        "large_scale: mark test as requiring large dataset (>10K rows)",
    )
    config.addinivalue_line(
        "markers",
        "memory_intensive: mark test as memory intensive (>4GB)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip stress tests unless explicitly requested."""
    if config.getoption("-m") and "stress" in config.getoption("-m"):
        # Stress tests explicitly requested
        return

    # Check if running from stress directory
    if any("stress" in str(item.fspath) for item in items):
        skip_stress = pytest.mark.skip(
            reason="Stress tests skipped by default. Run with: pytest tests/stress/ -m stress"
        )
        for item in items:
            if "stress" in str(item.fspath):
                item.add_marker(skip_stress)
