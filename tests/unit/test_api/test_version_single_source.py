"""L4: API version must be a single, consistent 4.2.0 across surfaces.

The ``from src.api.main import ...`` is hoisted to module level on purpose: the
app import is heavy (~25s) and the per-test ``timeout = 30`` (pyproject) would
fire if it ran inside a test body. Importing at collection time keeps each test
well under the budget (mirrors test_route_shadow_regression / test_cors_*).
"""

import pytest

import src.api as api_pkg
from src.api.main import API_VERSION, app, health_check


@pytest.mark.unit
def test_health_body_version_matches_openapi_version():
    assert app.version == "4.2.0"
    assert API_VERSION == app.version


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_endpoint_reports_canonical_version():
    body = await health_check()
    assert body["version"] == API_VERSION == "4.2.0"


@pytest.mark.unit
def test_module_version_matches():
    assert api_pkg.__version__ == "4.2.0"
