"""L4: API version must be single-sourced from pyproject.toml across surfaces.

The canonical version is ``[project] version`` in pyproject.toml; every
API surface (OpenAPI info.version, /health body, ``src.api.__version__``)
must report it. Hand-pinned copies drifted before (4.2.0 vs 4.2.1).

The ``from src.api.main import ...`` is hoisted to module level on purpose: the
app import is heavy (~25s) and the per-test ``timeout = 30`` (pyproject) would
fire if it ran inside a test body. Importing at collection time keeps each test
well under the budget (mirrors test_route_shadow_regression / test_cors_*).
"""

import tomllib
from pathlib import Path

import pytest

import src.api as api_pkg
from src.api.main import API_VERSION, app, health_check

_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"
with _PYPROJECT.open("rb") as _f:
    CANONICAL_VERSION = tomllib.load(_f)["project"]["version"]


@pytest.mark.unit
def test_health_body_version_matches_openapi_version():
    assert app.version == CANONICAL_VERSION
    assert API_VERSION == app.version


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_endpoint_reports_canonical_version():
    body = await health_check()
    assert body["version"] == API_VERSION == CANONICAL_VERSION


@pytest.mark.unit
def test_module_version_matches():
    assert api_pkg.__version__ == CANONICAL_VERSION
