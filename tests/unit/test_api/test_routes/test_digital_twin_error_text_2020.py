"""#2020 / D6 (Task 7b D1, D2): no library text in the Digital Twin 400s.

Both routes mapped ``except ValueError as e`` to ``HTTPException(400, detail=str(e))``, and the
app's HTTPException handler copies a 400's detail verbatim into the ``message`` the frontend
toasts. Probed through the REAL app on 2026-09-13, before this fix:

* ``POST /simulate`` with ``target_deciles=[11]`` answered 400 with pydantic's own text — the
  model name, ``input_value=[11]`` and an ``errors.pydantic.dev`` URL. The request model did not
  bound the deciles, so the domain ``InterventionConfig`` rejected them inside the ``try``.
* ``POST /simulations/compare`` with ``brand="NotABrand"`` answered 400 with
  ``'NotABrand' is not a valid Brand`` (``Brand(scenario.brand)`` runs inside the ``try``).

Driven through the real app, as ``test_http_exception_headers_1999.py`` does, so the assertions
are on the body a client receives rather than on ``HTTPException.detail``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Module-level on purpose: importing src.api.main is paid once at collection,
# outside pytest-timeout's per-test budget (see test_gaps_time_period_1834.py).
from src.api.dependencies.auth import require_operator
from src.api.main import app
from src.api.routes.digital_twin import BrandEnum, InterventionConfigRequest
from src.digital_twin.models.simulation_models import InterventionConfig

_ROUTE_LOGGER = "src.api.routes.digital_twin"
_SIMULATE_SENTENCE = (
    "The simulation request could not be processed. Check the request parameters and try again."
)
_COMPARE_SENTENCE = (
    "The scenario comparison request could not be processed. Check the scenario parameters and "
    "try again."
)
_OPERATOR = {"user_id": "test-2020", "role": "admin", "email": "test-2020@example.invalid"}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    app.dependency_overrides[require_operator] = lambda: _OPERATOR
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.pop(require_operator, None)


def _pydantic_error() -> ValidationError:
    """The real error the domain model raised for the probed request."""
    try:
        InterventionConfig(intervention_type="email_campaign", target_deciles=[11])
    except ValidationError as exc:
        return exc
    raise AssertionError("InterventionConfig accepted decile 11")


def _simulate_body(**intervention: Any) -> Dict[str, Any]:
    return {
        "intervention": {"intervention_type": "email_campaign", **intervention},
        "brand": list(BrandEnum)[0].value,
    }


@pytest.mark.unit
def test_an_out_of_range_decile_is_rejected_by_request_validation(client):
    """D1: the request model bounds each decile, so the domain model never sees 11."""
    resp = client.post("/api/digital-twin/simulate", json=_simulate_body(target_deciles=[11]))
    assert resp.status_code == 422, resp.text
    assert "target_deciles" in resp.text


@pytest.mark.unit
def test_in_range_deciles_and_the_default_still_validate():
    assert InterventionConfigRequest(
        intervention_type="x", target_deciles=[1, 10]
    ).target_deciles == [1, 10]
    assert InterventionConfigRequest(intervention_type="x").target_deciles == [1, 2, 3]


@pytest.mark.unit
def test_a_value_error_inside_simulate_is_a_fixed_400_and_the_raw_text_is_logged(client, caplog):
    raw = _pydantic_error()
    with (
        patch("src.api.routes.digital_twin._get_twin_repo", AsyncMock(side_effect=raw)),
        caplog.at_level(logging.WARNING, logger=_ROUTE_LOGGER),
    ):
        resp = client.post("/api/digital-twin/simulate", json=_simulate_body())

    assert resp.status_code == 400, resp.text
    assert resp.json()["message"] == _SIMULATE_SENTENCE
    for leak in ("Invalid decile", "pydantic.dev", "input_value", "validation error for"):
        assert leak not in resp.text, f"{leak!r} reached the response body"
    assert "Invalid decile: 11" in caplog.text
    # A server-side ValueError lands in this arm too, so the log keeps the traceback.
    rejected = [r for r in caplog.records if "Simulation request rejected" in r.getMessage()]
    assert len(rejected) == 1 and rejected[0].exc_info is not None


@pytest.mark.unit
def test_a_value_error_inside_compare_is_a_fixed_400_and_the_raw_text_is_logged(client, caplog):
    with (
        patch("src.api.routes.digital_twin._get_twin_repo", AsyncMock(return_value=MagicMock())),
        caplog.at_level(logging.WARNING, logger=_ROUTE_LOGGER),
    ):
        resp = client.post(
            "/api/digital-twin/simulations/compare",
            json={"base_scenario": {"intervention_type": "email_campaign", "brand": "NotABrand"}},
        )

    assert resp.status_code == 400, resp.text
    assert resp.json()["message"] == _COMPARE_SENTENCE
    assert "is not a valid Brand" not in resp.text
    assert "NotABrand" not in resp.text
    assert "'NotABrand' is not a valid Brand" in caplog.text
    rejected = [
        r for r in caplog.records if "Scenario comparison request rejected" in r.getMessage()
    ]
    assert len(rejected) == 1 and rejected[0].exc_info is not None
