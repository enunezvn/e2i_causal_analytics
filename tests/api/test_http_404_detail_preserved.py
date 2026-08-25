"""The global 404 handlers preserve in-app route details (#1814).

Regression: every in-app ``raise HTTPException(status_code=404, detail="...")``
was rewritten to a generic "Endpoint '<path>' not found" EndpointNotFoundError
body, discarding deliberate client-facing messages ("Unknown brand 'X'",
"KPI not found: Y") and misdescribing the failure — the endpoint exists, a
resource doesn't. Unmatched routes (Starlette's default "Not Found" detail)
must KEEP the generic endpoint envelope, so both halves are pinned here.

Two handlers are registered in src.api.main and starlette resolves the
status-code handler (``not_found_handler``) FIRST for HTTPException(404); the
404 branch of the class handler (``http_exception_handler``) is its mirror.
Both are exercised: the default client registers both exactly as main.py does,
and one test drops the status handler to pin the mirror branch directly.

The real app handlers are imported lazily inside the helper so collection
stays cheap; this runs in the CI integration lane (same pattern as the 503
sibling, tests/api/test_http_503_detail_unmasked.py).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _client_with_404_route(detail, *, status_handler=True):
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from src.api.main import http_exception_handler, not_found_handler

    app = FastAPI()
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    if status_handler:
        app.add_exception_handler(404, not_found_handler)

    @app.get("/things/lookup")
    async def lookup():
        if detail is ...:
            raise HTTPException(status_code=404)
        raise HTTPException(status_code=404, detail=detail)

    return TestClient(app, raise_server_exceptions=False)


def test_404_preserves_in_app_route_detail():
    resp = _client_with_404_route("Analysis abc-123 not found").get("/things/lookup")
    assert resp.status_code == 404
    body = resp.json()
    assert body["message"] == "Analysis abc-123 not found"
    assert body.get("category") == "not_found"
    # The misleading endpoint-missing framing must be gone.
    assert "Endpoint" not in body["message"]
    assert "/api/docs" not in (body.get("suggested_action") or "")


def test_unmatched_route_keeps_generic_endpoint_envelope():
    # Positive control: a path with no route gets Starlette's default
    # "Not Found" detail and must still render the endpoint-missing body.
    resp = _client_with_404_route("irrelevant").get("/no/such/route")
    assert resp.status_code == 404
    body = resp.json()
    assert body["message"] == "Endpoint '/no/such/route' not found"
    assert "/api/docs" in body["suggested_action"]


def test_404_without_detail_is_generic():
    # An in-app raise with no detail carries nothing to preserve — Starlette
    # fills "Not Found", which must not be promoted to a message.
    resp = _client_with_404_route(...).get("/things/lookup")
    assert resp.status_code == 404
    assert resp.json()["message"] == "Endpoint '/things/lookup' not found"


def test_class_handler_404_branch_mirrors_detail_preservation():
    # Defense-in-depth: if the status handler ever stops being registered,
    # the StarletteHTTPException class handler's 404 branch must behave
    # identically, not regress to the generic rewrite.
    resp = _client_with_404_route("KPI not found: WS3-BI-999", status_handler=False).get(
        "/things/lookup"
    )
    assert resp.status_code == 404
    assert resp.json()["message"] == "KPI not found: WS3-BI-999"
