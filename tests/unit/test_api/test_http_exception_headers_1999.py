"""#1999 item 1: an ``HTTPException``'s ``headers`` reach the client through the
REAL app's exception handlers.

Regression: ``http_exception_handler`` and ``not_found_handler`` (src.api.main)
rebuild a ``JSONResponse`` from ``status_code``/``detail`` and dropped
``exc.headers``, so the expert-review assessment route's 409 ``Retry-After: 5``
(#1993) and digital_twin's 503 ``Retry-After: 30`` reached the client without
the header, and Starlette's own 405 lost its ``Allow`` header. The #1993 route
tests asserted ``Retry-After`` on a bare ``FastAPI()`` without these handlers,
which is how the drop went unseen.

Starlette resolves an ``HTTPException`` by STATUS handler first (only 404 is
registered as one here), then by class (``StarletteHTTPException`` ->
``http_exception_handler``); 500/``Exception`` handlers live on
``ServerErrorMiddleware`` and never see an ``HTTPException``. So every code
below is answered by one of the two rebuilding handlers, through the real
middleware stack.

Throwaway routes are inserted at the front of the real router and removed
afterwards. The client is deliberately NOT ``with TestClient(app)`` (no
lifespan), as in test_gaps_time_period_1834.py.
"""

from __future__ import annotations

from typing import Dict, Iterator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Module-level on purpose: importing src.api.main is paid once at collection,
# outside pytest-timeout's per-test budget (see test_gaps_time_period_1834.py).
from src.api.errors import user_safe_503_detail
from src.api.main import app

PREFIX = "/api/__test_1999"

# status -> (detail, headers) raised by the throwaway route
RAISES: Dict[int, tuple] = {
    409: (
        "An assessment build for this review is still in progress; retry shortly.",
        {"Retry-After": "5"},
    ),
    503: ("model registry unreachable: connection refused to 10.0.0.7", {"Retry-After": "30"}),
    429: ("slow down", {"Retry-After": "60"}),
    404: ("Review abc was not found.", {"X-E2I-Test": "not-found"}),
    500: ("boom", {"X-E2I-Test": "server"}),
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    async def raise_with_headers(status: int):
        detail, headers = RAISES[status]
        raise HTTPException(status_code=status, detail=detail, headers=headers)

    async def raise_without_headers():
        raise HTTPException(status_code=409, detail="plain conflict")

    async def get_only():
        return {"ok": True}

    before = list(app.router.routes)
    app.add_api_route(f"{PREFIX}/raise/{{status}}", raise_with_headers, methods=["GET"])
    app.add_api_route(f"{PREFIX}/plain", raise_without_headers, methods=["GET"])
    app.add_api_route(f"{PREFIX}/get-only", get_only, methods=["GET"])
    added = [r for r in app.router.routes if r not in before]
    # Front of the router so no catch-all route can shadow them.
    app.router.routes[:] = added + before
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.router.routes[:] = before


@pytest.mark.unit
def test_409_retry_after_reaches_the_client_with_the_conflict_envelope(client):
    resp = client.get(f"{PREFIX}/raise/409")
    assert resp.status_code == 409, resp.text
    assert resp.headers.get("retry-after") == "5"
    body = resp.json()
    assert body["error"] == "ConflictError"
    assert body["message"] == RAISES[409][0]


@pytest.mark.unit
def test_503_retry_after_is_forwarded_while_the_raw_detail_stays_masked(client):
    resp = client.get(f"{PREFIX}/raise/503")
    assert resp.status_code == 503, resp.text
    assert resp.headers.get("retry-after") == "30"
    # The masking branch is about DETAIL text, which is unchanged.
    assert "10.0.0.7" not in resp.text


@pytest.mark.unit
def test_503_marked_user_safe_keeps_its_message_and_its_header(client, monkeypatch):
    monkeypatch.setitem(
        RAISES,
        503,
        (user_safe_503_detail("Store unavailable. Retry shortly."), {"Retry-After": "30"}),
    )
    resp = client.get(f"{PREFIX}/raise/503")
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "30"
    assert resp.json()["message"] == "Store unavailable. Retry shortly."


@pytest.mark.unit
def test_429_retry_after_is_forwarded(client):
    resp = client.get(f"{PREFIX}/raise/429")
    assert resp.status_code == 429, resp.text
    assert resp.headers.get("retry-after") == "60"
    assert resp.json()["error"] == "RateLimitError"


@pytest.mark.unit
def test_404_status_handler_forwards_headers_and_keeps_the_detail(client):
    resp = client.get(f"{PREFIX}/raise/404")
    assert resp.status_code == 404, resp.text
    assert resp.headers.get("x-e2i-test") == "not-found"
    assert resp.json()["message"] == RAISES[404][0]


@pytest.mark.unit
def test_500_http_exception_forwards_headers(client):
    resp = client.get(f"{PREFIX}/raise/500")
    assert resp.status_code == 500, resp.text
    assert resp.headers.get("x-e2i-test") == "server"


@pytest.mark.unit
def test_starlette_405_keeps_its_allow_header(client):
    """RFC 9110 15.5.6: a 405 MUST carry ``Allow``; Starlette's router sets it on
    the HTTPException it raises for a method mismatch."""
    resp = client.post(f"{PREFIX}/get-only")
    assert resp.status_code == 405, resp.text
    assert resp.headers.get("allow") == "GET"


@pytest.mark.unit
def test_unmatched_route_404_is_unchanged(client):
    resp = client.get(f"{PREFIX}/no-such-route")
    assert resp.status_code == 404
    assert resp.json()["error"] == "EndpointNotFoundError"


@pytest.mark.unit
def test_expert_review_store_outage_503_carries_retry_after_through_the_real_app(monkeypatch):
    """The expert-review routes' safe 503 now sends the ``Retry-After`` its body
    already promises (DependencyError: "try again in 30 seconds"); the route
    comment named the dropped header as the only reason it was absent.

    The real route and handler on the real app; only the store is replaced by
    one whose read fails (the outage), as in test_expert_review_detail_route.py."""
    import src.api.routes.expert_review as route_mod
    from src.api.dependencies.auth import require_operator

    class _DownRepo:
        async def get_review_summary(self, brand=None):
            raise ConnectionError("connection refused")

    async def _factory():
        return _DownRepo()

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _factory)
    app.dependency_overrides[require_operator] = lambda: {"id": "op-1", "role": "admin"}
    try:
        resp = TestClient(app, raise_server_exceptions=False).get("/api/expert-reviews/summary")
    finally:
        app.dependency_overrides.pop(require_operator, None)

    assert resp.status_code == 503, resp.text
    assert resp.headers.get("retry-after") == "30"
    body = resp.json()
    assert body["message"] == "Expert-review store unavailable. Retry shortly."
    assert "30 seconds" in body["suggested_action"]


@pytest.mark.unit
def test_raise_without_headers_adds_no_retry_after(client):
    """Positive control: the header comes from the exception, not the handler."""
    resp = client.get(f"{PREFIX}/plain")
    assert resp.status_code == 409
    assert "retry-after" not in resp.headers
    assert resp.json()["message"] == "plain conflict"
