"""The global 503 handler surfaces ONLY opt-in (user-safe-marked) details.

Regression (ai-insights CATE): the handler masked every HTTPException(503) as a
generic "Dependency 'service' is unavailable", hiding the causal pipeline's honest
"no real data backend wired ... pass demo_mode=true". A naive fix that promoted
EVERY 503 detail would leak raw exception text from sites that build
``detail=f"...: {e}"`` (explain.py / predictions.py). So surfacing is OPT-IN: an
endpoint marks a curated, exception-free detail via ``errors.user_safe_503_detail``;
raw/unmarked details stay masked. These tests pin both halves.

The real app handler is imported lazily inside the helper so collection stays
cheap; this runs in the CI integration lane.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

_GENERIC_503_MESSAGE = "Dependency 'service' is unavailable"


def _client_with_503_route(detail):
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from src.api.main import http_exception_handler

    app = FastAPI()
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    @app.get("/boom")
    async def boom():
        raise HTTPException(status_code=503, detail=detail)

    return TestClient(app, raise_server_exceptions=False)


def test_503_surfaces_user_safe_marked_detail_stripped():
    from src.api.errors import user_safe_503_detail

    reason = (
        "Causal pipeline endpoints have no real data backend wired. "
        "Pass demo_mode=true to get a clearly-labeled pinned-zero placeholder."
    )
    resp = _client_with_503_route(user_safe_503_detail(reason)).get("/boom")
    assert resp.status_code == 503
    body = resp.json()
    # The curated reason is surfaced verbatim, with the marker stripped.
    assert body["message"] == reason
    assert "\x1e" not in body["message"]
    assert body.get("category") == "dependency_error"


def test_503_masks_raw_unmarked_detail():
    # The dangerous case: a raw exception-bearing detail must NOT be echoed to clients.
    raw = "Feature store lookup failed: <psycopg2 connection refused host=db.internal port=5432>"
    resp = _client_with_503_route(raw).get("/boom")
    assert resp.status_code == 503
    body = resp.json()
    assert raw not in body["message"]
    assert "psycopg2" not in body["message"]
    assert body["message"] == _GENERIC_503_MESSAGE


def test_503_without_detail_is_generic():
    resp = _client_with_503_route(None).get("/boom")
    assert resp.status_code == 503
    assert resp.json()["message"] == _GENERIC_503_MESSAGE
