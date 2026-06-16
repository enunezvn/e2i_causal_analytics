"""The global 503 handler must surface the endpoint's honest reason.

Regression (ai-insights CATE): ``src/api/main.py`` mapped every
``HTTPException(503)`` to a generic ``"Dependency 'service' is unavailable"``,
discarding the endpoint's real detail (e.g. the causal pipeline's "no real data
backend wired ... pass demo_mode=true"). That is why the Heterogeneous Treatment
Effects card surfaced a misleading "service unavailable". The fix preserves
``exc.detail`` as the user-facing message.

The real app handler is imported lazily inside the helper so test collection
stays cheap; this runs in the CI integration lane.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


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


def test_503_preserves_endpoint_detail():
    detail = (
        "Causal pipeline endpoints have no real data backend wired. "
        "Pass demo_mode=true to get a clearly-labeled pinned-zero placeholder."
    )
    resp = _client_with_503_route(detail).get("/boom")
    assert resp.status_code == 503
    body = resp.json()
    # The honest reason is surfaced, NOT masked behind the generic message.
    assert body["message"] == detail
    assert body["message"] != "Dependency 'service' is unavailable"


def test_503_still_categorized_as_dependency_error():
    # The error envelope is unchanged (still a 503 dependency error); only the
    # human-readable message now carries the endpoint's real reason.
    detail = "Graph service unavailable"
    resp = _client_with_503_route(detail).get("/boom")
    assert resp.status_code == 503
    body = resp.json()
    assert body.get("category") == "dependency_error"
    assert body["message"] == detail
