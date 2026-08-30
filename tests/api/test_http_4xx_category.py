"""In-app 400/409/422 HTTPExceptions carry a CLIENT error category (#1831).

Regression: ``http_exception_handler`` (src.api.main) maps 401/403/404/429/503/504
to typed E2IError subclasses and drops every other status into a bare
``E2IError`` — whose class defaults are ``category=INTERNAL`` (documented as a
5xx category) and ``severity=MEDIUM``. So every deliberate in-app
``raise HTTPException(status_code=400, detail="...")`` (22 sites across
src/api/routes at the time of writing) rendered as::

    {"error": "E2IError", "category": "internal", "message": "<detail>", ...}

The status code was right; the envelope said "server error". 404 (#1814) and
503 (#1816) had already been corrected; this pins the remaining client codes
and keeps a 5xx positive control so the generic branch still says ``internal``
where that IS the truth.

Same shape as tests/api/test_http_404_detail_preserved.py: the real handler is
registered on a throwaway app (lazy import keeps collection cheap; CI
integration lane).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _client_raising(status_code: int, detail):
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from src.api.main import http_exception_handler

    app = FastAPI()
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    @app.get("/things/act")
    async def act():
        if detail is ...:
            raise HTTPException(status_code=status_code)
        raise HTTPException(status_code=status_code, detail=detail)

    return TestClient(app, raise_server_exceptions=False)


def test_400_is_a_validation_error_and_preserves_detail():
    detail = "'treatment_initiated -> persistent_180d' is not a modeled causal question"
    resp = _client_raising(400, detail).get("/things/act")
    assert resp.status_code == 400
    body = resp.json()
    assert body == {
        **body,
        "error": "ValidationError",
        "category": "validation",
        "message": detail,
    }


def test_422_in_app_raise_matches_the_request_validation_category():
    # The RequestValidationError handler already answers 422 with
    # category=validation; an explicit in-app HTTPException(422) must agree.
    resp = _client_raising(422, "Unsupported scenario shape").get("/things/act")
    assert resp.status_code == 422
    body = resp.json()
    assert body["category"] == "validation"
    assert body["message"] == "Unsupported scenario shape"


def test_409_is_a_conflict_error():
    resp = _client_raising(409, "Experiment already running").get("/things/act")
    assert resp.status_code == 409
    body = resp.json()
    assert body == {
        **body,
        "error": "ConflictError",
        "category": "conflict",
        "message": "Experiment already running",
    }


def test_unmapped_4xx_is_never_labelled_internal():
    # No in-app raise site uses 405; Starlette itself does. Whatever the
    # category, a client error must not be filed under the 5xx bucket.
    resp = _client_raising(405, ...).get("/things/act")
    assert resp.status_code == 405
    body = resp.json()
    assert body["category"] != "internal"
    assert body["message"]  # Starlette's default detail survives as the message


@pytest.mark.parametrize("status_code", [500, 502])
def test_5xx_positive_control_stays_internal(status_code):
    # The generic branch's "internal" label is CORRECT for server errors and
    # must survive the client-error mapping.
    resp = _client_raising(status_code, "upstream exploded").get("/things/act")
    assert resp.status_code == status_code
    body = resp.json()
    assert body["category"] == "internal"
    assert body["error"] == "E2IError"
    assert body["message"] == "upstream exploded"


def test_conflict_error_class_is_a_low_severity_409():
    from src.api.errors import ConflictError, ErrorCategory, ErrorSeverity

    err = ConflictError("Experiment already running")
    assert (err.status_code, err.category, err.severity) == (
        409,
        ErrorCategory.CONFLICT,
        ErrorSeverity.LOW,
    )
    assert err.message == "Experiment already running"
