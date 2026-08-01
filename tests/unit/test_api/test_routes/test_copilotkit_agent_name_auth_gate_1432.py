"""#1432: the CopilotKit ``agent/{name}`` sub-path must go through the same
JWT gate as the ``agent/run`` root, not fall through to the third-party SDK
handler ungated.

``copilotkit_custom_handler`` wires ``_require_auth_for_copilotkit_execution``
only into the ``path in ("", "info")`` root-POST branch. A request to
``POST /api/copilotkit/agent/{name}`` (path is neither "" nor "info") skips that
branch and reaches ``sdk_handler`` -> ``agent.execute()`` with no Authorization
check. The ``JWTAuthMiddleware`` allowlist already marks the sub-path non-public,
so this is defense-in-depth (and a docstring-correctness fix), but the handler
itself must enforce auth so its claim to gate "anything else before the SDK
handler" is actually true.

These tests exercise the handler directly (below the middleware) with a real
Starlette ``Request`` over a complete ASGI scope — the handler is the unit under
test and the fix lives in the handler. Auth symbols are patched on the module
directly (``verify_supabase_token`` / ``set_authenticated_user``) rather than
relying on a contextvar surviving an async boundary (this repo has known
pytest-asyncio contextvar fragility).
"""

from unittest.mock import MagicMock

import pytest
from fastapi.responses import JSONResponse
from starlette.requests import Request

_OWNER = "33333333-3333-3333-3333-333333333333"


def _make_request(
    method: str, path: str, headers: dict | None = None, body: bytes = b""
) -> Request:
    """Build a real Starlette Request over a COMPLETE ASGI scope.

    A complete scope (not a partial Mock/SimpleNamespace) is what the handler
    needs — it reads ``.method``, ``.headers``, ``.scope``, ``.path_params`` and
    awaits ``.body()``. Incomplete hand-crafted scopes are the fragile ones that
    crash xdist workers; this one carries every field the handler touches.
    """
    raw = f"/api/copilotkit/{path}".encode()
    header_list = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": f"/api/copilotkit/{path}",
        "raw_path": raw,
        "query_string": b"",
        "headers": header_list,
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
        "root_path": "",
        "path_params": {"path": path},
        "state": {},
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


@pytest.mark.asyncio
async def test_unauthenticated_agent_name_is_gated_not_delegated(monkeypatch):
    """RED before fix: an unauthenticated ``POST agent/{name}`` reaches the SDK
    handler ungated. GREEN after: it returns 401 and never touches sdk_handler."""
    import src.api.routes.copilotkit as ck

    monkeypatch.setattr(ck, "TESTING_MODE", False)

    called = {"sdk": False}

    async def _fake_sdk_handler(request, sdk):
        called["sdk"] = True
        return JSONResponse(content={"reached": "execution"})

    monkeypatch.setattr(ck, "sdk_handler", _fake_sdk_handler)

    request = _make_request("POST", "agent/default", headers={}, body=b'{"messages":[]}')
    response = await ck.copilotkit_custom_handler(request, MagicMock(), path="agent/default")

    assert called["sdk"] is False, "unauthenticated agent/{name} reached the SDK handler ungated"
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_authenticated_agent_name_still_reaches_execution_and_stashes_owner(monkeypatch):
    """Companion: an AUTHENTICATED ``agent/{name}`` request must still reach the
    SDK handler AND stash the verified owner for attribution — proving the gate
    did not break the legitimate (token-bearing) path the demo scripts use."""
    import src.api.routes.copilotkit as ck

    monkeypatch.setattr(ck, "TESTING_MODE", False)

    async def _verify(token):
        assert token == "good-token"
        return {"id": _OWNER, "email": "owner@example.com"}

    monkeypatch.setattr(ck, "verify_supabase_token", _verify)

    stashed = {}

    def _capture_stash(uid):
        stashed["uid"] = uid

    monkeypatch.setattr(ck, "set_authenticated_user", _capture_stash)

    called = {"sdk": False}

    async def _fake_sdk_handler(request, sdk):
        called["sdk"] = True
        return JSONResponse(content={"reached": "execution"})

    monkeypatch.setattr(ck, "sdk_handler", _fake_sdk_handler)

    request = _make_request(
        "POST",
        "agent/default",
        headers={"Authorization": "Bearer good-token"},
        body=b'{"messages":[]}',
    )
    response = await ck.copilotkit_custom_handler(request, MagicMock(), path="agent/default")

    assert called["sdk"] is True, "authenticated agent/{name} must still reach execution"
    assert stashed["uid"] == _OWNER, "verified owner must be stashed for chat attribution"
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_known_good_identity_skips_duplicate_auth_roundtrip(monkeypatch):
    """FAIL-SAFE guard: when identity is already established (``request.state.user``
    set — e.g. the root-POST branch authenticated then fell through on a later
    exception), the sub-path gate must NOT re-run the auth check (no duplicate
    Supabase round-trip) yet must still reach execution. Unknown identity still
    runs the gate (covered by the unauthenticated test above)."""
    import src.api.routes.copilotkit as ck

    monkeypatch.setattr(ck, "TESTING_MODE", False)

    async def _verify_must_not_be_called(token):
        raise AssertionError("re-auth ran despite known-good request.state.user")

    monkeypatch.setattr(ck, "verify_supabase_token", _verify_must_not_be_called)

    called = {"sdk": False}

    async def _fake_sdk_handler(request, sdk):
        called["sdk"] = True
        return JSONResponse(content={"reached": "execution"})

    monkeypatch.setattr(ck, "sdk_handler", _fake_sdk_handler)

    request = _make_request("POST", "agent/default", headers={}, body=b'{"messages":[]}')
    request.state.user = {"id": _OWNER}  # identity already established upstream

    response = await ck.copilotkit_custom_handler(request, MagicMock(), path="agent/default")

    assert called["sdk"] is True, "known-good identity must still reach execution"
    assert response.status_code == 200
