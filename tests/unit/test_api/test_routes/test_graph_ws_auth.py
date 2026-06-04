"""Tests for graph-stream WebSocket authentication.

These tests cover the auth handshake on ``@router.websocket("/stream")`` in
``src/api/routes/graph.py``. The HTTP-scoped auth middleware does NOT run on
WebSocket connections, so the endpoint must authenticate the handshake itself
(parity with PR #657, which removed graph data endpoints from the public HTTP
allowlist because they expose Patient/HCP PHI/PII).

Design (mirrors the project's INTENTIONAL fail-open posture):

* ``is_auth_enabled()`` True  -> a valid token is REQUIRED. Anonymous/invalid
  handshakes are rejected with close code 1008 (policy violation) BEFORE accept.
* ``is_auth_enabled()`` False -> ACCEPT without a token (fail-open preserved for
  testing mode / unconfigured Supabase, so dev + e2e keep working).

Token channel: the frontend (PR #679) cannot set WS headers from a browser, so
it carries the JWT in ``Sec-WebSocket-Protocol`` as the two subprotocols
``['bearer', base64url(jwt)]`` (see ``frontend/src/hooks/use-websocket.ts``
``encodeTokenForSubprotocol``: ``btoa(token)`` with ``+``->``-``, ``/``->``_``,
and ``=`` padding stripped).
"""

import base64
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.api.dependencies import auth as auth_dep
from src.api.routes import graph as graph_route
from src.api.routes.graph import router


def _encode_token_for_subprotocol(token: str) -> str:
    """Mirror the frontend ``encodeTokenForSubprotocol`` exactly.

    ``btoa(token).replace(/\\+/g,'-').replace(/\\//g,'_').replace(/=+$/,'')``
    """
    b = base64.b64encode(token.encode("latin-1")).decode("ascii")
    return b.replace("+", "-").replace("/", "_").rstrip("=")


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# Token whose STANDARD base64 is "YWI+YWI/eA==" -> it contains BOTH '+' and '/'
# AND '=' padding, so the url-safe '+'->'-' / '/'->'_' substitution and the
# padding-restoration on decode are all genuinely exercised. Deliberately NOT a
# JWT-shaped literal so secret scanners (Gitleaks/GitGuardian) don't false-
# positive on a fake credential — verify_supabase_token is mocked, so the only
# property that matters here is a faithful encode/decode round-trip.
VALID_TOKEN = "ab>ab?x"
VALID_USER = {
    "id": "user-1",
    "email": "u@example.com",
    "role": "authenticated",
    "app_metadata": {"role": "analyst"},
}


class TestGraphStreamAuthEnabled:
    """Auth ENABLED: a valid bearer-subprotocol token is required."""

    def test_no_token_subprotocol_is_rejected(self, client):
        """Anonymous handshake (no subprotocols) -> rejected with close 1008."""
        with patch.object(graph_route, "is_auth_enabled", return_value=True):
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect("/graph/stream"):
                    pass  # pragma: no cover - should never accept
        assert exc.value.code == 1008

    def test_valid_token_subprotocol_accepts_and_session_works(self, client):
        """Valid token -> accepts; subscription message gets a confirmation."""
        verify = AsyncMock(return_value=VALID_USER)
        with (
            patch.object(graph_route, "is_auth_enabled", return_value=True),
            patch.object(auth_dep, "verify_supabase_token", verify),
        ):
            encoded = _encode_token_for_subprotocol(VALID_TOKEN)
            with client.websocket_connect("/graph/stream", subprotocols=["bearer", encoded]) as ws:
                ws.send_json({"entity_types": ["HCP"]})
                data = ws.receive_json()
                assert data["type"] == "subscription_updated"
        # The handshake actually verified the DECODED token (not the encoded form).
        verify.assert_awaited_once_with(VALID_TOKEN)

    def test_invalid_token_is_rejected(self, client):
        """Garbage token that fails verification -> rejected with close 1008."""
        verify = AsyncMock(return_value=None)
        with (
            patch.object(graph_route, "is_auth_enabled", return_value=True),
            patch.object(auth_dep, "verify_supabase_token", verify),
        ):
            encoded = _encode_token_for_subprotocol("not-a-real-token")
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect("/graph/stream", subprotocols=["bearer", encoded]):
                    pass  # pragma: no cover - should never accept
        assert exc.value.code == 1008

    def test_malformed_base64url_is_handled_gracefully(self, client):
        """Bad base64url padding -> rejected (1008), never a 500/crash."""
        verify = AsyncMock(return_value=VALID_USER)
        with (
            patch.object(graph_route, "is_auth_enabled", return_value=True),
            patch.object(auth_dep, "verify_supabase_token", verify),
        ):
            # "!" is not a valid base64url alphabet character -> decode raises.
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect(
                    "/graph/stream", subprotocols=["bearer", "!!!not-base64!!!"]
                ):
                    pass  # pragma: no cover - should never accept
        assert exc.value.code == 1008
        # Verification must never even be attempted on undecodable input.
        verify.assert_not_awaited()

    def test_oversized_token_is_rejected_without_decoding(self, client):
        """A token exceeding the length bound -> rejected (1008), never decoded.

        Defense-in-depth: a huge subprotocol value must short-circuit BEFORE the
        buffer allocation / decode, and never reach verification.
        """
        verify = AsyncMock(return_value=VALID_USER)
        with (
            patch.object(graph_route, "is_auth_enabled", return_value=True),
            patch.object(auth_dep, "verify_supabase_token", verify),
        ):
            oversized = "A" * (auth_dep._MAX_SUBPROTOCOL_TOKEN_LEN + 1)
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect("/graph/stream", subprotocols=["bearer", oversized]):
                    pass  # pragma: no cover - should never accept
        assert exc.value.code == 1008
        verify.assert_not_awaited()


class TestGraphStreamAuthDisabled:
    """Auth DISABLED (testing mode / Supabase unset): fail-open preserved."""

    def test_no_token_accepts_when_auth_disabled(self, client):
        """No token + auth disabled -> accepts WITHOUT a token (fail-open)."""
        with patch.object(graph_route, "is_auth_enabled", return_value=False):
            with client.websocket_connect("/graph/stream") as ws:
                ws.send_json({"entity_types": ["HCP"]})
                data = ws.receive_json()
                assert data["type"] == "subscription_updated"
