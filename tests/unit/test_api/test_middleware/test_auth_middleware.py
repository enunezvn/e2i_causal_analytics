"""Unit tests for JWT authentication middleware.

Tests cover:
- Public path identification
- CopilotKit endpoints are public (rate-limited, not auth-gated)
- Public path pattern matching (regex)
- CORS headers on error responses
- Client info extraction
- Testing mode bypass
- Auth disabled bypass
- Token parsing and validation flow
- get_public_paths utility function

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.middleware.auth_middleware import (
    _get_client_info,
    _get_cors_headers,
    _is_public_path,
    get_public_paths,
)


@pytest.mark.unit
class TestIsPublicPath:
    """Test suite for _is_public_path function."""

    # =========================================================================
    # Exact match tests
    # =========================================================================

    def test_root_path_is_public(self):
        """Test root path is public."""
        assert _is_public_path("GET", "/") is True

    def test_health_endpoint_is_public(self):
        """Test /health is public for any method."""
        assert _is_public_path("GET", "/health") is True
        assert _is_public_path("POST", "/health") is True

    def test_healthz_endpoint_is_public(self):
        """Test /healthz is public."""
        assert _is_public_path("GET", "/healthz") is True

    def test_ready_endpoint_is_public(self):
        """Test /ready is public."""
        assert _is_public_path("GET", "/ready") is True

    def test_metrics_endpoint_is_public_get_only(self):
        """Test /metrics is public only for GET."""
        assert _is_public_path("GET", "/metrics") is True
        assert _is_public_path("POST", "/metrics") is False

    def test_docs_endpoint_is_public(self):
        """Test /api/docs is public."""
        assert _is_public_path("GET", "/api/docs") is True
        assert _is_public_path("POST", "/api/docs") is True

    def test_redoc_endpoint_is_public(self):
        """Test /api/redoc is public."""
        assert _is_public_path("GET", "/api/redoc") is True

    def test_openapi_endpoint_is_public(self):
        """Test /api/openapi.json is public."""
        assert _is_public_path("GET", "/api/openapi.json") is True

    # =========================================================================
    # Auth endpoints
    # =========================================================================

    def test_login_endpoint_is_public(self):
        """Test POST /api/auth/login is public."""
        assert _is_public_path("POST", "/api/auth/login") is True
        assert _is_public_path("GET", "/api/auth/login") is False

    def test_register_endpoint_is_public(self):
        """Test POST /api/auth/register is public."""
        assert _is_public_path("POST", "/api/auth/register") is True

    def test_refresh_endpoint_is_public(self):
        """Test POST /api/auth/refresh is public."""
        assert _is_public_path("POST", "/api/auth/refresh") is True

    # =========================================================================
    # CopilotKit endpoints — all public (rate-limited instead of auth-gated)
    # =========================================================================

    def test_copilotkit_root_is_public(self):
        """Test /api/copilotkit is public."""
        assert _is_public_path("POST", "/api/copilotkit") is True
        assert _is_public_path("GET", "/api/copilotkit") is True

    def test_copilotkit_status_is_public(self):
        """Test /api/copilotkit/status is public."""
        assert _is_public_path("GET", "/api/copilotkit/status") is True

    def test_copilotkit_info_is_public(self):
        """Test /api/copilotkit/info is public."""
        assert _is_public_path("GET", "/api/copilotkit/info") is True

    # =========================================================================
    # Protected paths
    # =========================================================================

    def test_arbitrary_api_path_is_not_public(self):
        """Test non-listed API paths require auth."""
        assert _is_public_path("GET", "/api/users") is False
        assert _is_public_path("POST", "/api/data/upload") is False

    def test_admin_path_is_not_public(self):
        """Test admin paths require auth."""
        assert _is_public_path("GET", "/api/admin/settings") is False

    # =========================================================================
    # Trailing slash normalization
    # =========================================================================

    def test_trailing_slash_normalized(self):
        """Test trailing slashes are stripped for matching."""
        assert _is_public_path("GET", "/health/") is True
        assert _is_public_path("GET", "/api/docs/") is True

    # =========================================================================
    # Pattern matching
    # =========================================================================

    def test_kpi_metadata_pattern_match(self):
        """Test KPI metadata dynamic route pattern matches."""
        assert _is_public_path("GET", "/api/kpis/revenue/metadata") is True
        assert _is_public_path("GET", "/api/kpis/churn_rate/metadata") is True

    def test_kpi_metadata_pattern_post_not_allowed(self):
        """Test KPI metadata pattern is GET-only."""
        assert _is_public_path("POST", "/api/kpis/revenue/metadata") is False


@pytest.mark.unit
class TestGetClientInfo:
    """Test suite for _get_client_info function."""

    def test_extracts_x_real_ip(self):
        """Test extraction of X-Real-IP header."""
        mock_request = MagicMock()
        mock_request.headers = {"X-Real-IP": "203.0.113.1", "User-Agent": "TestBot/1.0"}

        ip, ua = _get_client_info(mock_request)

        assert ip == "203.0.113.1"
        assert ua == "TestBot/1.0"

    def test_falls_back_to_client_host(self):
        """Test fallback to request.client.host when X-Real-IP missing."""
        mock_request = MagicMock()
        mock_request.headers = {"User-Agent": "TestBot/1.0"}
        mock_request.client.host = "192.168.1.1"

        ip, ua = _get_client_info(mock_request)

        assert ip == "192.168.1.1"

    def test_unknown_when_no_client(self):
        """Test 'unknown' when no client info available."""
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client = None

        ip, ua = _get_client_info(mock_request)

        assert ip == "unknown"

    def test_unknown_user_agent_default(self):
        """Test default user agent when header missing."""
        mock_request = MagicMock()
        mock_request.headers = {"X-Real-IP": "203.0.113.1"}

        ip, ua = _get_client_info(mock_request)

        assert ua == "unknown"


@pytest.mark.unit
class TestGetCorsHeaders:
    """Test suite for _get_cors_headers function."""

    def test_returns_cors_headers_for_allowed_origin(self):
        """Test CORS headers returned for allowed origin."""
        mock_request = MagicMock()
        mock_request.headers = {"origin": "http://localhost:5173"}

        headers = _get_cors_headers(mock_request)

        assert headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
        assert headers["Access-Control-Allow-Credentials"] == "true"

    def test_returns_empty_for_disallowed_origin(self):
        """Test empty headers for disallowed origin."""
        mock_request = MagicMock()
        mock_request.headers = {"origin": "https://evil.example.com"}

        headers = _get_cors_headers(mock_request)

        assert headers == {}

    def test_returns_empty_for_missing_origin(self):
        """Test empty headers when origin header is missing."""
        mock_request = MagicMock()
        mock_request.headers = {}

        headers = _get_cors_headers(mock_request)

        assert headers == {}


@pytest.mark.unit
class TestGetPublicPaths:
    """Test suite for get_public_paths function."""

    def test_returns_list(self):
        """Test get_public_paths returns a list."""
        paths = get_public_paths()
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_includes_health_endpoint(self):
        """Test health endpoint is in public paths."""
        paths = get_public_paths()
        assert "/health" in paths

    def test_includes_pattern_paths(self):
        """Test pattern paths are included with (pattern) suffix."""
        paths = get_public_paths()
        pattern_paths = [p for p in paths if "(pattern)" in p]
        assert len(pattern_paths) > 0


@pytest.mark.unit
class TestJWTAuthMiddleware:
    """Test suite for JWTAuthMiddleware dispatch."""

    @pytest.mark.asyncio
    async def test_public_path_bypasses_auth(self):
        """Test public paths bypass authentication entirely."""
        from src.api.middleware.auth_middleware import JWTAuthMiddleware

        app = MagicMock()
        middleware = JWTAuthMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/health"

        mock_response = MagicMock()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        call_next.assert_called_once_with(mock_request)
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_options_bypasses_auth(self):
        """Test OPTIONS (CORS preflight) bypasses authentication."""
        from src.api.middleware.auth_middleware import JWTAuthMiddleware

        app = MagicMock()
        middleware = JWTAuthMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "OPTIONS"
        mock_request.url.path = "/api/protected"

        mock_response = MagicMock()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        call_next.assert_called_once()
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_testing_mode_bypasses_auth(self):
        """Test testing mode bypasses authentication and sets mock user."""
        from src.api.middleware.auth_middleware import JWTAuthMiddleware

        app = MagicMock()
        middleware = JWTAuthMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/protected"
        mock_request.state = MagicMock()

        mock_response = MagicMock()
        call_next = AsyncMock(return_value=mock_response)

        with patch("src.api.middleware.auth_middleware.TESTING_MODE", True):
            await middleware.dispatch(mock_request, call_next)

        call_next.assert_called_once()
        # Verify mock user was set
        assert mock_request.state.user["id"] == "test-user-id"
        assert mock_request.state.user["role"] == "authenticated"

    @pytest.mark.asyncio
    async def test_auth_disabled_allows_request(self):
        """Test disabled auth allows unauthenticated requests."""
        from src.api.middleware.auth_middleware import JWTAuthMiddleware

        app = MagicMock()
        middleware = JWTAuthMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/protected"
        mock_request.headers = {}

        mock_response = MagicMock()
        call_next = AsyncMock(return_value=mock_response)

        with patch("src.api.middleware.auth_middleware.TESTING_MODE", False):
            with patch("src.api.middleware.auth_middleware.is_auth_enabled", return_value=False):
                await middleware.dispatch(mock_request, call_next)

        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_auth_header_returns_401(self):
        """Test missing Authorization header returns 401."""
        from src.api.middleware.auth_middleware import JWTAuthMiddleware

        app = MagicMock()
        middleware = JWTAuthMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/protected"
        mock_request.headers = {}

        call_next = AsyncMock()

        with patch("src.api.middleware.auth_middleware.TESTING_MODE", False):
            with patch("src.api.middleware.auth_middleware.is_auth_enabled", return_value=True):
                with patch("src.api.middleware.auth_middleware._AUDIT_ENABLED", False):
                    response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 401
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_auth_header_format_returns_401(self):
        """Test invalid Authorization header format returns 401."""
        from src.api.middleware.auth_middleware import JWTAuthMiddleware

        app = MagicMock()
        middleware = JWTAuthMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/protected"
        mock_request.headers = {"Authorization": "InvalidFormat"}

        call_next = AsyncMock()

        with patch("src.api.middleware.auth_middleware.TESTING_MODE", False):
            with patch("src.api.middleware.auth_middleware.is_auth_enabled", return_value=True):
                with patch("src.api.middleware.auth_middleware._AUDIT_ENABLED", False):
                    response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 401
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self):
        """Test invalid JWT token returns 401."""
        from src.api.middleware.auth_middleware import JWTAuthMiddleware

        app = MagicMock()
        middleware = JWTAuthMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/protected"
        mock_request.headers = {"Authorization": "Bearer invalid-token-123"}

        call_next = AsyncMock()

        with patch("src.api.middleware.auth_middleware.TESTING_MODE", False):
            with patch("src.api.middleware.auth_middleware.is_auth_enabled", return_value=True):
                with patch("src.api.middleware.auth_middleware._AUDIT_ENABLED", False):
                    with patch(
                        "src.api.middleware.auth_middleware.verify_supabase_token",
                        new_callable=AsyncMock,
                        return_value=None,
                    ):
                        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 401
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_token_allows_request(self):
        """Test valid JWT token allows the request and sets user on state."""
        from src.api.middleware.auth_middleware import JWTAuthMiddleware

        app = MagicMock()
        middleware = JWTAuthMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/protected"
        mock_request.headers = {"Authorization": "Bearer valid-token-abc"}
        mock_request.state = MagicMock()

        mock_user = {"id": "user-123", "email": "test@example.com", "role": "authenticated"}
        mock_response = MagicMock()
        call_next = AsyncMock(return_value=mock_response)

        with patch("src.api.middleware.auth_middleware.TESTING_MODE", False):
            with patch("src.api.middleware.auth_middleware.is_auth_enabled", return_value=True):
                with patch(
                    "src.api.middleware.auth_middleware.verify_supabase_token",
                    new_callable=AsyncMock,
                    return_value=mock_user,
                ):
                    response = await middleware.dispatch(mock_request, call_next)

        call_next.assert_called_once()
        assert mock_request.state.user == mock_user
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_copilotkit_endpoints_bypass_auth(self):
        """Test all CopilotKit endpoints are public (rate-limited, not auth-gated)."""
        from src.api.middleware.auth_middleware import JWTAuthMiddleware

        app = MagicMock()
        middleware = JWTAuthMiddleware(app)

        mock_response = MagicMock()
        call_next = AsyncMock(return_value=mock_response)

        for path in ["/api/copilotkit", "/api/copilotkit/status", "/api/copilotkit/info"]:
            mock_request = MagicMock()
            mock_request.method = "POST"
            mock_request.url.path = path

            response = await middleware.dispatch(mock_request, call_next)

            assert response == mock_response
