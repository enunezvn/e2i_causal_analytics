"""Unit tests for security headers middleware.

Tests cover:
- Security headers added to all responses
- X-Content-Type-Options (MIME sniffing prevention)
- X-Frame-Options (clickjacking prevention)
- X-XSS-Protection (legacy XSS protection)
- Strict-Transport-Security (HSTS)
- Content-Security-Policy (CSP)
- Referrer-Policy
- Permissions-Policy
- Cache-Control for sensitive endpoints
- HSTS configuration (environment-based and manual)

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import Response

from src.api.middleware.security_middleware import SecurityHeadersMiddleware


@pytest.mark.unit
class TestSecurityHeadersMiddleware:
    """Test suite for SecurityHeadersMiddleware."""

    @pytest.mark.asyncio
    async def test_basic_security_headers_added(self):
        """Test basic security headers are added to all responses."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app, enable_hsts=False)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        # Verify security headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "Permissions-Policy" in response.headers
        assert "Content-Security-Policy" in response.headers

    @pytest.mark.asyncio
    async def test_hsts_header_when_enabled(self):
        """Test HSTS header is added when enabled."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app, enable_hsts=True, hsts_max_age=31536000)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert "Strict-Transport-Security" in response.headers
        assert "max-age=31536000" in response.headers["Strict-Transport-Security"]
        assert "includeSubDomains" in response.headers["Strict-Transport-Security"]

    @pytest.mark.asyncio
    async def test_hsts_header_not_added_when_disabled(self):
        """Test HSTS header is not added when disabled."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app, enable_hsts=False)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert "Strict-Transport-Security" not in response.headers

    @pytest.mark.asyncio
    async def test_hsts_auto_detection_from_environment(self):
        """Test HSTS is auto-detected from environment."""
        app = MagicMock()

        # Test with HSTS enabled
        with patch.dict("os.environ", {"ENABLE_HSTS": "true"}):
            middleware = SecurityHeadersMiddleware(app)

            mock_request = MagicMock(spec=Request)
            mock_request.url.path = "/api/test"
            mock_request.method = "GET"

            mock_response = Response()
            call_next = AsyncMock(return_value=mock_response)

            response = await middleware.dispatch(mock_request, call_next)

            assert "Strict-Transport-Security" in response.headers

    @pytest.mark.asyncio
    async def test_hsts_auto_detection_defaults_to_false(self):
        """Test HSTS defaults to disabled if not in environment."""
        app = MagicMock()

        with patch.dict("os.environ", {}, clear=True):
            middleware = SecurityHeadersMiddleware(app)

            mock_request = MagicMock(spec=Request)
            mock_request.url.path = "/api/test"
            mock_request.method = "GET"

            mock_response = Response()
            call_next = AsyncMock(return_value=mock_response)

            response = await middleware.dispatch(mock_request, call_next)

            assert "Strict-Transport-Security" not in response.headers

    @pytest.mark.asyncio
    async def test_custom_hsts_max_age(self):
        """Test custom HSTS max-age can be set."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app, enable_hsts=True, hsts_max_age=7776000)  # 90 days

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert "max-age=7776000" in response.headers["Strict-Transport-Security"]

    @pytest.mark.asyncio
    async def test_default_csp_policy(self):
        """Test default Content-Security-Policy is applied."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        csp = response.headers["Content-Security-Policy"]

        # Verify key CSP directives
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "style-src 'self' 'unsafe-inline'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "base-uri 'self'" in csp
        assert "form-action 'self'" in csp

    @pytest.mark.asyncio
    async def test_custom_csp_policy(self):
        """Test custom Content-Security-Policy can be set."""
        app = MagicMock()
        custom_csp = "default-src 'none'; script-src 'self'; img-src *"
        middleware = SecurityHeadersMiddleware(app, csp_policy=custom_csp)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert response.headers["Content-Security-Policy"] == custom_csp

    @pytest.mark.asyncio
    async def test_permissions_policy_header(self):
        """Test Permissions-Policy header restricts browser features."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        permissions = response.headers["Permissions-Policy"]

        # Verify restricted features
        assert "accelerometer=()" in permissions
        assert "camera=()" in permissions
        assert "geolocation=()" in permissions
        assert "microphone=()" in permissions
        assert "payment=()" in permissions

    @pytest.mark.asyncio
    async def test_cache_control_for_api_post_requests(self):
        """Test Cache-Control headers for sensitive API POST requests."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/sensitive/data"
        mock_request.method = "POST"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert "Cache-Control" in response.headers
        assert "no-store" in response.headers["Cache-Control"]
        assert "no-cache" in response.headers["Cache-Control"]
        assert "must-revalidate" in response.headers["Cache-Control"]
        assert "Pragma" in response.headers
        assert response.headers["Pragma"] == "no-cache"

    @pytest.mark.asyncio
    async def test_no_cache_control_for_get_requests(self):
        """Test Cache-Control is not added for GET requests."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/data"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        # Cache-Control should not be added for GET requests
        # (unless already present in the response from the handler)
        if "Cache-Control" in response.headers:
            # If present, it should not be the no-store policy
            assert "no-store" not in response.headers["Cache-Control"]

    @pytest.mark.asyncio
    async def test_no_cache_control_for_non_api_paths(self):
        """Test Cache-Control is not added for non-API paths."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/docs"
        mock_request.method = "POST"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        # Cache-Control should not be added for non-/api/ paths
        if "Cache-Control" in response.headers:
            assert "no-store" not in response.headers["Cache-Control"]

    @pytest.mark.asyncio
    async def test_x_frame_options_deny(self):
        """Test X-Frame-Options is set to DENY to prevent clickjacking."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert response.headers["X-Frame-Options"] == "DENY"

    @pytest.mark.asyncio
    async def test_x_content_type_options_nosniff(self):
        """Test X-Content-Type-Options prevents MIME sniffing."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert response.headers["X-Content-Type-Options"] == "nosniff"

    @pytest.mark.asyncio
    async def test_x_xss_protection_enabled(self):
        """Test X-XSS-Protection is enabled for legacy browsers."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert response.headers["X-XSS-Protection"] == "1; mode=block"

    @pytest.mark.asyncio
    async def test_referrer_policy_set(self):
        """Test Referrer-Policy is set for privacy."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    @pytest.mark.asyncio
    async def test_all_headers_on_multiple_requests(self):
        """Test headers are consistently added across multiple requests."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app, enable_hsts=True)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        # Make multiple requests
        for _ in range(3):
            response = await middleware.dispatch(mock_request, call_next)

            # Verify all expected headers are present
            expected_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
                "Referrer-Policy",
                "Permissions-Policy",
                "Content-Security-Policy",
                "Strict-Transport-Security",
            ]

            for header in expected_headers:
                assert header in response.headers

    @pytest.mark.asyncio
    async def test_middleware_preserves_existing_response_headers(self):
        """Test middleware doesn't remove existing response headers."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        # Create response with existing headers
        mock_response = Response()
        mock_response.headers["X-Custom-Header"] = "custom-value"
        mock_response.headers["Content-Type"] = "application/json"

        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        # Verify existing headers are preserved
        assert response.headers["X-Custom-Header"] == "custom-value"
        assert response.headers["Content-Type"] == "application/json"

        # And security headers are added
        assert "X-Content-Type-Options" in response.headers
        assert "Content-Security-Policy" in response.headers

    @pytest.mark.asyncio
    async def test_csp_allows_inline_styles(self):
        """Test CSP allows inline styles for API docs."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/docs"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        csp = response.headers["Content-Security-Policy"]

        # Verify inline styles are allowed (for Swagger UI/docs)
        assert "'unsafe-inline'" in csp
        assert "style-src" in csp

    @pytest.mark.asyncio
    async def test_csp_prevents_framing(self):
        """Test CSP prevents page from being framed."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        csp = response.headers["Content-Security-Policy"]

        # Verify frame-ancestors is set to none
        assert "frame-ancestors 'none'" in csp

    @pytest.mark.asyncio
    async def test_middleware_calls_next_handler(self):
        """Test middleware properly calls the next handler in chain."""
        app = MagicMock()
        middleware = SecurityHeadersMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        mock_response = Response(content=b"test content", status_code=200)
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        # Verify next handler was called
        call_next.assert_called_once_with(mock_request)

        # Verify response content is preserved
        assert response.body == b"test content"
        assert response.status_code == 200
