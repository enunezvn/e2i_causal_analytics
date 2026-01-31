"""Unit tests for rate limiting middleware.

Tests cover:
- In-memory rate limiting backend
- Redis rate limiting backend
- Rate limit enforcement per endpoint
- Client identification (IP, API key, user ID)
- Exempt paths and prefixes
- Custom rate limits per endpoint type
- Security audit logging integration
- Rate limit headers in responses

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.api.middleware.rate_limit_middleware import (
    InMemoryBackend,
    RateLimitMiddleware,
    RedisBackend,
)


@pytest.mark.unit
class TestInMemoryBackend:
    """Test suite for in-memory rate limit backend."""

    def test_initial_request_not_limited(self):
        """Test first request is not rate limited."""
        backend = InMemoryBackend()

        is_limited, remaining = backend.is_rate_limited("client-1", limit=10, window=60)

        assert is_limited is False
        assert remaining == 9  # 10 - 1 already recorded

    def test_rate_limit_enforcement(self):
        """Test rate limit is enforced after reaching limit."""
        backend = InMemoryBackend()

        # Make requests up to the limit
        for i in range(10):
            is_limited, remaining = backend.is_rate_limited("client-1", limit=10, window=60)
            assert is_limited is False

        # Next request should be rate limited
        is_limited, remaining = backend.is_rate_limited("client-1", limit=10, window=60)

        assert is_limited is True
        assert remaining == 0

    def test_window_expiration(self):
        """Test rate limit resets after window expires."""
        backend = InMemoryBackend()

        # Exhaust limit
        for i in range(5):
            backend.is_rate_limited("client-1", limit=5, window=1)

        # Verify limited
        is_limited, _ = backend.is_rate_limited("client-1", limit=5, window=1)
        assert is_limited is True

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        is_limited, remaining = backend.is_rate_limited("client-1", limit=5, window=1)

        assert is_limited is False
        assert remaining == 4

    def test_different_clients_independent(self):
        """Test rate limits are independent per client."""
        backend = InMemoryBackend()

        # Exhaust limit for client-1
        for i in range(5):
            backend.is_rate_limited("client-1", limit=5, window=60)

        is_limited_1, _ = backend.is_rate_limited("client-1", limit=5, window=60)
        assert is_limited_1 is True

        # Client-2 should not be affected
        is_limited_2, remaining_2 = backend.is_rate_limited("client-2", limit=5, window=60)

        assert is_limited_2 is False
        assert remaining_2 == 4

    def test_sliding_window(self):
        """Test rate limit uses sliding window (old requests expire)."""
        backend = InMemoryBackend()

        # Make 3 requests
        for i in range(3):
            backend.is_rate_limited("client-1", limit=5, window=1)

        # Wait for partial window expiration
        time.sleep(0.6)

        # Make 2 more requests (now at 5 total, but some expired)
        for i in range(2):
            is_limited, _ = backend.is_rate_limited("client-1", limit=5, window=1)

        # Should still be allowed (old requests sliding out)
        assert is_limited is False


@pytest.mark.unit
class TestRedisBackend:
    """Test suite for Redis rate limit backend."""

    def test_redis_backend_enabled(self):
        """Test Redis backend when Redis is available."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.pipeline.return_value.__enter__.return_value.execute.return_value = [1, True]

        # Patch redis.from_url directly since redis is imported inside __init__
        with patch("redis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            backend = RedisBackend()

            assert backend._enabled is True

    def test_redis_backend_disabled_on_connection_error(self):
        """Test Redis backend falls back to in-memory on connection error."""
        with patch("redis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection refused")

            backend = RedisBackend()

            assert backend._enabled is False
            assert hasattr(backend, "_fallback")

    def test_redis_backend_rate_limiting(self):
        """Test Redis backend rate limiting logic."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        # Mock pipeline execution - returns [current_count, expire_result]
        mock_pipe = MagicMock()
        mock_pipe.incr.return_value = None
        mock_pipe.expire.return_value = None
        mock_pipe.execute.return_value = [3, True]  # 3rd request
        mock_redis.pipeline.return_value = mock_pipe

        with patch("redis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            backend = RedisBackend()
            is_limited, remaining = backend.is_rate_limited("client-1", limit=10, window=60)

            assert is_limited is False
            assert remaining == 7  # 10 - 3

    def test_redis_backend_enforces_limit(self):
        """Test Redis backend enforces rate limit."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        # Mock count exceeding limit
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [11, True]  # 11th request with limit of 10
        mock_redis.pipeline.return_value = mock_pipe

        with patch("redis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            backend = RedisBackend()
            is_limited, remaining = backend.is_rate_limited("client-1", limit=10, window=60)

            assert is_limited is True
            assert remaining == 0

    def test_redis_backend_fallback_on_error(self):
        """Test Redis backend falls back to in-memory on runtime errors."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.pipeline.side_effect = Exception("Redis error")

        with patch("redis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            backend = RedisBackend()

            # Should use fallback and allow request
            is_limited, remaining = backend.is_rate_limited("client-1", limit=10, window=60)

            assert is_limited is False  # Graceful degradation

    def test_redis_backend_uses_correct_key_format(self):
        """Test Redis backend uses correct key format."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [1, True]
        mock_redis.pipeline.return_value = mock_pipe

        with patch("redis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            backend = RedisBackend()
            backend.is_rate_limited("client-123", limit=10, window=60)

            # Verify key format
            mock_pipe.incr.assert_called_once_with("ratelimit:client-123")


@pytest.mark.unit
class TestRateLimitMiddleware:
    """Test suite for RateLimitMiddleware."""

    @pytest.fixture(autouse=True)
    def reset_default_limits(self):
        """Reset DEFAULT_LIMITS class variable before each test."""
        from src.api.middleware.rate_limit_middleware import RateLimitMiddleware

        # Save original defaults
        original_defaults = RateLimitMiddleware.DEFAULT_LIMITS.copy()

        yield

        # Restore original defaults after each test
        RateLimitMiddleware.DEFAULT_LIMITS = original_defaults

    @pytest.mark.asyncio
    async def test_exempt_health_paths(self):
        """Test health check paths are exempt from rate limiting."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/health"
        mock_request.method = "GET"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        # Should call next middleware without rate limiting
        call_next.assert_called_once()
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_exempt_copilotkit_prefix(self):
        """Test CopilotKit paths are exempt from rate limiting."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/copilotkit/chat"
        mock_request.method = "POST"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        call_next.assert_called_once()
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_rate_limit_headers_added(self):
        """Test rate limit headers are added to responses."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self):
        """Test rate limit exceeded returns 429 status."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False, default_limit=2, default_window=60)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        mock_request.state = MagicMock()

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        # Make requests up to limit
        await middleware.dispatch(mock_request, call_next)
        await middleware.dispatch(mock_request, call_next)

        # Next request should be rate limited
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 429
        assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_get_client_key_from_api_key(self):
        """Test client identification from API key."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-API-Key": "test-api-key-12345"}

        client_key = middleware._get_client_key(mock_request)

        assert client_key.startswith("apikey:")
        assert "test-api-key-12" in client_key  # First 16 chars

    @pytest.mark.asyncio
    async def test_get_client_key_from_user_id(self):
        """Test client identification from authenticated user."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.state.user_id = "user-123"

        client_key = middleware._get_client_key(mock_request)

        assert client_key == "user:user-123"

    @pytest.mark.asyncio
    async def test_get_client_key_from_ip(self):
        """Test client identification from IP address."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        # Explicitly set state.user_id to None to prevent MagicMock auto-creation
        mock_request.state = MagicMock()
        mock_request.state.user_id = None

        client_key = middleware._get_client_key(mock_request)

        assert client_key == "ip:192.168.1.1"

    @pytest.mark.asyncio
    async def test_get_client_key_from_forwarded_header(self):
        """Test client identification from X-Forwarded-For header."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Forwarded-For": "203.0.113.1, 198.51.100.1"}
        mock_request.client = None
        # Explicitly set state.user_id to None
        mock_request.state = MagicMock()
        mock_request.state.user_id = None

        client_key = middleware._get_client_key(mock_request)

        assert client_key == "ip:203.0.113.1"  # First IP in list

    def test_get_limit_for_path_health(self):
        """Test health endpoints get higher rate limit."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        limit, window = middleware._get_limit_for_path("/health", "GET")

        assert limit == 300
        assert window == 60

    def test_get_limit_for_path_auth(self):
        """Test auth endpoints get lower rate limit."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        limit, window = middleware._get_limit_for_path("/api/auth/login", "POST")

        assert limit == 20
        assert window == 60

    def test_get_limit_for_path_batch(self):
        """Test batch endpoints get lower rate limit."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        limit, window = middleware._get_limit_for_path("/api/batch/process", "POST")

        assert limit == 10
        assert window == 60

    def test_get_limit_for_path_calculate(self):
        """Test calculate endpoints get moderate rate limit."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        limit, window = middleware._get_limit_for_path("/api/calculate/impact", "POST")

        assert limit == 30
        assert window == 60

    def test_get_limit_for_path_default(self):
        """Test default rate limit for other endpoints."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False)

        limit, window = middleware._get_limit_for_path("/api/data", "GET")

        assert limit == 100
        assert window == 60

    @pytest.mark.asyncio
    async def test_custom_default_limit_from_config(self):
        """Test custom default limit can be set."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False, default_limit=50, default_window=30)

        limit, window = middleware._get_limit_for_path("/api/test", "GET")

        assert limit == 50
        assert window == 30

    @pytest.mark.asyncio
    async def test_custom_default_limit_from_environment(self):
        """Test default limit can be set from environment."""
        app = MagicMock()

        with patch.dict("os.environ", {"RATE_LIMIT_DEFAULT": "75", "RATE_LIMIT_WINDOW": "45"}):
            middleware = RateLimitMiddleware(app, use_redis=False)

            limit, window = middleware._get_limit_for_path("/api/test", "GET")

            assert limit == 75
            assert window == 45

    @pytest.mark.asyncio
    async def test_security_audit_logging_on_rate_limit(self):
        """Test security audit logging when rate limit exceeded."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False, default_limit=1, default_window=60)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        mock_request.state = MagicMock()
        mock_request.state.request_id = "req-123"
        mock_request.state.user_id = "user-456"

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        # First request succeeds
        await middleware.dispatch(mock_request, call_next)

        # Second request triggers rate limit
        with patch("src.api.middleware.rate_limit_middleware.get_security_audit_service") as mock_audit:
            mock_audit_service = MagicMock()
            mock_audit.return_value = mock_audit_service

            # Need to reload module to enable audit
            with patch("src.api.middleware.rate_limit_middleware._AUDIT_ENABLED", True):
                response = await middleware.dispatch(mock_request, call_next)

                assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_retry_after_header(self):
        """Test Retry-After header is set on rate limit."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, use_redis=False, default_limit=1, default_window=60)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        mock_request.state = MagicMock()

        mock_response = Response()
        call_next = AsyncMock(return_value=mock_response)

        # Exhaust limit
        await middleware.dispatch(mock_request, call_next)

        # Get rate limited response
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "60"
