"""Integration tests for Redis authentication.

Tests verify Redis client behavior with password-protected Redis instances:
- Connection with authentication
- Connection pooling and concurrency
- Rate limiting with authenticated backend
- Health checks with auth
- Error scenarios (wrong password, timeouts)

Prerequisites:
    - Redis running with REDIS_PASSWORD authentication
    - REDIS_URL environment variable configured

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import asyncio
import os
import time
from unittest.mock import patch

import pytest

# Check if Redis is available for integration tests
REDIS_AVAILABLE = False
REDIS_URL = os.environ.get("REDIS_URL", "")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

try:
    import redis

    if REDIS_URL:
        _test_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        _test_client.ping()
        REDIS_AVAILABLE = True
        _test_client.close()
except Exception:
    pass


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not REDIS_AVAILABLE,
        reason="Redis not available or not configured with authentication",
    ),
]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def redis_url():
    """Get the configured Redis URL."""
    return REDIS_URL


@pytest.fixture
def redis_password():
    """Get the configured Redis password."""
    return REDIS_PASSWORD


@pytest.fixture
async def clean_redis_client():
    """Provide a clean Redis client and clean up after test."""
    import src.api.dependencies.redis_client as redis_module

    # Reset global state
    original_client = redis_module._redis_client
    redis_module._redis_client = None

    yield

    # Cleanup
    if redis_module._redis_client is not None:
        await redis_module.close_redis()
    redis_module._redis_client = original_client


@pytest.fixture
def sync_redis_client():
    """Provide a synchronous Redis client for direct testing."""
    import redis

    client = redis.from_url(REDIS_URL, decode_responses=True)
    yield client
    client.close()


# =============================================================================
# CONNECTION WITH AUTHENTICATION TESTS
# =============================================================================


@pytest.mark.asyncio
class TestRedisConnectionWithAuth:
    """Tests for Redis connection with authentication."""

    async def test_init_redis_with_valid_auth(self, clean_redis_client):
        """Test that init_redis connects successfully with valid credentials."""
        from src.api.dependencies.redis_client import init_redis

        client = await init_redis()

        assert client is not None
        # Verify connection works
        pong = await client.ping()
        assert pong is True

    async def test_get_redis_returns_same_client(self, clean_redis_client):
        """Test that get_redis returns the same client instance."""
        from src.api.dependencies.redis_client import get_redis

        client1 = await get_redis()
        client2 = await get_redis()

        # Should be the same instance (connection pooling)
        assert client1 is client2

    async def test_redis_basic_operations_with_auth(self, clean_redis_client):
        """Test basic Redis operations work with authenticated connection."""
        from src.api.dependencies.redis_client import get_redis

        client = await get_redis()

        # Test SET/GET
        test_key = "test:auth:basic"
        test_value = "authenticated_value"

        await client.set(test_key, test_value, ex=60)
        result = await client.get(test_key)

        assert result == test_value

        # Cleanup
        await client.delete(test_key)

    async def test_close_redis_clears_connection(self, clean_redis_client):
        """Test that close_redis properly clears the connection."""
        import src.api.dependencies.redis_client as redis_module
        from src.api.dependencies.redis_client import close_redis, init_redis

        await init_redis()
        assert redis_module._redis_client is not None

        await close_redis()
        assert redis_module._redis_client is None


@pytest.mark.asyncio
class TestRedisConnectionPoolingWithAuth:
    """Tests for Redis connection pooling with authentication."""

    async def test_concurrent_connections_with_auth(self, clean_redis_client):
        """Test multiple concurrent operations on authenticated connection."""
        from src.api.dependencies.redis_client import get_redis

        client = await get_redis()

        # Run multiple concurrent operations
        async def set_and_get(i: int) -> str:
            key = f"test:concurrent:{i}"
            value = f"value_{i}"
            await client.set(key, value, ex=60)
            result = await client.get(key)
            await client.delete(key)
            return result

        # Execute 20 concurrent operations
        tasks = [set_and_get(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 20
        for i, result in enumerate(results):
            assert result == f"value_{i}"

    async def test_connection_reuse_across_operations(self, clean_redis_client):
        """Test that connection is reused across multiple operations."""
        from src.api.dependencies.redis_client import get_redis

        client = await get_redis()

        # Get connection info before
        info_before = await client.info("clients")
        connected_before = info_before.get("connected_clients", 0)

        # Perform many operations
        for i in range(50):
            key = f"test:reuse:{i}"
            await client.set(key, f"value_{i}", ex=60)
            await client.get(key)
            await client.delete(key)

        # Get connection info after
        info_after = await client.info("clients")
        connected_after = info_after.get("connected_clients", 0)

        # Connection count should not have increased significantly
        # (connection pooling should reuse connections)
        assert connected_after <= connected_before + 2


# =============================================================================
# HEALTH CHECK WITH AUTH TESTS
# =============================================================================


@pytest.mark.asyncio
class TestRedisHealthCheckWithAuth:
    """Tests for Redis health check with authentication."""

    async def test_health_check_returns_healthy(self, clean_redis_client):
        """Test that health check returns healthy status with auth."""
        from src.api.dependencies.redis_client import init_redis, redis_health_check

        await init_redis()

        result = await redis_health_check()

        assert result["status"] == "healthy"
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0
        assert "connected_clients" in result

    async def test_health_check_measures_latency(self, clean_redis_client):
        """Test that health check accurately measures latency."""
        from src.api.dependencies.redis_client import init_redis, redis_health_check

        await init_redis()

        # Run multiple health checks and verify latency is reasonable
        latencies = []
        for _ in range(5):
            result = await redis_health_check()
            if result["status"] == "healthy":
                latencies.append(result["latency_ms"])

        assert len(latencies) >= 3
        # Latency should be reasonable (< 100ms for local Redis)
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 100


# =============================================================================
# RATE LIMITING WITH AUTH TESTS
# =============================================================================


class TestRateLimitingWithAuth:
    """Tests for rate limiting with authenticated Redis backend."""

    def test_redis_backend_with_auth(self, sync_redis_client):
        """Test that RedisBackend works with authenticated connection."""
        from src.api.middleware.rate_limit_middleware import RedisBackend

        backend = RedisBackend(redis_url=REDIS_URL)

        # Should be enabled (not falling back to in-memory)
        assert backend._enabled is True

    def test_rate_limiting_tracks_requests(self, sync_redis_client):
        """Test that rate limiting correctly tracks requests with auth."""
        from src.api.middleware.rate_limit_middleware import RedisBackend

        backend = RedisBackend(redis_url=REDIS_URL)
        test_key = f"test:ratelimit:{time.time()}"

        # First request should not be limited
        is_limited, remaining = backend.is_rate_limited(test_key, limit=5, window=60)
        assert is_limited is False
        assert remaining == 4

        # Make more requests
        for i in range(4):
            is_limited, remaining = backend.is_rate_limited(test_key, limit=5, window=60)

        # Fifth request should hit the limit
        is_limited, remaining = backend.is_rate_limited(test_key, limit=5, window=60)
        assert is_limited is True
        assert remaining == 0

    def test_rate_limit_window_expiration(self, sync_redis_client):
        """Test that rate limits reset after window expires."""
        from src.api.middleware.rate_limit_middleware import RedisBackend

        backend = RedisBackend(redis_url=REDIS_URL)
        test_key = f"test:ratelimit:expiry:{time.time()}"

        # Use a 2-second window
        window = 2

        # Hit the limit
        for _ in range(3):
            backend.is_rate_limited(test_key, limit=3, window=window)

        # Should be limited now
        is_limited, _ = backend.is_rate_limited(test_key, limit=3, window=window)
        assert is_limited is True

        # Wait for window to expire
        time.sleep(window + 0.5)

        # Should not be limited anymore
        is_limited, remaining = backend.is_rate_limited(test_key, limit=3, window=window)
        assert is_limited is False
        assert remaining >= 1


# =============================================================================
# ERROR SCENARIO TESTS
# =============================================================================


@pytest.mark.asyncio
class TestRedisAuthErrorScenarios:
    """Tests for Redis authentication error scenarios."""

    async def test_wrong_password_raises_auth_error(self):
        """Test that wrong password raises authentication error."""
        # Parse the current URL to get host/port
        from urllib.parse import urlparse

        import redis.asyncio as aioredis

        parsed = urlparse(REDIS_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379

        # Build URL with wrong password
        wrong_url = f"redis://:wrong_password_12345@{host}:{port}"

        client = aioredis.from_url(wrong_url, decode_responses=True, socket_timeout=2)

        with pytest.raises(Exception) as exc_info:
            await client.ping()

        # Should be an auth error (different Redis versions may use different messages)
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ["auth", "noauth", "invalid", "wrong"])

        await client.close()

    async def test_init_redis_with_wrong_password_fails(self, clean_redis_client):
        """Test that init_redis fails gracefully with wrong password."""
        # Parse URL to get host/port
        from urllib.parse import urlparse

        import src.api.dependencies.redis_client as redis_module

        parsed = urlparse(REDIS_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379

        wrong_url = f"redis://:wrong_password@{host}:{port}"

        # Temporarily replace module constant
        original_url = redis_module.REDIS_URL

        with patch.object(redis_module, "REDIS_URL", wrong_url):
            # Reload to pick up patched URL
            import importlib

            importlib.reload(redis_module)
            redis_module.REDIS_URL = wrong_url
            redis_module._redis_client = None

            with pytest.raises(ConnectionError):
                await redis_module.init_redis()

        # Restore
        redis_module.REDIS_URL = original_url

    def test_rate_limit_backend_fallback_on_auth_failure(self):
        """Test that rate limit backend falls back to in-memory on auth failure."""
        from src.api.middleware.rate_limit_middleware import RedisBackend

        # Use invalid credentials
        backend = RedisBackend(redis_url="redis://:invalid_password@localhost:6379")

        # Should fall back to in-memory (not enabled for Redis)
        assert backend._enabled is False

        # Should still work via in-memory fallback
        is_limited, remaining = backend.is_rate_limited("test", limit=10, window=60)
        assert is_limited is False


@pytest.mark.asyncio
class TestRedisTimeoutHandling:
    """Tests for Redis connection timeout handling."""

    async def test_socket_timeout_configured(self, clean_redis_client):
        """Test that socket timeout is properly configured."""
        from src.api.dependencies.redis_client import REDIS_SOCKET_TIMEOUT

        # Should have a reasonable timeout
        assert REDIS_SOCKET_TIMEOUT > 0
        assert REDIS_SOCKET_TIMEOUT < 60

    async def test_connection_timeout_on_unreachable_host(self):
        """Test that connection times out on unreachable host."""
        import redis.asyncio as aioredis

        # Use a non-routable IP to test timeout
        client = aioredis.from_url(
            "redis://10.255.255.1:6379",
            decode_responses=True,
            socket_timeout=1,
            socket_connect_timeout=1,
        )

        start = time.time()
        with pytest.raises((TimeoutError, OSError, ConnectionError)):
            await client.ping()
        elapsed = time.time() - start

        # Should timeout within ~2 seconds (1s timeout + overhead)
        assert elapsed < 5

        await client.close()


# =============================================================================
# CIRCUIT BREAKER INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
class TestRedisCircuitBreakerIntegration:
    """Tests for Redis health check circuit breaker integration."""

    async def test_circuit_breaker_opens_on_failures(self, clean_redis_client):
        """Test that circuit breaker opens after repeated failures."""
        import src.api.dependencies.redis_client as redis_module
        from src.api.dependencies.redis_client import redis_health_check

        # Initialize connection first
        await redis_module.init_redis()

        # Reset circuit breaker
        redis_module._health_circuit_breaker._state = (
            redis_module._health_circuit_breaker.CircuitState.CLOSED
        )
        redis_module._health_circuit_breaker._failure_count = 0

        # Simulate failures by patching get_redis to fail
        async def failing_get_redis():
            raise ConnectionError("Simulated failure")

        # Record enough failures to trip the circuit breaker
        original_get_redis = redis_module.get_redis
        redis_module.get_redis = failing_get_redis

        # Make requests until circuit opens (threshold is 3)
        for _ in range(4):
            result = await redis_health_check()

        # Restore
        redis_module.get_redis = original_get_redis

        # Circuit should now be open
        result = await redis_health_check()
        assert result["status"] == "circuit_open"

        # Reset for other tests
        redis_module._health_circuit_breaker._state = (
            redis_module._health_circuit_breaker.CircuitState.CLOSED
        )
        redis_module._health_circuit_breaker._failure_count = 0

    async def test_circuit_breaker_resets_on_success(self, clean_redis_client):
        """Test that circuit breaker resets after successful health check."""
        import src.api.dependencies.redis_client as redis_module
        from src.api.dependencies.redis_client import init_redis, redis_health_check

        await init_redis()

        # Reset circuit breaker state
        redis_module._health_circuit_breaker._state = (
            redis_module._health_circuit_breaker.CircuitState.CLOSED
        )
        redis_module._health_circuit_breaker._failure_count = 2  # Just under threshold

        # Successful health check should reset failure count
        result = await redis_health_check()
        assert result["status"] == "healthy"

        # Failure count should be reset
        assert redis_module._health_circuit_breaker._failure_count == 0


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================


@pytest.mark.asyncio
class TestRedisDataIntegrityWithAuth:
    """Tests for data integrity with authenticated Redis."""

    async def test_set_get_various_data_types(self, clean_redis_client):
        """Test storing and retrieving various data types."""
        from src.api.dependencies.redis_client import get_redis

        client = await get_redis()
        prefix = f"test:integrity:{time.time()}"

        try:
            # String
            await client.set(f"{prefix}:string", "hello world", ex=60)
            assert await client.get(f"{prefix}:string") == "hello world"

            # Integer (stored as string)
            await client.set(f"{prefix}:int", "12345", ex=60)
            assert await client.get(f"{prefix}:int") == "12345"

            # JSON-like string
            json_data = '{"key": "value", "nested": {"a": 1}}'
            await client.set(f"{prefix}:json", json_data, ex=60)
            assert await client.get(f"{prefix}:json") == json_data

            # Unicode
            await client.set(f"{prefix}:unicode", "日本語テスト", ex=60)
            assert await client.get(f"{prefix}:unicode") == "日本語テスト"

            # Empty string
            await client.set(f"{prefix}:empty", "", ex=60)
            assert await client.get(f"{prefix}:empty") == ""

        finally:
            # Cleanup
            keys = await client.keys(f"{prefix}:*")
            if keys:
                await client.delete(*keys)

    async def test_expiration_works_with_auth(self, clean_redis_client):
        """Test that key expiration works correctly."""
        from src.api.dependencies.redis_client import get_redis

        client = await get_redis()
        test_key = f"test:expiry:{time.time()}"

        # Set with 2-second expiration
        await client.set(test_key, "temporary", ex=2)

        # Should exist immediately
        assert await client.get(test_key) == "temporary"

        # Wait for expiration
        await asyncio.sleep(2.5)

        # Should be gone
        assert await client.get(test_key) is None

    async def test_atomic_increment_with_auth(self, clean_redis_client):
        """Test atomic increment operations."""
        from src.api.dependencies.redis_client import get_redis

        client = await get_redis()
        test_key = f"test:incr:{time.time()}"

        try:
            # Initialize
            await client.set(test_key, "0", ex=60)

            # Concurrent increments
            async def increment():
                return await client.incr(test_key)

            tasks = [increment() for _ in range(100)]
            await asyncio.gather(*tasks)

            # All increments should succeed, final value should be 100
            final_value = await client.get(test_key)
            assert int(final_value) == 100

        finally:
            await client.delete(test_key)
