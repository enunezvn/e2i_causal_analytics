"""Unit tests for Redis client dependency.

Tests cover:
- Client initialization and connection
- Health check functionality
- Connection pooling configuration
- Error handling for connection failures
- Singleton pattern behavior
- Cleanup and close operations
- Tenacity retry decorator behavior
- Circuit breaker on health checks

Author: E2I Causal Analytics Team
Version: 2.0.0
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from src.utils.circuit_breaker import CircuitState


@pytest.mark.unit
class TestRedisClient:
    """Test suite for Redis client dependency."""

    @pytest.fixture(autouse=True)
    def reset_client(self):
        """Reset global client and circuit breaker before each test."""
        import src.api.dependencies.redis_client as redis_module
        from src.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        redis_module._redis_client = None
        redis_module._health_circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=3, reset_timeout_seconds=30.0)
        )
        yield
        redis_module._redis_client = None

    @pytest.mark.asyncio
    async def test_init_redis_success(self):
        """Test successful Redis initialization."""
        from src.api.dependencies.redis_client import init_redis

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            client = await init_redis()

            assert client is not None
            mock_from_url.assert_called_once()
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_redis_uses_environment_config(self):
        """Test Redis initialization uses environment variables."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with patch.dict(
            "os.environ",
            {
                "REDIS_URL": "redis://custom-host:6379",
                "REDIS_SOCKET_TIMEOUT": "5.0",
                "REDIS_MAX_CONNECTIONS": "20",
            },
        ):
            # Re-import module to pick up environment changes
            import importlib

            import src.api.dependencies.redis_client

            importlib.reload(src.api.dependencies.redis_client)
            from src.api.dependencies.redis_client import init_redis

            with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
                mock_from_url.return_value = mock_redis

                await init_redis()

                call_args = mock_from_url.call_args
                assert "redis://custom-host:6379" in str(call_args)
                assert call_args.kwargs["socket_timeout"] == 5.0
                assert call_args.kwargs["max_connections"] == 20
                assert call_args.kwargs["decode_responses"] is True

    @pytest.mark.asyncio
    async def test_init_redis_connection_failure(self):
        """Test Redis initialization handles connection failures."""
        from src.api.dependencies.redis_client import init_redis

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection refused")

            with pytest.raises(ConnectionError, match="Redis connection failed"):
                await init_redis()

    @pytest.mark.asyncio
    async def test_init_redis_ping_failure(self):
        """Test Redis initialization handles ping failures."""
        from src.api.dependencies.redis_client import init_redis

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("PING failed"))

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            with pytest.raises(ConnectionError, match="Redis connection failed"):
                await init_redis()

    @pytest.mark.asyncio
    async def test_init_redis_singleton_pattern(self):
        """Test Redis client uses singleton pattern."""
        from src.api.dependencies.redis_client import init_redis

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            client1 = await init_redis()
            client2 = await init_redis()

            assert client1 is client2
            # Should only call from_url once due to singleton
            assert mock_from_url.call_count == 1

    @pytest.mark.asyncio
    async def test_get_redis_initializes_if_needed(self):
        """Test get_redis initializes client if not already initialized."""
        from src.api.dependencies.redis_client import get_redis

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            client = await get_redis()

            assert client is not None
            mock_from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_redis_returns_existing_client(self):
        """Test get_redis returns existing client without reinitializing."""
        from src.api.dependencies.redis_client import get_redis, init_redis

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            await init_redis()
            client = await get_redis()

            assert client is mock_redis
            assert mock_from_url.call_count == 1

    @pytest.mark.asyncio
    async def test_close_redis(self):
        """Test Redis client cleanup."""
        from src.api.dependencies.redis_client import close_redis, init_redis

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.close = AsyncMock()

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis

            await init_redis()
            await close_redis()

            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_redis_when_not_initialized(self):
        """Test close_redis handles uninitialized client gracefully."""
        from src.api.dependencies.redis_client import close_redis

        # Should not raise any errors
        await close_redis()

    @pytest.mark.asyncio
    async def test_redis_health_check_healthy(self):
        """Test Redis health check returns healthy status."""
        from src.api.dependencies.redis_client import redis_health_check

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.info = AsyncMock(return_value={"connected_clients": 5})

        with patch("src.api.dependencies.redis_client.get_redis") as mock_get:
            mock_get.return_value = mock_redis

            result = await redis_health_check()

            assert result["status"] == "healthy"
            assert "latency_ms" in result
            assert result["connected_clients"] == 5
            assert result["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_redis_health_check_unhealthy(self):
        """Test Redis health check handles unhealthy state."""
        from src.api.dependencies.redis_client import redis_health_check

        with patch("src.api.dependencies.redis_client.get_redis") as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            result = await redis_health_check()

            assert result["status"] == "unhealthy"
            assert "error" in result
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_redis_health_check_ping_failure(self):
        """Test Redis health check handles ping failures."""
        from src.api.dependencies.redis_client import redis_health_check

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("PING failed"))

        with patch("src.api.dependencies.redis_client.get_redis") as mock_get:
            mock_get.return_value = mock_redis

            result = await redis_health_check()

            assert result["status"] == "unhealthy"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_redis_health_check_info_failure(self):
        """Test Redis health check handles info command failures gracefully."""
        from src.api.dependencies.redis_client import redis_health_check

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.info = AsyncMock(side_effect=Exception("INFO failed"))

        with patch("src.api.dependencies.redis_client.get_redis") as mock_get:
            mock_get.return_value = mock_redis

            result = await redis_health_check()

            # Should still report unhealthy but not crash
            assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_redis_logging(self, caplog):
        """Test Redis client logs appropriate messages."""
        from src.api.dependencies.redis_client import init_redis

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with caplog.at_level(logging.INFO):
            with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
                mock_from_url.return_value = mock_redis

                await init_redis()

                assert any("Initializing Redis connection" in msg for msg in caplog.messages)
                assert any(
                    "Redis connection established successfully" in msg for msg in caplog.messages
                )

    @pytest.mark.asyncio
    async def test_redis_error_logging(self, caplog):
        """Test Redis client logs errors appropriately."""
        from src.api.dependencies.redis_client import init_redis

        with caplog.at_level(logging.ERROR):
            with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
                mock_from_url.side_effect = Exception("Connection failed")

                with pytest.raises(ConnectionError):
                    await init_redis()

                assert any("Failed to connect to Redis" in msg for msg in caplog.messages)

    # =========================================================================
    # Stale reference handling tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_init_redis_stale_reference_ping_fails(self):
        """Test init_redis reconnects when existing client ping fails."""
        import src.api.dependencies.redis_client as redis_module
        from src.api.dependencies.redis_client import init_redis

        # Set a stale client that fails ping
        stale_client = AsyncMock()
        stale_client.ping = AsyncMock(side_effect=Exception("Connection lost"))
        redis_module._redis_client = stale_client

        # New client that works
        new_client = AsyncMock()
        new_client.ping = AsyncMock()

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.return_value = new_client

            client = await init_redis()

            assert client is new_client
            mock_from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_redis_healthy_reference_reuses_client(self):
        """Test init_redis reuses existing client when ping succeeds."""
        import src.api.dependencies.redis_client as redis_module
        from src.api.dependencies.redis_client import init_redis

        # Set a healthy existing client
        existing_client = AsyncMock()
        existing_client.ping = AsyncMock()
        redis_module._redis_client = existing_client

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            client = await init_redis()

            assert client is existing_client
            mock_from_url.assert_not_called()

    # =========================================================================
    # Retry decorator tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_init_redis_retries_on_connection_error(self):
        """Test init_redis retries on ConnectionError via tenacity."""
        from src.api.dependencies.redis_client import init_redis

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            return mock_redis

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.side_effect = side_effect

            client = await init_redis()

            assert client is mock_redis
            assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_init_redis_retries_on_timeout_error(self):
        """Test init_redis retries on TimeoutError via tenacity."""
        from src.api.dependencies.redis_client import init_redis

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Connection timed out")
            return mock_redis

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.side_effect = side_effect

            client = await init_redis()

            assert client is mock_redis
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_init_redis_wraps_all_errors_in_connection_error(self):
        """Test init_redis wraps non-connection exceptions in ConnectionError."""
        from src.api.dependencies.redis_client import init_redis

        with patch("src.api.dependencies.redis_client.aioredis.from_url") as mock_from_url:
            mock_from_url.side_effect = ValueError("Invalid config")

            # ValueError gets wrapped in ConnectionError, which tenacity retries.
            # After 5 attempts it reraises the final ConnectionError.
            with pytest.raises(ConnectionError, match="Redis connection failed"):
                await init_redis()

            assert mock_from_url.call_count == 5  # Retried max attempts

    # =========================================================================
    # Circuit breaker health check tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_health_check_circuit_open_returns_circuit_status(self):
        """Test health check returns circuit_open when breaker is tripped."""
        import src.api.dependencies.redis_client as redis_module
        from src.api.dependencies.redis_client import redis_health_check

        # Trip the circuit breaker
        redis_module._health_circuit_breaker.force_open()

        result = await redis_health_check()

        assert result["status"] == "circuit_open"

    @pytest.mark.asyncio
    async def test_health_check_records_success_on_breaker(self):
        """Test health check records success on circuit breaker."""
        import src.api.dependencies.redis_client as redis_module
        from src.api.dependencies.redis_client import redis_health_check

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.info = AsyncMock(return_value={"connected_clients": 3})

        with patch("src.api.dependencies.redis_client.get_redis") as mock_get:
            mock_get.return_value = mock_redis

            await redis_health_check()

            assert redis_module._health_circuit_breaker.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_health_check_records_failure_on_breaker(self):
        """Test health check records failure on circuit breaker."""
        import src.api.dependencies.redis_client as redis_module
        from src.api.dependencies.redis_client import redis_health_check

        with patch("src.api.dependencies.redis_client.get_redis") as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            await redis_health_check()

            assert redis_module._health_circuit_breaker.metrics.failed_calls == 1

    @pytest.mark.asyncio
    async def test_health_check_circuit_opens_after_repeated_failures(self):
        """Test circuit breaker opens after repeated health check failures."""
        import src.api.dependencies.redis_client as redis_module
        from src.api.dependencies.redis_client import redis_health_check

        with patch("src.api.dependencies.redis_client.get_redis") as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            # Trip the breaker (threshold=3)
            for _ in range(3):
                await redis_health_check()

            assert redis_module._health_circuit_breaker.state == CircuitState.OPEN

            # Next call should return circuit_open without hitting Redis
            result = await redis_health_check()
            assert result["status"] == "circuit_open"
