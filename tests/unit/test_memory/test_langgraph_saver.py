"""
Unit tests for E2I LangGraph Checkpointer Factory.

Tests focus on:
- create_checkpointer() function (sync Redis checkpointer)
- create_async_checkpointer() function (async Redis checkpointer)
- CheckpointerConfig class
- Error handling and fallback behavior
- Environment variable handling

All tests use mocked dependencies to avoid external services.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.memory.langgraph_saver import (
    CheckpointerConfig,
    create_async_checkpointer,
    create_checkpointer,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_redis_saver():
    """Create a mock RedisSaver."""
    saver = MagicMock()
    return saver


@pytest.fixture
def mock_memory_saver():
    """Create a mock MemorySaver."""
    return MagicMock()


@pytest.fixture
def clean_env():
    """Fixture to ensure clean environment for REDIS_URL tests."""
    original = os.environ.get("REDIS_URL")
    yield
    if original is not None:
        os.environ["REDIS_URL"] = original
    else:
        os.environ.pop("REDIS_URL", None)


# ============================================================================
# CREATE_CHECKPOINTER TESTS
# ============================================================================


class TestCreateCheckpointer:
    """Tests for create_checkpointer function."""

    def test_creates_redis_saver_with_url(self, mock_redis_saver):
        """Should create RedisSaver when Redis is available."""
        import sys

        mock_cls = MagicMock()
        mock_cls.from_conn_string.return_value = mock_redis_saver
        mock_module = MagicMock()
        mock_module.RedisSaver = mock_cls

        with patch.dict(
            sys.modules,
            {"langgraph.checkpoint.redis": mock_module},
        ):
            result = create_checkpointer(redis_url="redis://custom:6380")
            mock_cls.from_conn_string.assert_called_once_with("redis://custom:6380")
            assert result == mock_redis_saver

    def test_uses_env_var_when_no_url_provided(self, mock_redis_saver):
        """Should use REDIS_URL environment variable when no URL provided."""
        import sys

        mock_cls = MagicMock()
        mock_cls.from_conn_string.return_value = mock_redis_saver
        mock_module = MagicMock()
        mock_module.RedisSaver = mock_cls

        with patch.dict(os.environ, {"REDIS_URL": "redis://env-host:6379"}):
            with patch.dict(
                sys.modules,
                {"langgraph.checkpoint.redis": mock_module},
            ):
                create_checkpointer()
                mock_cls.from_conn_string.assert_called_once_with("redis://env-host:6379")

    def test_falls_back_to_memory_on_import_error(self, mock_memory_saver):
        """Should fall back to MemorySaver when import fails."""
        import sys

        # Create a mock module that raises ImportError when RedisSaver is accessed
        class MockModule:
            @property
            def RedisSaver(self):
                raise ImportError("Not installed")

        # First, force the import to work but have from_conn_string fail
        with patch.dict(sys.modules, {"langgraph.checkpoint.redis": None}):
            with patch(
                "langgraph.checkpoint.memory.MemorySaver",
                return_value=mock_memory_saver,
            ):
                result = create_checkpointer(fallback_to_memory=True)
                # When module is None in sys.modules, import raises ImportError
                assert result == mock_memory_saver

    def test_raises_import_error_when_fallback_disabled(self):
        """Should raise ImportError when fallback_to_memory is False."""
        import sys

        # Force ImportError by setting module to None
        with patch.dict(sys.modules, {"langgraph.checkpoint.redis": None}):
            with pytest.raises(ImportError) as exc_info:
                create_checkpointer(fallback_to_memory=False)
            assert "langgraph-checkpoint-redis" in str(exc_info.value)

    def test_falls_back_to_memory_on_connection_error(self, mock_memory_saver):
        """Should fall back to MemorySaver when Redis connection fails."""
        import sys

        mock_cls = MagicMock()
        mock_cls.from_conn_string.side_effect = ConnectionError("Connection refused")
        mock_module = MagicMock()
        mock_module.RedisSaver = mock_cls

        with patch.dict(
            sys.modules,
            {"langgraph.checkpoint.redis": mock_module},
        ):
            with patch(
                "langgraph.checkpoint.memory.MemorySaver",
                return_value=mock_memory_saver,
            ):
                result = create_checkpointer(fallback_to_memory=True)
                assert result == mock_memory_saver

    def test_raises_connection_error_when_fallback_disabled(self):
        """Should raise ConnectionError when fallback is disabled and connection fails."""
        import sys

        mock_cls = MagicMock()
        mock_cls.from_conn_string.side_effect = Exception("Connection refused")
        mock_module = MagicMock()
        mock_module.RedisSaver = mock_cls

        with patch.dict(
            sys.modules,
            {"langgraph.checkpoint.redis": mock_module},
        ):
            with pytest.raises(ConnectionError) as exc_info:
                create_checkpointer(fallback_to_memory=False)
            assert "Failed to connect to Redis" in str(exc_info.value)

    def test_default_redis_url(self, mock_redis_saver, clean_env):
        """Should use default localhost:6382 when no URL or env var."""
        import sys

        # Clear REDIS_URL
        os.environ.pop("REDIS_URL", None)

        mock_cls = MagicMock()
        mock_cls.from_conn_string.return_value = mock_redis_saver
        mock_module = MagicMock()
        mock_module.RedisSaver = mock_cls

        with patch.dict(
            sys.modules,
            {"langgraph.checkpoint.redis": mock_module},
        ):
            create_checkpointer()
            mock_cls.from_conn_string.assert_called_once_with("redis://localhost:6382")


# ============================================================================
# CREATE_ASYNC_CHECKPOINTER TESTS
# ============================================================================


class TestCreateAsyncCheckpointer:
    """Tests for create_async_checkpointer function."""

    def test_creates_async_redis_saver_with_url(self):
        """Should create AsyncRedisSaver when Redis is available."""
        import sys

        mock_saver = MagicMock()
        mock_cls = MagicMock()
        mock_cls.from_conn_string.return_value = mock_saver
        mock_module = MagicMock()
        mock_module.AsyncRedisSaver = mock_cls

        with patch.dict(
            sys.modules,
            {"langgraph.checkpoint.redis.aio": mock_module},
        ):
            result = create_async_checkpointer(redis_url="redis://custom:6380")
            mock_cls.from_conn_string.assert_called_once_with("redis://custom:6380")
            assert result == mock_saver

    def test_falls_back_to_memory_on_import_error(self, mock_memory_saver):
        """Should fall back to MemorySaver when async import fails."""
        import sys

        with patch.dict(sys.modules, {"langgraph.checkpoint.redis.aio": None}):
            with patch(
                "langgraph.checkpoint.memory.MemorySaver",
                return_value=mock_memory_saver,
            ):
                result = create_async_checkpointer(fallback_to_memory=True)
                assert result == mock_memory_saver

    def test_raises_import_error_when_fallback_disabled(self):
        """Should raise ImportError when fallback_to_memory is False."""
        import sys

        with patch.dict(sys.modules, {"langgraph.checkpoint.redis.aio": None}):
            with pytest.raises(ImportError) as exc_info:
                create_async_checkpointer(fallback_to_memory=False)
            assert "langgraph-checkpoint-redis" in str(exc_info.value)

    def test_falls_back_on_connection_error(self, mock_memory_saver):
        """Should fall back to MemorySaver when async connection fails."""
        import sys

        mock_cls = MagicMock()
        mock_cls.from_conn_string.side_effect = Exception("Connection refused")
        mock_module = MagicMock()
        mock_module.AsyncRedisSaver = mock_cls

        with patch.dict(
            sys.modules,
            {"langgraph.checkpoint.redis.aio": mock_module},
        ):
            with patch(
                "langgraph.checkpoint.memory.MemorySaver",
                return_value=mock_memory_saver,
            ):
                result = create_async_checkpointer(fallback_to_memory=True)
                assert result == mock_memory_saver

    def test_raises_connection_error_when_fallback_disabled(self):
        """Should raise ConnectionError when fallback is disabled."""
        import sys

        mock_cls = MagicMock()
        mock_cls.from_conn_string.side_effect = Exception("Connection refused")
        mock_module = MagicMock()
        mock_module.AsyncRedisSaver = mock_cls

        with patch.dict(
            sys.modules,
            {"langgraph.checkpoint.redis.aio": mock_module},
        ):
            with pytest.raises(ConnectionError) as exc_info:
                create_async_checkpointer(fallback_to_memory=False)
            assert "Failed to connect to Redis" in str(exc_info.value)


# ============================================================================
# CHECKPOINTER CONFIG TESTS
# ============================================================================


class TestCheckpointerConfig:
    """Tests for CheckpointerConfig class."""

    def test_default_values(self, clean_env):
        """Should use default values when not provided."""
        os.environ.pop("REDIS_URL", None)

        config = CheckpointerConfig()
        assert config.redis_url == "redis://localhost:6382"
        assert config.checkpoint_prefix == "e2i:checkpoint:"
        assert config.ttl_seconds == 86400
        assert config.fallback_to_memory is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = CheckpointerConfig(
            redis_url="redis://custom:9999",
            checkpoint_prefix="custom:prefix:",
            ttl_seconds=3600,
            fallback_to_memory=False,
        )
        assert config.redis_url == "redis://custom:9999"
        assert config.checkpoint_prefix == "custom:prefix:"
        assert config.ttl_seconds == 3600
        assert config.fallback_to_memory is False

    def test_uses_env_var_for_redis_url(self):
        """Should use REDIS_URL env var when redis_url not provided."""
        with patch.dict(os.environ, {"REDIS_URL": "redis://env-var:6379"}):
            config = CheckpointerConfig()
            assert config.redis_url == "redis://env-var:6379"

    def test_ttl_can_be_none(self):
        """Should allow None TTL for no expiry."""
        config = CheckpointerConfig(ttl_seconds=None)
        assert config.ttl_seconds is None

    def test_create_checkpointer_method(self):
        """Should create checkpointer using configuration."""
        import sys

        mock_saver = MagicMock()
        mock_cls = MagicMock()
        mock_cls.from_conn_string.return_value = mock_saver
        mock_module = MagicMock()
        mock_module.RedisSaver = mock_cls

        with patch.dict(
            sys.modules,
            {"langgraph.checkpoint.redis": mock_module},
        ):
            config = CheckpointerConfig(redis_url="redis://config-test:6379")
            config.create_checkpointer()
            mock_cls.from_conn_string.assert_called_once_with("redis://config-test:6379")

    def test_create_async_checkpointer_method(self):
        """Should create async checkpointer using configuration."""
        import sys

        mock_saver = MagicMock()
        mock_cls = MagicMock()
        mock_cls.from_conn_string.return_value = mock_saver
        mock_module = MagicMock()
        mock_module.AsyncRedisSaver = mock_cls

        with patch.dict(
            sys.modules,
            {"langgraph.checkpoint.redis.aio": mock_module},
        ):
            config = CheckpointerConfig(redis_url="redis://config-async:6379")
            config.create_async_checkpointer()
            mock_cls.from_conn_string.assert_called_once_with("redis://config-async:6379")


# ============================================================================
# INTEGRATION-STYLE TESTS (with real module if available)
# ============================================================================


class TestCheckpointerIntegration:
    """Integration-style tests that work with the real langgraph modules."""

    def test_real_checkpointer_creation_with_fallback(self):
        """Test that create_checkpointer works end-to-end with fallback."""
        # This test always succeeds because fallback is enabled
        # It will use RedisSaver if available, MemorySaver otherwise
        result = create_checkpointer(fallback_to_memory=True)
        assert result is not None
        # Verify it has the expected checkpointer interface
        assert hasattr(result, "put") or hasattr(result, "aget") or hasattr(result, "get")

    def test_real_async_checkpointer_creation_with_fallback(self):
        """Test that create_async_checkpointer works end-to-end with fallback."""
        result = create_async_checkpointer(fallback_to_memory=True)
        assert result is not None
        assert hasattr(result, "put") or hasattr(result, "aget") or hasattr(result, "get")

    def test_config_creates_working_checkpointer(self):
        """Test that CheckpointerConfig creates a working checkpointer."""
        config = CheckpointerConfig(fallback_to_memory=True)
        result = config.create_checkpointer()
        assert result is not None
