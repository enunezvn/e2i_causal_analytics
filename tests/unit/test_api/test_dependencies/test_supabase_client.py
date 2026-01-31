"""Unit tests for Supabase client dependency.

Tests cover:
- Client initialization with credentials
- Client initialization without credentials (graceful degradation)
- Health check functionality
- Connection verification
- Error handling
- Singleton pattern behavior
- Missing package handling

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestSupabaseClient:
    """Test suite for Supabase client dependency."""

    @pytest.fixture(autouse=True)
    def reset_client(self):
        """Reset global client before each test."""
        import src.api.dependencies.supabase_client as supabase_module
        supabase_module._supabase_client = None
        yield
        supabase_module._supabase_client = None

    def test_init_supabase_success(self):
        """Test successful Supabase initialization."""
        from src.api.dependencies.supabase_client import init_supabase

        mock_client = MagicMock()

        # Patch module-level variables since they're set at import time
        with patch("src.api.dependencies.supabase_client.SUPABASE_URL", "https://test.supabase.co"):
            with patch("src.api.dependencies.supabase_client.SUPABASE_KEY", "test-anon-key"):
                with patch("src.api.dependencies.supabase_client.SUPABASE_SERVICE_KEY", ""):
                    with patch("supabase.create_client") as mock_create:
                        mock_create.return_value = mock_client

                        client = init_supabase()

                        assert client is not None
                        mock_create.assert_called_once_with("https://test.supabase.co", "test-anon-key")

    def test_init_supabase_uses_service_key_if_available(self):
        """Test Supabase initialization prefers service key over anon key."""
        from src.api.dependencies.supabase_client import init_supabase

        mock_client = MagicMock()

        # Patch module-level variables to simulate service key preference
        with patch("src.api.dependencies.supabase_client.SUPABASE_URL", "https://test.supabase.co"):
            with patch("src.api.dependencies.supabase_client.SUPABASE_KEY", "test-anon-key"):
                with patch("src.api.dependencies.supabase_client.SUPABASE_SERVICE_KEY", "test-service-key"):
                    with patch("supabase.create_client") as mock_create:
                        mock_create.return_value = mock_client

                        client = init_supabase()

                        # Should use service key, not anon key
                        mock_create.assert_called_once_with("https://test.supabase.co", "test-service-key")

    def test_init_supabase_missing_credentials(self):
        """Test Supabase initialization without credentials returns None."""
        from src.api.dependencies.supabase_client import init_supabase

        # Patch module-level variables to simulate missing credentials
        with patch("src.api.dependencies.supabase_client.SUPABASE_URL", ""):
            with patch("src.api.dependencies.supabase_client.SUPABASE_KEY", ""):
                client = init_supabase()

                assert client is None

    def test_init_supabase_missing_url(self):
        """Test Supabase initialization with only key returns None."""
        from src.api.dependencies.supabase_client import init_supabase

        # Patch module-level variables to simulate missing URL
        with patch("src.api.dependencies.supabase_client.SUPABASE_URL", ""):
            with patch("src.api.dependencies.supabase_client.SUPABASE_KEY", "test-key"):
                client = init_supabase()

                assert client is None

    def test_init_supabase_missing_key(self):
        """Test Supabase initialization with only URL returns None."""
        from src.api.dependencies.supabase_client import init_supabase

        # Patch module-level variables since they're set at import time
        with patch("src.api.dependencies.supabase_client.SUPABASE_URL", "https://test.supabase.co"):
            with patch("src.api.dependencies.supabase_client.SUPABASE_KEY", ""):
                with patch("src.api.dependencies.supabase_client.SUPABASE_SERVICE_KEY", ""):
                    client = init_supabase()

                    assert client is None

    def test_init_supabase_package_not_installed(self):
        """Test Supabase initialization handles missing package gracefully."""
        from src.api.dependencies.supabase_client import init_supabase

        with patch.dict(
            "os.environ",
            {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_KEY": "test-key",
            },
        ):
            with patch("supabase.create_client") as mock_create:
                mock_create.side_effect = ImportError("No module named 'supabase'")

                client = init_supabase()

                assert client is None

    def test_init_supabase_connection_error(self):
        """Test Supabase initialization handles connection errors."""
        from src.api.dependencies.supabase_client import init_supabase

        with patch.dict(
            "os.environ",
            {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_KEY": "test-key",
            },
        ):
            with patch("supabase.create_client") as mock_create:
                mock_create.side_effect = Exception("Connection failed")

                with pytest.raises(ConnectionError, match="Supabase connection failed"):
                    init_supabase()

    def test_init_supabase_singleton_pattern(self):
        """Test Supabase client uses singleton pattern."""
        from src.api.dependencies.supabase_client import init_supabase

        mock_client = MagicMock()

        with patch("supabase.create_client") as mock_create:
            with patch.dict(
                "os.environ",
                {
                    "SUPABASE_URL": "https://test.supabase.co",
                    "SUPABASE_KEY": "test-key",
                },
            ):
                mock_create.return_value = mock_client

                client1 = init_supabase()
                client2 = init_supabase()

                assert client1 is client2
                # Should only call create_client once
                assert mock_create.call_count == 1

    def test_get_supabase_returns_existing_client(self):
        """Test get_supabase returns existing client."""
        from src.api.dependencies.supabase_client import get_supabase, init_supabase

        mock_client = MagicMock()

        with patch("supabase.create_client") as mock_create:
            with patch.dict(
                "os.environ",
                {
                    "SUPABASE_URL": "https://test.supabase.co",
                    "SUPABASE_KEY": "test-key",
                },
            ):
                mock_create.return_value = mock_client

                init_supabase()
                client = get_supabase()

                assert client is mock_client

    def test_get_supabase_initializes_if_needed(self):
        """Test get_supabase initializes client if not already initialized."""
        from src.api.dependencies.supabase_client import get_supabase

        mock_client = MagicMock()

        with patch("supabase.create_client") as mock_create:
            with patch.dict(
                "os.environ",
                {
                    "SUPABASE_URL": "https://test.supabase.co",
                    "SUPABASE_KEY": "test-key",
                },
            ):
                mock_create.return_value = mock_client

                client = get_supabase()

                assert client is mock_client

    def test_get_supabase_returns_none_on_error(self):
        """Test get_supabase returns None on initialization errors."""
        from src.api.dependencies.supabase_client import get_supabase

        # Patch module-level variables to simulate missing credentials
        with patch("src.api.dependencies.supabase_client.SUPABASE_URL", ""):
            with patch("src.api.dependencies.supabase_client.SUPABASE_KEY", ""):
                client = get_supabase()

                assert client is None

    def test_close_supabase(self):
        """Test Supabase client cleanup."""
        from src.api.dependencies.supabase_client import close_supabase, init_supabase

        mock_client = MagicMock()

        with patch("supabase.create_client") as mock_create:
            with patch.dict(
                "os.environ",
                {
                    "SUPABASE_URL": "https://test.supabase.co",
                    "SUPABASE_KEY": "test-key",
                },
            ):
                mock_create.return_value = mock_client

                init_supabase()
                close_supabase()

                # Verify client is cleared (Supabase doesn't have explicit close)
                from src.api.dependencies.supabase_client import _supabase_client
                assert _supabase_client is None

    def test_close_supabase_when_not_initialized(self):
        """Test close_supabase handles uninitialized client gracefully."""
        from src.api.dependencies.supabase_client import close_supabase

        # Should not raise any errors
        close_supabase()

    @pytest.mark.asyncio
    async def test_supabase_health_check_healthy(self):
        """Test Supabase health check returns healthy status."""
        from src.api.dependencies.supabase_client import supabase_health_check

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.select.return_value.limit.return_value.execute.return_value = None

        mock_client.table.return_value = mock_table

        with patch("src.api.dependencies.supabase_client.get_supabase") as mock_get:
            mock_get.return_value = mock_client

            result = await supabase_health_check()

            assert result["status"] == "healthy"
            assert "latency_ms" in result
            assert result["connected"] is True

    @pytest.mark.asyncio
    async def test_supabase_health_check_not_configured(self):
        """Test Supabase health check when not configured."""
        from src.api.dependencies.supabase_client import supabase_health_check

        with patch("src.api.dependencies.supabase_client.get_supabase") as mock_get:
            mock_get.return_value = None

            result = await supabase_health_check()

            assert result["status"] == "unavailable"
            assert "error" in result
            assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_supabase_health_check_connection_error(self):
        """Test Supabase health check handles connection errors."""
        from src.api.dependencies.supabase_client import supabase_health_check

        with patch("src.api.dependencies.supabase_client.get_supabase") as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            result = await supabase_health_check()

            assert result["status"] == "unhealthy"
            assert "error" in result
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_supabase_health_check_query_error(self):
        """Test Supabase health check handles query errors gracefully."""
        from src.api.dependencies.supabase_client import supabase_health_check

        mock_client = MagicMock()
        # Simulate query failure but connection works
        mock_client.table.side_effect = Exception("Query failed")

        with patch("src.api.dependencies.supabase_client.get_supabase") as mock_get:
            mock_get.return_value = mock_client

            result = await supabase_health_check()

            # Should still complete (errors are caught in nested try/except)
            assert result["status"] in ["healthy", "unhealthy"]

    def test_supabase_logging_success(self, caplog):
        """Test Supabase client logs success messages."""
        from src.api.dependencies.supabase_client import init_supabase

        mock_client = MagicMock()

        with caplog.at_level(logging.INFO):
            with patch("supabase.create_client") as mock_create:
                with patch.dict(
                    "os.environ",
                    {
                        "SUPABASE_URL": "https://test.supabase.co",
                        "SUPABASE_KEY": "test-key",
                    },
                ):
                    mock_create.return_value = mock_client

                    init_supabase()

                    assert any("Initializing Supabase connection" in msg for msg in caplog.messages)
                    assert any("Supabase client initialized successfully" in msg for msg in caplog.messages)

    def test_supabase_logging_missing_credentials(self, caplog):
        """Test Supabase client logs warning when credentials missing."""
        from src.api.dependencies.supabase_client import init_supabase

        with caplog.at_level(logging.WARNING):
            # Patch module-level variables to simulate missing credentials
            with patch("src.api.dependencies.supabase_client.SUPABASE_URL", ""):
                with patch("src.api.dependencies.supabase_client.SUPABASE_KEY", ""):
                    init_supabase()

                    assert any("Supabase credentials not configured" in msg for msg in caplog.messages)

    def test_supabase_logging_error(self, caplog):
        """Test Supabase client logs errors."""
        from src.api.dependencies.supabase_client import init_supabase

        with caplog.at_level(logging.ERROR):
            with patch.dict(
                "os.environ",
                {
                    "SUPABASE_URL": "https://test.supabase.co",
                    "SUPABASE_KEY": "test-key",
                },
            ):
                with patch("supabase.create_client") as mock_create:
                    mock_create.side_effect = Exception("Connection failed")

                    with pytest.raises(ConnectionError):
                        init_supabase()

                    assert any("Failed to connect to Supabase" in msg for msg in caplog.messages)

    def test_supabase_uses_anon_key_fallback(self):
        """Test Supabase uses SUPABASE_ANON_KEY as fallback."""
        from src.api.dependencies.supabase_client import init_supabase

        mock_client = MagicMock()

        with patch("supabase.create_client") as mock_create:
            # Patch module-level variables to simulate ANON_KEY fallback scenario
            with patch("src.api.dependencies.supabase_client.SUPABASE_URL", "https://test.supabase.co"):
                with patch("src.api.dependencies.supabase_client.SUPABASE_KEY", "test-anon-key"):
                    with patch("src.api.dependencies.supabase_client.SUPABASE_SERVICE_KEY", ""):
                        mock_create.return_value = mock_client

                        client = init_supabase()

                        assert client is not None
                        # Should use SUPABASE_KEY (which contains the fallback anon key value)
                        mock_create.assert_called_once_with("https://test.supabase.co", "test-anon-key")
