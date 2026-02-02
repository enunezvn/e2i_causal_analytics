"""Unit tests for FalkorDB client dependency.

Tests cover:
- Client initialization and connection
- Graph selection and management
- Health check functionality
- Error handling for connection failures
- Missing package handling
- Singleton pattern behavior
- Node and edge counting

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestFalkorDBClient:
    """Test suite for FalkorDB client dependency."""

    @pytest.fixture(autouse=True)
    def reset_client(self):
        """Reset global client before each test."""
        import src.api.dependencies.falkordb_client as falkordb_module

        falkordb_module._falkordb_client = None
        falkordb_module._graph = None
        yield
        falkordb_module._falkordb_client = None
        falkordb_module._graph = None

    @pytest.mark.asyncio
    async def test_init_falkordb_success(self):
        """Test successful FalkorDB initialization."""
        from src.api.dependencies.falkordb_client import init_falkordb

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client.list_graphs.return_value = ["e2i_causal", "other_graph"]

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.return_value = mock_client

            client = await init_falkordb()

            assert client is not None
            mock_falkordb.assert_called_once_with(host="localhost", port=6381)
            mock_client.select_graph.assert_called_once_with("e2i_causal")
            mock_client.list_graphs.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_falkordb_uses_environment_config(self):
        """Test FalkorDB initialization uses environment variables."""
        from src.api.dependencies.falkordb_client import init_falkordb

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client.list_graphs.return_value = []

        with patch("falkordb.FalkorDB") as mock_falkordb:
            with patch.dict(
                "os.environ",
                {
                    "FALKORDB_HOST": "custom-host",
                    "FALKORDB_PORT": "7000",
                    "FALKORDB_GRAPH_NAME": "custom_graph",
                },
            ):
                mock_falkordb.return_value = mock_client

                await init_falkordb()

                mock_falkordb.assert_called_once_with(host="custom-host", port=7000)
                mock_client.select_graph.assert_called_once_with("custom_graph")

    @pytest.mark.asyncio
    async def test_init_falkordb_package_not_installed(self):
        """Test FalkorDB initialization handles missing package gracefully."""
        from src.api.dependencies.falkordb_client import init_falkordb

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.side_effect = ImportError("No module named 'falkordb'")

            result = await init_falkordb()

            assert result is None

    @pytest.mark.asyncio
    async def test_init_falkordb_connection_error(self):
        """Test FalkorDB initialization handles connection errors."""
        from src.api.dependencies.falkordb_client import init_falkordb

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.side_effect = Exception("Connection refused")

            with pytest.raises(ConnectionError, match="FalkorDB connection failed"):
                await init_falkordb()

    @pytest.mark.asyncio
    async def test_init_falkordb_list_graphs_error(self):
        """Test FalkorDB initialization handles list_graphs errors."""
        from src.api.dependencies.falkordb_client import init_falkordb

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client.list_graphs.side_effect = Exception("Failed to list graphs")

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.return_value = mock_client

            with pytest.raises(ConnectionError, match="FalkorDB connection failed"):
                await init_falkordb()

    @pytest.mark.asyncio
    async def test_init_falkordb_singleton_pattern(self):
        """Test FalkorDB client uses singleton pattern."""
        from src.api.dependencies.falkordb_client import init_falkordb

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client.list_graphs.return_value = []

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.return_value = mock_client

            client1 = await init_falkordb()
            client2 = await init_falkordb()

            assert client1 is client2
            # Should only call FalkorDB constructor once
            assert mock_falkordb.call_count == 1

    @pytest.mark.asyncio
    async def test_get_falkordb_returns_existing_client(self):
        """Test get_falkordb returns existing client."""
        from src.api.dependencies.falkordb_client import get_falkordb, init_falkordb

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client.list_graphs.return_value = []

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.return_value = mock_client

            await init_falkordb()
            client = await get_falkordb()

            assert client is mock_client

    @pytest.mark.asyncio
    async def test_get_falkordb_initializes_if_needed(self):
        """Test get_falkordb initializes client if not already initialized."""
        from src.api.dependencies.falkordb_client import get_falkordb

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client.list_graphs.return_value = []

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.return_value = mock_client

            client = await get_falkordb()

            assert client is mock_client

    @pytest.mark.asyncio
    async def test_get_falkordb_returns_none_on_error(self):
        """Test get_falkordb returns None on initialization errors."""
        from src.api.dependencies.falkordb_client import get_falkordb

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.side_effect = Exception("Connection failed")

            client = await get_falkordb()

            assert client is None

    @pytest.mark.asyncio
    async def test_get_graph_returns_existing_graph(self):
        """Test get_graph returns existing graph instance."""
        from src.api.dependencies.falkordb_client import get_graph, init_falkordb

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client.list_graphs.return_value = []

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.return_value = mock_client

            await init_falkordb()
            graph = await get_graph()

            assert graph is mock_graph

    @pytest.mark.asyncio
    async def test_get_graph_initializes_graph_if_needed(self):
        """Test get_graph creates graph if client exists but graph doesn't."""
        from src.api.dependencies.falkordb_client import get_graph

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client.list_graphs.return_value = []

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.return_value = mock_client

            # Get graph without initializing client first
            graph = await get_graph()

            assert graph is mock_graph
            mock_client.select_graph.assert_called()

    @pytest.mark.asyncio
    async def test_get_graph_returns_none_when_client_unavailable(self):
        """Test get_graph returns None when client is unavailable."""
        from src.api.dependencies.falkordb_client import get_graph

        with patch("src.api.dependencies.falkordb_client.get_falkordb") as mock_get:
            mock_get.return_value = None

            graph = await get_graph()

            assert graph is None

    @pytest.mark.asyncio
    async def test_close_falkordb(self):
        """Test FalkorDB client cleanup."""
        from src.api.dependencies.falkordb_client import close_falkordb, init_falkordb

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client.list_graphs.return_value = []

        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_falkordb.return_value = mock_client

            await init_falkordb()
            await close_falkordb()

            # Verify client and graph are cleared
            from src.api.dependencies.falkordb_client import _falkordb_client, _graph

            assert _falkordb_client is None
            assert _graph is None

    @pytest.mark.asyncio
    async def test_close_falkordb_when_not_initialized(self):
        """Test close_falkordb handles uninitialized client gracefully."""
        from src.api.dependencies.falkordb_client import close_falkordb

        # Should not raise any errors
        await close_falkordb()

    @pytest.mark.asyncio
    async def test_falkordb_health_check_healthy(self):
        """Test FalkorDB health check returns healthy status."""
        from src.api.dependencies.falkordb_client import falkordb_health_check

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.list_graphs.return_value = ["e2i_causal", "other_graph"]

        # Mock query results for node and edge counts
        mock_node_result = MagicMock()
        mock_node_result.result_set = [[42]]
        mock_edge_result = MagicMock()
        mock_edge_result.result_set = [[15]]

        mock_graph.query.side_effect = [mock_node_result, mock_edge_result]

        with patch("src.api.dependencies.falkordb_client.get_falkordb") as mock_get_client:
            with patch("src.api.dependencies.falkordb_client.get_graph") as mock_get_graph:
                mock_get_client.return_value = mock_client
                mock_get_graph.return_value = mock_graph

                result = await falkordb_health_check()

                assert result["status"] == "healthy"
                assert "latency_ms" in result
                assert result["graphs"] == ["e2i_causal", "other_graph"]
                assert result["current_graph"] == "e2i_causal"
                assert result["node_count"] == 42
                assert result["edge_count"] == 15

    @pytest.mark.asyncio
    async def test_falkordb_health_check_not_configured(self):
        """Test FalkorDB health check when not configured."""
        from src.api.dependencies.falkordb_client import falkordb_health_check

        with patch("src.api.dependencies.falkordb_client.get_falkordb") as mock_get:
            mock_get.return_value = None

            result = await falkordb_health_check()

            assert result["status"] == "unavailable"
            assert "error" in result
            assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_falkordb_health_check_connection_error(self):
        """Test FalkorDB health check handles connection errors."""
        from src.api.dependencies.falkordb_client import falkordb_health_check

        with patch("src.api.dependencies.falkordb_client.get_falkordb") as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            result = await falkordb_health_check()

            assert result["status"] == "unhealthy"
            assert "error" in result
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_falkordb_health_check_query_error_graceful(self):
        """Test FalkorDB health check handles query errors gracefully."""
        from src.api.dependencies.falkordb_client import falkordb_health_check

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.list_graphs.return_value = ["e2i_causal"]
        mock_graph.query.side_effect = Exception("Query failed")

        with patch("src.api.dependencies.falkordb_client.get_falkordb") as mock_get_client:
            with patch("src.api.dependencies.falkordb_client.get_graph") as mock_get_graph:
                mock_get_client.return_value = mock_client
                mock_get_graph.return_value = mock_graph

                result = await falkordb_health_check()

                # Should still report healthy but with 0 counts
                assert result["status"] == "healthy"
                assert result["node_count"] == 0
                assert result["edge_count"] == 0

    @pytest.mark.asyncio
    async def test_falkordb_health_check_empty_graph(self):
        """Test FalkorDB health check handles empty graph."""
        from src.api.dependencies.falkordb_client import falkordb_health_check

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.list_graphs.return_value = ["e2i_causal"]

        # Empty result sets
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph.query.return_value = mock_result

        with patch("src.api.dependencies.falkordb_client.get_falkordb") as mock_get_client:
            with patch("src.api.dependencies.falkordb_client.get_graph") as mock_get_graph:
                mock_get_client.return_value = mock_client
                mock_get_graph.return_value = mock_graph

                result = await falkordb_health_check()

                assert result["status"] == "healthy"
                assert result["node_count"] == 0
                assert result["edge_count"] == 0

    @pytest.mark.asyncio
    async def test_falkordb_logging_success(self, caplog):
        """Test FalkorDB client logs success messages."""
        from src.api.dependencies.falkordb_client import init_falkordb

        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client.list_graphs.return_value = ["e2i_causal"]

        with caplog.at_level(logging.INFO):
            with patch("falkordb.FalkorDB") as mock_falkordb:
                mock_falkordb.return_value = mock_client

                await init_falkordb()

                assert any("Initializing FalkorDB connection" in msg for msg in caplog.messages)
                assert any("FalkorDB connected" in msg for msg in caplog.messages)

    @pytest.mark.asyncio
    async def test_falkordb_logging_package_not_installed(self, caplog):
        """Test FalkorDB client logs warning when package not installed."""
        from src.api.dependencies.falkordb_client import init_falkordb

        with caplog.at_level(logging.WARNING):
            with patch("falkordb.FalkorDB") as mock_falkordb:
                mock_falkordb.side_effect = ImportError("No module")

                await init_falkordb()

                assert any("falkordb package not installed" in msg for msg in caplog.messages)

    @pytest.mark.asyncio
    async def test_falkordb_logging_error(self, caplog):
        """Test FalkorDB client logs errors."""
        from src.api.dependencies.falkordb_client import init_falkordb

        with caplog.at_level(logging.ERROR):
            with patch("falkordb.FalkorDB") as mock_falkordb:
                mock_falkordb.side_effect = Exception("Connection failed")

                with pytest.raises(ConnectionError):
                    await init_falkordb()

                assert any("Failed to connect to FalkorDB" in msg for msg in caplog.messages)
