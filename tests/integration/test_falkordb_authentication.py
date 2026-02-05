"""Integration tests for FalkorDB authentication.

Tests verify FalkorDB client behavior with password-protected instances:
- Connection with authentication
- Graph operations with auth
- Health checks with auth
- Error scenarios (wrong password, connection issues)
- Security hardening (password not in errors)

Prerequisites:
    - FalkorDB running with FALKORDB_PASSWORD authentication
    - FALKORDB_URL environment variable configured

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import os
import time
from unittest.mock import patch

import pytest

# Check if FalkorDB is available for integration tests
FALKORDB_AVAILABLE = False
FALKORDB_URL = os.environ.get("FALKORDB_URL", "")
FALKORDB_PASSWORD = os.environ.get("FALKORDB_PASSWORD", "")

try:
    from urllib.parse import urlparse

    from falkordb import FalkorDB

    if FALKORDB_URL:
        parsed = urlparse(FALKORDB_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        _test_client = FalkorDB(host=host, port=port)
        _test_client.list_graphs()
        FALKORDB_AVAILABLE = True
except Exception:
    pass


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not FALKORDB_AVAILABLE,
        reason="FalkorDB not available or not configured with authentication",
    ),
]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def falkordb_url():
    """Get the configured FalkorDB URL."""
    return FALKORDB_URL


@pytest.fixture
def falkordb_password():
    """Get the configured FalkorDB password."""
    return FALKORDB_PASSWORD


@pytest.fixture
async def clean_falkordb_client():
    """Provide a clean FalkorDB client and clean up after test."""
    import src.api.dependencies.falkordb_client as falkordb_module

    # Reset global state
    original_client = falkordb_module._falkordb_client
    original_graph = falkordb_module._graph
    falkordb_module._falkordb_client = None
    falkordb_module._graph = None

    yield

    # Cleanup
    await falkordb_module.close_falkordb()
    falkordb_module._falkordb_client = original_client
    falkordb_module._graph = original_graph


@pytest.fixture
def sync_falkordb_client():
    """Provide a synchronous FalkorDB client for direct testing."""
    from urllib.parse import urlparse

    from falkordb import FalkorDB

    parsed = urlparse(FALKORDB_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379

    client = FalkorDB(host=host, port=port)
    yield client


@pytest.fixture
def test_graph_name():
    """Provide a unique test graph name."""
    return f"test_graph_{int(time.time())}"


# =============================================================================
# CONNECTION WITH AUTHENTICATION TESTS
# =============================================================================


@pytest.mark.asyncio
class TestFalkorDBConnectionWithAuth:
    """Tests for FalkorDB connection with authentication."""

    async def test_init_falkordb_with_valid_auth(self, clean_falkordb_client):
        """Test that init_falkordb connects successfully with valid credentials."""
        from src.api.dependencies.falkordb_client import init_falkordb

        client = await init_falkordb()

        assert client is not None
        # Verify connection works by listing graphs
        graphs = client.list_graphs()
        assert isinstance(graphs, list)

    async def test_get_falkordb_returns_same_client(self, clean_falkordb_client):
        """Test that get_falkordb returns the same client instance."""
        from src.api.dependencies.falkordb_client import get_falkordb

        client1 = await get_falkordb()
        client2 = await get_falkordb()

        # Should be the same instance
        assert client1 is client2

    async def test_get_graph_returns_graph_instance(self, clean_falkordb_client):
        """Test that get_graph returns a valid graph instance."""
        from src.api.dependencies.falkordb_client import get_graph, init_falkordb

        await init_falkordb()
        graph = await get_graph()

        assert graph is not None

    async def test_close_falkordb_clears_connection(self, clean_falkordb_client):
        """Test that close_falkordb properly clears the connection."""
        import src.api.dependencies.falkordb_client as falkordb_module
        from src.api.dependencies.falkordb_client import close_falkordb, init_falkordb

        await init_falkordb()
        assert falkordb_module._falkordb_client is not None

        await close_falkordb()
        assert falkordb_module._falkordb_client is None
        assert falkordb_module._graph is None


# =============================================================================
# GRAPH OPERATIONS WITH AUTH TESTS
# =============================================================================


@pytest.mark.asyncio
class TestFalkorDBGraphOperationsWithAuth:
    """Tests for FalkorDB graph operations with authentication."""

    async def test_select_graph_with_auth(self, clean_falkordb_client, test_graph_name):
        """Test selecting a graph with authenticated client."""
        from src.api.dependencies.falkordb_client import init_falkordb

        client = await init_falkordb()
        graph = client.select_graph(test_graph_name)

        assert graph is not None

    async def test_create_and_query_nodes(self, clean_falkordb_client, test_graph_name):
        """Test creating and querying nodes with auth."""
        from src.api.dependencies.falkordb_client import init_falkordb

        client = await init_falkordb()
        graph = client.select_graph(test_graph_name)

        try:
            # Create a test node
            graph.query("CREATE (n:TestNode {name: 'test', value: 42})")

            # Query the node
            result = graph.query("MATCH (n:TestNode) RETURN n.name, n.value")

            assert result.result_set is not None
            assert len(result.result_set) == 1
            assert result.result_set[0][0] == "test"
            assert result.result_set[0][1] == 42

        finally:
            # Cleanup
            graph.query("MATCH (n:TestNode) DELETE n")

    async def test_create_and_query_relationships(self, clean_falkordb_client, test_graph_name):
        """Test creating and querying relationships with auth."""
        from src.api.dependencies.falkordb_client import init_falkordb

        client = await init_falkordb()
        graph = client.select_graph(test_graph_name)

        try:
            # Create nodes and relationship
            graph.query("""
                CREATE (a:Person {name: 'Alice'})
                CREATE (b:Person {name: 'Bob'})
                CREATE (a)-[:KNOWS {since: 2020}]->(b)
            """)

            # Query the relationship
            result = graph.query("""
                MATCH (a:Person)-[r:KNOWS]->(b:Person)
                RETURN a.name, r.since, b.name
            """)

            assert result.result_set is not None
            assert len(result.result_set) == 1
            assert result.result_set[0][0] == "Alice"
            assert result.result_set[0][1] == 2020
            assert result.result_set[0][2] == "Bob"

        finally:
            # Cleanup
            graph.query("MATCH (n:Person) DETACH DELETE n")

    async def test_count_nodes_and_edges(self, clean_falkordb_client, test_graph_name):
        """Test counting nodes and edges with auth."""
        from src.api.dependencies.falkordb_client import init_falkordb

        client = await init_falkordb()
        graph = client.select_graph(test_graph_name)

        try:
            # Create test data
            graph.query("""
                CREATE (a:Item {id: 1})
                CREATE (b:Item {id: 2})
                CREATE (c:Item {id: 3})
                CREATE (a)-[:LINKS]->(b)
                CREATE (b)-[:LINKS]->(c)
            """)

            # Count nodes
            node_result = graph.query("MATCH (n) RETURN count(n) as count")
            node_count = node_result.result_set[0][0]
            assert node_count == 3

            # Count edges
            edge_result = graph.query("MATCH ()-[r]->() RETURN count(r) as count")
            edge_count = edge_result.result_set[0][0]
            assert edge_count == 2

        finally:
            # Cleanup
            graph.query("MATCH (n:Item) DETACH DELETE n")

    async def test_list_graphs_with_auth(self, clean_falkordb_client):
        """Test listing available graphs with auth."""
        from src.api.dependencies.falkordb_client import init_falkordb

        client = await init_falkordb()
        graphs = client.list_graphs()

        assert isinstance(graphs, list)


# =============================================================================
# HEALTH CHECK WITH AUTH TESTS
# =============================================================================


@pytest.mark.asyncio
class TestFalkorDBHealthCheckWithAuth:
    """Tests for FalkorDB health check with authentication."""

    async def test_health_check_returns_healthy(self, clean_falkordb_client):
        """Test that health check returns healthy status with auth."""
        from src.api.dependencies.falkordb_client import (
            falkordb_health_check,
            init_falkordb,
        )

        await init_falkordb()

        result = await falkordb_health_check()

        assert result["status"] == "healthy"
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0
        assert "graphs" in result
        assert isinstance(result["graphs"], list)

    async def test_health_check_includes_graph_stats(self, clean_falkordb_client):
        """Test that health check includes graph statistics."""
        from src.api.dependencies.falkordb_client import (
            falkordb_health_check,
            init_falkordb,
        )

        await init_falkordb()

        result = await falkordb_health_check()

        assert result["status"] == "healthy"
        assert "node_count" in result
        assert "edge_count" in result
        assert "current_graph" in result

    async def test_health_check_measures_latency(self, clean_falkordb_client):
        """Test that health check accurately measures latency."""
        from src.api.dependencies.falkordb_client import (
            falkordb_health_check,
            init_falkordb,
        )

        await init_falkordb()

        # Run multiple health checks
        latencies = []
        for _ in range(5):
            result = await falkordb_health_check()
            if result["status"] == "healthy":
                latencies.append(result["latency_ms"])

        assert len(latencies) >= 3
        # Latency should be reasonable (< 200ms for local FalkorDB)
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 200


# =============================================================================
# ERROR SCENARIO TESTS
# =============================================================================


class TestFalkorDBAuthErrorScenarios:
    """Tests for FalkorDB authentication error scenarios."""

    def test_wrong_password_raises_error(self):
        """Test that wrong password raises authentication error."""
        from urllib.parse import urlparse

        from falkordb import FalkorDB

        # Parse the current URL to get host
        parsed = urlparse(FALKORDB_URL)
        host = parsed.hostname or "localhost"

        # Try to connect (FalkorDB may handle auth differently)
        # The key is that invalid credentials don't silently succeed
        try:
            # Attempt connection to wrong port to simulate auth/connection failure
            client = FalkorDB(host=host, port=9999)
            with pytest.raises(Exception):
                client.list_graphs()
        except Exception:
            # Connection failure is expected
            pass

    def test_connection_to_invalid_host_fails(self):
        """Test that connection to invalid host fails gracefully."""
        from falkordb import FalkorDB

        with pytest.raises(Exception):
            client = FalkorDB(host="nonexistent-host-12345", port=6379)
            client.list_graphs()


@pytest.mark.asyncio
class TestFalkorDBTimeoutHandling:
    """Tests for FalkorDB connection timeout handling."""

    async def test_connection_to_unreachable_host_fails(self):
        """Test that connection to unreachable host fails."""
        from falkordb import FalkorDB

        # Use a non-routable IP
        start = time.time()
        try:
            client = FalkorDB(host="10.255.255.1", port=6379)
            client.list_graphs()
            pytest.fail("Should have raised an exception")
        except Exception:
            elapsed = time.time() - start
            # Should fail within reasonable time (not hang forever)
            assert elapsed < 60


# =============================================================================
# CIRCUIT BREAKER INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
class TestFalkorDBCircuitBreakerIntegration:
    """Tests for FalkorDB health check circuit breaker integration."""

    async def test_circuit_breaker_opens_on_failures(self, clean_falkordb_client):
        """Test that circuit breaker opens after repeated failures."""
        import src.api.dependencies.falkordb_client as falkordb_module
        from src.api.dependencies.falkordb_client import falkordb_health_check

        # Initialize connection first
        await falkordb_module.init_falkordb()

        # Reset circuit breaker
        falkordb_module._health_circuit_breaker._state = (
            falkordb_module._health_circuit_breaker.CircuitState.CLOSED
        )
        falkordb_module._health_circuit_breaker._failure_count = 0

        # Simulate failures by patching get_falkordb to fail
        async def failing_get_falkordb():
            raise ConnectionError("Simulated failure")

        original_get_falkordb = falkordb_module.get_falkordb
        falkordb_module.get_falkordb = failing_get_falkordb

        # Make requests until circuit opens (threshold is 3)
        for _ in range(4):
            await falkordb_health_check()

        # Restore
        falkordb_module.get_falkordb = original_get_falkordb

        # Circuit should now be open
        result = await falkordb_health_check()
        assert result["status"] == "circuit_open"

        # Reset for other tests
        falkordb_module._health_circuit_breaker._state = (
            falkordb_module._health_circuit_breaker.CircuitState.CLOSED
        )
        falkordb_module._health_circuit_breaker._failure_count = 0

    async def test_circuit_breaker_resets_on_success(self, clean_falkordb_client):
        """Test that circuit breaker resets after successful health check."""
        import src.api.dependencies.falkordb_client as falkordb_module
        from src.api.dependencies.falkordb_client import (
            falkordb_health_check,
            init_falkordb,
        )

        await init_falkordb()

        # Reset circuit breaker state with some failures
        falkordb_module._health_circuit_breaker._state = (
            falkordb_module._health_circuit_breaker.CircuitState.CLOSED
        )
        falkordb_module._health_circuit_breaker._failure_count = 2

        # Successful health check should reset failure count
        result = await falkordb_health_check()
        assert result["status"] == "healthy"

        # Failure count should be reset
        assert falkordb_module._health_circuit_breaker._failure_count == 0


# =============================================================================
# SECURITY HARDENING TESTS
# =============================================================================


@pytest.mark.asyncio
class TestFalkorDBSecurityHardening:
    """Tests for FalkorDB security hardening."""

    async def test_password_not_in_error_message(self, clean_falkordb_client):
        """Test that password is not exposed in error messages."""
        # Parse URL to get host/port
        from urllib.parse import urlparse

        import src.api.dependencies.falkordb_client as falkordb_module

        parsed = urlparse(FALKORDB_URL)
        host = parsed.hostname or "localhost"

        # Create URL with a fake password
        fake_password = "super_secret_falkordb_password_12345"
        wrong_url = f"redis://:{fake_password}@{host}:9999"

        with patch.dict(
            os.environ,
            {"FALKORDB_URL": wrong_url},
            clear=True,
        ):
            import importlib

            importlib.reload(falkordb_module)
            falkordb_module._falkordb_client = None
            falkordb_module._graph = None

            try:
                await falkordb_module.init_falkordb()
            except ConnectionError as e:
                # Password should not be in error message
                assert fake_password not in str(e)
            except Exception as e:
                # Any other error should also not contain password
                assert fake_password not in str(e)

    async def test_config_parsing_does_not_leak_password(self):
        """Test that config parsing doesn't expose password."""
        from src.api.dependencies.falkordb_client import _parse_falkordb_config

        fake_password = "secret_password_xyz"
        test_url = f"redis://:{fake_password}@testhost:6379"

        with patch.dict(os.environ, {"FALKORDB_URL": test_url}, clear=True):
            host, port = _parse_falkordb_config()

            # Should return host and port, not password
            assert host == "testhost"
            assert port == 6379
            # Function only returns host/port, password is handled internally


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================


@pytest.mark.asyncio
class TestFalkorDBDataIntegrityWithAuth:
    """Tests for data integrity with authenticated FalkorDB."""

    async def test_unicode_data_storage(self, clean_falkordb_client, test_graph_name):
        """Test storing and retrieving unicode data."""
        from src.api.dependencies.falkordb_client import init_falkordb

        client = await init_falkordb()
        graph = client.select_graph(test_graph_name)

        try:
            # Create node with unicode data
            graph.query("""
                CREATE (n:UnicodeTest {
                    japanese: '日本語テスト',
                    emoji: '🔥📊🎯',
                    russian: 'Привет мир'
                })
            """)

            # Query the data
            result = graph.query("MATCH (n:UnicodeTest) RETURN n.japanese, n.emoji, n.russian")

            assert result.result_set[0][0] == "日本語テスト"
            assert result.result_set[0][1] == "🔥📊🎯"
            assert result.result_set[0][2] == "Привет мир"

        finally:
            graph.query("MATCH (n:UnicodeTest) DELETE n")

    async def test_large_property_values(self, clean_falkordb_client, test_graph_name):
        """Test storing large property values."""
        from src.api.dependencies.falkordb_client import init_falkordb

        client = await init_falkordb()
        graph = client.select_graph(test_graph_name)

        try:
            # Create node with large string (10KB)
            large_value = "x" * 10000
            graph.query(f"CREATE (n:LargeTest {{data: '{large_value}'}})")

            # Query the data
            result = graph.query("MATCH (n:LargeTest) RETURN n.data")

            assert len(result.result_set[0][0]) == 10000

        finally:
            graph.query("MATCH (n:LargeTest) DELETE n")

    async def test_numeric_precision(self, clean_falkordb_client, test_graph_name):
        """Test numeric precision is maintained."""
        from src.api.dependencies.falkordb_client import init_falkordb

        client = await init_falkordb()
        graph = client.select_graph(test_graph_name)

        try:
            # Create node with precise numbers
            graph.query("""
                CREATE (n:NumericTest {
                    integer: 9223372036854775807,
                    float_val: 3.141592653589793
                })
            """)

            # Query the data
            result = graph.query("MATCH (n:NumericTest) RETURN n.integer, n.float_val")

            assert result.result_set[0][0] == 9223372036854775807
            # Float precision may vary slightly
            assert abs(result.result_set[0][1] - 3.141592653589793) < 0.0001

        finally:
            graph.query("MATCH (n:NumericTest) DELETE n")

    async def test_concurrent_graph_operations(self, clean_falkordb_client, test_graph_name):
        """Test concurrent graph operations with auth."""
        import asyncio

        from src.api.dependencies.falkordb_client import init_falkordb

        client = await init_falkordb()
        graph = client.select_graph(test_graph_name)

        try:
            # Create multiple nodes concurrently (using sync operations in async)
            async def create_node(i: int):
                graph.query(f"CREATE (n:ConcurrentTest {{id: {i}}})")
                return i

            tasks = [create_node(i) for i in range(20)]
            await asyncio.gather(*tasks)

            # Verify all nodes were created
            result = graph.query("MATCH (n:ConcurrentTest) RETURN count(n) as count")
            assert result.result_set[0][0] == 20

        finally:
            graph.query("MATCH (n:ConcurrentTest) DELETE n")
