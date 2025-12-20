"""
Tests for MemoryConnector.

Tests the memory backend integration for RAG retrieval.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.rag.memory_connector import (
    MemoryConnector,
    get_memory_connector,
    reset_memory_connector,
)
from src.rag.models.retrieval_models import RetrievalResult


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def memory_connector():
    """Create a fresh MemoryConnector instance."""
    reset_memory_connector()
    return MemoryConnector()


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client."""
    client = MagicMock()
    # Configure RPC mock to return chainable mock
    rpc_mock = MagicMock()
    execute_mock = MagicMock()
    execute_mock.data = []
    rpc_mock.execute.return_value = execute_mock
    client.rpc.return_value = rpc_mock
    return client


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = MagicMock()
    service.embed = AsyncMock(return_value=[0.1] * 384)
    return service


@pytest.fixture
def mock_semantic_memory():
    """Mock semantic memory."""
    memory = MagicMock()
    memory.traverse_causal_chain.return_value = []
    memory.get_patient_network.return_value = {}
    memory.get_hcp_influence_network.return_value = {}
    memory.find_common_paths.return_value = []
    memory.find_causal_paths_for_kpi.return_value = []
    return memory


# ============================================================================
# MemoryConnector Initialization Tests
# ============================================================================

class TestMemoryConnectorInit:
    """Test MemoryConnector initialization."""

    def test_initialization(self, memory_connector):
        """Test basic initialization."""
        assert memory_connector._embedding_service is None

    @pytest.mark.asyncio
    async def test_lazy_embedding_service(self, memory_connector, mock_embedding_service):
        """Test lazy initialization of embedding service."""
        with patch(
            "src.rag.memory_connector.get_embedding_service",
            return_value=mock_embedding_service
        ):
            service = await memory_connector.get_embedding_service()
            assert service is not None
            # Second call should return same instance
            service2 = await memory_connector.get_embedding_service()
            assert service is service2


# ============================================================================
# Vector Search Tests
# ============================================================================

class TestVectorSearch:
    """Test vector_search method."""

    @pytest.mark.asyncio
    async def test_vector_search_basic(self, memory_connector, mock_supabase_client):
        """Test basic vector search."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = [
            {
                "content": "Test content",
                "source_table": "episodic_memories",
                "id": "mem-123",
                "similarity": 0.85,
                "metadata": {"brand": "Kisqali"}
            }
        ]

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.vector_search(
                query_embedding=[0.1] * 384,
                k=10
            )

            assert len(results) == 1
            assert isinstance(results[0], RetrievalResult)
            assert results[0].content == "Test content"
            assert results[0].source == "episodic_memories"
            assert results[0].source_id == "mem-123"
            assert results[0].score == 0.85
            assert results[0].retrieval_method == "dense"
            assert results[0].metadata["brand"] == "Kisqali"

    @pytest.mark.asyncio
    async def test_vector_search_with_filters(self, memory_connector, mock_supabase_client):
        """Test vector search with filters."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = [
            {"content": "Filtered content", "source_table": "test", "id": "1", "similarity": 0.9, "metadata": {}}
        ]

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.vector_search(
                query_embedding=[0.1] * 384,
                k=5,
                filters={"brand": "Fabhalta", "region": "Northeast"}
            )

            # Verify RPC was called with filters
            call_args = mock_supabase_client.rpc.call_args
            assert call_args[0][0] == "hybrid_vector_search"
            assert "filters" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_vector_search_min_similarity_filter(self, memory_connector, mock_supabase_client):
        """Test that results below min_similarity are filtered out."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = [
            {"content": "High score", "source_table": "test", "id": "1", "similarity": 0.8, "metadata": {}},
            {"content": "Low score", "source_table": "test", "id": "2", "similarity": 0.3, "metadata": {}},
        ]

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.vector_search(
                query_embedding=[0.1] * 384,
                k=10,
                min_similarity=0.5
            )

            assert len(results) == 1
            assert results[0].content == "High score"

    @pytest.mark.asyncio
    async def test_vector_search_empty_results(self, memory_connector, mock_supabase_client):
        """Test vector search with no results."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = []

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.vector_search(
                query_embedding=[0.1] * 384,
                k=10
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_error_handling(self, memory_connector, mock_supabase_client):
        """Test vector search error handling."""
        mock_supabase_client.rpc.side_effect = Exception("Database error")

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.vector_search(
                query_embedding=[0.1] * 384,
                k=10
            )

            assert results == []


class TestVectorSearchByText:
    """Test vector_search_by_text method."""

    @pytest.mark.asyncio
    async def test_vector_search_by_text(self, memory_connector, mock_supabase_client, mock_embedding_service):
        """Test text-based vector search."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = [
            {"content": "Result", "source_table": "test", "id": "1", "similarity": 0.9, "metadata": {}}
        ]

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ), patch(
            "src.rag.memory_connector.get_embedding_service",
            return_value=mock_embedding_service
        ):
            results = await memory_connector.vector_search_by_text(
                query_text="What is the impact of Kisqali?",
                k=10
            )

            # Verify embedding was generated
            mock_embedding_service.embed.assert_called_once_with("What is the impact of Kisqali?")
            assert len(results) == 1


# ============================================================================
# Full-Text Search Tests
# ============================================================================

class TestFulltextSearch:
    """Test fulltext_search method."""

    @pytest.mark.asyncio
    async def test_fulltext_search_basic(self, memory_connector, mock_supabase_client):
        """Test basic full-text search."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = [
            {"content": "Causal path result", "source_table": "causal_paths", "id": "cp-1", "rank": 0.9, "metadata": {}},
            {"content": "Agent activity", "source_table": "agent_activities", "id": "aa-1", "rank": 0.6, "metadata": {}}
        ]

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.fulltext_search(
                query_text="Kisqali adoption",
                k=10
            )

            assert len(results) == 2
            assert results[0].retrieval_method == "sparse"
            # Score should be normalized
            assert results[0].score == 1.0  # Max score normalized to 1
            assert results[1].score == pytest.approx(0.666, rel=0.01)

    @pytest.mark.asyncio
    async def test_fulltext_search_with_filters(self, memory_connector, mock_supabase_client):
        """Test full-text search with filters."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = []

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            await memory_connector.fulltext_search(
                query_text="TRx trend",
                k=5,
                filters={"brand": "Remibrutinib"}
            )

            # Verify RPC was called with correct function
            call_args = mock_supabase_client.rpc.call_args
            assert call_args[0][0] == "hybrid_fulltext_search"
            assert call_args[0][1]["search_query"] == "TRx trend"

    @pytest.mark.asyncio
    async def test_fulltext_search_empty_results(self, memory_connector, mock_supabase_client):
        """Test full-text search with no results."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = []

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.fulltext_search(
                query_text="nonexistent query",
                k=10
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_fulltext_search_zero_rank(self, memory_connector, mock_supabase_client):
        """Test handling of zero rank results."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = [
            {"content": "Result", "source_table": "test", "id": "1", "rank": 0, "metadata": {}}
        ]

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.fulltext_search(
                query_text="test",
                k=10
            )

            # Should handle zero max_rank without division error
            assert len(results) == 1
            assert results[0].score == 0.0

    @pytest.mark.asyncio
    async def test_fulltext_search_error_handling(self, memory_connector, mock_supabase_client):
        """Test full-text search error handling."""
        mock_supabase_client.rpc.side_effect = Exception("Search failed")

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.fulltext_search(
                query_text="test query",
                k=10
            )

            assert results == []


# ============================================================================
# Graph Traversal Tests
# ============================================================================

class TestGraphTraverse:
    """Test graph_traverse method."""

    def test_graph_traverse_causal_path(self, memory_connector, mock_semantic_memory):
        """Test graph traversal for causal paths."""
        mock_semantic_memory.traverse_causal_chain.return_value = [
            {
                "path": ["Sales visit", "HCP awareness", "TRx increase"],
                "confidence": 0.85,
                "start_entity_id": "ent-1",
                "path_length": 3,
                "relationships": ["leads_to", "results_in"],
                "effect_sizes": [0.3, 0.5]
            }
        ]

        with patch(
            "src.rag.memory_connector.get_semantic_memory",
            return_value=mock_semantic_memory
        ):
            results = memory_connector.graph_traverse(
                entity_id="ent-1",
                relationship="causal_path",
                max_depth=3
            )

            assert len(results) == 1
            assert results[0].retrieval_method == "graph"
            assert results[0].source == "semantic_graph"
            assert "Sales visit" in results[0].content
            assert results[0].score == 0.85

    def test_graph_traverse_patient_network(self, memory_connector, mock_semantic_memory):
        """Test graph traversal for patient network."""
        mock_semantic_memory.get_patient_network.return_value = {
            "center_node": {"id": "patient-1", "properties": {"stage": "Treatment"}},
            "connections": [
                {"node_id": "hcp-1", "node_type": "HCP", "relationship": "treated_by", "depth": 1, "properties": {}}
            ]
        }

        with patch(
            "src.rag.memory_connector.get_semantic_memory",
            return_value=mock_semantic_memory
        ):
            results = memory_connector.graph_traverse(
                entity_id="patient-1",
                relationship="patient_network",
                max_depth=2
            )

            assert len(results) == 2  # Center + 1 connection
            # Center node should have score 1.0
            assert results[0].score == 1.0
            assert "PATIENT" in results[0].content.upper()

    def test_graph_traverse_hcp_network(self, memory_connector, mock_semantic_memory):
        """Test graph traversal for HCP influence network."""
        mock_semantic_memory.get_hcp_influence_network.return_value = {
            "center_node": {"id": "hcp-1", "properties": {"specialty": "Oncology"}},
            "connections": [
                {"node_id": "hcp-2", "node_type": "HCP", "relationship": "influences", "depth": 1, "properties": {}},
                {"node_id": "hcp-3", "node_type": "HCP", "relationship": "influences", "depth": 2, "properties": {}}
            ]
        }

        with patch(
            "src.rag.memory_connector.get_semantic_memory",
            return_value=mock_semantic_memory
        ):
            results = memory_connector.graph_traverse(
                entity_id="hcp-1",
                relationship="hcp_network",
                max_depth=3
            )

            assert len(results) == 3
            # Scores should decrease with depth
            assert results[1].score > results[2].score

    def test_graph_traverse_generic(self, memory_connector, mock_semantic_memory):
        """Test generic graph traversal."""
        mock_semantic_memory.find_common_paths.return_value = [
            {"nodes": ["A", "B", "C"], "path_id": "path-1", "confidence": 0.7, "length": 3, "relationships": []}
        ]

        with patch(
            "src.rag.memory_connector.get_semantic_memory",
            return_value=mock_semantic_memory
        ):
            results = memory_connector.graph_traverse(
                entity_id="ent-1",
                relationship="other_relationship",
                max_depth=2
            )

            assert len(results) == 1
            assert "A → B → C" in results[0].content

    def test_graph_traverse_error_handling(self, memory_connector, mock_semantic_memory):
        """Test graph traversal error handling."""
        mock_semantic_memory.traverse_causal_chain.side_effect = Exception("Graph error")

        with patch(
            "src.rag.memory_connector.get_semantic_memory",
            return_value=mock_semantic_memory
        ):
            results = memory_connector.graph_traverse(
                entity_id="ent-1",
                relationship="causal_path"
            )

            assert results == []


class TestGraphTraverseKPI:
    """Test graph_traverse_kpi method."""

    def test_graph_traverse_kpi_basic(self, memory_connector, mock_semantic_memory):
        """Test KPI path traversal."""
        mock_semantic_memory.find_causal_paths_for_kpi.return_value = [
            {
                "nodes": ["Rep visits", "HCP engagement", "TRx"],
                "path_id": "kpi-path-1",
                "confidence": 0.8,
                "length": 3,
                "relationships": ["causes", "impacts"],
                "kpi_impact": 0.15
            }
        ]

        with patch(
            "src.rag.memory_connector.get_semantic_memory",
            return_value=mock_semantic_memory
        ):
            results = memory_connector.graph_traverse_kpi(
                kpi_name="TRx",
                min_confidence=0.5
            )

            assert len(results) == 1
            assert results[0].metadata["kpi_impact"] == 0.15
            mock_semantic_memory.find_causal_paths_for_kpi.assert_called_with(
                "TRx", min_confidence=0.5
            )

    def test_graph_traverse_kpi_error_handling(self, memory_connector, mock_semantic_memory):
        """Test KPI traversal error handling."""
        mock_semantic_memory.find_causal_paths_for_kpi.side_effect = Exception("KPI error")

        with patch(
            "src.rag.memory_connector.get_semantic_memory",
            return_value=mock_semantic_memory
        ):
            results = memory_connector.graph_traverse_kpi(
                kpi_name="NRx"
            )

            assert results == []


# ============================================================================
# Result Conversion Helper Tests
# ============================================================================

class TestChainsToResults:
    """Test _chains_to_results helper method."""

    def test_chains_to_results_basic(self, memory_connector):
        """Test converting chains to results."""
        chains = [
            {
                "path": ["A", "B", "C"],
                "confidence": 0.9,
                "start_entity_id": "ent-1",
                "path_length": 3,
                "relationships": ["r1", "r2"],
                "effect_sizes": [0.5, 0.3]
            }
        ]

        results = memory_connector._chains_to_results(chains)

        assert len(results) == 1
        assert "A → B → C" in results[0].content
        assert results[0].score == 0.9
        assert results[0].metadata["path_length"] == 3

    def test_chains_to_results_empty_path(self, memory_connector):
        """Test with empty path list."""
        chains = [{"path": [], "confidence": 0.5}]

        results = memory_connector._chains_to_results(chains)

        assert len(results) == 1
        assert "Causal chain 1" in results[0].content

    def test_chains_to_results_multiple(self, memory_connector):
        """Test multiple chains."""
        chains = [
            {"path": ["X", "Y"], "confidence": 0.8, "start_entity_id": "e1"},
            {"path": ["P", "Q", "R"], "confidence": 0.7, "start_entity_id": "e2"}
        ]

        results = memory_connector._chains_to_results(chains)

        assert len(results) == 2


class TestNetworkToResults:
    """Test _network_to_results helper method."""

    def test_network_to_results_with_center(self, memory_connector):
        """Test converting network with center node."""
        network = {
            "center_node": {"id": "center-1", "properties": {"type": "HCP"}},
            "connections": []
        }

        results = memory_connector._network_to_results(network, "hcp")

        assert len(results) == 1
        assert results[0].score == 1.0
        assert "center-1" in results[0].content

    def test_network_to_results_with_connections(self, memory_connector):
        """Test converting network with connections."""
        network = {
            "center_node": {"id": "c1", "properties": {}},
            "connections": [
                {"node_id": "n1", "node_type": "Entity", "relationship": "relates", "depth": 1, "properties": {}},
                {"node_id": "n2", "node_type": "Entity", "relationship": "relates", "depth": 2, "properties": {}}
            ]
        }

        results = memory_connector._network_to_results(network, "entity")

        assert len(results) == 3
        # Scores decrease with depth
        assert results[1].score > results[2].score

    def test_network_to_results_empty(self, memory_connector):
        """Test empty network."""
        network = {"connections": []}

        results = memory_connector._network_to_results(network, "test")

        assert len(results) == 0


class TestPathsToResults:
    """Test _paths_to_results helper method."""

    def test_paths_to_results_basic(self, memory_connector):
        """Test converting paths to results."""
        paths = [
            {
                "nodes": ["Start", "Middle", "End"],
                "path_id": "p1",
                "confidence": 0.85,
                "length": 3,
                "relationships": ["r1", "r2"],
                "kpi_impact": 0.2
            }
        ]

        results = memory_connector._paths_to_results(paths)

        assert len(results) == 1
        assert "Start → Middle → End" in results[0].content
        assert results[0].metadata["kpi_impact"] == 0.2

    def test_paths_to_results_empty_nodes(self, memory_connector):
        """Test with empty nodes."""
        paths = [{"nodes": [], "confidence": 0.5}]

        results = memory_connector._paths_to_results(paths)

        assert len(results) == 1
        assert "Path 1" in results[0].content


# ============================================================================
# Singleton Tests
# ============================================================================

class TestSingleton:
    """Test singleton pattern for memory connector."""

    def test_get_memory_connector_singleton(self):
        """Test that get_memory_connector returns singleton."""
        reset_memory_connector()

        connector1 = get_memory_connector()
        connector2 = get_memory_connector()

        assert connector1 is connector2

    def test_reset_memory_connector(self):
        """Test resetting singleton."""
        reset_memory_connector()

        connector1 = get_memory_connector()
        reset_memory_connector()
        connector2 = get_memory_connector()

        assert connector1 is not connector2


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_vector_search_null_data(self, memory_connector, mock_supabase_client):
        """Test handling of null data response."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = None

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.vector_search(
                query_embedding=[0.1] * 384,
                k=10
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_missing_fields(self, memory_connector, mock_supabase_client):
        """Test handling of results with missing fields."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = [
            {"content": "Test", "similarity": 0.8}  # Missing source_table, id, metadata
        ]

        with patch(
            "src.rag.memory_connector.get_supabase_client",
            return_value=mock_supabase_client
        ):
            results = await memory_connector.vector_search(
                query_embedding=[0.1] * 384,
                k=10
            )

            assert len(results) == 1
            assert results[0].source == "unknown"
            assert results[0].source_id == ""
            assert results[0].metadata == {}

    def test_depth_score_calculation(self, memory_connector):
        """Test that connection scores decrease with depth correctly."""
        network = {
            "center_node": {"id": "c", "properties": {}},
            "connections": [
                {"node_id": "n1", "node_type": "E", "relationship": "r", "depth": 1, "properties": {}},
                {"node_id": "n2", "node_type": "E", "relationship": "r", "depth": 2, "properties": {}},
                {"node_id": "n3", "node_type": "E", "relationship": "r", "depth": 3, "properties": {}},
                {"node_id": "n4", "node_type": "E", "relationship": "r", "depth": 4, "properties": {}},
                {"node_id": "n5", "node_type": "E", "relationship": "r", "depth": 5, "properties": {}}
            ]
        }

        results = memory_connector._network_to_results(network, "test")

        # Check scores: 1.0 for center, then decreasing
        assert results[0].score == 1.0  # center
        assert results[1].score == 0.8  # depth 1: 1.0 - 0.2*1
        assert results[2].score == 0.6  # depth 2: 1.0 - 0.2*2
        assert results[3].score == 0.5  # depth 3: max(0.5, 1.0 - 0.6) = 0.5
        assert results[4].score == 0.5  # depth 4: max(0.5, 1.0 - 0.8) = 0.5
        assert results[5].score == 0.5  # depth 5: max(0.5, 1.0 - 1.0) = 0.5
