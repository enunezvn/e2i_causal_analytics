"""
Unit tests for RAG Memory Connector.

Tests the bridge between RAG retriever and memory backends.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# MODULE ISOLATION FIX
# ============================================================================
# CRITICAL: test_cognitive_backends.py pollutes sys.modules by replacing
# 'src.rag.memory_connector' with a MagicMock at import time. This causes
# all tests in this file to fail when run as part of the full suite.
# Solution: Force re-import of the real module before running tests.

# Store reference to real module (if already imported)
_real_memory_connector_module = sys.modules.get("src.rag.memory_connector")

# Remove any mock from sys.modules and force real import
if "src.rag.memory_connector" in sys.modules:
    if isinstance(sys.modules["src.rag.memory_connector"], MagicMock):
        del sys.modules["src.rag.memory_connector"]

# Import the real module
from src.rag.memory_connector import (
    MemoryConnector,
    get_memory_connector,
    reset_memory_connector,
)
from src.rag.types import RetrievalSource

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def ensure_real_module():
    """
    Ensure real memory_connector module is loaded before each test.

    Protects against pollution from test_cognitive_backends.py which
    replaces sys.modules['src.rag.memory_connector'] with a MagicMock.
    """
    # Store original (should be real module now after import above)
    original = sys.modules.get("src.rag.memory_connector")

    # Force real module if it got replaced
    if isinstance(sys.modules.get("src.rag.memory_connector"), MagicMock):
        sys.modules["src.rag.memory_connector"] = original

    # Reset singleton state before each test
    reset_memory_connector()

    yield

    # Reset singleton state after each test
    reset_memory_connector()


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value.data = []
    return client


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = MagicMock()
    service.embed = AsyncMock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def mock_semantic_memory():
    """Create a mock semantic memory."""
    memory = MagicMock()
    memory.traverse_causal_chain.return_value = []
    memory.get_patient_network.return_value = {"center_node": {}, "connections": []}
    memory.get_hcp_influence_network.return_value = {"center_node": {}, "connections": []}
    memory.find_common_paths.return_value = []
    memory.find_causal_paths_for_kpi.return_value = []
    return memory


@pytest.fixture
def memory_connector(mock_supabase, mock_embedding_service):
    """Create memory connector with mocks."""
    reset_memory_connector()
    with patch("src.rag.memory_connector.get_supabase_client", return_value=mock_supabase):
        with patch(
            "src.rag.memory_connector.get_embedding_service", return_value=mock_embedding_service
        ):
            connector = MemoryConnector()
            return connector


# ============================================================================
# VECTOR SEARCH TESTS
# ============================================================================


class TestVectorSearch:
    """Tests for vector similarity search."""

    @pytest.mark.asyncio
    async def test_vector_search_returns_results(self, mock_supabase, mock_embedding_service):
        """vector_search should return results from Supabase RPC."""
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {
                "id": "mem_1",
                "content": "Test memory content",
                "similarity": 0.85,
                "metadata": {"event_type": "query"},
                "source_table": "episodic_memories",
            }
        ]

        with patch("src.rag.memory_connector.get_supabase_client", return_value=mock_supabase):
            connector = MemoryConnector()
            results = await connector.vector_search(query_embedding=[0.1] * 1536, k=10)

        assert len(results) == 1
        assert results[0].content == "Test memory content"
        assert results[0].source == RetrievalSource.VECTOR.value
        assert results[0].score == 0.85
        assert results[0].retrieval_method == "dense"
        assert results[0].metadata.get("source_name") == "episodic_memories"

    @pytest.mark.asyncio
    async def test_vector_search_filters_by_similarity(self, mock_supabase):
        """vector_search should filter results below min_similarity."""
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {
                "id": "mem_1",
                "content": "High similarity",
                "similarity": 0.9,
                "metadata": {},
                "source_table": "episodic_memories",
            },
            {
                "id": "mem_2",
                "content": "Low similarity",
                "similarity": 0.3,
                "metadata": {},
                "source_table": "episodic_memories",
            },
        ]

        with patch("src.rag.memory_connector.get_supabase_client", return_value=mock_supabase):
            connector = MemoryConnector()
            results = await connector.vector_search(
                query_embedding=[0.1] * 1536, k=10, min_similarity=0.5
            )

        assert len(results) == 1
        assert results[0].content == "High similarity"

    @pytest.mark.asyncio
    async def test_vector_search_passes_filters(self, mock_supabase):
        """vector_search should pass filters to RPC call."""
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.rag.memory_connector.get_supabase_client", return_value=mock_supabase):
            connector = MemoryConnector()
            await connector.vector_search(
                query_embedding=[0.1] * 1536,
                k=10,
                filters={"brand": "Kisqali", "region": "northeast"},
            )

        mock_supabase.rpc.assert_called_once()
        call_args = mock_supabase.rpc.call_args
        assert call_args[0][0] == "hybrid_vector_search"
        # RPC uses positional dict argument
        assert call_args[0][1]["filters"] == {"brand": "Kisqali", "region": "northeast"}

    @pytest.mark.asyncio
    async def test_vector_search_by_text_generates_embedding(
        self, mock_supabase, mock_embedding_service
    ):
        """vector_search_by_text should auto-generate embedding."""
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.rag.memory_connector.get_supabase_client", return_value=mock_supabase):
            with patch(
                "src.rag.memory_connector.get_embedding_service",
                return_value=mock_embedding_service,
            ):
                connector = MemoryConnector()
                await connector.vector_search_by_text(query_text="Why did TRx drop?", k=10)

        mock_embedding_service.embed.assert_called_once_with("Why did TRx drop?")

    @pytest.mark.asyncio
    async def test_vector_search_handles_error(self, mock_supabase):
        """vector_search should return empty list on error."""
        mock_supabase.rpc.side_effect = Exception("Connection failed")

        with patch("src.rag.memory_connector.get_supabase_client", return_value=mock_supabase):
            connector = MemoryConnector()
            results = await connector.vector_search(query_embedding=[0.1] * 1536, k=10)

        assert results == []


# ============================================================================
# FULLTEXT SEARCH TESTS
# ============================================================================


class TestFulltextSearch:
    """Tests for full-text search."""

    @pytest.mark.asyncio
    async def test_fulltext_search_returns_results(self, mock_supabase):
        """fulltext_search should return results from Supabase RPC."""
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {
                "id": "path_1",
                "content": "HCP engagement → TRx increase",
                "rank": 0.85,
                "metadata": {"start_node": "HCP engagement"},
                "source_table": "causal_paths",
            }
        ]

        with patch("src.rag.memory_connector.get_supabase_client", return_value=mock_supabase):
            connector = MemoryConnector()
            results = await connector.fulltext_search(query_text="TRx increase", k=10)

        assert len(results) == 1
        assert results[0].content == "HCP engagement → TRx increase"
        assert results[0].source == RetrievalSource.FULLTEXT.value
        assert results[0].retrieval_method == "sparse"
        assert results[0].metadata.get("source_name") == "causal_paths"

    @pytest.mark.asyncio
    async def test_fulltext_search_normalizes_scores(self, mock_supabase):
        """fulltext_search should normalize rank scores to 0-1."""
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {
                "id": "1",
                "content": "High rank",
                "rank": 1.0,
                "metadata": {},
                "source_table": "causal_paths",
            },
            {
                "id": "2",
                "content": "Half rank",
                "rank": 0.5,
                "metadata": {},
                "source_table": "causal_paths",
            },
        ]

        with patch("src.rag.memory_connector.get_supabase_client", return_value=mock_supabase):
            connector = MemoryConnector()
            results = await connector.fulltext_search(query_text="test", k=10)

        assert results[0].score == 1.0
        assert results[1].score == 0.5

    @pytest.mark.asyncio
    async def test_fulltext_search_handles_empty_results(self, mock_supabase):
        """fulltext_search should handle empty results."""
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.rag.memory_connector.get_supabase_client", return_value=mock_supabase):
            connector = MemoryConnector()
            results = await connector.fulltext_search(query_text="nonexistent query", k=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_fulltext_search_handles_error(self, mock_supabase):
        """fulltext_search should return empty list on error."""
        mock_supabase.rpc.side_effect = Exception("Connection failed")

        with patch("src.rag.memory_connector.get_supabase_client", return_value=mock_supabase):
            connector = MemoryConnector()
            results = await connector.fulltext_search(query_text="test", k=10)

        assert results == []


# ============================================================================
# GRAPH TRAVERSAL TESTS
# ============================================================================


class TestGraphTraverse:
    """Tests for graph traversal."""

    def test_graph_traverse_causal_path(self, mock_semantic_memory):
        """graph_traverse should traverse causal chains."""
        mock_semantic_memory.traverse_causal_chain.return_value = [
            {
                "path": ["HCP engagement", "Script volume", "TRx"],
                "confidence": 0.85,
                "start_entity_id": "ent_1",
            }
        ]

        with patch(
            "src.rag.memory_connector.get_semantic_memory", return_value=mock_semantic_memory
        ):
            connector = MemoryConnector()
            results = connector.graph_traverse(
                entity_id="ent_1", relationship="causal_path", max_depth=3
            )

        assert len(results) == 1
        assert "HCP engagement" in results[0].content
        assert results[0].retrieval_method == "graph"
        mock_semantic_memory.traverse_causal_chain.assert_called_once_with("ent_1", max_depth=3)

    def test_graph_traverse_patient_network(self, mock_semantic_memory):
        """graph_traverse should traverse patient networks."""
        mock_semantic_memory.get_patient_network.return_value = {
            "center_node": {"id": "pat_123", "properties": {"name": "John"}},
            "connections": [
                {"node_id": "hcp_1", "node_type": "HCP", "relationship": "TREATED_BY", "depth": 1}
            ],
        }

        with patch(
            "src.rag.memory_connector.get_semantic_memory", return_value=mock_semantic_memory
        ):
            connector = MemoryConnector()
            results = connector.graph_traverse(
                entity_id="pat_123", relationship="patient_network", max_depth=2
            )

        assert len(results) == 2  # center + 1 connection
        mock_semantic_memory.get_patient_network.assert_called_once_with("pat_123", max_depth=2)

    def test_graph_traverse_hcp_network(self, mock_semantic_memory):
        """graph_traverse should traverse HCP networks."""
        mock_semantic_memory.get_hcp_influence_network.return_value = {
            "center_node": {"id": "hcp_1"},
            "connections": [],
        }

        with patch(
            "src.rag.memory_connector.get_semantic_memory", return_value=mock_semantic_memory
        ):
            connector = MemoryConnector()
            connector.graph_traverse(
                entity_id="hcp_1", relationship="hcp_network", max_depth=2
            )

        mock_semantic_memory.get_hcp_influence_network.assert_called_once_with("hcp_1", max_depth=2)

    def test_graph_traverse_handles_error(self, mock_semantic_memory):
        """graph_traverse should return empty list on error."""
        mock_semantic_memory.traverse_causal_chain.side_effect = Exception("Graph error")

        with patch(
            "src.rag.memory_connector.get_semantic_memory", return_value=mock_semantic_memory
        ):
            connector = MemoryConnector()
            results = connector.graph_traverse(entity_id="ent_1", relationship="causal_path")

        assert results == []


class TestGraphTraverseKPI:
    """Tests for KPI-targeted graph traversal."""

    def test_graph_traverse_kpi_returns_paths(self, mock_semantic_memory):
        """graph_traverse_kpi should find paths for KPI."""
        mock_semantic_memory.find_causal_paths_for_kpi.return_value = [
            {"nodes": ["HCP engagement", "TRx"], "confidence": 0.9, "kpi_impact": 0.15}
        ]

        with patch(
            "src.rag.memory_connector.get_semantic_memory", return_value=mock_semantic_memory
        ):
            connector = MemoryConnector()
            results = connector.graph_traverse_kpi(kpi_name="TRx", min_confidence=0.5)

        assert len(results) == 1
        mock_semantic_memory.find_causal_paths_for_kpi.assert_called_once_with(
            "TRx", min_confidence=0.5
        )

    def test_graph_traverse_kpi_handles_error(self, mock_semantic_memory):
        """graph_traverse_kpi should return empty list on error."""
        mock_semantic_memory.find_causal_paths_for_kpi.side_effect = Exception("Graph error")

        with patch(
            "src.rag.memory_connector.get_semantic_memory", return_value=mock_semantic_memory
        ):
            connector = MemoryConnector()
            results = connector.graph_traverse_kpi(kpi_name="TRx")

        assert results == []


# ============================================================================
# SINGLETON TESTS
# ============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_memory_connector_returns_singleton(self):
        """get_memory_connector should return same instance."""
        reset_memory_connector()

        with patch("src.rag.memory_connector.get_supabase_client"):
            connector1 = get_memory_connector()
            connector2 = get_memory_connector()

        assert connector1 is connector2

    def test_reset_clears_singleton(self):
        """reset_memory_connector should clear singleton."""
        reset_memory_connector()

        with patch("src.rag.memory_connector.get_supabase_client"):
            connector1 = get_memory_connector()
            reset_memory_connector()
            connector2 = get_memory_connector()

        assert connector1 is not connector2


# ============================================================================
# RESULT CONVERSION TESTS
# ============================================================================


class TestResultConversion:
    """Tests for result conversion helpers."""

    def test_chains_to_results_converts_correctly(self):
        """_chains_to_results should convert chain dicts to RetrievalResult."""
        connector = MemoryConnector()
        chains = [
            {
                "path": ["A", "B", "C"],
                "confidence": 0.85,
                "start_entity_id": "ent_1",
                "path_length": 3,
            }
        ]

        results = connector._chains_to_results(chains)

        assert len(results) == 1
        assert "A → B → C" in results[0].content
        assert results[0].score == 0.85
        assert results[0].retrieval_method == "graph"

    def test_network_to_results_includes_center(self):
        """_network_to_results should include center node."""
        connector = MemoryConnector()
        network = {
            "center_node": {"id": "pat_123", "properties": {"name": "Test"}},
            "connections": [],
        }

        results = connector._network_to_results(network, "patient")

        assert len(results) == 1
        assert results[0].score == 1.0  # Center has highest score

    def test_network_to_results_scores_by_depth(self):
        """_network_to_results should score connections by depth."""
        connector = MemoryConnector()
        network = {
            "center_node": {"id": "pat_123"},
            "connections": [
                {"node_id": "conn_1", "depth": 1, "node_type": "HCP"},
                {"node_id": "conn_2", "depth": 2, "node_type": "Treatment"},
            ],
        }

        results = connector._network_to_results(network, "patient")

        # Scores should decrease with depth
        depth1_result = next(r for r in results if r.source_id == "conn_1")
        depth2_result = next(r for r in results if r.source_id == "conn_2")
        assert depth1_result.score > depth2_result.score

    def test_paths_to_results_converts_correctly(self):
        """_paths_to_results should convert path dicts to RetrievalResult."""
        connector = MemoryConnector()
        paths = [
            {"nodes": ["X", "Y", "Z"], "confidence": 0.75, "path_id": "path_1", "kpi_impact": 0.1}
        ]

        results = connector._paths_to_results(paths)

        assert len(results) == 1
        assert "X → Y → Z" in results[0].content
        assert results[0].score == 0.75
        assert results[0].metadata["kpi_impact"] == 0.1
