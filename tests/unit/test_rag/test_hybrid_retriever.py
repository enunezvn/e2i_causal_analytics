"""
Unit tests for E2I Hybrid RAG Retriever.

Tests for HybridRetriever orchestration, RRF fusion, and graph boost.
All external dependencies are mocked.
"""

from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.config import FalkorDBConfig, HybridSearchConfig, RAGConfig
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.types import (
    BackendStatus,
    ExtractedEntities,
    RetrievalResult,
    RetrievalSource,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = []
    client.rpc.return_value.execute.return_value = mock_response
    return client


@pytest.fixture
def mock_falkordb_client():
    """Create a mock FalkorDB client."""
    client = MagicMock()
    mock_graph = MagicMock()
    mock_graph.query.return_value = MagicMock(result_set=[])
    client.select_graph.return_value = mock_graph
    return client


@pytest.fixture
def rag_config():
    """Create a RAG configuration for tests."""
    return RAGConfig(
        search=HybridSearchConfig(
            # Weight distribution (individual fields, fusion_weights is computed property)
            vector_weight=0.4,
            fulltext_weight=0.2,
            graph_weight=0.4,
            # Per-source limits
            vector_top_k=10,
            fulltext_top_k=10,
            graph_top_k=10,
            final_top_k=5,
            # Timeouts
            vector_timeout_ms=5000,
            fulltext_timeout_ms=3000,
            graph_timeout_ms=5000,
            # Thresholds
            vector_min_similarity=0.5,
            fulltext_min_rank=0.1,
            graph_min_relevance=0.2,
        ),
        falkordb=FalkorDBConfig(
            host="localhost",
            port=6381,
            graph_name="test_graph",
        ),
    )


@pytest.fixture
def hybrid_retriever(mock_supabase_client, mock_falkordb_client, rag_config):
    """Create a HybridRetriever with mocked backends."""
    return HybridRetriever(
        supabase_client=mock_supabase_client,
        falkordb_client=mock_falkordb_client,
        config=rag_config,
    )


def create_mock_result(
    id: str, content: str, source: RetrievalSource, score: float, graph_context: Dict = None
) -> RetrievalResult:
    """Helper to create mock RetrievalResult."""
    return RetrievalResult(
        id=id,
        content=content,
        source=source,
        score=score,
        metadata={},
        graph_context=graph_context,
    )


# ============================================================================
# HybridRetriever Initialization Tests
# ============================================================================


class TestHybridRetrieverInit:
    """Tests for HybridRetriever initialization."""

    def test_init_creates_all_backends(
        self, mock_supabase_client, mock_falkordb_client, rag_config
    ):
        """Test that all backends are initialized."""
        retriever = HybridRetriever(
            supabase_client=mock_supabase_client,
            falkordb_client=mock_falkordb_client,
            config=rag_config,
        )

        assert retriever.vector_backend is not None
        assert retriever.fulltext_backend is not None
        assert retriever.graph_backend is not None

    def test_init_with_defaults(self, mock_supabase_client, mock_falkordb_client):
        """Test initialization with default config."""
        retriever = HybridRetriever(
            supabase_client=mock_supabase_client,
            falkordb_client=mock_falkordb_client,
        )

        assert retriever.config is not None

    def test_repr(self, hybrid_retriever):
        """Test string representation."""
        repr_str = repr(hybrid_retriever)
        assert "HybridRetriever" in repr_str


# ============================================================================
# RRF Fusion Tests
# ============================================================================


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion algorithm."""

    def test_rrf_single_source(self, hybrid_retriever):
        """Test RRF with results from single source."""
        backend_results = {
            RetrievalSource.VECTOR: [
                create_mock_result("doc-1", "Content 1", RetrievalSource.VECTOR, 0.9),
                create_mock_result("doc-2", "Content 2", RetrievalSource.VECTOR, 0.8),
            ],
            RetrievalSource.FULLTEXT: [],
            RetrievalSource.GRAPH: [],
        }

        fused = hybrid_retriever._apply_rrf_fusion(backend_results, top_k=5)

        assert len(fused) == 2
        # First result should have higher RRF score
        assert fused[0].id == "doc-1"
        assert fused[0].score > fused[1].score

    def test_rrf_multiple_sources(self, hybrid_retriever):
        """Test RRF combines results from multiple sources."""
        backend_results = {
            RetrievalSource.VECTOR: [
                create_mock_result("doc-1", "Content 1", RetrievalSource.VECTOR, 0.9),
                create_mock_result("doc-2", "Content 2", RetrievalSource.VECTOR, 0.8),
            ],
            RetrievalSource.FULLTEXT: [
                create_mock_result("doc-2", "Content 2", RetrievalSource.FULLTEXT, 0.7),
                create_mock_result("doc-3", "Content 3", RetrievalSource.FULLTEXT, 0.6),
            ],
            RetrievalSource.GRAPH: [
                create_mock_result("doc-1", "Content 1", RetrievalSource.GRAPH, 0.85),
            ],
        }

        fused = hybrid_retriever._apply_rrf_fusion(backend_results, top_k=5)

        # doc-1 appears in vector and graph, should rank high
        # doc-2 appears in vector and fulltext
        assert len(fused) == 3

        # Get result IDs in order
        result_ids = [r.id for r in fused]

        # doc-1 should be first (appears in 2 sources with high ranks)
        assert result_ids[0] == "doc-1"

        # All results should have rrf_sources in metadata
        for result in fused:
            assert "rrf_sources" in result.metadata

    def test_rrf_limits_to_top_k(self, hybrid_retriever):
        """Test RRF respects top_k limit."""
        backend_results = {
            RetrievalSource.VECTOR: [
                create_mock_result(
                    f"doc-{i}", f"Content {i}", RetrievalSource.VECTOR, 0.9 - i * 0.1
                )
                for i in range(10)
            ],
            RetrievalSource.FULLTEXT: [],
            RetrievalSource.GRAPH: [],
        }

        fused = hybrid_retriever._apply_rrf_fusion(backend_results, top_k=3)

        assert len(fused) == 3


# ============================================================================
# Graph Boost Tests
# ============================================================================


class TestGraphBoost:
    """Tests for graph boost application."""

    def test_graph_boost_applied_to_graph_context(self, hybrid_retriever):
        """Test that graph boost is applied to results with graph context."""
        results = [
            create_mock_result(
                "doc-1",
                "With graph",
                RetrievalSource.VECTOR,
                0.5,
                graph_context={"connected_nodes": ["node-1"]},
            ),
            create_mock_result("doc-2", "No graph", RetrievalSource.VECTOR, 0.6),
        ]

        boosted = hybrid_retriever._apply_graph_boost(results)

        # doc-1 should be boosted (0.5 * 1.3 = 0.65)
        doc1 = next(r for r in boosted if r.id == "doc-1")
        assert doc1.score == pytest.approx(0.5 * 1.3)
        assert doc1.metadata.get("graph_boosted") is True

        # doc-2 should not be boosted
        doc2 = next(r for r in boosted if r.id == "doc-2")
        assert doc2.score == 0.6
        assert doc2.metadata.get("graph_boosted") is None

    def test_graph_boost_reorders_results(self, hybrid_retriever):
        """Test that graph boost can change result ordering."""
        results = [
            create_mock_result("doc-1", "Lower base", RetrievalSource.VECTOR, 0.5),
            create_mock_result(
                "doc-2",
                "Higher base but boostable",
                RetrievalSource.VECTOR,
                0.55,
                graph_context={"path_length": 2},
            ),
        ]

        boosted = hybrid_retriever._apply_graph_boost(results)
        sorted_results = sorted(boosted, key=lambda x: x.score, reverse=True)

        # doc-2 with graph context (0.55 * 1.3 = 0.715) should now beat doc-1 (0.5)
        assert sorted_results[0].id == "doc-2"


# ============================================================================
# Search Integration Tests
# ============================================================================


class TestHybridRetrieverSearch:
    """Tests for HybridRetriever search orchestration."""

    @pytest.mark.asyncio
    async def test_search_dispatches_to_all_backends(self, hybrid_retriever):
        """Test that search dispatches to all backends in parallel."""
        with (
            patch.object(
                hybrid_retriever, "_safe_vector_search", new_callable=AsyncMock
            ) as mock_vector,
            patch.object(
                hybrid_retriever, "_safe_fulltext_search", new_callable=AsyncMock
            ) as mock_fulltext,
            patch.object(
                hybrid_retriever, "_safe_graph_search", new_callable=AsyncMock
            ) as mock_graph,
        ):
            mock_vector.return_value = []
            mock_fulltext.return_value = []
            mock_graph.return_value = []

            await hybrid_retriever.search(
                query="test query",
                embedding=[0.1] * 1536,
                entities=ExtractedEntities(brands=["Test"]),
            )

            mock_vector.assert_called_once()
            mock_fulltext.assert_called_once()
            mock_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_skips_vector_without_embedding(self, hybrid_retriever):
        """Test that vector search is skipped when no embedding provided."""
        with (
            patch.object(
                hybrid_retriever, "_safe_vector_search", new_callable=AsyncMock
            ) as mock_vector,
            patch.object(
                hybrid_retriever, "_safe_fulltext_search", new_callable=AsyncMock
            ) as mock_fulltext,
            patch.object(
                hybrid_retriever, "_safe_graph_search", new_callable=AsyncMock
            ) as mock_graph,
        ):
            mock_fulltext.return_value = []
            mock_graph.return_value = []

            await hybrid_retriever.search(
                query="test query",
                embedding=None,  # No embedding
            )

            mock_vector.assert_not_called()
            mock_fulltext.assert_called_once()
            mock_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_returns_fused_results(self, hybrid_retriever):
        """Test that search returns properly fused and boosted results."""
        with (
            patch.object(
                hybrid_retriever, "_safe_vector_search", new_callable=AsyncMock
            ) as mock_vector,
            patch.object(
                hybrid_retriever, "_safe_fulltext_search", new_callable=AsyncMock
            ) as mock_fulltext,
            patch.object(
                hybrid_retriever, "_safe_graph_search", new_callable=AsyncMock
            ) as mock_graph,
        ):
            mock_vector.return_value = [
                create_mock_result("doc-1", "Vector result", RetrievalSource.VECTOR, 0.9)
            ]
            mock_fulltext.return_value = [
                create_mock_result("doc-2", "Fulltext result", RetrievalSource.FULLTEXT, 0.8)
            ]
            mock_graph.return_value = [
                create_mock_result(
                    "doc-1",
                    "Graph result",
                    RetrievalSource.GRAPH,
                    0.85,
                    graph_context={"path_length": 1},
                )
            ]

            results = await hybrid_retriever.search(
                query="test",
                embedding=[0.1] * 1536,
            )

            assert len(results) > 0
            # doc-1 appears in both vector and graph, should be ranked high
            assert results[0].id == "doc-1"
            # Should have graph boost applied
            assert results[0].metadata.get("graph_boosted") is True

    @pytest.mark.asyncio
    async def test_search_handles_backend_failures_gracefully(self, hybrid_retriever):
        """Test that search continues when a backend fails."""
        with (
            patch.object(
                hybrid_retriever, "_safe_vector_search", new_callable=AsyncMock
            ) as mock_vector,
            patch.object(
                hybrid_retriever, "_safe_fulltext_search", new_callable=AsyncMock
            ) as mock_fulltext,
            patch.object(
                hybrid_retriever, "_safe_graph_search", new_callable=AsyncMock
            ) as mock_graph,
        ):
            mock_vector.return_value = []  # Returns empty on error
            mock_fulltext.return_value = [
                create_mock_result("doc-1", "Fulltext result", RetrievalSource.FULLTEXT, 0.8)
            ]
            mock_graph.return_value = []  # Returns empty on error

            results = await hybrid_retriever.search(
                query="test",
                embedding=[0.1] * 1536,
            )

            # Should still get fulltext result
            assert len(results) == 1
            assert results[0].id == "doc-1"

    @pytest.mark.asyncio
    async def test_search_tracks_stats(self, hybrid_retriever):
        """Test that search tracks statistics."""
        with (
            patch.object(
                hybrid_retriever, "_safe_vector_search", new_callable=AsyncMock
            ) as mock_vector,
            patch.object(
                hybrid_retriever, "_safe_fulltext_search", new_callable=AsyncMock
            ) as mock_fulltext,
            patch.object(
                hybrid_retriever, "_safe_graph_search", new_callable=AsyncMock
            ) as mock_graph,
        ):
            mock_vector.return_value = [
                create_mock_result("doc-1", "V", RetrievalSource.VECTOR, 0.9),
                create_mock_result("doc-2", "V", RetrievalSource.VECTOR, 0.8),
            ]
            mock_fulltext.return_value = [
                create_mock_result("doc-3", "F", RetrievalSource.FULLTEXT, 0.7),
            ]
            mock_graph.return_value = []

            await hybrid_retriever.search(
                query="test",
                embedding=[0.1] * 1536,
            )

            stats = hybrid_retriever.last_search_stats
            assert stats is not None
            assert stats.vector_count == 2
            assert stats.fulltext_count == 1
            assert stats.graph_count == 0
            assert stats.total_latency_ms >= 0


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHybridRetrieverHealthCheck:
    """Tests for HybridRetriever health check."""

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, hybrid_retriever):
        """Test health check when all backends are healthy."""
        with (
            patch.object(
                hybrid_retriever.vector_backend, "health_check", new_callable=AsyncMock
            ) as mock_vector_health,
            patch.object(
                hybrid_retriever.fulltext_backend, "health_check", new_callable=AsyncMock
            ) as mock_fulltext_health,
            patch.object(
                hybrid_retriever.graph_backend, "health_check", new_callable=AsyncMock
            ) as mock_graph_health,
        ):
            mock_vector_health.return_value = {"status": "healthy", "latency_ms": 10}
            mock_fulltext_health.return_value = {"status": "healthy", "latency_ms": 5}
            mock_graph_health.return_value = {"status": "healthy", "latency_ms": 15}

            health = await hybrid_retriever.health_check()

            assert "vector" in health
            assert "fulltext" in health
            assert "graph" in health
            assert all(h.status == BackendStatus.HEALTHY for h in health.values())

    @pytest.mark.asyncio
    async def test_health_check_partial_failure(self, hybrid_retriever):
        """Test health check when some backends are unhealthy."""
        with (
            patch.object(
                hybrid_retriever.vector_backend, "health_check", new_callable=AsyncMock
            ) as mock_vector_health,
            patch.object(
                hybrid_retriever.fulltext_backend, "health_check", new_callable=AsyncMock
            ) as mock_fulltext_health,
            patch.object(
                hybrid_retriever.graph_backend, "health_check", new_callable=AsyncMock
            ) as mock_graph_health,
        ):
            mock_vector_health.return_value = {"status": "healthy", "latency_ms": 10}
            mock_fulltext_health.return_value = {
                "status": "unhealthy",
                "latency_ms": 0,
                "error": "DB down",
            }
            mock_graph_health.return_value = {"status": "healthy", "latency_ms": 15}

            health = await hybrid_retriever.health_check()

            assert health["vector"].status == BackendStatus.HEALTHY
            assert health["fulltext"].status == BackendStatus.UNHEALTHY
            assert health["graph"].status == BackendStatus.HEALTHY


# ============================================================================
# Graph Visualization Tests
# ============================================================================


class TestGraphVisualization:
    """Tests for graph visualization methods."""

    @pytest.mark.asyncio
    async def test_get_causal_subgraph_delegates_to_backend(self, hybrid_retriever):
        """Test that get_causal_subgraph delegates to graph backend."""
        expected_result = {
            "nodes": [{"id": "1", "label": "Brand"}],
            "edges": [{"source": "1", "target": "2", "type": "AFFECTS"}],
            "metadata": {"total_nodes": 1},
        }

        with patch.object(
            hybrid_retriever.graph_backend, "get_causal_subgraph", new_callable=AsyncMock
        ) as mock_subgraph:
            mock_subgraph.return_value = expected_result

            result = await hybrid_retriever.get_causal_subgraph(center_node_id="123", max_depth=2)

            mock_subgraph.assert_called_once_with(
                center_node_id="123",
                node_types=None,
                relationship_types=None,
                max_depth=2,
                limit=100,
            )
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_get_causal_path_delegates_to_backend(self, hybrid_retriever):
        """Test that get_causal_path delegates to graph backend."""
        expected_paths = [{"source_node": "1", "target_node": "2", "path_length": 2}]

        with patch.object(
            hybrid_retriever.graph_backend, "get_causal_path", new_callable=AsyncMock
        ) as mock_path:
            mock_path.return_value = expected_paths

            result = await hybrid_retriever.get_causal_path(
                source_id="1", target_id="2", max_length=3
            )

            mock_path.assert_called_once_with(source_id="1", target_id="2", max_length=3)
            assert result == expected_paths
