"""
Tests for HybridRetriever and component retrievers.

Tests the hybrid retrieval system combining dense, sparse, and graph retrieval.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.rag.retriever import (
    DenseRetriever,
    BM25Retriever,
    GraphRetriever,
    HybridRetriever,
    hybrid_search,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    GRAPH_WEIGHT,
)
from src.rag.models.retrieval_models import RetrievalResult
from src.rag.types import RetrievalSource


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_memory_connector():
    """Mock memory connector for testing."""
    connector = MagicMock()
    connector.vector_search_by_text = AsyncMock(return_value=[])
    connector.fulltext_search = AsyncMock(return_value=[])
    connector.graph_traverse = MagicMock(return_value=[])
    connector.graph_traverse_kpi = MagicMock(return_value=[])
    return connector


@pytest.fixture
def sample_dense_results():
    """Sample results from dense retrieval."""
    return [
        RetrievalResult(
            source_id="dense-1",
            content="Dense result 1 about Kisqali adoption",
            source=RetrievalSource.VECTOR,
            score=0.95,
            retrieval_method="dense",
            metadata={"brand": "Kisqali", "source_name": "episodic_memories"}
        ),
        RetrievalResult(
            source_id="dense-2",
            content="Dense result 2 about HCP engagement",
            source=RetrievalSource.VECTOR,
            score=0.85,
            retrieval_method="dense",
            metadata={"source_name": "procedural_memories"}
        ),
    ]


@pytest.fixture
def sample_sparse_results():
    """Sample results from sparse retrieval."""
    return [
        RetrievalResult(
            source_id="sparse-1",
            content="Sparse result 1 - causal path",
            source=RetrievalSource.FULLTEXT,
            score=0.9,
            retrieval_method="sparse",
            metadata={"source_name": "causal_paths"}
        ),
        RetrievalResult(
            source_id="sparse-2",
            content="Sparse result 2 - agent activity",
            source=RetrievalSource.FULLTEXT,
            score=0.7,
            retrieval_method="sparse",
            metadata={"source_name": "agent_activities"}
        ),
    ]


@pytest.fixture
def sample_graph_results():
    """Sample results from graph retrieval."""
    return [
        RetrievalResult(
            source_id="graph-1",
            content="Graph result: Sales → TRx",
            source=RetrievalSource.GRAPH,
            score=0.8,
            retrieval_method="graph",
            metadata={"path_length": 2, "source_name": "semantic_graph"}
        ),
    ]


# ============================================================================
# Default Weights Tests
# ============================================================================

class TestDefaultWeights:
    """Test default retrieval weights."""

    def test_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        total = DENSE_WEIGHT + SPARSE_WEIGHT + GRAPH_WEIGHT
        assert total == 1.0

    def test_dense_weight_value(self):
        """Test dense weight is highest."""
        assert DENSE_WEIGHT == 0.5
        assert DENSE_WEIGHT > SPARSE_WEIGHT
        assert DENSE_WEIGHT > GRAPH_WEIGHT


# ============================================================================
# DenseRetriever Tests
# ============================================================================

class TestDenseRetriever:
    """Test DenseRetriever class."""

    def test_initialization(self):
        """Test dense retriever initialization."""
        retriever = DenseRetriever()
        assert retriever.embedding_dim == 1536

    @pytest.mark.asyncio
    async def test_search_basic(self, mock_memory_connector, sample_dense_results):
        """Test basic dense search."""
        mock_memory_connector.vector_search_by_text.return_value = sample_dense_results

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = DenseRetriever()
            results = await retriever.search("Kisqali adoption trends", k=10)

            assert len(results) == 2
            assert all(r.retrieval_method == "dense" for r in results)
            mock_memory_connector.vector_search_by_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_memory_connector):
        """Test dense search with filters."""
        mock_memory_connector.vector_search_by_text.return_value = []

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = DenseRetriever()
            await retriever.search(
                "test query",
                k=5,
                filters={"brand": "Fabhalta", "region": "Northeast"}
            )

            call_kwargs = mock_memory_connector.vector_search_by_text.call_args.kwargs
            assert call_kwargs["filters"] == {"brand": "Fabhalta", "region": "Northeast"}
            assert call_kwargs["min_similarity"] == 0.5

    @pytest.mark.asyncio
    async def test_search_error_handling(self, mock_memory_connector):
        """Test dense search error handling."""
        mock_memory_connector.vector_search_by_text.side_effect = Exception("Embedding failed")

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = DenseRetriever()
            results = await retriever.search("test query")

            assert results == []


# ============================================================================
# BM25Retriever Tests
# ============================================================================

class TestBM25Retriever:
    """Test BM25Retriever class."""

    @pytest.mark.asyncio
    async def test_search_basic(self, mock_memory_connector, sample_sparse_results):
        """Test basic sparse search."""
        mock_memory_connector.fulltext_search.return_value = sample_sparse_results

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = BM25Retriever()
            results = await retriever.search("TRx trend analysis", k=10)

            assert len(results) == 2
            assert all(r.retrieval_method == "sparse" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_memory_connector):
        """Test sparse search with filters."""
        mock_memory_connector.fulltext_search.return_value = []

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = BM25Retriever()
            await retriever.search(
                "causal impact",
                k=5,
                filters={"brand": "Kisqali"}
            )

            call_kwargs = mock_memory_connector.fulltext_search.call_args.kwargs
            assert call_kwargs["filters"] == {"brand": "Kisqali"}

    @pytest.mark.asyncio
    async def test_search_error_handling(self, mock_memory_connector):
        """Test sparse search error handling."""
        mock_memory_connector.fulltext_search.side_effect = Exception("Search failed")

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = BM25Retriever()
            results = await retriever.search("test")

            assert results == []


# ============================================================================
# GraphRetriever Tests
# ============================================================================

class TestGraphRetriever:
    """Test GraphRetriever class."""

    def test_traverse_single_entity(self, mock_memory_connector, sample_graph_results):
        """Test graph traversal with single entity."""
        mock_memory_connector.graph_traverse.return_value = sample_graph_results

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = GraphRetriever()
            results = retriever.traverse(
                entities=["entity-1"],
                relationship="causal_path",
                max_depth=3
            )

            assert len(results) == 1
            assert results[0].retrieval_method == "graph"

    def test_traverse_multiple_entities(self, mock_memory_connector):
        """Test graph traversal with multiple entities."""
        # Return different results for different entities
        def mock_traverse(entity_id, relationship, max_depth):
            return [
                RetrievalResult(
                    source_id=f"g-{entity_id}",
                    content=f"Result for {entity_id}",
                    source=RetrievalSource.GRAPH,
                    score=0.8,
                    retrieval_method="graph",
                    metadata={"source_name": "semantic_graph"}
                )
            ]

        mock_memory_connector.graph_traverse.side_effect = mock_traverse

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = GraphRetriever()
            results = retriever.traverse(
                entities=["e1", "e2", "e3"],
                relationship="causal_path"
            )

            assert len(results) == 3

    def test_traverse_deduplication(self, mock_memory_connector):
        """Test that duplicate results are removed."""
        duplicate_result = RetrievalResult(
            source_id="same-id",
            content="Same result",
            source=RetrievalSource.GRAPH,
            score=0.9,
            retrieval_method="graph",
            metadata={"source_name": "semantic_graph"}
        )
        mock_memory_connector.graph_traverse.return_value = [duplicate_result]

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = GraphRetriever()
            results = retriever.traverse(entities=["e1", "e2"])

            # Should deduplicate by id
            assert len(results) == 1

    def test_traverse_error_handling(self, mock_memory_connector):
        """Test graph traversal error handling for one entity."""
        def mock_traverse(entity_id, relationship, max_depth):
            if entity_id == "bad":
                raise Exception("Traversal failed")
            return [
                RetrievalResult(
                    source_id=entity_id,
                    content=f"Result for {entity_id}",
                    source=RetrievalSource.GRAPH,
                    score=0.8,
                    retrieval_method="graph",
                    metadata={"source_name": "semantic_graph"}
                )
            ]

        mock_memory_connector.graph_traverse.side_effect = mock_traverse

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = GraphRetriever()
            # Should handle error for "bad" entity but continue with "good"
            results = retriever.traverse(entities=["bad", "good"])

            assert len(results) == 1
            assert "good" in results[0].source_id

    def test_traverse_kpi_basic(self, mock_memory_connector, sample_graph_results):
        """Test KPI graph traversal."""
        mock_memory_connector.graph_traverse_kpi.return_value = sample_graph_results

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = GraphRetriever()
            results = retriever.traverse_kpi("TRx", min_confidence=0.5)

            assert len(results) == 1
            mock_memory_connector.graph_traverse_kpi.assert_called_with(
                kpi_name="TRx",
                min_confidence=0.5
            )

    def test_traverse_kpi_error_handling(self, mock_memory_connector):
        """Test KPI traversal error handling."""
        mock_memory_connector.graph_traverse_kpi.side_effect = Exception("KPI error")

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = GraphRetriever()
            results = retriever.traverse_kpi("NRx")

            assert results == []


# ============================================================================
# HybridRetriever Tests
# ============================================================================

class TestHybridRetriever:
    """Test HybridRetriever class."""

    def test_initialization(self):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever()
        assert isinstance(retriever.dense, DenseRetriever)
        assert isinstance(retriever.sparse, BM25Retriever)
        assert isinstance(retriever.graph, GraphRetriever)

    @pytest.mark.asyncio
    async def test_search_combines_all_methods(
        self,
        mock_memory_connector,
        sample_dense_results,
        sample_sparse_results,
        sample_graph_results
    ):
        """Test that search combines all retrieval methods."""
        mock_memory_connector.vector_search_by_text.return_value = sample_dense_results
        mock_memory_connector.fulltext_search.return_value = sample_sparse_results
        mock_memory_connector.graph_traverse.return_value = sample_graph_results

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = HybridRetriever()
            results = await retriever.search(
                query="Kisqali TRx analysis",
                k=10,
                entities=["entity-1"]
            )

            # Should have results from all methods (fused)
            assert len(results) <= 10
            # Dense and sparse should have been called
            mock_memory_connector.vector_search_by_text.assert_called_once()
            mock_memory_connector.fulltext_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_custom_weights(self, mock_memory_connector):
        """Test search with custom weights."""
        mock_memory_connector.vector_search_by_text.return_value = []
        mock_memory_connector.fulltext_search.return_value = []

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = HybridRetriever()
            custom_weights = {"dense": 0.7, "sparse": 0.2, "graph": 0.1}
            results = await retriever.search(
                query="test",
                weights=custom_weights,
                k=5
            )

            # Should use custom weights (tested implicitly through execution)
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_kpi_name(self, mock_memory_connector, sample_graph_results):
        """Test search with KPI name for graph traversal."""
        mock_memory_connector.vector_search_by_text.return_value = []
        mock_memory_connector.fulltext_search.return_value = []
        mock_memory_connector.graph_traverse_kpi.return_value = sample_graph_results

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = HybridRetriever()
            results = await retriever.search(
                query="TRx trends",
                k=10,
                kpi_name="TRx"
            )

            mock_memory_connector.graph_traverse_kpi.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_concurrent_execution(self, mock_memory_connector):
        """Test that dense and sparse searches run concurrently."""
        import asyncio

        # Track call times to verify concurrent execution
        call_order = []

        async def mock_vector_search(*args, **kwargs):
            call_order.append("dense_start")
            await asyncio.sleep(0.01)
            call_order.append("dense_end")
            return []

        async def mock_fulltext(*args, **kwargs):
            call_order.append("sparse_start")
            await asyncio.sleep(0.01)
            call_order.append("sparse_end")
            return []

        mock_memory_connector.vector_search_by_text = mock_vector_search
        mock_memory_connector.fulltext_search = mock_fulltext

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = HybridRetriever()
            await retriever.search("test", k=5)

            # Both should start before either ends (concurrent)
            assert "dense_start" in call_order[:2]
            assert "sparse_start" in call_order[:2]


class TestReciprocalRankFusion:
    """Test RRF algorithm."""

    def test_rrf_basic(self):
        """Test basic RRF calculation."""
        retriever = HybridRetriever()

        results1 = [
            RetrievalResult(source_id="1", content="A", source=RetrievalSource.VECTOR, score=0.9, retrieval_method="dense", metadata={}),
            RetrievalResult(source_id="2", content="B", source=RetrievalSource.VECTOR, score=0.8, retrieval_method="dense", metadata={}),
        ]
        results2 = [
            RetrievalResult(source_id="2", content="B", source=RetrievalSource.FULLTEXT, score=0.85, retrieval_method="sparse", metadata={}),
            RetrievalResult(source_id="3", content="C", source=RetrievalSource.FULLTEXT, score=0.7, retrieval_method="sparse", metadata={}),
        ]

        fused = retriever._reciprocal_rank_fusion(
            result_lists=[results1, results2],
            weights=[0.6, 0.4]
        )

        # "B" should be ranked highest (appears in both lists)
        assert fused[0].source_id == "2"  # B

    def test_rrf_empty_lists(self):
        """Test RRF with empty result lists."""
        retriever = HybridRetriever()

        fused = retriever._reciprocal_rank_fusion(
            result_lists=[[], [], []],
            weights=[0.5, 0.3, 0.2]
        )

        assert fused == []

    def test_rrf_single_list(self):
        """Test RRF with single result list."""
        retriever = HybridRetriever()

        results = [
            RetrievalResult(source_id="1", content="A", source=RetrievalSource.VECTOR, score=0.9, retrieval_method="dense", metadata={}),
        ]

        fused = retriever._reciprocal_rank_fusion(
            result_lists=[results],
            weights=[1.0]
        )

        assert len(fused) == 1
        assert fused[0].source_id == "1"

    def test_rrf_score_in_metadata(self):
        """Test that RRF score is added to metadata."""
        retriever = HybridRetriever()

        results = [
            RetrievalResult(source_id="1", content="A", source=RetrievalSource.VECTOR, score=0.9, retrieval_method="dense", metadata={}),
        ]

        fused = retriever._reciprocal_rank_fusion(
            result_lists=[results],
            weights=[1.0]
        )

        assert "rrf_score" in fused[0].metadata
        assert "original_score" in fused[0].metadata
        assert fused[0].metadata["original_score"] == 0.9

    def test_rrf_preserves_retrieval_method(self):
        """Test that retrieval method is preserved in metadata."""
        retriever = HybridRetriever()

        results = [
            RetrievalResult(source_id="1", content="A", source=RetrievalSource.GRAPH, score=0.9, retrieval_method="graph", metadata={}),
        ]

        fused = retriever._reciprocal_rank_fusion(
            result_lists=[results],
            weights=[1.0]
        )

        assert fused[0].retrieval_method == "graph"


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestHybridSearchFunction:
    """Test hybrid_search convenience function."""

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self, mock_memory_connector):
        """Test hybrid_search function."""
        mock_memory_connector.vector_search_by_text.return_value = []
        mock_memory_connector.fulltext_search.return_value = []

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            results = await hybrid_search(
                query="test query",
                k=10
            )

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_with_all_params(self, mock_memory_connector):
        """Test hybrid_search with all parameters."""
        mock_memory_connector.vector_search_by_text.return_value = []
        mock_memory_connector.fulltext_search.return_value = []
        mock_memory_connector.graph_traverse_kpi.return_value = []

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            results = await hybrid_search(
                query="TRx analysis for Kisqali",
                k=5,
                entities=None,
                kpi_name="TRx",
                filters={"brand": "Kisqali"}
            )

            assert isinstance(results, list)


# ============================================================================
# Integration-style Tests
# ============================================================================

class TestRetrieverIntegration:
    """Integration-style tests for retriever components."""

    @pytest.mark.asyncio
    async def test_end_to_end_retrieval(
        self,
        mock_memory_connector,
        sample_dense_results,
        sample_sparse_results,
        sample_graph_results
    ):
        """Test complete retrieval flow."""
        mock_memory_connector.vector_search_by_text.return_value = sample_dense_results
        mock_memory_connector.fulltext_search.return_value = sample_sparse_results
        mock_memory_connector.graph_traverse.return_value = sample_graph_results

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = HybridRetriever()
            results = await retriever.search(
                query="Why did Kisqali adoption increase?",
                k=5,
                entities=["kisqali-region-northeast"],
                filters={"brand": "Kisqali"}
            )

            # Should return fused results
            assert len(results) <= 5
            # Results should have RRF scores in metadata
            for result in results:
                assert "rrf_score" in result.metadata or result.score > 0

    @pytest.mark.asyncio
    async def test_retrieval_ranking(self, mock_memory_connector):
        """Test that results are properly ranked by RRF."""
        # Create results where one item appears in multiple lists
        high_overlap_result = RetrievalResult(
            source_id="overlap-1",
            content="High value result",
            source=RetrievalSource.VECTOR,
            score=0.9,
            retrieval_method="dense",
            metadata={}
        )

        dense_results = [
            high_overlap_result,
            RetrievalResult(source_id="d1", content="Dense only", source=RetrievalSource.VECTOR, score=0.8, retrieval_method="dense", metadata={})
        ]
        sparse_results = [
            RetrievalResult(
                source_id="overlap-1",
                content="High value result",
                source=RetrievalSource.FULLTEXT,
                score=0.85,
                retrieval_method="sparse",
                metadata={}
            ),
        ]

        mock_memory_connector.vector_search_by_text.return_value = dense_results
        mock_memory_connector.fulltext_search.return_value = sparse_results

        with patch(
            "src.rag.retriever.get_memory_connector",
            return_value=mock_memory_connector
        ):
            retriever = HybridRetriever()
            results = await retriever.search(query="test", k=10)

            # The overlapping result should be ranked first
            assert results[0].source_id == "overlap-1"
