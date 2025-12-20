"""
Unit tests for RAG Hybrid Retriever.

Tests the hybrid retrieval implementation combining dense, sparse, and graph methods.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

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


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_memory_connector():
    """Create a mock memory connector."""
    connector = MagicMock()
    connector.vector_search_by_text = AsyncMock(return_value=[])
    connector.fulltext_search = AsyncMock(return_value=[])
    connector.graph_traverse.return_value = []
    connector.graph_traverse_kpi.return_value = []
    return connector


@pytest.fixture
def sample_dense_results():
    """Sample results from dense retrieval."""
    return [
        RetrievalResult(
            content="Dense result 1",
            source="episodic_memories",
            source_id="mem_1",
            score=0.9,
            retrieval_method="dense",
            metadata={}
        ),
        RetrievalResult(
            content="Dense result 2",
            source="episodic_memories",
            source_id="mem_2",
            score=0.8,
            retrieval_method="dense",
            metadata={}
        ),
    ]


@pytest.fixture
def sample_sparse_results():
    """Sample results from sparse retrieval."""
    return [
        RetrievalResult(
            content="Sparse result 1",
            source="causal_paths",
            source_id="path_1",
            score=0.85,
            retrieval_method="sparse",
            metadata={}
        ),
    ]


@pytest.fixture
def sample_graph_results():
    """Sample results from graph retrieval."""
    return [
        RetrievalResult(
            content="Graph result 1",
            source="semantic_graph",
            source_id="graph_1",
            score=0.75,
            retrieval_method="graph",
            metadata={}
        ),
    ]


# ============================================================================
# DENSE RETRIEVER TESTS
# ============================================================================

class TestDenseRetriever:
    """Tests for DenseRetriever."""

    def test_init_sets_embedding_dim(self):
        """DenseRetriever should initialize with correct embedding dimension."""
        retriever = DenseRetriever()
        assert retriever.embedding_dim == 1536

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_memory_connector, sample_dense_results):
        """search should return results from memory connector."""
        mock_memory_connector.vector_search_by_text.return_value = sample_dense_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = DenseRetriever()
            results = await retriever.search("test query", k=10)

        assert len(results) == 2
        assert results[0].retrieval_method == "dense"
        mock_memory_connector.vector_search_by_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_passes_parameters(self, mock_memory_connector):
        """search should pass correct parameters to memory connector."""
        mock_memory_connector.vector_search_by_text.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = DenseRetriever()
            await retriever.search(
                query="test query",
                k=20,
                filters={"brand": "Kisqali"}
            )

        mock_memory_connector.vector_search_by_text.assert_called_once_with(
            query_text="test query",
            k=20,
            filters={"brand": "Kisqali"},
            min_similarity=0.5
        )

    @pytest.mark.asyncio
    async def test_search_handles_error(self, mock_memory_connector):
        """search should return empty list on error."""
        mock_memory_connector.vector_search_by_text.side_effect = Exception("Error")

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = DenseRetriever()
            results = await retriever.search("test query")

        assert results == []


# ============================================================================
# BM25 RETRIEVER TESTS
# ============================================================================

class TestBM25Retriever:
    """Tests for BM25Retriever."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_memory_connector, sample_sparse_results):
        """search should return results from memory connector."""
        mock_memory_connector.fulltext_search.return_value = sample_sparse_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = BM25Retriever()
            results = await retriever.search("causal path TRx", k=10)

        assert len(results) == 1
        assert results[0].retrieval_method == "sparse"

    @pytest.mark.asyncio
    async def test_search_passes_parameters(self, mock_memory_connector):
        """search should pass correct parameters to memory connector."""
        mock_memory_connector.fulltext_search.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = BM25Retriever()
            await retriever.search(
                query="TRx drop",
                k=15,
                filters={"agent_name": "causal_impact"}
            )

        mock_memory_connector.fulltext_search.assert_called_once_with(
            query_text="TRx drop",
            k=15,
            filters={"agent_name": "causal_impact"}
        )

    @pytest.mark.asyncio
    async def test_search_handles_error(self, mock_memory_connector):
        """search should return empty list on error."""
        mock_memory_connector.fulltext_search.side_effect = Exception("Error")

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = BM25Retriever()
            results = await retriever.search("test query")

        assert results == []


# ============================================================================
# GRAPH RETRIEVER TESTS
# ============================================================================

class TestGraphRetriever:
    """Tests for GraphRetriever."""

    def test_traverse_returns_results(self, mock_memory_connector, sample_graph_results):
        """traverse should return results from memory connector."""
        mock_memory_connector.graph_traverse.return_value = sample_graph_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = GraphRetriever()
            results = retriever.traverse(
                entities=["ent_1"],
                relationship="causal_path",
                max_depth=3
            )

        assert len(results) == 1
        assert results[0].retrieval_method == "graph"

    def test_traverse_multiple_entities(self, mock_memory_connector, sample_graph_results):
        """traverse should handle multiple entities."""
        mock_memory_connector.graph_traverse.return_value = sample_graph_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = GraphRetriever()
            results = retriever.traverse(
                entities=["ent_1", "ent_2", "ent_3"],
                max_depth=2
            )

        # Called once per entity
        assert mock_memory_connector.graph_traverse.call_count == 3

    def test_traverse_deduplicates_results(self, mock_memory_connector):
        """traverse should deduplicate results by source_id."""
        duplicate_results = [
            RetrievalResult(
                content="Same result",
                source="graph",
                source_id="same_id",
                score=0.8,
                retrieval_method="graph",
                metadata={}
            )
        ]
        mock_memory_connector.graph_traverse.return_value = duplicate_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = GraphRetriever()
            results = retriever.traverse(
                entities=["ent_1", "ent_2"]  # Two entities, same result
            )

        # Should be deduplicated
        assert len(results) == 1

    def test_traverse_kpi_returns_results(self, mock_memory_connector, sample_graph_results):
        """traverse_kpi should return KPI-related paths."""
        mock_memory_connector.graph_traverse_kpi.return_value = sample_graph_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = GraphRetriever()
            results = retriever.traverse_kpi(
                kpi_name="TRx",
                min_confidence=0.6
            )

        assert len(results) == 1
        mock_memory_connector.graph_traverse_kpi.assert_called_once_with(
            kpi_name="TRx",
            min_confidence=0.6
        )


# ============================================================================
# HYBRID RETRIEVER TESTS
# ============================================================================

class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_init_creates_retrievers(self):
        """HybridRetriever should initialize all sub-retrievers."""
        retriever = HybridRetriever()
        assert isinstance(retriever.dense, DenseRetriever)
        assert isinstance(retriever.sparse, BM25Retriever)
        assert isinstance(retriever.graph, GraphRetriever)

    @pytest.mark.asyncio
    async def test_search_combines_results(
        self, mock_memory_connector, sample_dense_results, sample_sparse_results
    ):
        """search should combine results from all retrievers."""
        mock_memory_connector.vector_search_by_text.return_value = sample_dense_results
        mock_memory_connector.fulltext_search.return_value = sample_sparse_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = HybridRetriever()
            results = await retriever.search("test query", k=10)

        # Should have fused results from both dense and sparse
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_uses_custom_weights(self, mock_memory_connector, sample_dense_results):
        """search should use custom weights when provided."""
        mock_memory_connector.vector_search_by_text.return_value = sample_dense_results
        mock_memory_connector.fulltext_search.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = HybridRetriever()
            results = await retriever.search(
                query="test",
                weights={"dense": 0.8, "sparse": 0.1, "graph": 0.1},
                k=10
            )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_with_entities(
        self, mock_memory_connector, sample_dense_results, sample_graph_results
    ):
        """search should use graph traversal when entities provided."""
        mock_memory_connector.vector_search_by_text.return_value = sample_dense_results
        mock_memory_connector.fulltext_search.return_value = []
        mock_memory_connector.graph_traverse.return_value = sample_graph_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = HybridRetriever()
            results = await retriever.search(
                query="test",
                entities=["ent_1", "ent_2"],
                k=10
            )

        mock_memory_connector.graph_traverse.assert_called()

    @pytest.mark.asyncio
    async def test_search_with_kpi_name(
        self, mock_memory_connector, sample_dense_results, sample_graph_results
    ):
        """search should use KPI traversal when kpi_name provided."""
        mock_memory_connector.vector_search_by_text.return_value = sample_dense_results
        mock_memory_connector.fulltext_search.return_value = []
        mock_memory_connector.graph_traverse_kpi.return_value = sample_graph_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = HybridRetriever()
            results = await retriever.search(
                query="Why did TRx drop?",
                kpi_name="TRx",
                k=10
            )

        mock_memory_connector.graph_traverse_kpi.assert_called_once_with(kpi_name="TRx", min_confidence=0.5)


# ============================================================================
# RRF FUSION TESTS
# ============================================================================

class TestReciprocalRankFusion:
    """Tests for RRF fusion algorithm."""

    def test_rrf_ranks_by_combined_score(self):
        """RRF should rank results by combined score."""
        retriever = HybridRetriever()

        list1 = [
            RetrievalResult(content="A", source="s", source_id="1", score=0.9, retrieval_method="dense", metadata={}),
            RetrievalResult(content="B", source="s", source_id="2", score=0.8, retrieval_method="dense", metadata={}),
        ]
        list2 = [
            RetrievalResult(content="B", source="s", source_id="2", score=0.95, retrieval_method="sparse", metadata={}),
            RetrievalResult(content="C", source="s", source_id="3", score=0.85, retrieval_method="sparse", metadata={}),
        ]

        fused = retriever._reciprocal_rank_fusion(
            result_lists=[list1, list2],
            weights=[0.5, 0.5]
        )

        # B should be ranked higher (appears in both lists)
        assert fused[0].source_id == "2"

    def test_rrf_handles_empty_lists(self):
        """RRF should handle empty result lists."""
        retriever = HybridRetriever()

        fused = retriever._reciprocal_rank_fusion(
            result_lists=[[], [], []],
            weights=[0.5, 0.3, 0.2]
        )

        assert fused == []

    def test_rrf_includes_metadata(self):
        """RRF should include RRF score in metadata."""
        retriever = HybridRetriever()

        list1 = [
            RetrievalResult(content="A", source="s", source_id="1", score=0.9, retrieval_method="dense", metadata={"original": "data"}),
        ]

        fused = retriever._reciprocal_rank_fusion(
            result_lists=[list1],
            weights=[1.0]
        )

        assert "rrf_score" in fused[0].metadata
        assert "original_score" in fused[0].metadata
        assert fused[0].metadata["original"] == "data"


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestHybridSearchFunction:
    """Tests for hybrid_search convenience function."""

    @pytest.mark.asyncio
    async def test_hybrid_search_creates_retriever(self, mock_memory_connector, sample_dense_results):
        """hybrid_search should create and use HybridRetriever."""
        mock_memory_connector.vector_search_by_text.return_value = sample_dense_results
        mock_memory_connector.fulltext_search.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            results = await hybrid_search(
                query="test query",
                k=5
            )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_passes_parameters(self, mock_memory_connector):
        """hybrid_search should pass all parameters correctly."""
        mock_memory_connector.vector_search_by_text.return_value = []
        mock_memory_connector.fulltext_search.return_value = []
        mock_memory_connector.graph_traverse_kpi.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            await hybrid_search(
                query="TRx analysis",
                k=15,
                kpi_name="TRx",
                filters={"brand": "Kisqali"}
            )

        # Verify filters were passed
        mock_memory_connector.vector_search_by_text.assert_called_once()
        call_kwargs = mock_memory_connector.vector_search_by_text.call_args[1]
        assert call_kwargs["filters"] == {"brand": "Kisqali"}


# ============================================================================
# WEIGHT CONSTANT TESTS
# ============================================================================

class TestWeightConstants:
    """Tests for retrieval weight constants."""

    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        total = DENSE_WEIGHT + SPARSE_WEIGHT + GRAPH_WEIGHT
        assert total == pytest.approx(1.0)

    def test_dense_has_highest_weight(self):
        """Dense retrieval should have highest default weight."""
        assert DENSE_WEIGHT > SPARSE_WEIGHT
        assert DENSE_WEIGHT > GRAPH_WEIGHT

    def test_all_weights_positive(self):
        """All weights should be positive."""
        assert DENSE_WEIGHT > 0
        assert SPARSE_WEIGHT > 0
        assert GRAPH_WEIGHT > 0
