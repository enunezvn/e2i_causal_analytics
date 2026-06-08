"""
Unit tests for RAG Hybrid Retriever.

Tests the hybrid retrieval implementation combining dense, sparse, and graph methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.models.retrieval_models import RetrievalResult
from src.rag.retriever import (
    DENSE_WEIGHT,
    GRAPH_WEIGHT,
    SPARSE_WEIGHT,
    BM25Retriever,
    DenseRetriever,
    GraphRetriever,
    HybridRetriever,
    hybrid_search,
)
from src.rag.types import RetrievalSource

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
            source_id="mem_1",
            content="Dense result 1",
            source=RetrievalSource.VECTOR,
            score=0.9,
            retrieval_method="dense",
            metadata={"source_name": "episodic_memories"},
        ),
        RetrievalResult(
            source_id="mem_2",
            content="Dense result 2",
            source=RetrievalSource.VECTOR,
            score=0.8,
            retrieval_method="dense",
            metadata={"source_name": "episodic_memories"},
        ),
    ]


@pytest.fixture
def sample_sparse_results():
    """Sample results from sparse retrieval."""
    return [
        RetrievalResult(
            source_id="path_1",
            content="Sparse result 1",
            source=RetrievalSource.FULLTEXT,
            score=0.85,
            retrieval_method="sparse",
            metadata={"source_name": "causal_paths"},
        ),
    ]


@pytest.fixture
def sample_graph_results():
    """Sample results from graph retrieval."""
    return [
        RetrievalResult(
            source_id="graph_1",
            content="Graph result 1",
            source=RetrievalSource.GRAPH,
            score=0.75,
            retrieval_method="graph",
            metadata={"source_name": "semantic_graph"},
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
            await retriever.search(query="test query", k=20, filters={"brand": "Kisqali"})

        mock_memory_connector.vector_search_by_text.assert_called_once_with(
            query_text="test query",
            k=20,
            filters={"brand": "Kisqali"},
            min_similarity=0.5,
            max_staleness=None,
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
            await retriever.search(query="TRx drop", k=15, filters={"agent_name": "causal_impact"})

        mock_memory_connector.fulltext_search.assert_called_once_with(
            query_text="TRx drop",
            k=15,
            filters={"agent_name": "causal_impact"},
            max_staleness=None,
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
                entities=["ent_1"], relationship="causal_path", max_depth=3
            )

        assert len(results) == 1
        assert results[0].retrieval_method == "graph"

    def test_traverse_multiple_entities(self, mock_memory_connector, sample_graph_results):
        """traverse should handle multiple entities."""
        mock_memory_connector.graph_traverse.return_value = sample_graph_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = GraphRetriever()
            retriever.traverse(entities=["ent_1", "ent_2", "ent_3"], max_depth=2)

        # Called once per entity
        assert mock_memory_connector.graph_traverse.call_count == 3

    def test_traverse_deduplicates_results(self, mock_memory_connector):
        """traverse should deduplicate results by source_id."""
        duplicate_results = [
            RetrievalResult(
                source_id="same_id",
                content="Same result",
                source=RetrievalSource.GRAPH,
                score=0.8,
                retrieval_method="graph",
                metadata={},
            )
        ]
        mock_memory_connector.graph_traverse.return_value = duplicate_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = GraphRetriever()
            results = retriever.traverse(entities=["ent_1", "ent_2"])  # Two entities, same result

        # Should be deduplicated
        assert len(results) == 1

    def test_traverse_kpi_returns_results(self, mock_memory_connector, sample_graph_results):
        """traverse_kpi should return KPI-related paths."""
        mock_memory_connector.graph_traverse_kpi.return_value = sample_graph_results

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = GraphRetriever()
            results = retriever.traverse_kpi(kpi_name="TRx", min_confidence=0.6)

        assert len(results) == 1
        mock_memory_connector.graph_traverse_kpi.assert_called_once_with(
            kpi_name="TRx", min_confidence=0.6
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
                query="test", weights={"dense": 0.8, "sparse": 0.1, "graph": 0.1}, k=10
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
            await retriever.search(query="test", entities=["ent_1", "ent_2"], k=10)

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
            await retriever.search(query="Why did TRx drop?", kpi_name="TRx", k=10)

        mock_memory_connector.graph_traverse_kpi.assert_called_once_with(
            kpi_name="TRx", min_confidence=0.5
        )


# ============================================================================
# RRF FUSION TESTS
# ============================================================================


class TestReciprocalRankFusion:
    """Tests for RRF fusion algorithm."""

    def test_rrf_ranks_by_combined_score(self):
        """RRF should rank results by combined score."""
        retriever = HybridRetriever()

        list1 = [
            RetrievalResult(
                source_id="1",
                content="A",
                source=RetrievalSource.VECTOR,
                score=0.9,
                retrieval_method="dense",
                metadata={},
            ),
            RetrievalResult(
                source_id="2",
                content="B",
                source=RetrievalSource.VECTOR,
                score=0.8,
                retrieval_method="dense",
                metadata={},
            ),
        ]
        list2 = [
            RetrievalResult(
                source_id="2",
                content="B",
                source=RetrievalSource.FULLTEXT,
                score=0.95,
                retrieval_method="sparse",
                metadata={},
            ),
            RetrievalResult(
                source_id="3",
                content="C",
                source=RetrievalSource.FULLTEXT,
                score=0.85,
                retrieval_method="sparse",
                metadata={},
            ),
        ]

        fused = retriever._reciprocal_rank_fusion(result_lists=[list1, list2], weights=[0.5, 0.5])

        # B should be ranked higher (appears in both lists)
        assert fused[0].source_id == "2"

    def test_rrf_handles_empty_lists(self):
        """RRF should handle empty result lists."""
        retriever = HybridRetriever()

        fused = retriever._reciprocal_rank_fusion(
            result_lists=[[], [], []], weights=[0.5, 0.3, 0.2]
        )

        assert fused == []

    def test_rrf_includes_metadata(self):
        """RRF should include RRF score in metadata."""
        retriever = HybridRetriever()

        list1 = [
            RetrievalResult(
                source_id="1",
                content="A",
                source=RetrievalSource.VECTOR,
                score=0.9,
                retrieval_method="dense",
                metadata={"original": "data"},
            ),
        ]

        fused = retriever._reciprocal_rank_fusion(result_lists=[list1], weights=[1.0])

        assert "rrf_score" in fused[0].metadata
        assert "original_score" in fused[0].metadata
        assert fused[0].metadata["original"] == "data"

    def test_rrf_dedups_identical_content_under_distinct_source_ids(self):
        """Two rows with IDENTICAL content but DIFFERENT source_id must collapse
        to ONE fused row (not two) so duplicate content does not waste top-k slots
        or accrue double RRF mass. Faithful: exercises the real RRF, no mocks."""
        retriever = HybridRetriever()

        dup_content = "Causal analysis: hcp_engagement_level -> patient_conversion_rate, ATE=0.413"
        dense = [
            RetrievalResult(
                source_id="mem_aaaaaaaa",
                content=dup_content,
                source=RetrievalSource.VECTOR,
                score=0.811,
                retrieval_method="dense",
                metadata={},
            ),
            RetrievalResult(
                source_id="mem_bbbbbbbb",  # different id, SAME content
                content=dup_content,
                source=RetrievalSource.VECTOR,
                score=0.811,
                retrieval_method="dense",
                metadata={},
            ),
            RetrievalResult(
                source_id="mem_cccccccc",
                content="QC Report: passed. Score: 1.00.",
                source=RetrievalSource.VECTOR,
                score=0.726,
                retrieval_method="dense",
                metadata={},
            ),
        ]

        fused = retriever._reciprocal_rank_fusion(result_lists=[dense], weights=[1.0])

        contents = [r.content for r in fused]
        assert contents.count(dup_content) == 1, f"identical content survived twice: {contents}"
        # exactly the 2 distinct contents remain
        assert len(fused) == 2

    def test_rrf_surfaced_raw_score_matches_metadata_rrf_score(self):
        """Regression guard: the surfaced top-level score is the RAW RRF fusion sum
        (echoed identically in metadata['rrf_score']), NOT a 0-1 relevance, and the
        Pydantic model must accept that raw value (no le=1 rejection). Relevance is
        the DSPy relevance_score computed downstream, not this field."""
        retriever = HybridRetriever()

        only = [
            RetrievalResult(
                source_id="x1",
                content="solo row",
                source=RetrievalSource.VECTOR,
                score=0.9,
                retrieval_method="dense",
                metadata={},
            ),
        ]

        fused = retriever._reciprocal_rank_fusion(result_lists=[only], weights=[1.0])

        assert len(fused) == 1
        # raw RRF for a single rank-1 row at weight 1.0, k=60 -> 1/61 ~= 0.0164
        assert abs(fused[0].score - (1.0 / 61.0)) < 1e-9
        # top-level score == metadata rrf_score (single source of truth)
        assert fused[0].metadata["rrf_score"] == fused[0].score


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


class TestHybridSearchFunction:
    """Tests for hybrid_search convenience function."""

    @pytest.mark.asyncio
    async def test_hybrid_search_creates_retriever(
        self, mock_memory_connector, sample_dense_results
    ):
        """hybrid_search should create and use HybridRetriever."""
        mock_memory_connector.vector_search_by_text.return_value = sample_dense_results
        mock_memory_connector.fulltext_search.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            results = await hybrid_search(query="test query", k=5)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_passes_parameters(self, mock_memory_connector):
        """hybrid_search should pass all parameters correctly."""
        mock_memory_connector.vector_search_by_text.return_value = []
        mock_memory_connector.fulltext_search.return_value = []
        mock_memory_connector.graph_traverse_kpi.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            await hybrid_search(
                query="TRx analysis", k=15, kpi_name="TRx", filters={"brand": "Kisqali"}
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


# ============================================================================
# MAX_STALENESS PARAMETER TESTS (Phase 2 finishing, issue #373)
# ============================================================================


class TestHybridRetrieverMaxStaleness:
    """Tests for max_staleness parameter on HybridRetriever.search.

    Phase 2 finishing per .claude/plans/e2i_memory_subsystems_implementation_plan.md
    §Recommended-sequencing item 1. Under Decision 3 = KEEP BINARY adopted on
    2026-05-19, max_staleness < 1.0 excludes invalidated rows; >= 1.0 includes all.
    """

    @pytest.mark.asyncio
    async def test_search_default_max_staleness_is_none(self, mock_memory_connector):
        """search without max_staleness should explicitly forward max_staleness=None to connector."""
        mock_memory_connector.vector_search_by_text.return_value = []
        mock_memory_connector.fulltext_search.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = HybridRetriever()
            await retriever.search(query="test")

        call_kwargs_vec = mock_memory_connector.vector_search_by_text.call_args.kwargs
        assert "max_staleness" in call_kwargs_vec, (
            "max_staleness must be explicitly forwarded to vector_search_by_text"
        )
        assert call_kwargs_vec["max_staleness"] is None
        call_kwargs_ft = mock_memory_connector.fulltext_search.call_args.kwargs
        assert "max_staleness" in call_kwargs_ft, (
            "max_staleness must be explicitly forwarded to fulltext_search"
        )
        assert call_kwargs_ft["max_staleness"] is None

    @pytest.mark.asyncio
    async def test_search_max_staleness_zero_forwarded_to_dense_and_sparse(
        self, mock_memory_connector
    ):
        """search with max_staleness=0.0 should forward to dense and sparse connector calls."""
        mock_memory_connector.vector_search_by_text.return_value = []
        mock_memory_connector.fulltext_search.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = HybridRetriever()
            await retriever.search(query="test", max_staleness=0.0)

        call_kwargs_vec = mock_memory_connector.vector_search_by_text.call_args.kwargs
        assert call_kwargs_vec.get("max_staleness") == 0.0
        call_kwargs_ft = mock_memory_connector.fulltext_search.call_args.kwargs
        assert call_kwargs_ft.get("max_staleness") == 0.0

    @pytest.mark.asyncio
    async def test_hybrid_search_convenience_forwards_max_staleness(self, mock_memory_connector):
        """hybrid_search() convenience function should forward max_staleness to HybridRetriever."""
        mock_memory_connector.vector_search_by_text.return_value = []
        mock_memory_connector.fulltext_search.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            await hybrid_search(query="test", max_staleness=0.5)

        call_kwargs_vec = mock_memory_connector.vector_search_by_text.call_args.kwargs
        assert call_kwargs_vec.get("max_staleness") == 0.5

    @pytest.mark.asyncio
    async def test_search_max_staleness_one_forwarded(self, mock_memory_connector):
        """search with max_staleness=1.0 (no-op semantic) should still forward 1.0 to connectors."""
        mock_memory_connector.vector_search_by_text.return_value = []
        mock_memory_connector.fulltext_search.return_value = []

        with patch("src.rag.retriever.get_memory_connector", return_value=mock_memory_connector):
            retriever = HybridRetriever()
            await retriever.search(query="test", max_staleness=1.0)

        call_kwargs_vec = mock_memory_connector.vector_search_by_text.call_args.kwargs
        assert call_kwargs_vec.get("max_staleness") == 1.0
        call_kwargs_ft = mock_memory_connector.fulltext_search.call_args.kwargs
        assert call_kwargs_ft.get("max_staleness") == 1.0
