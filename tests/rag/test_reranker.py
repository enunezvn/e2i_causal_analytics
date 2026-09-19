"""
Unit tests for CrossEncoderReranker.

Tests cover:
- Basic reranking functionality
- Batch processing
- Model caching
- Edge cases (empty input, single result)
- Score normalization
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.rag.models.retrieval_models import RetrievalResult
from src.rag.reranker import _MODEL_CACHE, CrossEncoderReranker
from src.rag.types import RetrievalResult as LiveRetrievalResult
from src.rag.types import RetrievalSource


@pytest.fixture
def sample_results():
    """Create sample retrieval results for testing."""
    return [
        RetrievalResult(
            source_id="act_001",
            content="Kisqali shows strong TRx growth in Q4 2024",
            source=RetrievalSource.VECTOR,
            score=0.8,
            retrieval_method="dense",
            metadata={"brand": "Kisqali", "source_name": "agent_activities"},
        ),
        RetrievalResult(
            source_id="met_002",
            content="Market share analysis for breast cancer treatments",
            source=RetrievalSource.FULLTEXT,
            score=0.6,
            retrieval_method="sparse",
            metadata={"category": "oncology", "source_name": "business_metrics"},
        ),
        RetrievalResult(
            source_id="trg_003",
            content="HCP targeting recommendations for Fabhalta launch",
            source=RetrievalSource.GRAPH,
            score=0.7,
            retrieval_method="graph",
            metadata={"brand": "Fabhalta", "source_name": "triggers"},
        ),
    ]


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder to avoid loading actual model in tests."""
    with patch("src.rag.reranker.CrossEncoder") as mock:
        mock_model = MagicMock()
        mock.return_value = mock_model
        yield mock_model


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker class."""

    def setup_method(self):
        """Clear model cache before each test."""
        _MODEL_CACHE.clear()

    def teardown_method(self):
        """Do not leak mocked model instances into other test modules."""
        _MODEL_CACHE.clear()

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.batch_size == 32
        assert reranker.max_length == 512

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-TinyBERT-L-2",
            batch_size=16,
            max_length=256,
        )
        assert reranker.model_name == "cross-encoder/ms-marco-TinyBERT-L-2"
        assert reranker.batch_size == 16
        assert reranker.max_length == 256

    def test_rerank_empty_results(self, mock_cross_encoder):
        """Test reranking with empty input returns empty list."""
        reranker = CrossEncoderReranker()
        result = reranker.rerank([], "test query")
        assert result == []
        # Model should not be called for empty input
        mock_cross_encoder.predict.assert_not_called()

    def test_rerank_preserves_live_hybrid_result_type_and_fields(self, mock_cross_encoder):
        """The live API uses src.rag.types.RetrievalResult, not the legacy model."""
        mock_cross_encoder.predict.return_value = np.array([0.88])
        original = LiveRetrievalResult(
            id="doc-1",
            content="Kisqali TRx increased in Q4",
            source=RetrievalSource.VECTOR,
            score=0.016,
            metadata={"brand": "Kisqali"},
            graph_context={"nodes": ["brand:kisqali"]},
            query_latency_ms=12.5,
            raw_score=0.91,
        )

        reranked = CrossEncoderReranker().rerank([original], "Kisqali TRx", top_k=1)

        assert len(reranked) == 1
        assert isinstance(reranked[0], LiveRetrievalResult)
        assert reranked[0].id == "doc-1"
        assert reranked[0].graph_context == original.graph_context
        assert reranked[0].query_latency_ms == 12.5
        assert reranked[0].raw_score == 0.91
        assert reranked[0].metadata["original_score"] == 0.016
        assert reranked[0].metadata["reranker_score"] == reranked[0].score

    def test_rerank_basic(self, sample_results, mock_cross_encoder):
        """Test basic reranking functionality."""
        # Mock model to return descending scores (reverse of input order)
        mock_cross_encoder.predict.return_value = np.array([0.3, 0.9, 0.6])

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(sample_results, "TRx growth analysis", top_k=3)

        assert len(reranked) == 3
        # Results should be sorted by score descending
        # Original order: [0.3, 0.9, 0.6] -> Sorted: [0.9, 0.6, 0.3]
        assert reranked[0].source_id == "met_002"  # Score 0.9
        assert reranked[1].source_id == "trg_003"  # Score 0.6
        assert reranked[2].source_id == "act_001"  # Score 0.3

    def test_rerank_top_k(self, sample_results, mock_cross_encoder):
        """Test that top_k limits output correctly."""
        mock_cross_encoder.predict.return_value = np.array([0.8, 0.6, 0.4])

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(sample_results, "query", top_k=2)

        assert len(reranked) == 2

    def test_rerank_preserves_metadata(self, sample_results, mock_cross_encoder):
        """Test that original metadata is preserved with additions."""
        mock_cross_encoder.predict.return_value = np.array([0.8, 0.6, 0.4])

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(sample_results, "query", top_k=1)

        assert "brand" in reranked[0].metadata
        assert reranked[0].metadata["brand"] == "Kisqali"
        assert "reranker_score" in reranked[0].metadata
        assert "original_score" in reranked[0].metadata
        assert reranked[0].metadata["original_score"] == 0.8

    def test_rerank_score_normalization(self, sample_results, mock_cross_encoder):
        """Trust the single activation requested from CrossEncoder.predict."""
        mock_cross_encoder.predict.return_value = np.array([0.88, 0.5, 0.12])

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(sample_results, "query", top_k=3)

        # All scores should be in [0, 1]
        for result in reranked:
            assert 0.0 <= result.score <= 1.0

        # Values must not be compressed through a second sigmoid.
        assert reranked[0].score == pytest.approx(0.88)
        assert reranked[1].score == pytest.approx(0.5)
        assert reranked[2].score == pytest.approx(0.12)
        assert (
            type(mock_cross_encoder.predict.call_args.kwargs["activation_fn"]).__name__ == "Sigmoid"
        )

    @pytest.mark.parametrize("invalid_score", [np.nan, np.inf, -np.inf])
    def test_strict_reranker_rejects_non_finite_scores(
        self, sample_results, mock_cross_encoder, invalid_score
    ):
        mock_cross_encoder.predict.return_value = np.array([invalid_score, 0.5, 0.25])

        with pytest.raises(ValueError, match="non-finite"):
            CrossEncoderReranker(raise_on_error=True).rerank(sample_results, "query")

    def test_rerank_with_string_query(self, sample_results, mock_cross_encoder):
        """Test reranking with plain string query."""
        mock_cross_encoder.predict.return_value = np.array([0.5, 0.5, 0.5])

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(sample_results, "plain string query")

        assert len(reranked) == 3

    def test_rerank_with_parsed_query_object(self, sample_results, mock_cross_encoder):
        """Test reranking with ParsedQuery-like object."""
        mock_cross_encoder.predict.return_value = np.array([0.5, 0.5, 0.5])

        query = Mock()
        query.text = "parsed query text"

        reranker = CrossEncoderReranker()
        reranker.rerank(sample_results, query)

        # Verify the query text was extracted correctly
        call_args = mock_cross_encoder.predict.call_args[0][0]
        assert call_args[0][0] == "parsed query text"

    def test_batch_score_builds_correct_pairs(self, sample_results, mock_cross_encoder):
        """Test that batch scoring builds correct query-document pairs."""
        mock_cross_encoder.predict.return_value = np.array([0.5, 0.5, 0.5])

        reranker = CrossEncoderReranker()
        reranker.rerank(sample_results, "test query")

        # Check predict was called with correct pairs
        call_args = mock_cross_encoder.predict.call_args
        pairs = call_args[0][0]

        assert len(pairs) == 3
        assert all(pair[0] == "test query" for pair in pairs)
        assert pairs[0][1] == "Kisqali shows strong TRx growth in Q4 2024"

    def test_batch_score_uses_correct_batch_size(self, sample_results, mock_cross_encoder):
        """Test that batch_size is passed to model predict."""
        mock_cross_encoder.predict.return_value = np.array([0.5, 0.5, 0.5])

        reranker = CrossEncoderReranker(batch_size=16)
        reranker.rerank(sample_results, "query")

        call_kwargs = mock_cross_encoder.predict.call_args[1]
        assert call_kwargs["batch_size"] == 16

    def test_model_caching(self, mock_cross_encoder):
        """Test that model is cached and reused."""
        with patch("src.rag.reranker.CrossEncoder") as mock_constructor:
            mock_constructor.return_value = mock_cross_encoder
            mock_cross_encoder.predict.return_value = np.array([0.5])

            result1 = RetrievalResult(
                source_id="1",
                content="test",
                source=RetrievalSource.VECTOR,
                score=0.5,
                retrieval_method="dense",
                metadata={},
            )

            reranker1 = CrossEncoderReranker()
            reranker1.rerank([result1], "query1")

            reranker2 = CrossEncoderReranker()
            reranker2.rerank([result1], "query2")

            # Model should only be constructed once
            assert mock_constructor.call_count == 1

    def test_concurrent_cold_load_constructs_model_once(self):
        model = MagicMock()
        with patch("src.rag.reranker.CrossEncoder", return_value=model) as constructor:
            reranker = CrossEncoderReranker()

            with ThreadPoolExecutor(max_workers=2) as pool:
                loaded = list(pool.map(lambda _: reranker.model, range(2)))

        assert loaded == [model, model]
        constructor.assert_called_once()

    def test_error_handling_returns_fallback_scores(self, sample_results):
        """Test that errors in scoring return fallback scores."""
        with patch("src.rag.reranker.CrossEncoder") as mock_constructor:
            mock_model = MagicMock()
            mock_model.predict.side_effect = RuntimeError("Model error")
            mock_constructor.return_value = mock_model

            reranker = CrossEncoderReranker()
            reranked = reranker.rerank(sample_results, "query")

            # Should return results with fallback score of 0.5
            assert len(reranked) == 3
            for result in reranked:
                assert result.score == 0.5

    def test_strict_error_handling_preserves_orchestrator_fallback(self, sample_results):
        """The live service must be able to detect inference failure."""
        with patch("src.rag.reranker.CrossEncoder") as mock_constructor:
            mock_model = MagicMock()
            mock_model.predict.side_effect = RuntimeError("Model error")
            mock_constructor.return_value = mock_model

            reranker = CrossEncoderReranker(raise_on_error=True)

            with pytest.raises(RuntimeError, match="Model error"):
                reranker.rerank(sample_results, "query")

    def test_single_result(self, mock_cross_encoder):
        """Test reranking with single result."""
        mock_cross_encoder.predict.return_value = np.array([0.9])

        result = RetrievalResult(
            source_id="1",
            content="Single result",
            source=RetrievalSource.VECTOR,
            score=0.5,
            retrieval_method="dense",
            metadata={},
        )

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank([result], "query", top_k=1)

        assert len(reranked) == 1
        assert reranked[0].source_id == "1"

    def test_score_pair_method(self, mock_cross_encoder):
        """Test _score_pair convenience method."""
        mock_cross_encoder.predict.return_value = np.array([0.7])

        reranker = CrossEncoderReranker()
        score = reranker._score_pair("query", "document")

        assert score == pytest.approx(0.7)


class TestRerankerPerformance:
    """Performance-related tests for reranker."""

    def setup_method(self):
        """Clear model cache before each test."""
        _MODEL_CACHE.clear()

    def teardown_method(self):
        """Do not leak mocked model instances into other test modules."""
        _MODEL_CACHE.clear()

    def test_batch_score_empty_pairs(self, mock_cross_encoder):
        """Test batch scoring with empty pairs returns empty list."""
        reranker = CrossEncoderReranker()
        scores = reranker._batch_score([])
        assert scores == []

    def test_rerank_preserves_retrieval_method(self, sample_results, mock_cross_encoder):
        """Test that retrieval_method is preserved after reranking."""
        mock_cross_encoder.predict.return_value = np.array([0.9, 0.5, 0.3])

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(sample_results, "query")

        methods = [r.retrieval_method for r in reranked]
        assert "dense" in methods
        assert "sparse" in methods
        assert "graph" in methods
