"""
Unit tests for OpenAI Embedding Client.

Tests cover:
- Single and batch embedding generation
- Retry logic for rate limits
- Usage statistics tracking
- Error handling
- Async interface

Part of Phase 1, Checkpoint 1.2 validation.
"""

import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ensure_real_openai():
    """Restore real openai module if polluted by other test modules (e.g. test_evaluation.py)."""
    openai_mod = sys.modules.get("openai")
    if openai_mod is not None and isinstance(openai_mod, MagicMock):
        # Remove mock and all submodules
        del sys.modules["openai"]
        for key in list(sys.modules):
            if key.startswith("openai."):
                del sys.modules[key]
    # Import the real openai
    import openai

    if isinstance(openai, MagicMock) or not hasattr(openai, "RateLimitError"):
        importlib.reload(openai)
    # Reload embeddings so it picks up real openai (only if already loaded)
    if "src.rag.embeddings" in sys.modules:
        importlib.reload(sys.modules["src.rag.embeddings"])
    return openai


# Ensure real openai at import time
_ensure_real_openai()

from src.rag.config import EmbeddingConfig
from src.rag.embeddings import (
    EmbeddingUsageStats,
    OpenAIEmbeddingClient,
    get_embedding,
)
from src.rag.exceptions import ConfigurationError, EmbeddingError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding_response():
    """Create a mock OpenAI embedding response."""

    def create_response(texts, dimension=1536):
        """Factory for creating mock responses."""
        mock_data = []
        for i, _text in enumerate(texts if isinstance(texts, list) else [texts]):
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1 * (i + 1)] * dimension
            mock_data.append(mock_embedding)

        mock_usage = MagicMock()
        mock_usage.total_tokens = len(texts) * 10 if isinstance(texts, list) else 10

        response = MagicMock()
        response.data = mock_data
        response.usage = mock_usage
        return response

    return create_response


@pytest.fixture
def embedding_config():
    """Create test embedding configuration."""
    return EmbeddingConfig(
        model_name="text-embedding-3-small",
        model_provider="openai",
        api_key="test-api-key",
        embedding_dimension=1536,
        batch_size=10,
    )


@pytest.fixture
def embedding_client(embedding_config):
    """Create test embedding client with mocked OpenAI clients."""
    with (
        patch("src.rag.embeddings.OpenAI") as mock_sync,
        patch("src.rag.embeddings.AsyncOpenAI") as mock_async,
    ):

        client = OpenAIEmbeddingClient(embedding_config)
        client._mock_sync = mock_sync
        client._mock_async = mock_async
        yield client


# =============================================================================
# EmbeddingUsageStats Tests
# =============================================================================


class TestEmbeddingUsageStats:
    """Tests for usage statistics tracking."""

    def test_initial_stats(self):
        """Test initial statistics values."""
        stats = EmbeddingUsageStats()

        assert stats.total_tokens == 0
        assert stats.total_requests == 0
        assert stats.total_texts == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.retry_count == 0
        assert stats.total_latency_ms == 0.0

    def test_record_success(self):
        """Test recording successful requests."""
        stats = EmbeddingUsageStats()

        stats.record_success(tokens=100, texts=5, latency_ms=50.0)

        assert stats.total_tokens == 100
        assert stats.total_texts == 5
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.failed_requests == 0
        assert stats.total_latency_ms == 50.0

    def test_record_multiple_successes(self):
        """Test recording multiple successful requests."""
        stats = EmbeddingUsageStats()

        stats.record_success(tokens=100, texts=5, latency_ms=50.0)
        stats.record_success(tokens=200, texts=10, latency_ms=100.0)

        assert stats.total_tokens == 300
        assert stats.total_texts == 15
        assert stats.total_requests == 2
        assert stats.successful_requests == 2
        assert stats.total_latency_ms == 150.0

    def test_record_failure(self):
        """Test recording failed requests."""
        stats = EmbeddingUsageStats()

        stats.record_failure()

        assert stats.total_requests == 1
        assert stats.failed_requests == 1
        assert stats.successful_requests == 0

    def test_record_retry(self):
        """Test recording retry attempts."""
        stats = EmbeddingUsageStats()

        stats.record_retry()
        stats.record_retry()

        assert stats.retry_count == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = EmbeddingUsageStats()
        stats.record_success(tokens=100, texts=5, latency_ms=50.0)
        stats.record_success(tokens=100, texts=5, latency_ms=50.0)
        stats.record_failure()
        stats.record_retry()

        result = stats.to_dict()

        assert result["total_tokens"] == 200
        assert result["total_requests"] == 3
        assert result["total_texts"] == 10
        assert result["successful_requests"] == 2
        assert result["failed_requests"] == 1
        assert result["retry_count"] == 1
        assert result["avg_latency_ms"] == 50.0
        assert result["success_rate"] == pytest.approx(66.67, rel=0.01)

    def test_avg_latency_no_requests(self):
        """Test average latency with no successful requests."""
        stats = EmbeddingUsageStats()

        result = stats.to_dict()

        assert result["avg_latency_ms"] == 0.0


# =============================================================================
# OpenAIEmbeddingClient Initialization Tests
# =============================================================================


class TestOpenAIEmbeddingClientInit:
    """Tests for client initialization."""

    def test_init_with_config(self, embedding_config):
        """Test initialization with config object."""
        with patch("src.rag.embeddings.OpenAI"), patch("src.rag.embeddings.AsyncOpenAI"):

            client = OpenAIEmbeddingClient(embedding_config)

            assert client.model == "text-embedding-3-small"
            assert client.dimension == 1536
            assert client._batch_size == 10

    def test_init_with_explicit_params(self):
        """Test initialization with explicit parameters override config."""
        config = EmbeddingConfig(model_name="text-embedding-3-small", api_key="config-key")

        with patch("src.rag.embeddings.OpenAI"), patch("src.rag.embeddings.AsyncOpenAI"):

            client = OpenAIEmbeddingClient(
                config=config, api_key="override-key", model="text-embedding-3-large"
            )

            assert client._api_key == "override-key"
            assert client._model == "text-embedding-3-large"

    def test_init_without_api_key_raises(self):
        """Test initialization without API key raises error."""
        # Patch os.getenv to return None for OPENAI_API_KEY
        with patch("src.rag.config.os.getenv", return_value=None):
            config = EmbeddingConfig(model_name="text-embedding-3-small", api_key=None)

            with pytest.raises(ConfigurationError) as exc_info:
                OpenAIEmbeddingClient(config)

            assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_repr(self, embedding_client):
        """Test string representation."""
        repr_str = repr(embedding_client)

        assert "OpenAIEmbeddingClient" in repr_str
        assert "text-embedding-3-small" in repr_str
        assert "1536" in repr_str


# =============================================================================
# Synchronous Encoding Tests
# =============================================================================


class TestSyncEncoding:
    """Tests for synchronous embedding generation."""

    def test_encode_single_text(self, embedding_config, mock_embedding_response):
        """Test encoding a single text."""
        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
        ):

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_embedding_response(["test"])
            mock_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            result = client.encode("Hello world")

            assert isinstance(result, list)
            assert len(result) == 1536
            assert all(isinstance(x, float) for x in result)

    def test_encode_multiple_texts(self, embedding_config, mock_embedding_response):
        """Test encoding multiple texts."""
        texts = ["First text", "Second text", "Third text"]

        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
        ):

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_embedding_response(texts)
            mock_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            result = client.encode(texts)

            assert isinstance(result, list)
            assert len(result) == 3
            assert all(len(emb) == 1536 for emb in result)

    def test_encode_batch_splits_correctly(self, embedding_config, mock_embedding_response):
        """Test batch encoding splits into correct batches."""
        texts = [f"Text {i}" for i in range(25)]  # 3 batches with batch_size=10

        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
        ):

            mock_client = MagicMock()
            # Return different responses for each batch
            mock_client.embeddings.create.side_effect = [
                mock_embedding_response(texts[:10]),
                mock_embedding_response(texts[10:20]),
                mock_embedding_response(texts[20:]),
            ]
            mock_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            result = client.encode_batch(texts)

            assert len(result) == 25
            assert mock_client.embeddings.create.call_count == 3

    def test_encode_with_custom_dimensions(self, embedding_config, mock_embedding_response):
        """Test encoding with custom dimensions."""
        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
        ):

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_embedding_response(
                ["test"], dimension=512
            )
            mock_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            client.encode("Hello", dimensions=512)

            # Verify dimensions parameter was passed
            call_kwargs = mock_client.embeddings.create.call_args[1]
            assert call_kwargs.get("dimensions") == 512


# =============================================================================
# Asynchronous Encoding Tests
# =============================================================================


class TestAsyncEncoding:
    """Tests for asynchronous embedding generation."""

    @pytest.mark.asyncio
    async def test_encode_async_single_text(self, embedding_config, mock_embedding_response):
        """Test async encoding a single text."""
        with (
            patch("src.rag.embeddings.OpenAI"),
            patch("src.rag.embeddings.AsyncOpenAI") as mock_async_openai,
        ):

            mock_client = AsyncMock()
            mock_client.embeddings.create.return_value = mock_embedding_response(["test"])
            mock_async_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            result = await client.encode_async("Hello world")

            assert isinstance(result, list)
            assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_encode_async_multiple_texts(self, embedding_config, mock_embedding_response):
        """Test async encoding multiple texts."""
        texts = ["First", "Second", "Third"]

        with (
            patch("src.rag.embeddings.OpenAI"),
            patch("src.rag.embeddings.AsyncOpenAI") as mock_async_openai,
        ):

            mock_client = AsyncMock()
            mock_client.embeddings.create.return_value = mock_embedding_response(texts)
            mock_async_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            result = await client.encode_async(texts)

            assert len(result) == 3

    @pytest.mark.asyncio
    async def test_encode_batch_async_concurrent(self, embedding_config, mock_embedding_response):
        """Test async batch encoding with concurrency."""
        texts = [f"Text {i}" for i in range(25)]

        with (
            patch("src.rag.embeddings.OpenAI"),
            patch("src.rag.embeddings.AsyncOpenAI") as mock_async_openai,
        ):

            mock_client = AsyncMock()
            # Side effect to return correct number of embeddings per batch
            # With batch_size=10: batches of 10, 10, 5
            mock_client.embeddings.create.side_effect = [
                mock_embedding_response(["test"] * 10),  # First batch of 10
                mock_embedding_response(["test"] * 10),  # Second batch of 10
                mock_embedding_response(["test"] * 5),  # Third batch of 5
            ]
            mock_async_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            result = await client.encode_batch_async(texts, concurrency=2)

            assert len(result) == 25


# =============================================================================
# Retry Logic Tests
# =============================================================================


class TestRetryLogic:
    """Tests for retry logic on rate limits."""

    @pytest.fixture(autouse=True)
    def _restore_openai(self):
        """Ensure real openai module is loaded (guards against test_evaluation pollution)."""
        real_openai = _ensure_real_openai()
        # Re-import OpenAIEmbeddingClient after reload
        from src.rag.embeddings import OpenAIEmbeddingClient as _Client  # noqa: F811

        self._Client = _Client
        self._openai = real_openai
        yield

    def test_retry_on_rate_limit(self, embedding_config, mock_embedding_response):
        """Test retry on rate limit error."""
        openai = self._openai
        Client = self._Client

        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
            patch("time.sleep"),
        ):  # Skip actual sleep

            mock_client = MagicMock()
            # First call raises rate limit, second succeeds
            mock_client.embeddings.create.side_effect = [
                openai.RateLimitError(
                    message="Rate limit exceeded", response=MagicMock(status_code=429), body={}
                ),
                mock_embedding_response(["test"]),
            ]
            mock_openai.return_value = mock_client

            client = Client(embedding_config)
            result = client.encode("Hello")

            assert len(result) == 1536
            assert client._usage_stats.retry_count == 1

    def test_max_retries_exceeded(self, embedding_config):
        """Test error raised when max retries exceeded."""
        openai = self._openai
        Client = self._Client

        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
            patch("time.sleep"),
        ):

            mock_client = MagicMock()
            # All calls raise rate limit
            mock_client.embeddings.create.side_effect = openai.RateLimitError(
                message="Rate limit exceeded", response=MagicMock(status_code=429), body={}
            )
            mock_openai.return_value = mock_client

            client = Client(embedding_config)

            with pytest.raises(EmbeddingError) as exc_info:
                client.encode("Hello")

            assert "retries" in str(exc_info.value).lower()
            assert client._usage_stats.retry_count == 5  # MAX_RETRIES

    def test_api_error_not_retried(self, embedding_config):
        """Test API errors (non-rate-limit) are not retried."""
        openai = self._openai
        Client = self._Client

        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
        ):

            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = openai.APIError(
                message="API error", request=MagicMock(), body={}
            )
            mock_openai.return_value = mock_client

            client = Client(embedding_config)

            with pytest.raises(EmbeddingError):
                client.encode("Hello")

            assert mock_client.embeddings.create.call_count == 1  # No retry


# =============================================================================
# Usage Statistics Tests
# =============================================================================


class TestUsageStatistics:
    """Tests for usage tracking."""

    def test_stats_after_successful_call(self, embedding_config, mock_embedding_response):
        """Test usage stats updated after successful call."""
        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
        ):

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_embedding_response(["test"])
            mock_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            client.encode("Hello")

            stats = client.get_usage_stats()

            assert stats["total_tokens"] == 10
            assert stats["total_texts"] == 1
            assert stats["successful_requests"] == 1
            assert stats["failed_requests"] == 0
            assert stats["success_rate"] == 100.0

    def test_reset_usage_stats(self, embedding_config, mock_embedding_response):
        """Test resetting usage statistics."""
        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
        ):

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_embedding_response(["test"])
            mock_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            client.encode("Hello")

            client.reset_usage_stats()

            stats = client.get_usage_stats()
            assert stats["total_tokens"] == 0
            assert stats["total_requests"] == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Tests for get_embedding convenience function."""

    @pytest.mark.asyncio
    async def test_get_embedding_basic(self, mock_embedding_response):
        """Test basic usage of get_embedding function."""
        with (
            patch("src.rag.embeddings.OpenAI"),
            patch("src.rag.embeddings.AsyncOpenAI") as mock_async_openai,
        ):

            mock_client = AsyncMock()
            mock_client.embeddings.create.return_value = mock_embedding_response(["test"])
            mock_async_openai.return_value = mock_client

            result = await get_embedding(
                "Hello world", model="text-embedding-3-small", api_key="test-key"
            )

            assert isinstance(result, list)
            assert len(result) == 1536


# =============================================================================
# Embedding Dimension Tests
# =============================================================================


class TestEmbeddingDimensions:
    """Tests for embedding dimension handling."""

    def test_default_dimension(self, embedding_client):
        """Test default embedding dimension."""
        assert embedding_client.dimension == 1536

    def test_model_property(self, embedding_client):
        """Test model property."""
        assert embedding_client.model == "text-embedding-3-small"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text_list(self, embedding_config, mock_embedding_response):
        """Test encoding empty list."""
        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
        ):

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_embedding_response([])
            mock_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            result = client.encode([])

            assert result == []

    def test_batch_with_single_item(self, embedding_config, mock_embedding_response):
        """Test batch encoding with single item."""
        with (
            patch("src.rag.embeddings.OpenAI") as mock_openai,
            patch("src.rag.embeddings.AsyncOpenAI"),
        ):

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_embedding_response(["single"])
            mock_openai.return_value = mock_client

            client = OpenAIEmbeddingClient(embedding_config)
            result = client.encode_batch(["single text"])

            assert len(result) == 1
            assert len(result[0]) == 1536
