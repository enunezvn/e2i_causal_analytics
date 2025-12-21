"""
E2I Hybrid RAG - OpenAI Embedding Client

This module provides the embedding client for generating vector embeddings
using OpenAI's text-embedding models. Features include:
- Single and batch embedding generation
- Exponential backoff retry logic for rate limits
- Token usage tracking
- Async support

Part of Phase 1, Checkpoint 1.2.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import openai
from openai import AsyncOpenAI, OpenAI

from src.rag.config import EmbeddingConfig
from src.rag.exceptions import ConfigurationError, EmbeddingError

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingUsageStats:
    """Track token usage and API call statistics."""

    total_tokens: int = 0
    total_requests: int = 0
    total_texts: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retry_count: int = 0
    total_latency_ms: float = 0.0

    def record_success(self, tokens: int, texts: int, latency_ms: float) -> None:
        """Record a successful embedding request."""
        self.total_tokens += tokens
        self.total_texts += texts
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms

    def record_failure(self) -> None:
        """Record a failed embedding request."""
        self.total_requests += 1
        self.failed_requests += 1

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retry_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        avg_latency = (
            self.total_latency_ms / self.successful_requests
            if self.successful_requests > 0
            else 0.0
        )
        return {
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "total_texts": self.total_texts,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "retry_count": self.retry_count,
            "avg_latency_ms": round(avg_latency, 2),
            "success_rate": round(self.successful_requests / max(self.total_requests, 1) * 100, 2),
        }


class OpenAIEmbeddingClient:
    """
    OpenAI Embedding Client for generating vector embeddings.

    Features:
    - Supports text-embedding-3-small and text-embedding-3-large models
    - Exponential backoff retry for rate limits
    - Batch processing with configurable batch size
    - Token usage tracking
    - Both sync and async interfaces

    Example:
        ```python
        from src.rag.embeddings import OpenAIEmbeddingClient
        from src.rag.config import EmbeddingConfig

        config = EmbeddingConfig.from_env()
        client = OpenAIEmbeddingClient(config)

        # Single text
        embedding = await client.encode_async("Hello world")

        # Batch
        embeddings = await client.encode_batch_async([
            "First text",
            "Second text",
            "Third text"
        ])

        # Check usage
        stats = client.get_usage_stats()
        print(f"Total tokens used: {stats['total_tokens']}")
        ```
    """

    # Retry configuration
    MAX_RETRIES = 5
    INITIAL_RETRY_DELAY = 1.0  # seconds
    MAX_RETRY_DELAY = 60.0  # seconds
    RETRY_MULTIPLIER = 2.0

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAI Embedding Client.

        Args:
            config: EmbeddingConfig instance (preferred)
            api_key: OpenAI API key (overrides config)
            model: Model name (overrides config)
            base_url: API base URL (overrides config)
        """
        self.config = config or EmbeddingConfig()

        # Override config with explicit parameters
        self._api_key = api_key or self.config.api_key
        self._model = model or self.config.model_name
        self._base_url = base_url or self.config.api_base_url
        self._dimension = self.config.embedding_dimension
        self._batch_size = self.config.batch_size

        # Validate configuration
        if not self._api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY must be set for OpenAI embeddings. "
                "Set via environment variable or pass api_key parameter."
            )

        # Initialize clients
        self._sync_client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        self._async_client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

        # Usage tracking
        self._usage_stats = EmbeddingUsageStats()

        logger.info(
            f"Initialized OpenAIEmbeddingClient with model={self._model}, "
            f"dimension={self._dimension}"
        )

    # =========================================================================
    # SYNCHRONOUS INTERFACE
    # =========================================================================

    def encode(
        self, text: Union[str, List[str]], dimensions: Optional[int] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s) synchronously.

        Args:
            text: Single text or list of texts
            dimensions: Override embedding dimensions (for models that support it)

        Returns:
            Single embedding (List[float]) for single text,
            or list of embeddings for multiple texts.

        Raises:
            EmbeddingError: If embedding generation fails after retries
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        embeddings = self._embed_with_retry(texts, dimensions)

        return embeddings[0] if is_single else embeddings

    def encode_batch(
        self, texts: List[str], batch_size: Optional[int] = None, dimensions: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a large batch of texts synchronously.

        Automatically splits into smaller batches for API limits.

        Args:
            texts: List of texts to embed
            batch_size: Override default batch size
            dimensions: Override embedding dimensions

        Returns:
            List of embeddings in same order as input texts

        Raises:
            EmbeddingError: If embedding generation fails
        """
        batch_size = batch_size or self._batch_size
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_with_retry(batch, dimensions)
            all_embeddings.extend(batch_embeddings)

            # Log progress for large batches
            if len(texts) > batch_size:
                progress = min(i + batch_size, len(texts))
                logger.debug(f"Embedded {progress}/{len(texts)} texts")

        return all_embeddings

    def _embed_with_retry(
        self, texts: List[str], dimensions: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings with exponential backoff retry.

        Args:
            texts: List of texts to embed
            dimensions: Override embedding dimensions

        Returns:
            List of embeddings

        Raises:
            EmbeddingError: If all retries fail
        """
        last_error: Optional[Exception] = None
        delay = self.INITIAL_RETRY_DELAY

        for attempt in range(self.MAX_RETRIES):
            try:
                start_time = time.time()

                # Build request parameters
                params: Dict[str, Any] = {"input": texts, "model": self._model}

                # Add dimensions if model supports it
                dim = dimensions or self._dimension
                if dim and self._model in ("text-embedding-3-small", "text-embedding-3-large"):
                    params["dimensions"] = dim

                # Make API call
                response = self._sync_client.embeddings.create(**params)

                latency_ms = (time.time() - start_time) * 1000

                # Extract embeddings
                embeddings = [data.embedding for data in response.data]

                # Track usage
                self._usage_stats.record_success(
                    tokens=response.usage.total_tokens, texts=len(texts), latency_ms=latency_ms
                )

                return embeddings

            except openai.RateLimitError as e:
                last_error = e
                self._usage_stats.record_retry()
                logger.warning(
                    f"Rate limit hit, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                time.sleep(delay)
                delay = min(delay * self.RETRY_MULTIPLIER, self.MAX_RETRY_DELAY)

            except openai.APIError as e:
                last_error = e
                self._usage_stats.record_failure()
                logger.error(f"OpenAI API error: {e}")
                raise EmbeddingError(
                    f"OpenAI API error: {e}",
                    model=self._model,
                    batch_size=len(texts),
                    original_error=e,
                )

            except Exception as e:
                last_error = e
                self._usage_stats.record_failure()
                logger.error(f"Unexpected embedding error: {e}")
                raise EmbeddingError(
                    f"Unexpected embedding error: {e}",
                    model=self._model,
                    batch_size=len(texts),
                    original_error=e,
                )

        # All retries exhausted
        self._usage_stats.record_failure()
        raise EmbeddingError(
            f"Failed to generate embeddings after {self.MAX_RETRIES} retries",
            model=self._model,
            batch_size=len(texts),
            details={"last_error": str(last_error)},
            original_error=last_error,
        )

    # =========================================================================
    # ASYNCHRONOUS INTERFACE
    # =========================================================================

    async def encode_async(
        self, text: Union[str, List[str]], dimensions: Optional[int] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s) asynchronously.

        Args:
            text: Single text or list of texts
            dimensions: Override embedding dimensions

        Returns:
            Single embedding or list of embeddings

        Raises:
            EmbeddingError: If embedding generation fails
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        embeddings = await self._embed_with_retry_async(texts, dimensions)

        return embeddings[0] if is_single else embeddings

    async def encode_batch_async(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        dimensions: Optional[int] = None,
        concurrency: int = 3,
    ) -> List[List[float]]:
        """
        Generate embeddings for a large batch of texts asynchronously.

        Uses concurrent batch processing for improved performance.

        Args:
            texts: List of texts to embed
            batch_size: Override default batch size
            dimensions: Override embedding dimensions
            concurrency: Number of concurrent batch requests

        Returns:
            List of embeddings in same order as input texts
        """
        batch_size = batch_size or self._batch_size

        # Split into batches
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        # Process batches with limited concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def process_batch(batch: List[str]) -> List[List[float]]:
            async with semaphore:
                return await self._embed_with_retry_async(batch, dimensions)

        # Run all batches concurrently
        results = await asyncio.gather(*[process_batch(b) for b in batches])

        # Flatten results
        return [emb for batch_result in results for emb in batch_result]

    async def _embed_with_retry_async(
        self, texts: List[str], dimensions: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings with exponential backoff retry (async).

        Args:
            texts: List of texts to embed
            dimensions: Override embedding dimensions

        Returns:
            List of embeddings

        Raises:
            EmbeddingError: If all retries fail
        """
        last_error: Optional[Exception] = None
        delay = self.INITIAL_RETRY_DELAY

        for attempt in range(self.MAX_RETRIES):
            try:
                start_time = time.time()

                # Build request parameters
                params: Dict[str, Any] = {"input": texts, "model": self._model}

                # Add dimensions if model supports it
                dim = dimensions or self._dimension
                if dim and self._model in ("text-embedding-3-small", "text-embedding-3-large"):
                    params["dimensions"] = dim

                # Make async API call
                response = await self._async_client.embeddings.create(**params)

                latency_ms = (time.time() - start_time) * 1000

                # Extract embeddings
                embeddings = [data.embedding for data in response.data]

                # Track usage
                self._usage_stats.record_success(
                    tokens=response.usage.total_tokens, texts=len(texts), latency_ms=latency_ms
                )

                return embeddings

            except openai.RateLimitError as e:
                last_error = e
                self._usage_stats.record_retry()
                logger.warning(
                    f"Rate limit hit, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
                delay = min(delay * self.RETRY_MULTIPLIER, self.MAX_RETRY_DELAY)

            except openai.APIError as e:
                last_error = e
                self._usage_stats.record_failure()
                logger.error(f"OpenAI API error: {e}")
                raise EmbeddingError(
                    f"OpenAI API error: {e}",
                    model=self._model,
                    batch_size=len(texts),
                    original_error=e,
                )

            except Exception as e:
                last_error = e
                self._usage_stats.record_failure()
                logger.error(f"Unexpected embedding error: {e}")
                raise EmbeddingError(
                    f"Unexpected embedding error: {e}",
                    model=self._model,
                    batch_size=len(texts),
                    original_error=e,
                )

        # All retries exhausted
        self._usage_stats.record_failure()
        raise EmbeddingError(
            f"Failed to generate embeddings after {self.MAX_RETRIES} retries",
            model=self._model,
            batch_size=len(texts),
            details={"last_error": str(last_error)},
            original_error=last_error,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self._usage_stats.to_dict()

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._usage_stats = EmbeddingUsageStats()
        logger.info("Reset embedding usage statistics")

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    def __repr__(self) -> str:
        return (
            f"OpenAIEmbeddingClient("
            f"model={self._model}, "
            f"dimension={self._dimension}, "
            f"batch_size={self._batch_size})"
        )


# Convenience function for quick embedding
async def get_embedding(
    text: str, model: str = "text-embedding-3-small", api_key: Optional[str] = None
) -> List[float]:
    """
    Quick utility function to get a single embedding.

    For repeated use, prefer creating an OpenAIEmbeddingClient instance.

    Args:
        text: Text to embed
        model: Model name
        api_key: OpenAI API key (defaults to env var)

    Returns:
        Embedding vector
    """
    config = EmbeddingConfig(model_name=model, api_key=api_key)
    client = OpenAIEmbeddingClient(config)
    return await client.encode_async(text)
