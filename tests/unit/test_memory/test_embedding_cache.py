"""
Unit tests for the in-process embedding cache (L17).

The embedding cache on OpenAIEmbeddingService / LocalEmbeddingService must:
- Use a STABLE, deterministic key (hashlib.sha256 hexdigest of the text), not
  the per-process-randomized builtin hash(), so the same text maps to the same
  key across processes and there is no collision risk.
- Be BOUNDED (cap the number of cached entries; evict oldest on overflow) so a
  long-lived service does not grow the cache without limit.
- Preserve cache-HIT semantics: the same text within a process returns the same
  cached embedding without re-calling the backend.
"""

import hashlib
from typing import List

import pytest

from src.memory.services.factories import (
    EMBEDDING_CACHE_MAX_ENTRIES,
    LocalEmbeddingService,
    OpenAIEmbeddingService,
    _embedding_cache_key,
)


class _StubOpenAIClient:
    """Counts how many times the embeddings backend is invoked.

    ``create`` is async since #1475 migrated the service to
    ``openai.AsyncOpenAI`` (the awaited SDK surface).
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.embeddings = self

    async def create(self, model: str, input):  # noqa: A002 - mirror openai signature
        # input may be str (embed) or list (embed_batch)
        self.call_count += 1

        class _Item:
            def __init__(self, embedding: List[float]):
                self.embedding = embedding

        class _Resp:
            pass

        resp = _Resp()
        if isinstance(input, list):
            resp.data = [_Item([0.1, 0.2, 0.3]) for _ in input]
        else:
            resp.data = [_Item([0.1, 0.2, 0.3])]
        return resp


class TestEmbeddingCacheKey:
    """The cache key must be a stable sha256 hexdigest, not builtin hash()."""

    def test_key_is_stable_sha256_hexdigest(self):
        text = "remibrutinib reduces CSU symptoms"
        expected = hashlib.sha256(text.encode()).hexdigest()
        assert _embedding_cache_key(text) == expected

    def test_key_is_deterministic_across_calls(self):
        text = "deterministic key check"
        assert _embedding_cache_key(text) == _embedding_cache_key(text)

    def test_key_differs_for_different_text(self):
        assert _embedding_cache_key("alpha") != _embedding_cache_key("beta")


class TestEmbeddingCacheHit:
    """Repeated text must hit the cache and not re-call the backend."""

    @pytest.mark.asyncio
    async def test_repeated_text_hits_cache(self):
        service = OpenAIEmbeddingService()
        stub = _StubOpenAIClient()
        service._client = stub  # inject stub so no real API call happens

        first = await service.embed("hello world")
        second = await service.embed("hello world")

        assert first == second
        # Backend invoked only once: second call served from cache.
        assert stub.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_keyed_by_stable_digest(self):
        service = OpenAIEmbeddingService()
        stub = _StubOpenAIClient()
        service._client = stub

        text = "stable digest cache entry"
        await service.embed(text)

        # The entry must be stored under the sha256 hexdigest key.
        assert _embedding_cache_key(text) in service._cache


class TestEmbeddingCacheBound:
    """The cache must be bounded and evict oldest entries past the cap."""

    @pytest.mark.asyncio
    async def test_cache_size_never_exceeds_cap(self):
        service = OpenAIEmbeddingService()
        stub = _StubOpenAIClient()
        service._client = stub

        # Insert more distinct texts than the cap.
        n = EMBEDDING_CACHE_MAX_ENTRIES + 50
        for i in range(n):
            await service.embed(f"text-{i}")

        assert len(service._cache) <= EMBEDDING_CACHE_MAX_ENTRIES

    @pytest.mark.asyncio
    async def test_oldest_entry_evicted_first(self):
        service = OpenAIEmbeddingService()
        stub = _StubOpenAIClient()
        service._client = stub

        # Fill exactly to the cap.
        for i in range(EMBEDDING_CACHE_MAX_ENTRIES):
            await service.embed(f"fill-{i}")

        oldest_key = _embedding_cache_key("fill-0")
        assert oldest_key in service._cache

        # One more insertion should evict the oldest (FIFO).
        await service.embed("overflow")
        assert oldest_key not in service._cache
        assert _embedding_cache_key("overflow") in service._cache
        assert len(service._cache) <= EMBEDDING_CACHE_MAX_ENTRIES

    @pytest.mark.asyncio
    async def test_local_service_cache_is_bounded_and_stable(self):
        """LocalEmbeddingService shares the same bounded/stable cache contract."""
        service = LocalEmbeddingService()

        # Stub the model so no real sentence-transformers load happens.
        class _StubModel:
            def encode(self, text, convert_to_numpy=True):
                import numpy as np

                return np.array([0.0, 1.0, 2.0])

        service._model = _StubModel()

        text = "local stable digest"
        await service.embed(text)
        assert _embedding_cache_key(text) in service._cache

        n = EMBEDDING_CACHE_MAX_ENTRIES + 25
        for i in range(n):
            await service.embed(f"local-{i}")
        assert len(service._cache) <= EMBEDDING_CACHE_MAX_ENTRIES
