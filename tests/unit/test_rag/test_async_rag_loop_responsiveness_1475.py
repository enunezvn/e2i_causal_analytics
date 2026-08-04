"""#1475 WI2: the RAG leg must not starve the event loop.

Two confirmed loop-blocking call sites on the request path (issue #1475):

1. ``src/memory/services/factories.py`` — ``OpenAIEmbeddingService`` used a
   sync ``openai.OpenAI`` client inside ``async def embed``, so the whole
   embedding HTTP round-trip blocked the loop.
2. ``src/rag/memory_connector.py`` — ``vector_search`` / ``fulltext_search``
   called the sync ``get_supabase_client()`` + sync ``.rpc().execute()``.

These tests prove loop responsiveness with REAL client machinery: real
``openai`` SDK clients and real ``supabase`` clients, with only the HTTP
transport swapped for a local ``httpx.MockTransport`` whose handler models the
network wait faithfully — a blocking ``time.sleep`` in the sync transport
(exactly how a sync socket read blocks the calling thread) and an
``await asyncio.sleep`` in the async transport (exactly how an async socket
read yields to the loop). A heartbeat task ticks concurrently; if the call
under test holds the loop, the ticks stop — which is precisely how these
tests FAILED against the sync implementation (red) before the migration.

The harness offers BOTH sync and async clients simultaneously and identically
delayed; the implementation under test picks whichever it builds, so red and
green run against the same harness.
"""

import asyncio
import contextlib
import time

import httpx
import pytest

# The fake "network" wait per HTTP call, and the heartbeat interval. With a
# responsive loop a 0.6s call overlaps ~12 ticks at 50ms; a blocked loop
# yields ~0. The >=5 assertions leave a wide margin for a loaded box.
BLOCK_SECONDS = 0.6
TICK_SECONDS = 0.05
MIN_TICKS = 5

EMBED_RESPONSE = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 8}],
    "model": "text-embedding-ada-002",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}


async def _ticks_while(coro):
    """Count heartbeat ticks that land while ``coro`` runs."""
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(TICK_SECONDS)
            ticks += 1

    task = asyncio.create_task(heartbeat())
    # Let the heartbeat establish itself before the call under test starts.
    await asyncio.sleep(0)
    try:
        result = await coro
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return ticks, result


# =============================================================================
# Embedding service (factories.OpenAIEmbeddingService)
# =============================================================================


@pytest.fixture
def openai_transport(monkeypatch):
    """Real openai SDK clients over a local mock transport, sync and async.

    ``OpenAIEmbeddingService._get_client`` does ``import openai`` and builds a
    client from the module attribute, so patching the module attributes reaches
    it regardless of which client class the implementation chooses.
    """
    import openai

    real_sync_cls = openai.OpenAI
    real_async_cls = openai.AsyncOpenAI
    hits = {"sync": 0, "async": 0}

    def sync_handler(request):
        hits["sync"] += 1
        time.sleep(BLOCK_SECONDS)  # a sync socket wait blocks the calling thread
        return httpx.Response(200, json=EMBED_RESPONSE)

    async def async_handler(request):
        hits["async"] += 1
        await asyncio.sleep(BLOCK_SECONDS)  # an async socket wait yields to the loop
        return httpx.Response(200, json=EMBED_RESPONSE)

    def sync_ctor(**kwargs):
        kwargs.setdefault("http_client", httpx.Client(transport=httpx.MockTransport(sync_handler)))
        return real_sync_cls(**kwargs)

    def async_ctor(**kwargs):
        kwargs.setdefault(
            "http_client",
            httpx.AsyncClient(transport=httpx.MockTransport(async_handler)),
        )
        return real_async_cls(**kwargs)

    monkeypatch.setattr(openai, "OpenAI", sync_ctor)
    monkeypatch.setattr(openai, "AsyncOpenAI", async_ctor)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    return hits


class TestEmbedLoopResponsiveness:
    @pytest.mark.asyncio
    async def test_embed_does_not_starve_a_concurrent_heartbeat(self, openai_transport):
        from src.memory.services.factories import OpenAIEmbeddingService

        service = OpenAIEmbeddingService()
        ticks, embedding = await _ticks_while(service.embed("loop responsiveness 1475"))

        # Guard against a false green from an error/cache short-circuit: the
        # embedding must really have crossed the (mock) HTTP layer once.
        assert openai_transport["sync"] + openai_transport["async"] == 1
        assert embedding == [0.1] * 8
        assert ticks >= MIN_TICKS, (
            f"embed() held the event loop: only {ticks} heartbeat ticks landed "
            f"during a {BLOCK_SECONDS}s embedding call (expected >= {MIN_TICKS}). "
            "The embedding HTTP round-trip is running synchronously on the loop."
        )

    @pytest.mark.asyncio
    async def test_embed_batch_does_not_starve_a_concurrent_heartbeat(self, openai_transport):
        from src.memory.services.factories import OpenAIEmbeddingService

        service = OpenAIEmbeddingService()
        ticks, embeddings = await _ticks_while(service.embed_batch(["a 1475", "b 1475"]))

        assert openai_transport["sync"] + openai_transport["async"] == 1
        assert embeddings == [[0.1] * 8]
        assert ticks >= MIN_TICKS, (
            f"embed_batch() held the event loop: only {ticks} ticks landed "
            f"(expected >= {MIN_TICKS})."
        )


# =============================================================================
# MemoryConnector vector_search / fulltext_search
# =============================================================================


@pytest.fixture
def supabase_transport(monkeypatch):
    """Real supabase clients (sync AND async) over a local mock transport.

    Both factory names are patched in the memory_connector module namespace so
    whichever client the implementation resolves, it gets the same delayed
    local transport. The postgrest session swap preserves base_url/headers, so
    ``.rpc().execute()`` runs the real request-builder machinery end to end.
    """
    import src.rag.memory_connector as mc

    hits = {"sync": 0, "async": 0}

    def sync_handler(request):
        hits["sync"] += 1
        time.sleep(BLOCK_SECONDS)
        return httpx.Response(200, json=[])

    async def async_handler(request):
        hits["async"] += 1
        await asyncio.sleep(BLOCK_SECONDS)
        return httpx.Response(200, json=[])

    from supabase import create_client

    sync_client = create_client("http://supabase.test", "test-key")
    old_session = sync_client.postgrest.session
    sync_client.postgrest.session = httpx.Client(
        base_url=old_session.base_url,
        headers=old_session.headers,
        transport=httpx.MockTransport(sync_handler),
    )

    async_holder = {}

    async def get_async_client():
        if "client" not in async_holder:
            from supabase import acreate_client

            aclient = await acreate_client("http://supabase.test", "test-key")
            aold = aclient.postgrest.session
            aclient.postgrest.session = httpx.AsyncClient(
                base_url=aold.base_url,
                headers=aold.headers,
                transport=httpx.MockTransport(async_handler),
            )
            async_holder["client"] = aclient
        return async_holder["client"]

    monkeypatch.setattr(mc, "get_supabase_client", lambda: sync_client, raising=False)
    monkeypatch.setattr(mc, "get_async_supabase_client", get_async_client, raising=False)
    return hits


class TestConnectorLoopResponsiveness:
    @pytest.mark.asyncio
    async def test_vector_search_does_not_starve_a_concurrent_heartbeat(self, supabase_transport):
        from src.rag.memory_connector import MemoryConnector

        connector = MemoryConnector()
        ticks, results = await _ticks_while(connector.vector_search([0.1] * 8, k=1))

        assert supabase_transport["sync"] + supabase_transport["async"] == 1, (
            "the RPC never crossed the HTTP layer — a short-circuit (e.g. a "
            "swallowed exception) would fake a responsive loop"
        )
        assert results == []
        assert ticks >= MIN_TICKS, (
            f"vector_search() held the event loop: only {ticks} ticks landed "
            f"during a {BLOCK_SECONDS}s RPC (expected >= {MIN_TICKS})."
        )

    @pytest.mark.asyncio
    async def test_fulltext_search_does_not_starve_a_concurrent_heartbeat(self, supabase_transport):
        from src.rag.memory_connector import MemoryConnector

        connector = MemoryConnector()
        ticks, results = await _ticks_while(connector.fulltext_search("probe 1475", k=1))

        assert supabase_transport["sync"] + supabase_transport["async"] == 1
        assert results == []
        assert ticks >= MIN_TICKS, (
            f"fulltext_search() held the event loop: only {ticks} ticks landed "
            f"(expected >= {MIN_TICKS})."
        )
