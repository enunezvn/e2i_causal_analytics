"""Faithful smoke test for the /api/v1/rag/* REST router (audit C1).

The router was dead-on-construction: RAGService.retriever built
HybridRetriever(self.config) (wrong positional -> falkordb_client missing),
called the nonexistent get_last_query_stats(), passed wrong kwargs to the
subgraph/path methods, and could not resolve falkordb at all from a SYNC
property because get_falkordb() is async (a sync property would store a
coroutine -> AttributeError on .select_graph). All masked by `# type: ignore`.

Every prior test MOCKED the unit under test (get_rag_service override /
service._retriever = mock), so the break was invisible. This test exercises
the REAL RAGService construction and asserts the endpoint does NOT 500. It
doubles ONLY the true externals (supabase client + the async falkordb getter).

SCOPE: the goal is "router is callable (no 500)", NOT retrieval relevance --
the corpus (rag_document_chunks empty) is a separate remediation phase.
"""

import inspect
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import rag as rag_module
from src.api.routes.rag import RAGService
from src.rag.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_api_surface_is_what_the_router_calls():
    """Pin the real API names the router depends on (re-breaks on a rename)."""
    ctor = inspect.signature(HybridRetriever.__init__)
    assert "supabase_client" in ctor.parameters
    assert "falkordb_client" in ctor.parameters
    # last_search_stats is a property; get_last_query_stats must NOT exist.
    assert isinstance(inspect.getattr_static(HybridRetriever, "last_search_stats"), property)
    assert not hasattr(HybridRetriever, "get_last_query_stats")
    sg = inspect.signature(HybridRetriever.get_causal_subgraph)
    assert "center_node_id" in sg.parameters and "max_depth" in sg.parameters
    cp = inspect.signature(HybridRetriever.get_causal_path)
    assert {"source_id", "target_id", "max_length"} <= set(cp.parameters)


def test_get_falkordb_is_async_so_sync_property_would_hold_a_coroutine():
    """Pin the BLOCKER premise: get_falkordb is async; a sync property that
    stores its return value holds a coroutine, not a usable client."""
    from src.api.dependencies.falkordb_client import get_falkordb

    assert inspect.iscoroutinefunction(get_falkordb)
    coro = get_falkordb()
    assert inspect.iscoroutine(coro)
    coro.close()  # don't leak the unawaited coroutine


def _make_fake_falkordb():
    """A stub FalkorDB client: select_graph(name).query(...) -> empty result_set."""
    fake_falkordb = MagicMock()
    fake_graph = MagicMock()
    fake_falkordb.select_graph.return_value = fake_graph
    fake_graph.query.return_value = MagicMock(result_set=[])
    return fake_falkordb


@pytest.fixture
def real_service_with_doubled_backends(monkeypatch):
    """Build the REAL RAGService.retriever; double only the network clients.

    No mock of RAGService or HybridRetriever -- the actual construction runs.
    The doubled supabase/falkordb clients return empty result sets so the
    backends short-circuit to [] (faithful to the empty-corpus production
    reality) without making a real network call. get_falkordb is doubled with
    an ASYNC fake (it is `async def` in production), so the fix's
    `await get_falkordb()` path is exercised faithfully.
    """
    # Reset the RAGService singleton so a fresh instance is built.
    RAGService._instance = None
    RAGService._initialized = False

    fake_supabase = MagicMock()
    # supabase .rpc(...).execute() -> object with .data == [] (no rows)
    fake_supabase.rpc.return_value.execute.return_value = MagicMock(data=[])
    fake_falkordb = _make_fake_falkordb()

    async def _fake_get_falkordb():
        return fake_falkordb

    # Double the embedding provider so _get_retriever's dense-leg wiring is
    # exercised WITHOUT a real OpenAI call. embed() returns a fixed-length dummy
    # vector; the doubled supabase RPC returns [] regardless, so results stay
    # empty (faithful to the empty-corpus reality). EmbeddingConfig.from_env is
    # also doubled so it never depends on a real key in the test env.
    class _FakeEmbeddingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def embed(self, text):
            return [0.0] * 1536

    monkeypatch.setattr(
        "src.rag.embeddings.OpenAIEmbeddingClient", _FakeEmbeddingClient, raising=False
    )
    monkeypatch.setattr(
        "src.rag.config.EmbeddingConfig.from_env",
        classmethod(lambda cls: MagicMock()),
        raising=False,
    )

    # Patch the dependency getters the (fixed) construction will call by name.
    # raising=False is REQUIRED: on UNFIXED code rag.py has not yet imported
    # get_supabase/get_falkordb (the fix adds those imports), so raising=True
    # would AttributeError at fixture setup BEFORE the endpoint runs. With
    # raising=False the fakes are installed harmlessly; the unfixed sync
    # `retriever` property ignores them and still runs HybridRetriever(self.config)
    # -> TypeError -> the real 500 we assert against. After the fix, the
    # imported names exist and the fakes are used -> 200.
    monkeypatch.setattr(rag_module, "get_supabase", lambda: fake_supabase, raising=False)
    monkeypatch.setattr(rag_module, "get_falkordb", _fake_get_falkordb, raising=False)

    yield
    RAGService._instance = None
    RAGService._initialized = False


def test_search_endpoint_does_not_500(real_service_with_doubled_backends):
    """POST /api/v1/rag/search returns 200 (or a graceful 4xx/503), never 500.

    Before the fix this 500s with a TypeError on HybridRetriever construction
    (and would 500 again on a coroutine.select_graph if 'fixed' with a sync
    property). After the fix it returns 200 with an empty results list.
    """
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/v1/rag/search",
        json={"query": "Why did Kisqali TRx drop in the West during Q3?", "top_k": 5},
    )
    assert resp.status_code != 500, resp.text
    assert resp.status_code in (200, 401, 422, 503), resp.text
    if resp.status_code == 200:
        body = resp.json()
        # Valid SearchResponse shape; empty results are acceptable (empty corpus).
        assert "results" in body and "stats" in body and "search_id" in body
        assert isinstance(body["results"], list)
