"""Offline guards for the chat-RAG chunk corpus ingestion path (#1373).

Chat RAG (``HybridRetriever`` -> ``rag_vector_search`` RPC) reads
``rag_document_chunks`` embedded in the ``text-embedding-3-small`` space that
chat queries embed in (``src/rag/embeddings.py``). That table was NEVER
populated, so the dense leg returned 0 results (graph-only chat). This module
renders REAL ``business_metrics`` rows (values VERBATIM from the fact table --
F3 anti-mocking) via the SHARED ``render_business_metric`` renderer, embeds via
the RAG-side ``OpenAIEmbeddingClient`` (the same space chat queries use), and
upserts into ``rag_document_chunks``.

These tests use fluent supabase-py fakes + an injected embedding client (no DB,
no network, no mock of the unit under test). They pin:
  * rendering reuse (no forked renderer),
  * dedup-BEFORE-embed (a re-run must not re-embed unchanged prose -> $$),
  * provenance filtering on the reads,
  * column population (lowercased brand/region, model name, kpi_name, no
    is_synthetic key so the DB default false lands -> retrievable by chat),
  * the upsert conflict target (idempotency).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from src.rag import chunk_corpus_ingestion as cci
from src.rag import corpus_ingestion as ci

# A single real-shaped business_metrics row (display casing on prose, values
# verbatim). brand/region carry canonical (mixed) case as the fact table stores.
_ROW = {
    "metric_name": "TRx",
    "brand": "Kisqali",
    "region": "Northeast",
    "metric_date": "2025-01-01",
    "value": 684127.3,
    "target": 736992.12,
    "achievement_rate": 0.928,
    "year_over_year_change": 0.187,
    "roi": 4.09,
}


class _FakeQuery:
    """Fluent supabase-py query stub returning store-configured data and
    recording every ``.eq`` call + upsert payload."""

    def __init__(self, table: str, store: dict[str, Any]):
        self.table = table
        self.store = store
        self._select: tuple[Any, ...] = ()
        self._select_kw: dict[str, Any] = {}
        self._range: tuple[Any, ...] | None = None

    def select(self, *cols: Any, **kw: Any) -> "_FakeQuery":
        self._select = cols
        self._select_kw = kw
        return self

    def eq(self, *a: Any) -> "_FakeQuery":
        self.store.setdefault("eq_calls", {}).setdefault(self.table, []).append(a)
        return self

    @property
    def not_(self) -> "_FakeQuery":
        return self

    def is_(self, *_a: Any) -> "_FakeQuery":
        return self

    def order(self, *_a: Any, **_k: Any) -> "_FakeQuery":
        return self

    def limit(self, *_a: Any) -> "_FakeQuery":
        return self

    def range(self, *a: Any) -> "_FakeQuery":
        self._range = a
        return self

    def upsert(self, records: Any, **kw: Any) -> "_FakeQuery":
        self.store.setdefault("upserts", []).append({"records": records, "kw": kw})
        return self

    def execute(self) -> Any:
        if self._select_kw.get("count") is not None:
            return SimpleNamespace(data=[], count=self.store.get("count", 0))
        if self.table == "business_metrics":
            if self._select == ("brand",):
                data = self.store.get("brands_rows", [])
            else:
                data = self.store.get("metric_rows", [])
        elif self.table == "rag_document_chunks":
            data = self.store.get("existing_hash_rows", [])
        else:
            data = []
        return SimpleNamespace(data=data, count=self.store.get("count", 0))


class _FakeClient:
    def __init__(self, store: dict[str, Any]):
        self.store = store

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(name, self.store)


class _FakeEmbedder:
    """Injected embedding client recording every batch-embed call."""

    def __init__(self, model: str = "text-embedding-3-small", vec: list[float] | None = None):
        self.model = model
        self.calls: list[list[str]] = []
        self._vec = vec or [0.1, 0.2, 0.3]

    async def encode_batch_async(self, texts: list[str], **_kw: Any) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self._vec) for _ in texts]


def _run_index(store: dict[str, Any], embedder: _FakeEmbedder, **kw: Any) -> list[str]:
    return asyncio.run(
        cci.index_business_metric_chunks(
            brands=["Kisqali"],
            supabase_client=_FakeClient(store),
            embedding_client=embedder,
            **kw,
        )
    )


def test_renders_via_shared_business_metric_renderer() -> None:
    """The chunk content must be the SHARED renderer's output verbatim -- no
    forked rendering logic (F3: values come from the fact table)."""
    store: dict[str, Any] = {"metric_rows": [dict(_ROW)]}
    emb = _FakeEmbedder()
    _run_index(store, emb)
    upserts = store["upserts"]
    assert len(upserts) == 1
    rec = upserts[0]["records"][0]
    assert rec["content"] == ci.render_business_metric(_ROW)


def test_populates_columns_lowercased_and_no_is_synthetic() -> None:
    """brand/region lowercased on columns, embedding_model = the ACTUAL model,
    kpi_name populated, chunk_index 0, and NO is_synthetic key so the DB default
    (false) applies -> the chunk is retrievable by the live chat path (which
    never opts into synthetic)."""
    store: dict[str, Any] = {"metric_rows": [dict(_ROW)]}
    emb = _FakeEmbedder(model="text-embedding-3-small")
    _run_index(store, emb)
    rec = store["upserts"][0]["records"][0]
    assert rec["brand"] == "kisqali"
    assert rec["region"] == "northeast"
    assert rec["document_type"] == "kpi_snapshot"
    assert rec["agent_name"] == "corpus_ingestion"
    assert rec["kpi_name"] == "TRx"
    assert rec["chunk_index"] == 0
    assert rec["embedding"] == [0.1, 0.2, 0.3]
    assert rec["embedding_model"] == "text-embedding-3-small"
    assert rec["content_hash"] == cci._content_hash(ci.render_business_metric(_ROW))
    # is_synthetic MUST be absent -> DB column default false -> retrievable.
    assert "is_synthetic" not in rec


def test_upsert_uses_document_chunk_conflict_target() -> None:
    """Idempotency: upsert conflict target is (document_id, chunk_index) so a
    re-run of a CHANGED snapshot overwrites its combo row rather than duplicating."""
    store: dict[str, Any] = {"metric_rows": [dict(_ROW)]}
    _run_index(store, _FakeEmbedder())
    kw = store["upserts"][0]["kw"]
    assert kw.get("on_conflict") == "document_id,chunk_index"
    rec = store["upserts"][0]["records"][0]
    # document_id is stable per (brand, metric, region) combo, lowercased.
    assert rec["document_id"] == cci._chunk_document_id("TRx", "Kisqali", "Northeast")


def test_dedup_skips_embedding_for_already_indexed_prose() -> None:
    """A re-run must NOT re-embed prose already indexed (embedding = real $$).
    With the rendered row's content_hash already present, nothing is embedded
    and nothing is upserted."""
    content = ci.render_business_metric(_ROW)
    store: dict[str, Any] = {
        "metric_rows": [dict(_ROW)],
        "existing_hash_rows": [{"content_hash": cci._content_hash(content)}],
    }
    emb = _FakeEmbedder()
    out = _run_index(store, emb)
    assert out == []
    assert emb.calls == []  # no embed call -> no spend
    assert "upserts" not in store  # nothing written


def test_new_prose_is_embedded_once() -> None:
    """A new (unseen-hash) row IS embedded exactly once (the batch call)."""
    store: dict[str, Any] = {"metric_rows": [dict(_ROW)], "existing_hash_rows": []}
    emb = _FakeEmbedder()
    out = _run_index(store, emb)
    assert len(out) == 1
    assert len(emb.calls) == 1
    assert emb.calls[0] == [ci.render_business_metric(_ROW)]


def test_existing_chunk_hashes_applies_provenance_and_agent_filter(monkeypatch: Any) -> None:
    """The dedup read excludes synthetic chunks (real mode) and is scoped to the
    corpus agent_name -- it must not be a vacuous full-table scan, and a
    synthetic chunk must not suppress ingesting the real row (mirrors the
    episodic dedup guard, Shard 07 R15).

    Force real-mode explicitly: local pytest absorbs the main repo's .env (which
    sets E2I_INCLUDE_SYNTHETIC on the showcase box) via the opik->litellm
    load_dotenv chain, so the provenance predicate would otherwise be a no-op.
    """
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "0")
    store: dict[str, Any] = {}
    cci._existing_chunk_hashes(
        _FakeClient(store), agent_name="corpus_ingestion", document_type="kpi_snapshot"
    )
    eqs = store.get("eq_calls", {}).get("rag_document_chunks", [])
    assert ("is_synthetic", False) in eqs
    assert ("agent_name", "corpus_ingestion") in eqs


def test_brand_discovery_excludes_synthetic(monkeypatch: Any) -> None:
    """brands=None triggers shared brand discovery, which must default-exclude
    synthetic business_metrics rows. Force real-mode explicitly (see note in
    ``test_existing_chunk_hashes_applies_provenance_and_agent_filter``)."""
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "0")
    store: dict[str, Any] = {"brands_rows": []}  # empty -> early return after discovery
    asyncio.run(
        cci.index_business_metric_chunks(
            supabase_client=_FakeClient(store), embedding_client=_FakeEmbedder()
        )
    )
    eqs = store.get("eq_calls", {}).get("business_metrics", [])
    assert ("is_synthetic", False) in eqs


def test_chunk_corpus_health_warns_when_empty(caplog: Any) -> None:
    store: dict[str, Any] = {"count": 0}
    with caplog.at_level("WARNING"):
        health = asyncio.run(cci.chunk_corpus_health(supabase_client=_FakeClient(store)))
    assert health == {"chunk_count": 0, "empty": True}
    assert any("empty" in r.message.lower() or "0" in r.message for r in caplog.records)


def test_chunk_corpus_health_ok_when_populated(caplog: Any) -> None:
    store: dict[str, Any] = {"count": 42}
    with caplog.at_level("WARNING"):
        health = asyncio.run(cci.chunk_corpus_health(supabase_client=_FakeClient(store)))
    assert health == {"chunk_count": 42, "empty": False}
    assert not caplog.records  # no warning when populated
