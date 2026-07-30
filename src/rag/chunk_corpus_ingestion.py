"""Populate the chat-RAG chunk corpus (``rag_document_chunks``) in the
``text-embedding-3-small`` space (#1373).

Root cause (two stacked defects, measured in the prod container 2026-07-30):

1. ``rag_document_chunks`` -- the substrate the chat ``HybridRetriever`` /
   ``rag_vector_search`` RPC reads -- had ZERO rows; the dense leg of chat RAG
   returned 0 results (graph-only chat).
2. The RPC also UNIONs ``episodic_memories``, which DOES hold a populated
   operational corpus (120 rows written by :mod:`src.rag.corpus_ingestion`). But
   that corpus was embedded by the MEMORY path's ``OpenAIEmbeddingService``
   (default ``text-embedding-ada-002``), while chat queries embed via the
   RAG-side :class:`~src.rag.embeddings.OpenAIEmbeddingClient`
   (``text-embedding-3-small``). Cross-model embedding spaces are uncorrelated
   (cos ≈ -0.009 for identical text), and the RPC's 0.3 cosine floor means that
   episodic branch can NEVER match a chat query. So even the populated episodic
   corpus is invisible to chat RAG.

This module renders REAL ``business_metrics`` rows as analytic prose -- values
taken VERBATIM from the fact table (F3 anti-mocking) via the SHARED
:func:`src.rag.corpus_ingestion.render_business_metric` renderer -- embeds them
with the RAG-side client (SAME space chat queries use), and upserts them into
``rag_document_chunks``. It reuses the episodic corpus's proven row-selection
logic (``_fetch_brand_rows`` latest-per-combo + ``_discover_brands``).

Provenance (the #1373 trap, resolved deliberately -- see the PR body): the
chunks are written with NO ``is_synthetic`` value, so the column DEFAULT
(``false``, migration ``database/rag/005``) applies. This MIRRORS the episodic
corpus convention (its insert path never sets ``is_synthetic`` either -> the 120
live rows are ``is_synthetic=false``) and is the ONLY value that keeps the
corpus retrievable: the live chat path (``chatbot_dspy`` cognitive RAG ->
``hybrid_search(filters={'brand': ...})``) never passes ``include_synthetic``,
so ``src/rag/backends/vector.py`` fails closed to ``include_synthetic='false'``,
and the RPC predicate then admits ONLY rows with ``is_synthetic=false``. Marking
chunks synthetic (e.g. honoring a synthetic-gold source row's flag) would make
the whole corpus invisible again -- the exact bug this fixes. The showcase
deployment treats synthetic-gold as first-class data
(``deployment_includes_synthetic``), so a corpus rendered from it, with verbatim
values, is legitimately a first-class corpus artifact.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from src.rag.config import EmbeddingConfig
from src.rag.corpus_ingestion import (
    _DEFAULT_AGENT_NAME,
    _discover_brands,
    _fetch_brand_rows,
    render_business_metric,
)
from src.rag.embeddings import OpenAIEmbeddingClient

logger = logging.getLogger(__name__)

# document_type for KPI-snapshot chunks. rag_document_chunks.document_type is a
# free VARCHAR(50) (no enum), so no migration is needed for this value.
CHUNK_DOCUMENT_TYPE = "kpi_snapshot"
# Attribution matches the episodic corpus (e2i_agent_name 'corpus_ingestion') so
# the two corpora are selectively removable/re-syncable and share dedup scoping.
CHUNK_AGENT_NAME = _DEFAULT_AGENT_NAME


def _content_hash(content: str) -> str:
    """SHA-256 of the rendered prose (matches the DB ``content_hash`` column and
    the ``upsert_rag_chunk`` RPC's ``encode(sha256(...),'hex')``)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chunk_document_id(metric_name: Any, brand: Any, region: Any) -> str:
    """Stable ``document_id`` for one (metric, brand, region) combo, lowercased.

    Stable across runs so the upsert conflict target ``(document_id,
    chunk_index)`` overwrites a combo's row when its latest snapshot changes,
    rather than accumulating duplicates.
    """
    return (
        f"kpi::{(brand or '').lower()}::{(metric_name or '').lower()}::{(region or '').lower()}"
    )


def _existing_chunk_hashes(sb: Any, *, agent_name: str, document_type: str) -> set[str]:
    """Return already-indexed ``content_hash`` values for the KPI-snapshot corpus.

    Dedup-BEFORE-embed: a re-run skips rows whose rendered prose is already
    indexed so unchanged snapshots are never re-embedded (embedding is real
    spend). Scoped to ``agent_name`` + ``document_type`` and provenance-filtered
    (real mode excludes synthetic chunks) so a synthetic chunk can't suppress
    ingesting the real row -- mirrors the episodic dedup guard (Shard 07 R15).
    """
    from src.repositories.provenance import apply_provenance_filter

    seen: set[str] = set()
    page = 0
    page_size = 1000
    while True:
        q = (
            sb.table("rag_document_chunks")
            .select("content_hash")
            .eq("agent_name", agent_name)
            .eq("document_type", document_type)
        )
        q = apply_provenance_filter(q)
        resp = q.range(page * page_size, page * page_size + page_size - 1).execute()
        batch = resp.data or []
        for row in batch:
            if row.get("content_hash"):
                seen.add(row["content_hash"])
        if len(batch) < page_size:
            break
        page += 1
    return seen


def _count_chunk_corpus(sb: Any) -> int:
    """Exact row count of ``rag_document_chunks`` (chat-RAG substrate)."""
    resp = sb.table("rag_document_chunks").select("chunk_id", count="exact").limit(1).execute()
    return int(getattr(resp, "count", 0) or 0)


async def index_business_metric_chunks(
    *,
    brands: Optional[list[str]] = None,
    limit_per_brand: int = 50,
    supabase_client: Any = None,
    embedding_client: Any = None,
    agent_name: str = CHUNK_AGENT_NAME,
    document_type: str = CHUNK_DOCUMENT_TYPE,
    dedup: bool = True,
    latest_per_combo: bool = False,
) -> list[str]:
    """Render REAL ``business_metrics`` rows and index them into
    ``rag_document_chunks`` in the ``text-embedding-3-small`` space (#1373).

    Reads the fact table directly (no hand-authored corpus), renders each row
    faithfully via the shared renderer, embeds via the RAG-side
    :class:`OpenAIEmbeddingClient` (the SAME space chat queries use), and upserts
    on ``(document_id, chunk_index)``. Idempotent by default: rows whose rendered
    prose is already indexed under ``agent_name``/``document_type`` are skipped
    BEFORE embedding, so a re-run (or the scheduled re-sync) embeds only
    new/changed snapshots.

    Args:
        brands: restrict to these brands (defaults to all present).
        limit_per_brand: cap rows per brand when ``latest_per_combo`` is False.
        supabase_client: injected client (defaults to the prod client).
        embedding_client: injected RAG embedding client (defaults to one built
            from :meth:`EmbeddingConfig.from_env`, ``text-embedding-3-small``).
        agent_name: e2i attribution (default 'corpus_ingestion').
        document_type: chunk document_type (default 'kpi_snapshot').
        dedup: skip already-indexed prose (default True).
        latest_per_combo: index the latest snapshot of EVERY (metric, region)
            combo per brand (full coverage; ``limit_per_brand`` ignored). The
            durable Celery sync uses this so no combo is omitted.

    Returns:
        The ``document_id``s of the NEW/changed chunks upserted (empty when the
        corpus is already up to date).
    """
    from src.memory.services.factories import get_supabase_client

    sb = supabase_client or get_supabase_client()
    client = embedding_client or OpenAIEmbeddingClient(EmbeddingConfig.from_env())

    if brands is None:
        brands = _discover_brands(sb)

    already = _existing_chunk_hashes(sb, agent_name=agent_name, document_type=document_type) if dedup else set()

    # Build the pending set (dedup BEFORE embedding).
    pending: list[dict[str, Any]] = []
    batch_hashes: set[str] = set()
    skipped = 0
    for brand in brands:
        for row in _fetch_brand_rows(sb, brand, limit_per_brand, latest_per_combo):
            content = render_business_metric(row)
            chash = _content_hash(content)
            if chash in already or chash in batch_hashes:
                skipped += 1
                continue
            batch_hashes.add(chash)
            metric_name = row.get("metric_name")
            row_brand = row.get("brand")
            row_region = row.get("region")
            pending.append(
                {
                    "document_id": _chunk_document_id(metric_name, row_brand, row_region),
                    "content": content,
                    "content_hash": chash,
                    "brand": (row_brand or "").lower() or None,
                    "region": (row_region or "").lower() or None,
                    "kpi_name": metric_name,
                }
            )

    if skipped:
        logger.info("index_business_metric_chunks: skipped %d already-indexed/duplicate rows", skipped)
    if not pending:
        logger.info("index_business_metric_chunks: nothing new to index (chunk corpus up to date)")
        return []

    embeddings = await client.encode_batch_async([p["content"] for p in pending])

    records: list[dict[str, Any]] = []
    for p, embedding in zip(pending, embeddings, strict=True):
        record = {
            "document_id": p["document_id"],
            "document_type": document_type,
            "chunk_index": 0,
            "content": p["content"],
            "content_hash": p["content_hash"],
            "embedding": embedding,
            "agent_name": agent_name,
            "kpi_name": p["kpi_name"],
            # embedding_model = the ACTUAL model used (honest; not the column
            # default) so a deployment overriding EMBEDDING_MODEL is recorded.
            "embedding_model": client.model,
            # cheap word-count token estimate (matches upsert_rag_chunk's semantics).
            "token_count": len(p["content"].split()),
            "metadata": {"source": "business_metrics", "ingested_via": "chunk_corpus_ingestion"},
        }
        # brand/region omitted when null (keep the column NULL, not empty string).
        if p["brand"]:
            record["brand"] = p["brand"]
        if p["region"]:
            record["region"] = p["region"]
        # NOTE: is_synthetic intentionally NOT set -> DB default false -> the
        # chunk is retrievable by the live chat path (see module docstring).
        records.append(record)

    sb.table("rag_document_chunks").upsert(records, on_conflict="document_id,chunk_index").execute()
    logger.info(
        "index_business_metric_chunks: upserted %d chunks (agent=%s, model=%s)",
        len(records),
        agent_name,
        client.model,
    )
    return [p["document_id"] for p in pending]


async def chunk_corpus_health(supabase_client: Any = None) -> dict[str, Any]:
    """Report ``rag_document_chunks`` corpus size for chat-RAG observability (#1373 step 3).

    Emits a WARNING when the corpus is empty -- the honest signal that the dense
    leg of chat RAG will return 0 results until ``sync_chunk_corpus`` (or the
    scripts entrypoint) runs. Returns ``{"chunk_count": n, "empty": bool}``.
    """
    from src.memory.services.factories import get_supabase_client

    sb = supabase_client or get_supabase_client()
    count = _count_chunk_corpus(sb)
    if count == 0:
        logger.warning(
            "chat-RAG chunk corpus is EMPTY (rag_document_chunks) -- dense retrieval "
            "returns 0 results; run sync_chunk_corpus / scripts/rag/ingest_chunk_corpus.py"
        )
    return {"chunk_count": count, "empty": count == 0}
