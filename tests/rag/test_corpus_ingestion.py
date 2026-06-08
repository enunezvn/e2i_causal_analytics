"""Faithful full-corpus ingestion + retrieval gate (audit F3, Phase 5).

Ingests via the REAL durable corpus path (index_business_metrics reads the live
business_metrics fact table -- no hand-authored/fabricated corpus) and asserts
every query archetype (KPI / context / explanation) retrieves a topically-
relevant row above the effective 0.5 cosine floor. No mock of the unit under
test; the only externals are the live Supabase + prod OpenAI embedder.

DROPLET-ONLY MANUAL GATE: opt-in via E2I_RUN_LIVE_RAG=1. No CI lane collects
tests/rag/, and every CI lane sets SUPABASE_URL=http://localhost:54321, so a
presence-of-SUPABASE_URL guard would FAIL in CI. The explicit opt-in flag
(precedent: E2I_RUN_REAL_LLM_E2E) keeps it inert everywhere but the droplet.
"""

import os

import pytest

from src.rag.corpus_ingestion import index_business_metrics
from src.rag.memory_connector import get_memory_connector

EFFECTIVE_FLOOR = 0.5  # see 011_...sql:126 + memory_connector.py:152

# (query, tokens-any-of). All tokens are REAL values present in business_metrics
# (brands Kisqali/Fabhalta; KPI names TRx/Conversion/ROI; regions northeast/...).
ARCHETYPES = {
    "kpi": ("TRx trend for Kisqali in the Northeast", ("kisqali", "trx")),
    "context": ("Market context and conversion rate for Fabhalta", ("fabhalta",)),
    "explanation": ("Return on investment performance for Kisqali", ("kisqali", "roi")),
}


@pytest.mark.skipif(
    os.getenv("E2I_RUN_LIVE_RAG") != "1",
    reason="droplet-only faithful gate; set E2I_RUN_LIVE_RAG=1 to run",
)
@pytest.mark.asyncio
async def test_full_corpus_relevant_for_all_archetypes():
    # Faithful, IDEMPOTENT ingestion of a bounded REAL slice (no fabrication).
    # index_business_metrics dedups by default, so a warm corpus inserts nothing
    # new -- this is correct and the gate below (retrieval relevance) is the real
    # assertion, valid whether the rows were freshly inserted or already present.
    await index_business_metrics(brands=["Kisqali", "Fabhalta"], limit_per_brand=15)

    connector = get_memory_connector()
    for name, (query, tokens) in ARCHETYPES.items():
        results = await connector.vector_search_by_text(
            query_text=query, k=8, min_similarity=EFFECTIVE_FLOOR
        )
        assert results, f"{name}: zero rows for {query!r}"
        blob = " ".join(r.content.lower() for r in results)
        assert any(t in blob for t in tokens), (
            f"{name}: no retrieved row mentions {tokens}; "
            "top=" + " | ".join(r.content[:50] for r in results[:5])
        )
