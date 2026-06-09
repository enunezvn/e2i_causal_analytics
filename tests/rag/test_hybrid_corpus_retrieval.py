"""Faithful hybrid-retrieval gate for the operational corpus (audit F2/F3a/F4).

These extend ``test_corpus_substrate_relevance.py`` (which only proves the dense
leg with NO brand filter) to the three gaps a faithful re-run of the audit
exposed AFTER the corpus was populated:

* **F3a** — the live chatbot passes a TITLE-CASE brand filter (``"Kisqali"``)
  but the corpus rows store brand lowercase (``"kisqali"``); the dense RPC did an
  exact ``em.brand = filter`` match, so every corpus row was silently excluded
  and only NULL-brand ``[PROC]`` junk survived. Fix = case-insensitive brand/
  region matching in ``hybrid_vector_search``.
* **F2** — ``hybrid_fulltext_search`` UNIONs only causal_paths/agent_activities/
  triggers; ``episodic_memories`` (where the corpus lives) was never queried, so
  the sparse leg returned 0 for every commercial query. Fix = add an
  ``episodic_memories`` branch over the already-populated ``search_text``
  tsvector (GIN ``idx_episodic_search``).
* **F4** — with the corpus reachable on BOTH dense and sparse, the same row
  appears in two legs and ``_reciprocal_rank_fusion`` (content-aware
  ``dedup_key``) boosts it, so the "hybrid" pipeline stops degenerating to a
  single-leg identity re-rank of dense junk.

DROPLET-ONLY MANUAL GATE: opt-in via E2I_RUN_LIVE_RAG=1 (precedent:
E2I_RUN_REAL_LLM_E2E). No CI lane collects tests/rag/, and every CI lane sets
SUPABASE_URL=http://localhost:54321, so these are inert everywhere but the
droplet (the faithful environment: live Supabase + prod embedder). No mock of
the unit under test.
"""

import os

import pytest

from src.rag.memory_connector import get_memory_connector
from src.rag.retriever import hybrid_search

# Title-case brand exactly as the live chatbot forwards it
# (chatbot_dspy.py:1550 filters={"brand": brand_context}; the UI sends "Kisqali").
TITLECASE_BRAND = "Kisqali"
COMMERCIAL_QUERY = "TRx trend for Kisqali in the Northeast this quarter"
# REAL tokens present in business_metrics corpus prose (no invented vocabulary).
CORPUS_TOKENS = ("kisqali", "trx")
EFFECTIVE_FLOOR = 0.5  # see 011_...sql + memory_connector.py default min_similarity


def _mentions_corpus(content: str) -> bool:
    low = content.lower()
    return all(tok in low for tok in CORPUS_TOKENS)


@pytest.mark.skipif(
    os.getenv("E2I_RUN_LIVE_RAG") != "1",
    reason="droplet-only faithful gate; set E2I_RUN_LIVE_RAG=1 to run",
)
@pytest.mark.asyncio
async def test_dense_corpus_retrievable_with_titlecase_brand_filter():
    """F3a: a title-case brand filter must still surface the lowercase corpus."""
    connector = get_memory_connector()
    results = await connector.vector_search_by_text(
        query_text=COMMERCIAL_QUERY,
        k=8,
        filters={"brand": TITLECASE_BRAND},
        min_similarity=EFFECTIVE_FLOOR,
    )
    assert results, "dense leg returned ZERO rows under a title-case brand filter"
    assert any(_mentions_corpus(r.content) for r in results), (
        "title-case brand filter excluded the lowercase corpus rows; "
        "top=" + " | ".join(r.content[:60] for r in results[:5])
    )


@pytest.mark.skipif(
    os.getenv("E2I_RUN_LIVE_RAG") != "1",
    reason="droplet-only faithful gate; set E2I_RUN_LIVE_RAG=1 to run",
)
@pytest.mark.asyncio
async def test_sparse_leg_serves_operational_corpus():
    """F2: the sparse/full-text leg must return episodic corpus rows."""
    connector = get_memory_connector()
    results = await connector.fulltext_search(
        query_text=COMMERCIAL_QUERY,
        k=8,
        filters={"brand": TITLECASE_BRAND},
    )
    assert results, "sparse leg returned ZERO rows for a commercial query"
    from_episodic = [r for r in results if r.metadata.get("source_name") == "episodic_memories"]
    assert from_episodic, (
        "sparse leg surfaced no episodic_memories rows (operational corpus not "
        "indexed in hybrid_fulltext_search); sources="
        + ", ".join(sorted({str(r.metadata.get("source_name")) for r in results}))
    )
    assert any(_mentions_corpus(r.content) for r in from_episodic), (
        "sparse episodic rows did not include the queried corpus; "
        "top=" + " | ".join(r.content[:60] for r in from_episodic[:5])
    )


@pytest.mark.skipif(
    os.getenv("E2I_RUN_LIVE_RAG") != "1",
    reason="droplet-only faithful gate; set E2I_RUN_LIVE_RAG=1 to run",
)
@pytest.mark.asyncio
async def test_corpus_appears_in_both_legs_and_wins_fusion():
    """F4: the corpus row appears in BOTH legs and wins the fused ranking.

    Cross-list membership is the precondition for RRF reinforcement; the hybrid
    top result being a corpus KPI row (not [PROC] junk) is the user-visible
    outcome the audit said was broken.
    """
    connector = get_memory_connector()
    dense = await connector.vector_search_by_text(
        query_text=COMMERCIAL_QUERY,
        k=8,
        filters={"brand": TITLECASE_BRAND},
        min_similarity=EFFECTIVE_FLOOR,
    )
    sparse = await connector.fulltext_search(
        query_text=COMMERCIAL_QUERY,
        k=8,
        filters={"brand": TITLECASE_BRAND},
    )
    dense_corpus = {r.content for r in dense if _mentions_corpus(r.content)}
    sparse_corpus = {r.content for r in sparse if _mentions_corpus(r.content)}
    assert dense_corpus & sparse_corpus, (
        "no corpus row appears in BOTH dense and sparse legs -> RRF cannot "
        "reinforce; dense_corpus="
        f"{len(dense_corpus)} sparse_corpus={len(sparse_corpus)}"
    )

    fused = await hybrid_search(
        query=COMMERCIAL_QUERY,
        k=5,
        entities=[TITLECASE_BRAND, "Northeast", "TRx"],
        filters={"brand": TITLECASE_BRAND},
    )
    assert fused, "hybrid_search returned no fused results"
    assert _mentions_corpus(fused[0].content), (
        "fused top result is not an operational corpus row (pipeline still "
        f"degenerate); top={fused[0].content[:80]!r}"
    )
