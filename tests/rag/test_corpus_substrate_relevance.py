"""Faithful corpus-substrate relevance gate (audit F3 / Phase 0 disproof).

Proves the reachable vector corpus contains operational-analytics content a
user actually asks about. Calls the REAL hybrid_vector_search RPC via the
memory connector (no mock of the unit under test).

DROPLET-ONLY MANUAL GATE: opt-in via E2I_RUN_LIVE_RAG=1. No CI lane collects
tests/rag/, and every CI lane sets SUPABASE_URL=http://localhost:54321 (a dead
localhost), so a presence-of-SUPABASE_URL guard would FAIL in CI. The explicit
opt-in flag (precedent: E2I_RUN_REAL_LLM_E2E) keeps it inert everywhere except
the droplet, which has live backends + the prod embedding key (the faithful
environment for this gate).

Decision A (resolved by live-DB inspection): the corpus source is the REAL
`business_metrics` KPI fact table (4667 rows: TRx/NBRx/... per brand/region/
period). The query tokens below are all REAL values present in that table
(brand Kisqali, KPI TRx, region northeast).

CONTINGENT-GREEN NOTE: this gate measures whether the corpus is POPULATED with
relevant KPI rows. It is GREEN only while the substrate actually contains them
(the Phase-0 spike slice, or the Phase-5 durable ingestion). On a cold/emptied
corpus it returns RED ("dense corpus returned ZERO rows"). That is by design --
it validates the substrate premise, not a code-only invariant. Phase 5 carries
its own multi-archetype gate over the full corpus.
"""

import os

import pytest

from src.rag.memory_connector import get_memory_connector

COMMERCIAL_QUERY = "TRx trend for Kisqali in the Northeast this quarter"
# A relevant hit must mention at least one of the queried domain tokens. All
# three are REAL values present in business_metrics (no invented vocabulary).
RELEVANT_TOKENS = ("kisqali", "trx", "northeast")
# 0.5 is the EFFECTIVE floor: hybrid_vector_search hardcodes
# `(1 - (embedding <=> q)) > 0.5` (011_...sql:126) AND vector_search drops
# similarity < min_similarity (memory_connector.py:152, default 0.5). Passing
# 0.3 would be misleading -- the SQL already removed those rows.
EFFECTIVE_FLOOR = 0.5


@pytest.mark.skipif(
    os.getenv("E2I_RUN_LIVE_RAG") != "1",
    reason="droplet-only faithful gate; set E2I_RUN_LIVE_RAG=1 to run",
)
@pytest.mark.asyncio
async def test_dense_corpus_returns_topically_relevant_row():
    connector = get_memory_connector()
    results = await connector.vector_search_by_text(
        query_text=COMMERCIAL_QUERY, k=8, min_similarity=EFFECTIVE_FLOOR
    )
    assert results, "dense corpus returned ZERO rows for a commercial query"
    blob = " ".join(r.content.lower() for r in results)
    assert any(tok in blob for tok in RELEVANT_TOKENS), (
        "no retrieved row mentions any of "
        f"{RELEVANT_TOKENS}; top contents=" + " | ".join(r.content[:60] for r in results[:5])
    )
