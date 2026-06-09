"""Faithful full-coverage corpus snapshot gate (audit F3b).

Runs the REAL durable snapshot ingestion (index_business_metrics with
latest_per_combo=True reads the live business_metrics fact table) and asserts the
corpus now covers EVERY (brand, region) — including combos a naive recent-N
ingestion omitted entirely (Remibrutinib brand, `west` region). No mock; the only
externals are live Supabase + the prod OpenAI embedder. Idempotent: a warm corpus
inserts nothing new and the coverage assertions still hold.

DROPLET-ONLY MANUAL GATE: opt-in via E2I_RUN_LIVE_RAG=1.
"""

import os

import pytest

from src.memory.services.factories import get_supabase_client
from src.rag.corpus_ingestion import index_business_metrics

AGENT = "corpus_ingestion"


@pytest.mark.skipif(
    os.getenv("E2I_RUN_LIVE_RAG") != "1",
    reason="droplet-only faithful gate; set E2I_RUN_LIVE_RAG=1 to run",
)
@pytest.mark.asyncio
async def test_snapshot_ingestion_covers_every_brand_region_combo():
    # Faithful durable ingestion of the latest snapshot per (brand, metric, region).
    await index_business_metrics(latest_per_combo=True)

    sb = get_supabase_client()
    # All brands present in business_metrics must be present in the corpus.
    bm_brands = {
        (r["brand"] or "").lower()
        for r in (
            sb.table("business_metrics").select("brand").not_.is_("brand", "null").execute().data
            or []
        )
        if r.get("brand")
    }
    corpus_rows = (
        sb.table("episodic_memories").select("brand,region").eq("agent_name", AGENT).execute().data
        or []
    )
    corpus_brands = {(r.get("brand") or "") for r in corpus_rows}
    corpus_regions = {(r.get("region") or "") for r in corpus_rows}

    missing_brands = bm_brands - corpus_brands
    assert not missing_brands, f"corpus omits brands present in business_metrics: {missing_brands}"
    # `west` was entirely absent before F3b; full coverage must include all 4 regions.
    assert {"northeast", "south", "midwest", "west"} <= corpus_regions, (
        f"corpus missing regions; have={sorted(corpus_regions)}"
    )
    # Remibrutinib was entirely absent before F3b.
    assert "remibrutinib" in corpus_brands, "Remibrutinib snapshot still missing from corpus"
