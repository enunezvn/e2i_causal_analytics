#!/usr/bin/env python3
"""Populate the chat-RAG chunk corpus (``rag_document_chunks``) -- #1373 entrypoint.

Renders REAL ``business_metrics`` rows as prose (values VERBATIM from the fact
table) and indexes them into ``rag_document_chunks`` embedded in the
``text-embedding-3-small`` space the chat ``HybridRetriever`` queries in. Run
in-container against prod AFTER merge + deploy (the container carries
SUPABASE_* + OPENAI_API_KEY). Idempotent: a re-run embeds only new/changed
snapshots (content-hash dedup).

Examples (run inside the API/worker container):
    # Smoke: index the latest snapshot for a couple of brands only.
    python -m scripts.rag.ingest_chunk_corpus --brands Kisqali --limit 10

    # Full production sync (every brand, latest snapshot of every combo).
    python -m scripts.rag.ingest_chunk_corpus --latest-per-combo

    # Just report the current corpus size (no writes, no embedding spend).
    python -m scripts.rag.ingest_chunk_corpus --health-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from src.rag.chunk_corpus_ingestion import chunk_corpus_health, index_business_metric_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest_chunk_corpus")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--brands",
        nargs="*",
        default=None,
        help="Restrict to these brands (default: all brands in business_metrics).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Rows per brand when NOT --latest-per-combo (smoke flag).",
    )
    p.add_argument(
        "--latest-per-combo",
        action="store_true",
        help="Index the latest snapshot of EVERY (metric, region) combo per brand "
        "(full coverage; --limit ignored). Use for a full production sync.",
    )
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="Re-embed + upsert even prose already indexed (default: dedup by content_hash).",
    )
    p.add_argument(
        "--health-only",
        action="store_true",
        help="Only report the current chunk-corpus row count; no writes.",
    )
    return p.parse_args()


async def _run(args: argparse.Namespace) -> int:
    before = await chunk_corpus_health()
    logger.info("chunk corpus size BEFORE: %s", before)
    if args.health_only:
        return 0

    document_ids = await index_business_metric_chunks(
        brands=args.brands,
        limit_per_brand=args.limit,
        dedup=not args.no_dedup,
        latest_per_combo=args.latest_per_combo,
    )
    logger.info("indexed %d new/changed chunks", len(document_ids))

    after = await chunk_corpus_health()
    logger.info("chunk corpus size AFTER: %s", after)
    return 0


def main() -> int:
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
