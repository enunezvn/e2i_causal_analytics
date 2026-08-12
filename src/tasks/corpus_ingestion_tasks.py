"""Durable scheduled sync of the operational KPI corpus into the RAG substrate.

Audit F3b: the operational corpus (business_metrics rendered as prose, indexed
into episodic_memories by src/rag/corpus_ingestion.py) was populated ONCE by a
manual run and had no scheduler, so it could not pick up new facts and silently
omitted whole combos. This Celery beat task re-syncs the LATEST snapshot of every
(brand, metric_name, region) combo on a schedule. It is idempotent (prose dedup),
so a daily run only embeds NEW/changed snapshots. Scheduled after the
business_metrics ETL rollups (celery_app beat_schedule) so it sees fresh facts.
"""

import asyncio
import logging
from typing import Any, Optional

from src.rag.chunk_corpus_ingestion import index_business_metric_chunks
from src.rag.corpus_ingestion import index_business_metrics
from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="src.tasks.sync_operational_corpus")
def sync_operational_corpus(self, brands: Optional[list[str]] = None) -> dict[str, Any]:
    """Index the latest snapshot of every (brand, metric, region) KPI combo.

    Also reconciles the existing corpus against the current fact table (#1552,
    default ``reconcile=True`` inside :func:`index_business_metrics`): prose
    matching no current ``business_metrics`` row (stale, e.g. values the frozen
    substrate no longer attributes to that date) is deleted, and value-valid
    prose in the pre-grain-label template is re-indexed under the labeled one.

    Args:
        brands: optional restriction; defaults to all brands in business_metrics.

    Returns:
        ``{"indexed": <n_new_rows>, "brands": <brands|"all">}``.
    """
    inserted = asyncio.run(index_business_metrics(latest_per_combo=True, brands=brands))
    logger.info(
        "sync_operational_corpus: indexed %d new corpus rows (brands=%s)",
        len(inserted),
        brands or "all",
    )
    return {"indexed": len(inserted), "brands": brands or "all"}


@celery_app.task(bind=True, name="src.tasks.sync_chunk_corpus")
def sync_chunk_corpus(self, brands: Optional[list[str]] = None) -> dict[str, Any]:
    """Sync the chat-RAG chunk corpus (``rag_document_chunks``) (#1373).

    Mirrors :func:`sync_operational_corpus` but targets the OTHER RAG substrate:
    the ``text-embedding-3-small`` chunk table the chat ``HybridRetriever`` reads
    (the episodic corpus is embedded in the memory-path ada-002 space and is
    invisible to chat queries). Indexes the latest snapshot of every (brand,
    metric, region) combo; idempotent (content-hash dedup) so a daily run only
    embeds new/changed snapshots. Scheduled after the business_metrics ETL so it
    sees fresh facts.

    Args:
        brands: optional restriction; defaults to all brands in business_metrics.

    Returns:
        ``{"indexed": <n_new_or_changed_chunks>, "brands": <brands|"all">}``.
    """
    # Plain asyncio.run (no nest_asyncio.apply) -- same pattern as
    # sync_operational_corpus above; does NOT add a nest_asyncio callsite, so the
    # tests/integration/test_no_unconditional_nest_asyncio_apply.py pin is unaffected.
    inserted = asyncio.run(index_business_metric_chunks(latest_per_combo=True, brands=brands))
    logger.info(
        "sync_chunk_corpus: indexed %d new/changed chunks (brands=%s)",
        len(inserted),
        brands or "all",
    )
    return {"indexed": len(inserted), "brands": brands or "all"}
