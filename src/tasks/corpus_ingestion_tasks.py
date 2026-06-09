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

from src.rag.corpus_ingestion import index_business_metrics
from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="src.tasks.sync_operational_corpus")
def sync_operational_corpus(self, brands: Optional[list[str]] = None) -> dict[str, Any]:
    """Index the latest snapshot of every (brand, metric, region) KPI combo.

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
