"""Durable ingestion of the operational-analytics corpus into the RAG substrate.

Audit F3 remediation (Phase 5): the operational KPI corpus EXISTS as real rows
in the ``business_metrics`` fact table (TRx/NBRx/Market-Share/HCP-Engagement per
brand/region/period) but is NEVER indexed into the dense retrieval path. The
chatbot's ``hybrid_vector_search`` RPC reads ``episodic_memories`` +
``procedural_memories`` only. This module renders REAL ``business_metrics`` rows
as analytic prose (values taken VERBATIM from the fact table -- no invented KPI
numbers, F3 anti-mocking) and indexes them via the existing auto-embed path
(1536-dim, prod provider) into ``episodic_memories``, where the chatbot already
reads them.

brand/region are carried via ``E2IEntityReferences`` so they land on the
``episodic_memories.brand``/``.region`` columns the chatbot brand-filter matches
on (``chatbot_dspy`` passes ``filters={'brand': brand_context}`` ->
``011_…sql`` enforces ``filters->>'brand' IS NULL OR em.brand = filters->>'brand'``).

Corpus rows are attributed to ``agent_name='corpus_ingestion'`` (added in
migration ``database/memory/041``) so they are distinguishable from agent
bookkeeping and selectively removable/re-syncable.
"""

import logging
import uuid
from typing import Any, Optional

from src.memory.episodic_memory import (
    E2IEntityReferences,
    EpisodicMemoryInput,
    insert_episodic_memory_with_text,
)
from src.memory.services.factories import get_supabase_client

logger = logging.getLogger(__name__)

# 'system_event' is a valid memory_event_type. A dedicated 'kpi_snapshot' value
# would need an additive enum migration (precedent 039) -- optional, see the
# Phase-0 owner-decision doc.
_DEFAULT_EVENT_TYPE = "system_event"
# 'corpus_ingestion' is added in migration database/memory/041 (e2i_agent_name).
_DEFAULT_AGENT_NAME = "corpus_ingestion"

# Columns read from business_metrics for faithful rendering.
_METRIC_COLUMNS = (
    "metric_name,brand,region,metric_date,value,target,achievement_rate,year_over_year_change,roi"
)


def render_business_metric(row: dict[str, Any]) -> str:
    """Render a REAL ``business_metrics`` row as analytic prose.

    Every value is taken verbatim from the fact table -- no fabrication. Returns
    a sentence like::

        TRx for Kisqali in the northeast on 2025-01-01: value 684127.3,
        target 736992.12, achievement 92.8%, year-over-year +18.7%, ROI 4.09.
    """
    name = row.get("metric_name")
    brand = row.get("brand")
    region = row.get("region")
    date = row.get("metric_date")
    value = row.get("value")
    target = row.get("target")
    ach = row.get("achievement_rate")
    yoy = row.get("year_over_year_change")
    roi = row.get("roi")
    parts = [f"{name} for {brand} in the {region} on {date}: value {value}"]
    if target is not None:
        parts.append(f"target {target}")
    if ach is not None:
        parts.append(f"achievement {float(ach) * 100:.1f}%")
    if yoy is not None:
        parts.append(f"year-over-year {float(yoy) * 100:+.1f}%")
    if roi is not None:
        parts.append(f"ROI {roi}")
    return ", ".join(parts) + "."


async def index_operational_corpus(
    rows: list[tuple[str, Optional[str], Optional[str]]],
    *,
    event_type: str = _DEFAULT_EVENT_TYPE,
    agent_name: str = _DEFAULT_AGENT_NAME,
    session_id: Optional[str] = None,
) -> list[str]:
    """Index operational-analytics ``(text, brand, region)`` rows; return ids.

    Args:
        rows: ``(text, brand, region)`` tuples. ``text`` is REAL operational
            prose (e.g. from :func:`render_business_metric`) -- NEVER invented
            numbers (F3 anti-mocking). brand/region land on the
            ``episodic_memories.brand``/``.region`` columns via ``e2i_refs``.
        event_type: a VALID ``memory_event_type`` value (default 'system_event').
        agent_name: a VALID ``e2i_agent_name`` value (default 'corpus_ingestion',
            added in migration 041).
        session_id: optional shared session id (one is generated if omitted) so
            a batch is traceable/removable.

    Returns:
        The inserted episodic ``memory_id``s.
    """
    sid = session_id or str(uuid.uuid4())
    inserted: list[str] = []
    for text, brand, region in rows:
        mem = EpisodicMemoryInput(
            event_type=event_type,
            description=text,
            agent_name=agent_name,
            e2i_refs=E2IEntityReferences(brand=brand, region=region),
        )
        mid = await insert_episodic_memory_with_text(memory=mem, text_to_embed=text, session_id=sid)
        inserted.append(mid)
    logger.info("indexed %d operational-corpus rows (session=%s)", len(inserted), sid)
    return inserted


def _latest_per_combo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest row per (metric_name, region) from a date-DESC stream.

    The live ``business_metrics`` table is irregular: the most-recent date carries
    only a subset of (metric_name, region) combos, so a naive ``limit_per_brand``
    (order-by-date-desc, take N) silently omits combos (e.g. Remibrutinib and the
    ``west`` region were entirely absent from the corpus). Given rows already
    ordered by ``metric_date`` DESC, the FIRST occurrence of each combo is its
    latest snapshot; later (older) duplicates are dropped. Guarantees every combo
    present exactly once.
    """
    seen: set[tuple[Any, Any]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("metric_name"), row.get("region"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _fetch_brand_rows(
    sb: Any, brand: str, limit_per_brand: int, latest_per_combo: bool
) -> list[dict[str, Any]]:
    """Fetch the business_metrics rows to index for one brand.

    ``latest_per_combo=True`` paginates ALL of the brand's rows (date-DESC) and
    returns the latest snapshot of every (metric_name, region) combo -- full
    coverage, no omitted combos (audit F3b). ``False`` keeps the legacy bounded
    recent-N behavior.
    """
    # metric_date DESC drives latest-per-combo; metric_id (PK) is a deterministic
    # secondary key so offset pagination over same-date rows can neither skip nor
    # duplicate a row across page boundaries (codex MED) -> _latest_per_combo sees
    # a stable total order and cannot miss a combo.
    from src.repositories.provenance import apply_provenance_filter

    base = (
        sb.table("business_metrics")
        .select(_METRIC_COLUMNS)
        .eq("brand", brand)
        .not_.is_("metric_name", "null")
        .order("metric_date", desc=True)
        .order("metric_id")
    )
    # Provenance (Shard 07 R12): exclude synthetic KPI prose from the prod corpus in
    # real mode; a synthetic-gold showcase instance (E2I_INCLUDE_SYNTHETIC) includes
    # it so the chatbot RAG corpus is populated rather than empty. (WS-SYNTH)
    base = apply_provenance_filter(base)
    if not latest_per_combo:
        return list(base.limit(limit_per_brand).execute().data or [])

    all_rows: list[dict[str, Any]] = []
    page = 0
    page_size = 1000
    while True:
        batch = base.range(page * page_size, page * page_size + page_size - 1).execute().data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return _latest_per_combo(all_rows)


def _discover_brands(sb: Any) -> list[str]:
    """Return the sorted set of brands present in ``business_metrics``.

    Shared by the episodic corpus (:func:`index_business_metrics`) and the
    chat-RAG chunk corpus (``src/rag/chunk_corpus_ingestion.py``). Real-mode
    discovery excludes synthetic-only brands via ``apply_provenance_filter``; a
    synthetic-gold showcase instance (``E2I_INCLUDE_SYNTHETIC``) includes them so
    every synthetic-gold brand is discovered for indexing. (WS-SYNTH)
    """
    from src.repositories.provenance import apply_provenance_filter

    brand_q = sb.table("business_metrics").select("brand").not_.is_("brand", "null")
    brand_q = apply_provenance_filter(brand_q)
    r = brand_q.execute()
    return sorted({row["brand"] for row in (r.data or []) if row.get("brand")})


def _existing_corpus_descriptions(sb: Any, agent_name: str) -> set[str]:
    """Return the set of already-indexed corpus descriptions for idempotency.

    Best-effort dedup: paginate the corpus rows for ``agent_name`` and collect
    their ``description`` text so a re-run does not re-embed/re-insert identical
    prose. (For a very large corpus a stable content-hash column would be more
    efficient than fetching descriptions; this bounded paginate suffices at the
    current fact-table scale.)
    """
    seen: set[str] = set()
    page = 0
    page_size = 1000
    while True:
        from src.repositories.provenance import apply_provenance_filter

        dedup_q = sb.table("episodic_memories").select("description").eq("agent_name", agent_name)
        # Provenance (Shard 07 R12/R15): dedup against REAL corpus rows in real mode;
        # the showcase instance (E2I_INCLUDE_SYNTHETIC) dedups against the synthetic
        # corpus too so re-ingest stays idempotent on synthetic-gold data. (WS-SYNTH)
        dedup_q = apply_provenance_filter(dedup_q)
        resp = dedup_q.range(page * page_size, page * page_size + page_size - 1).execute()
        batch = resp.data or []
        for row in batch:
            if row.get("description"):
                seen.add(row["description"])
        if len(batch) < page_size:
            break
        page += 1
    return seen


async def index_business_metrics(
    *,
    brands: Optional[list[str]] = None,
    limit_per_brand: int = 50,
    supabase_client: Any = None,
    agent_name: str = _DEFAULT_AGENT_NAME,
    dedup: bool = True,
    latest_per_combo: bool = False,
) -> list[str]:
    """Read REAL ``business_metrics`` rows and index them into the RAG substrate.

    This is the production entry point: it reads the fact table directly (no
    hand-authored corpus), renders each row faithfully, and indexes via
    :func:`index_operational_corpus`. ``brand``/``region`` are written lowercased
    on the episodic row (the form the chatbot brand-filter uses) while the prose
    keeps the fact table's display casing.

    Idempotent by default: rows whose rendered prose is already indexed under
    ``agent_name`` are skipped, so a re-run (or a scheduled re-sync) does not
    accumulate duplicate corpus rows. Pass ``dedup=False`` to force re-insertion.

    Args:
        brands: restrict to these brands (defaults to all brands present).
        limit_per_brand: cap rows per brand (the full corpus is large; bound the
            embedding cost). Pass a large value for a full run.
        supabase_client: optional injected client (defaults to the prod client).
        agent_name: e2i_agent_name attribution (default 'corpus_ingestion').
        dedup: skip rows already indexed under ``agent_name`` (default True).
        latest_per_combo: when True, index the latest snapshot of EVERY
            (metric_name, region) combo per brand (full coverage; ``limit_per_brand``
            ignored). The durable Celery sync uses this so no combo is omitted
            (audit F3b). When False, the legacy bounded recent-N behavior.

    Returns:
        The inserted episodic ``memory_id``s (only the NEW rows when ``dedup``).
    """
    sb = supabase_client or get_supabase_client()
    if brands is None:
        brands = _discover_brands(sb)

    already = _existing_corpus_descriptions(sb, agent_name) if dedup else set()

    rows: list[tuple[str, Optional[str], Optional[str]]] = []
    batch_seen: set[str] = set()
    skipped = 0
    for brand in brands:
        for row in _fetch_brand_rows(sb, brand, limit_per_brand, latest_per_combo):
            text = render_business_metric(row)
            # Skip already-indexed prose and intra-batch duplicates (idempotency).
            if text in already or text in batch_seen:
                skipped += 1
                continue
            batch_seen.add(text)
            row_brand = (row.get("brand") or "").lower() or None
            row_region = (row.get("region") or "").lower() or None
            rows.append((text, row_brand, row_region))

    if skipped:
        logger.info("index_business_metrics: skipped %d already-indexed/duplicate rows", skipped)
    if not rows:
        logger.info("index_business_metrics: nothing new to index (corpus up to date)")
        return []
    return await index_operational_corpus(rows, agent_name=agent_name)
