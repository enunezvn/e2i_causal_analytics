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
from datetime import date
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


def _parse_metric_date(value: Any) -> Optional[date]:
    """Parse ``metric_date`` (supabase ISO string or ``datetime.date``)."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _metric_tail_parts(row: dict[str, Any]) -> list[str]:
    """The value/target/achievement/yoy/ROI clauses shared by both templates."""
    parts = [f"value {row.get('value')}"]
    target = row.get("target")
    ach = row.get("achievement_rate")
    yoy = row.get("year_over_year_change")
    roi = row.get("roi")
    if target is not None:
        parts.append(f"target {target}")
    if ach is not None:
        parts.append(f"achievement {float(ach) * 100:.1f}%")
    if yoy is not None:
        parts.append(f"year-over-year {float(yoy) * 100:+.1f}%")
    if roi is not None:
        parts.append(f"ROI {roi}")
    return parts


def _render_business_metric_legacy(row: dict[str, Any]) -> str:
    """The pre-#1552 template ('... on <date>: value ...') — kept ONLY so the
    reconciliation pass can recognize value-valid legacy prose and migrate it
    to the grain-labeled template (see ``_plan_corpus_reconciliation``)."""
    head = (
        f"{row.get('metric_name')} for {row.get('brand')} in the "
        f"{row.get('region')} on {row.get('metric_date')}"
    )
    tail = _metric_tail_parts(row)
    return ", ".join([f"{head}: " + tail[0]] + tail[1:]) + "."


def render_business_metric(row: dict[str, Any]) -> str:
    """Render a REAL ``business_metrics`` row as analytic prose.

    Every value is taken verbatim from the fact table -- no fabrication.

    Period grain (#1552): ``business_metrics`` is monthly-grain — every
    ``metric_date`` sits on a month start (measured 2026-08-12: DISTINCT
    ``metric_date - date_trunc('month', metric_date)`` = 0 across the live
    table). The old '... on 2026-08-01: value ...' form read as a single day
    and let the chat synthesizer invent 2-month and month-to-date buckets in
    one table ("Jun/Jul 2026" next to "Aug 2026") and call the width artifact
    "an unexplained scale discontinuity" (eval 6.5). Month-start rows are now
    labeled with their bucket explicitly::

        trx for Kisqali in the northeast for calendar month 2026-08
        (August 2026, monthly grain): value 48654.99, target 65800.04,
        achievement 73.9%, year-over-year +22.0%, ROI 3.22.

    A row NOT on the month-start lattice (defensive: none exist today) keeps
    the honest '... on <date>' form — never claim monthly grain for a row
    that isn't one.
    """
    metric_date = _parse_metric_date(row.get("metric_date"))
    if metric_date is None or metric_date.day != 1:
        return _render_business_metric_legacy(row)
    head = (
        f"{row.get('metric_name')} for {row.get('brand')} in the {row.get('region')} "
        f"for calendar month {metric_date:%Y-%m} ({metric_date:%B %Y}, monthly grain)"
    )
    tail = _metric_tail_parts(row)
    return ", ".join([f"{head}: " + tail[0]] + tail[1:]) + "."


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
    if latest_per_combo:
        return _latest_per_combo(_fetch_all_brand_rows(sb, brand))
    base = _brand_rows_query(sb, brand)
    return list(base.limit(limit_per_brand).execute().data or [])


def _brand_rows_query(sb: Any, brand: str) -> Any:
    """The provenance-filtered, deterministically-ordered brand rows query."""
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
    return apply_provenance_filter(base)


def _fetch_all_brand_rows(sb: Any, brand: str) -> list[dict[str, Any]]:
    """Paginate EVERY current ``business_metrics`` row for ``brand`` (date-DESC,
    provenance-filtered). Feeds both the latest-per-combo selection and the
    #1552 reconciliation pass (which must recognize prose derived from ANY
    current fact row — not just the latest snapshot — as still valid)."""
    base = _brand_rows_query(sb, brand)
    all_rows: list[dict[str, Any]] = []
    page = 0
    page_size = 1000
    while True:
        batch = base.range(page * page_size, page * page_size + page_size - 1).execute().data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return all_rows


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


def _corpus_rows_for_brand(sb: Any, agent_name: str, brand_lower: str) -> list[dict[str, Any]]:
    """Paginate the existing corpus rows (id + prose) for one brand.

    Scoped to ``agent_name`` (this module's attribution, migration 041) and the
    lowercased ``brand`` column its inserts stamp — the reconciliation pass
    must never see (or delete) rows this module did not write. Provenance
    parity with the dedup read (real mode touches real rows only).
    """
    from src.repositories.provenance import apply_provenance_filter

    rows: list[dict[str, Any]] = []
    page = 0
    page_size = 1000
    while True:
        q = (
            sb.table("episodic_memories")
            .select("memory_id,description")
            .eq("agent_name", agent_name)
            .eq("brand", brand_lower)
        )
        q = apply_provenance_filter(q)
        resp = q.range(page * page_size, page * page_size + page_size - 1).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return rows


def _plan_corpus_reconciliation(
    sb: Any, agent_name: str, brand: str, all_rows: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Classify one brand's existing corpus prose against the CURRENT facts.

    #1552: the corpus dedup is insert-only, so prose ingested from an earlier
    substrate state can silently contradict the current fact table (measured:
    a row attributing Jul-2026's Kisqali northeast TRx values to Jun-2026 sat
    next to the valid Jul row; the chat merged the byte-identical pair into an
    invented "Jun/Jul 2026" bucket and called the resulting scale gap an
    unexplained discontinuity — eval 6.5).

    Returns ``(stale_memory_ids, migrate_rows)``:

    * a corpus description matching a current fact row under the CURRENT
      template is valid history — kept;
    * one matching a current fact row only under the LEGACY (pre-grain-label)
      template is value-valid but unlabeled — deleted AND its fact row queued
      for re-indexing under the labeled template (``migrate_rows``);
    * anything else matches NO current fact row — stale — deleted.
    """
    valid_new: set[str] = set()
    legacy_map: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        valid_new.add(render_business_metric(row))
        legacy_map[_render_business_metric_legacy(row)] = row

    stale_ids: list[str] = []
    migrate_rows: list[dict[str, Any]] = []
    for crow in _corpus_rows_for_brand(sb, agent_name, (brand or "").lower()):
        desc = crow.get("description")
        mid = crow.get("memory_id")
        if desc in valid_new or not mid:
            continue
        stale_ids.append(mid)
        if desc in legacy_map:
            migrate_rows.append(legacy_map[desc])
    return stale_ids, migrate_rows


def _delete_corpus_rows(sb: Any, agent_name: str, memory_ids: list[str]) -> None:
    """Delete superseded corpus rows in bounded batches (agent-scoped)."""
    batch_size = 100
    for i in range(0, len(memory_ids), batch_size):
        (
            sb.table("episodic_memories")
            .delete()
            .eq("agent_name", agent_name)
            .in_("memory_id", memory_ids[i : i + batch_size])
            .execute()
        )


async def index_business_metrics(
    *,
    brands: Optional[list[str]] = None,
    limit_per_brand: int = 50,
    supabase_client: Any = None,
    agent_name: str = _DEFAULT_AGENT_NAME,
    dedup: bool = True,
    latest_per_combo: bool = False,
    reconcile: bool = True,
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
        reconcile: when True (default) AND ``latest_per_combo`` is True, run the
            #1552 reconciliation pass: corpus prose matching NO current fact row
            (stale, e.g. values the substrate no longer attributes to that date)
            is deleted, and value-valid prose in the legacy unlabeled template is
            re-indexed under the grain-labeled template. Requires the full fact
            scan, so the bounded path (``latest_per_combo=False``) NEVER
            reconciles — deleting against a partial scan would drop valid prose.

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
    stale_all: list[str] = []
    for brand in brands:
        if latest_per_combo:
            all_rows = _fetch_all_brand_rows(sb, brand)
            fetched: list[dict[str, Any]] = _latest_per_combo(all_rows)
            if reconcile and all_rows:
                # Empty-scan guard: a brand whose (provenance-filtered) fact scan
                # returns 0 rows is skipped — an env/provenance misconfiguration
                # (e.g. a showcase host missing E2I_INCLUDE_SYNTHETIC) must not
                # mass-delete an otherwise valid corpus.
                stale_ids, migrate_rows = _plan_corpus_reconciliation(
                    sb, agent_name, brand, all_rows
                )
                stale_all.extend(stale_ids)
                fetched = fetched + migrate_rows
        else:
            fetched = _fetch_brand_rows(sb, brand, limit_per_brand, latest_per_combo)
        for row in fetched:
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
    inserted: list[str] = []
    if rows:
        inserted = await index_operational_corpus(rows, agent_name=agent_name)
    if stale_all:
        # Insert-then-delete: a failure between the two leaves BOTH generations
        # present until the next idempotent sync — never a corpus with valid
        # prose missing.
        _delete_corpus_rows(sb, agent_name, stale_all)
        logger.info(
            "index_business_metrics: reconciled %d stale/superseded corpus rows (#1552)",
            len(stale_all),
        )
    if not rows and not stale_all:
        logger.info("index_business_metrics: nothing new to index (corpus up to date)")
    return inserted
