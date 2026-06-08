"""Phase-0 spike: index a SMALL REAL slice of the operational KPI corpus into the
table the chatbot's hybrid_vector_search already reads (episodic_memories), using
the EXISTING auto-embed path (1536-dim, same provider as prod).

GATE EXPERIMENT, not full production ingestion (that is Phase 5). Rows are tagged
agent_name='phase0_corpus_spike' so they can be removed if the owner picks a
different target table.

Decision A (resolved by live-DB inspection, NOT guessed): the corpus source is
the REAL `business_metrics` KPI fact table (4667 rows: TRx/NBRx/Conversion_Rate/
ROI/... per brand/region/period). This script READS that table directly and
renders each REAL row as prose -- NO invented KPI numbers (F3 anti-mocking: every
value comes verbatim from the fact table). brand/region are carried via e2i_refs
(the ONLY way to land them on the episodic_memories.brand/.region columns the
chatbot brand-filter matches on).

Run on the droplet (faithful live Supabase + real embedding provider):

    PYTHONPATH=. .venv/bin/dotenv run -- .venv/bin/python \\
      docs/reports/rag-hybrid-audit-20260608-repro/spike_index_small_slice.py
"""

import asyncio
import uuid

from src.memory.episodic_memory import (
    E2IEntityReferences,
    EpisodicMemoryInput,
    insert_episodic_memory_with_text,
)
from src.memory.services.factories import get_supabase_client

# Brands to include in the spike slice (real values present in business_metrics).
_SPIKE_BRANDS = ("Kisqali", "Fabhalta")
_ROWS_PER_BRAND = 10

# agent_name is the e2i_agent_name ENUM (not free text -- the faithful live insert
# proved this: an arbitrary value raises Postgres 22P02). 'observability_connector'
# is the valid enum value semantically closest to operational KPI facts. Phase 5's
# durable path should add a dedicated 'corpus_ingestion' value via an additive enum
# migration (precedent 029/039) for clean attribution. Spike rows are removable by
# the printed session_id.
_SPIKE_AGENT_NAME = "observability_connector"


def _render(row: dict) -> str:
    """Render a REAL business_metrics row as analytic prose. Every value is taken
    verbatim from the fact table -- no fabrication."""
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


def _fetch_real_rows() -> list[dict]:
    sb = get_supabase_client()
    rows: list[dict] = []
    for brand in _SPIKE_BRANDS:
        r = (
            sb.table("business_metrics")
            .select(
                "metric_name,brand,region,metric_date,value,target,"
                "achievement_rate,year_over_year_change,roi"
            )
            .eq("brand", brand)
            .not_.is_("metric_name", "null")
            .order("metric_date")
            .limit(_ROWS_PER_BRAND)
            .execute()
        )
        rows.extend(r.data or [])
    return rows


async def main() -> None:
    rows = _fetch_real_rows()
    if not rows:
        raise SystemExit("business_metrics returned no rows -- cannot run the faithful spike.")
    session_id = str(uuid.uuid4())
    inserted = []
    for row in rows:
        text = _render(row)
        mem = EpisodicMemoryInput(
            # 'system_event' is a REAL memory_event_type value. A dedicated
            # 'kpi_snapshot' value would need an additive enum migration
            # (precedent 039) -- a Phase-5 decision, not needed to disprove here.
            event_type="system_event",
            description=text,
            agent_name=_SPIKE_AGENT_NAME,
            # brand/region land on episodic_memories.brand/.region ONLY via
            # e2i_refs -- required so the chatbot's filters={'brand': ...} query
            # can match these rows.
            e2i_refs=E2IEntityReferences(
                brand=(row.get("brand") or "").lower() or None,
                region=row.get("region"),
            ),
        )
        mid = await insert_episodic_memory_with_text(
            memory=mem, text_to_embed=text, session_id=session_id
        )
        inserted.append(mid)
        print(f"  indexed: {text[:90]}")
    print(f"\nindexed {len(inserted)} REAL corpus-spike rows; session_id={session_id}")


if __name__ == "__main__":
    asyncio.run(main())
