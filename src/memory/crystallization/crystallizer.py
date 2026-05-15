"""
Executive insight crystallizer.

Aggregates 2+ related episodic memories (from different agents, same brand,
within a 7-day window, on the same causal_path or KPI) into a single
durable ``executive_insights`` row, plus ``insight_edges`` rows linking
back to every source.

The crystallizer is brand-strict: it NEVER co-aggregates across brands.
Aggregation keys: (brand, region, kpi, time_window).

v1: narrative composition is deterministic concatenation of source
executive_summary fields with a structured preamble. A DSPy-based
generator can be plugged in later behind the same interface — for v1
we want deterministic output so JIT provenance verification is
unambiguous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.memory.services.factories import get_supabase_client

logger = logging.getLogger(__name__)


DEFAULT_WINDOW_DAYS = 7
DEFAULT_MIN_AGENTS = 2  # require findings from at least 2 distinct agents


@dataclass
class CrystallizerResult:
    """Summary of a crystallizer run."""

    examined_groups: int = 0
    insights_created: int = 0
    edges_created: int = 0
    by_brand: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None


class Crystallizer:
    """
    Crystallizes episodic findings into executive_insights.

    Public entrypoint: ``run_for_brand(brand, region=None, window_days=7)``.
    The consolidator (subsystem 2) is the natural caller — after promoting
    a causal_path to semantic, it can trigger crystallization for that
    brand. Can also be invoked on a schedule.
    """

    def __init__(
        self,
        window_days: int = DEFAULT_WINDOW_DAYS,
        min_agents: int = DEFAULT_MIN_AGENTS,
    ):
        self.window_days = window_days
        self.min_agents = min_agents

    async def run_for_brand(
        self,
        brand: str,
        *,
        region: Optional[str] = None,
        crystallized_by_cycle_id: Optional[str] = None,
        crystallized_by_user_id: Optional[str] = None,
    ) -> CrystallizerResult:
        """
        Look for crystallization candidates within ``brand`` and create
        executive_insights for any group meeting the threshold.

        Brand is required and never collapsed across brands.
        """
        if not brand:
            raise ValueError("crystallize requires brand")

        result = CrystallizerResult()
        client = get_supabase_client()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.window_days)).isoformat()
        # Pull candidate episodic memories: same brand, recent, completed agent actions.
        query = (
            client.table("episodic_memories")
            .select(
                "memory_id, agent_name, brand, region, causal_path_id, "
                "event_type, description, outcome_type, occurred_at, raw_content"
            )
            .eq("brand", brand)
            .gte("occurred_at", cutoff)
            .in_("event_type", ["agent_action", "causal_discovery", "experiment_completed"])
        )
        if region:
            query = query.eq("region", region)
        memories = (query.execute().data) or []

        # Group by (causal_path_id OR fallback grouping key). For v1 we use
        # causal_path_id as the strongest signal of "these are talking about
        # the same finding"; fall back to (brand, region, event_subtype-ish).
        groups = _group_memories(memories)
        result.examined_groups = len(groups)

        for group_key, members in groups.items():
            distinct_agents = {m.get("agent_name") for m in members if m.get("agent_name")}
            if len(distinct_agents) < self.min_agents:
                continue
            try:
                insight_id, edges_added = await self._crystallize_group(
                    brand=brand,
                    region=region,
                    group_key=group_key,
                    members=members,
                    crystallized_by_cycle_id=crystallized_by_cycle_id,
                    crystallized_by_user_id=crystallized_by_user_id,
                )
                result.insights_created += 1
                result.edges_created += edges_added
                result.by_brand[brand] = result.by_brand.get(brand, 0) + 1
                logger.info(
                    f"crystallizer: created executive_insight {insight_id} "
                    f"brand={brand} group={group_key} agents={distinct_agents}"
                )
            except Exception as exc:
                logger.exception(f"crystallizer: failed for group {group_key}")
                result.errors.append(f"{group_key}: {exc}")

        result.finished_at = datetime.now(timezone.utc)
        return result

    # ----------------------------------------------------------- one group

    async def _crystallize_group(
        self,
        *,
        brand: str,
        region: Optional[str],
        group_key: str,
        members: List[Dict[str, Any]],
        crystallized_by_cycle_id: Optional[str],
        crystallized_by_user_id: Optional[str],
    ) -> Tuple[str, int]:  # (insight_id, edges_added)
        client = get_supabase_client()

        title, narrative, key_metrics = _compose_narrative(brand, region, members)
        # Time window: earliest..latest occurred_at of the source memories.
        times = sorted(t for t in (m.get("occurred_at") for m in members) if t is not None)
        time_start = times[0] if times else None
        time_end = times[-1] if times else None

        # Insert executive_insight row.
        insert = (
            client.table("executive_insights")
            .insert(
                {
                    "title": title[:500],
                    "narrative": narrative,
                    "brand": brand,
                    "region": region,
                    "kpi": _pick_kpi(members),
                    "time_window_start": time_start,
                    "time_window_end": time_end,
                    "key_metrics": key_metrics,
                    "crystallized_by_cycle_id": crystallized_by_cycle_id,
                    "crystallized_by_user_id": crystallized_by_user_id,
                    "source_count": len(members),
                }
            )
            .execute()
        )
        rows = insert.data or []
        if not rows:
            raise RuntimeError("executive_insight insert returned no rows")
        insight_id = rows[0]["insight_id"]

        # Insert insight_edges rows: one per source (episodic_memory) AND
        # one per distinct causal_path mentioned (so JIT verify can follow
        # the chain upward to causal_paths).
        edges_added = 0
        edge_rows = []
        seen_causal_paths = set()
        for m in members:
            edge_rows.append(
                {
                    "source_type": "episodic_memory",
                    "source_id": str(m["memory_id"]),
                    "target_type": "executive_insight",
                    "target_id": str(insight_id),
                    "edge_type": "summarizes",
                    "brand": brand,
                    "region": region,
                }
            )
            cp = m.get("causal_path_id")
            if cp and cp not in seen_causal_paths:
                edge_rows.append(
                    {
                        "source_type": "causal_path",
                        "source_id": str(cp),
                        "target_type": "executive_insight",
                        "target_id": str(insight_id),
                        "edge_type": "summarizes",
                        "brand": brand,
                        "region": region,
                    }
                )
                seen_causal_paths.add(cp)
        if edge_rows:
            # upsert-ish: insert may fail on unique constraint if rerun; that's
            # fine because we filter incomplete crystallizations elsewhere.
            try:
                client.table("insight_edges").insert(edge_rows).execute()
                edges_added = len(edge_rows)
            except Exception as exc:
                logger.warning(f"crystallizer: edge insert partially failed: {exc}")
        return insight_id, edges_added


# ============================================================================
# Helpers
# ============================================================================


def _group_memories(memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group memories by a stable key. Prefer causal_path_id; otherwise fall back
    to (region, event_type, agent_name-anchor) — but for v1 we require a
    causal_path_id to crystallize, since that's the strongest provenance link.

    Returns dict mapping group_key -> list of memory rows.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for m in memories:
        cp = m.get("causal_path_id")
        if not cp:
            continue
        key = f"causal_path:{cp}"
        groups.setdefault(key, []).append(m)
    return groups


def _compose_narrative(
    brand: str, region: Optional[str], members: List[Dict[str, Any]]
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Build (title, narrative, key_metrics) from source episodic memories.

    Deterministic: same inputs → same output. No LLM in v1.
    """
    agents = sorted({m.get("agent_name", "unknown") for m in members})
    causal_path_id = members[0].get("causal_path_id") if members else None
    region_str = f" in {region}" if region else ""
    title = f"{brand}{region_str}: cross-agent finding on causal path {causal_path_id}"

    lines = [
        f"Cross-agent crystallized insight for {brand}{region_str}.",
        f"Source agents ({len(agents)}): {', '.join(agents)}.",
        f"Source memories: {len(members)}.",
        "",
        "Findings:",
    ]
    for m in members:
        agent = m.get("agent_name", "unknown")
        desc = m.get("description") or ""
        outcome = m.get("outcome_type") or "n/a"
        lines.append(f"  - [{agent} / {outcome}] {desc}")

    narrative = "\n".join(lines)

    key_metrics = {
        "source_count": len(members),
        "distinct_agents": len(agents),
        "agents": agents,
        "causal_path_id": causal_path_id,
    }
    return title, narrative, key_metrics


def _pick_kpi(members: List[Dict[str, Any]]) -> Optional[str]:
    """Best-effort KPI extraction from raw_content. Returns first hit."""
    for m in members:
        rc = m.get("raw_content") or {}
        if isinstance(rc, dict):
            kpi = rc.get("kpi")
            if kpi:
                return str(kpi)
    return None


# Module-level convenience entrypoint.
async def crystallize_for_brand(
    brand: str,
    region: Optional[str] = None,
    *,
    crystallized_by_cycle_id: Optional[str] = None,
    crystallized_by_user_id: Optional[str] = None,
) -> CrystallizerResult:
    return await Crystallizer().run_for_brand(
        brand=brand,
        region=region,
        crystallized_by_cycle_id=crystallized_by_cycle_id,
        crystallized_by_user_id=crystallized_by_user_id,
    )
