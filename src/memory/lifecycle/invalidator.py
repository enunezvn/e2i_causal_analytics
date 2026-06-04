"""
Cascading invalidation across the provenance DAG.

When a causal_path is overturned (or a sentinel fires an invalidate action),
every downstream artifact derived from it must be marked stale: dependent
triggers, ml_predictions, and executive_insights.

The cascade is a BFS through ``insight_edges`` rooted at the source. Brand
is enforced on every hop: only edges with matching ``brand`` (or
``brand='all'``) are traversed. A Kisqali overturn cannot bleed into a
Fabhalta dependent unless the edge was explicitly cross-brand.

Side effect: each successful invalidation publishes to the Redis pub/sub
channel ``invalidation:e2i:{brand}`` so per-agent ``memory_hooks.invalidate_cache``
subscribers can drop their local caches without tight coupling.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from src.memory.services.factories import get_redis_client, get_supabase_client
from src.mlops.lifecycle_monitoring import record_cascade_complete

logger = logging.getLogger(__name__)


# Tables that carry invalidated_at columns. The cascade updates these on hit.
INVALIDATABLE_TABLES: Dict[str, str] = {
    "trigger": "triggers",
    "ml_prediction": "ml_predictions",
    "executive_insight": "executive_insights",
}

# Primary-key columns for each target table.
TARGET_PK: Dict[str, str] = {
    "trigger": "trigger_id",
    "ml_prediction": "prediction_id",
    "executive_insight": "insight_id",
}


@dataclass
class CascadeResult:
    """Summary of a cascade_invalidate run."""

    source_type: str
    source_id: str
    scope_brand: str
    reason: str
    visited: int = 0
    invalidated_by_type: Dict[str, int] = field(default_factory=dict)
    skipped_cross_brand: int = 0
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None

    def record_hit(self, target_type: str) -> None:
        self.invalidated_by_type[target_type] = self.invalidated_by_type.get(target_type, 0) + 1


async def cascade_invalidate(
    source_type: str,
    source_id: str,
    reason: str,
    scope_brand: str,
    *,
    publish_signal: bool = True,
    max_depth: int = 16,
) -> CascadeResult:
    """
    BFS downstream from (source_type, source_id), marking invalidatable
    artifacts as stale. Brand-scoped on every hop.

    Args:
        source_type: e.g. 'causal_path'
        source_id:   row id in that table (TEXT)
        reason:      free-text reason recorded on each invalidated row
        scope_brand: brand to filter edges by. Edges with this brand OR 'all'
                     are traversed.
        publish_signal: if True, fire ``invalidation:e2i:{brand}`` notifications
                        so per-agent caches can drop entries.
        max_depth:   safety bound on the BFS.

    Returns:
        CascadeResult with counts per target type.
    """
    if not scope_brand:
        raise ValueError("scope_brand is required (use 'all' for explicit cross-brand)")

    result = CascadeResult(
        source_type=source_type,
        source_id=source_id,
        scope_brand=scope_brand,
        reason=reason,
    )
    # Track BFS depth + edges visited for the #391 monitoring slice
    # (Opik trace + MLflow metric). Counted at the end so a partial
    # cascade (errors mid-traversal) still emits SOMETHING — the
    # values reflect what actually happened, not the optimistic plan.
    cascade_started_monotonic = time.monotonic()
    edges_visited_count = 0

    client = get_supabase_client()
    now_iso = datetime.now(timezone.utc).isoformat()

    # BFS frontier: list of (type, id) to expand. Visited prevents cycles
    # (insight_edges shouldn't cycle, but defense-in-depth is cheap).
    frontier: List[Tuple[str, str]] = [(source_type, source_id)]
    visited: Set[Tuple[str, str]] = set()
    depth = 0

    while frontier and depth < max_depth:
        next_frontier: List[Tuple[str, str]] = []
        for src_t, src_id in frontier:
            if (src_t, src_id) in visited:
                continue
            visited.add((src_t, src_id))
            result.visited += 1

            # Find direct downstream edges.
            try:
                edges_query = (
                    client.table("insight_edges")
                    .select("target_type, target_id, brand")
                    .eq("source_type", src_t)
                    .eq("source_id", src_id)
                )
                edges_result = edges_query.execute()
                edges = edges_result.data or []
            except Exception as exc:
                logger.exception(f"cascade: edge fetch failed for {src_t}:{src_id}")
                result.errors.append(f"edge fetch {src_t}:{src_id}: {exc}")
                continue

            for edge in edges:
                # Count every edge inspected (#391 monitoring: edges_visited
                # observable). Counted BEFORE the brand-scope filter so a
                # cross-brand-skipped edge still shows up in the "edges I
                # had to look at" tally — that's what the dashboard cares
                # about for I/O cost analysis.
                edges_visited_count += 1
                edge_brand = edge.get("brand")
                target_type = edge.get("target_type")
                target_id = edge.get("target_id")
                if not target_type or not target_id:
                    continue
                # Brand scoping: traverse only edges in our brand or explicit cross-brand.
                if edge_brand not in (scope_brand, "all"):
                    result.skipped_cross_brand += 1
                    continue
                # Mark invalidatable target rows.
                table = INVALIDATABLE_TABLES.get(target_type)
                if table:
                    pk_col = TARGET_PK[target_type]
                    try:
                        res = (
                            client.table(table)
                            .update(
                                {
                                    "invalidated_at": now_iso,
                                    "invalidation_reason": reason[:1000],  # truncate paranoia
                                }
                            )
                            .eq(pk_col, target_id)
                            .is_("invalidated_at", "null")
                            .execute()
                        )
                        # Only count a hit when the UPDATE actually matched a row.
                        # An already-invalidated target (guarded by
                        # is_("invalidated_at", "null")) matches nothing — counting
                        # it would over-report invalidated_by_type.
                        if res.data:
                            result.record_hit(target_type)
                    except Exception as exc:
                        logger.exception(f"cascade: update failed for {target_type}:{target_id}")
                        result.errors.append(f"update {target_type}:{target_id}: {exc}")
                # Walk further regardless of whether this row was invalidatable —
                # an episodic_memory has no invalidated_at but may have descendants.
                next_frontier.append((target_type, target_id))
        frontier = next_frontier
        depth += 1

    result.finished_at = datetime.now(timezone.utc)

    if publish_signal:
        await _publish_invalidation_signal(result)

    # #391 monitoring box 1.a + 2.a + 2.b: emit Opik trace + MLflow
    # cascade-frequency counter + propagation-depth gauge. Called at
    # the end of cascade execution. Best-effort by design — any
    # exception inside record_cascade_complete is swallowed there, so
    # the cascade's return value is not influenced by the
    # observability path. See [[feedback-codex-audits-within-existing-
    # signature-not-design]] — instrumentation runs AT the boundary
    # of the existing function, leaving the function signature + raises
    # contract unchanged.
    #
    # Codex iter-0 M3 closure: ``depth`` (the loop counter) counts
    # frontier-sweeps and is INCREMENTED after each sweep, including
    # the root-only sweep. So a cascade with no downstream edges
    # finishes with ``depth==1`` even though it propagated ZERO hops
    # past the source. The observable we WANT in the dashboard is
    # "hops past source" (==0 when no downstream), so subtract one
    # with a floor of 0. The internal ``depth`` variable's role
    # against ``max_depth`` is unchanged.
    propagation_depth = max(0, depth - 1)
    duration_ms = (time.monotonic() - cascade_started_monotonic) * 1000.0
    record_cascade_complete(
        brand=scope_brand,
        depth=propagation_depth,
        edges_visited=edges_visited_count,
        duration_ms=duration_ms,
        invalidated_by_type=dict(result.invalidated_by_type),
    )

    logger.info(
        f"cascade_invalidate({source_type}:{source_id}, brand={scope_brand}) "
        f"visited={result.visited} hits={result.invalidated_by_type} "
        f"skipped_cross_brand={result.skipped_cross_brand}"
    )
    return result


async def _publish_invalidation_signal(result: CascadeResult) -> None:
    """
    Notify per-agent cache hooks via brand-namespaced Redis pub/sub.

    Channel: ``invalidation:e2i:{brand}``
    Payload: JSON summary of the cascade.

    pub/sub (not Streams) because subscribers don't need replay — they just
    drop their local caches when notified.
    """
    try:
        redis = get_redis_client()
        channel = f"invalidation:e2i:{result.scope_brand}"
        payload = {
            "source_type": result.source_type,
            "source_id": result.source_id,
            "reason": result.reason,
            "invalidated_by_type": result.invalidated_by_type,
            "visited": result.visited,
            "at": result.finished_at.isoformat() if result.finished_at else None,
        }
        await redis.publish(channel, json.dumps(payload))
    except Exception:
        # Cache-invalidation broadcast is best-effort; never let it block the cascade.
        logger.exception("cascade: failed to publish invalidation signal")
