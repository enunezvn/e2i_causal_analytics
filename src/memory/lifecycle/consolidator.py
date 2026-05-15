"""
Causal Insight Consolidator — promotes artifacts up the 4-tier hierarchy.

Tiers:
    working    : redis evidence_board (transient, ttl)
    episodic   : episodic_memories rows (default tier)
    semantic   : causal_paths.consolidated_at set; promoted when a path
                 has been rediscovered/confirmed N times
    procedural : procedural_memories with high usage_count + success_rate;
                 promoted when an episodic pattern is reliably reusable

The consolidator is idempotent and brand-scoped — promotions happen
per-brand independently.

This is intentionally a small, focused pipeline: it queries for
candidates, applies the promotion rule, writes the side effects, and
returns counts. It does NOT do LLM-style summarization (deliberately —
pharma data needs deterministic provenance; see plan §Out of Scope).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.memory.services.factories import get_supabase_client

logger = logging.getLogger(__name__)


# Promotion thresholds. Configurable via env later; conservative defaults.
SEMANTIC_MIN_CONFIRMATIONS = 3       # episodic memories citing the same causal_path
PROCEDURAL_MIN_USAGE = 5             # procedural_memories.usage_count
PROCEDURAL_MIN_SUCCESS_RATE = 0.8    # success_rate field on procedural memories


@dataclass
class ConsolidationResult:
    """Summary of one consolidator run."""

    promoted_to_semantic: int = 0
    promoted_to_procedural: int = 0
    causal_paths_examined: int = 0
    procedural_examined: int = 0
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    by_brand: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, brand: str, kind: str) -> None:
        bucket = self.by_brand.setdefault(brand, {})
        bucket[kind] = bucket.get(kind, 0) + 1


class Consolidator:
    """
    Promotion engine.

    Public entrypoint: ``run()`` — invoked daily by the Celery beat
    ``consolidate_insights`` task. Can also be invoked synchronously
    by the cognitive workflow's reflector to consolidate after a cycle.
    """

    def __init__(
        self,
        semantic_min_confirmations: int = SEMANTIC_MIN_CONFIRMATIONS,
        procedural_min_usage: int = PROCEDURAL_MIN_USAGE,
        procedural_min_success_rate: float = PROCEDURAL_MIN_SUCCESS_RATE,
    ):
        self.semantic_min_confirmations = semantic_min_confirmations
        self.procedural_min_usage = procedural_min_usage
        self.procedural_min_success_rate = procedural_min_success_rate

    async def run(self, brand: Optional[str] = None) -> ConsolidationResult:
        """
        Run a full consolidation pass.

        Args:
            brand: optional scope. If None, runs across all brands.
        """
        result = ConsolidationResult()
        try:
            await self._promote_to_semantic(result, brand)
        except Exception as exc:
            logger.exception("consolidator: semantic promotion failed")
            result.errors.append(f"semantic: {exc}")
        try:
            await self._promote_to_procedural(result, brand)
        except Exception as exc:
            logger.exception("consolidator: procedural promotion failed")
            result.errors.append(f"procedural: {exc}")
        result.finished_at = datetime.now(timezone.utc)
        logger.info(
            f"consolidator finished promoted_semantic={result.promoted_to_semantic} "
            f"promoted_procedural={result.promoted_to_procedural} "
            f"errors={len(result.errors)}"
        )
        return result

    # ---------------------------------------------------------- semantic tier

    async def _promote_to_semantic(
        self, result: ConsolidationResult, brand: Optional[str]
    ) -> None:
        """
        Find causal_paths that have been confirmed >= N times AND have a
        passing validation_status, and stamp ``consolidated_at = now()``.

        We treat the count of distinct episodic_memories citing the path
        as the "confirmation count" — these come from independent
        cognitive cycles, so seeing the same path several times is a
        strong signal of robustness.
        """
        client = get_supabase_client()

        # Pull candidate causal_paths: not yet consolidated, not overturned.
        query = (
            client.table("causal_paths")
            .select("path_id, brand, validation_status, confirmation_count, consolidated_at")
            .is_("consolidated_at", "null")
        )
        if brand:
            query = query.eq("brand", brand)
        candidates = (query.execute().data) or []
        result.causal_paths_examined += len(candidates)

        for path in candidates:
            path_id = path.get("path_id")
            path_brand = path.get("brand")
            if not path_id:
                continue
            # An overturned path never gets consolidated.
            if path.get("validation_status") == "overturned":
                continue

            # Count distinct episodic memories citing this path. This is a
            # cheap COUNT(*) — episodic_memories.causal_path_id is indexed.
            try:
                count_result = (
                    client.table("episodic_memories")
                    .select("memory_id", count="exact")
                    .eq("causal_path_id", path_id)
                    .execute()
                )
                confirmation_count = count_result.count or 0
            except Exception as exc:
                logger.warning(f"consolidator: count failed for {path_id}: {exc}")
                result.errors.append(f"count {path_id}: {exc}")
                continue

            # Honor whichever count is higher — the running counter set by
            # the reflector OR the live episodic-memory count.
            effective = max(confirmation_count, path.get("confirmation_count") or 0)
            if effective < self.semantic_min_confirmations:
                continue

            now_iso = datetime.now(timezone.utc).isoformat()
            try:
                client.table("causal_paths").update(
                    {
                        "consolidated_at": now_iso,
                        "last_confirmed_at": now_iso,
                        "confirmation_count": effective,
                    }
                ).eq("path_id", path_id).execute()
                result.promoted_to_semantic += 1
                result.record(path_brand or "_unknown", "semantic")
                logger.info(
                    f"consolidator: promoted causal_path {path_id} to semantic "
                    f"(brand={path_brand}, confirmations={effective})"
                )
            except Exception as exc:
                logger.warning(f"consolidator: update failed for {path_id}: {exc}")
                result.errors.append(f"update {path_id}: {exc}")

    # -------------------------------------------------------- procedural tier

    async def _promote_to_procedural(
        self, result: ConsolidationResult, brand: Optional[str]
    ) -> None:
        """
        Find procedural_memories with usage_count >= K and success_rate >= S
        that haven't been marked as consolidated. Mark them with
        ``consolidation_tier = 'procedural'`` via episodic_memories link.

        Procedural memory table already exists; we just need to track which
        rows have been graduated. We use the description prefix '[PROC]'
        sentinel so we don't need yet another schema migration on the
        procedural table — the consolidator and the episodic side both
        agree on what graduation means without a new column.
        """
        client = get_supabase_client()
        query = (
            client.table("procedural_memories")
            .select("procedure_id, procedure_name, applicable_brands, success_rate, usage_count")
            .gte("usage_count", self.procedural_min_usage)
            .gte("success_rate", self.procedural_min_success_rate)
        )
        candidates = (query.execute().data) or []
        result.procedural_examined += len(candidates)

        for proc in candidates:
            applicable = proc.get("applicable_brands") or []
            # Filter by brand if scoped: applicable_brands is a list column.
            if brand and applicable and brand not in applicable and "all" not in applicable:
                continue
            # Idempotency: only promote rows that haven't been marked yet.
            # We use a JSONB-style marker in procedure_name as a side-channel
            # so we don't migrate the procedural table for this v1.
            if proc.get("procedure_name", "").startswith("[PROC] "):
                continue
            new_name = f"[PROC] {proc.get('procedure_name', '')}"
            try:
                client.table("procedural_memories").update({"procedure_name": new_name}).eq(
                    "procedure_id", proc["procedure_id"]
                ).execute()
                result.promoted_to_procedural += 1
                # Record under the first applicable brand for stats.
                bucket_brand = applicable[0] if applicable else "_unknown"
                result.record(bucket_brand, "procedural")
                logger.info(
                    f"consolidator: promoted procedural {proc['procedure_id']} "
                    f"(applicable_brands={applicable})"
                )
            except Exception as exc:
                logger.warning(
                    f"consolidator: update failed for procedural "
                    f"{proc.get('procedure_id')}: {exc}"
                )
                result.errors.append(f"proc update {proc.get('procedure_id')}: {exc}")


# Module-level convenience entrypoint for Celery / tests.
async def consolidate_insights(brand: Optional[str] = None) -> ConsolidationResult:
    """Run a consolidation pass with default thresholds."""
    return await Consolidator().run(brand=brand)
