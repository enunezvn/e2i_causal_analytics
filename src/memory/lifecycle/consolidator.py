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

Episodic deduplication (#388):
    Before the semantic promotion step, ``deduplicate_episodic`` collapses
    rows that share an exact-match key signature into a single canonical
    row. The signature is computed by ``_compute_dedup_signature`` (pure
    helper, also exported for testing). Strategy is exact-match only —
    embedding-similarity and fuzzy dedup are explicitly out of scope
    (issue #388 §Out of scope). DB-level race-condition safety is
    provided by a partial-unique-index on
    (brand, dedup_signature) WHERE dedup_signature IS NOT NULL added in
    migration 026_episodic_dedup.sql.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from src.memory.services.factories import get_supabase_client

logger = logging.getLogger(__name__)


# Promotion thresholds. Configurable via env later; conservative defaults.
SEMANTIC_MIN_CONFIRMATIONS = 3  # episodic memories citing the same causal_path
PROCEDURAL_MIN_USAGE = 5  # procedural_memories.usage_count
PROCEDURAL_MIN_SUCCESS_RATE = 0.8  # success_rate field on procedural memories


# Dedup signature version. Bump when the signature shape changes so the
# (brand, dedup_signature) partial-unique-index treats old + new
# generations as distinct rather than colliding mid-rollout.
DEDUP_SIGNATURE_VERSION = "v1"


def _compute_dedup_signature(row: Mapping[str, Any]) -> Optional[str]:
    """Pure helper returning a deterministic dedup key for one episodic row.

    Design (justifying the recommended primary + fallback key shapes from
    issue #388 §Design constraints item 3):

    * **Primary**: ``(brand, event_type, event_subtype, causal_path_id)``
      when ``causal_path_id`` is set. These four columns scope an
      episodic memory to one specific causal-path event under one brand;
      duplicates with this exact tuple are noise in the consolidator
      sweep (same observation logged N times by independent cognitive
      cycles).
    * **Fallback**: ``(brand, event_type, event_subtype, agent_name,
      description_hash)`` when ``causal_path_id IS NULL``. The
      description-hash backstops the cases where no causal path was
      involved (e.g. trigger-only events).

    Brand is included in EVERY variant so a brand difference always
    yields a distinct signature — defense in depth alongside the DB
    partial-unique-index on (brand, dedup_signature).

    Returns ``None`` when ``brand``, ``event_type``, or ``event_subtype``
    is missing. In that case the row is not safe to dedup and the caller
    skips it.
    """
    brand = row.get("brand")
    event_type = row.get("event_type")
    event_subtype = row.get("event_subtype")
    if not brand or not event_type or not event_subtype:
        return None

    causal_path_id = row.get("causal_path_id")
    if causal_path_id:
        # Primary key: (brand, event_type, event_subtype, causal_path_id).
        # str() coerces enum values (memory_event_type) consistently.
        payload = "|".join(
            (
                "primary",
                str(brand),
                str(event_type),
                str(event_subtype),
                str(causal_path_id),
            )
        )
        variant = "primary"
    else:
        # Fallback key: agent_name + description (hashed) backstop the
        # no-causal-path case.
        description = row.get("description") or ""
        agent_name = row.get("agent_name") or ""
        desc_hash = hashlib.sha256(description.encode("utf-8")).hexdigest()
        payload = "|".join(
            (
                "fallback",
                str(brand),
                str(event_type),
                str(event_subtype),
                str(agent_name),
                desc_hash,
            )
        )
        variant = "fallback"

    sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{DEDUP_SIGNATURE_VERSION}:{variant}:{sig}"


@dataclass
class ConsolidationResult:
    """Summary of one consolidator run."""

    promoted_to_semantic: int = 0
    promoted_to_procedural: int = 0
    causal_paths_examined: int = 0
    procedural_examined: int = 0
    # Dedup metrics (#388): episodic rows examined / collapsed.
    episodic_dedup_examined: int = 0
    episodic_dedup_collapsed: int = 0
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

        Order:
            1. ``deduplicate_episodic`` — collapses near-duplicate episodic
               rows so promotion thresholds see effective (deduplicated)
               counts. Runs first because semantic-promotion's
               confirmation-count threshold reads SUM(dedup_counter).
            2. ``_promote_to_semantic`` — stamps causal_paths consolidated.
            3. ``_promote_to_procedural`` — graduates procedural memories.
        """
        result = ConsolidationResult()
        try:
            await self.deduplicate_episodic(brand=brand, region=None, result=result)
        except Exception as exc:
            logger.exception("consolidator: episodic dedup failed")
            result.errors.append(f"dedup: {exc}")
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
            f"dedup_collapsed={result.episodic_dedup_collapsed} "
            f"errors={len(result.errors)}"
        )
        return result

    # ------------------------------------------------------- episodic dedup

    async def deduplicate_episodic(
        self,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        result: Optional[ConsolidationResult] = None,
    ) -> ConsolidationResult:
        """Collapse near-duplicate episodic memories under exact-match keys.

        Strategy (issue #388):

        * Each row's dedup signature is computed by
          ``_compute_dedup_signature`` (pure helper). Rows lacking the
          required fields are skipped (signature is ``None``).
        * Rows are grouped by ``(brand, signature)``. For each group of
          N >= 2 rows, the oldest row (lowest ``occurred_at``) is
          chosen as canonical; its ``dedup_signature`` is stamped and
          its ``dedup_counter`` is incremented by ``SUM(dedup_counter)``
          of the duplicates (so re-running on already-deduped data is a
          monotonic-counter no-op, not a re-merge).
        * Non-canonical duplicate rows are DELETED. The canonical row's
          incremented counter preserves the audit trail of how many
          underlying events were collapsed.
        * Singletons (group of 1) get their dedup_signature stamped so
          subsequent runs do not re-examine them and the DB
          partial-unique-index (brand, dedup_signature) covers future
          inserts.

        Brand-boundary: signatures embed ``brand``; groups are keyed
        on ``(brand, signature)``, so cross-brand collapse is
        structurally impossible. Defense in depth via the DB index
        which is per (brand, dedup_signature).

        Args:
            brand: optional scope. If None, sweeps all brands.
            region: optional regional scope. If None, sweeps all regions.
            result: optional ``ConsolidationResult`` to accumulate
                metrics into. If not supplied (called outside of
                ``run()``), a fresh one is created and returned.
        """
        if result is None:
            result = ConsolidationResult()
        client = get_supabase_client()

        # Pull rows that have NOT yet been deduped (dedup_signature IS NULL).
        # Idempotency hinges on this filter: previously-deduped rows have
        # their signature set and are excluded from subsequent passes.
        query = client.table("episodic_memories").select(
            "memory_id, brand, region, event_type, event_subtype, "
            "causal_path_id, agent_name, description, occurred_at, "
            "dedup_signature, dedup_counter"
        )
        if brand:
            query = query.eq("brand", brand)
        if region:
            query = query.eq("region", region)
        # IS NULL filter: only candidates without a signature yet.
        query = query.is_("dedup_signature", "null")
        try:
            rows = (query.execute().data) or []
        except Exception as exc:
            logger.warning(f"consolidator: dedup select failed: {exc}")
            result.errors.append(f"dedup select: {exc}")
            return result

        result.episodic_dedup_examined += len(rows)
        if not rows:
            return result

        # Group rows by (brand, signature). Brand is included in the key
        # explicitly so a None-brand fallback row (signature == None) cannot
        # ever collide with a real-brand row. Rows where signature is None
        # are bucketed into a noop group keyed by their memory_id (singleton
        # group) so they pass through untouched.
        groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        unkeyed: List[Dict[str, Any]] = []
        for row in rows:
            sig = _compute_dedup_signature(row)
            if sig is None:
                unkeyed.append(row)
                continue
            groups[(row.get("brand"), sig)].append(row)

        for (group_brand, signature), group in groups.items():
            if len(group) == 1:
                # Singleton: stamp the signature so future inserts hit the
                # DB partial-unique-index, but no counter increment.
                solo = group[0]
                self._stamp_dedup_signature(
                    client=client,
                    memory_id=solo["memory_id"],
                    signature=signature,
                    counter=int(solo.get("dedup_counter") or 1),
                    result=result,
                )
                continue

            # Multi-row group: pick canonical (oldest by occurred_at —
            # falls back to memory_id for tie-break determinism).
            group_sorted = sorted(
                group,
                key=lambda r: (
                    r.get("occurred_at") or "",
                    str(r.get("memory_id") or ""),
                ),
            )
            canonical = group_sorted[0]
            duplicates = group_sorted[1:]
            merged_counter = sum(int(r.get("dedup_counter") or 1) for r in group_sorted)

            # Stamp canonical with signature + merged counter.
            stamped = self._stamp_dedup_signature(
                client=client,
                memory_id=canonical["memory_id"],
                signature=signature,
                counter=merged_counter,
                result=result,
            )
            if not stamped:
                # If the stamp failed, skip deletion to preserve evidence.
                continue

            # Delete duplicates by memory_id. Use a single in_() if possible;
            # otherwise per-row delete. Per-row is robust against client
            # shims that don't implement in_().
            duplicate_ids = [d["memory_id"] for d in duplicates]
            try:
                # Try batch delete first.
                client.table("episodic_memories").delete().in_("memory_id", duplicate_ids).execute()
            except (AttributeError, TypeError):
                # Shim doesn't support in_() — fall back to per-row delete.
                for dup_id in duplicate_ids:
                    try:
                        client.table("episodic_memories").delete().eq("memory_id", dup_id).execute()
                    except Exception as exc:
                        logger.warning(f"consolidator: dedup delete failed for {dup_id}: {exc}")
                        result.errors.append(f"dedup delete {dup_id}: {exc}")
            except Exception as exc:
                logger.warning(f"consolidator: dedup batch delete failed: {exc}")
                result.errors.append(f"dedup batch delete: {exc}")
                # Fall back to per-row deletion as a safety net.
                for dup_id in duplicate_ids:
                    try:
                        client.table("episodic_memories").delete().eq("memory_id", dup_id).execute()
                    except Exception as exc2:
                        logger.warning(f"consolidator: dedup delete failed for {dup_id}: {exc2}")
                        result.errors.append(f"dedup delete {dup_id}: {exc2}")

            collapsed_n = len(duplicates)
            result.episodic_dedup_collapsed += collapsed_n
            result.record(group_brand or "_unknown", "dedup_collapsed")
            logger.info(
                f"consolidator: deduped {collapsed_n + 1} episodic rows "
                f"(brand={group_brand}, sig={signature[:24]}..., "
                f"canonical={canonical['memory_id']}, "
                f"merged_counter={merged_counter})"
            )

        # Unkeyed rows: nothing to do — they lack a safe dedup signature.
        if unkeyed:
            logger.info(
                f"consolidator: skipped {len(unkeyed)} episodic rows with "
                "no safe dedup signature (missing brand/event_type/event_subtype)"
            )

        return result

    def _stamp_dedup_signature(
        self,
        client: Any,
        memory_id: str,
        signature: str,
        counter: int,
        result: ConsolidationResult,
    ) -> bool:
        """UPDATE one episodic_memories row to set dedup_signature +
        dedup_counter. Returns True on success."""
        try:
            client.table("episodic_memories").update(
                {"dedup_signature": signature, "dedup_counter": counter}
            ).eq("memory_id", memory_id).execute()
            return True
        except Exception as exc:
            logger.warning(f"consolidator: dedup stamp failed for {memory_id}: {exc}")
            result.errors.append(f"dedup stamp {memory_id}: {exc}")
            return False

    # ---------------------------------------------------------- semantic tier

    async def _promote_to_semantic(self, result: ConsolidationResult, brand: Optional[str]) -> None:
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

            # Count effective episodic confirmations of this path. After
            # episodic deduplication (#388), a row's ``dedup_counter`` is
            # the number of underlying events it represents — so the
            # effective confirmation count is SUM(dedup_counter), not
            # COUNT(*). We pull memory_id + dedup_counter and aggregate
            # in Python because supabase-py doesn't expose a SUM() helper
            # via the .table() builder; the row set per causal_path is
            # small (bounded by N_confirmations after dedup), so this is
            # cheap. episodic_memories.causal_path_id is indexed.
            try:
                rows_result = (
                    client.table("episodic_memories")
                    .select("memory_id, dedup_counter")
                    .eq("causal_path_id", path_id)
                    .execute()
                )
                confirmation_rows = rows_result.data or []
                # SUM(dedup_counter) — treats missing counter as 1
                # (back-compat for rows from before migration 026).
                confirmation_count = sum(
                    int(r.get("dedup_counter") or 1) for r in confirmation_rows
                )
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
                    f"consolidator: update failed for procedural {proc.get('procedure_id')}: {exc}"
                )
                result.errors.append(f"proc update {proc.get('procedure_id')}: {exc}")


# Module-level convenience entrypoint for Celery / tests.
async def consolidate_insights(brand: Optional[str] = None) -> ConsolidationResult:
    """Run a consolidation pass with default thresholds."""
    return await Consolidator().run(brand=brand)
