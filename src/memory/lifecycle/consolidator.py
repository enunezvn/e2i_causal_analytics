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
          N >= 1 rows, ``_dedup_group`` decides whether to merge into
          an existing canonical (pre-check by SELECT on (brand,
          signature)) or to pick a local canonical and stamp it.
        * Non-canonical duplicate rows are DELETED. The canonical row's
          incremented counter preserves the audit trail of how many
          underlying events were collapsed.
        * Per-group atomicity: each group's (stamp + delete) sequence is
          compensating-rolled-back on partial failure, so a delete-fail
          never leaves the counter bumped while duplicates survive
          (which would inflate ``SUM(dedup_counter)`` for promotion).

        Brand-boundary: signatures embed ``brand``; groups are keyed
        on ``(brand, signature)``, so cross-brand collapse is
        structurally impossible. Defense in depth via the DB index
        which is per (COALESCE(brand,''), dedup_signature).

        Late-arrival contract (iter-1 H1 fix): when a candidate row's
        ``(brand, signature)`` matches a row that's ALREADY stamped
        (i.e. previously-deduped canonical from an earlier run), the
        new row is merged INTO that canonical rather than stamped as
        a new duplicate-canonical (which would hit the DB
        partial-unique-index UniqueViolation).

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
        # explicitly so a None-brand fallback row cannot ever collide
        # with a real-brand row. Rows where signature is None are
        # collected separately (cannot be safely deduped).
        groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        unkeyed: List[Dict[str, Any]] = []
        for row in rows:
            sig = _compute_dedup_signature(row)
            if sig is None:
                unkeyed.append(row)
                continue
            groups[(row.get("brand"), sig)].append(row)

        for (group_brand, signature), group in groups.items():
            self._dedup_group(
                client=client,
                group_brand=group_brand,
                signature=signature,
                group=group,
                result=result,
            )

        # Unkeyed rows: nothing to do — they lack a safe dedup signature.
        if unkeyed:
            logger.info(
                f"consolidator: skipped {len(unkeyed)} episodic rows with "
                "no safe dedup signature (missing brand/event_type/event_subtype)"
            )

        return result

    # ----------------------------------------- per-group dedup helpers

    def _dedup_group(
        self,
        client: Any,
        group_brand: Optional[str],
        signature: str,
        group: List[Dict[str, Any]],
        result: ConsolidationResult,
    ) -> None:
        """Process one (brand, signature) group of un-stamped rows.

        Two paths:

        1. **Existing-canonical merge** (H1 fix path): SELECT for a
           previously-stamped canonical with the same (brand,
           signature). If one exists, MERGE the new group into it —
           increment ITS counter by SUM(group counters), then DELETE
           the new group's rows. Never stamps a second row with the
           same signature (which would hit the DB partial-unique-index).

        2. **Fresh-canonical stamp** (multi-row OR singleton with no
           existing canonical): pick the oldest row in the group as
           canonical, stamp it with the signature + merged counter,
           then DELETE the rest. Compensating-rollback on failure
           (M1 fix path) keeps counter/deletes in sync.
        """
        # Path 1: probe for an already-stamped canonical with this
        # (brand, signature). The DB partial-unique-index guarantees at
        # most one such row.
        existing_canonical = self._find_canonical_for_signature(
            client=client, brand=group_brand, signature=signature
        )
        if existing_canonical is not None:
            self._merge_into_existing_canonical(
                client=client,
                group_brand=group_brand,
                signature=signature,
                existing_canonical=existing_canonical,
                incoming=group,
                result=result,
            )
            return

        # Path 2: no existing canonical. Stamp the oldest in the group.
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
        canonical_pre_counter = int(canonical.get("dedup_counter") or 1)

        # Multi-row vs singleton stamp behavior. For a singleton with no
        # existing canonical, we just stamp the row's signature (counter
        # stays at its current value); no duplicates to delete, so no
        # atomicity concern.
        if not duplicates:
            self._stamp_dedup_signature(
                client=client,
                memory_id=canonical["memory_id"],
                signature=signature,
                counter=canonical_pre_counter,
                result=result,
            )
            return

        # Multi-row: STAMP first, then DELETE. On DELETE failure, REVERT
        # the stamp so the group's counter + deletes stay in sync. Real
        # DB has SAVEPOINT semantics that would give us this for free;
        # we synthesize the same outcome with explicit compensating
        # writes so the contract holds for any client (including the
        # FakeSupabase used in unit tests).
        stamped = self._stamp_dedup_signature(
            client=client,
            memory_id=canonical["memory_id"],
            signature=signature,
            counter=merged_counter,
            result=result,
        )
        if not stamped:
            return  # stamp failure already recorded; no deletes to roll back

        delete_ok = self._delete_duplicates(
            client=client,
            memory_ids=[d["memory_id"] for d in duplicates],
            result=result,
        )
        if not delete_ok:
            # Compensate: revert the canonical's stamp to its pre-state
            # so the next run can retry the whole group cleanly. Best
            # effort — if the revert itself fails, log + record but
            # don't crash the consolidator.
            try:
                client.table("episodic_memories").update(
                    {"dedup_signature": None, "dedup_counter": canonical_pre_counter}
                ).eq("memory_id", canonical["memory_id"]).execute()
            except Exception as exc:
                logger.warning(
                    f"consolidator: dedup compensating revert failed for "
                    f"{canonical['memory_id']}: {exc}"
                )
                result.errors.append(f"dedup revert {canonical['memory_id']}: {exc}")
            return

        collapsed_n = len(duplicates)
        result.episodic_dedup_collapsed += collapsed_n
        result.record(group_brand or "_unknown", "dedup_collapsed")
        logger.info(
            f"consolidator: deduped {collapsed_n + 1} episodic rows "
            f"(brand={group_brand}, sig={signature[:24]}..., "
            f"canonical={canonical['memory_id']}, "
            f"merged_counter={merged_counter})"
        )

    def _find_canonical_for_signature(
        self, client: Any, brand: Optional[str], signature: str
    ) -> Optional[Dict[str, Any]]:
        """SELECT the (single) row already stamped with this (brand,
        signature). DB partial-unique-index guarantees at most one.

        Returns None if none exists OR if the lookup fails (defensive:
        a transient lookup failure should not be silently treated as
        "no canonical exists" because the caller would then double-
        stamp). Caller treats None as "no canonical" — for the
        defensive case a follow-up run will pick up the leftover
        candidates and retry.
        """
        try:
            query = client.table("episodic_memories").select(
                "memory_id, brand, dedup_signature, dedup_counter, occurred_at"
            )
            if brand is not None:
                query = query.eq("brand", brand)
            else:
                # None-brand case (rare in production but possible per the
                # nullable column on episodic_memories). The DB index uses
                # COALESCE(brand, '') so the application layer must match.
                query = query.is_("brand", "null")
            query = query.eq("dedup_signature", signature)
            data = (query.execute().data) or []
        except Exception as exc:
            logger.warning(
                f"consolidator: canonical lookup failed for sig={signature[:24]}...: {exc}"
            )
            return None
        if not data:
            return None
        # DB index guarantees at most one; defensively return the first.
        first: Dict[str, Any] = data[0]
        return first

    def _merge_into_existing_canonical(
        self,
        client: Any,
        group_brand: Optional[str],
        signature: str,
        existing_canonical: Dict[str, Any],
        incoming: List[Dict[str, Any]],
        result: ConsolidationResult,
    ) -> None:
        """Merge incoming un-stamped rows INTO an already-stamped
        canonical row. Increment canonical's counter by SUM(incoming
        counters); DELETE the incoming rows. On delete failure, revert
        the canonical's counter to its pre-merge state so the group's
        counter + deletes stay in sync (M1 atomicity fix path)."""
        if not incoming:
            return

        canonical_id = existing_canonical["memory_id"]
        canonical_pre_counter = int(existing_canonical.get("dedup_counter") or 1)
        incoming_total = sum(int(r.get("dedup_counter") or 1) for r in incoming)
        new_counter = canonical_pre_counter + incoming_total

        # Bump canonical counter first.
        try:
            client.table("episodic_memories").update({"dedup_counter": new_counter}).eq(
                "memory_id", canonical_id
            ).execute()
        except Exception as exc:
            logger.warning(
                f"consolidator: dedup merge counter-bump failed for {canonical_id}: {exc}"
            )
            result.errors.append(f"dedup merge bump {canonical_id}: {exc}")
            return

        # Then delete the incoming rows. On failure, revert the counter
        # so we don't end up with the canonical bumped + duplicates
        # still alive (double-counting in promotion).
        delete_ok = self._delete_duplicates(
            client=client,
            memory_ids=[r["memory_id"] for r in incoming],
            result=result,
        )
        if not delete_ok:
            try:
                client.table("episodic_memories").update(
                    {"dedup_counter": canonical_pre_counter}
                ).eq("memory_id", canonical_id).execute()
            except Exception as exc:
                logger.warning(
                    f"consolidator: dedup merge compensating revert failed "
                    f"for {canonical_id}: {exc}"
                )
                result.errors.append(f"dedup merge revert {canonical_id}: {exc}")
            return

        result.episodic_dedup_collapsed += len(incoming)
        result.record(group_brand or "_unknown", "dedup_collapsed")
        logger.info(
            f"consolidator: merged {len(incoming)} late-arrival rows into "
            f"canonical {canonical_id} (brand={group_brand}, "
            f"sig={signature[:24]}..., new_counter={new_counter})"
        )

    def _delete_duplicates(
        self,
        client: Any,
        memory_ids: List[Any],
        result: ConsolidationResult,
    ) -> bool:
        """Delete duplicate rows by memory_id. Returns True iff ALL
        deletions succeeded — caller uses this to drive compensating
        rollback on partial failure.

        Tries batch-delete via ``in_()`` first (one round-trip; atomic
        on real Postgres). Falls back to per-row delete only if the
        client shim doesn't implement ``in_()``. Per-row failures count
        as group-failure to preserve the atomicity contract — even if
        SOME duplicates were deleted, the caller will revert the
        canonical's stamp, which means the next run will retry the
        group cleanly (the still-deleted rows just won't reappear).
        """
        if not memory_ids:
            return True
        try:
            client.table("episodic_memories").delete().in_("memory_id", memory_ids).execute()
            return True
        except (AttributeError, TypeError):
            # Shim doesn't support in_() — fall back to per-row delete.
            all_ok = True
            for dup_id in memory_ids:
                try:
                    client.table("episodic_memories").delete().eq("memory_id", dup_id).execute()
                except Exception as exc:
                    logger.warning(f"consolidator: dedup delete failed for {dup_id}: {exc}")
                    result.errors.append(f"dedup delete {dup_id}: {exc}")
                    all_ok = False
            return all_ok
        except Exception as exc:
            logger.warning(f"consolidator: dedup batch delete failed: {exc}")
            result.errors.append(f"dedup batch delete: {exc}")
            return False

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
