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
from typing import Any, Dict, List, Mapping, Optional, Set

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
    """Summary of one consolidator run.

    ``errors`` policy (iter-2 M2 documentation): the consolidator's
    dedup path is best-effort and offers read-your-writes consistency
    only WITHIN a single ``deduplicate_episodic`` call. Across
    concurrent passes against the same brand, the compensating-rollback
    pattern is externally non-atomic (a concurrent reader can observe
    the intermediate bumped counter before the revert runs). A future
    PR may wrap each per-group sequence in a real DB transaction; the
    application-level pattern here is the fallback that works for the
    FakeSupabase used in unit tests AND for any client that doesn't
    expose transactional semantics.

    Unrevertable errors (iter-2 new-H1): when BOTH the original
    mutation AND its compensating revert fail, the brand is added to
    ``brands_with_dedup_errors`` so downstream promotion phases can
    short-circuit instead of double-counting on the inconsistent
    counter.
    """

    promoted_to_semantic: int = 0
    promoted_to_procedural: int = 0
    causal_paths_examined: int = 0
    procedural_examined: int = 0
    # Dedup metrics (#388): episodic rows examined / collapsed.
    episodic_dedup_examined: int = 0
    episodic_dedup_collapsed: int = 0
    errors: List[str] = field(default_factory=list)
    # iter-2 new-H1: typed set of brands whose dedup left an
    # unrevertable inconsistent state (original mutation failed AND
    # compensating revert also failed). _promote_to_semantic short-
    # circuits for these brands. Revertable failures stay clean.
    brands_with_dedup_errors: Set[str] = field(default_factory=set)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    by_brand: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, brand: str, kind: str) -> None:
        bucket = self.by_brand.setdefault(brand, {})
        bucket[kind] = bucket.get(kind, 0) + 1

    def mark_brand_dedup_error(self, brand: Optional[str]) -> None:
        """Mark a brand as having an unrevertable dedup error.
        Downstream promotion phases skip this brand to avoid
        double-counting on the inconsistent counter."""
        if brand:
            self.brands_with_dedup_errors.add(brand)


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

        Same-pass race recovery (iter-2 new-M1): if a concurrent
        consolidator wins the race to stamp a canonical between our
        SELECT (no canonical found) and our UPDATE (stamp), the DB
        partial-unique-index rejects our stamp with UniqueViolation.
        The stamp handler re-queries the canonical and merges the
        loser into it in the SAME pass via ``_recover_unique_violation``
        — does not leave the loser unstamped for the next run.

        Unrevertable-error contract (iter-2 new-H1): when BOTH the
        original mutation AND its compensating revert fail, the brand
        is added to ``result.brands_with_dedup_errors``. The
        downstream ``_promote_to_semantic`` short-circuits for any
        brand in that set to avoid double-counting on the
        inconsistent counter.

        Known limitation (iter-2 M2, accepted for this PR — see
        ``ConsolidationResult.errors`` policy in the dataclass
        docstring): the compensating-rollback is externally non-atomic
        across separate client calls. A concurrent consolidator pass
        can observe the intermediate bumped counter before the revert
        runs against the same brand. Closing this requires wrapping
        each per-group sequence in a real DB transaction, which only
        works against real Postgres (not the FakeSupabase used in unit
        tests). Filed as a forward-looking follow-up for the next
        sub-issue under #388. Production safety: even with the race,
        each consolidator pass eventually converges on a consistent
        state because the candidate filter and the canonical-lookup
        are idempotent across passes.

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
        # atomicity concern. The stamp may still race with a concurrent
        # winner — _stamp_dedup_signature handles that via
        # _recover_unique_violation (iter-2 new-M1).
        if not duplicates:
            self._stamp_dedup_signature(
                client=client,
                memory_id=canonical["memory_id"],
                signature=signature,
                counter=canonical_pre_counter,
                result=result,
                brand=group_brand,
                loser_row=canonical,
                siblings=[],
            )
            return

        # Multi-row: STAMP first, then DELETE. On DELETE failure, REVERT
        # the stamp so the group's counter + deletes stay in sync. Real
        # DB has SAVEPOINT semantics that would give us this for free;
        # we synthesize the same outcome with explicit compensating
        # writes so the contract holds for any client (including the
        # FakeSupabase used in unit tests).
        #
        # The stamp helper may dispatch to ``_recover_unique_violation``
        # if a concurrent winner stamped first (iter-2 new-M1 path) —
        # in that case the loser is already merged into the existing
        # canonical and we skip the rest of this fresh-stamp path.
        stamped, recovered_via_merge = self._stamp_dedup_signature(
            client=client,
            memory_id=canonical["memory_id"],
            signature=signature,
            counter=merged_counter,
            result=result,
            brand=group_brand,
            loser_row=canonical,
            siblings=duplicates,
        )
        if not stamped:
            return  # stamp failure already recorded; no deletes to roll back
        if recovered_via_merge:
            return  # loser was merged into a concurrent winner; nothing left to do

        delete_ok = self._delete_duplicates(
            client=client,
            memory_ids=[d["memory_id"] for d in duplicates],
            result=result,
        )
        if not delete_ok:
            # Compensate: revert the canonical's stamp to its pre-state
            # so the next run can retry the whole group cleanly. Best
            # effort — if the revert itself fails, log + record AND
            # mark the brand as unrevertable so promotion skips it
            # (iter-2 new-H1).
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
                result.mark_brand_dedup_error(group_brand)
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
        # still alive (double-counting in promotion). On revert failure,
        # mark the brand as unrevertable so promotion skips it (iter-2
        # new-H1).
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
                result.mark_brand_dedup_error(group_brand)
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

    @staticmethod
    def _is_unique_violation(exc: BaseException) -> bool:
        """Detect a DB partial-unique-index violation by REQUIRING BOTH
        class-name AND message signals (iter-3 new-NEW-M2 tightening).

        Class-name signal: substring ``"UniqueViolation"`` (case-
        insensitive). Matches both production ``psycopg.errors.
        UniqueViolation`` and the unit-test stand-in
        ``_UniqueViolationStub``. Rejects unrelated exception classes
        that happen to mention "unique" (e.g. ``UniqueIDError``,
        ``UniqueGenError``).

        Message signal: ``"unique"`` AND one of ``"constraint"`` /
        ``"index"``. Real Postgres unique violations always say
        "duplicate key value violates unique constraint X" or
        "duplicate key value violates unique index Y" — both tokens
        present. Rejects exceptions with class-name match but
        unrelated wording (e.g. ``CustomViolation("not really")``).

        Both signals required → no false positives from broad token
        match.
        """
        cls_name = type(exc).__name__.lower()
        if "uniqueviolation" not in cls_name:
            return False
        msg = str(exc).lower()
        if "unique" not in msg:
            return False
        return "constraint" in msg or "index" in msg

    def _stamp_dedup_signature(
        self,
        client: Any,
        memory_id: str,
        signature: str,
        counter: int,
        result: ConsolidationResult,
        brand: Optional[str] = None,
        loser_row: Optional[Dict[str, Any]] = None,
        siblings: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[bool, bool]:
        """UPDATE one episodic_memories row to set dedup_signature +
        dedup_counter.

        Returns ``(stamped, recovered_via_merge)``:

        * ``(True, False)`` — stamp succeeded; caller proceeds with
          its normal post-stamp work (e.g. deleting duplicates).
        * ``(True, True)`` — stamp hit a UniqueViolation from a
          concurrent winner AND we successfully recovered by merging
          the loser (``loser_row``) + its ``siblings`` into the
          existing canonical (iter-2 new-M1 same-pass recovery path).
          Caller MUST skip post-stamp work because the recovery
          already handled it.
        * ``(False, False)`` — stamp failed for a non-recoverable
          reason; caller skips post-stamp work and the error is
          recorded.

        ``loser_row`` (iter-3 new-NEW-H1 fix) carries the canonical
        row dict whose stamp UPDATE failed. The recovery path reads
        ITS OWN counter — NOT the ``counter`` parameter (which is the
        merged_counter the caller WANTED to stamp). Using the row's
        own counter avoids double-counting in
        ``_merge_into_existing_canonical`` (which sums incoming
        counters; the siblings already contribute their own).
        ``siblings`` are the rest of the multi-row group.
        """
        try:
            client.table("episodic_memories").update(
                {"dedup_signature": signature, "dedup_counter": counter}
            ).eq("memory_id", memory_id).execute()
            return (True, False)
        except Exception as exc:
            if self._is_unique_violation(exc):
                # iter-2 new-M1: a concurrent consolidator stamped the
                # canonical between our SELECT (no canonical found) and
                # our UPDATE. Re-query + merge in the same pass.
                # iter-3 new-NEW-H1: pass loser_row directly so the
                # recovery reads the loser's OWN counter, not
                # merged_counter (which already includes siblings).
                recovered = self._recover_unique_violation(
                    client=client,
                    brand=brand,
                    signature=signature,
                    loser_row=loser_row,
                    loser_memory_id=memory_id,
                    siblings=siblings or [],
                    result=result,
                )
                if recovered:
                    return (True, True)
            logger.warning(f"consolidator: dedup stamp failed for {memory_id}: {exc}")
            result.errors.append(f"dedup stamp {memory_id}: {exc}")
            return (False, False)

    def _recover_unique_violation(
        self,
        client: Any,
        brand: Optional[str],
        signature: str,
        loser_memory_id: str,
        siblings: List[Dict[str, Any]],
        result: ConsolidationResult,
        loser_row: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Iter-2 new-M1 + iter-3 new-NEW-H1: recover from a
        concurrent-winner race WITHOUT double-counting.

        Called from ``_stamp_dedup_signature`` when the stamp UPDATE
        raises a unique-violation-shaped exception. Re-queries the
        canonical (the winner has just stamped) and merges the loser
        (``loser_row`` carrying its OWN ``dedup_counter``) + any
        ``siblings`` into the existing canonical.

        Critical iter-3 invariant: the incoming list contains each
        row's OWN ``dedup_counter`` — never the merged_counter the
        failed-stamp UPDATE was trying to set. The merge-helper sums
        incoming counters, so passing merged_counter for the loser
        would inflate by ``sum(siblings.counter)`` (the bug closed
        in iter-3 new-NEW-H1).

        Returns True iff recovery succeeded. False means the
        unique-violation could not be reconciled (canonical re-lookup
        empty, or merge itself failed) — caller falls back to
        recording the failure in ``result.errors``.
        """
        existing = self._find_canonical_for_signature(
            client=client, brand=brand, signature=signature
        )
        if existing is None:
            # Re-query empty after a UniqueViolation means a different
            # bug shape — record it as unrevertable for the brand so
            # promotion skips this brand.
            logger.warning(
                f"consolidator: UniqueViolation recovery failed — re-query "
                f"returned no canonical for brand={brand}, sig={signature[:24]}..."
            )
            result.errors.append(
                f"dedup recovery {loser_memory_id}: re-query empty after UniqueViolation"
            )
            result.mark_brand_dedup_error(brand)
            return False

        # Build the synthetic incoming list: loser + siblings, each
        # carrying their OWN dedup_counter (iter-3 new-NEW-H1). When
        # loser_row is supplied, read its own counter; otherwise
        # default to 1 (the row was an unstamped fresh candidate).
        loser_own_counter = int(loser_row.get("dedup_counter") or 1) if loser_row is not None else 1
        incoming: List[Dict[str, Any]] = [
            {"memory_id": loser_memory_id, "dedup_counter": loser_own_counter}
        ]
        for s in siblings:
            incoming.append(
                {
                    "memory_id": s["memory_id"],
                    "dedup_counter": int(s.get("dedup_counter") or 1),
                }
            )

        # Capture pre-merge state so we can measure success.
        pre_errors = list(result.errors)
        self._merge_into_existing_canonical(
            client=client,
            group_brand=brand,
            signature=signature,
            existing_canonical=existing,
            incoming=incoming,
            result=result,
        )
        # Merge records its own per-row errors; if NEW ones were
        # appended above pre_errors, we treat as partial-failure for
        # this recovery. The brand-error marking inside
        # _merge_into_existing_canonical handles unrevertable cases.
        return len(result.errors) == len(pre_errors)

    # ---------------------------------------------------------- semantic tier

    async def _promote_to_semantic(self, result: ConsolidationResult, brand: Optional[str]) -> None:
        """
        Find causal_paths that have been confirmed >= N times AND have a
        passing validation_status, and stamp ``consolidated_at = now()``.

        We treat the count of distinct episodic_memories citing the path
        as the "confirmation count" — these come from independent
        cognitive cycles, so seeing the same path several times is a
        strong signal of robustness.

        Iter-2 new-H1: brands flagged in
        ``result.brands_with_dedup_errors`` are SKIPPED to avoid
        double-counting on a bumped-but-unreverted ``dedup_counter``.
        When ``brand`` is supplied AND flagged, the whole call
        short-circuits at entry; when ``brand`` is None (sweep mode),
        per-candidate paths are filtered out.
        """
        # Brand-scoped run with the scope already flagged: short-circuit
        # at entry. Record a skip-message so the caller sees why.
        if brand and brand in result.brands_with_dedup_errors:
            msg = f"skip-promotion-for-{brand}-due-to-dedup-error"
            logger.warning(f"consolidator: {msg}")
            result.errors.append(msg)
            return

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
            # iter-2 new-H1: skip paths whose brand had an unreverted
            # dedup error. Record once per brand to avoid spam.
            if path_brand and path_brand in result.brands_with_dedup_errors:
                msg = f"skip-promotion-for-{path_brand}-due-to-dedup-error"
                if msg not in result.errors:
                    logger.warning(f"consolidator: {msg}")
                    result.errors.append(msg)
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
