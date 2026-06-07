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
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

from pydantic import BaseModel, Field, field_validator

from src.memory.services.factories import get_supabase_client
from src.mlops.lifecycle_monitoring import record_consolidation_sweep

logger = logging.getLogger(__name__)


# Promotion thresholds. Configurable via env later; conservative defaults.
SEMANTIC_MIN_CONFIRMATIONS = 3  # episodic memories citing the same causal_path
PROCEDURAL_MIN_USAGE = 5  # procedural_memories.usage_count
PROCEDURAL_MIN_SUCCESS_RATE = 0.8  # success_rate field on procedural memories


# Dedup signature version. Bump when the signature shape changes so the
# (brand, dedup_signature) partial-unique-index treats old + new
# generations as distinct rather than colliding mid-rollout.
DEDUP_SIGNATURE_VERSION = "v1"


# -------------------------------------------------------------------------
# Procedural-template extraction (issue #389 — Phase 3 §3.4)
# -------------------------------------------------------------------------
# Template-signature version. Bump in lockstep with any change to
# ``_compute_template_signature`` so old + new generations live as
# distinct rows on the (brand, template_signature) partial-unique-index
# instead of colliding mid-rollout.
TEMPLATE_SIGNATURE_VERSION = "v1"

# Cluster N_MIN — minimum rows per (brand, event_type, event_subtype,
# sorted(action_keys)) cluster before an extraction attempt fires. Below
# this threshold the cluster is "too small to call a pattern" and is
# skipped. Mirrors the conservative N=3 default used by
# ``SEMANTIC_MIN_CONFIRMATIONS`` (above) — three independent
# observations are the minimum signal for "real recurring pattern" vs
# "incidental noise".
PROCEDURAL_TEMPLATE_MIN_CLUSTER_SIZE = 3

# Confidence threshold below which a template is NOT promoted. Cohesion
# below 0.3 indicates the cluster is mostly noise (rows that share the
# cluster key but have widely-divergent action_keys); promoting such a
# row would generate a misleading "template" that the downstream
# matching surface would over-trigger on. The threshold is documented
# in code comments alongside the formula calibration.
PROCEDURAL_TEMPLATE_MIN_CONFIDENCE = 0.3

# Feature flag for the LLM-augmented extraction path. When set to a
# truthy env value the symbolic-cohesion confidence is multiplied by an
# LLM-rated coherence in [0..1]; on SDK exception (narrow catch on the
# four anthropic.* error classes) the consolidator falls back to the
# pure symbolic path. Default off keeps the consolidator deterministic
# and free of per-cluster LLM cost in dev/CI.
PROCEDURAL_LLM_EXTRACTION_ENV_VAR = "PROCEDURAL_LLM_EXTRACTION_ENABLED"


# Anthropic client protocol — kept structural so the real SDK class
# ``anthropic.AsyncAnthropic`` (whose ``messages`` is a class-level
# property → "read-only attribute" in mypy structural subtyping)
# satisfies it without further annotation gymnastics. Mirrors the
# crystallizer's ``_AnthropicClientProtocol`` definition.
class _AnthropicMessagesProtocol(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _AnthropicClientProtocol(Protocol):
    @property
    def messages(self) -> _AnthropicMessagesProtocol: ...


_AnthropicClientFactory = Callable[[str], _AnthropicClientProtocol]


def _llm_extraction_enabled() -> bool:
    """Return True iff the operator explicitly opted in to the LLM
    extraction path via env. Default off.

    Truthy values: ``"1"``, ``"true"``, ``"yes"``, ``"on"`` (case-
    insensitive). Matches the crystallizer's ``_llm_narratives_enabled``
    so operators have a uniform on/off vocabulary across the memory
    lifecycle.
    """
    raw = os.environ.get(PROCEDURAL_LLM_EXTRACTION_ENV_VAR, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class ProceduralTemplate(BaseModel):
    """Pydantic schema for one extracted procedural template (issue #389).

    Serialized to JSONB via ``template_body`` on the
    ``procedural_templates`` table (migration 027).

    Design choices (per #389 binding decisions, mirror Decision 2 =
    HYBRID from PR #384):

    * **Pydantic, not Jinja2 / free-form text.** Cross-language
      consumers (Python + TypeScript dashboards) decode the JSONB
      payload by schema, not template engine. Jinja2 is fragile across
      runtimes; free-form ``{var}`` text loses type-safety on
      round-trip.
    * **``shared_action_keys``: List[str].** Concrete intersection of
      ``raw_content["action_keys"]`` across the cluster's rows — the
      KEYS the template captures as REQUIRED for every instance.
    * **``variables``: List[str].** Per-instance placeholder keys —
      the union-minus-intersection of OTHER ``raw_content`` keys
      across the cluster. These are the dimensions that vary between
      instances (e.g. ``hcp_id`` / ``region`` / ``cohort``); a
      consumer materializing the template would bind these.
    * **``extraction_confidence``: float in [0..1].** Mean pairwise
      Jaccard cohesion over ``action_keys`` sets (deterministic) ×
      optional LLM coherence multiplier when the flag is on. Range
      pinned by both the Pydantic ``Field(ge=0, le=1)`` validator AND
      the DB ``CHECK (extraction_confidence >= 0 AND <= 1)`` (defense
      in depth).
    * **``extraction_method``: Literal['symbolic' | 'llm_with_fallback'].**
      Pinned to the two paths the consolidator actually emits;
      mismatched values are caught at Pydantic construction AND at the
      DB CHECK constraint.

    V2 follow-ups (out of scope here, will be filed as separate issues
    once production observability confirms #389 V1 lands cleanly):

    * Template revision/versioning (e.g. ``template_version: int``).
    * Embedding-similarity clustering basis (replacing the
      exact-match key-tuple).
    * Cross-brand templates (forbidden in V1 per the brand-tenant
      boundary).
    """

    brand: str
    template_signature: str
    event_type: str
    event_subtype: str
    shared_action_keys: List[str]
    variables: List[str]
    derived_from_episodic_ids: List[str]
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: Literal["symbolic", "llm_with_fallback"]

    @field_validator("shared_action_keys", "variables", "derived_from_episodic_ids")
    @classmethod
    def _no_empty_strings(cls, v: List[str]) -> List[str]:
        # Defensive: reject any empty strings inside list payloads —
        # they cannot represent a real action key / variable name /
        # memory id and would silently break downstream matching.
        if any(not (isinstance(item, str) and item.strip()) for item in v):
            raise ValueError("list values must be non-empty strings")
        return v


def _compute_template_signature(
    *,
    brand: Optional[str],
    event_type: Optional[str],
    event_subtype: Optional[str],
    action_keys: List[str],
) -> Optional[str]:
    """Pure helper returning a deterministic template key for one
    cluster's (brand, event_type, event_subtype, sorted(action_keys))
    tuple.

    Mirrors ``_compute_dedup_signature`` (which exists at the top of
    this module) — same versioned-prefix shape, same brand-included-in-
    every-variant defense-in-depth, same SHA-256 over a delimited
    payload.

    Cluster basis (V1): exact-match key-tuples. Embedding-similarity
    is V2 follow-up. The four tuple-fields are the cluster key;
    different values for ANY field produce a distinct signature.

    Returns ``None`` when any of the four required inputs is missing
    or when ``action_keys`` is empty. In that case the row is not safe
    to template and the caller skips it.
    """
    if not brand or not event_type or not event_subtype or not action_keys:
        return None
    # sorted(action_keys) means the signature is invariant to encounter
    # order — two rows with action_keys=["a","b"] and ["b","a"] map to
    # the same cluster. Decision §3.4 explicitly calls this out.
    sorted_keys = sorted(action_keys)
    payload = "|".join(
        (
            "template",
            str(brand),
            str(event_type),
            str(event_subtype),
            "+".join(sorted_keys),
        )
    )
    sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{TEMPLATE_SIGNATURE_VERSION}:{sig}"


def _jaccard_cohesion(members: Sequence[Mapping[str, Any]]) -> float:
    """Mean pairwise Jaccard similarity over per-row ``action_keys`` sets.

    Confidence score for V1 extraction. Bounded in [0..1] by definition
    of the Jaccard index (|A∩B| / |A∪B|). Calibration:

    * All sets identical → cohesion = 1.0 (perfect agreement; the
      cluster IS a template).
    * All sets disjoint → cohesion = 0.0 (no agreement; just rows
      sharing a cluster key by coincidence — should be filtered).
    * Half-overlapping → cohesion in (0, 1); threshold 0.3 rejects
      "mostly noise" clusters before they reach the DB.

    A single-member cluster trivially has cohesion 1 (no pairs to
    average over; returns the limit).

    Empty members → 0 (defensive; the caller should never pass an
    empty list — the cluster-grouping step filters those out — but
    this keeps the function total).
    """
    if not members:
        return 0.0
    if len(members) == 1:
        # Singleton: the limit as the pair count -> 0 is "perfect"
        # agreement (there's nothing to disagree with). Conventional
        # choice — pyramid-up to the symbolic-only ceiling so the
        # N_MIN filter (not the cohesion filter) is the gate that
        # rejects too-small clusters.
        return 1.0

    sets: List[Set[str]] = [{str(k) for k in (m.get("action_keys") or [])} for m in members]
    n = len(sets)
    total = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            union = a | b
            if not union:
                # Both sets empty: convention — treat as agreement.
                total += 1.0
            else:
                total += len(a & b) / len(union)
            pair_count += 1
    return total / pair_count if pair_count else 0.0


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
    ``brands_with_dedup_errors`` so ``_promote_to_semantic`` can
    short-circuit instead of double-counting on the inconsistent
    counter. (``_promote_to_procedural`` intentionally ignores this
    set — procedural graduation keys off usage_count/success_rate, not
    the dedup counter, so it cannot be double-counted by a dedup error.)
    """

    promoted_to_semantic: int = 0
    promoted_to_procedural: int = 0
    causal_paths_examined: int = 0
    procedural_examined: int = 0
    # Dedup metrics (#388): episodic rows examined / collapsed.
    episodic_dedup_examined: int = 0
    episodic_dedup_collapsed: int = 0
    # Procedural-template metrics (#389): templates extracted from
    # clustered episodic memories in this pass. Reported in addition
    # to ``promoted_to_procedural`` (counter-threshold path); both
    # paths can fire in the same pass.
    procedural_templates_extracted: int = 0
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
        ``_promote_to_semantic`` skips this brand to avoid
        double-counting on the inconsistent counter.
        (``_promote_to_procedural`` ignores this set by design —
        it keys off usage_count/success_rate, not the dedup counter.)"""
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
        anthropic_client_factory: Optional[_AnthropicClientFactory] = None,
    ):
        self.semantic_min_confirmations = semantic_min_confirmations
        self.procedural_min_usage = procedural_min_usage
        self.procedural_min_success_rate = procedural_min_success_rate
        # Parameter-DI for the procedural-template LLM-augmented path
        # (issue #389). Mirror of the crystallizer's
        # ``anthropic_client_factory`` (PR #384). Tests inject a fake
        # factory; production passes nothing → the helper resolves the
        # default ``anthropic.AsyncAnthropic`` factory. Forbidden
        # alternative per [[feedback-test-must-exercise-real-catch-
        # not-mock]] + [[feedback-codex-audits-within-existing-
        # signature-not-design]]: monkey-patching
        # ``anthropic_mod.AsyncAnthropic`` (bypasses the real catch
        # surface AND is xdist-fragile).
        self._anthropic_client_factory = anthropic_client_factory

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
            3. ``_promote_to_procedural`` — graduates procedural memories
               via counter-threshold (the legacy path).
            4. ``extract_procedural_templates`` — emits one template
               per clustered episodic memories (#389 cluster-then-extract
               path). Runs LAST because it reads ``episodic_memories``
               in its already-deduplicated state.

        Iter-1 codex H1 fix: step 4 was previously a public method on
        Consolidator but not wired into ``run()``, so scheduled Celery
        passes would never emit templates. Now wired with the same
        try/except envelope as the other phases; failures record an
        error and do NOT short-circuit the rest of the pipeline.
        """
        result = ConsolidationResult()
        # Monotonic clock for the #391 monitoring duration metric. Wall
        # clock would be subject to system-time adjustments mid-run.
        sweep_started_monotonic = time.monotonic()
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
        try:
            # Iter-2 codex M1 fix: thread ``result`` through so non-
            # idempotent insert failures surface on
            # ``result.errors`` instead of being silently logged.
            n_templates = await self.extract_procedural_templates(brand=brand, result=result)
            result.procedural_templates_extracted += n_templates
        except Exception as exc:
            logger.exception("consolidator: procedural-template extraction failed")
            result.errors.append(f"procedural-template: {exc}")
        result.finished_at = datetime.now(timezone.utc)

        # #391 monitoring box 1.c + 2.b (promotion_rate): emit Opik
        # trace + MLflow promotion-rate gauge for the sweep.
        #
        # Codex iter-0 M1 closure: when ``brand`` is None (whole-
        # portfolio sweep) we EMIT ONE METRIC PER BRAND actually
        # touched (from ``result.by_brand``) instead of collapsing
        # everything into an ``_all_`` bucket that would obscure
        # cross-brand divergence. The Opik trace still emits ONCE
        # (full-sweep observable) tagged with ``_all_``.
        #
        # When ``brand`` is set (single-brand sweep), we just emit one
        # metric tagged with that brand. Best-effort by design —
        # helper swallows its own exceptions.
        sweep_duration_ms = (time.monotonic() - sweep_started_monotonic) * 1000.0
        if brand is None:
            # Per-brand fanout — emit one entry per brand touched.
            # Without the fanout the brand=None case would land in a
            # single ``_all_`` MLflow bucket and downstream dashboards
            # would lose per-brand promotion-rate signal.
            #
            # Per-brand denominators: result.causal_paths_examined is
            # the TOTAL across all brands; we don't have a per-brand
            # split here. The compromise is to attribute the SAME
            # global denominator to each brand's emission so the
            # ratio is comparable across brands (each brand sees the
            # same "what fraction of all sweep candidates got
            # promoted in MY brand" perspective). Per-brand
            # denominators would require deeper plumbing through
            # _promote_to_semantic — filed as a forward enhancement.
            touched_brands = list(result.by_brand.keys()) or ["_all_"]
            for b in touched_brands:
                bucket = result.by_brand.get(b, {})
                record_consolidation_sweep(
                    brand=b,
                    dedup_collapses=int(bucket.get("dedup_collapsed", 0)),
                    promotions_to_semantic=int(bucket.get("semantic", 0)),
                    promotions_to_procedural=int(bucket.get("procedural", 0)),
                    templates_extracted=result.procedural_templates_extracted
                    if b == touched_brands[0]
                    else 0,
                    duration_ms=sweep_duration_ms,
                    causal_paths_examined=result.causal_paths_examined,
                )
        else:
            # Single-brand sweep: just emit one metric for the brand.
            record_consolidation_sweep(
                brand=brand,
                dedup_collapses=result.episodic_dedup_collapsed,
                promotions_to_semantic=result.promoted_to_semantic,
                promotions_to_procedural=result.promoted_to_procedural,
                templates_extracted=result.procedural_templates_extracted,
                duration_ms=sweep_duration_ms,
                causal_paths_examined=result.causal_paths_examined,
            )

        logger.info(
            f"consolidator finished promoted_semantic={result.promoted_to_semantic} "
            f"promoted_procedural={result.promoted_to_procedural} "
            f"dedup_collapsed={result.episodic_dedup_collapsed} "
            f"procedural_templates_extracted={result.procedural_templates_extracted} "
            f"errors={len(result.errors)}"
        )
        return result

    # --------------------------------------------------------- pagination

    def _select_all_rows(
        self,
        build_query: Callable[[], Any],
        *,
        page_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Fetch every row for a query, paginating past PostgREST's row cap.

        PostgREST returns at most ``db-max-rows`` (default 1000) per request and
        the deployment configures no override, so an un-ranged SELECT silently
        truncates large candidate sets (review finding H5). Callers that need
        the full set MUST paginate. The supabase query object is single-use, so
        ``build_query`` is a thunk that re-creates the base query (filters
        included) for each page; we then walk ``.range()`` windows until a short
        page signals exhaustion.

        Args:
            build_query: zero-arg callable returning a fresh, filtered query
                builder (without ``.range()`` / ``.execute()`` applied).
            page_size: rows per page; must stay at or under the server cap.

        Returns:
            All matching rows across every page, in server order.
        """
        all_rows: List[Dict[str, Any]] = []
        offset = 0
        while True:
            page = (build_query().range(offset, offset + page_size - 1).execute().data) or []
            all_rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return all_rows

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
        def _build_dedup_query() -> Any:
            q = client.table("episodic_memories").select(
                "memory_id, brand, region, event_type, event_subtype, "
                "causal_path_id, agent_name, description, occurred_at, "
                "dedup_signature, dedup_counter"
            )
            if brand:
                q = q.eq("brand", brand)
            if region:
                q = q.eq("region", region)
            # IS NULL filter: only candidates without a signature yet.
            return q.is_("dedup_signature", "null")

        try:
            # Paginate past the PostgREST row cap so a large dedup backlog is
            # not silently truncated (review finding H5).
            rows = self._select_all_rows(_build_dedup_query)
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

    @staticmethod
    def _is_unique_violation_or_postgrest_23505(exc: BaseException) -> bool:
        """Detect a DB partial-unique-index violation across BOTH the
        psycopg/UniqueViolation shape AND the postgrest/APIError-with-
        SQLSTATE-23505 shape.

        Background (iter-1 codex M1 fix): the procedural-template
        writer goes through ``supabase-py``, which surfaces Postgres
        errors as ``postgrest.exceptions.APIError`` — class name does
        NOT contain "UniqueViolation". The pre-existing
        :meth:`_is_unique_violation` was designed for the dedup path
        which uses psycopg directly, so its class-name signal misses
        postgrest. This new helper is the WIDENED variant — used by
        the procedural-template insert path only.

        Path 1 (psycopg/test-stub): delegates to
        :meth:`_is_unique_violation` (class name + message both
        match).

        Path 2 (postgrest APIError):
          * Class name contains ``"APIError"`` (case-insensitive).
          * Object exposes ``code == "23505"`` (PostgreSQL SQLSTATE
            for unique_violation per https://www.postgresql.org/docs/
            current/errcodes-appendix.html).

        Both signals required for path 2 → no false positives from
        unrelated PostgREST errors (foreign-key violations,
        permission errors, etc.) that also surface as APIError.
        """
        # Path 1: psycopg/UniqueViolation shape (delegate to existing).
        if Consolidator._is_unique_violation(exc):
            return True

        # Path 2: postgrest APIError with SQLSTATE 23505.
        cls_name = type(exc).__name__.lower()
        if "apierror" not in cls_name:
            return False
        code = getattr(exc, "code", None)
        return str(code) == "23505"

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

        # First pass: select the promotable candidates (skip overturned paths
        # and brands flagged with an unrevertable dedup error), then count
        # confirmations for ALL of them in ONE batched query below.
        promotable: List[Dict[str, Any]] = []
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
            promotable.append(path)

        if not promotable:
            return

        # M3 (#694): count effective episodic confirmations for ALL promotable
        # paths in a SINGLE batched query, not one SELECT per path (was an N+1).
        # After episodic deduplication (#388), a row's ``dedup_counter`` is the
        # number of underlying events it represents — so the effective
        # confirmation count is SUM(dedup_counter) per causal_path, NOT
        # COUNT(*). supabase-py exposes no SUM() helper via the .table()
        # builder, so we pull (causal_path_id, dedup_counter) for the whole
        # candidate set in one .in_() query and aggregate in Python.
        # episodic_memories.causal_path_id is indexed, so the .in_() uses it.
        path_ids = [p["path_id"] for p in promotable]
        confirmations_by_path: Dict[str, int] = {}
        try:
            rows_result = (
                client.table("episodic_memories")
                .select("memory_id, dedup_counter, causal_path_id")
                .in_("causal_path_id", path_ids)
                .execute()
            )
            for r in rows_result.data or []:
                cpid = r.get("causal_path_id")
                if cpid is None:
                    continue
                # SUM(dedup_counter) — treats missing counter as 1
                # (back-compat for rows from before migration 026).
                confirmations_by_path[cpid] = confirmations_by_path.get(cpid, 0) + int(
                    r.get("dedup_counter") or 1
                )
        except Exception as exc:
            logger.warning(f"consolidator: batched confirmation count failed: {exc}")
            result.errors.append(f"count batch: {exc}")
            return

        # Second pass: threshold check + stamp consolidated_at per promoted path.
        for path in promotable:
            path_id = path["path_id"]
            path_brand = path.get("brand")
            confirmation_count = confirmations_by_path.get(path_id, 0)
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

    # ---------------------------------------- procedural-template extraction
    # Issue #389 Phase 3 §3.4. Public sibling to ``_promote_to_procedural``
    # below — same tier (procedural memory), different mechanism. The
    # counter-threshold promotion below is for INDIVIDUAL procedural
    # rows; this method emits TEMPLATES extracted from CLUSTERED episodic
    # rows (one template per cluster, capturing shared structure +
    # variables). Both can be enabled independently.

    async def extract_procedural_templates(
        self,
        brand: Optional[str] = None,
        *,
        result: Optional[ConsolidationResult] = None,
    ) -> int:
        """Extract procedural templates from clustered episodic memories.

        V1 design (binding decisions per issue #389):

        * **Clustering basis**: exact-match key-tuples ``(brand,
          event_type, event_subtype, sorted(action_keys))``. NOT
          embedding similarity (V2 follow-up).
        * **Threshold**: clusters with effective size <
          ``PROCEDURAL_TEMPLATE_MIN_CLUSTER_SIZE`` (default 3) are
          skipped. Iter-1 codex H1 fix: effective size is
          ``SUM(dedup_counter)`` across the cluster's rows (NOT row
          count) — mirrors the ``_promote_to_semantic`` threshold
          logic. Required because ``deduplicate_episodic`` runs FIRST
          in ``Consolidator.run()`` and collapses multiple
          observations into a single canonical row with a
          ``dedup_counter`` recording how many underlying events the
          canonical represents. Without this, 3 identical-key
          observations would collapse to 1 row and never meet the
          cluster threshold downstream.
        * **Confidence**: mean pairwise Jaccard cohesion over
          ``action_keys`` sets (deterministic, in [0..1]). When the
          LLM flag is on AND the SDK call succeeds, multiplied by an
          LLM-rated coherence in [0..1]. Below ``PROCEDURAL_TEMPLATE_
          MIN_CONFIDENCE`` (default 0.3), the template is NOT
          promoted (noise rejection).
        * **Symbolic always runs first**; LLM augments when flag on.
        * **Brand boundary preserved** — no cross-brand templates.
        * **Idempotency**: re-extraction on the same cluster swallows
          the DB partial-unique-index violation and reports 0 new
          templates added.

        Returns the count of templates ACTUALLY inserted (excludes
        skipped + idempotent re-extraction).

        Iter-2 codex M1 fix: when ``result`` is supplied, non-
        idempotent insert failures (e.g. PostgREST APIError with
        SQLSTATE 23503 = foreign-key-violation, 42P01 = undefined-
        table, permission errors) are recorded as
        ``result.errors`` entries so they surface in the
        consolidator's run summary instead of being silently
        logged. Idempotent (unique-violation) skips remain silent
        because they are expected on re-extraction. The legacy
        callers that pass no ``result`` are unaffected.
        """
        client = get_supabase_client()

        # Pull candidate episodic rows scoped to brand (when provided).
        # We need brand + event_type + event_subtype + raw_content
        # (for action_keys) + memory_id (for derived_from_episodic_ids).
        # occurred_at is included only for stable per-cluster ordering
        # of derived_from_episodic_ids.
        # ``dedup_counter`` is read so the cluster-size threshold can
        # use SUM(dedup_counter) (iter-1 codex H1 fix). Falls back to
        # 1 when missing (back-compat with rows from before migration
        # 026 or test fixtures that don't seed the column).
        def _build_template_query() -> Any:
            q = client.table("episodic_memories").select(
                "memory_id, brand, event_type, event_subtype, raw_content, occurred_at, dedup_counter"
            )
            if brand:
                q = q.eq("brand", brand)
            return q

        try:
            # Paginate past the PostgREST row cap: this candidate set has NO
            # IS NULL filter, so it grows with the table and would otherwise be
            # silently truncated, corrupting SUM(dedup_counter) cluster sizing
            # (review finding H5).
            rows = self._select_all_rows(_build_template_query)
        except Exception as exc:
            logger.warning(f"consolidator: procedural-template select failed: {exc}")
            return 0

        # Group by the cluster-key tuple. ``sorted(action_keys)`` is
        # part of the key — rows with different action_keys land in
        # different clusters (defense in depth alongside the per-row
        # Jaccard cohesion check).
        groups: Dict[Tuple[str, str, str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(
            list
        )
        for row in rows:
            row_brand = row.get("brand")
            row_event_type = row.get("event_type")
            row_event_subtype = row.get("event_subtype")
            raw_content = row.get("raw_content") or {}
            action_keys = raw_content.get("action_keys")
            if not (
                row_brand
                and row_event_type
                and row_event_subtype
                and isinstance(action_keys, list)
                and action_keys
            ):
                # Row lacks the required cluster-key fields — skip
                # (mirrors ``_compute_template_signature``'s None
                # return path).
                continue
            sorted_keys = tuple(sorted(str(k) for k in action_keys))
            key = (
                str(row_brand),
                str(row_event_type),
                str(row_event_subtype),
                sorted_keys,
            )
            # Normalise the row's action_keys to the sorted form so
            # ``_jaccard_cohesion`` sees the same values _compute_
            # template_signature uses.
            row_normalised = dict(row)
            row_normalised["action_keys"] = list(sorted_keys)
            groups[key].append(row_normalised)

        if not groups:
            return 0

        templates_inserted = 0
        llm_enabled = _llm_extraction_enabled()

        for (g_brand, g_event_type, g_event_subtype, g_sorted_keys), members in groups.items():
            # Iter-1 codex H1 fix: use SUM(dedup_counter) for cluster
            # sizing, not raw row count. ``deduplicate_episodic`` runs
            # before this method in ``run()`` and collapses identical
            # observations into a single canonical row with
            # ``dedup_counter`` recording the underlying count. A
            # cluster of 3 observations that share the same dedup
            # signature collapses to 1 row with dedup_counter=3 — that
            # should STILL trigger template extraction.
            effective_cluster_size = sum(int(m.get("dedup_counter") or 1) for m in members)
            if effective_cluster_size < PROCEDURAL_TEMPLATE_MIN_CLUSTER_SIZE:
                continue

            # Symbolic-cohesion confidence — ALWAYS computed first.
            symbolic_confidence = _jaccard_cohesion(members)

            # LLM augmentation: only when flag on. The helper handles
            # the SDK exception narrow-catch internally and returns
            # ``(extraction_method, confidence_multiplier)``. On
            # exception path the multiplier is None and the method
            # downgrades to 'symbolic' so the row insertion records
            # the actual code path that ran (not the WANTED path).
            extraction_method: Literal["symbolic", "llm_with_fallback"] = "symbolic"
            effective_confidence = symbolic_confidence

            if llm_enabled:
                multiplier = await _invoke_llm_coherence_rater(
                    brand=g_brand,
                    event_type=g_event_type,
                    event_subtype=g_event_subtype,
                    action_keys=list(g_sorted_keys),
                    member_count=len(members),
                    client_factory=self._anthropic_client_factory,
                )
                if multiplier is not None:
                    extraction_method = "llm_with_fallback"
                    effective_confidence = symbolic_confidence * multiplier

            if effective_confidence < PROCEDURAL_TEMPLATE_MIN_CONFIDENCE:
                logger.info(
                    f"consolidator: template noise-rejected "
                    f"(brand={g_brand}, subtype={g_event_subtype}, "
                    f"cohesion={effective_confidence:.3f} < "
                    f"{PROCEDURAL_TEMPLATE_MIN_CONFIDENCE})"
                )
                continue

            signature = _compute_template_signature(
                brand=g_brand,
                event_type=g_event_type,
                event_subtype=g_event_subtype,
                action_keys=list(g_sorted_keys),
            )
            if signature is None:
                # Defensive: _compute_template_signature returns None
                # only when the cluster-key fields are missing — but
                # the grouping above filters those out, so this branch
                # is unreachable in practice. Skip silently.
                continue

            # shared_action_keys = sorted intersection of action_keys
            # sets across the cluster's members. Sort for stable
            # ordering (matches the cluster-key sort).
            action_key_sets: List[Set[str]] = [{str(k) for k in m["action_keys"]} for m in members]
            shared_action_keys = sorted(set.intersection(*action_key_sets))

            # variables = per-instance differing keys across the
            # cluster's raw_content. A key is a "variable" if EITHER
            # (a) it appears in some rows but not others (presence
            # variance — e.g. only m1 carries "region"), OR (b) it
            # appears in ALL rows but with DIFFERENT values across
            # rows (value variance — e.g. every row has "hcp_id" but
            # each row's hcp_id is distinct). Iter-1 fix: codex iter-0
            # H2 — the original formula missed case (b) and the
            # ProceduralTemplate docstring's own example (hcp_id /
            # region / cohort) explicitly listed value-variance keys.
            #
            # The cluster basis key ("action_keys") is excluded — it
            # is captured separately as shared_action_keys.
            raw_key_union: Set[str] = set()
            raw_key_intersection: Optional[Set[str]] = None
            # Track distinct values per key for value-variance
            # detection. Use a hashable repr (JSON-serialise + sort
            # at the boundary) because raw_content values may be
            # nested dicts/lists which are not directly hashable.
            import json as _value_repr_json

            values_per_key: Dict[str, Set[str]] = defaultdict(set)
            for m in members:
                raw_content = m.get("raw_content") or {}
                keys = set(raw_content.keys()) - {"action_keys"}
                raw_key_union |= keys
                raw_key_intersection = (
                    keys if raw_key_intersection is None else (raw_key_intersection & keys)
                )
                for k in keys:
                    try:
                        repr_val = _value_repr_json.dumps(
                            raw_content[k], sort_keys=True, default=str
                        )
                    except (TypeError, ValueError):
                        # Defensive: defer to str() repr if the value
                        # isn't JSON-serialisable. The repr is for
                        # set-membership only, never persisted, so a
                        # lossy fallback is acceptable.
                        repr_val = str(raw_content[k])
                    values_per_key[k].add(repr_val)

            presence_variant = raw_key_union - (raw_key_intersection or set())
            value_variant = {
                k for k in (raw_key_intersection or set()) if len(values_per_key[k]) > 1
            }
            variables = sorted(presence_variant | value_variant)

            # derived_from_episodic_ids: stable sort by occurred_at +
            # memory_id (matches the dedup canonical-pick rule for
            # consistency).
            members_sorted = sorted(
                members,
                key=lambda r: (
                    r.get("occurred_at") or "",
                    str(r.get("memory_id") or ""),
                ),
            )
            derived_from_episodic_ids = [str(m["memory_id"]) for m in members_sorted]

            # Pydantic ProceduralTemplate construction. Pydantic
            # validates the confidence range + method literal at
            # this point — invalid values raise ValidationError and
            # are NOT silently caught (programming error per the
            # narrow-catch contract).
            template = ProceduralTemplate(
                brand=g_brand,
                template_signature=signature,
                event_type=g_event_type,
                event_subtype=g_event_subtype,
                shared_action_keys=shared_action_keys,
                variables=variables,
                derived_from_episodic_ids=derived_from_episodic_ids,
                extraction_confidence=float(effective_confidence),
                extraction_method=extraction_method,
            )

            # Persist. JSONB-shaped ``template_body`` excludes
            # provenance / signature / confidence / method (those go
            # to dedicated columns); body is the BUSINESS surface
            # consumers read.
            template_body = {
                "event_type": template.event_type,
                "event_subtype": template.event_subtype,
                "shared_action_keys": template.shared_action_keys,
                "variables": template.variables,
            }
            insert_payload = {
                "brand": template.brand,
                "template_signature": template.template_signature,
                "template_body": template_body,
                "derived_from_episodic_ids": template.derived_from_episodic_ids,
                "extraction_confidence": template.extraction_confidence,
                "extraction_method": template.extraction_method,
            }
            try:
                client.table("procedural_templates").insert(insert_payload).execute()
                templates_inserted += 1
                logger.info(
                    f"consolidator: extracted procedural template "
                    f"(brand={g_brand}, sig={signature[:24]}..., "
                    f"effective_cluster_size={effective_cluster_size}, "
                    f"row_count={len(members)}, "
                    f"confidence={effective_confidence:.3f}, "
                    f"method={extraction_method})"
                )
            except Exception as exc:
                # Idempotency: DB partial-unique-index UniqueViolation
                # is the EXPECTED shape on re-extraction (V1 has no
                # revision logic). Swallow as no-op; other exception
                # shapes are RECORDED on result.errors (iter-2 codex M1
                # fix) so they surface in the consolidator's run
                # summary instead of being silently logged.
                #
                # Iter-1 codex M1 fix: use the widened helper that
                # ALSO accepts postgrest.APIError with SQLSTATE 23505
                # (supabase-py surfaces unique-violations through this
                # class, NOT through psycopg's UniqueViolation). The
                # dedup path is unchanged because it doesn't go
                # through supabase-py.
                if self._is_unique_violation_or_postgrest_23505(exc):
                    logger.info(
                        f"consolidator: procedural template already exists "
                        f"(brand={g_brand}, sig={signature[:24]}...) — "
                        "skipping idempotent re-insert"
                    )
                else:
                    logger.warning(
                        f"consolidator: procedural template insert failed "
                        f"(brand={g_brand}, sig={signature[:24]}...): {exc}"
                    )
                    if result is not None:
                        result.errors.append(
                            f"procedural-template insert "
                            f"(brand={g_brand}, sig={signature[:24]}...): {exc}"
                        )

        return templates_inserted

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
            # L10 (#694): do NOT guard on ``applicable`` being truthy — an EMPTY
            # applicable_brands lists no brand, so under a scoped run it must be
            # SKIPPED (not promoted). The old ``and applicable`` clause let an
            # empty list fall through and promote. (Write paths coerce to
            # ['all'] today, so this is a defensive tightening.)
            if brand and brand not in applicable and "all" not in applicable:
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


async def _invoke_llm_coherence_rater(
    *,
    brand: str,
    event_type: str,
    event_subtype: str,
    action_keys: List[str],
    member_count: int,
    client_factory: Optional[_AnthropicClientFactory] = None,
) -> Optional[float]:
    """Invoke the Haiku coherence rater for one procedural-template cluster.

    Returns the rater's coherence score in [0..1], or ``None`` if the
    LLM path failed and the caller should fall back to the symbolic-
    only confidence.

    PRODUCTION shape: thin async wrapper around
    :class:`anthropic.AsyncAnthropic` (NOT the sync
    :class:`anthropic.Anthropic` — the consolidator is async-end-to-
    end; a sync client would block the event loop).

    UNIT TESTS: pass a fake ``client_factory`` so no network call
    happens. Forbidden alternative per [[feedback-test-must-exercise-
    real-catch-not-mock]]: ``monkeypatch.setattr(anthropic, ...)`` —
    bypasses the production catch surface.

    Exception path: narrow-caught on the four anthropic.* error
    classes (``APIConnectionError``, ``APITimeoutError``,
    ``RateLimitError``, ``APIStatusError``). Programming errors
    (TypeError, AttributeError, KeyError) PROPAGATE per the narrow-
    catch contract (mirror crystallizer ``_invoke_llm_narrator``).

    Returning ``None`` (rather than 1.0 / 0.0 / raising) keeps the
    caller's downgrade-to-symbolic logic explicit at the call site
    (``extraction_method = 'symbolic'`` when multiplier is None).
    """
    # Lazy import — flag-off module load does not pay for the SDK.
    anthropic_module: Optional[Any]
    try:
        import anthropic as _anthropic_module

        anthropic_module = _anthropic_module
    except ImportError:
        if client_factory is None:
            logger.warning(
                "consolidator: anthropic SDK unavailable for procedural-template "
                "LLM rater; falling back to symbolic-only"
            )
            return None
        anthropic_module = None

    # Narrow catch tuple — same shape as crystallizer narrator.
    caught_api_errors: Tuple[type[BaseException], ...]
    if anthropic_module is None:
        caught_api_errors = ()
    else:
        caught_api_errors = (
            anthropic_module.APIConnectionError,
            anthropic_module.APITimeoutError,
            anthropic_module.RateLimitError,
            anthropic_module.APIStatusError,
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key.startswith("sk-ant-"):
        # Memory `[[feedback-live-lm-skip-must-check-key-shape]]`:
        # presence-only check would let the CI placeholder
        # ANTHROPIC_API_KEY=test-key through and produce 401s. Use
        # the prefix check so empty + placeholder both short-circuit
        # to symbolic-only.
        logger.info(
            "consolidator: ANTHROPIC_API_KEY missing or placeholder; "
            "procedural-template LLM rater falling back to symbolic-only"
        )
        return None

    effective_factory: _AnthropicClientFactory
    if client_factory is not None:
        effective_factory = client_factory
    else:
        assert anthropic_module is not None  # narrowed by the import-fail branch above

        def _default_factory(key: str) -> _AnthropicClientProtocol:
            return anthropic_module.AsyncAnthropic(api_key=key)  # type: ignore[no-any-return,union-attr]

        effective_factory = _default_factory
    client = effective_factory(api_key)

    # Minimal prompt — Haiku-friendly, ~300 input tokens. Asks for a
    # single-number coherence rating + JSON response shape so parsing
    # is deterministic and cheap.
    prompt = (
        "You are rating the coherence of a candidate procedural-memory "
        "template extracted from clustered episodic memories.\n\n"
        f"Brand: {brand}\n"
        f"Event type: {event_type}\n"
        f"Event subtype: {event_subtype}\n"
        f"Shared action keys: {action_keys}\n"
        f"Number of observations: {member_count}\n\n"
        "Output a JSON object with a single key `coherence` whose value "
        "is a float in [0, 1]:\n"
        "  - 1.0 = the action keys form a coherent, reusable procedural template\n"
        "  - 0.5 = somewhat coherent but with noise / overgeneralisation\n"
        "  - 0.0 = the action keys do not form a coherent template\n\n"
        "Respond with ONLY the JSON object, no other text."
    )

    started = time.monotonic()
    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
    except caught_api_errors as exc:
        # Narrow catch: SDK / API transient errors fall back to
        # symbolic-only (return None). Programming errors propagate.
        logger.warning(
            "consolidator: procedural-template LLM rater %s: %s",
            type(exc).__name__,
            exc,
        )
        return None

    latency_ms = (time.monotonic() - started) * 1000.0
    logger.debug("consolidator: procedural-template LLM rater latency_ms=%.1f", latency_ms)

    # Extract text from response. Anthropic SDK returns
    # ``response.content`` as a list of content blocks.
    text = ""
    if hasattr(response, "content") and response.content:
        first = response.content[0]
        if hasattr(first, "text"):
            text = first.text or ""

    # Parse JSON. Bad JSON → fall back to symbolic-only (return None).
    import json as _json

    try:
        parsed = _json.loads(text.strip())
        coherence = parsed.get("coherence")
        if not isinstance(coherence, (int, float)):
            logger.warning(
                "consolidator: procedural-template LLM rater returned non-numeric "
                "coherence: %r — falling back to symbolic-only",
                coherence,
            )
            return None
        # Clamp to [0..1] — defensive against rater drift.
        return max(0.0, min(1.0, float(coherence)))
    except (_json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning(
            "consolidator: procedural-template LLM rater parse failed: %s — "
            "falling back to symbolic-only",
            exc,
        )
        return None


# Module-level convenience entrypoint for Celery / tests.
async def consolidate_insights(brand: Optional[str] = None) -> ConsolidationResult:
    """Run a consolidation pass with default thresholds."""
    return await Consolidator().run(brand=brand)
