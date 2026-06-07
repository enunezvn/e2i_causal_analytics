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

Phase 4 (#376) — schema completion
----------------------------------
Decision 2 = HYBRID (adopted 2026-05-19): each crystal carries 13
deterministic CrystalDigest fields derived from estimator state /
insight_edges / episodic-memory key_metrics, plus 2 LLM-narrative
prose fields (``limitations``, ``recommended_next_analysis``) wrapped
in an :class:`src.data.kg.types.LLMCrystalNarrativeAudit`. The LLM
path is feature-flagged via the env var
``E2I_CRYSTAL_LLM_NARRATIVES_ENABLED``; flag-off falls back to a
deterministic heuristic.

Decision 3 = KEEP BINARY (adopted 2026-05-19): the
``staleness_score`` field is intentionally NOT carried. Staleness
remains boolean via ``invalidated_at IS NULL``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from src.data.kg.types import LLMCrystalNarrativeAudit
from src.memory.services.factories import get_supabase_client
from src.mlops.lifecycle_monitoring import record_provenance_write

logger = logging.getLogger(__name__)


DEFAULT_WINDOW_DAYS = 7
DEFAULT_MIN_AGENTS = 2  # require findings from at least 2 distinct agents

# Feature flag: gates the LLM-narrator path. Empty/missing/"0" → off
# (deterministic fallback). Any of {"1", "true", "yes", "on"} (case-
# insensitive) → on.
LLM_NARRATIVE_ENV_VAR = "E2I_CRYSTAL_LLM_NARRATIVES_ENABLED"
DEFAULT_NARRATOR_MODEL = "claude-haiku-4-5-20251001"

# Default portfolio brands when crystallize_portfolio() is called with
# no explicit list. Mirrors SUPPORTED_BRANDS in
# ``src.agents.cohort_constructor.constants`` but is hard-coded here
# (not imported) to avoid a heavy import at module load time.
DEFAULT_PORTFOLIO_BRANDS: Tuple[str, ...] = ("remibrutinib", "fabhalta", "kisqali")


class _AnthropicMessagesProtocol(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _AnthropicClientProtocol(Protocol):
    # Read-only property so the real ``anthropic.AsyncAnthropic`` (whose
    # ``messages`` attribute is class-level @property → not "settable"
    # in mypy's structural-subtype check) satisfies this Protocol.
    # Without ``@property`` here mypy complains
    #   "expected settable variable, got read-only attribute".
    @property
    def messages(self) -> _AnthropicMessagesProtocol: ...


_AnthropicClientFactory = Callable[[str], _AnthropicClientProtocol]


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


def _llm_narratives_enabled() -> bool:
    """Return True iff the operator explicitly opted in to the LLM
    narrator path via env. Default off."""
    raw = os.environ.get(LLM_NARRATIVE_ENV_VAR, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class Crystallizer:
    """
    Crystallizes episodic findings into executive_insights.

    Public entrypoints:
      * ``run_for_brand(brand, region=None, window_days=7)`` — original.
      * ``crystallize_finding(finding_id, *, brand)`` — single-finding
        path; resolves the finding to its brand context and runs the
        same aggregation logic.
      * ``crystallize_portfolio(brands=None)`` — iterates the
        configured portfolio brand list and aggregates results.

    The consolidator (subsystem 2) is the natural caller — after promoting
    a causal_path to semantic, it can trigger crystallization for that
    brand. Can also be invoked on a schedule.
    """

    def __init__(
        self,
        window_days: int = DEFAULT_WINDOW_DAYS,
        min_agents: int = DEFAULT_MIN_AGENTS,
        anthropic_client_factory: Optional[_AnthropicClientFactory] = None,
        candidate_page_size: int = 1000,
    ):
        self.window_days = window_days
        self.min_agents = min_agents
        self._anthropic_client_factory = anthropic_client_factory
        # L7 (#694): page size for the candidate SELECT. PostgREST silently caps
        # a single response at 1000 rows, so we page through ``.range()`` windows
        # of this size until exhausted (overridable for tests).
        self.candidate_page_size = candidate_page_size

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
        # ``consolidation_tier`` (migration 021 line 99) is required so the
        # crystal's tier inherits from the highest tier among sources; without
        # this column in the SELECT, _derive_crystal_digest_fields defaults
        # every crystal to 'episodic' regardless of whether sources had been
        # promoted to semantic/procedural by the consolidator. Codex iter-1
        # M1 silent-bug repair.
        # L7 (#694): paginate the candidate SELECT. Without ``.range()`` PostgREST
        # silently caps the response at 1000 rows, truncating the candidate set
        # at scale and making the grouping + provenance sha256 non-deterministic.
        # Page through ``.range()`` windows until a short page signals the end.
        # Each blocking Supabase HTTP call is off-loaded to a worker thread so
        # the synchronous ``.execute()`` does not stall the FastAPI event loop on
        # the operator ``/crystallize`` path (M2-supabase, #694).
        page_size = self.candidate_page_size
        offset = 0
        memories: List[Dict[str, Any]] = []
        while True:
            page_query = (
                client.table("episodic_memories")
                .select(
                    "memory_id, agent_name, brand, region, causal_path_id, "
                    "event_type, description, outcome_type, occurred_at, "
                    "raw_content, consolidation_tier"
                )
                .eq("brand", brand)
                .gte("occurred_at", cutoff)
                .in_("event_type", ["agent_action", "causal_discovery", "experiment_completed"])
            )
            if region:
                page_query = page_query.eq("region", region)
            # L7 (codex MED): order by the unique PK before paging. Offset
            # pagination without a stable sort can skip or duplicate rows across
            # pages if the query plan reorders — defeating the determinism this
            # pagination is meant to give. memory_id is the unique PK.
            page_query = page_query.order("memory_id").range(offset, offset + page_size - 1)
            page = ((await asyncio.to_thread(page_query.execute)).data) or []
            memories.extend(page)
            if len(page) < page_size:
                break
            offset += page_size

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
                if not insight_id:
                    # Skip-signal from _crystallize_group: an active row for
                    # this (brand, region, kpi, causal_path) already exists.
                    # Backed by the partial-unique-index in migration 021. Not
                    # an error — concurrent crystallizer runs are expected.
                    continue
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

    async def crystallize_finding(
        self,
        finding_id: str,
        *,
        brand: str,
        region: Optional[str] = None,
        crystallized_by_cycle_id: Optional[str] = None,
        crystallized_by_user_id: Optional[str] = None,
    ) -> CrystallizerResult:
        """Crystallize a single finding by ID (#376 DoD §D).

        Resolves the finding's brand context via the supplied ``brand``
        kwarg (the brand is the tenant boundary and never inferred from
        the finding row itself — explicit-over-implicit). Then runs the
        same aggregation logic as ``run_for_brand``; downstream callers
        get a result that is brand-scoped exactly like the periodic
        path.

        The ``finding_id`` is reserved for future use (post-#376) when
        the crystallizer will narrow the grouping to the supplied
        finding's causal_path. For now it is logged for traceability;
        the brand-scoped aggregation runs unchanged.
        """
        if not brand:
            raise ValueError("crystallize_finding requires brand")
        logger.info(
            f"crystallize_finding: finding_id={finding_id} brand={brand} region={region or 'all'}"
        )
        return await self.run_for_brand(
            brand=brand,
            region=region,
            crystallized_by_cycle_id=crystallized_by_cycle_id,
            crystallized_by_user_id=crystallized_by_user_id,
        )

    async def crystallize_portfolio(
        self,
        *,
        brands: Optional[List[str]] = None,
        crystallized_by_cycle_id: Optional[str] = None,
        crystallized_by_user_id: Optional[str] = None,
    ) -> CrystallizerResult:
        """Iterate ``brands`` and aggregate results (#376 DoD §D).

        ``brands=None`` resolves to :data:`DEFAULT_PORTFOLIO_BRANDS`.
        Each brand is crystallized independently — the result counts
        sum across brands; ``by_brand`` carries per-brand counts; an
        exception for one brand is surfaced in ``errors`` and does not
        prevent the other brands from running.
        """
        target_brands = list(brands) if brands else list(DEFAULT_PORTFOLIO_BRANDS)
        aggregate = CrystallizerResult()
        for brand in target_brands:
            try:
                per_brand = await self.run_for_brand(
                    brand=brand,
                    crystallized_by_cycle_id=crystallized_by_cycle_id,
                    crystallized_by_user_id=crystallized_by_user_id,
                )
                aggregate.examined_groups += per_brand.examined_groups
                aggregate.insights_created += per_brand.insights_created
                aggregate.edges_created += per_brand.edges_created
                for b, n in per_brand.by_brand.items():
                    aggregate.by_brand[b] = aggregate.by_brand.get(b, 0) + n
                aggregate.errors.extend(per_brand.errors)
            except Exception as exc:
                logger.exception(f"crystallize_portfolio: brand {brand} failed")
                aggregate.errors.append(f"brand={brand}: {exc}")
        aggregate.finished_at = datetime.now(timezone.utc)
        return aggregate

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

        # --- Derive the 13 deterministic CrystalDigest fields (#376 §B) ---
        derived = _derive_crystal_digest_fields(brand=brand, members=members)

        # --- Compose the 2 narrative-prose fields (LLM-flagged) ---
        # ``audit`` is captured for persistence to
        # ``crystal_narrative_audits`` (#391 box 4 — codex iter-2 H1
        # closure). On the flag-off path it remains None — only the LLM
        # path produces an audit row. The codex L1 tighten replaces the
        # legacy ``Any`` annotation with ``Optional[LLMCrystalNarrativeAudit]``
        # so downstream ``.limitations`` / ``.key_finding`` accesses go
        # through mypy.
        audit: Optional[LLMCrystalNarrativeAudit] = None
        if _llm_narratives_enabled():
            audit = await _invoke_llm_narrator(
                brand=brand,
                region=region,
                members=members,
                derived=derived,
                client_factory=self._anthropic_client_factory,
            )
            limitations = audit.limitations
            recommended_next = audit.recommended_next_analysis
            if audit.key_finding:
                # LLM-emitted key_finding overrides the heuristic title
                # ONLY if non-empty (defensive against partial outputs).
                title = audit.key_finding[:500]
        else:
            limitations, recommended_next = _deterministic_narrative_prose(
                brand=brand, members=members, derived=derived
            )

        # Insert executive_insight row.
        # Migration 021 has a partial-unique-index on
        # (brand, region, kpi, key_metrics->>'causal_path_id') WHERE
        # invalidated_at IS NULL. Concurrent crystallizer runs (Celery beat
        # + operator-triggered POST /crystallize) collide here; catch and
        # return ("", 0) as a skip-signal that the caller observes.
        try:
            insert_query = client.table("executive_insights").insert(
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
                    # --- analytical (#376 §A.1-8) ---
                    "effect_size": derived["effect_size"],
                    "effect_ci_lower": derived["effect_ci_lower"],
                    "effect_ci_upper": derived["effect_ci_upper"],
                    "effect_direction": derived["effect_direction"],
                    "cohort_size": derived["cohort_size"],
                    "confounders_controlled": derived["confounders_controlled"],
                    "sensitivity_checks_passed": derived["sensitivity_checks_passed"],
                    "sensitivity_checks_failed": derived["sensitivity_checks_failed"],
                    # --- narrative prose (#376 §A.9-10) ---
                    "limitations": (limitations or "")[:500],
                    "recommended_next_analysis": (recommended_next or "")[:500],
                    # --- lineage (#376 §A.11-15) ---
                    "provenance_chain_id": derived["provenance_chain_id"],
                    "provenance_depth": derived["provenance_depth"],
                    "consolidation_tier": derived["consolidation_tier"],
                    "replication_count": derived["replication_count"],
                    "data_version": derived["data_version"],
                }
            )
            # Off-load the blocking insert to a worker thread (M2-supabase, #694).
            insert = await asyncio.to_thread(insert_query.execute)
        except Exception as exc:
            # Narrow match: only treat as a duplicate-skip if the error
            # message names the specific partial-unique-index from migration
            # 021. A broader match (just "unique" or "duplicate" anywhere in
            # the message) would mis-classify any unrelated RuntimeError /
            # API error that happens to contain those substrings, silently
            # swallowing real failures. See codex-rescue iter-0 HIGH-1.
            err_str = str(exc).lower()
            if "uix_executive_insights_active_causal_path" in err_str:
                logger.info(
                    f"crystallizer: skipping duplicate for brand={brand} "
                    f"group={group_key} (existing active row): {exc}"
                )
                return ("", 0)
            raise
        rows = insert.data or []
        if not rows:
            raise RuntimeError("executive_insight insert returned no rows")
        insight_id = rows[0]["insight_id"]

        # --- Persist crystal_narrative_audits (#391 box 4, codex iter-2 H1) ---
        # The PHI audit harness reads ``cna.input_prompt`` from this table
        # via the SQL JOIN at
        # ``scripts/audit_phi_in_crystal_narratives.py:_load_records_from_postgres``.
        # Without persistence here, every real crystal lands with NULL
        # ``audit_input_prompt`` and the PHI scanner cannot audit LLM
        # INPUTS (only LLM outputs via ``key_finding`` / ``narrative``).
        #
        # Persistence is BEST-EFFORT (audit-only telemetry, NOT
        # crystallization-gating): a narrow ``except Exception`` here
        # logs a warning + lets the crystallization continue. The audit
        # row is a sidecar for PHI auditing — losing it for one crystal
        # is preferable to failing the whole pipeline. Migration 028's
        # ``ON DELETE CASCADE`` keeps orphan rows out, and the
        # ``UNIQUE(insight_id)`` constraint there makes the path
        # idempotent across retries.
        if audit is not None:
            try:
                audit_query = client.table("crystal_narrative_audits").insert(
                    {
                        "insight_id": str(insight_id),
                        "narrator_model": audit.narrator_model,
                        "key_finding": audit.key_finding or "",
                        "limitations": audit.limitations or "",
                        "recommended_next": audit.recommended_next_analysis or "",
                        "input_prompt": audit.input_prompt or "",
                        "latency_ms": audit.latency_ms,
                        "input_tokens": audit.input_tokens,
                        "output_tokens": audit.output_tokens,
                        "cost_usd": audit.cost_usd,
                    }
                )
                # Off-load the blocking insert to a worker thread (M2-supabase, #694).
                await asyncio.to_thread(audit_query.execute)
            except Exception as exc:
                # Narrow LOG-warning-on-failure shape (NOT crystallization-
                # gating). The audit row is for offline PHI auditing; its
                # absence does not invalidate the crystal, so we must not
                # break the per-group try/except envelope in
                # ``run_for_brand`` (which would mark the whole group as
                # failed). Surface enough context for the audit harness to
                # alert on persistent failures via a separate ops query
                # (``SELECT COUNT(*) FROM executive_insights ei LEFT JOIN
                # crystal_narrative_audits cna ON cna.insight_id =
                # ei.insight_id WHERE cna.insight_id IS NULL AND
                # ei.created_at > ...``).
                logger.warning(
                    "crystal_narrative_audits insert failed for insight_id=%s: %s",
                    insight_id,
                    exc,
                )

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
                # Off-load the blocking insert to a worker thread (M2-supabase, #694).
                edge_query = client.table("insight_edges").insert(edge_rows)
                await asyncio.to_thread(edge_query.execute)
                edges_added = len(edge_rows)
            except Exception as exc:
                logger.warning(f"crystallizer: edge insert partially failed: {exc}")

        # #391 monitoring box 1.b + 2.c (count_by_brand): emit Opik
        # trace + MLflow brand-tagged crystal-count counter on every
        # successful crystallization. ``source_count`` is the
        # member-count for the group (what the database column stores
        # as ``source_count``). ``edges_added`` is the number of
        # insight_edges rows actually written — useful for tracing
        # partial-failure shapes. Best-effort: helper swallows its
        # own exceptions, so a broken Opik/MLflow backend doesn't
        # gate crystallization.
        record_provenance_write(
            insight_id=str(insight_id),
            source_count=len(members),
            brand=brand,
            edges_added=edges_added,
        )

        return insight_id, edges_added


# ============================================================================
# Helpers — grouping + narrative composition (unchanged from v1)
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


# ============================================================================
# Issue #376 — deterministic CrystalDigest field derivation
# ============================================================================


def _derive_crystal_digest_fields(*, brand: str, members: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive the 13 deterministic CrystalDigest fields from source members.

    All fields are extracted from the ``raw_content`` JSONB blob the
    causal_impact agent writes (per
    ``src/agents/causal_impact/memory_hooks.py:442-452``). When a field
    is missing or has the wrong shape, the function returns ``None`` /
    empty list rather than failing — schema completion is a forward
    step, not a hard contract on existing episodic memories.

    Returns a dict with the 13 derived keys; the row-insert site
    consumes it directly.
    """
    # Pull primary causal-impact memory (highest-effort signal). Prefer the
    # one with the most populated raw_content; fall back to the first.
    primary = _pick_primary_member(members)
    rc = (primary or {}).get("raw_content") or {}
    if not isinstance(rc, dict):
        rc = {}

    # --- effect_size + CI bounds ---
    effect_size = _coerce_float(rc.get("ate_estimate"))
    ci = rc.get("confidence_interval")
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    if isinstance(ci, (list, tuple)) and len(ci) >= 2:
        ci_lower = _coerce_float(ci[0])
        ci_upper = _coerce_float(ci[1])

    # --- effect_direction (deterministic from sign + CI bounds) ---
    effect_direction: Optional[str] = None
    if effect_size is not None and ci_lower is not None and ci_upper is not None:
        if ci_lower <= 0.0 <= ci_upper:
            effect_direction = "null"
        elif effect_size > 0.0:
            effect_direction = "positive"
        else:
            effect_direction = "negative"
    elif effect_size is not None:
        # Fallback: sign-only direction when CI unavailable
        if effect_size > 0.0:
            effect_direction = "positive"
        elif effect_size < 0.0:
            effect_direction = "negative"
        else:
            effect_direction = "null"

    # --- cohort_size ---
    cohort_size = _coerce_int(rc.get("sample_size"))

    # --- confounders_controlled (union/dedup across all members) ---
    # SORTED for deterministic serialization (codex iter-1 M2):
    # encounter-order dedup is non-deterministic across runs because
    # the upstream episodic_memories query has no stable secondary
    # ordering. Stable sort here means JSONB diffs are minimal and
    # the row hash is reproducible across re-crystallization passes.
    confounders_set: set = set()
    for m in members:
        m_rc = m.get("raw_content") or {}
        if not isinstance(m_rc, dict):
            continue
        for c in m_rc.get("confounders") or []:
            cs = str(c).strip()
            if cs:
                confounders_set.add(cs)
    confounders: List[str] = sorted(confounders_set)

    # --- sensitivity_checks_passed / failed (union/dedup across members) ---
    # Same sort-for-stability contract as confounders (codex iter-1 M2).
    passed_set: set = set()
    failed_set: set = set()
    for m in members:
        m_rc = m.get("raw_content") or {}
        if not isinstance(m_rc, dict):
            continue
        for t in m_rc.get("refutation_passed_tests") or []:
            ts = str(t).strip()
            if ts:
                passed_set.add(ts)
        for t in m_rc.get("refutation_failed_tests") or []:
            ts = str(t).strip()
            if ts:
                failed_set.add(ts)
    passed: List[str] = sorted(passed_set)
    failed: List[str] = sorted(failed_set)

    # --- provenance_chain_id (deterministic hash of source set) ---
    member_ids = sorted(str(m.get("memory_id", "")) for m in members)
    causal_paths = sorted(
        {str(m.get("causal_path_id", "")) for m in members if m.get("causal_path_id")}
    )
    chain_input = "|".join(causal_paths + member_ids)
    provenance_chain_id = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()[:32]

    # --- provenance_depth (BFS hop count; v1 = 1 because the source
    # memories are direct ancestors; the causal_path edge adds one
    # implicit hop, so we report 2 when a causal_path is wired) ---
    provenance_depth = 2 if causal_paths else 1

    # --- consolidation_tier ---
    # v1: source rows are tier='episodic' by default (migration 021
    # line 99). When called from the consolidator post-promotion the
    # consolidator can override; for the periodic crystallizer path
    # the tier defaults to 'episodic'. If any source carries a higher
    # tier (semantic/procedural), the crystal inherits the highest.
    consolidation_tier = "episodic"
    tier_rank = {"working": 0, "episodic": 1, "semantic": 2, "procedural": 3}
    for m in members:
        m_tier = m.get("consolidation_tier")
        if m_tier in tier_rank and tier_rank[m_tier] > tier_rank[consolidation_tier]:
            consolidation_tier = m_tier

    # --- replication_count (v1: source_count) ---
    replication_count = len(members)

    # --- data_version (first non-empty raw_content.data_version) ---
    data_version: Optional[str] = None
    for m in members:
        m_rc = m.get("raw_content") or {}
        if isinstance(m_rc, dict):
            dv = m_rc.get("data_version")
            if dv:
                data_version = str(dv)
                break

    return {
        "effect_size": effect_size,
        "effect_ci_lower": ci_lower,
        "effect_ci_upper": ci_upper,
        "effect_direction": effect_direction,
        "cohort_size": cohort_size,
        "confounders_controlled": confounders,
        "sensitivity_checks_passed": passed,
        "sensitivity_checks_failed": failed,
        "provenance_chain_id": provenance_chain_id,
        "provenance_depth": provenance_depth,
        "consolidation_tier": consolidation_tier,
        "replication_count": replication_count,
        "data_version": data_version,
    }


def _pick_primary_member(members: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the member with the richest raw_content (most causal keys).

    We prefer the causal_impact agent's memory when present because its
    raw_content carries the estimator state. Falls back to the most-
    populated raw_content if no agent_name='causal_impact' is found.
    """
    if not members:
        return None
    for m in members:
        if m.get("agent_name") == "causal_impact":
            rc = m.get("raw_content") or {}
            if isinstance(rc, dict) and "ate_estimate" in rc:
                return m
    # Otherwise the most-populated raw_content
    best = max(
        members,
        key=lambda m: (
            len(m.get("raw_content") or {}) if isinstance(m.get("raw_content"), dict) else 0
        ),
    )
    return best


def _coerce_float(value: Any) -> Optional[float]:
    """Defensive float coercion. Returns None on missing/unparseable.

    ``bool`` is explicitly excluded to mirror the
    ``src/rag/memory_connector.py`` pattern (memory:
    [[feat-373-phase2-finishing-close-20260519]]) — ``False`` would
    coerce to ``0.0`` and silently activate downstream checks.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    """Defensive int coercion. Returns None on missing/unparseable.

    ``bool`` excluded for the same reason as :func:`_coerce_float`.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _deterministic_narrative_prose(
    *,
    brand: str,
    members: List[Dict[str, Any]],
    derived: Dict[str, Any],
) -> Tuple[str, str]:
    """Heuristic ``(limitations, recommended_next_analysis)`` when the
    LLM narrator is flag-off.

    Both fields are non-empty so the dashboard does not show blank
    cells when an operator has not opted in to the LLM path.
    """
    cohort = derived.get("cohort_size") or 0
    failed = derived.get("sensitivity_checks_failed") or []
    direction = derived.get("effect_direction")

    limit_parts: List[str] = []
    if cohort and cohort < 500:
        limit_parts.append(f"small cohort (n={cohort})")
    if failed:
        limit_parts.append(f"sensitivity-check failures: {', '.join(failed)}")
    if not limit_parts:
        limit_parts.append("standard limitations apply (deterministic heuristic)")
    limitations = "; ".join(limit_parts) + "."

    rec_parts: List[str] = []
    if direction == "null":
        rec_parts.append("re-power study with larger cohort or longer follow-up window")
    elif failed:
        rec_parts.append("re-run analysis under the failed sensitivity test(s) with adjusted spec")
    else:
        rec_parts.append("replicate on an independent cohort to confirm generalizability")
    recommended = "; ".join(rec_parts) + "."

    return limitations, recommended


async def _invoke_llm_narrator(
    *,
    brand: str,
    region: Optional[str],
    members: List[Dict[str, Any]],
    derived: Dict[str, Any],
    client_factory: Optional[_AnthropicClientFactory] = None,
) -> Any:
    """Invoke the Haiku narrator and return an
    :class:`src.data.kg.types.LLMCrystalNarrativeAudit`.

    PRODUCTION shape: thin wrapper around
    :class:`anthropic.AsyncAnthropic` (NOT the sync
    :class:`anthropic.Anthropic` — the crystallizer is async-end-to-
    end; a sync client here would block the event loop and stall the
    FastAPI / Celery worker pool). Telemetry (latency_ms,
    input_tokens, output_tokens, cost_usd) is captured into the audit.

    UNIT TESTS: pass a fake ``client_factory`` so no network call
    happens and the audit is fully deterministic.

    Exception path: SDK / API exceptions narrow-caught on the four
    anthropic.* error classes (APIConnectionError, APITimeoutError,
    RateLimitError, APIStatusError); programming errors
    (TypeError/AttributeError/KeyError) propagate per the codex-rescue
    H2 / #378-iter-0-M1 narrow-catch contract.
    """
    # ``LLMCrystalNarrativeAudit`` is imported at module top (#391 box 4
    # codex iter-2 L1 closure: the lazy-load form here would defeat the
    # ``Optional[LLMCrystalNarrativeAudit]`` annotation introduced by the
    # same closure on the call-site).

    # We carry the imported module under a separate name (``anthropic_module``)
    # so mypy does not type ``anthropic`` as ``Module`` and then reject the
    # ``anthropic = None`` fallback assignment. The local-None sentinel lets
    # tests inject a ``client_factory`` even when the SDK is not present.
    anthropic_module: Optional[Any]
    try:
        import anthropic as _anthropic_module

        anthropic_module = _anthropic_module
    except ImportError:
        if client_factory is None:
            logger.warning("anthropic SDK unavailable; falling back to empty narrator audit")
            return LLMCrystalNarrativeAudit(narrator_model=DEFAULT_NARRATOR_MODEL)
        anthropic_module = None

    # Narrow catch tuple is derived from the real SDK module when present;
    # empty tuple when only the injected factory is available (tests that
    # explicitly inject a fake client without anthropic installed).
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
        # Memory `[[feedback-live-lm-skip-must-check-key-shape]]`: a
        # presence-only check would let CI placeholders (e.g.
        # ANTHROPIC_API_KEY=test-key) through and produce 401s. Use
        # the prefix check so empty + placeholder both short-circuit.
        logger.info("narrator: ANTHROPIC_API_KEY missing or placeholder; emitting empty audit")
        return LLMCrystalNarrativeAudit(narrator_model=DEFAULT_NARRATOR_MODEL)

    # Resolve the effective factory. When no factory is injected we
    # construct the real :class:`anthropic.AsyncAnthropic`; the local
    # ``effective_factory`` keeps mypy from re-narrowing the parameter.
    effective_factory: _AnthropicClientFactory
    if client_factory is not None:
        effective_factory = client_factory
    else:
        assert anthropic_module is not None  # narrowed by the import-fail branch above

        def _default_factory(key: str) -> _AnthropicClientProtocol:
            return anthropic_module.AsyncAnthropic(api_key=key)  # type: ignore[no-any-return,union-attr]

        effective_factory = _default_factory
    client = effective_factory(api_key)

    prompt = _build_narrator_prompt(brand=brand, region=region, members=members, derived=derived)

    started = time.monotonic()
    try:
        response = await client.messages.create(
            model=DEFAULT_NARRATOR_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
    except caught_api_errors as exc:
        # Narrow catch (codex iter-1 H2 + sibling #378 iter-0 M1):
        # SDK-level transient errors fall back to empty audit; the
        # row insert still completes with empty prose. Programming
        # errors (TypeError, AttributeError, KeyError) MUST propagate
        # so they surface in CI / DLQ instead of being silently
        # swallowed as "empty narrator audit".
        logger.warning("crystal-narrator-haiku-failure %s: %s", type(exc).__name__, exc)
        return LLMCrystalNarrativeAudit(
            narrator_model=DEFAULT_NARRATOR_MODEL,
            latency_ms=(time.monotonic() - started) * 1000.0,
            # Still capture the input prompt on the failure path so the
            # PHI scanner can audit even the no-output case (#391 box 4).
            input_prompt=prompt,
        )
    latency_ms = (time.monotonic() - started) * 1000.0

    text = ""
    if hasattr(response, "content") and response.content:
        first = response.content[0]
        if hasattr(first, "text"):
            text = first.text or ""

    parsed = _parse_narrator_response(text)

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
    from src.data.causal_role_evaluator import compute_haiku_cost_usd

    cost_usd = (
        compute_haiku_cost_usd(input_tokens=input_tokens, output_tokens=output_tokens)
        if (input_tokens is not None or output_tokens is not None)
        else None
    )

    return LLMCrystalNarrativeAudit(
        narrator_model=DEFAULT_NARRATOR_MODEL,
        key_finding=parsed.get("key_finding", "")[:500],
        limitations=parsed.get("limitations", "")[:500],
        recommended_next_analysis=parsed.get("recommended_next_analysis", "")[:500],
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        # #391 security box 4: capture the FULL prompt so the offline
        # PHI scanner can audit LLM inputs (not just outputs) for PHI
        # leaks. This is the input side of the audit harness contract.
        input_prompt=prompt,
    )


def _build_narrator_prompt(
    *,
    brand: str,
    region: Optional[str],
    members: List[Dict[str, Any]],
    derived: Dict[str, Any],
) -> str:
    """Compose the Haiku narrator prompt. Deterministic + JSON-shaped."""
    region_str = f" in {region}" if region else ""
    return (
        "You are auditing a crystallized cross-agent causal finding for a "
        "pharmaceutical commercial-analytics platform. Produce three short "
        "narrative fields based on the deterministic findings below.\n\n"
        f"Brand: {brand}{region_str}\n"
        f"Effect size (ATE): {derived.get('effect_size')}\n"
        f"95% CI: [{derived.get('effect_ci_lower')}, {derived.get('effect_ci_upper')}]\n"
        f"Effect direction: {derived.get('effect_direction')}\n"
        f"Cohort size: {derived.get('cohort_size')}\n"
        f"Confounders controlled: {derived.get('confounders_controlled')}\n"
        f"Sensitivity checks passed: {derived.get('sensitivity_checks_passed')}\n"
        f"Sensitivity checks failed: {derived.get('sensitivity_checks_failed')}\n"
        f"Source memories: {len(members)}\n\n"
        "Respond ONLY with valid JSON in this exact shape:\n"
        "{\n"
        '  "key_finding": "1-2 sentence headline distilling the finding",\n'
        '  "limitations": "1-2 sentence enumeration of known limitations",\n'
        '  "recommended_next_analysis": "1-2 sentence follow-up guidance"\n'
        "}\n"
    )


def _parse_narrator_response(text: str) -> Dict[str, str]:
    """Best-effort JSON parse of the narrator response."""
    import json
    import re

    if not text:
        return {}
    # Strip code fences if the model wrapped the JSON
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return {
                "key_finding": str(parsed.get("key_finding", "")),
                "limitations": str(parsed.get("limitations", "")),
                "recommended_next_analysis": str(parsed.get("recommended_next_analysis", "")),
            }
    except (json.JSONDecodeError, ValueError):
        logger.warning("narrator: failed to parse JSON response — empty fields")
    return {}


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
