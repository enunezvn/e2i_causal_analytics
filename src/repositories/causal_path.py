"""
Causal Path Repository.

Handles discovered causal relationships.
"""

import re
from typing import Any, List, Optional

from src.repositories.base import BaseRepository
from src.utils.type_helpers import parse_supabase_rows


def outcome_match_tokens(term: str) -> List[str]:
    """Normalize a chat-supplied KPI/outcome term into node-match prefixes.

    Node names are snake_case (``treatment_initiated``); chat terms are free
    text ("treatment initiation"). Lowercase, split on non-alphanumeric
    boundaries, drop tokens under 3 chars, truncate longer tokens to a 6-char
    stem-ish prefix (so morphology bridges: "persistence" -> "persis" matches
    ``persistent_180d``; "initiation" -> "initia" matches
    ``treatment_initiated``), and dedupe — each surviving prefix becomes a
    case-insensitive substring match against ``start_node``/``end_node``.
    Prefix matching strictly widens the whole-token match it replaced. A term
    with no hit (e.g. "TRx" against a patient-journey registry) is a genuine
    substrate-coverage miss, not an error.
    """
    seen: dict[str, None] = {}
    for token in re.split(r"[^a-z0-9]+", term.lower()):
        if len(token) < 3:
            continue
        prefix = token[:6]
        if prefix not in seen:
            seen[prefix] = None
    return list(seen)


# ---------------------------------------------------------------------- #1716
# Retrieval dedup for search_paths_for_outcome (both twins). Repeated
# synthetic loads insert a FRESH random ``path_id`` per run for the SAME
# causal question: the cohort family of CausalPathsGenerator mints
# ``scp_<uuid4-hex>`` ids, so the loader's upsert-on-path_id is only
# idempotent for the content-addressed scp_a*/scp_f* families. Measured
# 2026-08-19 on the live registry: 2,729 rows collapse to 193 distinct
# (cause, outcome, brand) questions, same-identity copies differing only in
# path_id/discovery_date/created_at and RNG-jittered confidence — and 10
# copies of one 0.945 path filled the whole 15-row cap for
# 'treatment_initiated', crowding out distinct drivers (the 0.892
# trigger_accepted path that turn 4.7 needed). The cap must therefore count
# DISTINCT paths, which requires paging past the duplicates (the first
# distinct rank of that trigger path was raw row ~866).

_DEDUP_PAGE_SIZE = 500
# Scan cap: 6 pages = 3,000 raw rows per read (the whole registry is ~2.7k
# rows today). If the registry outgrows the cap, the LEAST-confident
# identities may be missed — the cap bounds the read; it never duplicates.
_DEDUP_MAX_PAGES = 6


def causal_path_identity(row: dict) -> tuple:
    """Dedup identity of a registry row: the causal QUESTION it answers.

    (start_node, end_node, brand) — deliberately the same key
    :meth:`CausalPathRepository.get_distinct_questions` calls a "distinct
    causal question". Mediator-set variants of the same pair are the synthetic
    generator's per-row random decoration, not distinct drivers (measured:
    mediator-level identity leaves the whole top-15 ``treatment_arm``).
    """
    return (
        str(row.get("start_node") or "").strip().lower(),
        str(row.get("end_node") or "").strip().lower(),
        str(row.get("brand") or "").strip().lower(),
    )


def _fold_deduped(rows: List[dict], deduped: dict, limit: int) -> bool:
    """Fold confidence-desc ``rows`` into ``deduped`` keyed by identity.

    First-seen wins — with the input ordered by confidence descending that IS
    the max-confidence representative, and insertion order keeps the
    representatives confidence-desc. Returns True once ``limit`` distinct
    identities are collected.
    """
    for row in rows:
        key = causal_path_identity(row)
        if key not in deduped:
            deduped[key] = row
            if len(deduped) >= limit:
                return True
    return len(deduped) >= limit


class CausalPathRepository(BaseRepository):
    """
    Repository for causal_paths table.

    Supports:
    - Causal relationship queries
    - Path traversal
    - Effect estimation retrieval

    Provenance (#893): ``causal_paths`` carries the ``is_synthetic`` column
    (migration 063) and the synthetic loader stamps every loaded row, so all
    real-mode reads default-exclude synthetic paths. Validation/agent-context
    callers opt in per read with ``include_synthetic=True``.
    """

    table_name = "causal_paths"
    model_class = None  # Set to CausalPath model when available
    HAS_PROVENANCE = True  # causal_paths carries is_synthetic (migration 063, #893)
    # The live schema keys on a varchar ``path_id`` (there is no ``id`` column);
    # base get_by_id/update/delete were latent 42703s until #894.
    id_column = "path_id"

    async def get_paths_for_cause(
        self,
        cause: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get all causal paths originating from a cause.

        The live ``causal_paths`` schema stores the cause/effect pair as
        ``start_node``/``end_node`` (there are no ``cause``/``effect`` columns
        — filtering on them was a latent 42703 until #894). The argument names
        keep the caller-facing causal vocabulary.

        Args:
            cause: Cause entity name (matched against ``start_node``)
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"start_node": cause},
            limit=limit,
            include_synthetic=include_synthetic,
        )

    async def get_paths_for_effect(
        self,
        effect: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get all causal paths leading to an effect.

        Args:
            effect: Effect entity name (matched against ``end_node`` — see
                :meth:`get_paths_for_cause` for the schema mapping)
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"end_node": effect},
            limit=limit,
            include_synthetic=include_synthetic,
        )

    async def get_path_between(
        self,
        cause: str,
        effect: str,
        include_synthetic: bool = False,
    ) -> Optional[List]:
        """
        Get the causal path between two entities.

        Args:
            cause: Starting entity (matched against ``start_node``)
            effect: Ending entity (matched against ``end_node``)
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            CausalPath if exists, None otherwise
        """
        results = await self.get_many(
            filters={"start_node": cause, "end_node": effect},
            limit=1,
            include_synthetic=include_synthetic,
        )
        return results[0] if results else None

    async def get_by_brand(
        self,
        brand: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get causal paths related to a brand.

        Args:
            brand: Brand name
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"brand": brand},
            limit=limit,
            include_synthetic=include_synthetic,
        )

    async def search_paths_for_outcome(
        self,
        outcome_term: str,
        *,
        brand: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 15,
        include_synthetic: bool = False,
    ) -> List[dict]:
        """Highest-confidence causal paths whose cause/effect nodes match a term.

        This is the chat-facing registry query (2026-07-07 causal_analysis_tool
        rewire): ``confidence_level`` here is a REAL causal-confidence value
        (0-1, method-attributed), unlike the RAG layer's RRF rank-fusion score
        (ceiling ~0.03) the old tool compared against 0.7 — a filter that could
        never pass. An empty return for an unmatched term means the registry
        does not model that outcome (substrate-coverage gap), not "no causal
        drivers exist".

        A SYNC twin of this read — same filters, same token SSOT — lives at
        :func:`search_paths_for_outcome_sync` for callers that cannot await
        (the orchestrator's input resolvers run inside ``asyncio.to_thread``).
        Divergence is guarded by
        ``test_sync_and_async_causal_path_search_build_the_same_filters``.

        #1716: ``limit`` counts DISTINCT paths (see
        :func:`causal_path_identity`), not raw registry rows — duplicates from
        repeated loads are collapsed to their max-confidence representative
        BEFORE the cap, paging past duplicate floods up to
        ``_DEDUP_MAX_PAGES * _DEDUP_PAGE_SIZE`` raw rows.

        Args:
            outcome_term: Free-text KPI/outcome name; tokenized via
                :func:`outcome_match_tokens` against ``start_node``/``end_node``.
            brand: Optional brand, matched case-insensitively.
            min_confidence: Floor on ``confidence_level``.
            limit: Maximum DISTINCT paths, highest confidence first.
            include_synthetic: When True, do not exclude synthetic rows (opt-in).
        """
        if not self.client:
            return []
        tokens = outcome_match_tokens(outcome_term)
        if not tokens:
            return []

        def build_query() -> Any:
            # Rebuilt per page: the postgrest builder's .range() APPENDS
            # params (params.add), so a builder is not reusable across pages.
            query = self.client.table(self.table_name).select("*")
            query = query.or_(
                ",".join(
                    f"{col}.ilike.%{token}%"
                    for token in tokens
                    for col in ("start_node", "end_node")
                )
            )
            if brand:
                query = query.ilike("brand", brand)
            query = query.gte("confidence_level", min_confidence)
            if not include_synthetic and getattr(self, "HAS_PROVENANCE", False):
                from src.repositories.provenance import apply_provenance_filter

                query = apply_provenance_filter(query, include_synthetic=False)
            return query.order("confidence_level", desc=True)

        deduped: dict[tuple, dict] = {}
        for page in range(_DEDUP_MAX_PAGES):
            start = page * _DEDUP_PAGE_SIZE
            result = await build_query().range(start, start + _DEDUP_PAGE_SIZE - 1).execute()
            rows = parse_supabase_rows(result.data)
            if _fold_deduped(rows, deduped, limit) or len(rows) < _DEDUP_PAGE_SIZE:
                break
        return list(deduped.values())[:limit]

    async def get_distinct_outcomes(
        self,
        *,
        limit: int = 1000,
        include_synthetic: bool = False,
    ) -> List[str]:
        """Distinct ``end_node`` outcome names the registry actually models.

        Chat uses this to disclose substrate coverage honestly when a requested
        KPI has no matching paths (instead of implying an analysis ran and
        found nothing above threshold).
        """
        rows = await self.get_many(filters={}, limit=limit, include_synthetic=include_synthetic)
        seen: dict[str, None] = {}
        for row in rows:
            node = (row or {}).get("end_node") if isinstance(row, dict) else None
            if node and node not in seen:
                seen[node] = None
        return sorted(seen)

    async def get_distinct_questions(
        self,
        *,
        brand: Optional[str] = None,
        limit: int = 2000,
        include_synthetic: bool = True,
    ) -> List[dict]:
        """Distinct (treatment, outcome, brand) causal questions from the SSOT,
        each carrying its modeled backdoor set (``confounders_controlled``).

        Source of truth for the discovery leaderboard's questions (replaces the
        hand-curated cross-product). ``include_synthetic`` defaults True because
        the gold-standard substrate is synthetic.
        """
        # Empty dict (not None) = no brand filter; get_many iterates
        # filters.items(), which would raise AttributeError on None.
        filters = {"brand": brand} if brand else {}
        rows = await self.get_many(
            filters=filters, limit=limit, include_synthetic=include_synthetic
        )
        seen: dict = {}
        for r in rows:
            key = (r.get("start_node"), r.get("end_node"), r.get("brand"))
            if key[0] is None or key[1] is None or key in seen:
                continue
            seen[key] = {
                "treatment": r["start_node"],
                "outcome": r["end_node"],
                "brand": r.get("brand"),
                "confounders": list(r.get("confounders_controlled") or []),
            }
        return list(seen.values())

    # ------------------------------------------------------------------ #1352
    # RefutationNode promoter support (item 3). causal_paths.validation_status
    # semantics are pinned by migration 119: real rows enter 'pending' and ONLY
    # the RefutationNode may move them — after persisting passed refutation
    # evidence under causal_path_estimate_id(path_id). These helpers are that
    # promoter's read/write surface.

    async def get_path_row(self, path_id: str) -> Optional[dict]:
        """Fetch one causal_paths row by id, INCLUDING synthetic rows.

        The promoter must SEE a synthetic row (to refuse to promote it — a DGP
        fiction is not validated by a real run) rather than treat it as absent,
        so this read deliberately opts in to synthetic visibility.
        """
        rows = await self.get_many(filters={"path_id": path_id}, limit=1, include_synthetic=True)
        return rows[0] if rows else None

    async def find_real_paths_for_pair(
        self,
        treatment: str,
        outcome: str,
        brand: Optional[str] = None,
        limit: int = 5,
    ) -> List[dict]:
        """REAL (non-synthetic) path rows for a (treatment, outcome[, brand]).

        The promoter's auto-linkage read: real-mode default-exclude keeps DGP
        rows out, and the caller promotes only a UNIQUE match (ambiguity binds
        nothing — a mass promotion from one run would overclaim). ``limit`` is
        just enough to detect ambiguity.
        """
        filters: dict = {"start_node": treatment, "end_node": outcome}
        if brand:
            filters["brand"] = brand
        return await self.get_many(filters=filters, limit=limit, include_synthetic=False)

    async def set_validation_status(
        self,
        path_id: str,
        new_status: str,
        allowed_current: tuple,
    ) -> bool:
        """Conditionally move a path's ``validation_status`` (SOLE-promoter write).

        The transition is guarded server-side: the UPDATE matches only when the
        row's CURRENT status is in ``allowed_current``, so a concurrent writer
        (or an operator adjudication) is never silently overwritten. Returns
        True iff a row was actually updated. Raises on query errors — the
        caller (RefutationNode) degrades with a logged warning; a silent False
        on infra failure would be indistinguishable from a legitimate
        no-transition.
        """
        if not self.client:
            return False
        result = await (
            self.client.table(self.table_name)
            .update({"validation_status": new_status})
            .eq(self.id_column, path_id)
            .in_("validation_status", list(allowed_current))
            .execute()
        )
        return bool(result.data)


def search_paths_for_outcome_sync(
    outcome_term: str,
    *,
    client: Any = None,
    brand: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 15,
    include_synthetic: bool = False,
) -> List[dict]:
    """SYNC twin of :meth:`CausalPathRepository.search_paths_for_outcome` (#1475).

    The orchestrator's ``INPUT_RESOLVERS`` are sync by contract — the dispatcher
    offloads them with ``asyncio.to_thread`` — so the async repository method
    cannot be called from the explainer resolver that needs this substrate. Both
    reads build their node filters from the SAME :func:`outcome_match_tokens`
    and apply the same brand / confidence / provenance predicates in the same
    order, so a chat answer and an orchestrator answer can never disagree about
    what the registry models; ``test_explainer_evidence_binding_1475.py`` pins
    that equivalence.

    ``client`` is a SYNC supabase client. It defaults to the API-layer client
    (the one ``KPICalculator`` is built with), and a missing/unconfigured client
    returns ``[]`` — an honest "nothing resolved", never a fabricated path.

    #1716: shares the async twin's dedup-before-cap semantics — ``limit``
    counts DISTINCT :func:`causal_path_identity` paths, paged identically.
    """
    if client is None:
        from src.api.dependencies.supabase_client import get_supabase

        client = get_supabase()
    if not client:
        return []
    tokens = outcome_match_tokens(outcome_term)
    if not tokens:
        return []

    def build_query() -> Any:
        # Rebuilt per page — see the async twin: .range() appends params.
        query = client.table(CausalPathRepository.table_name).select("*")
        query = query.or_(
            ",".join(
                f"{col}.ilike.%{token}%" for token in tokens for col in ("start_node", "end_node")
            )
        )
        if brand:
            query = query.ilike("brand", brand)
        query = query.gte("confidence_level", min_confidence)
        if not include_synthetic and CausalPathRepository.HAS_PROVENANCE:
            from src.repositories.provenance import apply_provenance_filter

            query = apply_provenance_filter(query, include_synthetic=False)
        return query.order("confidence_level", desc=True)

    deduped: dict[tuple, dict] = {}
    for page in range(_DEDUP_MAX_PAGES):
        start = page * _DEDUP_PAGE_SIZE
        result = build_query().range(start, start + _DEDUP_PAGE_SIZE - 1).execute()
        rows = parse_supabase_rows(result.data)
        if _fold_deduped(rows, deduped, limit) or len(rows) < _DEDUP_PAGE_SIZE:
            break
    return list(deduped.values())[:limit]
