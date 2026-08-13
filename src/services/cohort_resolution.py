"""Cohort-resolution service: resolve ``(brand, region)`` to a real cohort frame.

Background
----------
The tool-composer-remediation (issue #770/#774-777) wired the chat caller
``chatbot_tools._resolve_cohort_frame`` so that a real cohort DataFrame could be
threaded into the Tool Composer context (``context["estimation_data"]``). That
wiring only produced data when the caller supplied an explicit ``data_source``
(a parquet/s3 path or table name); an arbitrary ``(brand, region)`` pair could
NOT be resolved to a production frame.

This service is the deferred data-layer follow-up (issue #779). It resolves a
``(brand, region)`` pair to a real cohort DataFrame from the canonical
``patient_journeys`` table (which carries ``brand`` and ``geographic_region``
columns plus the causal variables the composable tools need) WITHOUT requiring
an explicit ``data_source``.

Resolution order (first non-empty frame wins):

1. **Explicit ``data_source``** -> the tier0 ``CohortConstructorAgent`` loader
   (parquet/s3/table). Preserves the exact behavior R4 shipped.
2. **Canonical ``patient_journeys``** filtered by normalized brand +
   ``geographic_region``.

Anti-mocking discipline
-----------------------
This service NEVER fabricates a synthetic cohort. It returns ``None`` (fail
closed) when:

* an explicit ``data_source`` yields nothing,
* a supplied brand/region is not a recognized enum member (for regions: a
  label or a platform-synonym of one, #1517 — so we do not silently return a
  wrong-population cohort), or
* the canonical query returns zero rows.

Callers then honestly proceed without ``estimation_data`` and the composable
tools fail closed in turn (descriptive ``RuntimeError``), rather than returning
plausible-but-fake values.

Both ``chatbot_tools`` (chat path) and the ``cohort_builder`` composable tool
(issue #778) route through :func:`resolve_cohort_frame` so there is a SINGLE
cohort-loading code path, not two divergent ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.services.enum_labels import resolve_brand_label, resolve_region_label

logger = logging.getLogger(__name__)

# Canonical cohort source. ``patient_journeys`` carries ``brand`` (brand_type),
# ``geographic_region`` (region_type) and the causal variables (engagement_score,
# treatment_initiated, disease_severity, academic_hcp, age_at_diagnosis).
CANONICAL_COHORT_TABLE = "patient_journeys"

# PostgREST imposes a configured max row count per request (default 1000). When a
# canonical-table query returns at least this many rows without an explicit
# ``limit``, the cohort may have been silently truncated to a sample.
_POSTGREST_DEFAULT_MAX_ROWS = 1000

# brand_type / region_type labels and resolvers are shared with the chat KPI
# tool (src.services.enum_labels, #1505) so an enum change lands in one place.
# Drift mode stays fail-closed: a newly-added brand resolves to None until added
# to the shared label set, never a wrong/fabricated cohort.
_normalize_brand = resolve_brand_label


def _normalize_region(region: Optional[str]) -> Optional[str]:
    """Map a region string to its canonical ``region_type`` label, else None.

    Synonym-tolerant since #1517 (a deliberate product decision, NOT a
    consolidation side effect — #1505 kept this strict until it was decided).
    Evidence behind the flip: every production consumer feeding this service
    passes chat/LLM-derived or frontend-typed region strings —

    * ``chatbot_tools.tool_composer_tool`` / ``cohort_builder`` (tool
      registrations): LLM tool-call / planner arguments from chat text,
    * the orchestrator dispatcher (``_extract_brand_region``): NLP entity
      extractions, frontend ``user_context`` (typed to the four canonical
      labels), or ``query_entities.region_from_text`` (canonical labels and,
      since #1572, unambiguous aliases from the same shared vocabulary).

    No consumer passes market/territory identifiers a ``REGION_ALIASES``
    entry could falsely match, and the chat KPI tool already resolves the
    SAME chat strings with synonyms — strict mode here meant "conversion in
    the Pacific" got KPI numbers from one tool and a fail-closed cohort from
    the other. Anything outside the alias table ("US", "EMEA", typos) still
    fails closed; the service never guesses.
    """
    return resolve_region_label(region, allow_synonyms=True)


def _load_tier0_agent() -> Any:
    """Return a tier0 ``CohortConstructorAgent`` (indirection eases testing)."""
    from src.agents.cohort_constructor.tier0_integration import (
        CohortConstructorAgent,
    )

    return CohortConstructorAgent()


def _resolve_via_data_source(
    brand: Optional[str],
    region: Optional[str],
    data_source: str,
) -> Optional[pd.DataFrame]:
    """Resolve via the tier0 loader (explicit parquet/s3/table data_source).

    Mirrors the behavior R4 shipped in ``chatbot_tools._resolve_cohort_frame``.
    """
    agent = _load_tier0_agent()
    result = agent.run(
        {
            "scope_spec": {
                "brand": brand or "",
                "indication": "",
                "target_population": region or "",
                "business_objective": "tool_composer_estimation",
            },
            "patient_data_source": data_source,
            "use_existing_config": True,
        }
    )
    frame = result.get("eligible_patients")
    if frame is None or getattr(frame, "empty", True):
        return None
    return frame


def _resolve_via_patient_journeys(
    brand: Optional[str],
    region: Optional[str],
    *,
    supabase_client: Optional[Any] = None,
    limit: Optional[int] = None,
    include_synthetic: bool = False,
) -> Optional[pd.DataFrame]:
    """Resolve from the canonical ``patient_journeys`` table.

    Filters by normalized brand and ``geographic_region``. A supplied brand or
    region that is not a recognized enum member returns ``None`` WITHOUT issuing
    a query (we cannot faithfully resolve the requested population, so we fail
    closed rather than return a wrong cohort or trigger a DB enum error).
    """
    norm_brand = _normalize_brand(brand)
    norm_region = _normalize_region(region)

    # A NON-EMPTY brand/region that does not map to a known enum member -> fail
    # closed (never silently widen to an all-population cohort). Empty/whitespace
    # is treated as "not specified" (no filter), identical to None.
    if brand and brand.strip() and norm_brand is None:
        logger.info("cohort_resolution: unrecognized brand %r -> fail closed", brand)
        return None
    if region and region.strip() and norm_region is None:
        logger.info("cohort_resolution: unrecognized region %r -> fail closed", region)
        return None

    client = supabase_client if supabase_client is not None else _default_client()

    query = client.table(CANONICAL_COHORT_TABLE).select("*")
    if norm_brand:
        query = query.eq("brand", norm_brand)
    if norm_region:
        query = query.eq("geographic_region", norm_region)
    # Shard 07 R11: default-exclude is_synthetic so real-mode cohort resolution never
    # blends synthetic rows; a validation run passes include_synthetic=True.
    from src.repositories.provenance import apply_provenance_filter

    query = apply_provenance_filter(query, include_synthetic)
    if limit:
        query = query.limit(limit)

    response = query.execute()
    rows = getattr(response, "data", None) or []
    if not rows:
        return None
    # No silent caps: if we hit the PostgREST default max without an explicit
    # limit, the cohort may be a truncated sample -- surface it rather than
    # presenting a partial frame as a complete cohort.
    if limit is None and len(rows) >= _POSTGREST_DEFAULT_MAX_ROWS:
        logger.warning(
            "cohort_resolution: patient_journeys returned %d rows (>= PostgREST "
            "default cap %d) for brand=%r region=%r; cohort may be truncated -- "
            "pass limit= explicitly or paginate.",
            len(rows),
            _POSTGREST_DEFAULT_MAX_ROWS,
            norm_brand,
            norm_region,
        )
    return pd.DataFrame(rows)


def _default_client() -> Any:
    """Return the cached service-role Supabase client."""
    from src.repositories import get_supabase_client

    return get_supabase_client()


def resolve_cohort_frame(
    brand: Optional[str],
    region: Optional[str],
    *,
    data_source: Optional[str] = None,
    supabase_client: Optional[Any] = None,
    limit: Optional[int] = None,
    include_synthetic: bool = False,
) -> Optional[pd.DataFrame]:
    """Resolve a ``(brand, region)`` pair to a real cohort DataFrame.

    Args:
        brand: Brand context (e.g. ``"Kisqali"``); case-insensitive, mapped to
            the ``brand_type`` enum. Unrecognized -> ``None`` (fail closed).
        region: Region context (e.g. ``"Northeast"``); case-insensitive, mapped
            to the ``region_type`` enum (US census regions). Unrecognized ->
            ``None``.
        data_source: Optional explicit parquet/s3 path or table name. When
            supplied, the tier0 loader is used (preserves R4 behavior) and the
            canonical ``patient_journeys`` path is skipped.
        supabase_client: Optional injected client (testing / reuse). Defaults to
            the cached service-role client.
        limit: Optional row cap for the canonical ``patient_journeys`` path only
            (ignored on the explicit-``data_source`` path, where the tier0 loader
            controls its own bounds). PostgREST also imposes its own configured
            max (default 1000), so very large cohorts may be a sample -- a
            WARNING is logged when that cap is hit without an explicit ``limit``.

    Returns:
        A non-empty ``pd.DataFrame`` on success, else ``None``. NEVER a
        fabricated frame.

    Raises:
        Genuine infrastructure errors (e.g. ``ServiceConnectionError`` from the
        client factory) propagate so the caller can log-and-proceed; the tools
        then fail closed honestly.
    """
    if data_source:
        return _resolve_via_data_source(brand, region, data_source)
    return _resolve_via_patient_journeys(
        brand,
        region,
        supabase_client=supabase_client,
        limit=limit,
        include_synthetic=include_synthetic,
    )


@dataclass
class CohortOutcomeSpec:
    """A resolved cohort with a runnable causal var-set."""

    cohort: str
    frame: pd.DataFrame
    outcome_column: str
    treatment_column: str
    covariate_columns: list[str]


# Per-cohort (outcome, treatment, covariates) on the patient_journeys grain. Treatment
# is the canonical per-unit `treatment_arm` (Shard 03 / M2); hcp_adoption resolves a
# DIFFERENT grain (hcp_profiles) below. Cohort is resolved by OUTCOME COLUMN — there is
# no stored `cohort` column.
_PJ_COHORTS = {
    "initiation": (
        "treatment_initiated",
        "treatment_arm",
        ["disease_severity", "academic_hcp", "geographic_region"],
    ),
    "discontinuation": (
        "discontinued_180d",
        "treatment_arm",
        ["disease_severity", "academic_hcp", "geographic_region"],
    ),
    "persistence": (
        "persistent_180d",
        "treatment_arm",
        ["disease_severity", "academic_hcp", "geographic_region"],
    ),
}


def resolve_cohort_outcome_frame(
    cohort: str,
    brand: Optional[str],
    region: Optional[str],
    *,
    supabase_client: Optional[Any] = None,
    limit: Optional[int] = None,
    include_synthetic: bool = False,
) -> Optional[CohortOutcomeSpec]:
    """Resolve a named cohort to a frame + runnable causal var-set.

    cohort in {initiation, discontinuation, persistence, hcp_adoption}. Fails closed
    (None) on unknown cohort, unrecognized brand/region, or empty data — never
    fabricates (mirrors resolve_cohort_frame's contract).
    """
    key = str(cohort).strip().lower()
    if key == "hcp_adoption":
        return _resolve_hcp_adoption(
            brand,
            region,
            supabase_client=supabase_client,
            limit=limit,
            include_synthetic=include_synthetic,
        )
    if key not in _PJ_COHORTS:
        logger.info("cohort_resolution: unknown cohort %r -> fail closed", cohort)
        return None
    outcome, treatment, covars = _PJ_COHORTS[key]
    df = _resolve_via_patient_journeys(
        brand,
        region,
        supabase_client=supabase_client,
        limit=limit,
        include_synthetic=include_synthetic,
    )
    # Fail closed if the outcome or the canonical treatment_arm is absent (e.g. M2 not
    # applied / Shard-03 arm not populated) — never hand back a frame missing its
    # treatment column.
    if df is None or df.empty or outcome not in df.columns or treatment not in df.columns:
        return None
    df = df[df[outcome].notna()].copy()
    if df.empty:
        return None
    present_covars = [c for c in covars if c in df.columns]
    # Shard 06: retention_benefit (>=0) is RECOMPUTED at resolve time (not persisted —
    # Task 06.2) so resource_optimizer's problem_formulator (rejects expected_response<0)
    # can read it for the disc/persist cohorts. = per-severity scale * disease_severity *
    # persistent_180d. Only materializes when persistent_180d is populated (synthetic);
    # real rows have NULL persistent_180d and are dropped above, so the synthetic-DGP
    # constant never touches a real cohort.
    if key in ("discontinuation", "persistence") and {"disease_severity", "persistent_180d"} <= set(
        df.columns
    ):
        from src.ml.synthetic.generators.cohort_outcomes import (
            PERSISTENCE_RETENTION_BENEFIT_PER_SEVERITY,
        )

        # Coerce defensively: the discontinuation cohort filters on discontinued_180d
        # (not persistent_180d), so a row could carry a null persistent_180d. NaN/None
        # -> 0 keeps retention_benefit a non-negative float with no NaN (a non-persistent
        # or unknown-persistence patient has 0 retention benefit), so resource_optimizer
        # never sees a NaN/negative expected_response.
        sev = pd.to_numeric(df["disease_severity"], errors="coerce").fillna(0.0)
        pers = pd.to_numeric(df["persistent_180d"], errors="coerce").fillna(0.0)
        df["retention_benefit"] = (PERSISTENCE_RETENTION_BENEFIT_PER_SEVERITY * sev * pers).clip(
            lower=0.0
        )
        present_covars = present_covars + ["retention_benefit"]
    return CohortOutcomeSpec(
        cohort=key,
        frame=df.reset_index(drop=True),
        outcome_column=outcome,
        treatment_column=treatment,
        covariate_columns=present_covars,
    )


def _resolve_hcp_adoption(
    brand: Optional[str],
    region: Optional[str],
    *,
    supabase_client: Optional[Any] = None,
    limit: Optional[int] = None,
    include_synthetic: bool = False,
) -> Optional[CohortOutcomeSpec]:
    """Resolve the hcp_adoption cohort from hcp_profiles (the DB grain the runtime
    has). Outcome = the canonical adoption_category (ADOPTER/NON_ADOPTER); a binary
    `adopted` helper is derived in-frame for estimators. treatment = the HCP arm
    proxy (peer_influence_score, the exogenous centrality/engagement score);
    covariate = influence_network_size.

    Fails closed (mirroring the patient-journey branch) on an unrecognized brand or
    region, on empty/unlabeled data, or when the treatment column (peer_influence_score)
    is absent/all-null — never hands back a spec downstream estimators cannot run.
    """
    norm_brand = _normalize_brand(brand)
    if brand and brand.strip() and norm_brand is None:
        logger.info("cohort_resolution: unrecognized brand %r -> fail closed", brand)
        return None
    norm_region = _normalize_region(region)
    if region and region.strip() and norm_region is None:
        logger.info("cohort_resolution: unrecognized region %r -> fail closed", region)
        return None
    client = supabase_client if supabase_client is not None else _default_client()
    q = client.table("hcp_profiles").select(
        "hcp_id,peer_influence_score,influence_network_size,adoption_category,geographic_region"
    )
    # NOTE: hcp_profiles has NO `brand` column (HCPs are not brand-partitioned at this
    # grain — the per-brand CATE lives in the Shard-06.3 parquet artifact). We VALIDATE
    # brand above (fail closed on a bogus literal) but do not filter the query by it.
    if norm_region:
        q = q.eq("geographic_region", norm_region)
    # Shard 07 R11: default-exclude is_synthetic (hcp_profiles carries it); validation
    # opts in with include_synthetic=True.
    from src.repositories.provenance import apply_provenance_filter

    q = apply_provenance_filter(q, include_synthetic)
    if limit:
        q = q.limit(limit)
    rows = getattr(q.execute(), "data", None) or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "adoption_category" not in df.columns:
        return None
    # A cohort member must carry an adoption label; drop unlabeled rows (e.g. baseline
    # non-synthetic HCPs whose adoption_category is NULL) so the cohort is the populated
    # adoption population, not a NULL-diluted mix.
    df = df[df["adoption_category"].notna()].copy()
    # The treatment column must exist and be non-null, else the spec is unrunnable —
    # fail closed (parallel to the patient-journey branch's treatment guard).
    if "peer_influence_score" not in df.columns:
        return None
    df = df[df["peer_influence_score"].notna()].copy()
    if df.empty:
        return None
    # Binary helper for estimators; the canonical OUTCOME column stays adoption_category.
    df["adopted"] = (df["adoption_category"].astype(str).str.upper() == "ADOPTER").astype(int)
    covars = [c for c in ("influence_network_size",) if c in df.columns]
    return CohortOutcomeSpec(
        cohort="hcp_adoption",
        frame=df.reset_index(drop=True),
        outcome_column="adoption_category",
        treatment_column="peer_influence_score",
        covariate_columns=covars,
    )
