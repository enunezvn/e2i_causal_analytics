"""Estimation-frame loaders for the causal routes package.

Every read that turns a Supabase cohort into a numeric DataFrame the executors
can consume: paged selects, the HCP and NBA joins, covariate-role enforcement,
one-hot encoding and the agent estimation frame.

Import rule: may import ``_common``, ``datasets`` and non-package modules only;
never the package root.
"""

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException

if TYPE_CHECKING:
    from src.repositories.causal_path import CausalPathRepository

# Module-level so tests can patch this seam as ``loaders.get_async_supabase_client``
# — but ONLY for the two join paths that read it from module scope:
# ``_load_hcp_adoption_join_frame`` and ``_load_nba_triggers_join_frame``.
# ``_get_causal_path_repo`` and ``_load_agent_estimation_frame`` SHADOW this name
# with their own function-local import, so patching it here does NOT reach them
# and would be a false green; patch
# ``src.memory.services.factories.get_async_supabase_client`` for those paths
# (as tests/unit/test_api/test_causal_nba_baselines.py already does).
from src.memory.services.factories import get_async_supabase_client
from src.repositories.provenance import apply_provenance_filter

from .datasets import (
    _CAUSAL_BRAND_COLUMN,
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_FILL_ZERO_OUTCOMES,
    _CAUSAL_NUMERIC_COLUMNS,
    _CAUSAL_NUMERIC_DERIVATIONS,
    _CAUSAL_PHYSICAL_TABLE,
    _NBA_JOINED_COVARIATES,
)

logger = logging.getLogger(__name__)

# Paged-read page size. PostgREST returns at most ~1000 rows/request by default;
# patient cohorts have ~8.4k rows and the HCP cohorts 5k — a single unpaged
# .select() would SILENTLY truncate the cohort to a non-representative sample and
# misreport the ATE. We page with .range() until a short page is returned.
_TE_PAGE_SIZE = 1000

# Hard ceiling on pages so a runaway loop cannot exhaust memory; 20 pages * 1000
# = 20k rows comfortably covers the largest cohort (~8.4k) with headroom.
_TE_MAX_PAGES = 20


async def _te_paged_select(
    client: Any,
    table: str,
    columns: str,
    brand: str,
) -> List[Dict[str, Any]]:
    """Read ALL synthetic rows for ``brand`` from ``table`` via paged .range().

    Mirrors the audit.py .range() paging pattern. Returns the full row list —
    NEVER a silently-truncated sample (the single highest-risk fabrication bug for
    this surface). Raises on PostgREST/transport errors so the caller fail-closes.
    """
    rows: List[Dict[str, Any]] = []
    for page in range(_TE_MAX_PAGES):
        offset = page * _TE_PAGE_SIZE
        query = (
            client.table(table)
            .select(columns)
            .eq("brand", brand)
            .eq("is_synthetic", True)
            .range(offset, offset + _TE_PAGE_SIZE - 1)
        )
        result = await query.execute()
        batch: List[Dict[str, Any]] = result.data or []
        rows.extend(batch)
        if len(batch) < _TE_PAGE_SIZE:
            break
    return rows


async def _get_causal_path_repo() -> "CausalPathRepository":
    # ASYNC client is required: CausalPathRepository.get_many() awaits
    # query.execute(), which only works on the async postgrest builder. Building
    # the repo with the SYNC client raises "object APIResponse can't be used in
    # 'await'" the moment a read runs (precedent: chatbot_tools._query_causal_chains).
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.causal_path import CausalPathRepository

    return CausalPathRepository(supabase_client=await get_async_supabase_client())


def _coerce_estimation_row(
    row: Dict[str, Any],
    *,
    select_cols: List[str],
    treatment_var: str,
    outcome_var: str,
    numeric_cols: set,
    categorical_cols: "frozenset[str]" = frozenset(),
    derivations: Optional[Dict[str, Any]] = None,
    fill_zero: "frozenset[str]" = frozenset(),
) -> Optional[Dict[str, Any]]:
    """Coerce one raw DB row into an estimation record, or None if unusable
    (a treatment/outcome value is missing). Shared by every grain loader:
      - numeric_cols: float-coerced (non-coercible -> None), UNLESS also in categorical_cols
      - categorical_cols: passed through unchanged (P1b geo one-hot)
      - derivations: per-column callable applied before coercion (P3 text->0/1)
      - fill_zero: a None in these columns becomes 0.0 (P3 generated/nullable outcomes)
    For P1 only numeric_cols is passed, so behavior is identical to the prior inline loop."""
    derivations = derivations or {}
    record: Dict[str, Any] = {}
    for col in select_cols:
        value = row.get(col)
        if value is not None and col in derivations:
            value = derivations[col](value)
        if col in numeric_cols and col not in categorical_cols and value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
        if value is None and col in fill_zero:
            value = 0.0
        if col in (treatment_var, outcome_var) and value is None:
            return None
        record[col] = value
    return record


async def _load_hcp_profile_centrality(client: Any) -> List[Dict[str, Any]]:
    """Paged read of hcp_profiles centrality covariates (hcp_id,
    peer_influence_score, influence_network_size). hcp_profiles is NOT
    brand-partitioned, so this reads across the whole (synthetic) table — mirrors
    the proven treatment-effects HCP loader. Provenance-aware (synthetic-gold)."""
    rows: List[Dict[str, Any]] = []
    for page in range(_TE_MAX_PAGES):
        offset = page * _TE_PAGE_SIZE
        query = client.table("hcp_profiles").select(
            "hcp_id,peer_influence_score,influence_network_size"
        )
        query = apply_provenance_filter(query)
        query = query.range(offset, offset + _TE_PAGE_SIZE - 1)
        result = await query.execute()
        batch: List[Dict[str, Any]] = result.data or []
        rows.extend(batch)
        if len(batch) < _TE_PAGE_SIZE:
            break
    return rows


async def _te_paged_select_all_brands(client: Any) -> List[Dict[str, Any]]:
    """Brand-agnostic paged read of hcp_brand_adoption (all brands) for the
    hcp_adoption frame when no brand filter is set."""
    rows: List[Dict[str, Any]] = []
    for page in range(_TE_MAX_PAGES):
        offset = page * _TE_PAGE_SIZE
        query = client.table("hcp_brand_adoption").select("hcp_id,treatment_arm,adopted")
        query = apply_provenance_filter(query)
        query = query.range(offset, offset + _TE_PAGE_SIZE - 1)
        result = await query.execute()
        batch: List[Dict[str, Any]] = result.data or []
        rows.extend(batch)
        if len(batch) < _TE_PAGE_SIZE:
            break
    return rows


def _require_covariate_role(
    dataset: str, spec: Dict[str, List[str]], covariates: List[str]
) -> None:
    """Role-aware covariate validation (recovery-benchmark gap 3).

    The union allowlist admits any spec column into any slot, so an analyst's
    explicit picks could place an outcome/treatment-role column
    (``adherent_180d``, ``treatment_initiated``) in the COVARIATE slot. Those
    are post-treatment relative to the curated causal questions; the engine
    then trusts the declaration — guided tiers force the covariate
    pre-treatment, REVERSING its true edge, and the adjustment guarantee ships
    it in the adjustment set (measured on the benchmark's mediator DGP:
    conditioning on the mediator attenuated the ATE 60%). Engine-side
    detection is measured-impossible (the mediator DGP's true graph is
    complete, so every orientation is Markov-equivalent, and FCI returns an
    all-circle PAG), which makes this boundary — where the specs' curated
    pre-treatment role lists live — the only seam that can enforce it.

    The rule is membership in ``spec["covariate"]``: dual-role columns
    (``disease_stage`` et al., treatment AND covariate) stay accepted.
    Question slots are deliberately NOT tightened (the reversed-estimand gate
    already rejects outcome-as-treatment runs; no measured harm there).
    """
    allowed_covariate_role = set(spec["covariate"])
    role_crossed = [c for c in covariates if c not in allowed_covariate_role]
    if role_crossed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) {role_crossed} are not permitted as covariates for "
                f"dataset '{dataset}': they hold no covariate role in the curated "
                "spec (treatment/outcome-role columns are post-treatment relative "
                "to the causal questions — adjusting on them biases the estimate). "
                f"Allowed covariates: {sorted(allowed_covariate_role)}"
            ),
        )


async def _load_hcp_adoption_join_frame(
    *,
    treatment_var: str,
    outcome_var: str,
    covariates: List[str],
    limit: int,
    brand: Optional[str],
) -> tuple["pd.DataFrame", List[str]]:  # type: ignore[name-defined] # noqa: F821
    """Build the hcp_adoption estimation frame: hcp_brand_adoption (treatment_arm,
    adopted, hcp_id) JOIN hcp_profiles (peer_influence_score,
    influence_network_size) on hcp_id, deriving centrality_z =
    zscore(log1p(influence_network_size)).

    Reuses the proven two-reads-plus-pandas-merge HCP pattern (``_te_paged_select``
    + a separate hcp_profiles read) rather than a single-table select, because
    hcp_profiles is not brand-partitioned. Applies the SAME column-allowlist +
    numeric-coercion + drop-missing-treatment/outcome security gate as
    :func:`_load_agent_estimation_frame`. Fail-closed: 400 disallowed column, 503
    no store / no usable rows. Never fabricates rows.
    """
    import pandas as pd

    spec = _CAUSAL_DATASET_SPECS["hcp_adoption"]
    allowed = set(spec["treatment"]) | set(spec["outcome"]) | set(spec["covariate"])
    requested = [treatment_var, outcome_var, *covariates]
    not_allowed = [c for c in requested if c not in allowed]
    if not_allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) {not_allowed} are not permitted for dataset "
                f"'hcp_adoption'. Allowed: {sorted(allowed)}"
            ),
        )
    _require_covariate_role("hcp_adoption", spec, covariates)
    select_cols = list(dict.fromkeys(requested))

    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Causal data store unavailable")

    # brand scopes the adoption read (brand lives on hcp_brand_adoption); the
    # centrality covariates are brand-agnostic on hcp_profiles.
    if brand:
        adoption_rows = await _te_paged_select(
            client, "hcp_brand_adoption", "hcp_id,treatment_arm,adopted", brand
        )
    else:
        adoption_rows = await _te_paged_select_all_brands(client)
    if not adoption_rows:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable estimation rows for the requested variables "
                f"({treatment_var} -> {outcome_var}) in dataset 'hcp_adoption'."
            ),
        )
    profile_rows = await _load_hcp_profile_centrality(client)
    if not profile_rows:
        raise HTTPException(status_code=503, detail="hcp_profiles centrality unavailable")

    adoption_df = pd.DataFrame(adoption_rows)
    profile_df = pd.DataFrame(profile_rows).drop_duplicates(subset="hcp_id")
    merged = adoption_df.merge(profile_df, on="hcp_id", how="inner")
    if merged.empty:
        raise HTTPException(status_code=503, detail="hcp_adoption JOIN produced no rows")

    # Derive centrality_z = zscore(log1p(influence_network_size)) — matches the DGP
    # (hcp_adoption_artifact.py) the rep-engagement arm is confounded on.
    ins = pd.to_numeric(merged["influence_network_size"], errors="coerce")
    centrality = ins.map(lambda v: math.log1p(v) if pd.notna(v) else None).astype(float)
    std = centrality.std(ddof=0)
    merged["centrality_z"] = (centrality - centrality.mean()) / std if std and std > 0 else 0.0
    merged["peer_influence_score"] = pd.to_numeric(
        merged.get("peer_influence_score"), errors="coerce"
    )

    # Same numeric-coercion + drop-missing-treatment/outcome gate as the patient loader.
    numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get("hcp_adoption", set())
    records: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        record: Dict[str, Any] = {}
        usable = True
        for col in select_cols:
            value = row.get(col)
            if col in numeric_cols and value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
            if col in (treatment_var, outcome_var) and (value is None or pd.isna(value)):
                usable = False
                break
            record[col] = None if (value is not None and pd.isna(value)) else value
        if usable:
            records.append(record)
        if len(records) >= limit:
            break

    if not records:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable estimation rows for the requested variables "
                f"({treatment_var} -> {outcome_var}) in dataset 'hcp_adoption'."
            ),
        )
    return pd.DataFrame(records), select_cols


def _one_hot_categoricals(
    df: "pd.DataFrame",  # type: ignore[name-defined] # noqa: F821
    categorical_cols: List[str],
) -> tuple["pd.DataFrame", List[str]]:  # type: ignore[name-defined] # noqa: F821
    """One-hot encode the categorical columns present in ``df`` into stable
    ``<col>=<level>`` 0/1 float dummies, dropping the original column. drop_first
    drops the first sorted level as the reference category (avoids the
    dummy-variable trap). Level order is sorted for deterministic dummy names.
    Columns absent from ``df`` are skipped. A NULL cell gets its OWN
    ``<col>=__missing__`` dummy (1.0 for that row, 0.0 elsewhere) rather than
    silently collapsing into the drop_first reference level — a real cohort's
    NULL (e.g. 8.1% of ``optum_biologic_persistence``'s geographic_region) is
    not evidence the row belongs to the reference category. Returns
    ``(expanded_df, dummy_names)``."""
    present = [c for c in categorical_cols if c in df.columns]
    if not present:
        return df, []
    out = df.copy()
    dummy_names: List[str] = []
    for col in present:
        levels = sorted(str(v) for v in out[col].dropna().unique())
        for level in levels[1:]:  # drop_first: first sorted level = reference
            name = f"{col}={level}"
            out[name] = (out[col].astype(str) == level).astype(float)
            dummy_names.append(name)
        if out[col].isna().any():
            name = f"{col}=__missing__"
            out[name] = out[col].isna().astype(float)
            dummy_names.append(name)
        out = out.drop(columns=[col])
    return out, dummy_names


def _resolve_requested_baselines(dataset: str, adjust_baselines: bool) -> List[str]:
    """#1188: resolve the opt-in baseline-adjustment flag to the dataset's
    curated baseline covariate list.

    Fail-closed: requesting baseline adjustment on a dataset with no curated
    baseline role is an honest 400 (only randomized grains with a
    pre-treatment baseline join — nba_triggers — support it), never a silent
    no-op that would mislabel an unadjusted run as adjusted."""
    if not adjust_baselines:
        return []
    spec = _CAUSAL_DATASET_SPECS.get(dataset) or {}
    baselines = list(spec.get("baseline_covariate", []))
    if not baselines:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dataset '{dataset}' has no curated baseline covariates; "
                "adjust_baselines is only available for randomized datasets "
                "with a pre-treatment baseline role (nba_triggers)."
            ),
        )
    return baselines


# geographic_region is categorical -> one-hot expanded by the baseline loader
# (same _one_hot_categoricals machinery as patient_journeys).
_NBA_BASELINE_CATEGORICALS: frozenset = frozenset({"geographic_region"})

# Page cap for the baseline-join reads. Deliberately LARGER than _TE_MAX_PAGES
# (20 -> 20k rows): the live triggers table is ~37.5k rows and patient_journeys
# ~25k — a 20-page cap would silently truncate both, dropping ~5k patients'
# triggers from the joined sample (codex iter-1 MED). 60 pages x 1000 = 60k
# rows of headroom per read.
_NBA_JOIN_MAX_PAGES = 60


async def _load_trigger_question_rows(
    client: Any, select_cols: List[str], brand: Optional[str]
) -> List[Dict[str, Any]]:
    """PAGED triggers read for the baseline join: question columns + patient_id
    (the join key), provenance- and brand-aware.

    Reads the WHOLE (brand-scoped) table rather than ``.limit(limit)``: the
    request limit is applied POST-join by the frame builder, after orphan /
    missing-baseline rows are dropped — a pre-join cap would underfill (and
    order-bias) the joined sample (codex iter-1 MED)."""
    fetch_cols = list(dict.fromkeys([*select_cols, "patient_id"]))
    rows: List[Dict[str, Any]] = []
    for page in range(_NBA_JOIN_MAX_PAGES):
        offset = page * _TE_PAGE_SIZE
        query = client.table(_CAUSAL_PHYSICAL_TABLE.get("nba_triggers", "triggers")).select(
            ",".join(fetch_cols)
        )
        query = apply_provenance_filter(query)
        if brand:
            query = query.eq(_CAUSAL_BRAND_COLUMN.get("nba_triggers", "brand"), brand)
        query = query.range(offset, offset + _TE_PAGE_SIZE - 1)
        result = await query.execute()
        batch: List[Dict[str, Any]] = result.data or []
        rows.extend(batch)
        if len(batch) < _TE_PAGE_SIZE:
            break
    return rows


async def _load_patient_baseline_rows(
    client: Any, baseline_cols: List[str]
) -> List[Dict[str, Any]]:
    """Paged patient_journeys read of patient_id + the requested PRE-TREATMENT
    baseline columns (provenance-aware; patient_journeys is not paged by brand
    here — the trigger read already scopes the cohort). Pages up to
    _NBA_JOIN_MAX_PAGES so the whole ~25k-patient table is covered."""
    cols = ",".join(["patient_id", *baseline_cols])
    rows: List[Dict[str, Any]] = []
    for page in range(_NBA_JOIN_MAX_PAGES):
        offset = page * _TE_PAGE_SIZE
        query = client.table("patient_journeys").select(cols)
        query = apply_provenance_filter(query)
        query = query.range(offset, offset + _TE_PAGE_SIZE - 1)
        result = await query.execute()
        batch: List[Dict[str, Any]] = result.data or []
        rows.extend(batch)
        if len(batch) < _TE_PAGE_SIZE:
            break
    return rows


async def _load_nba_triggers_join_frame(
    *,
    treatment_var: str,
    outcome_var: str,
    baseline_covariates: List[str],
    limit: int,
    brand: Optional[str],
    covariates: Optional[List[str]] = None,
) -> tuple["pd.DataFrame", List[str]]:  # type: ignore[name-defined] # noqa: F821
    """#1188/#1872: build the nba_triggers estimation frame WITH patient-joined
    columns — triggers (question columns, patient_id) JOIN patient_journeys on
    patient_id. Two channels ride the same join: opt-in pre-treatment BASELINES
    (#1188 ANCOVA efficiency role) and the acceptance edge's BACKDOOR
    covariates (#1872 de-confounding role, curated in ``spec["covariate"]``).

    Same fail-closed discipline as every grain loader: 400 on any column
    outside the curated allowlists, 503 on no store / no usable rows; rows
    missing a treatment/outcome value are dropped (designed-NULL outcomes fill
    to 0 first), and rows missing a REQUESTED joined column are dropped too — a
    partially-observed row would silently degrade the adjustment.
    geographic_region is one-hot expanded. Never fabricates rows.
    """
    import pandas as pd

    spec = _CAUSAL_DATASET_SPECS["nba_triggers"]
    allowed_questions = set(spec["treatment"]) | set(spec["outcome"])
    not_allowed_q = [c for c in (treatment_var, outcome_var) if c not in allowed_questions]
    if not_allowed_q:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) {not_allowed_q} are not permitted for dataset "
                f"'nba_triggers'. Allowed: {sorted(allowed_questions)}"
            ),
        )
    covariates = list(covariates or [])
    allowed_covs = set(spec["covariate"])
    not_allowed_c = [c for c in covariates if c not in allowed_covs]
    if not_allowed_c:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) {not_allowed_c} are not permitted as nba_triggers "
                f"covariates. Allowed: {sorted(allowed_covs)}"
            ),
        )
    allowed_baselines = set(spec.get("baseline_covariate", []))
    not_allowed_b = [c for c in baseline_covariates if c not in allowed_baselines]
    if not_allowed_b:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) {not_allowed_b} are not permitted as nba_triggers "
                f"baseline covariates. Allowed: {sorted(allowed_baselines)}"
            ),
        )

    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Causal data store unavailable")

    question_cols = list(dict.fromkeys([treatment_var, outcome_var]))
    trigger_rows = await _load_trigger_question_rows(client, question_cols, brand)
    if not trigger_rows:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable estimation rows for the requested variables "
                f"({treatment_var} -> {outcome_var}) in dataset 'nba_triggers'."
            ),
        )
    # #1872: covariates first (backdoor role leads the resolved order), then the
    # baselines that are not already covariates — one fetch, one value per column.
    joined_cols = list(dict.fromkeys([*covariates, *baseline_covariates]))
    patient_rows = await _load_patient_baseline_rows(client, joined_cols)
    if not patient_rows:
        raise HTTPException(
            status_code=503, detail="patient_journeys joined covariates unavailable"
        )

    trigger_df = pd.DataFrame(trigger_rows)
    patient_df = pd.DataFrame(patient_rows).drop_duplicates(subset="patient_id")
    merged = trigger_df.merge(patient_df, on="patient_id", how="inner")
    if merged.empty:
        raise HTTPException(status_code=503, detail="nba_triggers patient JOIN produced no rows")

    numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get("nba_triggers", set())
    derivations = _CAUSAL_NUMERIC_DERIVATIONS.get("nba_triggers")
    fill_zero = frozenset(_CAUSAL_FILL_ZERO_OUTCOMES.get("nba_triggers", set()))
    numeric_joined = [c for c in joined_cols if c not in _NBA_BASELINE_CATEGORICALS]
    categorical_joined = [c for c in joined_cols if c in _NBA_BASELINE_CATEGORICALS]

    records: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        rec = _coerce_estimation_row(
            {c: row.get(c) for c in question_cols},
            select_cols=question_cols,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            numeric_cols=numeric_cols,
            derivations=derivations,
            fill_zero=fill_zero,
        )
        if rec is None:
            continue
        usable = True
        for col in joined_cols:
            value = row.get(col)
            if value is not None and pd.isna(value):
                value = None
            if col in numeric_joined and value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
            if value is None:
                usable = False  # missing joined column -> row cannot be adjusted
                break
            rec[col] = value
        if not usable:
            continue
        records.append(rec)
        if len(records) >= limit:
            break

    if not records:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable estimation rows for the requested variables "
                f"({treatment_var} -> {outcome_var}) with joined columns "
                f"{joined_cols} in dataset 'nba_triggers'."
            ),
        )

    frame = pd.DataFrame(records)
    frame, dummy_names = _one_hot_categoricals(frame, categorical_joined)
    select_cols = [*question_cols, *numeric_joined, *dummy_names]
    return frame, select_cols


async def _load_agent_estimation_frame(
    *,
    dataset: str,
    treatment_var: str,
    outcome_var: str,
    covariates: List[str],
    limit: int,
    brand: Optional[str] = None,
    baseline_covariates: Optional[List[str]] = None,
    passthrough_columns: Optional[List[str]] = None,
) -> tuple["pd.DataFrame", List[str]]:  # type: ignore[name-defined] # noqa: F821
    """Load a REAL estimation DataFrame for the causal_impact agent.

    Mirrors :func:`get_causal_estimation_data` (validates columns against the
    dataset's curated allowlist, provenance-filters, drops rows missing a
    treatment/outcome value) but returns a pandas DataFrame ready for the agent's
    ``data_cache['estimation_data']``. Fail-closed: raises ``HTTPException`` (404
    unknown dataset, 400 disallowed column, 503 no data store / no usable rows) —
    never fabricates rows.

    ``passthrough_columns`` (#2007): extra columns fetched on the SAME rows
    for a consumer other than the estimator — today the negative-control
    outcome the refutation node re-fits. They are validated against the
    dataset's allowlist like every other column but hold NO covariate role
    (``_require_covariate_role`` does not see them), get the dataset's numeric
    coercion / fill_zero like any outcome column, are NEVER one-hot expanded,
    NEVER appear in the returned ``expanded_cols`` (the adjustment set the
    caller passes as confounders), and their NULLs NEVER drop a row from the
    primary frame — an all-NULL passthrough column is kept (the node reports
    ``negative_control_column_missing`` / ``too_few_rows`` itself). The
    JOIN datasets refuse them (400) rather than silently dropping them.
    """
    passthrough = [str(c) for c in dict.fromkeys(passthrough_columns or [])]

    # hcp_adoption is a JOIN dataset (hcp_brand_adoption JOIN hcp_profiles), not a
    # single table — route it to the JOIN-aware loader (same allowlist/coercion gate).
    if passthrough and (
        dataset == "hcp_adoption"
        or (dataset == "nba_triggers" and (covariates or baseline_covariates))
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"passthrough column(s) {passthrough} are not supported on the JOIN "
                f"loader for dataset '{dataset}' (no negative-control outcome is "
                "declared for it; extend the JOIN loader before declaring one)."
            ),
        )
    if dataset == "hcp_adoption":
        return await _load_hcp_adoption_join_frame(
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            covariates=covariates,
            limit=limit,
            brand=brand,
        )

    # #1188/#1872: nba_triggers becomes JOIN-aware whenever any patient-joined
    # column is requested — opt-in baselines (ANCOVA) or backdoor covariates
    # (the acceptance edge's de-confounding set). The bare RCT question keeps
    # the single-table path below.
    if dataset == "nba_triggers" and (covariates or baseline_covariates):
        return await _load_nba_triggers_join_frame(
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            covariates=covariates,
            baseline_covariates=list(baseline_covariates or []),
            limit=limit,
            brand=brand,
        )

    spec = _CAUSAL_DATASET_SPECS.get(dataset)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown causal dataset '{dataset}'. "
                f"Known datasets: {sorted(_CAUSAL_DATASET_SPECS)}"
            ),
        )

    allowed = set(spec["treatment"]) | set(spec["outcome"]) | set(spec["covariate"])
    requested = [treatment_var, outcome_var, *covariates]
    not_allowed = [c for c in [*requested, *passthrough] if c not in allowed]
    if not_allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) {not_allowed} are not permitted for dataset "
                f"'{dataset}'. Allowed: {sorted(allowed)}"
            ),
        )
    _require_covariate_role(dataset, spec, covariates)
    # A passthrough column already in a question/covariate slot is that slot's
    # column (deduped below); only the passthrough-ONLY names are excluded from
    # the adjustment set, the one-hot expansion and the all-NULL drop.
    passthrough_only = [c for c in passthrough if c not in requested]

    # #1872 (codex iter-2): the union allowlist above is role-insensitive, so a
    # patient-JOINED covariate could ride a question slot into this
    # single-table path (only reachable with an empty covariate set — non-empty
    # routes to the join loader, which validates questions by role). It is not
    # a triggers column: fail closed before the read, never a PostgREST 42703.
    if dataset == "nba_triggers":
        joined_req = [c for c in requested if c in _NBA_JOINED_COVARIATES]
        if joined_req:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Column(s) {joined_req} are patient-joined on nba_triggers "
                    "(patient_journeys columns) and cannot serve as single-table "
                    "treatment/outcome columns."
                ),
            )

    select_cols = list(dict.fromkeys([*requested, *passthrough_only]))

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Causal data store unavailable")

    # ``brand`` is a categorical FILTER (row subset), NOT a causal variable — it
    # scopes the cohort to one brand and stays out of the estimation columns
    # (categorical confounders would need encoding the executors don't do here).
    fetch_cols = list(select_cols)
    brand_col = _CAUSAL_BRAND_COLUMN.get(dataset, "brand")
    if brand:
        fetch_cols = list(dict.fromkeys([*select_cols, brand_col]))
    query = client.table(_CAUSAL_PHYSICAL_TABLE.get(dataset, dataset)).select(",".join(fetch_cols))
    query = apply_provenance_filter(query)
    if brand:
        query = query.eq(brand_col, brand)
    result = await query.limit(limit).execute()
    rows = result.data or []

    return _resolve_agent_estimation_frame(
        rows,
        dataset=dataset,
        treatment_var=treatment_var,
        outcome_var=outcome_var,
        select_cols=select_cols,
        passthrough_only=passthrough_only,
        brand=brand,
    )


def _resolve_agent_estimation_frame(
    rows: List[Dict[str, Any]],
    *,
    dataset: str,
    treatment_var: str,
    outcome_var: str,
    select_cols: List[str],
    passthrough_only: List[str],
    brand: Optional[str],
) -> tuple["pd.DataFrame", List[str]]:  # type: ignore[name-defined] # noqa: F821
    """Everything the agent loader does AFTER the rows are fetched: per-row
    coercion, the constant-treatment refusal, the all-NULL covariate drop, the
    one-hot expansion and the exact-collinearity prune -- returning
    ``(frame, [treatment, outcome, *resolved covariates])``.

    Split out of :func:`_load_agent_estimation_frame` so an offline run on the
    same rows (the Lane A pre-flight reads the exported parquet) goes through
    the IDENTICAL resolution; a re-implementation of this path in the
    pre-flight script silently skipped the collinearity prune on 2026-09-22.
    """
    numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get(dataset, set())
    categorical_cols = _CAUSAL_CATEGORICAL_COLUMNS.get(dataset, set())
    records: List[Dict[str, Any]] = []
    for row in rows:
        rec = _coerce_estimation_row(
            row,
            select_cols=select_cols,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            numeric_cols=numeric_cols,
            categorical_cols=frozenset(categorical_cols),
            derivations=_CAUSAL_NUMERIC_DERIVATIONS.get(dataset),
            fill_zero=frozenset(_CAUSAL_FILL_ZERO_OUTCOMES.get(dataset, set())),
        )
        if rec is not None:
            records.append(rec)

    if not records:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable estimation rows for the requested variables "
                f"({treatment_var} -> {outcome_var}) in dataset '{dataset}'."
            ),
        )

    import pandas as pd

    frame = pd.DataFrame(records)

    # A causal contrast needs both arms. On a dataset where the brand filter IS
    # the treatment label (optum_biologic_persistence: index_biologic_brand ==
    # treatment_dupixent), scoping to one brand makes the treatment column
    # constant — NOT a loud failure downstream: DoWhy still returns a finite
    # estimate on a constant treatment, and refutation.py's nunique()==2 check
    # silently switches to the continuous-treatment path instead of refusing.
    # Fail closed HERE, before that can happen.
    if frame[treatment_var].nunique(dropna=True) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Treatment '{treatment_var}' is constant in the loaded rows "
                f"(n={len(frame)}, value={frame[treatment_var].iloc[0]!r}) for "
                f"dataset '{dataset}' brand={brand!r}; a causal contrast needs "
                "both arms — widen the cohort (omit `brand` when the brand IS "
                "the treatment)."
            ),
        )

    # Defensive net (Phase 2 brand-gating): after gating, an off-brand clinical
    # covariate can reach here 100% NULL (e.g. a stale/direct request that bypassed
    # the brand-aware adjustment-set selection). An all-NULL covariate carries no
    # information and would crash EconML ("Input contains NaN"), so drop it from the
    # adjustment set — NEVER the treatment/outcome (those already drop rows on a
    # missing value via _coerce_estimation_row), and never a passthrough column
    # (#2007: it is not adjusted on, so an all-NULL one cannot crash EconML; the
    # node that consumes it reports the missing/too-few condition itself).
    protected = {treatment_var, outcome_var, *passthrough_only}
    dropped_null = [
        c
        for c in select_cols
        if c not in protected and c in frame.columns and bool(frame[c].isna().all())
    ]
    if dropped_null:
        logger.warning(
            "causal loader: dropping all-NULL covariate(s) %s for dataset '%s' "
            "brand=%s (brand-gated off-brand clinical column)",
            dropped_null,
            dataset,
            brand,
        )
        frame = frame.drop(columns=dropped_null)
        select_cols = [c for c in select_cols if c not in dropped_null]
    null_passthrough = [
        c for c in passthrough_only if c in frame.columns and bool(frame[c].isna().all())
    ]
    if null_passthrough:
        logger.debug(
            "causal loader: passthrough column(s) %s are all-NULL for dataset '%s' "
            "brand=%s — kept (the consumer reports the condition)",
            null_passthrough,
            dataset,
            brand,
        )

    requested_categoricals = [
        c for c in select_cols if c in categorical_cols and c not in passthrough_only
    ]
    frame, dummy_names = _one_hot_categoricals(frame, requested_categoricals)
    expanded_cols = [
        c for c in select_cols if c not in categorical_cols and c not in passthrough_only
    ] + dummy_names

    # Exactly collinear covariates carry no information and make econml's
    # statsmodels final stage warn "Co-variance matrix is underdetermined.
    # Inference will be invalid!" (the wrappers now REFUSE such a fit). Prune
    # them ONCE here, on the full loaded frame, in registry order (an earlier
    # column wins), so the estimation node and the refutation rebuild -- which
    # takes its ``common_causes`` from this same resolved list -- see the SAME
    # design. Measured 2026-09-22 on optum_biologic_persistence (n=15,209):
    # 16 of 77 resolved columns were exact linear combinations of earlier ones
    # (Elixhauser flags duplicating Charlson flags, a risk band implied by its
    # score, payer dummies implied by a coarser payer axis); dropping them left
    # the ATE at 0.03353 (was 0.03353) and the SE at 0.00858 (was 0.00855),
    # warning gone. A full-rank frame (every synthetic dataset) is untouched.
    covariate_only = [c for c in expanded_cols if c not in (treatment_var, outcome_var)]
    kept, dropped_collinear = _prune_exactly_collinear(frame, covariate_only)
    if dropped_collinear:
        logger.warning(
            "causal loader: dropping exactly collinear covariate(s) %s for dataset "
            "'%s' brand=%s (linear combinations of earlier registry columns; design "
            "rank %d of %d)",
            dropped_collinear,
            dataset,
            brand,
            len(kept),
            len(covariate_only),
        )
        keep_set = set(kept)
        expanded_cols = [
            c for c in expanded_cols if c in (treatment_var, outcome_var) or c in keep_set
        ]
    return frame, expanded_cols


_COLLINEARITY_REL_TOL = 1e-8


def _prune_exactly_collinear(
    frame: "pd.DataFrame",  # type: ignore[name-defined] # noqa: F821
    columns: List[str],
) -> tuple[List[str], List[str]]:
    """Return ``(kept, dropped)``: ``columns`` minus those that are exact linear
    combinations of the intercept and the EARLIER kept columns (order preserved).

    Incremental Gram-Schmidt against the intercept: a column whose residual after
    projection onto the current basis is below ``_COLLINEARITY_REL_TOL`` of its
    own norm adds no rank. Exact redundancy sits at the 1e-15 level and any real
    near-collinear pair far above 1e-8, so the tolerance separates the two.

    Skipped (nothing dropped) when the frame cannot rank the columns
    (``n <= k + 1`` -- every such design is rank-deficient, which says nothing
    about the columns) or when a column is non-finite (the estimators' own
    guards own NaN). A constant column is collinear with the intercept and is
    dropped like any other.
    """
    if not columns or len(frame) <= len(columns) + 1:
        return list(columns), []
    import numpy as np

    X = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        return list(columns), []
    n = X.shape[0]
    basis = [np.full(n, 1.0 / np.sqrt(n))]  # the intercept, unit norm
    kept: List[str] = []
    dropped: List[str] = []
    for j, name in enumerate(columns):
        x = X[:, j]
        x_norm = float(np.linalg.norm(x))
        resid = x.copy()
        for _ in range(2):  # re-orthogonalise once for numerical stability
            for q in basis:
                resid = resid - q * float(q @ resid)
        r_norm = float(np.linalg.norm(resid))
        if x_norm == 0.0 or r_norm <= _COLLINEARITY_REL_TOL * x_norm:
            dropped.append(name)
            continue
        basis.append(resid / r_norm)
        kept.append(name)
    return kept, dropped
