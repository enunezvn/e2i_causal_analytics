"""KPI synthetic-visibility mode (demo / review instances).

Production default-EXCLUDES synthetic-tagged rows from every KPI read: migration
066 wraps each taggable ``kpi_query_registry`` statement in
``(SELECT * FROM <t> WHERE is_synthetic = false)`` and registers a parallel
``{query_id}_include_synthetic`` family holding the verbatim (unwrapped)
originals. That gate is the right default for a true-production instance carrying
real RWD alongside synthetic.

For a demo / review instance whose ONLY data is synthetic-gold (the 2026-06-11
"synthetic-gold-only" cleanup), the gate makes the whole KPI dashboard read
honest-empty. The ``E2I_KPI_INCLUDE_SYNTHETIC`` flag flips KPI reads to the
``*_include_synthetic`` twins so the dashboard renders from the existing
synthetic data —

* **reversibly**: unset the flag and the strict gate is restored verbatim (this
  module only rewrites the ``query_id`` string; the SQL is untouched), and
* **honestly**: callers that observe :func:`kpi_include_synthetic` surface a
  ``data_source="synthetic"`` provenance so the UI badges the figures rather than
  passing them off as real-world data.

Only the 41 query_ids that touch synthetic-taggable business tables HAVE a twin
(see :data:`SYNTHETIC_TWINNED_QUERY_IDS`, sourced from migrations 066/085/095). Every
other registry statement reads a non-taggable view / reference / ops table and
is therefore NOT synthetic-gated — its base id is already correct.
:func:`resolve_kpi_query_id` swaps to the twin ONLY when one exists; otherwise it
returns the base id unchanged, so it is safe and idempotent to apply at every
``kpi_query`` call site.
"""

from __future__ import annotations

import os

_SYNTHETIC_SUFFIX = "_include_synthetic"

#: Truthy spellings accepted for the ``E2I_*`` runtime flags (mirrors the
#: ``E2I_ENABLE_SIMULATED_FALLBACK`` / ``E2I_ENABLE_PLACEHOLDER_ACTIONS`` idiom in
#: ``src/api/routes/copilotkit.py``).
_TRUTHY = ("1", "true", "yes")

#: The 41 base ``kpi_query_registry`` ids that have an
#: ``{id}_include_synthetic`` twin: 36 sourced verbatim from
#: ``database/migrations/066_kpi_query_synthetic_exclusion.sql``, plus
#: ``business_impact_patient_touch_rate`` from
#: ``database/migrations/085_kpi_patient_touch_rate_include_synthetic.sql``
#: (#1064 — the touch-rate KPI reads a VIEW that migration 067 made
#: synthetic-excluding, so it was absent from 066's table-wrapping pass and
#: needed a view-backed twin), plus the four view-backed WS1 data-quality ids
#: (cross_source_match / stacking_lift / data_lag / time_to_release) from
#: ``database/migrations/095_kpi_dq_view_include_synthetic_twins.sql`` (the
#: same 066 gap as #1064 — their views are synthetic-excluding per 067, so
#: /data-quality read honest-empty on a synthetic-gold instance). The twins are
#: the synthetic-taggable statements; everything else in the registry is not
#: synthetic-gated. This literal is kept in lock-step with the migrations by
#: ``tests/unit/test_kpi/test_synthetic_mode.py`` (it parses 066 + 085 + 095
#: and asserts equality — drift fails CI), so a future twin added by a later
#: migration is a one-line update guarded by a red test, never a silent miss.
SYNTHETIC_TWINNED_QUERY_IDS: frozenset[str] = frozenset(
    {
        "brand_specific_fabhalta_pnh_tested",
        "brand_specific_kisqali_dx_adoption",
        "brand_specific_kisqali_oncologist_reach",
        "brand_specific_remi_ah_uncontrolled",
        "brand_specific_remi_intent_delta_fallback",
        "business_impact_conversion_rate",
        "business_impact_data_through",
        "business_impact_hcp_coverage",
        "business_impact_hcp_reach",
        "business_impact_mau_fallback",
        "business_impact_nbrx",
        "business_impact_nrx",
        "business_impact_patient_touch_rate",
        "business_impact_roi_agent_activities",
        "business_impact_roi_business_metrics",
        "business_impact_trx",
        "business_impact_trx_share",
        "business_impact_wau_fallback",
        "causal_metrics_ate",
        "causal_metrics_cate",
        "causal_metrics_causal_impact",
        "causal_metrics_counterfactual",
        "causal_metrics_mediation",
        "data_quality_completeness_pass_rate",
        "data_quality_cross_source_match",
        "data_quality_data_lag",
        "data_quality_geographic_consistency",
        "data_quality_source_coverage_hcps",
        "data_quality_source_coverage_patients",
        "data_quality_stacking_lift",
        "data_quality_time_to_release",
        "model_performance_roc_auc",
        "model_performance_shap_coverage",
        "trigger_performance_acceptance_rate",
        "trigger_performance_action_rate_uplift",
        "trigger_performance_cfr",
        "trigger_performance_false_alert_rate",
        "trigger_performance_lead_time",
        "trigger_performance_override_rate",
        "trigger_performance_precision",
        "trigger_performance_recall",
    }
)


def kpi_include_synthetic() -> bool:
    """Whether KPI reads should INCLUDE synthetic-tagged rows (demo / review mode).

    Reads ``E2I_KPI_INCLUDE_SYNTHETIC`` fresh on every call (truthy:
    ``1`` / ``true`` / ``yes``, case-insensitive), mirroring the other
    ``E2I_*`` runtime flags so it can be toggled per-deployment without a
    restart-coupled import-time capture. ALSO honors the deployment-wide
    ``E2I_INCLUDE_SYNTHETIC`` showcase switch (the generalized flag in
    :func:`src.repositories.provenance.deployment_includes_synthetic`) so ONE env
    makes the whole synthetic-gold instance run at full potential. Defaults to
    ``False`` → production's strict synthetic-exclusion gate (migration 066) is
    preserved untouched.
    """
    return (
        os.getenv("E2I_KPI_INCLUDE_SYNTHETIC", "0").strip().lower() in _TRUTHY
        or os.getenv("E2I_INCLUDE_SYNTHETIC", "0").strip().lower() in _TRUTHY
    )


def resolve_kpi_query_id(query_id: str) -> str:
    """Map a base ``kpi_query`` id to its synthetic-inclusive twin in demo mode.

    Returns ``{query_id}_include_synthetic`` iff :func:`kpi_include_synthetic`
    is true AND a twin exists for ``query_id`` (per migration 066). Otherwise —
    flag off, an already-resolved twin id, or a twinless (non-gated) id — returns
    ``query_id`` unchanged. The pass-through cases make this idempotent and safe
    to call at every ``kpi_query`` RPC site without per-id bookkeeping.
    """
    if not kpi_include_synthetic():
        return query_id
    if query_id.endswith(_SYNTHETIC_SUFFIX):
        return query_id
    if query_id in SYNTHETIC_TWINNED_QUERY_IDS:
        return query_id + _SYNTHETIC_SUFFIX
    return query_id


def region_query_id(base_query_id: str) -> str:
    """Region-scoped query id for a base KPI query (migrations 077 / 078).

    The ``*_region`` variants are ADDITIVE and deliberately ABSENT from
    :data:`SYNTHETIC_TWINNED_QUERY_IDS` (which mirrors migration 066 and is
    drift-checked in CI), so :func:`resolve_kpi_query_id` will NOT auto-swap them
    to the synthetic-inclusive twin. We therefore append ``_include_synthetic``
    HERE under the showcase flag, so a region-scoped read honors the same
    synthetic-visibility gate as the base query it parallels. Passing the result
    back through :func:`resolve_kpi_query_id` (as every ``_execute_query`` does)
    is a safe no-op on the already-suffixed id.
    """
    qid = f"{base_query_id}_region"
    return f"{qid}{_SYNTHETIC_SUFFIX}" if kpi_include_synthetic() else qid


def brand_scoped_query_id(base_query_id: str) -> str:
    """Brand-scoped query id for a base KPI query whose BASE takes no params
    (migration 111 — conversion_rate).

    Most business_impact base statements already accept a NULLable ``$1``
    brand, so they need no ``_brand`` variant. ``business_impact_conversion_
    rate``'s certified base is param-less (brand-agnostic by original design),
    so the brand-scoped read is an ADDITIVE ``{base}_brand`` sibling instead of
    an in-place edit. Same suffixing rules as :func:`region_query_id`.
    """
    qid = f"{base_query_id}_brand"
    return f"{qid}{_SYNTHETIC_SUFFIX}" if kpi_include_synthetic() else qid


def brand_region_query_id(base_query_id: str) -> str:
    """Brand+region-scoped query id (migration 113 — WS2 trigger variants).

    ``{base}_brand_region`` binds brand as ``$1`` and region as ``$2``. Same
    additive-sibling + self-suffixing rules as :func:`region_query_id` /
    :func:`brand_scoped_query_id` (absent from
    :data:`SYNTHETIC_TWINNED_QUERY_IDS`; ``resolve_kpi_query_id`` no-ops on the
    already-suffixed id).
    """
    qid = f"{base_query_id}_brand_region"
    return f"{qid}{_SYNTHETIC_SUFFIX}" if kpi_include_synthetic() else qid


def windowed_query_id(base_query_id: str, *, region: bool) -> str:
    """Windowed variant id for a base KPI query (Phase 1, additive).

    Canonical suffix order: ``{base}_windowed[_region][_include_synthetic]``.
    Parallels :func:`region_query_id`: the ``_windowed*`` variants are ADDITIVE
    and absent from :data:`SYNTHETIC_TWINNED_QUERY_IDS`, so we append the
    ``_include_synthetic`` suffix HERE under the showcase flag. Passing the
    result back through :func:`resolve_kpi_query_id` is a safe no-op.
    """
    qid = f"{base_query_id}_windowed"
    if region:
        qid = f"{qid}_region"
    return f"{qid}{_SYNTHETIC_SUFFIX}" if kpi_include_synthetic() else qid


def _axis_query_id(base_query_id: str, suffix: str) -> str:
    """Shared helper behind :func:`segment_query_id` / :func:`line_query_id`.

    Appends ``suffix`` (``"_segment"`` / ``"_line"``) to ``base_query_id`` and,
    same as :func:`region_query_id`, tacks on ``_include_synthetic`` HERE under
    the showcase flag -- the axis variants (migration 105) are ADDITIVE and
    absent from :data:`SYNTHETIC_TWINNED_QUERY_IDS`, so
    :func:`resolve_kpi_query_id` will not auto-swap them.
    """
    qid = f"{base_query_id}{suffix}"
    return f"{qid}{_SYNTHETIC_SUFFIX}" if kpi_include_synthetic() else qid


def segment_query_id(base_query_id: str) -> str:
    """Severity-tier-scoped query id for a base KPI query (migration 105).

    Severity tier is ``patient_journeys.segment_assignment`` (low/medium/high).
    Parallels :func:`region_query_id`; see :func:`_axis_query_id`.
    """
    return _axis_query_id(base_query_id, "_segment")


def line_query_id(base_query_id: str) -> str:
    """Line-of-therapy-scoped query id for a base KPI query (migration 105).

    Line of therapy is ``patient_journeys.prior_therapy_lines`` (0-3).
    Parallels :func:`region_query_id`; see :func:`_axis_query_id`.
    """
    return _axis_query_id(base_query_id, "_line")


def biologic_query_id(base_query_id: str) -> str:
    """Biologic-status-scoped query id for a base KPI query (migration 108).

    Biologic status is ``patient_journeys.biologic_experienced`` mapped to
    ``'naive'`` / ``'experienced'``; populated for Remibrutinib (CSU) rows only.
    Parallels :func:`region_query_id`; see :func:`_axis_query_id`.
    """
    return _axis_query_id(base_query_id, "_biologic")


def ige_tier_query_id(base_query_id: str) -> str:
    """IgE-tertile-scoped query id for a base KPI query (migration 108).

    IgE tertile bins ``patient_journeys.ige_level`` into data-driven
    ``'low'`` / ``'medium'`` / ``'high'`` (empirical p33/p66 of the Remibrutinib
    distribution); populated for Remibrutinib rows only.
    Parallels :func:`region_query_id`; see :func:`_axis_query_id`.
    """
    return _axis_query_id(base_query_id, "_ige_tier")


def windowed_axis_query_id(base_query_id: str, *, axis: str) -> str:
    """Windowed variant id for a segment/line/biologic/ige_tier-scoped base query.

    Canonical suffix order: ``{base}_{axis}_windowed[_include_synthetic]``
    (mirrors :func:`windowed_query_id`'s ``{base}_windowed[_region]`` order,
    axis-first since the axis is baked into the query id rather than a
    trailing modifier). ``axis`` is ``"segment"`` / ``"line"`` (migration 105)
    or ``"biologic"`` / ``"ige_tier"`` (migration 108).
    """
    qid = f"{base_query_id}_{axis}_windowed"
    return f"{qid}{_SYNTHETIC_SUFFIX}" if kpi_include_synthetic() else qid


def trigger_effectiveness_query_id(metric: str, *, windowed: bool, regioned: bool = False) -> str:
    """Ask-bound trigger-effectiveness statement id (migrations 118/120, #1360/#1388).

    Canonical order: ``trigger_effectiveness_{metric}[_windowed][_region]
    [_include_synthetic]``. ``metric`` is ``"precision"`` / ``"acceptance_rate"``
    / ``"override_rate"`` / ``"funnel_conversion"`` — the four KPIs the #1360
    ruling assigned to the chat KPI path.

    ``regioned=True`` selects the migration-120 ``_windowed_region`` variant that
    co-binds region with an explicit window ($1 brand, $2 region, $3 trigger_type,
    $4/$5 window) — only meaningful WITH ``windowed=True``, since the non-windowed
    form already binds region as a nullable param and needs no id suffix.
    ``regioned=True`` without ``windowed=True`` is a programming error and raises.

    Same additive/suffixing rules as :func:`windowed_query_id`: these ids are
    ADDITIVE and absent from :data:`SYNTHETIC_TWINNED_QUERY_IDS`, so the
    ``_include_synthetic`` suffix is appended HERE under the showcase flag and
    :func:`resolve_kpi_query_id` is a safe no-op on the result.
    """
    if regioned and not windowed:
        raise ValueError(
            "trigger_effectiveness_query_id: regioned=True is only valid with "
            "windowed=True (the non-windowed form binds region as a nullable "
            "param, no id suffix)"
        )
    qid = f"trigger_effectiveness_{metric}"
    if windowed:
        qid = f"{qid}_windowed"
        if regioned:
            qid = f"{qid}_region"
    return f"{qid}{_SYNTHETIC_SUFFIX}" if kpi_include_synthetic() else qid


def nowcast_triangle_query_id(base_query_id: str) -> str:
    """Claims-arrival lag-triangle variant id for an Rx-volume base query
    (migration 116, backlog #45).

    ``{base}_nowcast_triangle[_include_synthetic]``: one call returns the full
    per-service-month arrival-offset histogram — rows of (service_month,
    arrival_offset_days, n) plus the prescription data_min/frontier scalars —
    feeding the completion-factor nowcast estimator
    (:mod:`src.kpi.nowcast.completion_factor`). Same additive/suffixing rules
    as :func:`monthly_axis_query_id`: the variants are ADDITIVE and absent from
    :data:`SYNTHETIC_TWINNED_QUERY_IDS`, so the suffix is appended HERE under
    the showcase flag.
    """
    qid = f"{base_query_id}_nowcast_triangle"
    return f"{qid}{_SYNTHETIC_SUFFIX}" if kpi_include_synthetic() else qid


def monthly_axis_query_id(base_query_id: str, *, axis: str) -> str:
    """Monthly-series-grouped variant id for an axis-scoped base query.

    Canonical suffix order: ``{base}_monthly_by_{axis}[_include_synthetic]``
    (migration 110): one call returns the full monthly series for ALL buckets
    of the axis — rows of (month_start, bucket, value). ``axis`` is
    ``"segment"`` / ``"line"``. Same additive/suffixing rules as
    :func:`_axis_query_id`.
    """
    qid = f"{base_query_id}_monthly_by_{axis}"
    return f"{qid}{_SYNTHETIC_SUFFIX}" if kpi_include_synthetic() else qid
