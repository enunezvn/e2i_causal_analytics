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

Only the 36 query_ids that touch synthetic-taggable business tables HAVE a twin
(see :data:`SYNTHETIC_TWINNED_QUERY_IDS`, sourced from migration 066). Every
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

#: The 37 base ``kpi_query_registry`` ids that have an
#: ``{id}_include_synthetic`` twin: 36 sourced verbatim from
#: ``database/migrations/066_kpi_query_synthetic_exclusion.sql`` plus
#: ``business_impact_patient_touch_rate`` from
#: ``database/migrations/085_kpi_patient_touch_rate_include_synthetic.sql``
#: (#1064 — the touch-rate KPI reads a VIEW that migration 067 made
#: synthetic-excluding, so it was absent from 066's table-wrapping pass and
#: needed a view-backed twin). The twins are the synthetic-taggable statements;
#: everything else in the registry is not synthetic-gated. This literal is kept
#: in lock-step with the migrations by ``tests/unit/test_kpi/test_synthetic_mode.py``
#: (it parses 066 + 085 and asserts equality — drift fails CI), so a future twin
#: added by a later migration is a one-line update guarded by a red test, never a
#: silent miss.
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
        "data_quality_geographic_consistency",
        "data_quality_source_coverage_hcps",
        "data_quality_source_coverage_patients",
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
