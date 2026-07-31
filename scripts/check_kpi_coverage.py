"""KPI coverage probe (Shard 09).

Maps every one of the 45 KPIs in config/kpi_definitions.yaml to its BASE
kpi_query registry id, then resolves each id through the SAME production
synthetic-mode resolver the deployed chat KPI path uses (src.kpi.synthetic_mode)
before running it against the faithful docker Supabase, and reports non-NULL /
EMPTY per KPI. Storing base ids and resolving dynamically keeps the probe in
lock-step with the twin migrations (066/085/095/118): a future twin is picked up
automatically instead of stranding a row on a now-empty base id -- the #1389
regression, where migrations 085/095 added twins the hand-maintained map never
adopted, so five rows kept probing base ids that read honest-null after the
synthetic-gold provenance reseed.

Resolution mirrors production exactly (see resolved_probe_id):
  * synthetic-gated ids (the 066/085/095 twin family) -> resolve_kpi_query_id,
    which swaps to the `{id}_include_synthetic` twin when the synthetic flag is on
    and is a safe no-op on twinless ids (e.g. model_performance_feature_drift);
  * the additive WS2-TR-009 trigger-effectiveness id (migration 118, #1360) ->
    trigger_effectiveness_query_id, its own resolver (deliberately absent from the
    resolve_kpi_query_id twin family).

The serving mode is read from the live E2I_INCLUDE_SYNTHETIC /
E2I_KPI_INCLUDE_SYNTHETIC flags (kpi_include_synthetic) and printed, so a reader
knows which mode was measured. On the synthetic-gold instance the flag is ON, so
the probe measures the synthetic substrate the chat path actually serves; with
the flag off it measures production's strict synthetic-exclusion gate.

Run (faithful):  E2I_DB_INTEGRATION=1 E2I_INCLUDE_SYNTHETIC=1 python scripts/check_kpi_coverage.py
Run (offline):   E2I_INCLUDE_SYNTHETIC=1 python scripts/check_kpi_coverage.py   # prints the resolved map only

A KPI is MAPPED when its probe returns a row whose primary metric value is
non-NULL after a synthetic load. Param values are chosen to exercise the
synthetic substrate (brand = Kisqali where a brand param is required).
"""

import json
import os
import subprocess
import sys

import yaml

from src.kpi.synthetic_mode import (
    kpi_include_synthetic,
    resolve_kpi_query_id,
    trigger_effectiveness_query_id,
)

# Direct-substrate COUNT probes for KPIs the model-performance agent computes by
# reading ml_predictions columns directly (no kpi_query registry entry exists for
# WS1-MP-002/003/004/005/006/008). The KPI is MAPPED when the column is populated
# on the synthetic substrate (is_synthetic=true). These exercise the SAME columns
# the agent reads; an RPC would only wrap an identical aggregate.
DIRECT_PROBES: dict[str, str] = {
    "WS1-MP-002": "SELECT count(*) AS n FROM ml_predictions WHERE is_synthetic AND model_pr_auc IS NOT NULL",
    "WS1-MP-003": "SELECT count(*) AS n FROM ml_predictions WHERE is_synthetic AND model_precision IS NOT NULL AND model_recall IS NOT NULL",
    "WS1-MP-004": "SELECT count(*) AS n FROM ml_predictions WHERE is_synthetic AND rank_metrics IS NOT NULL",
    "WS1-MP-005": "SELECT count(*) AS n FROM ml_predictions WHERE is_synthetic AND brier_score IS NOT NULL",
    "WS1-MP-006": "SELECT count(*) AS n FROM ml_predictions WHERE is_synthetic AND calibration_score IS NOT NULL",
}

# KPI id -> (BASE registry query_id, params_json, value_key_substring_or_None)
# The stored id is ALWAYS the base registry id; resolved_probe_id() maps it to the
# id the deployed KPI path serves (the `_include_synthetic` twin under the
# synthetic flag). Do NOT hard-code a twin id here -- that is the #1389 drift.
# value_key_substring_or_None: a substring that must appear as a non-null JSON key
# in the returned row (None = any returned row counts).
PROBES: dict[str, tuple[str, str, str | None]] = {
    # --- WS1 data quality ---
    "WS1-DQ-001": ("data_quality_source_coverage_patients", '["Kisqali"]', "covered"),
    "WS1-DQ-002": ("data_quality_source_coverage_hcps", "[]", "covered"),
    "WS1-DQ-003": ("data_quality_cross_source_match", "[]", "match_rate"),
    "WS1-DQ-004": ("data_quality_stacking_lift", "[]", "lift_score"),
    "WS1-DQ-005": ("data_quality_completeness_pass_rate", "[]", "pass_rate"),
    "WS1-DQ-006": ("data_quality_geographic_consistency", '["Kisqali"]', "max_gap"),
    "WS1-DQ-007": ("data_quality_data_lag", "[]", "median_lag_days"),
    # WS1-DQ-008 (Label Quality / IAA) removed in T8 — product decision (working metric).
    "WS1-DQ-009": ("data_quality_time_to_release", "[]", "avg_ttr_hours"),
    # --- WS1 model performance ---
    "WS1-MP-001": ("model_performance_roc_auc", "[]", "roc_auc"),
    # WS1-MP-002/003/004/005/006 -> DIRECT_PROBES (no kpi_query registry entry).
    # WS1-MP-009 model_performance_feature_drift is NOT synthetic-gated (no twin);
    # resolved_probe_id passes it through unchanged even under the synthetic flag.
    "WS1-MP-007": ("model_performance_shap_coverage", "[]", "coverage"),
    "WS1-MP-009": ("model_performance_feature_drift", "[]", "avg_psi"),
    # --- WS2 triggers ---
    "WS2-TR-001": ("trigger_performance_precision", "[]", "precision"),
    "WS2-TR-002": ("trigger_performance_recall", "[]", "recall"),
    "WS2-TR-003": ("trigger_performance_action_rate_uplift", "[]", "action_rate_uplift"),
    "WS2-TR-004": ("trigger_performance_acceptance_rate", "[]", "acceptance_rate"),
    "WS2-TR-005": ("trigger_performance_false_alert_rate", "[]", "false_alert_rate"),
    "WS2-TR-006": ("trigger_performance_override_rate", "[]", "override_rate"),
    "WS2-TR-007": ("trigger_performance_lead_time", "[]", "median_lead_time"),
    "WS2-TR-008": ("trigger_performance_cfr", "[]", "cfr"),
    # #1360 trigger-effectiveness family (migration 118): the base statement binds
    # $1 brand / $2 region / $3 trigger_type — all nullable. This id is ADDITIVE and
    # outside the resolve_kpi_query_id twin family; resolved_probe_id routes it
    # through trigger_effectiveness_query_id (its own resolver).
    "WS2-TR-009": (
        "trigger_effectiveness_funnel_conversion",
        "[null, null, null]",
        "funnel_conversion",
    ),
    # --- WS3 business ---
    "WS3-BI-001": ("business_impact_mau_fallback", "[]", "mau"),
    "WS3-BI-002": ("business_impact_wau_fallback", "[]", "wau"),
    "WS3-BI-003": ("business_impact_patient_touch_rate", '["Kisqali"]', "touch_rate"),
    "WS3-BI-004": ("business_impact_hcp_coverage", "[]", "coverage"),
    "WS3-BI-005": ("business_impact_trx", '["Kisqali"]', "trx"),
    "WS3-BI-006": ("business_impact_nrx", '["Kisqali"]', "nrx"),
    "WS3-BI-007": ("business_impact_nbrx", '["Kisqali"]', "nbrx"),
    "WS3-BI-008": ("business_impact_trx_share", '["Kisqali"]', "share"),
    "WS3-BI-009": ("business_impact_conversion_rate", "[]", "conversion_rate"),
    "WS3-BI-010": ("business_impact_roi_business_metrics", "[]", "avg_roi"),
    # --- brand-specific ---
    "BR-001": ("brand_specific_remi_ah_uncontrolled", '["3"]', "uncontrolled_rate"),
    "BR-002": ("brand_specific_remi_intent_delta_fallback", "[]", "intent_delta"),
    "BR-003": ("brand_specific_fabhalta_pnh_tested", "[]", "tested_rate"),
    "BR-004": ("brand_specific_kisqali_dx_adoption", "[]", "median_days"),
    "BR-005": ("brand_specific_kisqali_oncologist_reach", "[]", "reach"),
    # --- causal metrics ---
    "CM-001": ("causal_metrics_ate", "[]", "ate"),
    "CM-002": ("causal_metrics_cate", '["high_severity"]', "cate"),
    "CM-003": ("causal_metrics_causal_impact", '[""]', "effect"),
    "CM-004": ("causal_metrics_counterfactual", '["churn"]', "mean_counterfactual"),
    "CM-005": ("causal_metrics_mediation", "[]", "proportion_mediated"),
}

# WS2-TR-009's trigger-effectiveness statement (migration 118, #1360) is an
# ADDITIVE id outside the 066/085/095 twin family that resolve_kpi_query_id owns.
# Production resolves it via trigger_effectiveness_query_id; resolved_probe_id
# mirrors that so the probe measures what the chat KPI path actually serves.
_ADDITIVE_TRIGGER_EFFECTIVENESS_BASE = "trigger_effectiveness_funnel_conversion"


def resolved_probe_id(base_query_id: str) -> str:
    """Resolve a PROBES base id to the id the deployed KPI path serves.

    Mirrors production resolution under the live synthetic flag: the additive
    trigger-effectiveness id (#1360) goes through trigger_effectiveness_query_id;
    every other id goes through resolve_kpi_query_id, which swaps a synthetic-
    gated id to its `_include_synthetic` twin when the flag is on and is a safe
    no-op on twinless ids. Idempotent and side-effect free.
    """
    if base_query_id == _ADDITIVE_TRIGGER_EFFECTIVENESS_BASE:
        return trigger_effectiveness_query_id("funnel_conversion", windowed=False)
    return resolve_kpi_query_id(base_query_id)


def _kpi_ids() -> list[str]:
    cfg = yaml.safe_load(open("config/kpi_definitions.yaml"))
    ids: list[str] = []
    for s in (
        "ws1_data_quality",
        "ws1_model_performance",
        "ws2_triggers",
        "ws3_business",
        "brand_specific",
        "causal_metrics",
    ):
        ids += [v["id"] for v in cfg[s].values()]
    return ids


def _psql(sql: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tAc", sql],
        capture_output=True,
        text=True,
    )
    return (out.stdout.strip().splitlines() or [""])[0]


def _run_probe(query_id: str, params: str) -> str:
    return _psql(f"SELECT * FROM kpi_query('{query_id}', $${params}$$::jsonb);")


def _run_direct(sql: str) -> str:
    return _psql(sql)


def _value_present(row_json: str, key: str | None) -> bool:
    if not row_json or row_json.startswith("ERROR"):
        return False
    try:
        obj = json.loads(row_json)
    except json.JSONDecodeError:
        return False
    if key is None:
        return bool(obj)
    return obj.get(key) is not None


def main() -> int:
    ids = _kpi_ids()
    # Drift-proof sanity check: the enumerated section entries must match the YAML's own
    # declared summary total (no hardcoded magic number that silently drifts on add/remove).
    summary_total = yaml.safe_load(open("config/kpi_definitions.yaml"))["summary"]["total_kpis"]
    assert len(ids) == summary_total, (
        f"YAML drift: {len(ids)} enumerated KPI entries vs summary.total_kpis={summary_total}"
    )
    faithful = os.environ.get("E2I_DB_INTEGRATION") == "1"
    # Make the serving mode explicit so a reader knows which mode the coverage
    # numbers reflect (the deployed synthetic-gold instance runs with the flag on).
    include_synthetic = kpi_include_synthetic()
    mode = "synthetic-inclusive" if include_synthetic else "strict-exclusion"
    print(
        f"MODE: {mode} "
        f"(kpi_include_synthetic()={include_synthetic}; "
        f"set E2I_INCLUDE_SYNTHETIC=1 to probe the synthetic substrate)"
    )
    mapped = empty = na = 0
    for kid in ids:
        if kid in DIRECT_PROBES:
            if not faithful:
                print(
                    f"{kid} MAPPED (direct ml_predictions column; E2I_DB_INTEGRATION=1 for counts)"
                )
                mapped += 1
                continue
            row = _run_direct(DIRECT_PROBES[kid])
            n = int(row or "0")
            if n > 0:
                print(f"{kid} MAPPED rows={n}")
                mapped += 1
            else:
                print(f"{kid} EMPTY rows=0")
                empty += 1
            continue
        if kid not in PROBES:
            print(f"{kid} N/A: no substrate probe defined")
            na += 1
            continue
        qid, params, key = PROBES[kid]
        served = resolved_probe_id(qid)
        if not faithful:
            print(f"{kid} MAPPED (probe={served}; run with E2I_DB_INTEGRATION=1 for counts)")
            mapped += 1
            continue
        row = _run_probe(served, params)
        if _value_present(row, key):
            print(f"{kid} MAPPED {row}")
            mapped += 1
        else:
            print(f"{kid} EMPTY {row}")
            empty += 1
    print("-" * 60)
    print(f"TOTAL {len(ids)}  MAPPED {mapped}  EMPTY {empty}  N/A {na}")
    # non-zero exit if any KPI is EMPTY or N/A under a faithful run
    return 0 if (not faithful or (empty == 0 and na == 0)) else 1


if __name__ == "__main__":
    sys.exit(main())
