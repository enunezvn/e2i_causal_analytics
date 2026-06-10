"""KPI coverage probe (Shard 09).

Maps every one of the 46 KPIs in config/kpi_definitions.yaml to the kpi_query
registry query_id (the *_include_synthetic / view variant) that exercises the
synthetic substrate, runs it against the faithful docker Supabase, and reports
non-NULL / EMPTY per KPI.

Run (faithful):  E2I_DB_INTEGRATION=1 python scripts/check_kpi_coverage.py
Run (offline):   python scripts/check_kpi_coverage.py   # prints the map only

A KPI is MAPPED when its probe returns a row whose primary metric value is
non-NULL after a synthetic load. Param values are chosen to exercise the
synthetic substrate (brand = Kisqali where a brand param is required).
"""

import json
import os
import subprocess
import sys

import yaml

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
    "WS1-MP-008": "SELECT count(*) AS n FROM ml_predictions WHERE is_synthetic AND fairness_metrics IS NOT NULL",
}

# KPI id -> (registry query_id, params_json, value_key_substring_or_None)
# value_key_substring_or_None: a substring that must appear as a non-null JSON key
# in the returned row (None = any returned row counts).
PROBES: dict[str, tuple[str, str, str | None]] = {
    # --- WS1 data quality ---
    "WS1-DQ-001": (
        "data_quality_source_coverage_patients_include_synthetic",
        '["Kisqali"]',
        "covered",
    ),
    "WS1-DQ-002": ("data_quality_source_coverage_hcps_include_synthetic", "[]", "covered"),
    "WS1-DQ-003": ("data_quality_cross_source_match", "[]", "match_rate"),
    "WS1-DQ-004": ("data_quality_stacking_lift", "[]", "lift_score"),
    "WS1-DQ-005": ("data_quality_completeness_pass_rate_include_synthetic", "[]", "pass_rate"),
    "WS1-DQ-006": (
        "data_quality_geographic_consistency_include_synthetic",
        '["Kisqali"]',
        "max_gap",
    ),
    "WS1-DQ-007": ("data_quality_data_lag", "[]", "median_lag_days"),
    "WS1-DQ-008": ("data_quality_label_quality", "[]", "iaa_group_id"),
    "WS1-DQ-009": ("data_quality_time_to_release", "[]", "avg_ttr_hours"),
    # --- WS1 model performance ---
    "WS1-MP-001": ("model_performance_roc_auc_include_synthetic", "[]", "roc_auc"),
    # WS1-MP-002/003/004/005/006/008 -> DIRECT_PROBES (no kpi_query registry entry).
    "WS1-MP-007": ("model_performance_shap_coverage_include_synthetic", "[]", "coverage"),
    "WS1-MP-009": ("model_performance_feature_drift", "[]", "avg_psi"),
    # --- WS2 triggers ---
    "WS2-TR-001": ("trigger_performance_precision_include_synthetic", "[]", "precision"),
    "WS2-TR-002": ("trigger_performance_recall_include_synthetic", "[]", "recall"),
    "WS2-TR-003": (
        "trigger_performance_action_rate_uplift_include_synthetic",
        "[]",
        "action_rate_uplift",
    ),
    "WS2-TR-004": (
        "trigger_performance_acceptance_rate_include_synthetic",
        "[]",
        "acceptance_rate",
    ),
    "WS2-TR-005": (
        "trigger_performance_false_alert_rate_include_synthetic",
        "[]",
        "false_alert_rate",
    ),
    "WS2-TR-006": ("trigger_performance_override_rate_include_synthetic", "[]", "override_rate"),
    "WS2-TR-007": ("trigger_performance_lead_time_include_synthetic", "[]", "median_lead_time"),
    "WS2-TR-008": ("trigger_performance_cfr_include_synthetic", "[]", "cfr"),
    # --- WS3 business ---
    "WS3-BI-001": ("business_impact_mau_fallback_include_synthetic", "[]", "mau"),
    "WS3-BI-002": ("business_impact_wau_fallback_include_synthetic", "[]", "wau"),
    "WS3-BI-003": ("business_impact_patient_touch_rate", '["Kisqali"]', "touch_rate"),
    "WS3-BI-004": ("business_impact_hcp_coverage_include_synthetic", "[]", "coverage"),
    "WS3-BI-005": ("business_impact_trx_include_synthetic", '["Kisqali"]', "trx"),
    "WS3-BI-006": ("business_impact_nrx_include_synthetic", '["Kisqali"]', "nrx"),
    "WS3-BI-007": ("business_impact_nbrx_include_synthetic", '["Kisqali"]', "nbrx"),
    "WS3-BI-008": ("business_impact_trx_share_include_synthetic", '["Kisqali"]', "share"),
    "WS3-BI-009": ("business_impact_conversion_rate_include_synthetic", "[]", "conversion_rate"),
    "WS3-BI-010": ("business_impact_roi_business_metrics_include_synthetic", "[]", "avg_roi"),
    # --- brand-specific ---
    "BR-001": (
        "brand_specific_remi_ah_uncontrolled_include_synthetic",
        '["3"]',
        "uncontrolled_rate",
    ),
    "BR-002": ("brand_specific_remi_intent_delta_fallback_include_synthetic", "[]", "intent_delta"),
    "BR-003": ("brand_specific_fabhalta_pnh_tested_include_synthetic", "[]", "tested_rate"),
    "BR-004": ("brand_specific_kisqali_dx_adoption_include_synthetic", "[]", "median_days"),
    "BR-005": ("brand_specific_kisqali_oncologist_reach_include_synthetic", "[]", "reach"),
    # --- causal metrics ---
    "CM-001": ("causal_metrics_ate_include_synthetic", "[]", "ate"),
    "CM-002": ("causal_metrics_cate_include_synthetic", '["high_severity"]', "cate"),
    "CM-003": ("causal_metrics_causal_impact_include_synthetic", '[""]', "effect"),
    "CM-004": (
        "causal_metrics_counterfactual_include_synthetic",
        '["churn"]',
        "mean_counterfactual",
    ),
    "CM-005": ("causal_metrics_mediation_include_synthetic", "[]", "proportion_mediated"),
}


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
    assert len(ids) == 46, f"expected 46 KPIs, found {len(ids)}"
    faithful = os.environ.get("E2I_DB_INTEGRATION") == "1"
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
        if not faithful:
            print(f"{kid} MAPPED (probe={qid}; run with E2I_DB_INTEGRATION=1 for counts)")
            mapped += 1
            continue
        row = _run_probe(qid, params)
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
