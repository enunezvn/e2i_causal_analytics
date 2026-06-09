import json
import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful docker-Supabase test; set E2I_DB_INTEGRATION=1",
)

# Every taggable-base-table query_id must default-exclude synthetic rows.
EXCLUDE_IDS = [
    "causal_metrics_ate", "causal_metrics_cate", "business_impact_hcp_coverage",
    "business_impact_trx", "business_impact_nrx", "business_impact_nbrx",
    "business_impact_trx_share", "business_impact_conversion_rate",
    "business_impact_roi_business_metrics", "business_impact_roi_agent_activities",
    "brand_specific_remi_intent_delta_fallback", "brand_specific_kisqali_dx_adoption",
    "brand_specific_kisqali_oncologist_reach", "trigger_performance_precision",
    "trigger_performance_recall", "trigger_performance_acceptance_rate",
    "trigger_performance_false_alert_rate", "trigger_performance_override_rate",
    "trigger_performance_lead_time", "trigger_performance_cfr",
    "trigger_performance_action_rate_uplift", "data_quality_completeness_pass_rate",
    "model_performance_shap_coverage", "business_impact_mau_fallback",
    "business_impact_wau_fallback", "causal_metrics_causal_impact",
    "business_impact_hcp_reach",
]


def _sql_for(query_id: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres",
         "-tAc", f"SELECT sql FROM kpi_query_registry WHERE query_id='{query_id}';"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_every_taggable_statement_default_excludes_synthetic():
    for qid in EXCLUDE_IDS:
        sql = _sql_for(qid)
        assert sql, f"{qid} not registered"
        assert "is_synthetic = false" in sql, f"{qid} missing default-exclude"


def test_opt_in_include_synthetic_family_exists():
    probe = "business_impact_trx_include_synthetic"
    sql = _sql_for(probe)
    assert sql, f"{probe} not registered (opt-in path missing)"
    assert "is_synthetic = false" not in sql, f"{probe} must NOT exclude synthetic"
    assert "treatment_events" in sql


def test_real_kpi_value_unchanged_by_synthetic_rows():
    subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres",
         "-c",
         "INSERT INTO treatment_events (treatment_event_id, patient_id, event_date, "
         "event_type, brand, is_synthetic) VALUES "
         "('SYN_TEST_M4', 'PSYNM4', CURRENT_DATE, 'prescription', 'Kisqali', true);"],
        check=True, capture_output=True, text=True,
    )
    try:
        default = subprocess.run(
            ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres",
             "-tAc", "SELECT * FROM kpi_query('business_impact_trx', '[\"Kisqali\"]'::jsonb);"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        incl = subprocess.run(
            ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres",
             "-tAc",
             "SELECT * FROM kpi_query('business_impact_trx_include_synthetic', "
             "'[\"Kisqali\"]'::jsonb);"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        default_trx = int(json.loads(default)["trx"])
        incl_trx = int(json.loads(incl)["trx"])
        assert incl_trx == default_trx + 1, (default_trx, incl_trx)
    finally:
        subprocess.run(
            ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres",
             "-c", "DELETE FROM treatment_events WHERE treatment_event_id='SYN_TEST_M4';"],
            check=True, capture_output=True, text=True,
        )
