import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful docker-Supabase test; set E2I_DB_INTEGRATION=1",
)

# KPI views that read a taggable table and must default-exclude synthetic rows
# (codex Shard-01 HIGH: view-backed KPIs bypass the migration-066 FROM/JOIN rewrite).
PATCHED_VIEWS = [
    "v_patient_eligibility",
    "v_kpi_active_users",
    "v_kpi_intent_to_prescribe",
    "v_kpi_data_lag",
    "v_kpi_cross_source_match",
    "v_kpi_stacking_lift",
    "v_kpi_time_to_release",
    "v_kpi_change_fail_rate",
]

# view-backed tables M1 missed; needed so Shard 09's synthetic loads are excludable.
NEW_TAGGED_TABLES = ["data_source_tracking", "etl_pipeline_metrics", "ml_annotations"]


def _psql(sql: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tAc", sql],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_every_kpi_view_default_excludes_synthetic():
    for v in PATCHED_VIEWS:
        defn = _psql(
            f"SELECT definition FROM pg_views WHERE schemaname='public' AND viewname='{v}';"
        )
        assert defn, f"{v} missing"
        assert "is_synthetic = false" in defn, f"{v} does not default-exclude synthetic"


def test_view_backed_tables_now_tagged():
    for t in NEW_TAGGED_TABLES:
        dt = _psql(
            "SELECT data_type FROM information_schema.columns WHERE table_schema='public' "
            f"AND table_name='{t}' AND column_name='is_synthetic';"
        )
        assert dt == "boolean", f"{t}.is_synthetic missing/wrong ({dt})"


def test_no_leaky_kpi_statement_reads_an_unfiltered_taggable_view():
    # Every kpi_query statement that reads a v_kpi_*/v_patient_eligibility view now
    # gets exclusion transitively from the view definition. Assert no registry
    # statement reads one of these views whose definition lacks the filter.
    bad = _psql(
        "SELECT string_agg(r.query_id, ',') FROM kpi_query_registry r "
        "JOIN pg_views v ON r.sql ILIKE '%' || v.viewname || '%' "
        "WHERE v.schemaname='public' AND v.viewname = ANY(ARRAY["
        + ",".join(f"'{v}'" for v in PATCHED_VIEWS)
        + "]) AND v.definition NOT ILIKE '%is_synthetic = false%';"
    )
    assert not bad, f"leaky view-backed KPI statements: {bad}"
