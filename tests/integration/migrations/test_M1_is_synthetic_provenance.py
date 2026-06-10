import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful docker-Supabase test; set E2I_DB_INTEGRATION=1",
)

TAGGABLE = [
    "triggers",
    "business_metrics",
    "ml_predictions",
    "agent_activities",
    "causal_paths",
    "patient_journeys",
    "treatment_events",
    "hcp_profiles",
    "user_sessions",
    "hcp_intent_surveys",
    "episodic_memories",
    "ab_experiment_assignments",
]


def _psql(sql: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tAc", sql],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_every_taggable_table_has_is_synthetic_boolean_default_false():
    for tbl in TAGGABLE:
        row = _psql(
            "SELECT data_type, is_nullable, column_default "
            "FROM information_schema.columns "
            f"WHERE table_schema='public' AND table_name='{tbl}' "
            "AND column_name='is_synthetic';"
        )
        assert row, f"{tbl}.is_synthetic missing"
        data_type, is_nullable, default = row.split("|")
        assert data_type == "boolean", f"{tbl}.is_synthetic type={data_type}"
        assert is_nullable == "NO", f"{tbl}.is_synthetic nullable"
        assert "false" in default, f"{tbl}.is_synthetic default={default}"


def test_high_fanout_tables_have_is_synthetic_index():
    for tbl in [
        "treatment_events",
        "triggers",
        "ml_predictions",
        "business_metrics",
        "patient_journeys",
        "episodic_memories",
    ]:
        idx = _psql(
            "SELECT indexname FROM pg_indexes WHERE schemaname='public' "
            f"AND tablename='{tbl}' AND indexname='idx_{tbl}_is_synthetic';"
        )
        assert idx == f"idx_{tbl}_is_synthetic", f"{tbl} missing partial index"
