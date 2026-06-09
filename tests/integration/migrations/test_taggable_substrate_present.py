import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful docker-Supabase test; set E2I_DB_INTEGRATION=1",
)

REQUIRED = [
    "ml_experiments", "ml_model_registry", "ml_training_runs", "ml_deployments",
    "ab_experiment_assignments", "ab_experiment_enrollments", "ab_experiment_results",
    "agent_activities", "causal_paths", "episodic_memories",
    # 5 view-backed KPI tables
    "user_sessions", "hcp_intent_surveys", "data_source_tracking",
    "etl_pipeline_metrics", "ml_annotations",
]


def test_all_required_substrate_tables_present():
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres",
         "-tAc",
         "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename = ANY("
         "ARRAY[" + ",".join(f"'{t}'" for t in REQUIRED) + "]);"],
        capture_output=True, text=True, check=True,
    )
    present = set(out.stdout.split())
    missing = [t for t in REQUIRED if t not in present]
    assert not missing, f"substrate drift -- missing tables: {missing}"
