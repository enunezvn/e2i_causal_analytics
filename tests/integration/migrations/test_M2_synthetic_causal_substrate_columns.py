import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful docker-Supabase test; set E2I_DB_INTEGRATION=1",
)

# Canonical per-unit causal columns -> expected pg data_type.
EXPECTED = {
    "treatment_arm": "smallint",
    "propensity_score": "double precision",
    "segment_assignment": "text",
    "discontinued_180d": "smallint",
    "persistent_180d": "smallint",
}


def _psql(sql: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres",
         "-d", "postgres", "-tAc", sql],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_patient_journeys_has_all_five_causal_substrate_columns():
    for col, expected_type in EXPECTED.items():
        data_type = _psql(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='patient_journeys' "
            f"AND column_name='{col}';"
        )
        assert data_type, f"patient_journeys.{col} missing"
        assert data_type == expected_type, (
            f"patient_journeys.{col} type={data_type}, want {expected_type}"
        )
