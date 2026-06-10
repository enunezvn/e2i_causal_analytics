import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful docker-Supabase test; set E2I_DB_INTEGRATION=1",
)


def _enum_labels(type_name: str) -> set[str]:
    out = subprocess.run(
        [
            "docker",
            "exec",
            "supabase-db",
            "psql",
            "-U",
            "postgres",
            "-d",
            "postgres",
            "-tAc",
            "SELECT e.enumlabel FROM pg_enum e JOIN pg_type t ON t.oid=e.enumtypid "
            f"WHERE t.typname='{type_name}';",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


def test_prediction_event_types_added():
    labels = _enum_labels("memory_event_type")
    assert "prediction_completed" in labels
    assert "prediction_delivered" in labels
