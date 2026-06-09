"""Faithful Shard-02 substrate gate (Gate B/C): real --small load to the docker
Supabase, then assert the four COUNT checks.

Gated by E2I_DB_INTEGRATION=1 and run with -n0 (no live-DB pollution in the
default unit run). The rows are is_synthetic-tagged and excluded by default from
real analyses (Shard 07).

NOTE: the loader's authorized write path uses the Supabase anon/authenticated
role, which on the local docker stack lacks INSERT grants (only service_role can
write; verified has_table_privilege). If the run environment supplies a
service-role key to the loader (SUPABASE_KEY/SUPABASE_ANON_KEY resolving to a
service-role token), this gate loads and asserts; otherwise the load fails closed
and the test is skipped with a clear reason rather than asserting on a half-load.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1", reason="faithful DB only"
)

REPO = Path(__file__).resolve().parents[2]


def _psql_count(sql: str) -> int:
    out = subprocess.check_output(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d",
         "postgres", "-tAc", sql],
        text=True,
    )
    return int(out.strip())


def test_shard02_substrate_lands_in_docker_supabase():
    # Column presence is write-free and always assertable.
    assert _psql_count(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_name='ml_predictions' AND column_name IN "
        "('treatment_effect_estimate','heterogeneous_effect','segment_assignment');"
    ) == 3, "ml_predictions causal columns must exist (Shard 01 DDL)"

    # Pre-check the write path (write-free): the loader's anon/authenticated role
    # must be able to INSERT, else the sanctioned load fails closed (42501) and we
    # skip rather than burn ~160s of per-batch retries to assert on a half-load.
    anon_can_write = _psql_count(
        "SELECT has_table_privilege('anon','business_metrics','INSERT')::int "
        "+ has_table_privilege('authenticated','business_metrics','INSERT')::int;"
    )
    if anon_can_write == 0:
        pytest.skip(
            "loader anon/authenticated role lacks INSERT grant on the docker "
            "stack (only service_role can write) -- supply a service-role key "
            "to the loader to run this faithful gate"
        )

    # Real --small load via the sanctioned loader script.
    proc = subprocess.run(
        [sys.executable, "scripts/load_synthetic_data.py", "--small"],
        cwd=REPO, capture_output=True, text=True, timeout=420,
        env={**os.environ, "LOKY_MAX_CPU_COUNT": "1"},
    )
    combined = proc.stdout + proc.stderr
    if "permission denied" in combined or "0 loaded" in combined:
        pytest.skip(
            "loader write path is not authorized in this environment "
            "(permission denied / 0 loaded)"
        )

    # business_metrics: gap-connector key present + ONLY the 5 lowercase keys.
    assert _psql_count(
        "SELECT count(*) FROM business_metrics "
        "WHERE metric_name='trx' AND is_synthetic=true;"
    ) > 0
    assert _psql_count(
        "SELECT count(DISTINCT metric_name) FROM business_metrics "
        "WHERE is_synthetic=true;"
    ) == 5
    # provenance stamp survived the loader.
    assert _psql_count(
        "SELECT count(*) FROM ml_predictions WHERE is_synthetic=true;"
    ) > 0
