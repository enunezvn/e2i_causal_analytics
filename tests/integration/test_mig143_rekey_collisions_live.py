"""Migration 143's kpi_history re-key is collision-safe (codex r1 HIGH).

Runs the GENERATED re-key block against a TEMP copy of public.kpi_history inside
BEGIN ... ROLLBACK on the live Postgres, so nothing persists. Skips without docker.
"""

import shutil
import subprocess

import pytest

from scripts import gen_canonical_volume_registry as gen

pytestmark = pytest.mark.integration

AUDIT = "pg_temp.kh_rekey"
SETUP = (
    "BEGIN;\nCREATE TEMP TABLE kh (LIKE public.kpi_history INCLUDING ALL) ON COMMIT DROP;\n"
    + gen.audit_table_sql(AUDIT)
    + "\n"
)
REKEY = gen.rekey_sql("kh", audit=AUDIT) + "\n"
RESTORE = gen.restore_sql("kh", audit=AUDIT) + "\n"
COUNT = "SELECT kpi_id || '=' || count(*) FROM kh GROUP BY kpi_id ORDER BY kpi_id;\n"
#: every row's identity AND payload, so a round trip is compared exactly
STATE = (
    "SELECT kpi_id || '|' || metric_date || '|' || value || '|' || coalesce(status, '-') || '|' "
    "|| source || '|' || is_synthetic || '|' || id FROM kh ORDER BY kpi_id, metric_date, id;\n"
)
MARK = "SELECT '---';\n"


def _psql(script: str) -> subprocess.CompletedProcess:
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    if (
        subprocess.run(["docker", "exec", "supabase-db", "true"], capture_output=True).returncode
        != 0
    ):
        pytest.skip("supabase-db container not reachable")
    return subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "supabase-db",
            "psql",
            "-U",
            "postgres",
            "-d",
            "postgres",
            "-v",
            "ON_ERROR_STOP=1",
            "-qtA",
        ],
        input=script,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _row(kpi_id, month, value, source="treatment_events.event_date", status="informational"):
    return (
        "INSERT INTO kh (kpi_id, brand, region, metric_date, value, status, source, is_synthetic) "
        f"VALUES ('{kpi_id}', '', '', '{month}', {value}, '{status}', '{source}', true);\n"
    )


def test_empty_destination_moves_every_event_row_and_records_it():
    out = _psql(
        SETUP
        + _row("WS3-BI-005", "2026-07-01", 10)
        + _row("WS3-BI-005", "2026-08-01", 12)
        + REKEY
        + COUNT
        + f"SELECT disposition || '=' || count(*) FROM {AUDIT} GROUP BY disposition;\n"
        + "ROLLBACK;\n"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["WS3-BI-011=2", "moved=2"]


def test_identical_collision_is_absorbed():
    out = _psql(
        SETUP
        + _row("WS3-BI-005", "2026-07-01", 10)
        + _row("WS3-BI-011", "2026-07-01", 10)
        + REKEY
        + COUNT
        + f"SELECT disposition || '=' || count(*) FROM {AUDIT} GROUP BY disposition;\n"
        + "ROLLBACK;\n"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["WS3-BI-011=1", "absorbed=1"]


def test_conflicting_collision_aborts_with_a_clear_error():
    out = _psql(
        SETUP
        + _row("WS3-BI-005", "2026-07-01", 10)
        + _row("WS3-BI-011", "2026-07-01", 99)
        + REKEY
        + COUNT
        + "ROLLBACK;\n"
    )
    assert out.returncode != 0
    assert "re-key refused" in out.stderr and "WS3-BI-011" in out.stderr


def test_a_status_only_difference_is_a_conflict_and_aborts():
    """codex r2: payload equivalence includes status (079_kpi_history.sql:23)."""
    out = _psql(
        SETUP
        + _row("WS3-BI-005", "2026-07-01", 10)
        + _row("WS3-BI-011", "2026-07-01", 10, status="critical")
        + REKEY
        + COUNT
        + "ROLLBACK;\n"
    )
    assert out.returncode != 0
    assert "re-key refused" in out.stderr and "WS3-BI-011" in out.stderr


def test_rerun_is_a_no_op_and_canonical_rows_are_untouched():
    out = _psql(
        SETUP
        + _row("WS3-BI-005", "2026-07-01", 10)
        + _row("WS3-BI-005", "2026-06-01", 800000, source="business_metrics.value")
        + REKEY
        + REKEY
        + COUNT
        + "ROLLBACK;\n"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["WS3-BI-005=1", "WS3-BI-011=1"]


def test_forward_then_rollback_restores_the_exact_pre_migration_rows():
    """Identical pre-existing destination + destination-only rows: every id and payload
    after forward+rollback equals the state before the migration."""
    rows = (
        _row("WS3-BI-005", "2026-06-01", 10)
        + _row("WS3-BI-005", "2026-07-01", 11)
        + _row("WS3-BI-011", "2026-07-01", 11)  # identical pre-existing destination (absorbed)
        + _row("WS3-BI-011", "2025-01-01", 5)  # destination-only
        + _row("WS3-BI-008", "2026-07-01", 0.4)
    )
    out = _psql(
        SETUP
        + rows
        + STATE
        + MARK
        + REKEY
        + RESTORE
        + STATE
        + f"SELECT coalesce(to_regclass('{AUDIT}')::text, 'dropped');\n"
        + "ROLLBACK;\n"
    )
    assert out.returncode == 0, out.stderr
    before, after = out.stdout.split("---")
    assert len(before.split()) == 5
    assert after.split() == before.split() + ["dropped"]


def test_rollback_never_deletes_destination_rows_the_migration_did_not_create():
    out = _psql(
        SETUP
        + _row("WS3-BI-005", "2026-06-01", 10)
        + _row("WS3-BI-005", "2026-07-01", 11)
        + _row("WS3-BI-011", "2026-07-01", 11)  # pre-existing, absorbed
        + REKEY
        + _row("WS3-BI-011", "2026-08-01", 12)  # written after the deploy
        + "UPDATE kh SET value = 13 WHERE kpi_id = 'WS3-BI-011' AND metric_date = '2026-06-01';\n"
        + RESTORE
        + COUNT
        + "SELECT kpi_id || '@' || metric_date || '=' || value FROM kh ORDER BY kpi_id, metric_date;\n"
        + "ROLLBACK;\n"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == [
        "WS3-BI-005=2",
        "WS3-BI-011=2",
        "WS3-BI-005@2026-06-01=10",
        "WS3-BI-005@2026-07-01=11",
        "WS3-BI-011@2026-07-01=11",
        "WS3-BI-011@2026-08-01=12",
    ]


def test_restore_refuses_without_provenance_or_over_a_differing_row():
    no_audit = _psql(
        SETUP
        + _row("WS3-BI-005", "2026-06-01", 10)
        + REKEY
        + f"DROP TABLE {AUDIT};\n"
        + RESTORE
        + "ROLLBACK;\n"
    )
    assert no_audit.returncode != 0 and "restore refused" in no_audit.stderr
    differing = _psql(
        SETUP
        + _row("WS3-BI-005", "2026-06-01", 10)
        + REKEY
        + _row("WS3-BI-005", "2026-06-01", 77)
        + RESTORE
        + "ROLLBACK;\n"
    )
    assert differing.returncode != 0 and "restore refused" in differing.stderr
