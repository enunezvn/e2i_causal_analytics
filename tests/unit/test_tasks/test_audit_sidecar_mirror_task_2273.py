"""#2273: the #238 sidecar -> adaptive_validity_verdicts mirror runs NIGHTLY, gated dark.

#238 specified "a nightly batch job that mirrors the sidecar payload" and
``scripts/mirror_audit_sidecar_to_supabase.py`` implemented the batch, but nothing
ever scheduled it (no beat entry, no cron), so the table's 0 rows on prod were
structural. The repo's convention for an in-app nightly data job is a celery beat
crontab entry on the ``analytics`` queue (the ETL rollups and corpus syncs), not
the host crontab (owner ops: backup, reseed), so this is a beat entry.

It ships DARK behind ``AUDIT_SIDECAR_MIRROR_ENABLED``: #2260 changes the mirror's
ON CONFLICT key together with migration 157, and the mirror must not write prod
before 157 is applied. The NPPES refresh is the precedent for a schedule that is
always wired but does real work only once an operator arms it.

Two facts the task depends on were measured on the prod worker (2026-09-23): the
image has ``psycopg2`` but NOT ``psycopg`` (v3), which the script imported, so the
script could not run in any app container; and ``SUPABASE_DB_URL`` (not
``DATABASE_URL``) is what x-common-env gives the workers.

The real-Postgres tests are opt-in (``E2I_DB_INTEGRATION=1`` + docker): a throwaway
container of prod's own Postgres image with migrations 040-043 applied verbatim, and
the REAL script run as the task's subprocess. Nothing here touches ``supabase-db``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml
from celery.schedules import crontab

import src.tasks  # noqa: F401 — registers every src.tasks.* task
from src.workers.celery_app import celery_app

pytestmark = pytest.mark.timeout(300)

REPO = Path(__file__).resolve().parents[3]
BASE_COMPOSE = REPO / "docker" / "docker-compose.yml"
MIRROR_SCRIPT = REPO / "scripts" / "mirror_audit_sidecar_to_supabase.py"
MIGRATIONS = REPO / "database" / "migrations"
VERDICT_MIGRATIONS = (
    "040_adaptive_validity_verdicts.sql",
    "041_role_attributions.sql",
    "042_audit_evaluator_shadow_columns.sql",
    "043_audit_evaluator_soft_gate_columns.sql",
)

BEAT_ENTRY = "audit-sidecar-mirror-nightly"
TASK_NAME = "src.tasks.mirror_audit_sidecars"
FLAG = "AUDIT_SIDECAR_MIRROR_ENABLED"


def _task():
    from src.tasks.audit_sidecar_mirror_tasks import mirror_audit_sidecars

    return mirror_audit_sidecars


def _write_sidecar(root: Path, experiment_id: str, *features: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sub = root / experiment_id
    sub.mkdir(parents=True, exist_ok=True)
    out = sub / f"adaptive_verdicts_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_id": experiment_id,
                "data_source": "csu",
                "written_at": stamp,
                "leakage_severity": "none",
                "leaked_features": [],
                "adaptive_flagged_features": list(features),
                "adaptive_verdicts": [
                    {"feature": f, "layer": "3", "severity": "moderate", "z_score": 4.2}
                    for f in features
                ],
            }
        )
    )
    return out


# --------------------------------------------------------------------------
# Wiring (always run)
# --------------------------------------------------------------------------


def test_the_mirror_is_a_nightly_beat_entry_on_the_analytics_queue() -> None:
    celery_app.finalize()
    entry = celery_app.conf.beat_schedule.get(BEAT_ENTRY)
    assert entry is not None, f"no `{BEAT_ENTRY}` beat entry: the #238 mirror is unscheduled"
    assert entry["task"] == TASK_NAME
    assert TASK_NAME in celery_app.tasks, f"{TASK_NAME} is not a registered task"
    sched = entry["schedule"]
    assert isinstance(sched, crontab), "a daily entry must be a wall-clock crontab (#1645)"
    assert (sched.hour, sched.minute) == ({0}, {15})
    assert entry.get("options", {}).get("queue") == "analytics"


def test_the_queue_is_consumed_by_a_worker_that_mounts_the_sidecar_volume() -> None:
    """The task reads the canonical sidecars, so whichever worker consumes its queue
    must mount ``audit_artifacts``; a worker without the mount would mirror nothing."""
    compose = yaml.safe_load(BASE_COMPOSE.read_text())
    consumers = []
    for name, svc in (compose.get("services") or {}).items():
        command = (
            " ".join(svc.get("command") or [])
            if isinstance(svc.get("command"), list)
            else str(svc.get("command") or "")
        )
        queues = next(
            (tok.split("=", 1)[1] for tok in command.split() if tok.startswith("--queues=")), ""
        )
        if "analytics" in queues.split(","):
            consumers.append(name)
    assert consumers, "no compose service consumes the analytics queue"
    for name in consumers:
        mounts = compose["services"][name].get("volumes") or []
        assert "audit_artifacts:/app/data/audit_artifacts" in mounts, (
            f"{name} consumes `analytics` but does not mount audit_artifacts read-write"
        )


def test_the_enable_flag_is_forwarded_dark() -> None:
    """Forwarded on x-common-env so the operator CAN arm it from the host .env, with an
    empty default that the task reads as off."""
    compose = yaml.safe_load(BASE_COMPOSE.read_text())
    assert (compose.get("x-common-env") or {}).get(FLAG) == "${" + FLAG + ":-}"


# --------------------------------------------------------------------------
# Dark by default (always run)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "false", "0", "no"])
def test_disabled_unless_armed(value, tmp_path, monkeypatch) -> None:
    _write_sidecar(tmp_path, "exp_dark", "age")
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    # A DB that refuses connections: if the task ran the mirror it would fail loudly.
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://nobody@127.0.0.1:1/none")
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    assert _task()()["status"] == "disabled"


def test_armed_with_an_unreachable_db_fails_loudly(tmp_path, monkeypatch) -> None:
    from src.tasks.audit_sidecar_mirror_tasks import AuditSidecarMirrorError

    _write_sidecar(tmp_path, "exp_fail", "age")
    monkeypatch.setenv(FLAG, "true")
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://nobody@127.0.0.1:1/none")
    with pytest.raises(AuditSidecarMirrorError):
        _task()()


# --------------------------------------------------------------------------
# Real Postgres (opt-in)
# --------------------------------------------------------------------------

_OPT_IN = "E2I_DB_INTEGRATION"


@pytest.fixture(scope="module")
def verdicts_db() -> Iterator[Any]:
    if os.environ.get(_OPT_IN) != "1":
        pytest.skip(
            f"real-DB rehearsal: opt-in with {_OPT_IN}=1 (docker, prod's Postgres image); "
            "a skipped real-DB test is not coverage"
        )
    if shutil.which("docker") is None:
        pytest.skip("docker is not reachable")
    proc = subprocess.run(
        ["docker", "inspect", "supabase-db", "--format", "{{.Config.Image}}"],
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        pytest.skip("the supabase-db container is not reachable")
    from tests.unit.test_database.learning_loop._pg import PgConn, ThrowawayPg

    pg = ThrowawayPg(image=proc.stdout.decode().strip())
    pg.start()
    try:
        pg.rows("postgres", "CREATE DATABASE t2273 OWNER postgres")
        for name in VERDICT_MIGRATIONS:
            done = pg.run_script("t2273", (MIGRATIONS / name).read_bytes(), user="postgres")
            assert done.returncode == 0, f"{name}: {done.stderr.decode()}"
        yield PgConn(pg, "t2273")
    finally:
        pg.stop()


def _arm(monkeypatch, pg_conn, artifacts: Path) -> None:
    monkeypatch.setenv(FLAG, "true")
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("SUPABASE_DB_URL", pg_conn.pg.dsn(pg_conn.db))
    monkeypatch.setenv("PGPASSWORD", pg_conn.pg.client_env()["PGPASSWORD"])


def _count(pg_conn, experiment_id: str) -> int:
    return int(
        pg_conn.rows(
            f"SELECT count(*) FROM adaptive_validity_verdicts WHERE experiment_id = '{experiment_id}'"
        )[0]
    )


def test_armed_task_mirrors_the_real_sidecars_idempotently(
    verdicts_db, tmp_path, monkeypatch
) -> None:
    _write_sidecar(tmp_path, "exp_real_2273", "age", "gender")
    _arm(monkeypatch, verdicts_db, tmp_path)

    first = _task()()
    assert first["status"] == "ok", first
    assert _count(verdicts_db, "exp_real_2273") == 2
    second = _task()()
    assert second["status"] == "ok", second
    assert _count(verdicts_db, "exp_real_2273") == 2, "the nightly re-run must be idempotent"


def test_the_script_runs_on_psycopg2_when_psycopg3_is_absent(verdicts_db, tmp_path) -> None:
    """The app image ships psycopg2 only. ``sys.modules["psycopg"] = None`` makes
    ``import psycopg`` raise ImportError exactly as it does in the container."""
    _write_sidecar(tmp_path, "exp_pg2_2273", "age")
    shim = (
        "import sys, runpy; sys.modules['psycopg'] = None; "
        f"sys.argv = [{str(MIRROR_SCRIPT)!r}, '--artifacts-dir', {str(tmp_path)!r}]; "
        f"runpy.run_path({str(MIRROR_SCRIPT)!r}, run_name='__main__')"
    )
    env = {
        **verdicts_db.pg.client_env(),
        "DATABASE_URL": verdicts_db.pg.dsn(verdicts_db.db),
        "PYTHONPATH": str(REPO),
    }
    proc = subprocess.run(
        [sys.executable, "-c", shim], cwd=str(REPO), env=env, capture_output=True, text=True,
        timeout=240,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert _count(verdicts_db, "exp_pg2_2273") == 1
