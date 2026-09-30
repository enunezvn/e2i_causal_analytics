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
from datetime import datetime, timedelta, timezone
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
# Every forward migration that touches the table, in order (040-043 today). Globbed,
# not listed, so a later one — #2260's 157 changes the mirror's conflict key — is
# applied here as soon as it lands instead of leaving the rehearsal on a stale schema.
VERDICT_MIGRATIONS = tuple(
    p.name
    for p in sorted(MIGRATIONS.glob("[0-9]*.sql"))
    if "adaptive_validity_verdicts" in p.read_text()
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

    assert VERDICT_MIGRATIONS[0] == "040_adaptive_validity_verdicts.sql", VERDICT_MIGRATIONS

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


def test_a_sidecar_the_mirror_skips_makes_the_run_degraded_not_ok(
    verdicts_db, tmp_path, monkeypatch, caplog
) -> None:
    """codex r1 MED: the reader skips an unreadable sidecar with a WARNING and the script
    still exits 0. That canonical record never reaches the table, so the task must not
    report ``ok``: it reports ``degraded``, names the file, and logs at ERROR."""
    _write_sidecar(tmp_path, "exp_degraded_2273", "age")
    broken = tmp_path / "exp_degraded_2273" / "adaptive_verdicts_20260923T000000Z.json"
    broken.write_text("{ not json")
    _arm(monkeypatch, verdicts_db, tmp_path)

    with caplog.at_level("WARNING"):
        result = _task()()
    assert result["status"] == "degraded", result
    assert any(str(broken) in line for line in result["skipped_sidecars"]), result
    assert _count(verdicts_db, "exp_degraded_2273") == 1, "the readable sidecar still lands"
    assert any(r.levelname == "ERROR" and str(broken) in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": "1.0", "experiment_id": "exp_lost_2273", "written_at": "not-a-date"},
        {
            "schema_version": "1.0",
            "experiment_id": "exp_lost_2273",
            # Clock-relative: a fixed stamp falls behind the module DB's cursor once an
            # earlier test has imported rows, and the reader would never look at it.
            "written_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "adaptive_verdicts": {"feature": "age"},
        },
    ],
    ids=["unparseable_written_at", "non_list_verdicts"],
)
def test_every_reader_path_that_drops_a_sidecar_makes_the_run_degraded(
    payload, verdicts_db, tmp_path, monkeypatch
) -> None:
    """codex r2 MED: the reader also drops a whole sidecar on an unparseable
    ``written_at`` and empties it on a non-list ``adaptive_verdicts`` — both exit 0."""
    sub = tmp_path / "exp_lost_2273"
    sub.mkdir()
    lost = sub / "adaptive_verdicts_lost.json"
    lost.write_text(json.dumps(payload))
    _arm(monkeypatch, verdicts_db, tmp_path)

    result = _task()()
    assert result["status"] == "degraded", result
    assert any(str(lost) in line for line in result["skipped_sidecars"]), result


def test_the_subprocess_argv_carries_nothing_from_the_environment() -> None:
    """Semgrep ``dangerous-subprocess-use-tainted-env-args``: the argv the task hands
    ``subprocess.run`` must be exactly ``[sys.executable, str(_MIRROR_SCRIPT)]``. The
    artifacts dir reaches the script through the inherited environment (the script
    defaults ``--artifacts-dir`` to ``$ADAPTIVE_VALIDITY_ARTIFACTS_DIR``), never argv.
    The armed real-Postgres tests above prove the child still finds the sidecars."""
    import ast

    source = (REPO / "src" / "tasks" / "audit_sidecar_mirror_tasks.py").read_text()
    tree = ast.parse(source)
    runs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(runs) == 1, "expected exactly one subprocess.run in the task module"
    argv = runs[0].args[0]
    if isinstance(argv, ast.Name):  # resolve `cmd = [...]`
        assigns = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == argv.id for t in node.targets)
        ]
        assert len(assigns) == 1, f"`{argv.id}` must be assigned exactly once"
        argv = assigns[0]
    assert isinstance(argv, ast.List), ast.unparse(argv)
    assert [ast.unparse(e) for e in argv.elts] == ["sys.executable", "str(_MIRROR_SCRIPT)"]


# --------------------------------------------------------------------------
# #2278: a sidecar the cursor has passed is still mirrored (real Postgres)
# --------------------------------------------------------------------------
#
# The cursor is ``max(imported_at) - 1h`` compared against a sidecar's
# ``written_at``. A sidecar that is not mirrored on the run that first could see it
# (it landed late, or it was unreadable mid-write) is behind the cursor by the next
# run, and before #2278 no run ever read it again.


def _write_sidecar_at(root: Path, experiment_id: str, written_at: datetime, *features: str) -> Path:
    stamp = written_at.strftime("%Y%m%dT%H%M%SZ")
    sub = root / experiment_id
    sub.mkdir(parents=True, exist_ok=True)
    out = sub / f"adaptive_verdicts_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_id": experiment_id,
                "written_at": stamp,
                "adaptive_verdicts": [
                    {"feature": f, "layer": "3", "severity": "moderate"} for f in features
                ],
            }
        )
    )
    return out


def _yesterday() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=1)


def test_a_sidecar_that_lands_behind_the_cursor_is_still_mirrored_2278(
    verdicts_db, tmp_path, monkeypatch
) -> None:
    """Run 1 imports a fresh sidecar, so the cursor is ~now - 1h. A sidecar written
    yesterday then lands (a late file). Run 2 must mirror it."""
    _arm(monkeypatch, verdicts_db, tmp_path)
    _write_sidecar(tmp_path, "exp_fresh_2278", "age")
    assert _task()()["status"] == "ok"
    assert _count(verdicts_db, "exp_fresh_2278") == 1

    _write_sidecar_at(tmp_path, "exp_late_2278", _yesterday(), "age", "gender")
    second = _task()()
    assert second["status"] == "ok", second
    assert _count(verdicts_db, "exp_late_2278") == 2, (
        "a sidecar written before the cursor but never mirrored must be mirrored"
    )


def test_the_backfill_runs_on_psycopg2_2278(verdicts_db, tmp_path, monkeypatch) -> None:
    """Prod's image has psycopg2 only, and the backfill's key lookup binds a Python
    list of aware datetimes to ``= ANY(%s)``. Run that path under psycopg2."""
    _arm(monkeypatch, verdicts_db, tmp_path)
    _write_sidecar(tmp_path, "exp_pg2fresh_2278", "age")
    assert _task()()["status"] == "ok"
    _write_sidecar_at(tmp_path, "exp_pg2late_2278", _yesterday(), "age")
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
    assert "backfilling 1 row(s)" in proc.stderr, proc.stderr[-2000:]
    assert _count(verdicts_db, "exp_pg2late_2278") == 1


def test_a_sidecar_unreadable_on_one_run_is_mirrored_once_readable_2278(
    verdicts_db, tmp_path, monkeypatch
) -> None:
    """The file is caught mid-write (unreadable, so the run is degraded). Once it is
    whole, a later run must mirror it although imports in between moved the cursor."""
    _arm(monkeypatch, verdicts_db, tmp_path)
    path = _write_sidecar_at(tmp_path, "exp_partial_2278", _yesterday(), "age")
    whole = path.read_text()
    path.write_text(whole[: len(whole) // 2])
    _write_sidecar(tmp_path, "exp_other_2278", "age")

    first = _task()()
    assert first["status"] == "degraded", first
    assert _count(verdicts_db, "exp_other_2278") == 1

    path.write_text(whole)
    second = _task()()
    assert second["status"] == "ok", second
    assert _count(verdicts_db, "exp_partial_2278") == 1


def test_an_already_mirrored_sidecar_behind_the_cursor_is_not_re_upserted_2278(
    verdicts_db, tmp_path, monkeypatch
) -> None:
    """The cursor still bounds RE-upserts: a sidecar behind it whose rows are already
    in the table does not go through the upsert again (the write amplification the
    cursor exists to prevent). Only rows missing from the table are backfilled."""
    _arm(monkeypatch, verdicts_db, tmp_path)
    path = _write_sidecar_at(tmp_path, "exp_settled_2278", _yesterday(), "age")
    _write_sidecar(tmp_path, "exp_fresh2_2278", "age")
    assert _task()()["status"] == "ok"
    assert _count(verdicts_db, "exp_settled_2278") == 1

    payload = json.loads(path.read_text())
    payload["adaptive_verdicts"][0]["severity"] = "high"
    path.write_text(json.dumps(payload))
    assert _task()()["status"] == "ok"
    severity = verdicts_db.rows(
        "SELECT verdict->>'severity' FROM adaptive_validity_verdicts "
        "WHERE experiment_id = 'exp_settled_2278'"
    )
    assert severity == ["moderate"], severity


def test_a_non_dict_verdict_entry_makes_the_run_degraded_2278(
    verdicts_db, tmp_path, monkeypatch
) -> None:
    """The reader drops a non-dict entry of ``adaptive_verdicts``; that verdict never
    reaches the table, so the run is degraded and names the file, like the other
    drop paths. The dict entries of the same sidecar still land."""
    _arm(monkeypatch, verdicts_db, tmp_path)
    path = _write_sidecar(tmp_path, "exp_nondict_2278", "age")
    payload = json.loads(path.read_text())
    payload["adaptive_verdicts"].append("gender")
    path.write_text(json.dumps(payload))

    result = _task()()
    assert result["status"] == "degraded", result
    assert any(str(path) in line for line in result["skipped_sidecars"]), result
    assert _count(verdicts_db, "exp_nondict_2278") == 1
