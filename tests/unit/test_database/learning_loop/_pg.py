"""Throwaway Postgres for the learning-loop real-DB tests, and read-only access to prod.

The live ``supabase-db`` is never written: every write goes to a throwaway container of prod's
own image. It is memory-capped, bound to 127.0.0.1, uniquely named and removed at teardown. Prod
contributes a schema-only dump of ``public`` (no data rows) and read-only SELECTs for the
equivalence checks. The rows the lane needs are rebuilt from the repository.
"""

from __future__ import annotations

import atexit
import os
import re
import secrets
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
PROD_CONTAINER = "supabase-db"
CONTAINER_PREFIX = "e2i-learnloop-pg-"
MEMORY_CAP = "1g"

# Restore errors we expect, by exact message, with the reason. Exact on purpose: an error that is
# not listed here fails the sanity test, and a listed error that never occurs fails it too.
EXPECTED_RESTORE_ERRORS: Dict[str, str] = {
    'ERROR:  schema "public" already exists': (
        "the dump's CREATE SCHEMA public; the template copy of the image's postgres database "
        "already has the schema"
    ),
}

# The base schema is prod's as of the lane's base commit; these files are applied by the tests
# themselves, so they are not in the rebuilt ledger.
LANE_MIGRATIONS = (
    "ml/039_tool_category_cohort.sql",
    "ml/040_tool_registry_startup_sync.sql",
    "ml/041_composer_learning_loop_recording.sql",
)


def _run(cmd: Sequence[str], *, input_bytes: Optional[bytes] = None, env=None, timeout=600):
    return subprocess.run(
        list(cmd), input=input_bytes, capture_output=True, timeout=timeout, env=env
    )


def _rows(stdout: bytes) -> List[str]:
    return [line for line in stdout.decode().splitlines() if line != ""]


# ---------------------------------------------------------------------------
# Prod: read-only
# ---------------------------------------------------------------------------


class ProdReadOnly:
    """SELECTs against the live database inside a read-only session."""

    def rows(self, sql: str) -> List[str]:
        proc = _run(
            [
                "docker", "exec", "-e", "PGOPTIONS=-c default_transaction_read_only=on",
                PROD_CONTAINER, "psql", "-U", "postgres", "-d", "postgres", "-X", "-tA",
                "-v", "ON_ERROR_STOP=1", "-c", sql,
            ]
        )  # fmt: skip
        if proc.returncode != 0:
            raise RuntimeError(f"prod read failed: {proc.stderr.decode()}")
        return _rows(proc.stdout)

    def schema_dump(self) -> bytes:
        proc = _run(
            ["docker", "exec", PROD_CONTAINER, "pg_dump", "-U", "postgres", "--schema-only",
             "-n", "public", "postgres"]
        )  # fmt: skip
        if proc.returncode != 0:
            raise RuntimeError(f"schema dump failed: {proc.stderr.decode()}")
        return proc.stdout

    def image(self) -> str:
        proc = _run(["docker", "inspect", PROD_CONTAINER, "--format", "{{.Config.Image}}"])
        return proc.stdout.decode().strip()


# ---------------------------------------------------------------------------
# Throwaway container
# ---------------------------------------------------------------------------


@dataclass
class ThrowawayPg:
    image: str
    name: str = field(default_factory=lambda: CONTAINER_PREFIX + secrets.token_hex(4))
    password: str = field(default_factory=lambda: secrets.token_hex(16), repr=False)
    host_port: Optional[int] = None

    def start(self, ready_timeout_s: int = 180) -> None:
        proc = _run(
            [
                "docker", "run", "-d", "--name", self.name,
                "--memory", MEMORY_CAP, "--memory-swap", MEMORY_CAP,
                "-e", f"POSTGRES_PASSWORD={self.password}",
                "-p", "127.0.0.1::5432",
                self.image,
                "postgres", "-c", "listen_addresses=*", "-c", "shared_buffers=128MB", "-c", "max_connections=60",
            ]
        )  # fmt: skip
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {proc.stderr.decode()}")
        atexit.register(self.stop)
        deadline = time.monotonic() + ready_timeout_s
        while time.monotonic() < deadline:
            ready = _run(
                ["docker", "exec", self.name, "pg_isready", "-U", "postgres", "-h", "127.0.0.1"]
            )
            # The image restarts postgres once after its init scripts; require a real query.
            if ready.returncode == 0 and self.try_rows("postgres", "select 1") == ["1"]:
                break
            time.sleep(2)
        else:
            self.stop()
            raise RuntimeError(f"{self.name} not ready after {ready_timeout_s}s")
        port = _run(["docker", "port", self.name, "5432/tcp"]).stdout.decode().strip()
        self.host_port = int(port.rsplit(":", 1)[1])

    def stop(self) -> None:
        _run(["docker", "rm", "-f", self.name])

    def _psql_cmd(self, db: str, *args: str, user: str = "supabase_admin") -> List[str]:
        # Fixture plumbing runs as supabase_admin (the image's superuser). Migrations under test
        # run as ``postgres``, the role scripts/run_migrations.sh uses against prod.
        return ["docker", "exec", "-i", self.name, "psql", "-U", user, "-d", db, "-X", *args]

    def try_rows(self, db: str, sql: str) -> Optional[List[str]]:
        proc = _run(self._psql_cmd(db, "-tA", "-c", sql))
        return _rows(proc.stdout) if proc.returncode == 0 else None

    def rows(self, db: str, sql: str, *, user: str = "supabase_admin") -> List[str]:
        proc = _run(self._psql_cmd(db, "-tA", "-v", "ON_ERROR_STOP=1", "-c", sql, user=user))
        if proc.returncode != 0:
            raise RuntimeError(f"{db}: {proc.stderr.decode()}")
        return _rows(proc.stdout)

    def run_script(
        self,
        db: str,
        script: bytes,
        *,
        stop_on_error: bool = True,
        single_transaction: bool = False,
        user: str = "supabase_admin",
    ) -> subprocess.CompletedProcess:
        args = ["-q", "-v", f"ON_ERROR_STOP={1 if stop_on_error else 0}"]
        if single_transaction:
            args.append("--single-transaction")
        return _run(self._psql_cmd(db, *args, user=user), input_bytes=script)

    def url(self, db: str) -> str:
        return f"postgresql://postgres:{self.password}@127.0.0.1:{self.host_port}/{db}"


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


class PgConn:
    """A database inside the throwaway container (psql for reads/DDL, psycopg URL for clients)."""

    def __init__(self, pg: ThrowawayPg, db: str):
        self.pg = pg
        self.db = db

    def rows(self, sql: str, *, user: str = "supabase_admin") -> List[str]:
        return self.pg.rows(self.db, sql, user=user)

    def execute(self, sql: str, *, user: str = "supabase_admin") -> None:
        self.pg.rows(self.db, sql, user=user)

    @property
    def url(self) -> str:
        return self.pg.url(self.db)

    def connect(self, **kwargs: Any):
        import psycopg

        return psycopg.connect(self.url, **kwargs)


# ---------------------------------------------------------------------------
# Rebuilding the base database
# ---------------------------------------------------------------------------


@dataclass
class RestoreLog:
    unexpected: List[str]
    unseen_expected: List[str]


def runner_migration_keys(project_root: Path = REPO_ROOT) -> List[str]:
    """Ledger keys the real runner would record, in its own order and with its own skip rules.

    Parsed from ``scripts/run_migrations.sh`` (``MIGRATION_DIRS`` and the ``case`` skip list) so the
    rebuilt ledger cannot drift from the runner.
    """
    script = (project_root / "scripts" / "run_migrations.sh").read_text()
    dirs = re.findall(r'"\$PROJECT_ROOT/(database/[^":]*)::([^"]*)"', script)
    assert dirs, "MIGRATION_DIRS not found in scripts/run_migrations.sh"
    skip = re.search(r"case \"\$filename\" in\s*\n\s*([^)]*)\)\s*continue", script)
    assert skip, "skip rules not found in scripts/run_migrations.sh"
    patterns = [p.strip() for p in skip.group(1).split("|")]

    def skipped(name: str) -> bool:
        import fnmatch

        return any(fnmatch.fnmatch(name, p) for p in patterns)

    keys: List[str] = []
    for rel_dir, prefix in dirs:
        for path in sorted((project_root / rel_dir).glob("*.sql")):
            if not skipped(path.name):
                keys.append(prefix + path.name)
    return keys


def build_base(pg: ThrowawayPg, prod: ProdReadOnly, db: str = "learning_loop_base") -> RestoreLog:
    # The image's ``postgres`` database carries the Supabase schemas (auth, extensions, …);
    # a bare CREATE DATABASE does not (probe 2026-09-11), so copy it.
    pg.rows("template1", f"create database {db} template postgres")

    # Roles prod grants to that the image does not create.
    image_roles = set(pg.rows(db, "select rolname from pg_roles"))
    for role in prod.rows("select rolname from pg_roles where rolname like 'e2i%' order by 1"):
        if role not in image_roles:
            pg.rows(db, f'create role "{role}" nologin')

    # Extensions the public schema needs, at prod's versions and schemas.
    for line in prod.rows(
        "select extname || '|' || extversion || '|' || extnamespace::regnamespace from pg_extension "
        "where extname in ('vector', 'pgcrypto', 'uuid-ossp') order by 1"
    ):
        name, version, schema = line.split("|")
        present = pg.rows(
            db,
            f"select extversion || '|' || extnamespace::regnamespace from pg_extension where extname = '{name}'",
        )
        if present != [f"{version}|{schema}"]:
            assert not present, f"{name} present at {present}, prod has {version}|{schema}"
            pg.rows(db, f"create extension \"{name}\" version '{version}' schema {schema}")

    # Object ACLs: pg_dump writes an object's GRANTs relative to the owner-only default, so any
    # default privileges active in the target at CREATE time would be ADDED on restore. The image
    # grants anon/authenticated on new tables in public; prod revoked those (ml/058), so objects
    # restored under the image's defaults came out with grants prod does not have (measured
    # 2026-09-11 on classification_logs). Clear the defaults that apply to public objects, restore,
    # and let the dump's own ALTER DEFAULT PRIVILEGES statements recreate prod's.
    pg.rows(db, "delete from pg_default_acl where defaclnamespace in (0, 'public'::regnamespace)")

    proc = pg.run_script(db, prod.schema_dump(), stop_on_error=False)
    errors = [
        re.sub(r"^psql:<stdin>:\d+: ", "", line)
        for line in proc.stderr.decode().splitlines()
        if "ERROR:" in line
    ]
    unexpected = [e for e in errors if e not in EXPECTED_RESTORE_ERRORS]
    unseen = [e for e in EXPECTED_RESTORE_ERRORS if e not in errors]

    # Ledger rebuilt from the repository, as the runner would record it.
    keys = [k for k in runner_migration_keys() if k not in LANE_MIGRATIONS]
    values = ",".join("('" + k.replace("'", "''") + "')" for k in keys)
    pg.rows(
        db, f"insert into public.schema_migrations(filename) values {values} on conflict do nothing"
    )

    # tool_registry / tool_dependencies rows rebuilt from the migrations that wrote them.
    for rel in ("database/ml/013_tool_composer_tables.sql", "database/ml/027_causal_discovery_tool_deps.sql",
                "database/ml/037_tool_registry_schema_sync.sql"):  # fmt: skip
        applied = pg.run_script(db, (REPO_ROOT / rel).read_bytes(), single_transaction=True)
        if applied.returncode != 0:
            raise RuntimeError(f"re-applying {rel} failed: {applied.stderr.decode()}")

    return RestoreLog(unexpected=unexpected, unseen_expected=unseen)


def clone(pg: ThrowawayPg, name: str, template: str = "learning_loop_base") -> PgConn:
    db = f"learning_loop_{re.sub(r'[^a-z0-9_]', '_', name.lower())}"
    pg.rows("template1", f"drop database if exists {db} with (force)")
    pg.rows("template1", f"create database {db} template {template}")
    return PgConn(pg, db)


def db_integration_enabled() -> bool:
    return os.getenv("E2I_DB_INTEGRATION") == "1"
