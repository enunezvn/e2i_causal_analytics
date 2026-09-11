"""Throwaway Postgres for the learning-loop real-DB tests, and read-only access to prod.

The live ``supabase-db`` is never written. Every write goes to a throwaway container of prod's
own image: memory-capped, bound to 127.0.0.1, uniquely named, labelled with the owning process
and removed at teardown (or by :func:`reap_orphans` after a hard exit). Prod contributes a
schema-only dump of ``public`` (no data rows) and single read-only SELECTs for the equivalence
checks; the rows the lane needs are rebuilt from the repository.

Run ``python -m tests.unit.test_database.learning_loop._pg --reap`` to remove containers left
by a pytest process that was killed before its finalizers ran.
"""

from __future__ import annotations

import atexit
import fnmatch
import os
import re
import secrets
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
PROD_CONTAINER = "supabase-db"
CONTAINER_PREFIX = "e2i-learnloop-pg-"
OWNER_LABEL = "e2i.learnloop.owner_pid"
MEMORY_CAP = "1g"
BASE_DB = "learning_loop_base"
CLONE_PREFIX = "learning_loop_c_"

# Restore errors we expect, by exact message, with the reason. Exact on purpose: an error that is
# not listed here fails the fixture, and a listed error that never occurs fails the sanity test.
EXPECTED_RESTORE_ERRORS: Dict[str, str] = {
    'ERROR:  schema "public" already exists': (
        "the dump's CREATE SCHEMA public; the template copy of the image's postgres database "
        "already has the schema"
    ),
}

# Applied by the tests themselves, in this order; never part of the rebuilt ledger.
LANE_MIGRATIONS = (
    "ml/039_tool_category_cohort.sql",
    "ml/040_tool_registry_startup_sync.sql",
    "ml/041_composer_learning_loop_recording.sql",
)

# The runner's un-wrap detector (scripts/run_migrations.sh), applied per line after stripping
# ``--`` comments, exactly like its ``sed 's/--.*$//' | grep -qiE``.
_RUNNER_UNWRAP = re.compile(
    r"ALTER[ \t]+TYPE[ \t].*ADD[ \t]+VALUE|CONCURRENTLY|^[ \t]*COMMIT[ \t]*;", re.IGNORECASE
)

# Prod queries: only exact SQL strings approved in code (``ProdReadOnly.approve``) reach prod.
# Approval itself refuses anything but ONE read statement: no statement separator (so it cannot
# COMMIT out of the read-only transaction it runs in), no write or session-changing keyword, and
# no function with a side effect a READ ONLY transaction does not block (backend signalling,
# advisory locks, sequence advance, notifications, file/large-object access, config, stats reset).
_READ_START = re.compile(r"^\s*(select|show|with)\b", re.IGNORECASE)
# Positive allowlist of what may stand before "(" in an approved query: the catalog functions the
# fixture reads with, and the SQL keywords that take a parenthesis. A denylist cannot be complete
# (query_to_xml('select pg_' || 'terminate_backend(1)', …) executes SQL from a string), so any
# other function name is refused. Extend this list only with side-effect-free functions.
_ALLOWED_BEFORE_PAREN = frozenset(
    {
        # functions
        "acldefault", "aclexplode", "array_to_string", "bool_or", "coalesce", "count",
        "current_database", "format_type", "md5", "pg_get_constraintdef", "pg_get_expr",
        "pg_get_functiondef", "pg_get_indexdef", "pg_get_triggerdef", "pg_get_userbyid",
        "pg_get_viewdef", "string_agg",
        # keywords
        "all", "and", "any", "as", "exists", "from", "in", "not", "on", "or", "select", "union",
        "where",
    }
)  # fmt: skip
_CALLED = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
_SIDE_EFFECT_FUNCTIONS = re.compile(
    r"\b(pg_terminate_backend|pg_cancel_backend|pg_reload_conf|pg_rotate_logfile|pg_advisory\w*|"
    r"pg_try_advisory\w*|nextval|setval|set_config|pg_notify|lo_\w+|dblink\w*|pg_read_\w*file|"
    r"pg_ls_\w+|pg_stat_reset\w*|pg_switch_wal|pg_create_\w+|pg_drop_\w+|pg_replication_\w+|"
    r"pg_logical_\w+|pg_promote|pg_log_backend_memory_contexts|txid_current|pg_current_xact_id|"
    r"pg_import_system_collations|pg_sleep\w*)\s*\(",
    re.IGNORECASE,
)
_WRITE_WORDS = re.compile(
    r"\b(insert|update|delete|merge|copy|call|do|create|alter|drop|grant|revoke|truncate|set|reset|"
    r"begin|commit|rollback|start|savepoint|lock|vacuum|analyze|cluster|reindex|refresh|notify|"
    r"listen|prepare|execute|discard|import|security|comment)\b",
    re.IGNORECASE,
)


class DbFixtureError(RuntimeError):
    """A fixture step failed. Messages never carry the throwaway password."""


def _run(
    cmd: Sequence[str],
    *,
    input_bytes: Optional[bytes] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(cmd), input=input_bytes, capture_output=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        # TimeoutExpired's message repeats the full command; report the program only.
        raise DbFixtureError(f"{' '.join(cmd[:2])} timed out after {timeout}s") from None


def _rows(stdout: bytes) -> List[str]:
    return [line for line in stdout.decode().splitlines() if line != ""]


# ---------------------------------------------------------------------------
# Prod: read-only
# ---------------------------------------------------------------------------


class ProdReadOnly:
    """Approved read statements against the live database, inside a READ ONLY transaction."""

    def __init__(self) -> None:
        self._approved: set = set()
        self.approve(*PROD_QUERIES)

    @staticmethod
    def check_read_only(sql: str) -> None:
        # Quoted identifiers, comments, dollar quoting and backslash escapes can all hide a name
        # from the word checks ("pg_terminate_backend"(1), pg_terminate_backend/**/(1)), and no
        # approved catalog query needs them, so they are refused outright.
        if (
            ";" in sql
            or any(token in sql for token in ('"', "/*", "*/", "--", "$", "\\"))
            or not _READ_START.match(sql)
            or _WRITE_WORDS.search(sql)
            or _SIDE_EFFECT_FUNCTIONS.search(sql)
            or any(
                name.lower().rsplit(".", 1)[-1] not in _ALLOWED_BEFORE_PAREN
                or ("." in name and not name.lower().startswith("pg_catalog."))
                for name in _CALLED.findall(re.sub(r"'[^']*'", "''", sql))
            )
        ):
            raise ValueError(
                f"not a single side-effect-free read, refused before reaching prod: {sql[:80]!r}"
            )

    def approve(self, *queries: str) -> None:
        for sql in queries:
            self.check_read_only(sql)
            self._approved.add(sql)

    def rows(self, sql: str) -> List[str]:
        if sql not in self._approved:
            raise ValueError(
                f"not an approved prod query, refused before reaching prod: {sql[:80]!r}"
            )
        proc = _run(
            [
                "docker", "exec", "-e", "PGOPTIONS=-c default_transaction_read_only=on",
                PROD_CONTAINER, "psql", "-U", "postgres", "-d", "postgres", "-X", "-tA",
                "-v", "ON_ERROR_STOP=1",
                "-c", "BEGIN TRANSACTION READ ONLY", "-c", sql, "-c", "ROLLBACK",
            ]
        )  # fmt: skip
        if proc.returncode != 0:
            raise DbFixtureError(f"prod read failed: {proc.stderr.decode()}")
        # psql echoes the transaction commands' tags only without -q; -tA prints BEGIN/ROLLBACK.
        return [r for r in _rows(proc.stdout) if r not in ("BEGIN", "ROLLBACK")]

    def schema_dump(self) -> bytes:
        # pg_dump reads inside its own REPEATABLE READ, READ ONLY transaction.
        proc = _run(
            [
                "docker", "exec", "-e", "PGOPTIONS=-c default_transaction_read_only=on",
                PROD_CONTAINER, "pg_dump", "-U", "postgres", "--schema-only", "-n", "public",
                "postgres",
            ]
        )  # fmt: skip
        if proc.returncode != 0:
            raise DbFixtureError(f"schema dump failed: {proc.stderr.decode()}")
        return proc.stdout

    def image(self) -> str:
        return (
            _run(["docker", "inspect", PROD_CONTAINER, "--format", "{{.Config.Image}}"])
            .stdout.decode()
            .strip()
        )


# ---------------------------------------------------------------------------
# Throwaway container
# ---------------------------------------------------------------------------


_GONE = "gone"
_UNKNOWN = "unknown"


def _process_start_ticks(pid: int) -> str:
    """The process start time in clock ticks, ``"gone"`` when no such process, ``"unknown"`` otherwise."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return _GONE
    except OSError:
        return _UNKNOWN
    # Field 22 (starttime) counted after the ")" that closes the command name.
    fields = stat.rsplit(")", 1)[-1].split()
    return fields[19] if len(fields) > 19 and fields[19].isdigit() else _UNKNOWN


def owner_identity(pid: Optional[int] = None) -> str:
    """``host/boot_id/pid-namespace/pid/start-ticks`` of a process on THIS kernel.

    A PID alone is not an identity: another Docker client (a different host, container or PID
    namespace) can hold the same number, and PIDs are reused. Two identities describe the same
    live process only when every field matches.
    """
    pid = os.getpid() if pid is None else pid
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    pid_ns = os.stat("/proc/self/ns/pid").st_ino
    return f"{os.uname().nodename}/{boot_id}/{pid_ns}/{pid}/{_process_start_ticks(pid)}"


def _owned_containers() -> List[Dict[str, str]]:
    proc = _run(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_PREFIX}",
         "--format", '{{.Names}}|{{.Label "' + OWNER_LABEL + '"}}']
    )  # fmt: skip
    out = []
    for line in _rows(proc.stdout):
        name, _, owner = line.partition("|")
        if name.startswith(CONTAINER_PREFIX):
            out.append({"name": name, "owner": owner})
    return out


def owner_is_gone(owner: str) -> bool:
    """True only when the owner provably lived on this host, boot and PID namespace and is dead.

    Anything else (another host or namespace, a malformed or missing label) is NOT reaped: it may
    belong to a live session this process cannot see.
    """
    parts = owner.split("/")
    if len(parts) != 5 or not all(parts[:3]) or not parts[2].isdigit():
        return False
    host, boot_id, pid_ns, pid, start = parts
    if not pid.isdigit() or not start.isdigit():
        return False
    here = owner_identity().split("/")
    if (host, boot_id, pid_ns) != tuple(here[:3]):
        return False
    now = _process_start_ticks(int(pid))
    if now == _UNKNOWN:
        return False
    return now == _GONE or now != start


def reap_orphans() -> List[str]:
    """Remove throwaway containers whose owning process is provably gone (hard-killed runs)."""
    removed = []
    for c in _owned_containers():
        if owner_is_gone(c["owner"]):
            _run(["docker", "rm", "-f", c["name"]])
            removed.append(c["name"])
    return removed


@dataclass
class ThrowawayPg:
    image: str
    name: str = field(default_factory=lambda: CONTAINER_PREFIX + secrets.token_hex(4))
    _password: str = field(default_factory=lambda: secrets.token_hex(16), repr=False)
    host_port: Optional[int] = None

    def start(self, ready_timeout_s: int = 180) -> None:
        # Cleanup is registered BEFORE the container exists, and any failure (including
        # KeyboardInterrupt) removes it before re-raising.
        atexit.register(self.stop)
        try:
            proc = _run(
                [
                    "docker", "run", "-d", "--name", self.name,
                    "--label", f"{OWNER_LABEL}={owner_identity()}",
                    "--memory", MEMORY_CAP, "--memory-swap", MEMORY_CAP,
                    "-e", "POSTGRES_PASSWORD",  # value comes from the environment, not argv
                    "-p", "127.0.0.1::5432",
                    self.image,
                    "postgres", "-c", "listen_addresses=*", "-c", "shared_buffers=128MB",
                    "-c", "max_connections=60",
                ],
                env={**os.environ, "POSTGRES_PASSWORD": self._password},
            )  # fmt: skip
            if proc.returncode != 0:
                raise DbFixtureError(f"docker run failed: {proc.stderr.decode()}")
            deadline = time.monotonic() + ready_timeout_s
            while time.monotonic() < deadline:
                # The image runs its init scripts on a temporary server that listens on the unix
                # socket only (listen_addresses=''), then shuts it down and starts the real one.
                # A socket query can therefore succeed seconds before "the database system is
                # shutting down" (measured 2026-09-11); require a TCP query, which only the real
                # server answers.
                if self._tcp_ready():
                    break
                time.sleep(2)
            else:
                raise DbFixtureError(f"{self.name} not ready after {ready_timeout_s}s")
            port = _run(["docker", "port", self.name, "5432/tcp"]).stdout.decode().strip()
            self.host_port = int(port.rsplit(":", 1)[1])
        except BaseException:
            self.stop()
            raise

    def _tcp_ready(self) -> bool:
        proc = _run(
            [
                "docker", "exec", "-e", "PGPASSWORD", self.name,
                "psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "postgres", "-X", "-tA",
                "-c", "select 1",
            ],
            env={**os.environ, "PGPASSWORD": self._password},
        )  # fmt: skip
        return proc.returncode == 0 and _rows(proc.stdout) == ["1"]

    def exists(self) -> bool:
        return _run(["docker", "inspect", self.name]).returncode == 0

    def stop(self) -> None:
        _run(["docker", "rm", "-f", self.name])
        if self.exists():
            raise DbFixtureError(f"could not remove throwaway container {self.name}")

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
            raise DbFixtureError(f"{db}: {proc.stderr.decode()}")
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

    def dsn(self, db: str, user: str = "postgres") -> str:
        """A connection string WITHOUT the password; pair it with :meth:`client_env`."""
        return f"postgresql://{user}@127.0.0.1:{self.host_port}/{db}"

    def client_env(self) -> Dict[str, str]:
        return {**os.environ, "PGPASSWORD": self._password}

    def connect(self, db: str, user: str = "postgres", **kwargs: Any):
        import psycopg

        return psycopg.connect(self.dsn(db, user), password=self._password, **kwargs)

    def terminate_connections(self, db: str) -> None:
        self.rows(
            "template1",
            "select count(pg_terminate_backend(pid)) from pg_stat_activity "
            f"where datname = '{db}' and pid <> pg_backend_pid()",
        )


# ---------------------------------------------------------------------------
# Databases inside the container
# ---------------------------------------------------------------------------


class PgConn:
    """One database inside the throwaway container."""

    def __init__(self, pg: ThrowawayPg, db: str):
        self.pg = pg
        self.db = db

    def rows(self, sql: str, *, user: str = "supabase_admin") -> List[str]:
        return self.pg.rows(self.db, sql, user=user)

    def execute(self, sql: str, *, user: str = "supabase_admin") -> None:
        self.pg.rows(self.db, sql, user=user)

    def connect(self, user: str = "postgres", **kwargs: Any):
        return self.pg.connect(self.db, user, **kwargs)

    @contextmanager
    def rolled_back(self, user: str = "postgres") -> Iterator[Any]:
        """A psycopg connection whose transaction is always rolled back."""
        conn = self.connect(user)
        try:
            yield conn
        finally:
            conn.rollback()
            conn.close()


def runner_unwraps(sql_text: str) -> bool:
    """True when scripts/run_migrations.sh applies the file un-wrapped (no --single-transaction)."""
    return any(_RUNNER_UNWRAP.search(re.sub(r"--.*$", "", line)) for line in sql_text.splitlines())


def apply_migration(conn: PgConn, path: Path, *, user: str = "postgres") -> str:
    """Apply one migration file the way the runner does; returns ``"wrapped"`` or ``"unwrapped"``."""
    text = path.read_bytes()
    unwrapped = runner_unwraps(text.decode())
    proc = conn.pg.run_script(conn.db, text, single_transaction=not unwrapped, user=user)
    if proc.returncode != 0:
        raise DbFixtureError(f"applying {path.name} failed: {proc.stderr.decode()}")
    return "unwrapped" if unwrapped else "wrapped"


def migrate(conn: PgConn, upto: Optional[str]) -> List[str]:
    """Apply the lane's migrations in order through ``upto`` (a LANE_MIGRATIONS key), as postgres."""
    if upto is None:
        return []
    if upto not in LANE_MIGRATIONS:
        raise ValueError(f"unknown lane migration {upto!r}; expected one of {LANE_MIGRATIONS}")
    applied = []
    for key in LANE_MIGRATIONS[: LANE_MIGRATIONS.index(upto) + 1]:
        path = REPO_ROOT / "database" / key
        if not path.exists():
            raise DbFixtureError(f"{key} does not exist yet")
        apply_migration(conn, path)
        applied.append(key)
    return applied


# ---------------------------------------------------------------------------
# Rebuilding the base database
# ---------------------------------------------------------------------------


@dataclass
class RestoreLog:
    errors: List[str]
    unexpected: List[str]
    unseen_expected: List[str]


def runner_migration_keys(project_root: Path = REPO_ROOT) -> List[str]:
    """Ledger keys the real runner would record, in its own order and with its own skip rules.

    Parsed from ``scripts/run_migrations.sh`` (``MIGRATION_DIRS`` and the ``case`` skip list) so the
    rebuilt ledger cannot drift from the runner.
    """
    script = (project_root / "scripts" / "run_migrations.sh").read_text()
    dirs = re.findall(r'"\$PROJECT_ROOT/(database/[^":]*)::([^"]*)"', script)
    if not dirs:
        raise DbFixtureError("MIGRATION_DIRS not found in scripts/run_migrations.sh")
    skip = re.search(r"case \"\$filename\" in\s*\n\s*([^)]*)\)\s*continue", script)
    if not skip:
        raise DbFixtureError("skip rules not found in scripts/run_migrations.sh")
    patterns = [p.strip() for p in skip.group(1).split("|")]
    keys: List[str] = []
    for rel_dir, prefix in dirs:
        for path in sorted((project_root / rel_dir).glob("*.sql")):
            if not any(fnmatch.fnmatch(path.name, p) for p in patterns):
                keys.append(prefix + path.name)
    return keys


# Extensions and non-public objects that public objects depend on, per pg_depend.
PROD_EXTENSION_INVENTORY = """
with public_objs as (
  select 'pg_class'::regclass as classid, oid as objid from pg_class where relnamespace = 'public'::regnamespace
  union all select 'pg_proc'::regclass, oid from pg_proc where pronamespace = 'public'::regnamespace
  union all select 'pg_type'::regclass, oid from pg_type where typnamespace = 'public'::regnamespace
  union all select 'pg_attrdef'::regclass, d.oid from pg_attrdef d join pg_class c on c.oid = d.adrelid
            where c.relnamespace = 'public'::regnamespace
  union all select 'pg_constraint'::regclass, oid from pg_constraint where connamespace = 'public'::regnamespace
  union all select 'pg_rewrite'::regclass, r.oid from pg_rewrite r join pg_class c on c.oid = r.ev_class
            where c.relnamespace = 'public'::regnamespace
), refs as (
  select distinct d.refclassid, d.refobjid from pg_depend d
  join public_objs p on p.classid = d.classid and p.objid = d.objid where d.deptype in ('n', 'a')
)
select distinct e.extname || '|' || e.extversion || '|' || e.extnamespace::regnamespace
from refs r join pg_depend x on x.classid = r.refclassid and x.objid = r.refobjid and x.deptype = 'e'
join pg_extension e on e.oid = x.refobjid order by 1
""".strip()

PROD_NONPUBLIC_REFERENCES = """
with public_objs as (
  select 'pg_class'::regclass as classid, oid as objid from pg_class where relnamespace = 'public'::regnamespace
  union all select 'pg_proc'::regclass, oid from pg_proc where pronamespace = 'public'::regnamespace
  union all select 'pg_attrdef'::regclass, d.oid from pg_attrdef d join pg_class c on c.oid = d.adrelid
            where c.relnamespace = 'public'::regnamespace
  union all select 'pg_constraint'::regclass, oid from pg_constraint where connamespace = 'public'::regnamespace
  union all select 'pg_rewrite'::regclass, r.oid from pg_rewrite r join pg_class c on c.oid = r.ev_class
            where c.relnamespace = 'public'::regnamespace
)
select distinct case d.refclassid
    when 'pg_class'::regclass then 'rel:' || d.refobjid::regclass::text
    when 'pg_proc'::regclass then 'proc:' || d.refobjid::regprocedure::text
    when 'pg_type'::regclass then 'type:' || format_type(d.refobjid, null)
  end as ref
from pg_depend d join public_objs p on p.classid = d.classid and p.objid = d.objid
where d.deptype in ('n', 'a') and d.refclassid in ('pg_class'::regclass, 'pg_proc'::regclass, 'pg_type'::regclass)
  and coalesce(
    (select relnamespace from pg_class where oid = d.refobjid and d.refclassid = 'pg_class'::regclass),
    (select pronamespace from pg_proc where oid = d.refobjid and d.refclassid = 'pg_proc'::regclass),
    (select typnamespace from pg_type where oid = d.refobjid and d.refclassid = 'pg_type'::regclass)
  ) not in ('public'::regnamespace, 'pg_catalog'::regnamespace)
order by 1
""".strip()


LANE_ROLES = (
    "anon",
    "authenticated",
    "authenticator",
    "postgres",
    "service_role",
    "supabase_admin",
)
_lane_roles_sql = "(" + ",".join("'" + r + "'" for r in LANE_ROLES) + ")"
LANE_ROLE_MEMBERSHIPS = (
    "select r.rolname || '>' || m.rolname || ':' || am.admin_option from pg_auth_members am "
    "join pg_roles r on r.oid = am.roleid join pg_roles m on m.oid = am.member "
    f"where m.rolname in {_lane_roles_sql} or r.rolname in {_lane_roles_sql} order by 1"
)


LANE_MIGRATIONS_IN_LEDGER = (
    "select filename from public.schema_migrations where filename in ("
    + ",".join("'" + k + "'" for k in LANE_MIGRATIONS)
    + ") order by 1"
)
PROD_DB_OWNER = "select datdba::regrole::text from pg_database where datname = current_database()"
PROD_E2I_ROLES = "select rolname from pg_roles where rolname like 'e2i%' order by 1"

# Every query build_base and the fixtures send to prod.
PROD_QUERIES = (
    LANE_MIGRATIONS_IN_LEDGER,
    PROD_DB_OWNER,
    PROD_E2I_ROLES,
    LANE_ROLE_MEMBERSHIPS,
    PROD_EXTENSION_INVENTORY,
    PROD_NONPUBLIC_REFERENCES,
)


def acl_text(acl_expr: str) -> str:
    """SQL rendering an aclitem[] as sorted, merged ``grantee=privilege[*]/grantor`` items.

    ``*`` marks WITH GRANT OPTION. Merging matters only for concatenated arrays (the
    effective-default-ACL model); a real object ACL never repeats an item.
    """
    # Items are merged the way PostgreSQL merges ACLs: one privilege per (grantee, privilege,
    # grantor), grantable if any merged item was.
    return (
        "(select string_agg(i.item, ',' order by i.item) from (select e.g || '=' || e.p "
        "|| case when bool_or(e.gr) then '*' else '' end || '/' || e.gto as item from (select "
        "a.grantee::regrole::text as g, a.privilege_type as p, a.is_grantable as gr, "
        f"a.grantor::regrole::text as gto from aclexplode({acl_expr}) a) e group by e.g, e.p, e.gto) i)"
    )


def lane_migrations_already_in_prod(prod: ProdReadOnly) -> List[str]:
    return prod.rows(LANE_MIGRATIONS_IN_LEDGER)


def build_base(pg: ThrowawayPg, prod: ProdReadOnly, db: str = BASE_DB) -> RestoreLog:
    # The image's ``postgres`` database carries the Supabase schemas (auth, extensions, …) and
    # event triggers; a bare CREATE DATABASE does not (probe 2026-09-11), so copy it.
    # Same owner as prod's database: PG15's public schema is owned by pg_database_owner, so the
    # database owner decides whether ``postgres`` (the migration role) may CREATE in public.
    (owner,) = prod.rows(PROD_DB_OWNER)
    pg.terminate_connections("postgres")
    pg.rows("template1", f'create database {db} template postgres owner "{owner}"')

    # Roles prod grants to that the image does not create, and prod's memberships of the roles
    # the lane runs as (role attributes of those roles are compared by the sanity suite).
    image_roles = set(pg.rows(db, "select rolname from pg_roles"))
    for role in prod.rows(PROD_E2I_ROLES):
        if role not in image_roles:
            pg.rows(db, f'create role "{role}" nologin')
            image_roles.add(role)
    image_members = set(pg.rows(db, LANE_ROLE_MEMBERSHIPS))
    for line in prod.rows(LANE_ROLE_MEMBERSHIPS):
        if line in image_members:
            continue
        granted, rest = line.split(">", 1)
        member, admin = rest.split(":")
        if granted not in image_roles:
            pg.rows(db, f'create role "{granted}" nologin')
            image_roles.add(granted)
        admin_clause = " with admin option" if admin == "true" else ""
        pg.rows(db, f'grant "{granted}" to "{member}"{admin_clause}')

    # Extensions public objects depend on (pg_depend inventory), at prod's versions and schemas.
    for line in prod.rows(PROD_EXTENSION_INVENTORY):
        name, version, schema = line.split("|")
        present = pg.rows(
            db,
            "select extversion || '|' || extnamespace::regnamespace from pg_extension "
            f"where extname = '{name}'",
        )
        if present != [f"{version}|{schema}"]:
            if present:
                raise DbFixtureError(
                    f"{name} is {present} in the image, prod has {version}|{schema}"
                )
            pg.rows(db, f"create extension \"{name}\" version '{version}' schema {schema}")

    # Object ACLs: pg_dump writes an object's GRANTs relative to the owner-only default, so any
    # default privileges active in the target at CREATE time would be ADDED on restore. The image
    # grants anon/authenticated on new tables in public; prod revoked those (ml/058), so objects
    # restored under the image's defaults came out with grants prod does not have (measured
    # 2026-09-11 on classification_logs). Clear the defaults that apply to public objects, restore,
    # and let the dump's own ALTER DEFAULT PRIVILEGES statements recreate prod's.
    pg.rows(db, "delete from pg_default_acl where defaclnamespace in (0, 'public'::regnamespace)")

    proc = pg.run_script(db, prod.schema_dump(), stop_on_error=False)
    stderr = proc.stderr.decode().splitlines()
    errors = [re.sub(r"^psql:<stdin>:\d+: ", "", line) for line in stderr if "ERROR:" in line]
    fatal = [line for line in stderr if "FATAL:" in line or "PANIC:" in line]
    unexpected = [e for e in errors if e not in EXPECTED_RESTORE_ERRORS]
    unseen = [e for e in EXPECTED_RESTORE_ERRORS if e not in errors]
    if proc.returncode != 0 or fatal or unexpected:
        raise DbFixtureError(
            f"schema restore failed (rc={proc.returncode}); fatal={fatal[:5]} unexpected={unexpected[:10]}"
        )

    # Ledger rebuilt from the repository, as the runner would record it.
    keys = [k for k in runner_migration_keys() if k not in LANE_MIGRATIONS]
    values = ",".join("('" + k.replace("'", "''") + "')" for k in keys)
    pg.rows(
        db, f"insert into public.schema_migrations(filename) values {values} on conflict do nothing"
    )

    # tool_registry / tool_dependencies rows rebuilt from the migrations that wrote them.
    for rel in (
        "database/ml/013_tool_composer_tables.sql",
        "database/ml/027_causal_discovery_tool_deps.sql",
        "database/ml/037_tool_registry_schema_sync.sql",
    ):
        applied = pg.run_script(db, (REPO_ROOT / rel).read_bytes(), single_transaction=True)
        if applied.returncode != 0:
            raise DbFixtureError(f"re-applying {rel} failed: {applied.stderr.decode()}")

    return RestoreLog(errors=errors, unexpected=unexpected, unseen_expected=unseen)


def clone(pg: ThrowawayPg, label: str, template: str = BASE_DB) -> PgConn:
    """A new database copied from ``template``, with a unique bounded name.

    Names are ``learning_loop_c_<label[:20]>_<hex6>``: never the base's name, never truncated by
    Postgres' 63-byte identifier limit, never colliding between callers.
    """
    slug = re.sub(r"[^a-z0-9_]", "_", label.lower())[:20]
    db = f"{CLONE_PREFIX}{slug}_{secrets.token_hex(3)}"
    assert db != BASE_DB and len(db) <= 63
    (owner,) = pg.rows(
        "template1", f"select datdba::regrole::text from pg_database where datname = '{template}'"
    )
    pg.terminate_connections(template)
    pg.rows("template1", f'create database {db} template {template} owner "{owner}"')
    return PgConn(pg, db)


def drop(conn: PgConn) -> None:
    if not conn.db.startswith(CLONE_PREFIX):
        raise DbFixtureError(f"refusing to drop {conn.db}: not a clone")
    conn.pg.rows("template1", f"drop database if exists {conn.db} with (force)")


def db_integration_enabled() -> bool:
    return os.getenv("E2I_DB_INTEGRATION") == "1"


if __name__ == "__main__":
    if sys.argv[1:] == ["--reap"]:
        print("removed:", reap_orphans())
    else:
        print(__doc__)
