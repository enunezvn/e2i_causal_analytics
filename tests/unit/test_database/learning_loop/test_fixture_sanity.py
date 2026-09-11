"""The throwaway database is a faithful copy of prod for everything the learning loop touches.

Every later real-DB test in this package (migrations ml/039–041, the recording RPCs, the
registry sync client) runs against a clone of ``base_db``, so this module proves the copy
before anything depends on it:

* it lives in a throwaway, memory-capped container of prod's own image, bound to 127.0.0.1, and
  the live ``supabase-db`` is only ever read (spec §9, dispatcher constraint 2026-09-11);
* its schema came from a schema-only dump of prod's ``public`` schema, and every restore error
  is one expected by name;
* the rows the lane needs were rebuilt from the repository (the migration ledger; the
  tool_registry / tool_dependencies seeds of ml/013, 027 and 037) and equal prod's;
* the lane's objects equal prod's in columns, constraints, indexes, ownership, RLS, policies,
  ACLs (object and default, by effective result), view and function bodies, triggers, enums,
  extensions, event triggers, role attributes, memberships and settings;
* the migration helper applies a file the way scripts/run_migrations.sh does.

Opt-in: ``E2I_DB_INTEGRATION=1`` (skipped in CI, which has no database). Run with ``-n 0``.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(
        os.getenv("E2I_DB_INTEGRATION") != "1",
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    # The session fixture starts a container and restores a schema dump inside the first test's
    # setup; the repo-wide 30 s pytest-timeout would cut that off on a loaded box.
    pytest.mark.timeout(600),
]

LANE_TABLES = [
    "classification_logs",
    "composer_episodes",
    "composition_steps",
    "episodic_memories",
    "schema_migrations",
    "tool_dependencies",
    "tool_performance",
    "tool_registry",
]
LANE_VIEWS = [
    "v_active_compositions",
    "v_classification_accuracy",
    "v_composition_success_rate",
    "v_tool_reliability",
]
LANE_FUNCTIONS = [
    "find_similar_compositions",
    "get_tool_execution_order",
    "trigger_log_step_performance",
    "trigger_update_tool_registry_timestamp",
    "update_tool_registry_metrics",
]
LANE_ENUMS = ["composition_status", "e2i_agent_name", "routing_pattern", "tool_category"]
LANE_ROLES = [
    "anon",
    "authenticated",
    "authenticator",
    "postgres",
    "service_role",
    "supabase_admin",
]


def _in(names: list[str]) -> str:
    return "(" + ",".join("'" + n + "'" for n in names) + ")"


_PUBLIC_REL = "c.relnamespace = 'public'::regnamespace"

# aspect -> (query, may the result legitimately be empty on both sides?)
EQUIVALENCE_QUERIES = {
    "columns": (
        "select c.relname || '.' || a.attname || ':' || format_type(a.atttypid, a.atttypmod) || ':' "
        "|| a.attnotnull || ':' || coalesce(pg_get_expr(d.adbin, d.adrelid), '') || ':' "
        "|| a.attidentity::text || a.attgenerated::text "
        "from pg_attribute a join pg_class c on c.oid = a.attrelid "
        "left join pg_attrdef d on d.adrelid = a.attrelid and d.adnum = a.attnum "
        f"where {_PUBLIC_REL} and c.relname in {_in(LANE_TABLES + LANE_VIEWS)} "
        "and a.attnum > 0 and not a.attisdropped order by 1",
        False,
    ),
    "constraints": (
        "select c.relname || ':' || k.conname || ':' || pg_get_constraintdef(k.oid) "
        "from pg_constraint k join pg_class c on c.oid = k.conrelid "
        f"where {_PUBLIC_REL} and c.relname in {_in(LANE_TABLES)} order by 1",
        False,
    ),
    "indexes": (
        "select c.relname || ':' || pg_get_indexdef(i.indexrelid) from pg_index i "
        f"join pg_class c on c.oid = i.indrelid where {_PUBLIC_REL} and c.relname in {_in(LANE_TABLES)} "
        "order by 1",
        False,
    ),
    "ownership_rls": (
        "select c.relname || ':' || c.relkind::text || ':' || pg_get_userbyid(c.relowner) || ':' "
        "|| c.relrowsecurity || c.relforcerowsecurity from pg_class c "
        f"where {_PUBLIC_REL} and c.relname in {_in(LANE_TABLES + LANE_VIEWS)} order by 1",
        False,
    ),
    "relation_acls": (
        "select c.relname || ':' || coalesce((select string_agg(a.grantee::regrole::text || '=' "
        "|| a.privilege_type || '/' || a.grantor::regrole::text, ',' order by 1) "
        "from aclexplode(c.relacl) a), '<owner-default>') from pg_class c "
        f"where {_PUBLIC_REL} and c.relname in {_in(LANE_TABLES + LANE_VIEWS)} order by 1",
        False,
    ),
    "policies": (
        "select tablename || ':' || policyname || ':' || cmd || ':' || roles::text || ':' "
        "|| coalesce(qual, '') || ':' || coalesce(with_check, '') from pg_policies "
        f"where schemaname = 'public' and tablename in {_in(LANE_TABLES)} order by 1",
        True,
    ),
    "owned_sequences": (
        "select t.relname || ':' || s.relname || ':' || coalesce(s.relacl::text, '') from pg_depend d "
        "join pg_class s on s.oid = d.objid and s.relkind = 'S' join pg_class t on t.oid = d.refobjid "
        f"where t.relnamespace = 'public'::regnamespace and t.relname in {_in(LANE_TABLES)} order by 1",
        True,
    ),
    "view_definitions": (
        "select c.relname || ':' || md5(pg_get_viewdef(c.oid)) from pg_class c "
        f"where {_PUBLIC_REL} and c.relname in {_in(LANE_VIEWS)} order by 1",
        False,
    ),
    "functions": (
        "select p.oid::regprocedure || ':' || md5(pg_get_functiondef(p.oid)) || ':' "
        "|| pg_get_userbyid(p.proowner) || ':' || coalesce((select string_agg(a.grantee::regrole::text "
        "|| '=' || a.privilege_type, ',' order by 1) from aclexplode(p.proacl) a), '<owner-default>') "
        f"from pg_proc p where p.pronamespace = 'public'::regnamespace and p.proname in {_in(LANE_FUNCTIONS)} "
        "order by 1",
        False,
    ),
    "triggers": (
        "select c.relname || ':' || t.tgname || ':' || t.tgenabled::text || ':' || pg_get_triggerdef(t.oid) "
        "from pg_trigger t join pg_class c on c.oid = t.tgrelid "
        f"where not t.tgisinternal and {_PUBLIC_REL} and c.relname in {_in(LANE_TABLES)} order by 1",
        False,
    ),
    "enums": (
        "select t.typname || ':' || string_agg(e.enumlabel, ',' order by e.enumsortorder) "
        "from pg_type t join pg_enum e on e.enumtypid = t.oid "
        f"where t.typname in {_in(LANE_ENUMS)} group by t.typname order by 1",
        False,
    ),
    "public_dependency_extensions": (_pg.PROD_EXTENSION_INVENTORY, False),
    "event_triggers": (
        "select evtname || ':' || evtevent || ':' || evtenabled::text || ':' || evtfoid::regproc::text "
        "|| ':' || coalesce(array_to_string(evttags, ','), '') from pg_event_trigger order by 1",
        False,
    ),
    "default_acls": (
        "select defaclrole::regrole || ':' || defaclnamespace::regnamespace || ':' || defaclobjtype::text "
        "|| ':' || (select string_agg(a.grantee::regrole::text || '=' || a.privilege_type, ',' order by 1) "
        "from aclexplode(defaclacl) a) from pg_default_acl "
        "where defaclnamespace in (0, 'public'::regnamespace) order by 1",
        False,
    ),
    "role_attributes": (
        "select rolname || ':' || rolsuper || rolcreaterole || rolcreatedb || rolbypassrls || rolinherit "
        f"|| rolcanlogin from pg_roles where rolname in {_in(LANE_ROLES)} order by 1",
        False,
    ),
    "role_memberships": (_pg.LANE_ROLE_MEMBERSHIPS, False),
    "database_owner": (
        "select datdba::regrole::text from pg_database where datname = current_database()",
        False,
    ),
    "public_schema": (
        "select nspowner::regrole::text || ':' || (select string_agg(a.grantee::regrole::text || '=' "
        "|| a.privilege_type, ',' order by 1) from aclexplode(nspacl) a) from pg_namespace "
        "where nspname = 'public'",
        False,
    ),
    "role_settings": (
        "select r.rolname || ':' || (s.setdatabase <> 0) || ':' || s.setconfig::text "
        "from pg_db_role_setting s join pg_roles r on r.oid = s.setrole "
        f"where r.rolname in {_in(LANE_ROLES)} order by 1",
        True,
    ),
}


def test_container_is_throwaway_capped_and_local(pg_container):
    info = json.loads(
        subprocess.run(
            ["docker", "inspect", pg_container.name], capture_output=True, text=True, check=True
        ).stdout
    )[0]
    assert pg_container.name.startswith(_pg.CONTAINER_PREFIX)
    assert pg_container.name != _pg.PROD_CONTAINER
    assert info["HostConfig"]["Memory"] == 1024**3
    assert info["HostConfig"]["MemorySwap"] == 1024**3
    assert [b["HostIp"] for b in info["HostConfig"]["PortBindings"]["5432/tcp"]] == ["127.0.0.1"]
    assert info["Config"]["Labels"][_pg.OWNER_LABEL] == str(os.getpid())
    assert info["Config"]["Image"] == _pg.ProdReadOnly().image()
    # The password travels in the environment, never in argv (docker inspect Args).
    assert pg_container._password not in json.dumps(info["Args"])
    assert pg_container._password not in pg_container.dsn("postgres")


@pytest.mark.parametrize(
    "sql",
    [
        "insert into tool_registry(name) values ('x')",
        "select 1; commit",
        "with d as (delete from tool_registry returning 1) select count(*) from d",
        "update tool_registry set version = 'x'",
        "set default_transaction_read_only = off",
        "begin read write",
        "drop table tool_registry",
    ],
)
def test_prod_access_refuses_anything_but_one_read(sql):
    with pytest.raises(ValueError, match="refused before reaching prod"):
        _pg.ProdReadOnly().rows(sql)


def test_prod_reads_run_in_a_read_only_transaction(prod_readonly):
    assert prod_readonly.rows("show transaction_read_only") == ["on"]


def test_restore_errors_were_exactly_the_expected_ones(base_db):
    assert base_db.restore_log.unexpected == []
    assert base_db.restore_log.unseen_expected == []


def test_ledger_is_the_repository_ledger_and_prod_has_nothing_older(base_db, prod_readonly):
    # Sorted in Python: the database collation orders punctuation differently from byte order.
    copy = sorted(base_db.rows("select filename from public.schema_migrations"))
    expected = sorted(k for k in _pg.runner_migration_keys() if k not in _pg.LANE_MIGRATIONS)
    assert copy == expected
    prod = set(prod_readonly.rows("select filename from public.schema_migrations"))
    assert set(copy) <= prod, sorted(set(copy) - prod)
    # Anything prod has beyond the repository must be NEWER than the repository's last file in
    # the same directory (deployed after this branch's base), never an older gap.
    for extra in sorted(prod - set(copy)):
        prefix = extra.rsplit("/", 1)[0] + "/" if "/" in extra else ""
        same_dir = [k for k in copy if (k.rsplit("/", 1)[0] + "/" if "/" in k else "") == prefix]
        assert not same_dir or extra > max(same_dir), (
            f"prod has {extra}, older than the repository's"
        )


def test_tool_registry_seed_fields_equal_prod(base_db, prod_readonly):
    query = (
        "select name || '|' || category || '|' || source_agent || '|' || md5(description) || '|' "
        "|| md5(input_schema::text) || '|' || md5(output_schema::text) || '|' "
        "|| coalesce(avg_latency_ms::text, '') || '|' || coalesce(version, '') from tool_registry order by name"
    )
    rows = base_db.rows(query)
    assert len(rows) == 16
    assert rows == prod_readonly.rows(query)


def test_tool_dependencies_equal_prod(base_db, prod_readonly):
    query = (
        "select c.name || '<-' || p.name || '|' || coalesce(d.output_field, '') || '|' "
        "|| coalesce(d.input_field, '') from tool_dependencies d "
        "join tool_registry c on c.tool_id = d.consumer_tool_id "
        "join tool_registry p on p.tool_id = d.producer_tool_id order by 1"
    )
    rows = base_db.rows(query)
    assert len(rows) == 11
    assert rows == prod_readonly.rows(query)


def test_loop_tables_start_empty(base_db):
    assert base_db.rows(
        "select (select count(*) from tool_performance) || '|' || (select count(*) from composer_episodes) "
        "|| '|' || (select count(*) from composition_steps)"
    ) == ["0|0|0"]


def test_required_lane_objects_exist_on_both_sides(base_db, prod_readonly):
    query = (
        "select c.relkind::text || ':' || c.relname from pg_class c "
        f"where {_PUBLIC_REL} and c.relname in {_in(LANE_TABLES + LANE_VIEWS)} "
        "union all select 'f:' || p.proname from pg_proc p "
        f"where p.pronamespace = 'public'::regnamespace and p.proname in {_in(LANE_FUNCTIONS)} "
        f"union all select 'e:' || t.typname from pg_type t where t.typname in {_in(LANE_ENUMS)} order by 1"
    )
    expected = sorted(
        [f"r:{t}" for t in LANE_TABLES]
        + [f"v:{v}" for v in LANE_VIEWS]
        + [f"f:{f}" for f in LANE_FUNCTIONS]
        + [f"e:{e}" for e in LANE_ENUMS]
    )
    assert base_db.rows(query) == expected
    assert prod_readonly.rows(query) == expected


@pytest.mark.parametrize("aspect", sorted(EQUIVALENCE_QUERIES))
def test_schema_equivalent_for_lane_objects(base_db, prod_readonly, aspect):
    query, may_be_empty = EQUIVALENCE_QUERIES[aspect]
    got = base_db.rows(query)
    want = prod_readonly.rows(query)
    if not may_be_empty:
        assert want, f"{aspect}: prod returned nothing, so equality would be vacuous"
    assert got == want


def test_non_public_objects_public_depends_on_exist_in_the_copy(base_db, prod_readonly):
    refs = prod_readonly.rows(_pg.PROD_NONPUBLIC_REFERENCES)
    assert refs, "public objects reference no non-public object; the check would be vacuous"
    missing = []
    for ref in refs:
        kind, _, name = ref.partition(":")
        fn = {"rel": "to_regclass", "proc": "to_regprocedure", "type": "to_regtype"}[kind]
        if base_db.rows(f"select {fn}('{name}') is not null") != ["t"]:
            missing.append(ref)
    assert missing == []


_EXPLODED = (
    "(select string_agg(g || '=' || p, ',' order by g, p) from (select distinct "
    "a.grantee::regrole::text as g, a.privilege_type as p from aclexplode({acl}) a) x)"
)


@pytest.mark.parametrize("objtype,ddl,acl_query", [
    ("r", "create table public.zz_default_acl_probe (id int)",
     "select {e} from pg_class where oid = 'public.zz_default_acl_probe'::regclass"),
    ("S", "create sequence public.zz_default_acl_probe_seq",
     "select {e} from pg_class where oid = 'public.zz_default_acl_probe_seq'::regclass"),
    ("f", "create function public.zz_default_acl_probe_fn() returns int language sql as 'select 1'",
     "select {e} from pg_proc where oid = 'public.zz_default_acl_probe_fn()'::regprocedure"),
])  # fmt: skip
def test_new_objects_get_prods_effective_default_privileges(
    clone_db, prod_readonly, objtype, ddl, acl_query
):
    # What a migration running as postgres actually gets on a new public object must equal what
    # prod would give it: the built-in default for the object type (acldefault; for functions
    # that includes EXECUTE to PUBLIC) plus prod's schema-level default ACL for (postgres, public).
    # Prod has no database-wide (namespace 0) default ACL rows, so nothing replaces acldefault;
    # the default_acls aspect above pins that.
    want = prod_readonly.rows(
        "select (select string_agg(g || '=' || p, ',' order by g, p) from (select distinct "
        "a.grantee::regrole::text as g, a.privilege_type as p from aclexplode("
        f"acldefault('{objtype}', 'postgres'::regrole) || d.defaclacl) a) x) "
        "from pg_default_acl d where d.defaclrole = 'postgres'::regrole "
        f"and d.defaclnamespace = 'public'::regnamespace and d.defaclobjtype = '{objtype}'"
    )
    assert len(want) == 1
    db = clone_db("default_acl")
    db.execute(ddl, user="postgres")
    column = "proacl" if objtype == "f" else "relacl"
    assert db.rows(acl_query.format(e=_EXPLODED.format(acl=column))) == want


def test_clone_is_independent_and_uniquely_named(base_db, clone_db):
    a, b = clone_db("same label"), clone_db("same label")
    assert a.db != b.db and a.db.startswith(_pg.CLONE_PREFIX) and len(a.db) <= 63
    a.execute("insert into public.schema_migrations(filename) values ('zz/clone_probe.sql')")
    probe = "select count(*) from public.schema_migrations where filename = 'zz/clone_probe.sql'"
    assert a.rows(probe) == ["1"]
    assert b.rows(probe) == ["0"]
    assert base_db.rows(probe) == ["0"]
    with pytest.raises(_pg.DbFixtureError, match="not a clone"):
        _pg.drop(base_db)


def test_postgres_role_reaches_the_copy_by_exec_and_by_client(base_db):
    assert base_db.rows("select current_user", user="postgres") == ["postgres"]
    with base_db.rolled_back() as conn:
        assert conn.execute("select current_user, current_database()").fetchone() == (
            "postgres",
            _pg.BASE_DB,
        )


def test_migration_helper_follows_the_runner_branches(clone_db, tmp_path):
    db = clone_db("runner_branches")
    wrapped = tmp_path / "900_wrapped.sql"
    wrapped.write_text(
        "create table public.zz_wrapped (id int);\n-- ALTER TYPE x ADD VALUE in a comment only\nselect 1/0;\n"
    )
    with pytest.raises(_pg.DbFixtureError):
        _pg.apply_migration(db, wrapped)
    # Wrapped: the failing statement rolled back the CREATE TABLE before it.
    assert db.rows("select to_regclass('public.zz_wrapped') is null") == ["t"]

    unwrapped = tmp_path / "901_unwrapped.sql"
    unwrapped.write_text(
        "do $$ begin create type public.zz_enum as enum ('a'); exception when duplicate_object then null; end $$;\n"
        "alter type public.zz_enum add value if not exists 'b';\n"
    )
    assert _pg.runner_unwraps(unwrapped.read_text()) is True
    assert _pg.apply_migration(db, unwrapped) == "unwrapped"
    assert db.rows("select string_agg(enumlabel, ',' order by enumsortorder) from pg_enum "
                   "where enumtypid = 'public.zz_enum'::regtype") == ["a,b"]  # fmt: skip
    assert _pg.runner_unwraps(wrapped.read_text()) is False


def test_migrate_rejects_unknown_or_missing_lane_files(clone_db):
    db = clone_db("migrate_guard")
    with pytest.raises(ValueError, match="unknown lane migration"):
        _pg.migrate(db, "ml/999_nope.sql")
    assert _pg.migrate(db, None) == []


def test_reap_removes_only_containers_whose_owner_is_gone(prod_readonly):
    import secrets

    image = prod_readonly.image()
    dead = _pg.CONTAINER_PREFIX + "reapdead" + secrets.token_hex(3)
    alive = _pg.CONTAINER_PREFIX + "reapalive" + secrets.token_hex(3)
    try:
        for name, pid in ((dead, "999999999"), (alive, str(os.getpid()))):
            subprocess.run(
                ["docker", "create", "--name", name, "--label", f"{_pg.OWNER_LABEL}={pid}", image],
                capture_output=True,
                check=True,
            )
        removed = _pg.reap_orphans()
        assert dead in removed and alive not in removed
        assert subprocess.run(["docker", "inspect", dead], capture_output=True).returncode != 0
        assert subprocess.run(["docker", "inspect", alive], capture_output=True).returncode == 0
    finally:
        subprocess.run(["docker", "rm", "-f", dead, alive], capture_output=True)
