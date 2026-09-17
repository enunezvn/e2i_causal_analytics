"""The throwaway database is a faithful copy of prod for everything the learning loop touches.

Every later real-DB test in this package (the recording RPCs, the registry sync client, the
runner rehearsal of pending migrations) runs against a clone of ``base_db`` or of the deployed
template built from it, so this module proves the copy before anything depends on it:

* it lives in a throwaway, memory-capped container of prod's own image, bound to 127.0.0.1, and
  the live ``supabase-db`` is only ever read (spec §9, dispatcher constraint 2026-09-11);
* its schema came from a schema-only dump of prod's ``public`` schema, and every restore error
  is one expected by name;
* its migration ledger is prod's, and what is pending is derived from it (#2065);
* its tool_registry / tool_dependencies rows are the running code's, written through
  ``sync_tool_registry`` the way the API's startup sync writes prod's (no prod data rows);
* the deployed template is the base with exactly the pending migrations applied;
* the lane's objects equal prod's in columns, constraints, indexes, ownership, RLS, policies,
  ACLs (object and default, by effective result), view and function bodies, triggers, enums,
  extensions, event triggers, role attributes, memberships and settings;
* the migration helper applies a file the way scripts/run_migrations.sh does.

Opt-in: ``E2I_DB_INTEGRATION=1`` (skipped in CI, which has no database; runs in every deploy).
Run with ``-n 0``.
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
        reason=_pg.OPT_IN_SKIP_REASON,
    ),
    # The session fixture starts a container and restores a schema dump inside the first test's
    # setup; the repo-wide 30 s pytest-timeout would cut that off on a loaded box.
    pytest.mark.timeout(300),
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
        f"select c.relname || ':' || coalesce({_pg.acl_text('c.relacl')}, '<owner-default>') from pg_class c "
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
        f"select t.relname || ':' || s.relname || ':' || coalesce({_pg.acl_text('s.relacl')}, '<owner-default>') "
        "from pg_depend d "
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
        f"|| pg_get_userbyid(p.proowner) || ':' || coalesce({_pg.acl_text('p.proacl')}, '<owner-default>') "
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
        f"|| ':' || {_pg.acl_text('defaclacl')} from pg_default_acl "
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
        f"select nspowner::regrole::text || ':' || {_pg.acl_text('nspacl')} from pg_namespace "
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


READ_ONLY_PROBE = "show transaction_read_only"
REGISTRY_AS_PAYLOAD = (
    "select coalesce(jsonb_agg(jsonb_build_object('name', name, 'description', description, "
    "'category', category, 'source_agent', source_agent, 'input_schema', input_schema, "
    "'output_schema', output_schema, 'avg_latency_ms', avg_latency_ms, 'version', version) "
    "order by name), '[]') from tool_registry where deprecated_at is null"
)
DEPENDENCIES_AS_PAYLOAD = (
    "select coalesce(jsonb_agg(jsonb_build_object('consumer', c.name, 'producer', p.name, "
    "'output_field', d.output_field, 'input_field', d.input_field) order by c.name, p.name), '[]') "
    "from tool_dependencies d join tool_registry c on c.tool_id = d.consumer_tool_id "
    "join tool_registry p on p.tool_id = d.producer_tool_id"
)
REQUIRED_OBJECTS_QUERY = (
    "select c.relkind::text || ':' || c.relname from pg_class c "
    f"where {_PUBLIC_REL} and c.relname in {_in(LANE_TABLES + LANE_VIEWS)} "
    "union all select 'f:' || p.proname from pg_proc p "
    f"where p.pronamespace = 'public'::regnamespace and p.proname in {_in(LANE_FUNCTIONS)} "
    f"union all select 'e:' || t.typname from pg_type t where t.typname in {_in(LANE_ENUMS)} order by 1"
)


@pytest.fixture(scope="module", autouse=True)
def _approve_this_modules_prod_queries(request):
    if os.getenv("E2I_DB_INTEGRATION") != "1":
        return
    prod = request.getfixturevalue("prod_readonly")
    prod.approve(
        READ_ONLY_PROBE,
        REQUIRED_OBJECTS_QUERY,
        *(query for query, _ in EQUIVALENCE_QUERIES.values()),
        *(_effective_default_acl_query(t) for t in PROBES),
    )


@pytest.fixture(scope="module")
def prod_state_db(base_db, deployed_db):
    """The copy whose ledger is prod's, for comparing schema with prod.

    That is the base, except under E2I_DB_SIMULATE_PENDING: the base then lacks the simulated
    migration on purpose, and the deployed template (base + that migration) is prod's state.
    """
    return deployed_db if base_db.build.simulated else base_db


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
    assert info["Config"]["Labels"][_pg.OWNER_LABEL] == _pg.owner_identity()
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
        "select pg_terminate_backend(pid) from pg_stat_activity where pid <> pg_backend_pid()",
        "select nextval('some_seq')",
        "select pg_advisory_lock(1)",
        'select "pg_terminate_backend"(123)',
        "select pg_terminate_backend/**/(123)",
        "select 1 -- ; commit",
        "select $$x$$",
        "select query_to_xml('select pg_' || 'terminate_backend(123)', false, false, '')",
        "select cursor_to_xml(null, 1, false, false, '')",
        "select public.some_function(1)",
        "select lower(relname) from pg_class",
    ],
)
def test_prod_access_refuses_anything_but_approved_side_effect_free_reads(sql):
    prod = _pg.ProdReadOnly()
    with pytest.raises(ValueError, match="refused before reaching prod"):
        prod.rows(sql)
    with pytest.raises(ValueError, match="refused before reaching prod"):
        prod.approve(sql)


def test_prod_access_refuses_an_unapproved_read():
    with pytest.raises(ValueError, match="not an approved prod query"):
        _pg.ProdReadOnly().rows("select count(*) from tool_registry")


def test_prod_reads_run_in_a_read_only_transaction(prod_readonly):
    assert prod_readonly.rows(READ_ONLY_PROBE) == ["on"]


def test_restore_errors_were_exactly_the_expected_ones(base_db):
    assert base_db.restore_log.unexpected == []
    assert base_db.restore_log.unseen_expected == []


def test_ledger_is_prods_and_pending_is_derived_from_it(base_db, prod_readonly, pending_migrations):
    # Sorted in Python: the database collation orders punctuation differently from byte order.
    prod = sorted(prod_readonly.rows(_pg.PROD_LEDGER))
    assert prod, "prod's ledger is empty; the derivation would be vacuous"
    simulated = base_db.build.simulated
    assert sorted(base_db.rows(_pg.PROD_LEDGER)) == sorted(k for k in prod if k != simulated)
    # In runner order, and exactly what the runner itself would call pending on the copy.
    assert pending_migrations == _pg.derive_pending(_pg.runner_migration_keys(), prod, simulated)
    if simulated is not None:
        assert simulated in pending_migrations


def test_tool_registry_is_the_running_codes(base_db):
    """The rows are what the app's own sync writes, not a copy of prod's rows.

    Prod's rows are NOT the expectation: during a deploy prod still runs the previous code, so a
    change to a tool's schema would read as a fixture fault on exactly the deploy that ships it.
    """
    from src.agents.tool_composer.registry_sync import build_sync_payload

    tools, dependencies = build_sync_payload()
    assert tools and dependencies
    rejected = set(base_db.build.registry_rejected)
    if not base_db.build.pending:
        assert rejected == set(), "with nothing pending the schema must accept every tool"
    tools = [t for t in tools if t["name"] not in rejected]
    dependencies = [d for d in dependencies if not {d["consumer"], d["producer"]} & rejected]
    with base_db.rolled_back() as conn:
        assert conn.execute(REGISTRY_AS_PAYLOAD).fetchone()[0] == tools
        stored = conn.execute(DEPENDENCIES_AS_PAYLOAD).fetchone()[0]
    key = lambda d: (d["consumer"], d["producer"])  # noqa: E731
    assert sorted(stored, key=key) == sorted(dependencies, key=key)
    assert base_db.build.registry_sync["inserted"] == len(tools)
    assert base_db.rows("select count(*) from tool_registry where deprecated_at is not null") == [
        "0"
    ]


def _empty_registry(db):
    db.execute("delete from tool_dependencies")
    db.execute("delete from tool_registry")


def test_registry_sync_never_drops_a_tool_unless_asked(clone_db):
    """Behaviour mode, and the deployed template: a sync the schema refuses is a failure."""
    db = clone_db("sync_strict")
    _empty_registry(db)
    # Stands in for a schema BEFORE a pending migration that the code's COHORT tools need.
    db.execute(
        "alter table tool_registry add constraint zz_pre_migration check (category::text <> 'COHORT')"
    )
    with pytest.raises(_pg.DbFixtureError, match="zz_pre_migration"):
        _pg.sync_registry_from_code(db)


def test_upgrade_base_gets_every_tool_the_pre_migration_schema_can_hold(clone_db):
    """Upgrade mode: production migrates BEFORE the new code syncs, so a tool that needs a
    pending migration cannot be in the pre-upgrade registry, and must not fail the base."""
    from src.agents.tool_composer.registry_sync import build_sync_payload

    tools, dependencies = build_sync_payload()
    cohort = sorted(t["name"] for t in tools if t["category"] == "COHORT")
    assert cohort, "the code registers no COHORT tool; this probe would be vacuous"
    db = clone_db("sync_tolerant")
    _empty_registry(db)
    db.execute(
        "alter table tool_registry add constraint zz_pre_migration check (category::text <> 'COHORT')"
    )

    counts, rejected = _pg.sync_registry_from_code(db, drop_rejected=True)

    assert rejected == cohort
    assert counts["inserted"] == len(tools) - len(cohort)
    kept = {t["name"] for t in tools} - set(cohort)
    stored = set(db.rows("select name from tool_registry"))
    assert stored == kept
    expected_deps = [d for d in dependencies if d["consumer"] in kept and d["producer"] in kept]
    assert db.rows("select count(*) from tool_dependencies") == [str(len(expected_deps))]


def test_deployed_template_is_the_base_plus_exactly_the_pending_migrations(
    base_db, deployed_db, pending_migrations
):
    if not pending_migrations:
        assert deployed_db.db == base_db.db
        return
    assert deployed_db.db == _pg.DEPLOYED_DB
    assert sorted(deployed_db.rows(_pg.PROD_LEDGER)) == sorted(
        base_db.rows(_pg.PROD_LEDGER) + pending_migrations
    )
    assert _pg.derive_pending(_pg.runner_migration_keys(), deployed_db.rows(_pg.PROD_LEDGER)) == []


def test_loop_tables_start_empty(base_db):
    assert base_db.rows(
        "select (select count(*) from tool_performance) || '|' || (select count(*) from composer_episodes) "
        "|| '|' || (select count(*) from composition_steps)"
    ) == ["0|0|0"]


def test_required_lane_objects_exist_on_both_sides(prod_state_db, prod_readonly):
    # LANE_FUNCTIONS names functions as they were BEFORE the lane (the rollbacks restore them);
    # 040/041 dropped four of them, so only the copy's agreement with prod is required of those.
    required = {f"r:{t}" for t in LANE_TABLES} | {f"v:{v}" for v in LANE_VIEWS}
    required |= {f"e:{e}" for e in LANE_ENUMS}
    copy = sorted(prod_state_db.rows(REQUIRED_OBJECTS_QUERY))
    assert copy == sorted(prod_readonly.rows(REQUIRED_OBJECTS_QUERY))
    assert required <= set(copy), sorted(required - set(copy))
    assert any(row.startswith("f:") for row in copy), "no lane function on either side"


@pytest.mark.parametrize("aspect", sorted(EQUIVALENCE_QUERIES))
def test_schema_equivalent_for_lane_objects(prod_state_db, prod_readonly, aspect):
    query, may_be_empty = EQUIVALENCE_QUERIES[aspect]
    got = prod_state_db.rows(query)
    want = prod_readonly.rows(query)
    if not may_be_empty:
        assert want, f"{aspect}: prod returned nothing, so equality would be vacuous"
    assert got == want


def test_non_public_objects_public_depends_on_exist_in_the_copy(prod_state_db, prod_readonly):
    refs = prod_readonly.rows(_pg.PROD_NONPUBLIC_REFERENCES)
    assert refs, "public objects reference no non-public object; the check would be vacuous"
    missing = []
    for ref in refs:
        kind, _, name = ref.partition(":")
        fn = {"rel": "to_regclass", "proc": "to_regprocedure", "type": "to_regtype"}[kind]
        if prod_state_db.rows(f"select {fn}('{name}') is not null") != ["t"]:
            missing.append(ref)
    assert missing == []


# pg_default_acl object type -> acldefault() object type (sequences differ: 'S' vs 's').
_ACLDEFAULT_TYPE = {"r": "r", "S": "s", "f": "f"}


def _effective_default_acl_query(objtype: str) -> str:
    # PG15 (aclchk.c get_user_default_acl): a database-wide (namespace 0) default ACL row for the
    # creating role REPLACES the built-in acldefault(); a schema row is then ADDED.
    return "select " + _pg.acl_text(
        "coalesce((select defaclacl from pg_default_acl where defaclrole = 'postgres'::regrole "
        f"and defaclnamespace = 0 and defaclobjtype = '{objtype}'), "
        f"acldefault('{_ACLDEFAULT_TYPE[objtype]}', 'postgres'::regrole)) "
        "|| coalesce((select defaclacl from pg_default_acl where defaclrole = 'postgres'::regrole "
        f"and defaclnamespace = 'public'::regnamespace and defaclobjtype = '{objtype}'), '{{}}'::aclitem[])"
    )


PROBES = {
    "r": ("create table public.zz_default_acl_probe (id int)",
          "select {acl} from pg_class where oid = 'public.zz_default_acl_probe'::regclass", "relacl"),
    "S": ("create sequence public.zz_default_acl_probe_seq",
          "select {acl} from pg_class where oid = 'public.zz_default_acl_probe_seq'::regclass", "relacl"),
    "f": ("create function public.zz_default_acl_probe_fn() returns int language sql as 'select 1'",
          "select {acl} from pg_proc where oid = 'public.zz_default_acl_probe_fn()'::regprocedure", "proacl"),
}  # fmt: skip


@pytest.mark.parametrize("objtype", sorted(PROBES))
def test_new_objects_get_prods_effective_default_privileges(clone_db, prod_readonly, objtype):
    # What a migration running as postgres actually gets on a new public object must equal what
    # prod's default ACLs prescribe for it (the model above; namespace-0 rows are covered too).
    want = prod_readonly.rows(_effective_default_acl_query(objtype))
    assert len(want) == 1 and want[0]
    ddl, acl_query, column = PROBES[objtype]
    db = clone_db("default_acl")
    db.execute(ddl, user="postgres")
    assert db.rows(acl_query.format(acl=_pg.acl_text(column))) == want


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


def test_migrate_skips_ledgered_keys_and_rejects_unknown_ones(clone_db):
    db = clone_db("migrate_guard")
    with pytest.raises(ValueError, match="not a runner migration key"):
        _pg.migrate(db, "ml/999_nope.sql")
    assert _pg.migrate(db, None) == []
    # The copy is post-deploy: every runner key is in its ledger, so nothing is re-applied.
    before = db.rows("select count(*) from public.schema_migrations")
    assert _pg.migrate(db, "ml/041_composer_learning_loop_recording.sql") == []
    assert _pg.migrate(db, _pg.ALL_PENDING) == []
    assert db.rows("select count(*) from public.schema_migrations") == before


def test_migrate_applies_and_records_an_unapplied_key_like_the_runner(clone_db):
    db = clone_db("migrate_records")
    key = "ml/044_composer_episodes_feedback_id.sql"
    assert _pg.rollback_file(key).exists()
    assert _pg.apply_rollback(db, _pg.rollback_file(key).name).returncode == 0
    assert db.rows(f"select count(*) from public.schema_migrations where filename = '{key}'") == [
        "0"
    ]
    assert _pg.migrate(db, key) == [key]
    assert db.rows(f"select count(*) from public.schema_migrations where filename = '{key}'") == [
        "1"
    ]
    assert db.rows(
        "select count(*) from pg_attribute where attrelid = 'composer_episodes'::regclass "
        "and attname = 'feedback_id' and not attisdropped"
    ) == ["1"]


def test_owner_is_gone_needs_the_same_host_boot_and_pid_namespace():
    me = _pg.owner_identity()
    host, boot_id, pid_ns, pid, start = me.split("/")
    assert _pg.owner_is_gone(me) is False
    # Same host, boot and namespace, but the PID's start time differs: that process is gone.
    assert _pg.owner_is_gone(f"{host}/{boot_id}/{pid_ns}/{pid}/0") is True
    # Another host, boot or PID namespace: may be a live session we cannot see; never reaped.
    assert _pg.owner_is_gone(f"other-host/{boot_id}/{pid_ns}/{pid}/0") is False
    assert _pg.owner_is_gone(f"{host}/other-boot/{pid_ns}/{pid}/0") is False
    assert _pg.owner_is_gone(f"{host}/{boot_id}/1/{pid}/0") is False
    assert _pg.owner_is_gone("12345") is False
    assert _pg.owner_is_gone("") is False
    # A live PID with a malformed start field is uncertain, never proof of death.
    assert _pg.owner_is_gone(f"{host}/{boot_id}/{pid_ns}/{pid}/") is False
    assert _pg.owner_is_gone(f"{host}/{boot_id}/{pid_ns}/{pid}/abc") is False
    assert _pg.owner_is_gone(f"{host}/{boot_id}//{pid}/{start}") is False


def test_acl_rendering_merges_grant_options_like_postgres(base_db):
    # The same (grantee, privilege, grantor) once plain and once WITH GRANT OPTION renders as one
    # grantable item, as PostgreSQL's ACL merge would produce.
    rendered = base_db.rows(
        "select "
        + _pg.acl_text("array['anon=r/postgres'::aclitem] || array['anon=r*/postgres'::aclitem]")
    )
    assert rendered == ["anon=SELECT*/postgres"]


def test_reap_removes_only_containers_whose_owner_is_provably_gone(prod_readonly):
    import secrets

    image = prod_readonly.image()
    host, boot_id, pid_ns, pid, _ = _pg.owner_identity().split("/")
    labelled = {
        _pg.CONTAINER_PREFIX
        + "reapdead"
        + secrets.token_hex(3): f"{host}/{boot_id}/{pid_ns}/{pid}/0",
        _pg.CONTAINER_PREFIX + "reapalive" + secrets.token_hex(3): _pg.owner_identity(),
        _pg.CONTAINER_PREFIX
        + "reapforeign"
        + secrets.token_hex(3): f"other-host/{boot_id}/{pid_ns}/1/0",
    }
    try:
        for name, owner in labelled.items():
            subprocess.run(
                [
                    "docker",
                    "create",
                    "--name",
                    name,
                    "--label",
                    f"{_pg.OWNER_LABEL}={owner}",
                    image,
                ],
                capture_output=True,
                check=True,
            )
        removed = set(_pg.reap_orphans())
        dead, alive, foreign = labelled
        assert dead in removed and alive not in removed and foreign not in removed
        assert subprocess.run(["docker", "inspect", dead], capture_output=True).returncode != 0
        for survivor in (alive, foreign):
            assert (
                subprocess.run(["docker", "inspect", survivor], capture_output=True).returncode == 0
            )
    finally:
        subprocess.run(["docker", "rm", "-f", *labelled], capture_output=True)
