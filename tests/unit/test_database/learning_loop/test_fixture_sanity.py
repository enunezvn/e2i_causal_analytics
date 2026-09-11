"""The throwaway database is a faithful copy of prod for everything the learning loop touches.

Every later real-DB test in this package (migrations ml/039–041, the recording RPCs, the
registry sync client) runs against ``base_db`` or a clone of it, so this module proves the
copy before anything depends on it:

* the copy lives in a throwaway, memory-capped container of prod's own image, bound to
  127.0.0.1, and the live ``supabase-db`` is only ever read (spec §9, dispatcher constraint
  2026-09-11);
* the schema came from a schema-only dump of prod's ``public`` schema, and every restore error
  is one we expected by name;
* the rows the lane needs were rebuilt from the repository (the migration ledger, and the
  tool_registry / tool_dependencies seeds of ml/013, 027 and 037) and equal prod's;
* constraints, indexes, ACLs, function bodies and enum labels of the lane's objects equal prod's.

Opt-in: ``E2I_DB_INTEGRATION=1`` (skipped in CI, which has no database). Run with ``-n 0``.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

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
    "tool_registry",
    "tool_dependencies",
    "tool_performance",
    "composer_episodes",
    "composition_steps",
    "classification_logs",
    "episodic_memories",
    "schema_migrations",
]
LANE_VIEWS = [
    "v_tool_reliability",
    "v_composition_success_rate",
    "v_active_compositions",
    "v_classification_accuracy",
]
LANE_FUNCTIONS = [
    "update_tool_registry_metrics",
    "find_similar_compositions",
    "get_tool_execution_order",
    "trigger_log_step_performance",
    "trigger_update_tool_registry_timestamp",
]
LANE_ENUMS = ["tool_category", "composition_status", "routing_pattern", "e2i_agent_name"]


def _in(names: list[str]) -> str:
    return "(" + ",".join("'" + n + "'" for n in names) + ")"


EQUIVALENCE_QUERIES = {
    "constraints": (
        "select conrelid::regclass || ':' || conname || ':' || pg_get_constraintdef(oid) "
        f"from pg_constraint where conrelid::regclass::text in {_in(LANE_TABLES)} order by 1"
    ),
    "indexes": (
        "select indexrelid::regclass || ':' || pg_get_indexdef(indexrelid) from pg_index "
        f"where indrelid::regclass::text in {_in(LANE_TABLES)} order by 1"
    ),
    "relation_acls": (
        "select c.relname || ':' || c.relkind::text || ':' || coalesce(c.relacl::text, '') from pg_class c "
        "where c.relnamespace = 'public'::regnamespace "
        f"and c.relname in {_in(LANE_TABLES + LANE_VIEWS)} order by 1"
    ),
    "view_definitions": (
        "select c.relname || ':' || md5(pg_get_viewdef(c.oid)) from pg_class c "
        f"where c.relnamespace = 'public'::regnamespace and c.relname in {_in(LANE_VIEWS)} order by 1"
    ),
    "functions": (
        "select p.oid::regprocedure || ':' || md5(pg_get_functiondef(p.oid)) || ':' "
        "|| coalesce(p.proacl::text, '') from pg_proc p "
        f"where p.pronamespace = 'public'::regnamespace and p.proname in {_in(LANE_FUNCTIONS)} "
        "order by 1"
    ),
    "triggers": (
        "select tgrelid::regclass || ':' || tgname || ':' || pg_get_triggerdef(oid) from pg_trigger "
        f"where not tgisinternal and tgrelid::regclass::text in {_in(LANE_TABLES)} order by 1"
    ),
    "enums": (
        "select t.typname || ':' || string_agg(e.enumlabel, ',' order by e.enumsortorder) "
        "from pg_type t join pg_enum e on e.enumtypid = t.oid "
        f"where t.typname in {_in(LANE_ENUMS)} group by t.typname order by 1"
    ),
    "lane_extensions": (
        "select extname || ':' || extversion || ':' || extnamespace::regnamespace from pg_extension "
        "where extname in ('vector', 'pgcrypto', 'uuid-ossp') order by 1"
    ),
    "default_acls_for_public_objects": (
        "select defaclrole::regrole || ':' || defaclobjtype::text || ':' || defaclacl::text "
        "from pg_default_acl where defaclnamespace in (0, 'public'::regnamespace) order by 1"
    ),
}


def test_container_is_throwaway_capped_and_local(pg_container):
    info = json.loads(
        subprocess.run(
            ["docker", "inspect", pg_container.name], capture_output=True, text=True, check=True
        ).stdout
    )[0]
    assert pg_container.name.startswith("e2i-learnloop-pg-")
    assert pg_container.name != "supabase-db"
    assert info["HostConfig"]["Memory"] == 1024**3
    assert info["HostConfig"]["MemorySwap"] == 1024**3
    bindings = info["HostConfig"]["PortBindings"]["5432/tcp"]
    assert [b["HostIp"] for b in bindings] == ["127.0.0.1"]
    prod_image = subprocess.run(
        ["docker", "inspect", "supabase-db", "--format", "{{.Config.Image}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert info["Config"]["Image"] == prod_image


def test_prod_session_is_read_only(prod_readonly):
    assert prod_readonly.rows("show default_transaction_read_only") == ["on"]


def test_restore_reported_no_unexpected_errors(base_db):
    assert base_db.restore_log.unexpected == [], base_db.restore_log.unexpected
    # The expectation dict is exact, not a pattern: every expected error was actually seen.
    assert base_db.restore_log.unseen_expected == [], base_db.restore_log.unseen_expected


def test_ledger_equals_prod(base_db, prod_readonly):
    query = "select filename from public.schema_migrations order by 1"
    assert base_db.rows(query) == prod_readonly.rows(query)


def test_tool_registry_rows_equal_prod(base_db, prod_readonly):
    query = (
        "select name || '|' || category || '|' || source_agent || '|' || md5(description) || '|' "
        "|| md5(input_schema::text) || '|' || md5(output_schema::text) || '|' "
        "|| coalesce(avg_latency_ms::text, '') || '|' || coalesce(success_rate::text, '') || '|' "
        "|| coalesce(version, '') || '|' || composable from tool_registry order by name"
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


def test_loop_tables_start_empty_like_prod(base_db, prod_readonly):
    query = (
        "select (select count(*) from tool_performance) || '|' || "
        "(select count(*) from composer_episodes) || '|' || (select count(*) from composition_steps)"
    )
    assert base_db.rows(query) == prod_readonly.rows(query) == ["0|0|0"]


@pytest.mark.parametrize("aspect", sorted(EQUIVALENCE_QUERIES))
def test_schema_equivalent_for_lane_objects(base_db, prod_readonly, aspect):
    query = EQUIVALENCE_QUERIES[aspect]
    got = base_db.rows(query)
    assert got, f"{aspect}: the query returned nothing, so equality would be vacuous"
    assert got == prod_readonly.rows(query)


def test_clone_is_independent_of_base(base_db, clone_db):
    clone = clone_db("sanity")
    clone.execute("insert into public.schema_migrations(filename) values ('zz/clone_probe.sql')")
    assert clone.rows(
        "select count(*) from public.schema_migrations where filename = 'zz/clone_probe.sql'"
    ) == ["1"]
    assert base_db.rows(
        "select count(*) from public.schema_migrations where filename = 'zz/clone_probe.sql'"
    ) == ["0"]


def test_role_attributes_equal_prod(base_db, prod_readonly):
    # Migrations run as postgres (not a superuser on prod); a superuser postgres in the copy would
    # let a migration pass here that fails on deploy.
    query = (
        "select rolname || ':' || rolsuper || rolcreaterole || rolcreatedb || rolbypassrls || rolinherit "
        "from pg_roles where rolname in ('postgres','supabase_admin','anon','authenticated',"
        "'service_role','authenticator') order by 1"
    )
    rows = base_db.rows(query)
    assert len(rows) == 6
    assert rows == prod_readonly.rows(query)


def test_postgres_role_reaches_the_copy_by_docker_exec_and_url(base_db):
    assert base_db.rows("select current_user", user="postgres") == ["postgres"]
    with base_db.connect(connect_timeout=5) as conn:
        assert conn.execute("select current_user, current_database()").fetchone() == (
            "postgres",
            "learning_loop_base",
        )
