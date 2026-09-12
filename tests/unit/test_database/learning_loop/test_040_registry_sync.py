"""ml/039 (COHORT category) and ml/040 (``sync_tool_registry``) on a prod-faithful copy.

The sync replaces the migration-per-schema-change regime (spec §4): the API calls
``sync_tool_registry(p_tools, p_dependencies, p_max_deprecations)`` at startup, and the DB
registry becomes exactly the running code's tools. These tests pin the function's contract:
idempotent counts, pre-write guards, atomicity, serialisation of concurrent callers, and
service_role-only access. They also pin 040's schema changes (agent CHECK follows the
``e2i_agent_name`` taxonomy; the unguarded metrics writer, its column and the execution-order
function are gone — owner decision O3).

Opt-in: ``E2I_DB_INTEGRATION=1``. Run with ``-n 0``.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Tuple

import pytest

from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(
        not _pg.db_integration_enabled(),
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    pytest.mark.timeout(300),
]

UPTO = "ml/040_tool_registry_startup_sync.sql"
SYNC = "sync_tool_registry(jsonb,jsonb,integer)"

# Four tools the code registers that were never seeded (spec §2.1, §9 cert item 1).
NEW_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "cohort_builder",
        "description": "Build a patient cohort from eligibility criteria",
        "category": "COHORT",
        "source_agent": "cohort_constructor",
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object", "properties": {}},
        "avg_latency_ms": 1500.0,
        "version": "1.0.0",
    },
    {
        "name": "cohort_validator",
        "description": "Validate a cohort definition",
        "category": "COHORT",
        "source_agent": "cohort_constructor",
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object", "properties": {}},
        "avg_latency_ms": 800.0,
        "version": "1.0.0",
    },
    {
        "name": "cohort_statistics",
        "description": "Profile a cohort",
        "category": "COHORT",
        "source_agent": "cohort_profiler",
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object", "properties": {}},
        "avg_latency_ms": 900.0,
        "version": "1.0.0",
    },
    {
        "name": "model_inference",
        "description": "Score rows with a deployed model",
        "category": "PREDICTION",
        "source_agent": "prediction_synthesizer",
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object", "properties": {}},
        "avg_latency_ms": 2000.0,
        "version": "1.0.0",
    },
]

ZERO = {
    "inserted": 0,
    "updated": 0,
    "deprecated": 0,
    "dependencies_upserted": 0,
    "dependencies_deleted": 0,
}

TOOLS_AS_PAYLOAD = """
select coalesce(jsonb_agg(jsonb_build_object(
    'name', name, 'description', description, 'category', category, 'source_agent', source_agent,
    'input_schema', input_schema, 'output_schema', output_schema,
    'avg_latency_ms', avg_latency_ms, 'version', version) order by name), '[]'::jsonb)
from tool_registry where deprecated_at is null
"""
DEPENDENCIES_AS_PAYLOAD = """
select coalesce(jsonb_agg(jsonb_build_object(
    'consumer', c.name, 'producer', p.name, 'output_field', d.output_field,
    'input_field', d.input_field) order by c.name, p.name), '[]'::jsonb)
from tool_dependencies d
join tool_registry c on c.tool_id = d.consumer_tool_id
join tool_registry p on p.tool_id = d.producer_tool_id
"""


def _sync(conn, tools, deps, max_deprecations: int = 3) -> Dict[str, int]:
    (result,) = conn.execute(
        "select sync_tool_registry(%s::jsonb, %s::jsonb, %s)",
        (json.dumps(tools), json.dumps(deps), max_deprecations),
    ).fetchone()
    return result


def _current_payload(conn) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    (tools,) = conn.execute(TOOLS_AS_PAYLOAD).fetchone()
    (deps,) = conn.execute(DEPENDENCIES_AS_PAYLOAD).fetchone()
    return tools, deps


@pytest.fixture
def migrated(module_db) -> _pg.PgConn:
    """The module's clone with 039 + 040 applied; tests isolate with ``rolled_back()``."""
    return module_db(UPTO)


@pytest.fixture
def fresh(clone_db) -> _pg.PgConn:
    """A clone of its own with 039 + 040 applied, for tests that must commit."""
    db = clone_db("sync_commit")
    _pg.migrate(db, UPTO)
    return db


# ---------------------------------------------------------------------------
# 039 + 040 schema
# ---------------------------------------------------------------------------


def test_cohort_category_insertable_after_039_commit(migrated):
    with migrated.rolled_back() as conn:
        conn.execute(
            "insert into tool_registry (name, description, category, source_agent) "
            "values ('cohort_builder', 'd', 'COHORT', 'cohort_constructor')"
        )
        (category,) = conn.execute(
            "select category::text from tool_registry where name = 'cohort_builder'"
        ).fetchone()
    assert category == "COHORT"


def test_valid_agent_follows_e2i_agent_name(migrated):
    import psycopg

    with migrated.rolled_back() as conn:
        with pytest.raises(psycopg.errors.CheckViolation, match="valid_agent"):
            conn.execute(
                "insert into tool_registry (name, description, category, source_agent) "
                "values ('typo_tool', 'd', 'CAUSAL', 'casual_impact')"
            )
    with migrated.rolled_back() as conn:
        conn.execute(
            "insert into tool_registry (name, description, category, source_agent) "
            "values ('profile_tool', 'd', 'COHORT', 'cohort_profiler')"
        )
    (definition,) = migrated.rows(
        "select pg_get_constraintdef(oid) from pg_constraint "
        "where conrelid = 'tool_registry'::regclass and conname = 'valid_agent'"
    )
    assert "e2i_agent_name" in definition


def test_success_rate_and_dead_functions_dropped(migrated):
    assert migrated.rows(
        "select count(*) from pg_attribute where attrelid = 'tool_registry'::regclass "
        "and attname = 'success_rate' and not attisdropped"
    ) == ["0"]
    assert migrated.rows(
        "select count(*) from pg_constraint where conrelid = 'tool_registry'::regclass "
        "and conname = 'tool_registry_success_rate_check'"
    ) == ["0"]
    assert migrated.rows(
        "select coalesce(to_regprocedure('update_tool_registry_metrics()')::text, 'NULL') || ','"
        " || coalesce(to_regprocedure('get_tool_execution_order(text[])')::text, 'NULL')"
    ) == ["NULL,NULL"]


def test_avg_latency_documented_as_declared(migrated):
    (comment,) = migrated.rows(
        "select col_description('tool_registry'::regclass, attnum) from pg_attribute "
        "where attrelid = 'tool_registry'::regclass and attname = 'avg_latency_ms'"
    )
    assert "DECLARED" in comment and "get_tool_reliability" in comment


# ---------------------------------------------------------------------------
# sync_tool_registry: counts and idempotency
# ---------------------------------------------------------------------------


def test_sync_first_call_then_idempotent(fresh):
    with fresh.connect() as conn:
        tools, deps = _current_payload(conn)
    assert len(tools) == 16 and len(deps) == 11
    payload = tools + NEW_TOOLS

    with fresh.connect() as conn:
        first = _sync(conn, payload, deps)
    assert first == {**ZERO, "inserted": 4}

    before = fresh.rows("select name || '|' || updated_at from tool_registry order by name")
    time.sleep(0.05)
    with fresh.connect() as conn:
        second = _sync(conn, payload, deps)
    assert second == ZERO
    assert fresh.rows("select name || '|' || updated_at from tool_registry order by name") == before

    with fresh.connect() as conn:
        after_tools, after_deps = _current_payload(conn)
    assert after_tools == sorted(payload, key=lambda t: t["name"])
    assert after_deps == deps


def test_sync_counts_use_xmax(fresh):
    with fresh.connect() as conn:
        tools, deps = _current_payload(conn)
    changed = [dict(t) for t in tools]
    changed[0]["description"] = changed[0]["description"] + " (changed)"
    before = dict(
        line.split("|", 1)
        for line in fresh.rows("select name || '|' || updated_at from tool_registry")
    )
    time.sleep(0.05)
    with fresh.connect() as conn:
        result = _sync(conn, changed, deps)
    assert result == {**ZERO, "updated": 1}
    after = dict(
        line.split("|", 1)
        for line in fresh.rows("select name || '|' || updated_at from tool_registry")
    )
    moved = sorted(name for name in before if before[name] != after[name])
    assert moved == [changed[0]["name"]]


def test_sync_updates_dependency_fields_and_replaces_the_set(fresh):
    with fresh.connect() as conn:
        tools, deps = _current_payload(conn)
    edited = [dict(d) for d in deps]
    edited[0]["output_field"] = "renamed_field"
    dropped = edited.pop()
    with fresh.connect() as conn:
        result = _sync(conn, tools, edited)
    assert result == {**ZERO, "dependencies_upserted": 1, "dependencies_deleted": 1}
    with fresh.connect() as conn:
        _, after = _current_payload(conn)
    assert after == edited
    assert dropped not in after


# ---------------------------------------------------------------------------
# Guards: validated before any write
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tools_sql, deps, message",
    [
        ("'[]'", "[]", "empty tool payload"),
        ("'null'", "[]", "empty tool payload"),
        ("'{}'", "[]", "empty tool payload"),
    ],
)
def test_sync_refuses_empty_payload(migrated, tools_sql, deps, message):
    import psycopg

    with migrated.rolled_back() as conn:
        with pytest.raises(psycopg.errors.RaiseException, match=message):
            conn.execute(f"select sync_tool_registry({tools_sql}::jsonb, %s::jsonb, 3)", (deps,))


@pytest.mark.parametrize("deps_sql", ["null", "'null'", "'{}'"])
def test_sync_refuses_non_array_dependencies(migrated, deps_sql):
    import psycopg

    with migrated.rolled_back() as conn:
        tools, _ = _current_payload(conn)
        with pytest.raises(
            psycopg.errors.RaiseException, match="dependency payload must be an array"
        ):
            conn.execute(
                f"select sync_tool_registry(%s::jsonb, {deps_sql}::jsonb, 3)", (json.dumps(tools),)
            )


@pytest.mark.parametrize("limit_sql", ["null", "-1"])
def test_sync_refuses_unbounded_deprecation_limit(migrated, limit_sql):
    import psycopg

    with migrated.rolled_back() as conn:
        tools, deps = _current_payload(conn)
        with pytest.raises(psycopg.errors.RaiseException, match="non-negative"):
            conn.execute(
                f"select sync_tool_registry(%s::jsonb, %s::jsonb, {limit_sql})",
                (json.dumps(tools[:2]), json.dumps([])),
            )


def test_sync_refuses_duplicate_dependency_pairs(migrated):
    import psycopg

    with migrated.rolled_back() as conn:
        tools, deps = _current_payload(conn)
        with pytest.raises(psycopg.errors.RaiseException, match="duplicate dependency pair"):
            _sync(conn, tools, deps + [dict(deps[0])])


def test_sync_refuses_duplicate_names(migrated):
    import psycopg

    with migrated.rolled_back() as conn:
        tools, deps = _current_payload(conn)
        with pytest.raises(psycopg.errors.RaiseException, match="duplicate tool name"):
            _sync(conn, tools + [dict(tools[0])], deps)


def test_sync_refuses_missing_name(migrated):
    import psycopg

    with migrated.rolled_back() as conn:
        tools, deps = _current_payload(conn)
        nameless = {k: v for k, v in NEW_TOOLS[0].items() if k != "name"}
        with pytest.raises(psycopg.errors.RaiseException, match="without a name"):
            _sync(conn, tools + [nameless], deps)


def test_sync_refuses_dependency_endpoint_not_in_payload(migrated):
    import psycopg

    with migrated.rolled_back() as conn:
        tools, deps = _current_payload(conn)
        orphan = deps + [
            {"consumer": "no_such_tool", "producer": tools[0]["name"], "output_field": "a"}
        ]
        with pytest.raises(psycopg.errors.RaiseException, match="no_such_tool"):
            _sync(conn, tools, orphan)


@pytest.mark.parametrize(
    "entry",
    [
        {"producer": "gap_calculator"},
        {"consumer": None, "producer": "gap_calculator"},
        {"consumer": 5, "producer": "gap_calculator"},
        {"consumer": "roi_estimator"},
        "roi_estimator<-gap_calculator",
    ],
)
def test_sync_refuses_dependency_without_endpoint_names(migrated, entry):
    import psycopg

    with migrated.rolled_back() as conn:
        tools, deps = _current_payload(conn)
        # A tool whose name is what a NULL endpoint used to be coerced to must not let a
        # nameless endpoint through.
        placeholder = {**NEW_TOOLS[0], "name": "<null>"}
        with pytest.raises(psycopg.errors.RaiseException, match="without a consumer and producer"):
            _sync(conn, tools + [placeholder], deps + [entry])


@pytest.mark.parametrize("name", [None, 5, ""])
def test_sync_refuses_tool_without_a_string_name(migrated, name):
    import psycopg

    with migrated.rolled_back() as conn:
        tools, deps = _current_payload(conn)
        with pytest.raises(psycopg.errors.RaiseException, match="without a name"):
            _sync(conn, tools + [{**NEW_TOOLS[0], "name": name}], deps)


def test_sync_duplicate_pair_detection_keeps_fields_apart(migrated):
    # (consumer 'a<-b', producer 'c') and (consumer 'a', producer 'b<-c') are different pairs
    # that a concatenated key would collide.
    with migrated.rolled_back() as conn:
        tools, deps = _current_payload(conn)
        names = ["a<-b", "c", "a", "b<-c"]
        odd = [{**NEW_TOOLS[3], "name": n} for n in names]
        pairs = [
            {"consumer": "a<-b", "producer": "c", "output_field": "x", "input_field": "y"},
            {"consumer": "a", "producer": "b<-c", "output_field": "x", "input_field": "y"},
        ]
        assert _sync(conn, tools + odd, deps + pairs) == {
            **ZERO,
            "inserted": 4,
            "dependencies_upserted": 2,
        }


WRITE_TRAP = """
create function pg_temp.learning_loop_write_trap() returns trigger language plpgsql as $t$
begin
    raise exception 'write trap: % reached %', tg_op, tg_table_name;
end $t$;
create trigger learning_loop_write_trap before insert or update or delete on tool_registry
    for each statement execute function pg_temp.learning_loop_write_trap();
create trigger learning_loop_write_trap before insert or update or delete on tool_dependencies
    for each statement execute function pg_temp.learning_loop_write_trap();
"""


def test_deprecation_guard_raises_before_any_dml(migrated):
    import psycopg

    with migrated.rolled_back() as conn:
        tools, _ = _current_payload(conn)
        conn.execute(WRITE_TRAP)
        with pytest.raises(psycopg.errors.RaiseException) as refused:
            _sync(conn, tools[:2], [])
    assert "would deprecate 14" in str(refused.value)
    assert "write trap" not in str(refused.value)

    # Positive control: the trap does fire on a payload that passes the guard, so the
    # assertion above cannot pass vacuously.
    with migrated.rolled_back() as conn:
        tools, deps = _current_payload(conn)
        conn.execute(WRITE_TRAP)
        with pytest.raises(psycopg.errors.RaiseException, match="write trap: INSERT reached"):
            _sync(conn, tools, deps)


def test_sync_refuses_more_than_max_deprecations_before_any_write(fresh):
    import psycopg

    with fresh.connect() as conn:
        tools, _ = _current_payload(conn)
    two = tools[:2]
    with fresh.connect() as conn:
        with pytest.raises(psycopg.errors.RaiseException, match=r"would deprecate 14 .*limit 3"):
            _sync(conn, two, [])
    assert fresh.rows(
        "select count(*) from tool_registry where deprecated_at is not null or not composable"
    ) == ["0"]
    assert fresh.rows("select count(*) from tool_dependencies") == ["11"]
    assert fresh.rows("select count(*) from tool_registry") == ["16"]


def test_sync_deprecates_within_limit_then_reactivates(fresh):
    with fresh.connect() as conn:
        tools, deps = _current_payload(conn)
    gone = "roi_estimator"
    touching = [d for d in deps if gone in (d["consumer"], d["producer"])]
    kept_tools = [t for t in tools if t["name"] != gone]
    kept_deps = [d for d in deps if d not in touching]

    with fresh.connect() as conn:
        result = _sync(conn, kept_tools, kept_deps)
    assert result == {**ZERO, "deprecated": 1, "dependencies_deleted": len(touching)}
    assert fresh.rows(
        f"select (deprecated_at is not null)::text || ',' || composable::text "
        f"from tool_registry where name = '{gone}'"
    ) == ["true,false"]
    assert fresh.rows(
        "select count(*) from tool_dependencies d join tool_registry t "
        f"on t.tool_id in (d.consumer_tool_id, d.producer_tool_id) where t.name = '{gone}'"
    ) == ["0"]

    with fresh.connect() as conn:
        back = _sync(conn, tools, deps)
    assert back == {**ZERO, "updated": 1, "dependencies_upserted": len(touching)}
    assert fresh.rows(
        f"select (deprecated_at is null)::text || ',' || composable::text "
        f"from tool_registry where name = '{gone}'"
    ) == ["true,true"]


@pytest.mark.parametrize(
    "bad_field, bad_value, error",
    [
        ("source_agent", "casual_impact", "CheckViolation"),
        ("category", "NOT_A_CATEGORY", "InvalidTextRepresentation"),
    ],
)
def test_sync_atomic_on_bad_row(fresh, bad_field, bad_value, error):
    import psycopg

    with fresh.connect() as conn:
        tools, deps = _current_payload(conn)
    before = fresh.rows("select name || '|' || description || '|' || updated_at from tool_registry")
    changed = [dict(t) for t in tools]
    changed[0]["description"] = "would be updated"
    bad = {**NEW_TOOLS[0], bad_field: bad_value}
    with fresh.connect() as conn:
        with pytest.raises(getattr(psycopg.errors, error)):
            _sync(conn, changed + [bad], deps)
    after = fresh.rows("select name || '|' || description || '|' || updated_at from tool_registry")
    assert sorted(after) == sorted(before)


# ---------------------------------------------------------------------------
# Concurrency: the advisory lock serialises callers
# ---------------------------------------------------------------------------


def test_sync_concurrent_calls_serialize(fresh):
    with fresh.connect() as conn:
        tools, deps = _current_payload(conn)
    payload = tools + NEW_TOOLS

    first = fresh.connect()
    second = fresh.connect()
    try:
        first_result = _sync(first, payload, deps)  # holds the xact lock until commit
        (second_pid,) = second.execute("select pg_backend_pid()").fetchone()
        outcome: Dict[str, Any] = {}

        def run_second() -> None:
            try:
                outcome["result"] = _sync(second, payload, deps)
                second.commit()
            except Exception as exc:  # surfaced by the assertion below
                outcome["error"] = exc

        thread = threading.Thread(target=run_second)
        thread.start()

        # The second caller must be waiting on the ADVISORY lock specifically (a row-lock
        # wait on the unique index would also block it, and would not prove the lock).
        waiting = False
        with fresh.connect(autocommit=True) as watcher:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                (n,) = watcher.execute(
                    "select count(*) from pg_locks where pid = %s and locktype = 'advisory' "
                    "and not granted",
                    (second_pid,),
                ).fetchone()
                if n == 1:
                    waiting = True
                    break
                time.sleep(0.1)
        assert waiting, "second sync_tool_registry call did not wait on the advisory lock"
        assert thread.is_alive()

        first.commit()
        thread.join(timeout=60)
        assert not thread.is_alive()
    finally:
        first.close()
        second.close()

    assert "error" not in outcome, outcome.get("error")
    assert first_result == {**ZERO, "inserted": 4}
    assert outcome["result"] == ZERO
    with fresh.connect() as conn:
        after_tools, after_deps = _current_payload(conn)
    assert after_tools == sorted(payload, key=lambda t: t["name"])
    assert after_deps == deps


def test_deprecation_guard_counts_after_the_lock(fresh):
    """The waiting caller's guard sees what the lock holder committed.

    The holder inserts 4 tools; the waiter's payload lacks them. Counted after the lock, the
    waiter would deprecate 4 > limit 3 and is refused. Counted before the lock, it would see 0
    and deprecate all 4 anyway.
    """
    import psycopg

    with fresh.connect() as conn:
        tools, deps = _current_payload(conn)

    holder = fresh.connect()
    waiter = fresh.connect()
    try:
        assert _sync(holder, tools + NEW_TOOLS, deps) == {**ZERO, "inserted": 4}
        (waiter_pid,) = waiter.execute("select pg_backend_pid()").fetchone()
        outcome: Dict[str, Any] = {}

        def run_waiter() -> None:
            try:
                outcome["result"] = _sync(waiter, tools, deps, max_deprecations=3)
            except Exception as exc:
                outcome["error"] = exc
            finally:
                waiter.rollback()

        thread = threading.Thread(target=run_waiter)
        thread.start()
        with fresh.connect(autocommit=True) as watcher:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                (n,) = watcher.execute(
                    "select count(*) from pg_locks where pid = %s and locktype = 'advisory' "
                    "and not granted",
                    (waiter_pid,),
                ).fetchone()
                if n == 1:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("waiter did not wait on the advisory lock")
        holder.commit()
        thread.join(timeout=60)
        assert not thread.is_alive()
    finally:
        holder.close()
        waiter.close()

    assert "result" not in outcome, outcome.get("result")
    assert isinstance(outcome["error"], psycopg.errors.RaiseException)
    assert "would deprecate 4 active tools, limit 3" in str(outcome["error"])
    assert fresh.rows("select count(*) from tool_registry where deprecated_at is null") == ["20"]


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------


def test_grants(migrated):
    assert migrated.rows(
        f"select has_function_privilege('anon', '{SYNC}', 'EXECUTE')::text || ','"
        f" || has_function_privilege('authenticated', '{SYNC}', 'EXECUTE')::text || ','"
        f" || has_function_privilege('service_role', '{SYNC}', 'EXECUTE')::text"
    ) == ["false,false,true"]
    # No PUBLIC item at all (a PUBLIC EXECUTE would make every role pass).
    assert migrated.rows(
        "select count(*) from pg_proc p, aclexplode(p.proacl) a "
        "where p.oid = 'sync_tool_registry(jsonb,jsonb,integer)'::regprocedure and a.grantee = 0"
    ) == ["0"]
    (security,) = migrated.rows(
        "select prosecdef::text || ',' || array_to_string(proconfig, ';') from pg_proc "
        "where oid = 'sync_tool_registry(jsonb,jsonb,integer)'::regprocedure"
    )
    assert security == "false,search_path=public"


def test_service_role_can_sync(migrated):
    with migrated.rolled_back(user="supabase_admin") as conn:
        tools, deps = _current_payload(conn)
        conn.execute("set local role service_role")
        assert _sync(conn, tools + NEW_TOOLS, deps) == {**ZERO, "inserted": 4}
