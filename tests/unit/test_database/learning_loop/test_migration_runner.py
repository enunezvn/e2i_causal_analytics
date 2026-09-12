"""ml/039–041 through the real ``scripts/run_migrations.sh``, failure first, and the rollbacks.

Spec §9: on a fresh prod-faithful copy the runner is first pointed at a repository copy whose
041 fails, which must leave 039 and 040 applied and recorded and nothing of 041 (body and
ledger row share one transaction); the real files then apply exactly 041, and a further run
has nothing pending. Direct re-application is a no-op for data. The rollbacks return every
lane object to its pre-migration definition (compared with the base copy of prod, object by
object), refuse when rows would violate the restored constraints, and are idempotent.

Opt-in: ``E2I_DB_INTEGRATION=1``. Run with ``-n 0``.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from tests.unit.test_database.learning_loop import _pg
from tests.unit.test_database.learning_loop.test_fixture_sanity import (
    _PUBLIC_REL,
    EQUIVALENCE_QUERIES,
    LANE_FUNCTIONS,
    LANE_TABLES,
    LANE_VIEWS,
    _in,
)

pytestmark = [
    pytest.mark.skipif(
        not _pg.db_integration_enabled(),
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    pytest.mark.timeout(300),
]

ML = _pg.REPO_ROOT / "database" / "ml"
LEDGER = "select filename from public.schema_migrations where filename like 'ml/04%' or filename like 'ml/039%' order by 1"
NEW_FUNCTIONS = (
    "select coalesce(string_agg(p.oid::regprocedure::text, ',' order by 1), 'none') from pg_proc p "
    "where p.pronamespace = 'public'::regnamespace "
    "and (p.proname like 'composer\\_%' or p.proname in ('get_tool_reliability', 'sync_tool_registry'))"
)
ROLLBACK_ASPECTS = (
    "columns",
    "constraints",
    "indexes",
    "ownership_rls",
    "relation_acls",
    "policies",
    "owned_sequences",
    "view_definitions",
    "functions",
    "triggers",
)
REGISTRY_ROWS = (
    "select name || '|' || coalesce(success_rate::text, '') || '|' || coalesce(avg_latency_ms::text, '') "
    "|| '|' || source_agent || '|' || updated_at::text from tool_registry order by name"
)


def _out(proc) -> str:
    # The runner colours its status words; compare plain text.
    return re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout.decode() + proc.stderr.decode())


def _repo_copy(tmp_path, *, break_041: bool):
    repo = tmp_path / "repo"
    shutil.copytree(_pg.REPO_ROOT / "scripts", repo / "scripts")
    shutil.copytree(_pg.REPO_ROOT / "database", repo / "database")
    if break_041:
        target = repo / "database" / "ml" / "041_composer_learning_loop_recording.sql"
        target.write_text(target.read_text() + "\nSELECT 1/0;\n")
    return repo


def test_runner_failure_first_then_real_files_then_nothing_pending(clone_db, tmp_path):
    db = clone_db("runner_fail")
    shims = tmp_path / "shims"
    assert db.rows(LEDGER) == []

    broken = _repo_copy(tmp_path, break_041=True)
    failed = _pg.run_runner(db, broken, shims)
    assert failed.returncode != 0, _out(failed)
    assert "Applying ml/039_tool_category_cohort.sql ... " in _out(failed)
    assert "Migration ml/041_composer_learning_loop_recording.sql failed" in _out(failed)
    assert db.rows(LEDGER) == [
        "ml/039_tool_category_cohort.sql",
        "ml/040_tool_registry_startup_sync.sql",
    ]
    # 040 is in; nothing of 041 is (its body and ledger row share one transaction).
    assert db.rows(
        "select to_regprocedure('sync_tool_registry(jsonb,jsonb,integer)') is not null"
    ) == ["t"]
    assert db.rows(
        "select coalesce(to_regprocedure('composer_record_steps(jsonb,jsonb)')::text, 'NULL') || ','"
        " || (select count(*) from pg_attribute where attrelid = 'composer_episodes'::regclass"
        " and attname in ('outcome', 'last_activity_at') and not attisdropped)::text || ','"
        " || (select count(*) from pg_attribute where attrelid = 'composer_episodes'::regclass"
        " and attname = 'query_embedding' and not attisdropped)::text"
    ) == ["NULL,0,1"]
    assert db.rows(
        "select string_agg(enumlabel, ',' order by enumsortorder) from pg_enum "
        "where enumtypid = 'tool_category'::regtype"
    ) == ["CAUSAL,SEGMENTATION,GAP,EXPERIMENT,PREDICTION,MONITORING,COHORT"]

    pending = _pg.run_runner(db, _pg.REPO_ROOT, shims, "--dry-run")
    assert pending.returncode == 0, _out(pending)
    assert [line for line in _out(pending).splitlines() if "[PENDING]" in line] == [
        "[PENDING] ml/041_composer_learning_loop_recording.sql"
    ]

    real = _pg.run_runner(db, _pg.REPO_ROOT, shims)
    assert real.returncode == 0, _out(real)
    assert _out(real).count("Applying ") == 1
    assert "Applied 1 migration(s) successfully." in _out(real)
    assert db.rows(LEDGER) == [
        "ml/039_tool_category_cohort.sql",
        "ml/040_tool_registry_startup_sync.sql",
        "ml/041_composer_learning_loop_recording.sql",
    ]

    again = _pg.run_runner(db, _pg.REPO_ROOT, shims)
    assert again.returncode == 0, _out(again)
    assert "Database is up to date. No migrations to apply." in _out(again)


SNAPSHOT = (
    "select 'r|' || row_to_json(t)::text from tool_registry t "
    "union all select 'd|' || row_to_json(t)::text from tool_dependencies t "
    "union all select 'e|' || row_to_json(t)::text from composer_episodes t "
    "union all select 's|' || row_to_json(t)::text from composition_steps t "
    "union all select 'p|' || row_to_json(t)::text from tool_performance t order by 1"
)


def test_direct_reapply_is_a_no_op_for_data(clone_db):
    db = clone_db("reapply_all")
    _pg.migrate(db, "ml/041_composer_learning_loop_recording.sql")
    with db.connect() as conn:
        tools = conn.execute(
            "select jsonb_agg(jsonb_build_object('name', name, 'description', description, "
            "'category', category, 'source_agent', source_agent, 'input_schema', input_schema, "
            "'output_schema', output_schema, 'avg_latency_ms', avg_latency_ms, 'version', version)) "
            "from tool_registry"
        ).fetchone()[0]
        cohort = {
            **tools[0],
            "name": "cohort_builder",
            "category": "COHORT",
            "source_agent": "cohort_constructor",
        }
        deps = conn.execute(
            "select coalesce(jsonb_agg(jsonb_build_object('consumer', c.name, 'producer', p.name, "
            "'output_field', d.output_field, 'input_field', d.input_field)), '[]') "
            "from tool_dependencies d join tool_registry c on c.tool_id = d.consumer_tool_id "
            "join tool_registry p on p.tool_id = d.producer_tool_id"
        ).fetchone()[0]
        conn.execute(
            "select sync_tool_registry(%s::jsonb, %s::jsonb, 3)",
            (json.dumps(tools + [cohort]), json.dumps(deps)),
        )
        seed = json.dumps({"composition_id": "reapply", "query_text": "q", "is_synthetic": True})
        conn.execute(
            "select composer_record_steps(%s::jsonb, %s::jsonb)",
            (
                seed,
                json.dumps(
                    [
                        {
                            "step_number": 0,
                            "tool_name": "cohort_builder",
                            "outcome_class": "succeeded",
                            "attempts": 1,
                            "latency_ms": 5,
                        }
                    ]
                ),
            ),
        )
        conn.execute(
            "select composer_record_finish(%s::jsonb, %s::jsonb)",
            (
                seed,
                json.dumps({"status": "COMPLETED", "outcome": "success", "total_latency_ms": 9}),
            ),
        )
    before = db.rows(SNAPSHOT)
    assert len([r for r in before if r.startswith("p|")]) == 1

    for key in _pg.LANE_MIGRATIONS:
        _pg.apply_migration(db, _pg.REPO_ROOT / "database" / key)
    assert db.rows(SNAPSHOT) == before


COMMENTS = (
    "select 'rel:' || c.relname || ':' || coalesce(obj_description(c.oid, 'pg_class'), '') "
    f"from pg_class c where {_PUBLIC_REL} and c.relname in {_in(LANE_TABLES + LANE_VIEWS)} "
    "union all select 'col:' || c.relname || '.' || a.attname || ':' "
    "|| coalesce(col_description(c.oid, a.attnum), '') from pg_attribute a "
    f"join pg_class c on c.oid = a.attrelid where {_PUBLIC_REL} and c.relname in {_in(LANE_TABLES)} "
    "and a.attnum > 0 and not a.attisdropped "
    "union all select 'fn:' || p.oid::regprocedure::text || ':' "
    "|| coalesce(obj_description(p.oid, 'pg_proc'), '') from pg_proc p "
    f"where p.pronamespace = 'public'::regnamespace and p.proname in {_in(LANE_FUNCTIONS)} order by 1"
)


def _aspects(conn: _pg.PgConn) -> dict:
    got = {aspect: conn.rows(EQUIVALENCE_QUERIES[aspect][0]) for aspect in ROLLBACK_ASPECTS}
    got["comments"] = conn.rows(COMMENTS)
    return got


def _enums(conn: _pg.PgConn) -> list:
    return conn.rows(EQUIVALENCE_QUERIES["enums"][0])


def test_rollbacks_restore_every_lane_object_and_are_idempotent(base_db, clone_db, tmp_path):
    db = clone_db("rollbacks")
    shims = tmp_path / "shims"
    # Migrated the way a deploy does, so the ledger rows the rollbacks must remove exist.
    forward = _pg.run_runner(db, _pg.REPO_ROOT, shims)
    assert forward.returncode == 0 and "Applied 3 migration(s)" in _out(forward), _out(forward)
    expected = _aspects(base_db)
    for aspect in ROLLBACK_ASPECTS:
        if not EQUIVALENCE_QUERIES[aspect][1]:
            assert expected[aspect], (
                f"{aspect} is empty on the base copy; the comparison would be vacuous"
            )
    assert expected["comments"]
    assert _aspects(db) != expected  # the migrations changed something to roll back
    registry_before = base_db.rows(REGISTRY_ROWS)
    stamps = "select name || '|' || updated_at::text from tool_registry order by name"
    stamps_migrated = db.rows(stamps)

    for attempt in ("first", "second"):
        for name in ("rollback_041.sql", "rollback_040.sql"):
            proc = _pg.apply_rollback(db, name)
            assert proc.returncode == 0, (attempt, name, proc.stderr.decode())
        got = _aspects(db)
        for aspect in (*ROLLBACK_ASPECTS, "comments"):
            assert got[aspect] == expected[aspect], (attempt, aspect)
        assert db.rows(NEW_FUNCTIONS) == ["none"], attempt
        # Restored seed values; updated_at untouched by the rollback (equal to before 040/041,
        # which did not change these rows either).
        assert db.rows(REGISTRY_ROWS) == registry_before, attempt
        assert db.rows(stamps) == stamps_migrated == base_db.rows(stamps), attempt
        # ml/039 has no rollback: COHORT stays (an enum value cannot be removed).
        assert _enums(db) == [
            line.replace("MONITORING", "MONITORING,COHORT")
            if line.startswith("tool_category:")
            else line
            for line in _enums(base_db)
        ]
        assert db.rows(LEDGER) == ["ml/039_tool_category_cohort.sql"], attempt

    # The ledger no longer claims 040/041: re-deploying the learning-loop code re-applies them.
    replay = _pg.run_runner(db, _pg.REPO_ROOT, shims)
    assert replay.returncode == 0, _out(replay)
    assert "Applied 2 migration(s) successfully." in _out(replay)
    assert db.rows(
        "select (to_regprocedure('sync_tool_registry(jsonb,jsonb,integer)') is not null)::text"
        " || ',' || (to_regprocedure('composer_record_steps(jsonb,jsonb)') is not null)::text"
    ) == ["true,true"]


def test_rollback_040_refuses_rows_the_six_agent_check_would_reject(clone_db):
    db = clone_db("rollback_guard_040")
    _pg.migrate(db, "ml/041_composer_learning_loop_recording.sql")
    assert _pg.apply_rollback(db, "rollback_041.sql").returncode == 0
    db.execute(
        "insert into tool_registry (name, description, category, source_agent) "
        "values ('cohort_builder', 'd', 'COHORT', 'cohort_constructor')",
        user="postgres",
    )
    proc = _pg.apply_rollback(db, "rollback_040.sql")
    assert proc.returncode != 0
    assert "cohort_builder (cohort_constructor)" in proc.stderr.decode()
    # One transaction: nothing of the rollback applied.
    assert db.rows(
        "select to_regprocedure('sync_tool_registry(jsonb,jsonb,integer)') is not null"
    ) == ["t"]
    assert db.rows(
        "select count(*) from pg_attribute where attrelid = 'tool_registry'::regclass "
        "and attname = 'success_rate' and not attisdropped"
    ) == ["0"]


def test_rollback_041_refuses_unfinished_episodes(clone_db):
    db = clone_db("rollback_guard_041")
    _pg.migrate(db, "ml/041_composer_learning_loop_recording.sql")
    db.execute(
        'select composer_record_start(\'{"composition_id": "open_one", "query_text": "q"}\'::jsonb)',
        user="postgres",
    )
    proc = _pg.apply_rollback(db, "rollback_041.sql")
    assert proc.returncode != 0
    assert "have no total_latency_ms" in proc.stderr.decode()
    assert "open_one" in proc.stderr.decode()
    assert db.rows("select to_regprocedure('composer_record_steps(jsonb,jsonb)') is not null") == [
        "t"
    ]
