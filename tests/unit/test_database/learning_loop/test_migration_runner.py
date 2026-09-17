"""Pending migrations through the real ``scripts/run_migrations.sh``, failure first, and the
learning-loop lane's rollbacks. All but the last section are UPGRADE-PATH tests (#2065); the
last section runs the rollback safety contracts on the current schema, as behaviour tests.

``test_pending_migrations_apply_through_the_runner_failure_first`` is generic: it rehearses, on a
copy of prod's current schema, exactly the migrations this deploy's runner is about to apply, and
runs whenever any is pending. The lane tests below it need ml/039–044 themselves pending, which
they no longer are, so they skip naming what they wait for. Their behavioural consequences
(the schema 040/041/044 left, the sync and recording contracts) are behaviour tests in
test_040_registry_sync.py and test_041_recording.py, run on every pass.

The lane tests, as written for the lane:

Spec §9: on a fresh prod-faithful copy the runner is first pointed at a repository copy whose
041 fails, which must leave everything before it applied and recorded and nothing of 041 (body
and ledger row share one transaction); the real files then apply exactly what was left pending
— 041 and 044 — and a further run has nothing pending. Direct re-application is a no-op for data. The rollbacks return every
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
        reason=_pg.OPT_IN_SKIP_REASON,
    ),
    pytest.mark.timeout(300),
]

ML = _pg.REPO_ROOT / "database" / "ml"

# Every expectation below is DERIVED from _pg.LANE_MIGRATIONS, never spelled out: the tuple grows
# (ml/044 for #2035) and a hard-coded count, pending list or ledger literal then asserts the
# previous lane.
#
# The ledger query is scoped to the lane's own keys for the same reason. It used to match
# "ml/04%", which also caught ml/042_twin_simulations_estimate_scope.sql — a NON-lane migration
# that build_base writes into the rebuilt ledger, so the opening `== []` below had already become
# false when #2053 landed. That went unseen because this whole module self-skips wherever prod
# already carries the lane (see conftest's base_db).
LEDGER = _pg.LANE_MIGRATIONS_IN_LEDGER

#: The file the failure-first case corrupts, and what that implies for the two runner passes.
BROKEN_KEY = "ml/041_composer_learning_loop_recording.sql"
_BREAK_AT = _pg.LANE_MIGRATIONS.index(BROKEN_KEY)
#: Applied and recorded before the failure; the broken file and everything after it is not.
APPLIED_BEFORE_FAILURE = list(_pg.LANE_MIGRATIONS[:_BREAK_AT])
PENDING_AFTER_FAILURE = list(_pg.LANE_MIGRATIONS[_BREAK_AT:])

#: Rollback files, newest first — the order they must be applied in. ml/039 has none (an enum
#: value cannot be removed), so it is filtered out rather than listed as an exception.
ROLLBACKS = tuple(
    name
    for name in (f"rollback_{key.split('/')[1][:3]}.sql" for key in reversed(_pg.LANE_MIGRATIONS))
    if (ML / name).exists()
)
#: What the ledger still claims once every available rollback has run.
LEDGER_AFTER_ROLLBACKS = [
    key for key in _pg.LANE_MIGRATIONS if f"rollback_{key.split('/')[1][:3]}.sql" not in ROLLBACKS
]
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


def _repo_copy(tmp_path, *, break_041: bool = False, break_key=None):
    repo = tmp_path / "repo"
    shutil.copytree(_pg.REPO_ROOT / "scripts", repo / "scripts")
    shutil.copytree(_pg.REPO_ROOT / "database", repo / "database")
    for key in ([BROKEN_KEY] if break_041 else []) + ([break_key] if break_key else []):
        target = _pg.migration_file(key, repo)
        target.write_text(target.read_text() + "\nSELECT 1/0;\n")
    return repo


def _ledger_of(keys) -> str:
    listed = ",".join("'" + k.replace("'", "''") + "'" for k in keys)
    return f"select filename from public.schema_migrations where filename in ({listed})"


@pytest.mark.realdb_upgrade()
def test_pending_migrations_apply_through_the_runner_failure_first(
    base_clone_db, pending_migrations, tmp_path
):
    """What this deploy's migration step will do, on a copy of prod as it is now.

    Failure first on the last pending file the runner WRAPS (body and ledger row in one
    transaction, so a failure must leave neither); an un-wrapped file applies statement by
    statement and a planted failure would leave a half-applied body behind, which is not what
    this rehearses. Then the real files apply exactly what is pending, and a rerun has nothing.
    """
    pending = list(pending_migrations)
    db = base_clone_db("pending_runner")
    shims = tmp_path / "shims"
    assert db.rows(_ledger_of(pending)) == []

    dry = _pg.run_runner(db, _pg.REPO_ROOT, shims, "--dry-run")
    assert dry.returncode == 0, _out(dry)
    assert [line for line in _out(dry).splitlines() if "[PENDING]" in line] == [
        f"[PENDING] {key}" for key in pending
    ]

    wrapped = [k for k in pending if not _pg.runner_unwraps(_pg.migration_file(k).read_text())]
    if wrapped:
        broken = wrapped[-1]
        failed = _pg.run_runner(db, _repo_copy(tmp_path, break_key=broken), shims)
        assert failed.returncode != 0, _out(failed)
        assert f"Migration {broken} failed" in _out(failed)
        before = pending[: pending.index(broken)]
        assert sorted(db.rows(_ledger_of(pending))) == sorted(before)
        remaining = pending[len(before) :]
    else:
        remaining = pending

    real = _pg.run_runner(db, _pg.REPO_ROOT, shims)
    assert real.returncode == 0, _out(real)
    assert f"Applied {len(remaining)} migration(s) successfully." in _out(real)
    assert sorted(db.rows(_ledger_of(pending))) == sorted(pending)

    again = _pg.run_runner(db, _pg.REPO_ROOT, shims)
    assert again.returncode == 0, _out(again)
    assert "Database is up to date. No migrations to apply." in _out(again)


LANE = _pg.LANE_MIGRATIONS


@pytest.mark.realdb_upgrade(*LANE)
def test_runner_failure_first_then_real_files_then_nothing_pending(base_clone_db, tmp_path):
    db = base_clone_db("runner_fail")
    shims = tmp_path / "shims"
    assert db.rows(LEDGER) == []

    broken = _repo_copy(tmp_path, break_041=True)
    failed = _pg.run_runner(db, broken, shims)
    assert failed.returncode != 0, _out(failed)
    assert f"Applying {_pg.LANE_MIGRATIONS[0]} ... " in _out(failed)
    assert f"Migration {BROKEN_KEY} failed" in _out(failed)
    assert db.rows(LEDGER) == APPLIED_BEFORE_FAILURE
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
        f"[PENDING] {key}" for key in PENDING_AFTER_FAILURE
    ]

    real = _pg.run_runner(db, _pg.REPO_ROOT, shims)
    assert real.returncode == 0, _out(real)
    assert _out(real).count("Applying ") == len(PENDING_AFTER_FAILURE)
    assert f"Applied {len(PENDING_AFTER_FAILURE)} migration(s) successfully." in _out(real)
    assert db.rows(LEDGER) == list(_pg.LANE_MIGRATIONS)

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


@pytest.mark.realdb_upgrade(*LANE)
def test_direct_reapply_is_a_no_op_for_data(base_clone_db):
    db = base_clone_db("reapply_all")
    # Migrated through the WHOLE lane before the snapshot. Seeding at an intermediate migration
    # made the comparison fail for the wrong reason: a later ADD COLUMN (ml/044's feedback_id)
    # shows up in row_to_json as a new null key, which is schema drift between the two
    # snapshots, not the data change this test is looking for.
    _pg.migrate(db, _pg.LANE_MIGRATIONS[-1])
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


@pytest.mark.realdb_upgrade(*LANE)
def test_rollbacks_restore_every_lane_object_and_are_idempotent(base_db, base_clone_db, tmp_path):
    db = base_clone_db("rollbacks")
    shims = tmp_path / "shims"
    # Migrated the way a deploy does, so the ledger rows the rollbacks must remove exist.
    forward = _pg.run_runner(db, _pg.REPO_ROOT, shims)
    assert forward.returncode == 0 and (
        f"Applied {len(_pg.LANE_MIGRATIONS)} migration(s)" in _out(forward)
    ), _out(forward)
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
        for name in ROLLBACKS:
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
        assert db.rows(LEDGER) == LEDGER_AFTER_ROLLBACKS, attempt

    # The ledger no longer claims the rolled-back migrations: re-deploying the learning-loop code
    # re-applies exactly those.
    replay = _pg.run_runner(db, _pg.REPO_ROOT, shims)
    assert replay.returncode == 0, _out(replay)
    assert f"Applied {len(ROLLBACKS)} migration(s) successfully." in _out(replay)
    assert db.rows(
        "select (to_regprocedure('sync_tool_registry(jsonb,jsonb,integer)') is not null)::text"
        " || ',' || (to_regprocedure('composer_record_steps(jsonb,jsonb)') is not null)::text"
    ) == ["true,true"]


@pytest.mark.realdb_upgrade(*LANE[: LANE.index("ml/041_composer_learning_loop_recording.sql") + 1])
def test_rollback_040_refuses_rows_the_six_agent_check_would_reject(base_clone_db):
    db = base_clone_db("rollback_guard_040")
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


@pytest.mark.realdb_upgrade("ml/041_composer_learning_loop_recording.sql")
def test_rollback_041_refuses_unfinished_episodes(base_clone_db):
    db = base_clone_db("rollback_guard_041")
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


# ---------------------------------------------------------------------------
# Rollback safety contracts, from the CURRENT schema (behaviour; run on every pass)
# ---------------------------------------------------------------------------
#
# The lane tests above can no longer run: they need a pre-lane base. What the rollback files
# promise an operator TODAY, on the schema prod actually has, still can: the runbook order is
# newest first, so reaching rollback_041 means undoing 044 and 043 (the refusal-code lane that
# rewrote 041's functions) first. What cannot be checked without a pre-lane base is that the
# rollbacks restore each object to its exact pre-lane definition.
#
# If a later migration changes the objects these files touch, these tests are where that
# first shows: the rollback files are then stale and need the same attention as the migration.

ROLLBACK_CHAIN = ("rollback_044.sql", "rollback_043.sql", "rollback_041.sql", "rollback_040.sql")
ROLLED_BACK_KEYS = (
    "ml/040_tool_registry_startup_sync.sql",
    "ml/041_composer_learning_loop_recording.sql",
    "ml/043_composer_refusal_reason_codes.sql",
    "ml/044_composer_episodes_feedback_id.sql",
)


def _roll_back_through(db: _pg.PgConn, last: str) -> None:
    for name in ROLLBACK_CHAIN[: ROLLBACK_CHAIN.index(last)]:
        proc = _pg.apply_rollback(db, name)
        assert proc.returncode == 0, (name, proc.stderr.decode())


def test_rollback_041_refuses_unfinished_episodes_on_the_current_schema(clone_db):
    db = clone_db("rollback_guard_041_now")
    db.execute(
        'select composer_record_start(\'{"composition_id": "open_one", "query_text": "q"}\'::jsonb)',
        user="postgres",
    )
    _roll_back_through(db, "rollback_041.sql")
    proc = _pg.apply_rollback(db, "rollback_041.sql")
    assert proc.returncode != 0
    assert "have no total_latency_ms" in proc.stderr.decode() and "open_one" in proc.stderr.decode()
    # One transaction: nothing of 041's rollback applied.
    assert db.rows("select to_regprocedure('composer_record_steps(jsonb,jsonb)') is not null") == [
        "t"
    ]


def test_rollback_040_refuses_while_the_synced_registry_names_agents_outside_the_six(clone_db):
    db = clone_db("rollback_guard_040_now")
    _roll_back_through(db, "rollback_040.sql")
    proc = _pg.apply_rollback(db, "rollback_040.sql")
    assert proc.returncode != 0
    # The code's own COHORT tool, written by the startup sync, is exactly what the guard names.
    assert "cohort_builder (cohort_constructor)" in proc.stderr.decode()
    assert db.rows(
        "select to_regprocedure('sync_tool_registry(jsonb,jsonb,integer)') is not null"
    ) == ["t"]
    assert db.rows(
        "select count(*) from pg_attribute where attrelid = 'tool_registry'::regclass "
        "and attname = 'success_rate' and not attisdropped"
    ) == ["0"]


def test_rollback_chain_is_idempotent_and_the_runner_reapplies_what_it_removed(clone_db, tmp_path):
    db = clone_db("rollback_chain_now")
    # The operator step both guards ask for: no unfinished episode, no row outside the six agents.
    db.execute("delete from tool_dependencies")
    db.execute("delete from tool_registry")
    snapshots = []
    for attempt in ("first", "second"):
        for name in ROLLBACK_CHAIN:
            proc = _pg.apply_rollback(db, name)
            assert proc.returncode == 0, (attempt, name, proc.stderr.decode())
        snapshots.append(_aspects(db))
        assert db.rows(_ledger_of(ROLLED_BACK_KEYS)) == [], attempt
        assert db.rows(NEW_FUNCTIONS) == ["none"], attempt
    assert snapshots[0] == snapshots[1]

    replay = _pg.run_runner(db, _pg.REPO_ROOT, tmp_path / "shims")
    assert replay.returncode == 0, _out(replay)
    assert f"Applied {len(ROLLED_BACK_KEYS)} migration(s) successfully." in _out(replay)
    assert sorted(db.rows(_ledger_of(ROLLED_BACK_KEYS))) == sorted(ROLLED_BACK_KEYS)
