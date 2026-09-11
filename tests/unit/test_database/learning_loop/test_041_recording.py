"""ml/041: composition recording schema, seeded idempotent RPCs, reliability, views.

Spec §5.2 (schema), §5.4 (five seeded RPCs: any write that lands creates the episode; each is
idempotent so a lost response can be re-sent), §5.5 (structure only: the RPCs re-check what the
serializer sends and never store error text), §6 (``get_tool_reliability`` denominators,
provenance, window; views without fan-out; heartbeat-derived ``abandoned``). Owner decision O3:
the vector column, its index, ``find_similar_compositions`` and the step trigger are dropped.

Opt-in: ``E2I_DB_INTEGRATION=1``. Run with ``-n 0``.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(
        not _pg.db_integration_enabled(),
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    pytest.mark.timeout(600),
]

UPTO = "ml/041_composer_learning_loop_recording.sql"

INVOKED = ("succeeded", "refused", "input_rejected", "timeout", "error")
NOT_INVOKED = ("cache_hit", "plan_defect", "dependency_unmet", "circuit_open", "not_registered")

RPC_FUNCTIONS = (
    "composer_record_start(jsonb)",
    "composer_record_phase(jsonb,text,jsonb)",
    "composer_record_steps(jsonb,jsonb)",
    "composer_record_heartbeat(jsonb)",
    "composer_record_finish(jsonb,jsonb)",
    "composer_public_column_names()",
    "composer_steps_for(text[])",
    "get_tool_reliability(integer,boolean)",
)
VIEWS = ("v_tool_reliability", "v_composition_success_rate", "v_active_compositions")
NO_MISMATCH = {"schema_mismatch_tools": []}
REPAIR_REASONS = (
    "no_groups",
    "step_missing_from_groups",
    "step_repeated_in_groups",
    "unknown_step_in_groups",
    "dependency_not_in_earlier_group",
)


@pytest.fixture
def db(module_db) -> _pg.PgConn:
    return module_db(UPTO)


def seed(cid: str, **over: Any) -> Dict[str, Any]:
    return {
        "composition_id": cid,
        "query_text": "Which regions drive Kisqali TRx?",
        "session_id": f"sess-{cid}",
        "user_id": "user-1",
        "entry_point": "chat_tool",
        "brand": "Kisqali",
        "region": "US",
        "audit_workflow_id": None,
        "is_synthetic": True,
        **over,
    }


def step(n: int, tool: str, cls: str, **over: Any) -> Dict[str, Any]:
    return {
        "step_number": n,
        "tool_name": tool,
        # Neutral for every tool; tests about structure pass declared names explicitly.
        "input_params": {},
        "output_keys": {"keys": [], "other_keys": 0},
        "depends_on_steps": [],
        "serves_sub_question": "0",
        # Relative to the run, so the reliability window (executed_at = completed_at) holds
        # whatever day the suite runs.
        "started_at": (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat(),
        "completed_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        "latency_ms": 120.0,
        "outcome_class": cls,
        "attempts": 1 if cls in INVOKED else 0,
        "cache_hit": cls == "cache_hit",
        "error_type": None if cls in ("succeeded", "cache_hit") else "ToolRefusalError",
        **over,
    }


def final(**over: Any) -> Dict[str, Any]:
    return {
        "status": "COMPLETED",
        "outcome": "partial",
        "failed_phase": None,
        "error_type": None,
        "total_latency_ms": 4200.0,
        "decompose_latency_ms": 1000.0,
        "plan_latency_ms": 900.0,
        "execute_latency_ms": 1300.0,
        "synthesize_latency_ms": 1000.0,
        "sub_questions": [{"index": 0, "intent": "CAUSAL"}, {"index": 1, "intent": "OTHER"}],
        "tool_plan": {
            "steps": [
                {
                    "step_number": 0,
                    "tool_name": "gap_calculator",
                    "depends_on_steps": [],
                    "input_params": {"metric": {"type": "column", "name": "brand"}},
                }
            ],
            "execution_order_repaired": None,
        },
        "plan_source": "llm",
        "parallelizable_groups": [[0]],
        "tools_executed": 1,
        "tools_succeeded": 1,
        **over,
    }


def call(conn, fn: str, *args: Any) -> Any:
    placeholders = ", ".join("%s::jsonb" if not isinstance(a, str) else "%s" for a in args)
    params = [json.dumps(a) if not isinstance(a, str) else a for a in args]
    (result,) = conn.execute(f"select {fn}({placeholders})", params).fetchone()
    return result


def episode(conn, cid: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "select row_to_json(e)::jsonb from composer_episodes e where composition_id = %s", (cid,)
    ).fetchone()
    return row[0] if row else None


def steps_of(conn, cid: str) -> List[Dict[str, Any]]:
    return [
        r[0]
        for r in conn.execute(
            "select row_to_json(s)::jsonb from composition_steps s join composer_episodes e "
            "using (episode_id) where e.composition_id = %s order by s.step_number",
            (cid,),
        ).fetchall()
    ]


def perf_of(conn, cid: str) -> List[Dict[str, Any]]:
    return [
        r[0]
        for r in conn.execute(
            "select row_to_json(p)::jsonb from tool_performance p where composition_id = %s "
            "order by outcome_class, attempts",
            (cid,),
        ).fetchall()
    ]


def reliability(conn, tool: str, days: int = 30, include_synthetic: bool = True) -> Dict[str, Any]:
    (row,) = conn.execute(
        "select row_to_json(r)::jsonb from get_tool_reliability(%s, %s) r where tool_name = %s",
        (days, include_synthetic, tool),
    ).fetchall()
    return row[0]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    ("composer_episodes", "audit_workflow_id"): ("uuid", "YES"),
    ("composer_episodes", "outcome"): ("text", "YES"),
    ("composer_episodes", "failed_phase"): ("text", "YES"),
    ("composer_episodes", "error_type"): ("text", "YES"),
    ("composer_episodes", "plan_source"): ("text", "YES"),
    ("composer_episodes", "entry_point"): ("text", "YES"),
    ("composer_episodes", "brand"): ("text", "YES"),
    ("composer_episodes", "region"): ("text", "YES"),
    ("composer_episodes", "is_synthetic"): ("boolean", "NO"),
    ("composer_episodes", "tools_executed"): ("integer", "YES"),
    ("composer_episodes", "tools_succeeded"): ("integer", "YES"),
    ("composer_episodes", "last_activity_at"): ("timestamp with time zone", "NO"),
    ("composer_episodes", "total_latency_ms"): ("double precision", "YES"),
    ("composition_steps", "outcome_class"): ("text", "YES"),
    ("composition_steps", "attempts"): ("integer", "YES"),
    ("composition_steps", "cache_hit"): ("boolean", "NO"),
    ("composition_steps", "error_type"): ("text", "YES"),
    ("composition_steps", "serves_sub_question"): ("character varying", "YES"),
    ("tool_performance", "outcome_class"): ("text", "YES"),
    ("tool_performance", "attempts"): ("integer", "YES"),
    ("tool_performance", "is_synthetic"): ("boolean", "NO"),
    ("tool_performance", "tool_version"): ("text", "YES"),
}


def test_columns_added(db):
    rows = db.rows(
        "select table_name || '|' || column_name || '|' || data_type || '|' || is_nullable "
        "|| '|' || coalesce(character_maximum_length::text, '') from information_schema.columns "
        "where table_schema = 'public' and table_name in "
        "('composer_episodes', 'composition_steps', 'tool_performance')"
    )
    found = {tuple(r.split("|")[:2]): tuple(r.split("|")[2:]) for r in rows}
    for key, (dtype, nullable) in EXPECTED_COLUMNS.items():
        assert key in found, key
        assert found[key][:2] == (dtype, nullable), (key, found[key])
    assert found[("composition_steps", "serves_sub_question")][2] == "20"


@pytest.mark.parametrize(
    "sql",
    [
        "update composer_episodes set outcome = 'timeout'",
        "update composer_episodes set plan_source = 'guess'",
        "update composer_episodes set failed_phase = 'deploy'",
    ],
)
def test_episode_checks(db, sql):
    import psycopg

    with db.rolled_back() as conn:
        call(conn, "composer_record_start", seed("chk"))
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(sql + " where composition_id = 'chk'")


@pytest.mark.parametrize("table", ["composition_steps", "tool_performance"])
def test_outcome_class_checks(db, table):
    import psycopg

    with db.rolled_back() as conn:
        call(conn, "composer_record_start", seed("cls"))
        call(conn, "composer_record_steps", seed("cls"), [step(0, "gap_calculator", "error")])
        allowed = "cache_hit" if table == "composition_steps" else "refused"
        conn.execute(f"update {table} set outcome_class = %s", (allowed,))
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(f"update {table} set outcome_class = 'exploded'")
    if table == "tool_performance":
        with db.rolled_back() as conn:
            call(conn, "composer_record_start", seed("cls2"))
            call(conn, "composer_record_steps", seed("cls2"), [step(0, "gap_calculator", "error")])
            with pytest.raises(psycopg.errors.CheckViolation):
                conn.execute("update tool_performance set outcome_class = 'cache_hit'")


def test_dropped(db):
    assert db.rows(
        "select count(*) from pg_attribute where attrelid = 'composer_episodes'::regclass "
        "and attname = 'query_embedding' and not attisdropped"
    ) == ["0"]
    assert db.rows(
        "select coalesce(to_regclass('idx_composer_episodes_embedding')::text, 'NULL') || ','"
        " || coalesce(to_regprocedure('find_similar_compositions(vector,integer,double precision)')"
        "::text, 'NULL') || ',' || coalesce(to_regprocedure('trigger_log_step_performance()')::text,"
        " 'NULL')"
    ) == ["NULL,NULL,NULL"]
    assert db.rows(
        "select count(*) from pg_trigger where tgrelid = 'composition_steps'::regclass "
        "and not tgisinternal"
    ) == ["0"]


# ---------------------------------------------------------------------------
# Seeded, idempotent RPCs
# ---------------------------------------------------------------------------

FIRST_CALLS = {
    "start": lambda conn, s: call(conn, "composer_record_start", s),
    "phase": lambda conn, s: call(conn, "composer_record_phase", s, "PLANNING", {}),
    "steps": lambda conn, s: call(
        conn, "composer_record_steps", s, [step(0, "gap_calculator", "succeeded")]
    ),
    "heartbeat": lambda conn, s: call(conn, "composer_record_heartbeat", s),
    "finish": lambda conn, s: call(conn, "composer_record_finish", s, final()),
}


@pytest.mark.parametrize("first", sorted(FIRST_CALLS))
def test_every_rpc_creates_episode_from_seed(db, first):
    cid = f"first_{first}"
    with db.rolled_back() as conn:
        FIRST_CALLS[first](conn, seed(cid))
        FIRST_CALLS[first](conn, seed(cid))
        (n,) = conn.execute(
            "select count(*) from composer_episodes where composition_id = %s", (cid,)
        ).fetchone()
        ep = episode(conn, cid)
    assert n == 1
    assert ep["session_id"] == f"sess-{cid}" and ep["entry_point"] == "chat_tool"
    assert ep["brand"] == "Kisqali" and ep["region"] == "US" and ep["is_synthetic"] is True
    assert ep["query_text"] == "Which regions drive Kisqali TRx?"


def test_seed_carries_audit_workflow_id_and_refuses_missing_id(db):
    import psycopg

    audit = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    with db.rolled_back() as conn:
        call(conn, "composer_record_start", seed("audit", audit_workflow_id=audit))
        assert episode(conn, "audit")["audit_workflow_id"] == audit
    for bad in ({}, {"composition_id": ""}, {"composition_id": 7}):
        with db.rolled_back() as conn:
            with pytest.raises(psycopg.errors.RaiseException, match="composition_id"):
                call(conn, "composer_record_start", {**seed("x"), **bad} if bad else {})


def test_phase_updates_only_non_terminal(db):
    with db.rolled_back() as conn:
        s = seed("phase")
        call(conn, "composer_record_start", s)
        patch = {
            "decompose_latency_ms": 1000.0,
            "sub_questions": [{"index": 0, "intent": "CAUSAL"}],
        }
        assert call(conn, "composer_record_phase", s, "PLANNING", patch) is True
        ep = episode(conn, "phase")
        assert ep["status"] == "PLANNING" and ep["decompose_latency_ms"] == 1000.0
        assert ep["sub_questions"] == [{"index": 0, "intent": "CAUSAL"}]

        plan_patch = {"plan_latency_ms": 900.0, "plan_source": "plan_cache"}
        assert call(conn, "composer_record_phase", s, "EXECUTING", plan_patch) is True
        ep = episode(conn, "phase")
        assert ep["decompose_latency_ms"] == 1000.0  # an absent key keeps the earlier value
        assert ep["plan_source"] == "plan_cache"

        call(conn, "composer_record_finish", s, final())
        assert (
            call(conn, "composer_record_phase", s, "SYNTHESIZING", {"plan_source": "llm"}) is False
        )
        ep = episode(conn, "phase")
    assert (
        ep["status"] == "COMPLETED" and ep["plan_source"] == "llm"
    )  # from finish, not the late phase


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "TIMEOUT", "PENDING", "nope"])
def test_phase_refuses_terminal_or_unknown_status(db, status):
    import psycopg

    with db.rolled_back() as conn:
        with pytest.raises(psycopg.errors.RaiseException, match="phase status"):
            call(conn, "composer_record_phase", seed("badphase"), status, {})


def test_steps_idempotent_receipt(db):
    s = seed("idem")
    batch = [
        step(0, "gap_calculator", "succeeded"),
        step(1, "roi_estimator", "refused"),
        step(2, "segment_ranker", "dependency_unmet"),
    ]
    with db.rolled_back() as conn:
        first = call(conn, "composer_record_steps", s, batch)
        second = call(conn, "composer_record_steps", s, batch)
        third = call(
            conn, "composer_record_steps", s, batch[:1] + [step(3, "psi_calculator", "error")]
        )
        n_steps = len(steps_of(conn, "idem"))
        n_perf = len(perf_of(conn, "idem"))
    assert first == {"recorded": 3, "already_present": 0, "unknown_tools": [], **NO_MISMATCH}
    assert second == {"recorded": 0, "already_present": 3, "unknown_tools": [], **NO_MISMATCH}
    assert third == {"recorded": 1, "already_present": 1, "unknown_tools": [], **NO_MISMATCH}
    assert n_steps == 4
    assert n_perf == 3  # succeeded, refused, error — dependency_unmet never ran the tool


def test_resend_restores_a_missing_performance_row_once(db):
    s = seed("recover")
    batch = [step(0, "gap_calculator", "succeeded"), step(1, "roi_estimator", "error")]
    with db.rolled_back() as conn:
        call(conn, "composer_record_steps", s, batch)
        original_steps = steps_of(conn, "recover")
        lost = next(p for p in perf_of(conn, "recover") if p["outcome_class"] == "error")
        conn.execute(
            "delete from tool_performance where performance_id = %s", (lost["performance_id"],)
        )
        assert len(perf_of(conn, "recover")) == 1
        receipt = call(conn, "composer_record_steps", s, batch)
        again = call(conn, "composer_record_steps", s, batch)
        steps_after = steps_of(conn, "recover")
        perf_after = perf_of(conn, "recover")
    assert receipt == {"recorded": 0, "already_present": 2, "unknown_tools": [], **NO_MISMATCH}
    assert again == receipt
    assert steps_after == original_steps
    assert sorted(p["outcome_class"] for p in perf_after) == ["error", "succeeded"]
    restored = next(p for p in perf_after if p["outcome_class"] == "error")
    assert restored["step_id"] == lost["step_id"]
    assert {k: v for k, v in restored.items() if k != "performance_id"} == {
        k: v for k, v in lost.items() if k != "performance_id"
    }


def test_perf_unique_index_allows_null_step_ids(db):
    with db.rolled_back() as conn:
        for _ in range(2):
            conn.execute(
                "insert into tool_performance (tool_id, tool_name, latency_ms, success, called_by) "
                "select tool_id, name, 10, true, 'direct' from tool_registry "
                "where name = 'gap_calculator'"
            )
        (n,) = conn.execute(
            "select count(*) from tool_performance where step_id is null"
        ).fetchone()
    assert n == 2
    (indexdef,) = db.rows("select pg_get_indexdef('uq_tool_performance_step'::regclass)")
    assert "UNIQUE" in indexdef and "WHERE" not in indexdef


def test_steps_perf_rows_only_for_invoked_classes(db):
    s = seed("classes", is_synthetic=False)
    batch = [step(i, "gap_calculator", cls) for i, cls in enumerate(INVOKED + NOT_INVOKED)]
    with db.rolled_back() as conn:
        receipt = call(conn, "composer_record_steps", s, batch)
        stored = steps_of(conn, "classes")
        perf = perf_of(conn, "classes")
        (version,) = conn.execute(
            "select version from tool_registry where name = 'gap_calculator'"
        ).fetchone()
    assert receipt["recorded"] == 10
    assert sorted(p["outcome_class"] for p in perf) == sorted(INVOKED)
    for p in perf:
        assert p["success"] is (p["outcome_class"] == "succeeded")
        assert p["is_synthetic"] is False
        assert p["tool_version"] == version
        assert p["called_by"] == "composer"
        assert p["step_id"] in {row["step_id"] for row in stored}
    for row in stored:
        assert row["status"] == (
            "COMPLETED" if row["outcome_class"] in ("succeeded", "cache_hit") else "FAILED"
        )
        assert row["step_name"] == f"step_{row['step_number']}"
        assert row["retry_count"] == max(row["attempts"] - 1, 0)
        assert row["error_message"] is None
        assert row["cache_hit"] is (row["outcome_class"] == "cache_hit")


def test_steps_unknown_tools_reported_and_rest_recorded(db):
    s = seed("unknown")
    batch = [
        step(0, "gap_calculator", "succeeded"),
        step(1, "cohort_builder", "succeeded"),
        step(2, "roi_estimator", "succeeded"),
        step(3, "cohort_builder", "error"),
    ]
    with db.rolled_back() as conn:
        receipt = call(conn, "composer_record_steps", s, batch)
        stored = steps_of(conn, "unknown")
    assert receipt == {
        "recorded": 2,
        "already_present": 0,
        "unknown_tools": ["cohort_builder"],
        **NO_MISMATCH,
    }
    assert [r["step_number"] for r in stored] == [0, 2]


def test_steps_work_after_finish(db):
    s = seed("late")
    with db.rolled_back() as conn:
        call(conn, "composer_record_finish", s, final())
        receipt = call(conn, "composer_record_steps", s, [step(0, "gap_calculator", "succeeded")])
        assert episode(conn, "late")["status"] == "COMPLETED"
    assert receipt["recorded"] == 1


def test_finish_restores_snapshot_and_second_finish_is_noop(db):
    s = seed("finish")
    snapshot = final(
        failed_phase="execute",
        error_type="ExecutionError",
        outcome="failed",
        status="FAILED",
        plan_source="kpi_deterministic",
    )
    with db.rolled_back() as conn:
        call(conn, "composer_record_start", s)  # the phase writes were "lost"
        first = call(conn, "composer_record_finish", s, snapshot)
        (set_by_finish,) = conn.execute(
            "select completed_at = now() and last_activity_at = now() from composer_episodes "
            "where composition_id = 'finish'"
        ).fetchone()
        # now() is frozen per transaction: backdate the terminal timestamps so a replay that
        # rewrote them would show.
        conn.execute(
            "update composer_episodes set completed_at = now() - interval '1 hour', "
            "last_activity_at = now() - interval '1 hour' where composition_id = 'finish'"
        )
        after_first = episode(conn, "finish")
        second = call(
            conn, "composer_record_finish", s, final(outcome="success", total_latency_ms=1.0)
        )
        after_second = episode(conn, "finish")
    assert first == {"recorded": True}
    assert set_by_finish is True
    assert second == {"recorded": False, "already_terminal": True}
    assert after_second == after_first
    for key in (
        "outcome", "failed_phase", "error_type", "total_latency_ms", "decompose_latency_ms",
        "plan_latency_ms", "execute_latency_ms", "synthesize_latency_ms", "sub_questions",
        "plan_source", "parallelizable_groups", "tools_executed", "tools_succeeded",
    ):  # fmt: skip
        assert after_first[key] == snapshot[key], key
    assert after_first["status"] == "FAILED"
    assert after_first["tool_plan"] == snapshot["tool_plan"]
    assert after_first["completed_at"] is not None
    assert after_first["error_message"] is None
    assert after_first["synthesized_response"] is None and after_first["tool_outputs"] == {}


@pytest.mark.parametrize("status", ["EXECUTING", "PENDING", None])
def test_finish_refuses_non_terminal_status(db, status):
    import psycopg

    with db.rolled_back() as conn:
        with pytest.raises(psycopg.errors.RaiseException, match="terminal status"):
            call(conn, "composer_record_finish", seed("nt"), final(status=status))


def test_cancelled_outcome_recorded(db):
    s = seed("cancel")
    with db.rolled_back() as conn:
        call(conn, "composer_record_start", s)
        call(
            conn,
            "composer_record_finish",
            s,
            final(status="FAILED", outcome="cancelled", failed_phase="execute"),
        )
        ep = episode(conn, "cancel")
    assert (ep["status"], ep["outcome"], ep["failed_phase"]) == ("FAILED", "cancelled", "execute")


def test_every_rpc_bumps_activity_only_while_non_terminal(db):
    s = seed("beat")
    stale = (
        "update composer_episodes set last_activity_at = now() - interval '10 minutes' "
        "where composition_id = 'beat'"
    )
    bumped_sql = (
        "select (last_activity_at >= now() - interval '1 second') from composer_episodes "
        "where composition_id = 'beat'"
    )
    numbers = itertools.count()
    writes = {
        "start": lambda conn: call(conn, "composer_record_start", s),
        "heartbeat": lambda conn: call(conn, "composer_record_heartbeat", s),
        "phase": lambda conn: call(conn, "composer_record_phase", s, "EXECUTING", {}),
        "steps": lambda conn: call(
            conn, "composer_record_steps", s, [step(next(numbers), "gap_calculator", "succeeded")]
        ),
    }
    with db.rolled_back() as conn:
        call(conn, "composer_record_start", s)
        for name, write in writes.items():
            conn.execute(stale)
            write(conn)
            (bumped,) = conn.execute(bumped_sql).fetchone()
            assert bumped is True, f"{name} did not bump an open episode"
        conn.execute(stale)
        call(conn, "composer_record_finish", s, final())
        (bumped,) = conn.execute(bumped_sql).fetchone()
        assert bumped is True, "finish did not bump"

        for name, write in {
            **writes,
            "finish": lambda conn: call(conn, "composer_record_finish", s, final()),
        }.items():
            conn.execute(stale)
            write(conn)
            (bumped,) = conn.execute(bumped_sql).fetchone()
            assert bumped is False, f"{name} bumped a terminal episode"


def test_steps_for_returns_steps_in_order(db):
    with db.rolled_back() as conn:
        call(conn, "composer_record_steps", seed("ref_a"), [
            step(2, "roi_estimator", "refused"),
            step(0, "gap_calculator", "succeeded"),
            step(1, "segment_ranker", "cache_hit"),
        ])  # fmt: skip
        call(conn, "composer_record_start", seed("ref_b"))
        (result,) = conn.execute(
            "select composer_steps_for(%s::text[])", (["ref_a", "ref_b", "ref_missing"],)
        ).fetchone()
    assert result == {
        "ref_a": [
            {"step_number": 0, "tool_name": "gap_calculator", "outcome_class": "succeeded"},
            {"step_number": 1, "tool_name": "segment_ranker", "outcome_class": "cache_hit"},
            {"step_number": 2, "tool_name": "roi_estimator", "outcome_class": "refused"},
        ],
        "ref_b": [],
    }


# ---------------------------------------------------------------------------
# Structure only: the RPCs re-check what the serializer sends
# ---------------------------------------------------------------------------


def test_catalog_allowlist_rpc(db):
    with db.rolled_back() as conn:
        (names,) = conn.execute("select composer_public_column_names()").fetchone()
    # Measured 2026-09-11 on prod: brand, region and treatment_arm are public column names;
    # a bare "treatment" is not (the spec's example was wrong; recorded in the task report), so
    # the positive control uses names that exist.
    assert {"brand", "region", "treatment_arm"} <= set(names)
    assert "PT-0001" not in names
    assert names == sorted(set(names))


SENTINEL = "SENTINEL_7f3"


def test_rpc_rechecks_column_entries(db):
    s = seed("recheck")
    params = {
        "metric": {"type": "column", "name": SENTINEL},
        "group_by": {"type": "column", "name": "brand"},
    }
    plan = final()["tool_plan"]
    plan["steps"][0]["input_params"] = params
    with db.rolled_back() as conn:
        call(
            conn,
            "composer_record_steps",
            s,
            [step(0, "gap_calculator", "succeeded", input_params=params)],
        )
        call(conn, "composer_record_phase", s, "EXECUTING", {"tool_plan": plan})
        stored_step = steps_of(conn, "recheck")[0]["input_params"]
        phase_plan = episode(conn, "recheck")["tool_plan"]
        call(conn, "composer_record_finish", s, final(tool_plan=plan))
        finish_plan = episode(conn, "recheck")["tool_plan"]
    expected = {
        "metric": {"type": "str", "len": len(SENTINEL)},
        "group_by": {"type": "column", "name": "brand"},
    }
    assert stored_step == expected
    assert phase_plan["steps"][0]["input_params"] == expected
    assert finish_plan["steps"][0]["input_params"] == expected


def test_rpc_reduces_anything_that_is_not_structure(db):
    """Every LLM- or data-authored position the RPCs accept, planted with a sentinel.

    Declared parameter names come from the registry (measured): gap_calculator IN metric,
    entities, group_by, entity_type / OUT gap, entity_values, top_performer, bottom_performer;
    causal_effect_estimator IN method, outcome, treatment, confounders; psi_calculator IN
    feature, period_column, current_period, baseline_period.
    """
    s = seed("reduce")
    gap_params = {
        "metric": SENTINEL,
        "group_by": {"type": "str", "len": 3, "value": SENTINEL},
        "entities": {"type": SENTINEL, "k": 1},
        "entity_type": {SENTINEL: {"a": 1}},
        SENTINEL: 1,
        "listed": [SENTINEL],
        "undeclared_params": 2,
    }
    cee_params = {
        "treatment": {"type": "ref", "step": 0, "field": "gap", "raw": SENTINEL},
        "outcome": {"type": "ref", "step": 0, "field": SENTINEL},
        "confounders": [SENTINEL, 1],
        "method": {"type": "frame", "rows": 10, "columns": 3, "cols": [SENTINEL]},
    }
    psi_params = {
        "current_period": 0.05,
        "baseline_period": True,
        "feature": {"type": "column", "name": "brand"},
    }
    bad_steps = [
        step(
            0,
            "gap_calculator",
            "refused",
            input_params=gap_params,
            serves_sub_question=SENTINEL,
            error_type=f"Refused {SENTINEL}!",
            error_message=SENTINEL,
            step_name=SENTINEL,
            output_keys={"keys": ["gap", SENTINEL, "ate"], "other_keys": 1, SENTINEL: SENTINEL},
            depends_on_steps=[SENTINEL, 0],
        ),
        step(1, "causal_effect_estimator", "succeeded", input_params=cee_params),
        step(2, "psi_calculator", "succeeded", input_params=psi_params),
    ]
    snapshot = final(
        sub_questions=[
            {"index": 0, "intent": SENTINEL, "text": SENTINEL},
            {"index": 1, "intent": "causal"},
        ],
        parallelizable_groups=[[0, SENTINEL]],
        error_type=f"{SENTINEL} boom",
        error_message=SENTINEL,
        synthesized_response=SENTINEL,
        tool_outputs={SENTINEL: 1},
    )
    snapshot["tool_plan"]["steps"][0]["input_params"] = gap_params
    snapshot["tool_plan"]["steps"].append(
        {"step_number": 1, "tool_name": SENTINEL, "input_params": {"metric": 1}, SENTINEL: SENTINEL}
    )
    snapshot["tool_plan"]["reasoning"] = SENTINEL
    snapshot["tool_plan"]["execution_order_repaired"] = SENTINEL
    with db.rolled_back() as conn:
        receipt = call(conn, "composer_record_steps", s, bad_steps)
        call(
            conn,
            "composer_record_phase",
            s,
            "PLANNING",
            {"sub_questions": snapshot["sub_questions"], "error_message": SENTINEL},
        )
        call(conn, "composer_record_finish", s, snapshot)
        rows = [episode(conn, "reduce")] + steps_of(conn, "reduce") + perf_of(conn, "reduce")
        stored = steps_of(conn, "reduce")
        ep = episode(conn, "reduce")
    assert all(SENTINEL not in json.dumps(r) for r in rows), [
        r for r in rows if SENTINEL in json.dumps(r)
    ]
    # Names the registry does not declare make the RPC report the tool (the recorder then runs
    # the lazy sync); the receipt names tools, never the offending keys.
    assert receipt["schema_mismatch_tools"] == ["causal_effect_estimator", "gap_calculator"]
    assert SENTINEL not in json.dumps(receipt)
    reduced_gap = {
        "metric": {"type": "str", "len": len(SENTINEL)},
        "group_by": {"type": "str", "len": 3},
        "entities": {"type": "dict", "len": 2},
        "entity_type": {"type": "dict", "len": 1},
        "undeclared_params": 4,  # SENTINEL and "listed", plus the 2 the serializer counted
    }
    assert stored[0]["input_params"] == reduced_gap
    assert stored[1]["input_params"] == {
        "treatment": {"type": "ref", "step": 0, "field": "gap"},
        "outcome": {"type": "ref", "step": 0, "field": None},
        "confounders": {"type": "list", "len": 2},
        "method": {"type": "frame", "rows": 10, "columns": 3},
    }
    # Positive control: declared names, numbers, booleans and catalog columns survive.
    assert stored[2]["input_params"] == psi_params
    assert stored[0]["serves_sub_question"] is None and stored[0]["error_type"] is None
    assert stored[0]["depends_on_steps"] == [0]
    assert stored[0]["output_result"] == {"keys": ["gap"], "other_keys": 3}
    assert ep["sub_questions"] == [
        {"index": 0, "intent": "OTHER"},
        {"index": 1, "intent": "CAUSAL"},
    ]
    assert ep["parallelizable_groups"] == []
    assert ep["tool_plan"] == {
        "steps": [
            {
                "step_number": 0,
                "tool_name": "gap_calculator",
                "depends_on_steps": [],
                "input_params": reduced_gap,
            },
            {
                "step_number": 1,
                "tool_name": None,
                "depends_on_steps": [],
                "input_params": {"undeclared_params": 1},
            },
        ],
        "execution_order_repaired": None,
    }


@pytest.mark.parametrize("reason", REPAIR_REASONS + ("patient_sentinel", "", None, 3))
def test_order_repair_reason_vocabulary(db, reason):
    s = seed(f"repair_{reason}")
    plan = final()["tool_plan"]
    plan["execution_order_repaired"] = reason
    with db.rolled_back() as conn:
        call(conn, "composer_record_finish", s, final(tool_plan=plan))
        stored = episode(conn, s["composition_id"])["tool_plan"]["execution_order_repaired"]
    assert stored == (reason if reason in REPAIR_REASONS else None)


def test_stale_registry_schema_reports_the_tool_and_still_records_the_outcome(db):
    declared = {
        "metric": {"type": "column", "name": "brand"},
        "group_by": {"type": "column", "name": "region"},
    }
    with db.rolled_back() as conn:
        (original_in, original_out) = conn.execute(
            "select input_schema, output_schema from tool_registry where name = 'gap_calculator'"
        ).fetchone()
        conn.execute(
            "update tool_registry set input_schema = '{}', output_schema = '{}' where name = 'gap_calculator'"
        )
        stale = call(
            conn,
            "composer_record_steps",
            seed("stale"),
            [
                step(
                    0,
                    "gap_calculator",
                    "error",
                    input_params=declared,
                    output_keys={"keys": ["gap"], "other_keys": 0},
                )
            ],
        )
        stale_step = steps_of(conn, "stale")[0]
        stale_perf = perf_of(conn, "stale")

        # The sync lands (the registry again describes the code): later writes keep the names.
        conn.execute(
            "update tool_registry set input_schema = %s, output_schema = %s where name = 'gap_calculator'",
            (json.dumps(original_in), json.dumps(original_out)),
        )
        fresh = call(
            conn,
            "composer_record_steps",
            seed("synced"),
            [
                step(
                    0,
                    "gap_calculator",
                    "error",
                    input_params=declared,
                    output_keys={"keys": ["gap"], "other_keys": 0},
                )
            ],
        )
        synced_step = steps_of(conn, "synced")[0]
    assert stale == {
        "recorded": 1,
        "already_present": 0,
        "unknown_tools": [],
        "schema_mismatch_tools": ["gap_calculator"],
    }
    assert stale_step["outcome_class"] == "error" and [p["outcome_class"] for p in stale_perf] == [
        "error"
    ]
    assert stale_step["input_params"] == {"undeclared_params": 2}
    assert stale_step["output_result"] == {"keys": [], "other_keys": 1}
    assert fresh == {"recorded": 1, "already_present": 0, "unknown_tools": [], **NO_MISMATCH}
    assert synced_step["input_params"] == declared
    assert synced_step["output_result"] == {"keys": ["gap"], "other_keys": 0}


def test_error_type_keeps_class_names(db):
    s = seed("errtype")
    with db.rolled_back() as conn:
        call(
            conn,
            "composer_record_steps",
            s,
            [step(0, "gap_calculator", "error", error_type="asyncio.TimeoutError")],
        )
        call(conn, "composer_record_finish", s, final(error_type="PlanningError"))
        assert steps_of(conn, "errtype")[0]["error_type"] == "asyncio.TimeoutError"
        assert perf_of(conn, "errtype")[0]["error_type"] == "asyncio.TimeoutError"
        assert episode(conn, "errtype")["error_type"] == "PlanningError"


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def _plant(conn, cid: str, classes, tool: str = "psi_calculator", **seed_over: Any) -> None:
    batch = []
    for i, entry in enumerate(classes):
        cls, over = (entry, {}) if isinstance(entry, str) else entry
        batch.append(step(i, tool, cls, **over))
    call(conn, "composer_record_steps", seed(cid, **seed_over), batch)


def test_reliability_denominators(db):
    with db.rolled_back() as conn:
        _plant(conn, "rel", ["succeeded"] * 3 + ["refused", "input_rejected", "timeout", "error"]
               + ["cache_hit", "plan_defect"])  # fmt: skip
        r = reliability(conn, "psi_calculator")
    assert (
        r["n_invoked"],
        r["n_succeeded"],
        r["n_refused"],
        r["n_health_failures"],
        r["n_health"],
    ) == (7, 3, 2, 2, 5)


def test_reliability_n_retried_and_latency_only_succeeded(db):
    with db.rolled_back() as conn:
        _plant(conn, "lat", [
            ("succeeded", {"attempts": 2, "latency_ms": 100.0}),
            ("succeeded", {"attempts": 1, "latency_ms": 300.0}),
            ("error", {"attempts": 3, "latency_ms": 9000.0, "error_type": "KeyError"}),
            ("timeout", {"attempts": 3, "latency_ms": 9000.0, "error_type": "TimeoutError"}),
            ("timeout", {"attempts": 3, "latency_ms": 9000.0, "error_type": "TimeoutError"}),
        ])  # fmt: skip
        r = reliability(conn, "psi_calculator")
    assert r["n_retried"] == 1
    assert float(r["p50_latency_ms"]) == 200.0 and float(r["p95_latency_ms"]) == 290.0
    assert r["most_common_health_error"] == "TimeoutError"
    assert r["declared_latency_ms"] is not None and r["version"] is not None


def test_reliability_provenance_filter(db):
    with db.rolled_back() as conn:
        _plant(conn, "syn", ["error", "error"], is_synthetic=True)
        _plant(conn, "real", ["succeeded"], is_synthetic=False)
        with_syn = reliability(conn, "psi_calculator", include_synthetic=True)
        without = reliability(conn, "psi_calculator", include_synthetic=False)
    assert (with_syn["n_invoked"], with_syn["n_health_failures"], with_syn["n_synthetic"]) == (
        3,
        2,
        2,
    )
    assert (without["n_invoked"], without["n_health_failures"], without["n_synthetic"]) == (1, 0, 2)


def test_reliability_window(db):
    with db.rolled_back() as conn:
        _plant(conn, "old", ["error"])
        conn.execute(
            "update tool_performance set executed_at = now() - interval '31 days' where composition_id = 'old'"
        )
        assert reliability(conn, "psi_calculator", days=30)["n_invoked"] == 0
        assert reliability(conn, "psi_calculator", days=60)["n_invoked"] == 1


def test_reliability_lists_active_tools_only(db):
    with db.rolled_back() as conn:
        (before,) = conn.execute("select count(*) from get_tool_reliability(30, true)").fetchone()
        conn.execute("update tool_registry set deprecated_at = now() where name = 'psi_calculator'")
        (after,) = conn.execute("select count(*) from get_tool_reliability(30, true)").fetchone()
    assert (before, after) == (16, 15)


@pytest.mark.parametrize("days", [None, 0, -5])
def test_reliability_refuses_bad_window(db, days):
    import psycopg

    with db.rolled_back() as conn:
        with pytest.raises(psycopg.errors.RaiseException, match="p_days"):
            conn.execute("select * from get_tool_reliability(%s, true)", (days,))


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def test_success_rate_view_no_fanout(db):
    with db.rolled_back() as conn:
        (before,) = conn.execute(
            "select coalesce(sum(total_compositions), 0) from v_composition_success_rate"
        ).fetchone()
        s = seed("fan")
        call(
            conn,
            "composer_record_steps",
            s,
            [step(i, "gap_calculator", "succeeded") for i in range(3)],
        )
        call(conn, "composer_record_finish", s, final(outcome="success", plan_source="plan_cache"))
        (row,) = conn.execute(
            "select row_to_json(v)::jsonb from v_composition_success_rate v where day = date_trunc('day', now())"
        ).fetchone()
    assert before == 0
    assert (row["total_compositions"], row["succeeded"], row["plan_cache"], row["unfinished"]) == (
        1,
        1,
        1,
        0,
    )


def test_active_view_abandoned_derived(db):
    with db.rolled_back() as conn:
        for cid, age in (("stale", "6 minutes"), ("live", "1 minute")):
            call(conn, "composer_record_start", seed(cid))
            conn.execute(
                f"update composer_episodes set last_activity_at = now() - interval '{age}' "
                "where composition_id = %s",
                (cid,),
            )
        call(conn, "composer_record_finish", seed("done"), final())
        rows = dict(
            conn.execute("select composition_id, abandoned from v_active_compositions").fetchall()
        )
    assert rows == {"stale": True, "live": False}


def test_v_tool_reliability_is_the_30_day_function(db):
    with db.rolled_back() as conn:
        _plant(conn, "view", ["succeeded", "error"])
        view = conn.execute(
            "select row_to_json(v)::jsonb from v_tool_reliability v order by tool_name"
        ).fetchall()
        fn = conn.execute(
            "select row_to_json(r)::jsonb from get_tool_reliability(30, true) r order by tool_name"
        ).fetchall()
    assert view == fn


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------


def test_grants_all_new_objects(db):
    public_composer = db.rows(
        "select p.oid::regprocedure::text from pg_proc p where p.pronamespace = 'public'::regnamespace "
        "and (p.proname like 'composer\\_%' or p.proname = 'get_tool_reliability') order by 1"
    )
    # Every composer_* function (helpers included) is covered below, so none can escape.
    assert set(RPC_FUNCTIONS) <= set(public_composer)
    for fn in public_composer:
        assert db.rows(
            f"select has_function_privilege('anon', '{fn}', 'EXECUTE')::text || ','"
            f" || has_function_privilege('authenticated', '{fn}', 'EXECUTE')::text || ','"
            f" || has_function_privilege('service_role', '{fn}', 'EXECUTE')::text"
        ) == ["false,false,true"], fn
        assert db.rows(
            f"select count(*) from pg_proc p, aclexplode(p.proacl) a where p.oid = '{fn}'::regprocedure "
            "and a.grantee = 0"
        ) == ["0"], fn
        assert db.rows(
            f"select prosecdef::text || ',' || coalesce(array_to_string(proconfig, ';'), '') "
            f"from pg_proc where oid = '{fn}'::regprocedure"
        ) == ["false,search_path=public"], fn
    for view in VIEWS:
        assert db.rows(
            f"select has_table_privilege('anon', '{view}', 'SELECT')::text || ','"
            f" || has_table_privilege('authenticated', '{view}', 'SELECT')::text || ','"
            f" || has_table_privilege('service_role', '{view}', 'SELECT')::text"
        ) == ["false,false,true"], view


def test_service_role_can_record_and_read(db):
    s = seed("svc")
    with db.rolled_back(user="supabase_admin") as conn:
        conn.execute("set local role service_role")
        call(conn, "composer_record_start", s)
        call(conn, "composer_record_phase", s, "PLANNING", {"plan_source": "llm"})
        call(conn, "composer_record_heartbeat", s)
        assert (
            call(conn, "composer_record_steps", s, [step(0, "gap_calculator", "succeeded")])[
                "recorded"
            ]
            == 1
        )
        assert call(conn, "composer_record_finish", s, final()) == {"recorded": True}
        (refs,) = conn.execute("select composer_steps_for(%s::text[])", (["svc"],)).fetchone()
        assert refs["svc"][0]["tool_name"] == "gap_calculator"
        conn.execute("select composer_public_column_names()").fetchone()
        assert reliability(conn, "gap_calculator")["n_succeeded"] == 1
        for view in VIEWS:
            conn.execute(f"select * from {view}").fetchall()


def test_reapply_041_keeps_recorded_rows(clone_db):
    fresh = clone_db("reapply_041")
    _pg.migrate(fresh, UPTO)
    s = seed("keep")
    with fresh.connect() as conn:
        call(conn, "composer_record_steps", s, [step(0, "gap_calculator", "succeeded")])
        call(conn, "composer_record_finish", s, final())
    snapshot = sorted(
        fresh.rows("select row_to_json(e)::text from composer_episodes e")
        + fresh.rows("select row_to_json(s)::text from composition_steps s")
        + fresh.rows("select row_to_json(p)::text from tool_performance p")
    )
    assert _pg.apply_migration(fresh, _pg.REPO_ROOT / "database" / UPTO) == "wrapped"
    assert (
        sorted(
            fresh.rows("select row_to_json(e)::text from composer_episodes e")
            + fresh.rows("select row_to_json(s)::text from composition_steps s")
            + fresh.rows("select row_to_json(p)::text from tool_performance p")
        )
        == snapshot
    )
