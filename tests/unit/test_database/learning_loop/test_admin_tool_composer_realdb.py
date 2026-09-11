"""The admin observability aggregates, computed over real recorded rows (spec §8).

Opt-in: ``E2I_DB_INTEGRATION=1``. Episodes, steps and performance rows are written through the
real recording RPCs, and the service reads them back over psycopg — a real database through
another transport, the same substitution ``PsycopgRpcPort`` makes for PostgREST.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.agents.tool_composer.registry_sync import RegistrySync
from src.agents.tool_composer.reliability import ToolReliabilityReader
from src.api.schemas.admin_tool_composer import ToolComposerObservability
from src.services.tool_composer_observability_service import ToolComposerObservabilityService
from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(
        not _pg.db_integration_enabled(),
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    pytest.mark.timeout(600),
]

UPTO = "ml/041_composer_learning_loop_recording.sql"


class PsycopgQuery:
    """The query-builder calls the observability service makes, over psycopg."""

    def __init__(self, conn: _pg.PgConn, table: str):
        self.conn = conn
        self.table = table
        self._where: List[str] = []
        self._params: List[Any] = []
        self._order: List[str] = []
        self._columns: Optional[List[str]] = None
        self._range: Optional[tuple] = None

    def select(self, *columns: str) -> "PsycopgQuery":
        # Honour the projection, so a column the service forgot to request is missing here too.
        self._columns = [c.strip() for spec in columns for c in spec.split(",") if c.strip()]
        return self

    def gte(self, column: str, value: Any) -> "PsycopgQuery":
        self._where.append(f"{column} >= %s::timestamptz")
        self._params.append(value)
        return self

    def eq(self, column: str, value: Any) -> "PsycopgQuery":
        self._where.append(f"{column} = %s")
        self._params.append(value)
        return self

    def in_(self, column: str, values: List[Any]) -> "PsycopgQuery":
        self._where.append(f"{column}::text = ANY(%s)")
        self._params.append([str(v) for v in values])
        return self

    def order(self, column: str, desc: bool = False) -> "PsycopgQuery":
        self._order.append(f"{column}{' DESC' if desc else ''}")
        return self

    def range(self, start: int, end: int) -> "PsycopgQuery":
        self._range = (start, end - start + 1)
        return self

    def execute(self) -> Any:
        projection = (
            "row_to_json(t)::text"
            if self._columns is None
            else "json_build_object("
            + ", ".join(f"'{c}', t.{c}" for c in self._columns)
            + ")::text"
        )
        sql = f"SELECT {projection} FROM {self.table} t"
        if self._where:
            sql += " WHERE " + " AND ".join(self._where)
        if self._order:
            sql += " ORDER BY " + ", ".join(self._order)
        if self._range:
            sql += f" OFFSET {self._range[0]} LIMIT {self._range[1]}"
        with self.conn.connect() as conn:
            rows = conn.execute(sql, self._params).fetchall()
        return type("Result", (), {"data": [json.loads(r[0]) for r in rows]})()


class PsycopgSupabase:
    def __init__(self, conn: _pg.PgConn):
        self.conn = conn

    def table(self, name: str) -> PsycopgQuery:
        return PsycopgQuery(self.conn, name)


@pytest.fixture
def synced(clone_db) -> _pg.PgConn:
    db = clone_db("admin_obs")
    _pg.migrate(db, UPTO)
    asyncio.run(RegistrySync(port=_pg.PsycopgRpcPort(db)).sync_once())
    return db


def _seed(cid: str) -> Dict[str, Any]:
    return {
        "composition_id": cid,
        "query_text": "q" * 400,  # longer than the preview, so truncation is exercised
        "session_id": f"sess-{cid}",
        "user_id": "user-1",
        "entry_point": "chat_tool",
        "brand": "Kisqali",
        "region": "US",
        "audit_workflow_id": None,
        "is_synthetic": False,
    }


async def _record(
    port: _pg.PsycopgRpcPort,
    cid: str,
    *,
    steps: Optional[List[Dict[str, Any]]] = None,
    final: Optional[Dict[str, Any]] = None,
) -> None:
    seed = _seed(cid)
    await port.call("composer_record_start", {"p_seed": seed})
    if steps:
        await port.call("composer_record_steps", {"p_seed": seed, "p_steps": steps})
    if final:
        await port.call("composer_record_finish", {"p_seed": seed, "p_final": final})


def _step(number: int, tool: str, outcome: str) -> Dict[str, Any]:
    return {
        "step_number": number,
        "tool_name": tool,
        "outcome_class": outcome,
        "latency_ms": 500 + number,
    }


def _final(**over: Any) -> Dict[str, Any]:
    final = {
        "status": "COMPLETED",
        "outcome": "success",
        "plan_source": "llm",
        "total_latency_ms": 1200,
        "tools_executed": 2,
        "tools_succeeded": 2,
    }
    final.update(over)
    return final


async def _seed_window(port: _pg.PsycopgRpcPort) -> None:
    await _record(
        port,
        "comp_ok",
        steps=[_step(0, "cohort_builder", "succeeded"), _step(1, "cate_analyzer", "succeeded")],
        final=_final(),
    )
    await _record(
        port,
        "comp_partial",
        steps=[_step(0, "cohort_builder", "succeeded"), _step(1, "gap_calculator", "refused")],
        final=_final(outcome="partial", plan_source="plan_cache", tools_succeeded=1),
    )
    await _record(
        port,
        "comp_failed",
        steps=[_step(0, "gap_calculator", "error"), _step(1, "cate_analyzer", "dependency_unmet")],
        final=_final(
            status="FAILED",
            outcome="failed",
            failed_phase="execute",
            error_type="RuntimeError",
            plan_source="kpi_deterministic",
            tools_succeeded=0,
        ),
    )
    await _record(port, "comp_running")  # started, never finished


def _service(db: _pg.PgConn) -> ToolComposerObservabilityService:
    return ToolComposerObservabilityService(client=PsycopgSupabase(db))


async def test_overview_counts_outcomes_plan_sources_and_unfinished(synced):
    port = _pg.PsycopgRpcPort(synced)
    await _seed_window(port)

    overview = _service(synced).overview(30)

    compositions = overview["compositions"]
    assert compositions["total"] == 4
    assert (compositions["success"], compositions["partial"], compositions["failed"]) == (1, 1, 1)
    assert compositions["unfinished"] == 1
    assert compositions["by_plan_source"] == {"llm": 1, "plan_cache": 1, "kpi_deterministic": 1}
    assert compositions["p50_latency_ms"] == 1200
    # The response validates against the published schema, extra keys forbidden.
    ToolComposerObservability.model_validate(overview)


async def test_an_unfinished_episode_gone_quiet_reads_abandoned(synced):
    port = _pg.PsycopgRpcPort(synced)
    await _record(port, "comp_stale")
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    synced.execute(
        "UPDATE composer_episodes SET last_activity_at = "
        f"'{stale}'::timestamptz WHERE composition_id = 'comp_stale'"
    )

    compositions = _service(synced).overview(30)["compositions"]

    assert compositions["unfinished"] == 1 and compositions["abandoned"] == 1


async def test_recent_failures_carry_the_phase_the_step_classes_and_a_bounded_preview(synced):
    port = _pg.PsycopgRpcPort(synced)
    await _seed_window(port)

    failures = _service(synced).overview(30)["recent_failures"]

    by_id = {row["composition_id"]: row for row in failures}
    assert set(by_id) == {"comp_failed", "comp_partial"}
    failed = by_id["comp_failed"]
    assert failed["failed_phase"] == "execute" and failed["error_type"] == "RuntimeError"
    assert len(failed["query_preview"]) == 100
    assert {(s["tool_name"], s["outcome_class"]) for s in failed["step_classes"]} == {
        ("gap_calculator", "error"),
        ("cate_analyzer", "dependency_unmet"),
    }
    # A succeeded step is not a reason the composition failed.
    assert all(s["outcome_class"] != "succeeded" for s in by_id["comp_partial"]["step_classes"])


async def test_tool_rows_carry_the_readers_verdict_worst_first(synced):
    port = _pg.PsycopgRpcPort(synced)
    await _seed_window(port)
    # 28 succeeded + 12 health failures: enough evidence for a caveat. Planted on a tool the
    # window above never touches, so the counts here are exactly the planted ones.
    caveat_steps = [
        _step(n, "sensitivity_analyzer", "succeeded" if n < 28 else "error") for n in range(40)
    ]
    await _record(port, "comp_caveat", steps=caveat_steps, final=_final())
    verdicts = await ToolReliabilityReader(port=port).get(30)

    tools = _service(synced).overview(30, verdicts)["tools"]

    assert tools[0]["verdict"] == "caveat" and tools[0]["tool_name"] == "sensitivity_analyzer"
    assert tools[0]["n_health"] == 40 and tools[0]["n_health_failures"] == 12
    # Below 20 successful runs nothing measured is shown, and declared stays separate.
    quiet = [t for t in tools if t["tool_name"] == "cate_analyzer"][0]
    assert quiet["p50_latency_ms"] is None and quiet["verdict"] == "too_few_runs"
    assert quiet["declared_latency_ms"] is not None


async def test_the_window_is_respected_by_when_the_composition_started(synced):
    """Membership is `created_at`: a long-running episode still belongs to the day it began."""
    port = _pg.PsycopgRpcPort(synced)
    await _seed_window(port)
    old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    synced.execute(
        "UPDATE composer_episodes SET created_at = "
        f"'{old}'::timestamptz WHERE composition_id = 'comp_ok'"
    )

    assert _service(synced).overview(30)["compositions"]["total"] == 3
    assert _service(synced).overview(365)["compositions"]["total"] == 4

    # Recent activity on an old episode does not pull it back into the window.
    synced.execute(
        "UPDATE composer_episodes SET last_activity_at = now() WHERE composition_id = 'comp_ok'"
    )
    assert _service(synced).overview(30)["compositions"]["total"] == 3


async def test_synthetic_episodes_follow_the_deployment_provenance_flag(synced, monkeypatch):
    """The tool rows exclude synthetic runs; the composition counts must agree with them."""
    port = _pg.PsycopgRpcPort(synced)
    await _seed_window(port)
    synced.execute(
        "UPDATE composer_episodes SET is_synthetic = true WHERE composition_id = 'comp_ok'"
    )

    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    excluded = _service(synced).overview(30)
    assert excluded["include_synthetic"] is False
    assert excluded["compositions"]["total"] == 3 and excluded["compositions"]["success"] == 0

    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "1")
    included = _service(synced).overview(30)
    assert included["include_synthetic"] is True
    assert included["compositions"]["total"] == 4 and included["compositions"]["success"] == 1
