"""One REAL composition, recorded end to end into a throwaway database (spec §5.3, §9).

Opt-in twice over: ``E2I_DB_INTEGRATION=1`` (docker + the prod schema copy) and ``E2I_LIVE_LLM=1``
(production LLM spend, about one composition per run — gated on G-LLM). Everything on the path is
real: the LLM client the composer builds per phase, the planner, the executor, the recorder, and
the RPCs, reached over psycopg instead of PostgREST.

Run twice to pin the audit identity at its call site:

- no audit service wired → the episode's ``audit_workflow_id`` is NULL;
- an audit service wired → it equals the id that composition's own audit start returned, which is
  only true if the seed is built AFTER the audit block.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from uuid import UUID

import pytest

from src.agents.base.audit_chain_mixin import set_audit_chain_service
from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.learning_recorder import CompositionRecorder, drain
from src.agents.tool_composer.registry_sync import RegistrySync
from src.utils.audit_chain import AuditChainService
from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(
        not _pg.db_integration_enabled(),
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    pytest.mark.skipif(
        os.getenv("E2I_LIVE_LLM") != "1",
        reason="real LLM spend; set E2I_LIVE_LLM=1 once the dispatcher records G-LLM",
    ),
    pytest.mark.timeout(900),
]

UPTO = "ml/041_composer_learning_loop_recording.sql"
QUERY = "What drove Kisqali TRx in the last quarter, and how does it differ by region?"


class PsycopgTable:
    """The few PostgREST query-builder calls ``AuditChainService`` makes, over psycopg.

    Real SQL against the throwaway database — the same substitution the recorder's
    ``PsycopgRpcPort`` makes for ``POST /rpc/<name>``, for a database with no PostgREST in front.
    """

    def __init__(self, conn: _pg.PgConn, table: str):
        self.conn = conn
        self.table = table
        self._filters: List[tuple] = []
        self._order: Optional[tuple] = None
        self._limit: Optional[int] = None

    def select(self, *_columns: str) -> "PsycopgTable":
        return self

    def eq(self, column: str, value: Any) -> "PsycopgTable":
        self._filters.append((column, value))
        return self

    def order(self, column: str, desc: bool = False) -> "PsycopgTable":
        self._order = (column, desc)
        return self

    def limit(self, count: int) -> "PsycopgTable":
        self._limit = count
        return self

    def insert(self, row: Dict[str, Any]) -> "PsycopgInsert":
        return PsycopgInsert(self.conn, self.table, row)

    def execute(self) -> Any:
        sql = f"select row_to_json(t)::text from {self.table} t"
        params: List[Any] = []
        if self._filters:
            sql += " where " + " and ".join(f"{c} = %s" for c, _ in self._filters)
            params = [v for _, v in self._filters]
        if self._order is not None:
            sql += f" order by {self._order[0]}{' desc' if self._order[1] else ''}"
        if self._limit is not None:
            sql += f" limit {self._limit}"
        with self.conn.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return type("Result", (), {"data": [json.loads(r[0]) for r in rows]})()


class PsycopgInsert:
    def __init__(self, conn: _pg.PgConn, table: str, row: Dict[str, Any]):
        self.conn = conn
        self.table = table
        self.row = row

    def execute(self) -> Any:
        columns = list(self.row)
        values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in self.row.values()]
        placeholders = ", ".join(["%s"] * len(columns))
        sql = f"insert into {self.table} ({', '.join(columns)}) values ({placeholders})"
        with self.conn.connect() as conn:
            conn.execute(sql, values)
            conn.commit()
        return type("Result", (), {"data": [self.row]})()


class PsycopgSupabase:
    """The client surface ``AuditChainService`` uses: ``table(...)`` and ``rpc(...)``."""

    def __init__(self, conn: _pg.PgConn):
        self.conn = conn

    def table(self, name: str) -> PsycopgTable:
        return PsycopgTable(self.conn, name)

    def rpc(self, name: str, params: Dict[str, Any]) -> Any:
        raise NotImplementedError(f"{name} is not part of this test's audit path")


@pytest.fixture
def live_db(clone_db) -> _pg.PgConn:
    """Migrated through 041 and synced against the live tool registry."""
    db = clone_db("live_llm")
    _pg.migrate(db, UPTO)
    asyncio.run(RegistrySync(port=_pg.PsycopgRpcPort(db)).sync_once())
    return db


def _one(db: _pg.PgConn, sql: str, *params: Any) -> Any:
    with db.connect() as conn:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None


async def _compose_recorded(db: _pg.PgConn, audit: Optional[AuditChainService]) -> Any:
    port = _pg.PsycopgRpcPort(db)
    sync = RegistrySync(port=port)
    set_audit_chain_service(audit)
    try:
        composer = ToolComposer(
            enable_memory_contribution=False,
            recorder_factory=lambda cid, seed: CompositionRecorder(
                cid, seed, port=port, sync=sync, heartbeat_s=3600.0
            ),
        )
        result = await composer.compose(QUERY, {"brand": "Kisqali", "entry_point": "direct"})
    finally:
        set_audit_chain_service(None)
    assert await drain(timeout=30) == 0
    return result


async def test_a_real_composition_records_one_episode_with_a_null_audit_id(live_db):
    result = await _compose_recorded(live_db, None)

    episode = _one(
        live_db,
        "select row_to_json(e)::jsonb from composer_episodes e where composition_id = %s",
        result.composition_id,
    )
    assert episode is not None
    assert episode["audit_workflow_id"] is None
    assert episode["status"] in ("COMPLETED", "FAILED")
    assert episode["outcome"] in ("success", "partial", "failed")
    assert episode["query_text"] and episode["entry_point"] == "direct"

    steps = _one(
        live_db,
        "select count(*) from composition_steps s join composer_episodes e using (episode_id) "
        "where e.composition_id = %s",
        result.composition_id,
    )
    assert steps == len(result.execution.step_results)

    invoked = {
        r.tool_name
        for r in result.execution.step_results
        if getattr(r, "outcome_class", None) not in (None, "dependency_unmet", "not_registered")
    }
    perf = _one(
        live_db,
        "select count(distinct tool_name) from tool_performance where composition_id = %s",
        result.composition_id,
    )
    assert perf == len(invoked)


async def test_a_wired_audit_service_puts_its_own_workflow_id_on_the_episode(live_db):
    audit = AuditChainService(PsycopgSupabase(live_db))
    result = await _compose_recorded(live_db, audit)

    recorded = _one(
        live_db,
        "select audit_workflow_id from composer_episodes where composition_id = %s",
        result.composition_id,
    )
    assert recorded is not None
    started = _one(
        live_db,
        "select workflow_id from audit_chain_entries where agent_name = 'tool_composer' "
        "and action_type = 'workflow_start' order by created_at desc limit 1",
    )
    assert UUID(str(recorded)) == UUID(str(started))
