"""The registry sync client against a prod-faithful database migrated through ml/041.

``RegistrySync.sync_once()`` builds the payload from the live registry and calls
``sync_tool_registry`` through an ``RpcPort`` (here psycopg as service_role, in production the
service-role Supabase client). After one call the DB registry equals the running code: 20 active
tools, 13 dependencies, every field as the code declares it. It runs at most once per process
(concurrent callers share one RPC), a failure is logged and never raised, and the column
allowlist for the recorder's serializer is fetched through the same port.

Opt-in: ``E2I_DB_INTEGRATION=1``. Run with ``-n 0``.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from src.agents.tool_composer import registry_sync
from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(
        not _pg.db_integration_enabled(),
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    pytest.mark.timeout(300),
]

UPTO = "ml/041_composer_learning_loop_recording.sql"

REGISTRY_AS_JSON = (
    "select jsonb_agg(jsonb_build_object('name', name, 'description', description, "
    "'category', category, 'source_agent', source_agent, 'input_schema', input_schema, "
    "'output_schema', output_schema, 'avg_latency_ms', avg_latency_ms, 'version', version) "
    "order by name) from tool_registry where deprecated_at is null"
)
DEPENDENCIES_AS_JSON = (
    "select jsonb_agg(jsonb_build_object('consumer', c.name, 'producer', p.name, "
    "'output_field', d.output_field, 'input_field', d.input_field) order by c.name, p.name) "
    "from tool_dependencies d join tool_registry c on c.tool_id = d.consumer_tool_id "
    "join tool_registry p on p.tool_id = d.producer_tool_id"
)


@pytest.fixture
def migrated(clone_db) -> _pg.PgConn:
    db = clone_db("sync_client")
    _pg.migrate(db, UPTO)
    return db


def _json(db: _pg.PgConn, sql: str):
    with db.connect() as conn:
        return conn.execute(sql).fetchone()[0]


async def test_sync_client_applies_payload(migrated):
    port = _pg.PsycopgRpcPort(migrated)
    counts = await registry_sync.RegistrySync(port=port).sync_once()

    tools, deps = registry_sync.build_sync_payload()
    assert counts is not None and counts["inserted"] == 4 and counts["deprecated"] == 0
    assert port.calls == ["sync_tool_registry"]
    assert _json(migrated, REGISTRY_AS_JSON) == tools
    assert len(tools) == 20
    stored_deps = _json(migrated, DEPENDENCIES_AS_JSON)
    assert len(stored_deps) == 13
    assert sorted(stored_deps, key=lambda d: (d["consumer"], d["producer"])) == sorted(
        deps, key=lambda d: (d["consumer"], d["producer"])
    )


async def test_second_call_in_process_is_noop(migrated):
    port = _pg.PsycopgRpcPort(migrated)
    sync = registry_sync.RegistrySync(port=port)
    first = await sync.sync_once()
    second = await sync.sync_once()
    assert first is not None and second is None
    assert port.calls == ["sync_tool_registry"]
    assert sync.synced is True


async def test_concurrent_callers_share_one_rpc(migrated):
    port = _pg.PsycopgRpcPort(migrated)
    sync = registry_sync.RegistrySync(port=port)
    results = await asyncio.gather(sync.sync_once(), sync.sync_once(), sync.sync_once())
    assert port.calls == ["sync_tool_registry"]
    assert sum(r is not None for r in results) == 1


async def test_failure_is_logged_cooled_down_then_retried(migrated, clone_db, caplog):
    gone = clone_db("sync_gone")
    _pg.migrate(gone, UPTO)
    port = _pg.PsycopgRpcPort(gone)
    _pg.drop(gone)
    sync = registry_sync.RegistrySync(port=port, retry_after_s=0.5)
    with caplog.at_level(logging.WARNING, logger=registry_sync.__name__):
        results = await asyncio.gather(sync.sync_once(), sync.sync_once(), sync.sync_once())
        assert results == [None, None, None]
        assert await sync.sync_once() is None  # within the cooldown: no new attempt
    assert sync.synced is False
    assert port.calls == ["sync_tool_registry"]  # queued callers shared the failing attempt
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "sync_tool_registry" in warnings[0].getMessage()

    # The database is back after the cooldown: the next call syncs.
    port.conn = migrated
    await asyncio.sleep(0.6)
    counts = await sync.sync_once()
    assert counts is not None and counts["inserted"] == 4
    assert sync.synced is True and port.calls == ["sync_tool_registry", "sync_tool_registry"]


async def test_column_allowlist_fetch(migrated, clone_db):
    port = _pg.PsycopgRpcPort(migrated)
    sync = registry_sync.RegistrySync(port=port)
    allowlist = await sync.column_allowlist()
    assert isinstance(allowlist, frozenset)
    assert {"brand", "region", "treatment_arm"} <= allowlist and "PT-0001" not in allowlist
    assert await sync.column_allowlist() is allowlist  # cached
    assert port.calls == ["composer_public_column_names"]

    gone = clone_db("allowlist_gone")
    _pg.migrate(gone, UPTO)
    failing = _pg.PsycopgRpcPort(gone)
    _pg.drop(gone)
    # Never fetched: no names at all (privacy fails closed).
    assert await registry_sync.RegistrySync(port=failing).column_allowlist() is None


async def test_allowlist_refresh_failure_keeps_last_set_cools_down_and_recovers(migrated, clone_db):
    gone = clone_db("allowlist_blip")
    _pg.migrate(gone, UPTO)
    port = _pg.PsycopgRpcPort(migrated)
    sync = registry_sync.RegistrySync(port=port, allowlist_ttl_s=0.3, retry_after_s=0.5)
    first = await sync.column_allowlist()
    assert first is not None

    await asyncio.sleep(0.4)  # expired
    port.conn = gone
    _pg.drop(gone)
    stale = await sync.column_allowlist()
    assert stale is first  # a failed refresh keeps the fetched catalog names
    assert await sync.column_allowlist() is first  # cooling down: no new attempt
    assert port.calls == ["composer_public_column_names"] * 2

    port.conn = migrated
    await asyncio.sleep(0.6)
    recovered = await sync.column_allowlist()
    assert recovered == first and recovered is not first
    assert port.calls == ["composer_public_column_names"] * 3


async def test_startup_runs_sync_and_allowlist_and_never_raises(migrated, clone_db):
    port = _pg.PsycopgRpcPort(migrated)
    sync = registry_sync.RegistrySync(port=port)
    await registry_sync.learning_loop_startup(sync)
    assert port.calls == ["sync_tool_registry", "composer_public_column_names"]
    assert sync.synced is True
    assert await sync.column_allowlist() is not None
    assert port.calls == ["sync_tool_registry", "composer_public_column_names"]  # cached
    assert _json(migrated, "select count(*) from tool_registry where deprecated_at is null") == 20

    gone = clone_db("startup_gone")
    _pg.migrate(gone, UPTO)
    failing = registry_sync.RegistrySync(port=_pg.PsycopgRpcPort(gone))
    _pg.drop(gone)
    await registry_sync.learning_loop_startup(failing)  # logs, does not raise
    assert failing.synced is False
