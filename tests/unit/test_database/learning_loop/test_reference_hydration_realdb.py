"""Episodic references are hydrated with their recorded steps, read from the database (spec §7.3).

Opt-in: ``E2I_DB_INTEGRATION=1``. The steps are written through the real recording RPCs and read
back through ``composer_steps_for`` (ml/041), over psycopg — a real database, another transport.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from src.agents.tool_composer.memory_hooks import hydrate_reference_steps
from src.agents.tool_composer.registry_sync import RegistrySync
from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(
        not _pg.db_integration_enabled(),
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    pytest.mark.timeout(300),
]

UPTO = "ml/041_composer_learning_loop_recording.sql"


@pytest.fixture
def synced(clone_db) -> _pg.PgConn:
    """Migrated through 041 and synced against the live tool registry."""
    db = clone_db("hydration")
    _pg.migrate(db, UPTO)
    asyncio.run(RegistrySync(port=_pg.PsycopgRpcPort(db)).sync_once())
    return db


def _seed(cid: str) -> Dict[str, Any]:
    return {
        "composition_id": cid,
        "query_text": "which regions drive TRx?",
        "session_id": f"sess-{cid}",
        "user_id": "user-1",
        "entry_point": "chat_tool",
        "brand": "Kisqali",
        "region": "US",
        "audit_workflow_id": None,
        "is_synthetic": True,
    }


def _reference(cid: str) -> Dict[str, Any]:
    return {"memory_id": cid, "raw_content": {"composition_id": cid, "success": True}}


async def _record(port: _pg.PsycopgRpcPort, cid: str, steps: List[Dict[str, Any]]) -> None:
    await port.call("composer_record_start", {"p_seed": _seed(cid)})
    if steps:
        await port.call("composer_record_steps", {"p_seed": _seed(cid), "p_steps": steps})


async def test_hydration_reads_steps_by_composition_id(synced):
    port = _pg.PsycopgRpcPort(synced)
    cid = "comp_hydrate1"
    # Recorded out of order on purpose: the reader must return them in step order.
    await _record(
        port,
        cid,
        [
            {"step_number": 2, "tool_name": "cate_analyzer", "outcome_class": "succeeded"},
            {"step_number": 0, "tool_name": "gap_calculator", "outcome_class": "refused"},
            {
                "step_number": 1,
                "tool_name": "causal_effect_estimator",
                "outcome_class": "succeeded",
            },
        ],
    )

    (reference,) = await hydrate_reference_steps([_reference(cid)], port=port)

    steps = reference["recorded_steps"]
    assert [s["step_number"] for s in steps] == [0, 1, 2]
    assert [s["tool_name"] for s in steps] == [
        "gap_calculator",
        "causal_effect_estimator",
        "cate_analyzer",
    ]
    assert [s["outcome_class"] for s in steps] == ["refused", "succeeded", "succeeded"]
    assert port.calls.count("composer_steps_for") == 1


async def test_one_read_serves_every_reference(synced):
    port = _pg.PsycopgRpcPort(synced)
    await _record(
        port,
        "comp_multi_a",
        [{"step_number": 0, "tool_name": "cohort_builder", "outcome_class": "succeeded"}],
    )
    await _record(
        port,
        "comp_multi_b",
        [{"step_number": 0, "tool_name": "cohort_statistics", "outcome_class": "cache_hit"}],
    )
    before = port.calls.count("composer_steps_for")

    references = await hydrate_reference_steps(
        [_reference("comp_multi_a"), _reference("comp_multi_b")], port=port
    )

    assert [r["recorded_steps"][0]["tool_name"] for r in references] == [
        "cohort_builder",
        "cohort_statistics",
    ]
    assert port.calls.count("composer_steps_for") == before + 1


async def test_an_episode_without_steps_and_an_unknown_id_hydrate_to_no_steps(synced):
    port = _pg.PsycopgRpcPort(synced)
    await _record(port, "comp_nosteps", [])

    references = await hydrate_reference_steps(
        [_reference("comp_nosteps"), _reference("comp_never_recorded")], port=port
    )

    assert [r["recorded_steps"] for r in references] == [[], []]


async def test_hydration_failure_leaves_the_references_usable(synced):
    """A read that raises must not lose the references; they fall back to the legacy shapes."""

    class FailingPort:
        calls: List[str] = []

        async def call(self, name: str, params: Dict[str, Any]) -> Any:
            raise ConnectionError("steps read is down")

    references = await hydrate_reference_steps([_reference("comp_any")], port=FailingPort())

    assert [r["recorded_steps"] for r in references] == [[]]
    assert references[0]["raw_content"]["composition_id"] == "comp_any"
