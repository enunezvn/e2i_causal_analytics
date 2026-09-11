"""The reliability reader and the planner caveat, against a real database (spec §7.1, §7.2).

Opt-in: ``E2I_DB_INTEGRATION=1``. Every count here is computed by ``get_tool_reliability`` in
Postgres over rows written by the real recording RPCs — the rule is never handed numbers a test
made up, and the flag is exercised through ``ToolPlanner.plan()`` rather than the formatter alone.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import pytest

from src.agents.tool_composer.models.composition_models import DecompositionResult, SubQuestion
from src.agents.tool_composer.planner import ToolPlanner
from src.agents.tool_composer.registry_sync import RegistrySync
from src.agents.tool_composer.reliability import ToolReliabilityReader
from src.tool_registry.registry import get_registry
from tests.unit.test_agents.test_tool_composer.conftest import MockLLMClient
from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(
        not _pg.db_integration_enabled(),
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    pytest.mark.timeout(600),
]

UPTO = "ml/041_composer_learning_loop_recording.sql"
CAVEATED = "gap_calculator"


@pytest.fixture
def synced(clone_db) -> _pg.PgConn:
    """Migrated through 041 and synced against the live tool registry."""
    db = clone_db("reliability")
    _pg.migrate(db, UPTO)
    asyncio.run(RegistrySync(port=_pg.PsycopgRpcPort(db)).sync_once())
    return db


def _seed(cid: str, *, synthetic: bool = False) -> Dict[str, Any]:
    return {
        "composition_id": cid,
        "query_text": "which regions drive TRx?",
        "session_id": f"sess-{cid}",
        "user_id": "user-1",
        "entry_point": "chat_tool",
        "brand": "Kisqali",
        "region": "US",
        "audit_workflow_id": None,
        "is_synthetic": synthetic,
    }


async def _record_runs(
    port: _pg.PsycopgRpcPort,
    cid: str,
    *,
    tool: str = CAVEATED,
    succeeded: int,
    failed: int,
    refused: int = 0,
    synthetic: bool = False,
) -> None:
    """One episode carrying the given mix of step outcomes for ``tool``."""
    steps: List[Dict[str, Any]] = []
    for number in range(succeeded + failed + refused):
        if number < succeeded:
            outcome = "succeeded"
        elif number < succeeded + failed:
            outcome = "error"
        else:
            outcome = "refused"
        steps.append(
            {
                "step_number": number,
                "tool_name": tool,
                "outcome_class": outcome,
                "latency_ms": 1000 + number,
                "error_type": "RuntimeError" if outcome == "error" else None,
            }
        )
    seed = _seed(cid, synthetic=synthetic)
    await port.call("composer_record_start", {"p_seed": seed})
    await port.call("composer_record_steps", {"p_seed": seed, "p_steps": steps})


def _decomposition() -> DecompositionResult:
    return DecompositionResult(
        original_query="what drove Kisqali TRx?",
        sub_questions=[
            SubQuestion(id="sq_1", question="drivers?", intent="CAUSAL", entities=["Kisqali"])
        ],
        decomposition_reasoning="t",
    )


def _planner(reader: ToolReliabilityReader, llm: MockLLMClient) -> ToolPlanner:
    return ToolPlanner(
        llm_client=llm,
        tool_registry=get_registry(),
        use_episodic_memory=False,
        reliability_reader=reader,
    )


def _system_prompt(llm: MockLLMClient) -> str:
    assert llm.call_history, "the planner never called the LLM"
    return llm.call_history[-1]["system"]


async def test_planner_fetches_verdicts_and_shows_the_caveat_when_the_flag_is_on(
    synced, monkeypatch
):
    monkeypatch.setenv("TOOL_COMPOSER_RELIABILITY_IN_PLANNER", "1")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    port = _pg.PsycopgRpcPort(synced)
    # 28 succeeded + 12 health failures: n_health 40, Wilson lower bound above 10%.
    await _record_runs(port, "comp_caveat", succeeded=28, failed=12)
    llm = MockLLMClient()

    await _planner(ToolReliabilityReader(port=port), llm).plan(_decomposition())

    assert port.calls.count("get_tool_reliability") == 1
    caveats = [
        line for line in _system_prompt(llm).splitlines() if line.startswith("Reliability caveat:")
    ]
    assert caveats == ["Reliability caveat: 12 of 40 runs failed on tool errors (RuntimeError)"]


async def test_planner_never_reads_reliability_when_the_flag_is_off(synced, monkeypatch):
    monkeypatch.delenv("TOOL_COMPOSER_RELIABILITY_IN_PLANNER", raising=False)
    port = _pg.PsycopgRpcPort(synced)
    await _record_runs(port, "comp_off", succeeded=28, failed=12)
    llm = MockLLMClient()

    await _planner(ToolReliabilityReader(port=port), llm).plan(_decomposition())

    assert port.calls.count("get_tool_reliability") == 0
    assert "Reliability caveat:" not in _system_prompt(llm)


async def test_reader_cache_expires(synced):
    port = _pg.PsycopgRpcPort(synced)
    await _record_runs(port, "comp_ttl_1", succeeded=28, failed=12)
    reader = ToolReliabilityReader(port=port, ttl_s=1.0)

    first = await reader.get(30)
    await _record_runs(port, "comp_ttl_2", succeeded=40, failed=0)
    cached = await reader.get(30)
    assert cached[CAVEATED].n_health == first[CAVEATED].n_health == 40

    time.sleep(1.1)
    after = await reader.get(30)
    assert after[CAVEATED].n_health == 80
    assert after[CAVEATED].verdict == "inconclusive"


async def test_synthetic_runs_are_excluded_unless_the_deployment_includes_them(synced, monkeypatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    port = _pg.PsycopgRpcPort(synced)
    await _record_runs(port, "comp_real", succeeded=28, failed=12)
    await _record_runs(port, "comp_synth", succeeded=0, failed=40, synthetic=True)

    excluded = await ToolReliabilityReader(port=port).get(30)
    assert excluded[CAVEATED].n_health == 40
    assert excluded[CAVEATED].n_synthetic == 40  # counted, but not in the denominator

    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "1")
    included = await ToolReliabilityReader(port=port).get(30)
    assert included[CAVEATED].n_health == 80
    assert included[CAVEATED].n_health_failures == 52
