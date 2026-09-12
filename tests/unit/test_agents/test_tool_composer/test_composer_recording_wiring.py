"""compose() records every composition, once, under one id (spec §5.3).

- The composition id is generated before anything else and carried by every CompositionResult,
  success and failure alike, and by every recording write.
- The episode seed is built after the audit start, from the LOCAL audit workflow id (never a
  value a caller left in ``context``).
- Each phase boundary, each finished step and the terminal outcome reach the recorder; a cancel
  is recorded as ``cancelled`` in the phase it interrupted and still propagates.
- Both entry points say who called: ``chat_tool`` and ``orchestrator_agent``.
- Recording happens only where the learning loop is enabled (the API sets the flag); a composer
  run anywhere else (tests, scripts) sends nothing.

The LLM transport is the conftest double; the recorder is the real CompositionRecorder over a
port that records payloads, so what would cross the network is what is asserted.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.agents.tool_composer import learning_recorder
from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.learning_recorder import CompositionRecorder, drain
from src.agents.tool_composer.registry_sync import RegistrySync
from src.tool_registry.registry import ToolSchema
from tests.unit.test_agents.test_tool_composer.test_learning_recorder_failopen import ScriptedPort

QUERY = "what drove rx volume and how does it vary by region?"


@pytest.fixture(autouse=True)
async def _no_leftover_recording():
    yield
    await drain(timeout=0, cancel_heartbeats=True)
    for task in list(learning_recorder._pending):
        task.cancel()
    await asyncio.sleep(0)


class Capture:
    """A recorder factory: real recorders over one recording port."""

    def __init__(self) -> None:
        self.port = ScriptedPort()
        self.sync = RegistrySync(port=self.port)
        self.seeds: List[Dict[str, Any]] = []

    def __call__(self, composition_id: str, seed: Dict[str, Any]) -> CompositionRecorder:
        self.seeds.append(dict(seed))
        return CompositionRecorder(composition_id, seed, port=self.port, sync=self.sync)

    def payloads(self, rpc: str) -> List[Dict[str, Any]]:
        return [params for name, params in self.port.payloads if name == rpc]

    @property
    def calls(self) -> List[str]:
        return self.port.calls


def _composer(
    mock_llm_client, mock_tool_registry, capture: Optional[Capture], **config: Any
) -> ToolComposer:
    return ToolComposer(
        llm_client=mock_llm_client,
        tool_registry=mock_tool_registry,
        enable_memory_contribution=False,
        recorder_factory=capture,
        config={"phases": {"execute": {"max_retries": 0}}, **config},
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argument, stale",
    [
        (None, None),
        (None, uuid.uuid4()),
        (uuid.UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7"), uuid.uuid4()),
    ],
)
def test_seed_uses_local_audit_id_not_context(mock_llm_client, mock_tool_registry, argument, stale):
    composer = _composer(mock_llm_client, mock_tool_registry, None)
    context: Dict[str, Any] = {
        "session_id": "s",
        "user_id": "u",
        "brand": "Kisqali",
        "region": "NE",
    }
    if stale is not None:
        context["audit_workflow_id"] = stale
    seed = composer._recording_seed("comp_x", "q" * 900, context, argument)
    assert seed["audit_workflow_id"] == argument
    assert seed["composition_id"] == "comp_x"
    assert len(seed["query_text"]) <= 503  # redact_query(query, 500)
    assert (seed["session_id"], seed["user_id"], seed["brand"], seed["region"]) == (
        "s",
        "u",
        "Kisqali",
        "NE",
    )
    assert seed["entry_point"] == "direct"


def test_audit_block_yields_none_when_absent(mock_llm_client, mock_tool_registry):
    composer = _composer(mock_llm_client, mock_tool_registry, None)
    assert composer._start_audit(None, "q", {}) is None


async def test_recording_is_off_unless_the_learning_loop_is_enabled(
    mock_llm_client, mock_tool_registry, monkeypatch
):
    monkeypatch.delenv("TOOL_COMPOSER_LEARNING_LOOP_ENABLED", raising=False)
    composer = _composer(mock_llm_client, mock_tool_registry, None)
    result = await composer.compose(QUERY)
    assert result.success is True
    assert learning_recorder.pending_count() == 0


async def test_composition_records_every_phase_under_one_id(mock_llm_client, mock_tool_registry):
    capture = Capture()
    composer = _composer(mock_llm_client, mock_tool_registry, capture)
    result = await composer.compose(QUERY, {"session_id": "sess-1", "brand": "Kisqali"})
    assert await drain(timeout=10) == 0

    assert result.success is True
    assert [c for c in capture.calls if c.startswith("composer_record_")][:1] == [
        "composer_record_start"
    ]
    ids = {
        params["p_seed"]["composition_id"]
        for _, params in capture.port.payloads
        if "p_seed" in params
    }
    assert ids == {result.composition_id}
    assert [p["p_status"] for p in capture.payloads("composer_record_phase")] == [
        "PLANNING",
        "EXECUTING",
        "SYNTHESIZING",
    ]
    steps = [s for batch in capture.payloads("composer_record_steps") for s in batch["p_steps"]]
    assert {s["step_number"] for s in steps} == {0, 1}
    (finish,) = capture.payloads("composer_record_finish")
    final = finish["p_final"]
    assert (final["status"], final["outcome"], final["plan_source"]) == (
        "COMPLETED",
        "success",
        "llm",
    )
    assert (final["tools_executed"], final["tools_succeeded"]) == (2, 2)
    assert final["failed_phase"] is None
    assert capture.seeds[0]["session_id"] == "sess-1" and capture.seeds[0]["brand"] == "Kisqali"


async def test_decomposition_failure_records_the_failed_phase_and_inner_class(
    mock_llm_client, mock_tool_registry
):
    capture = Capture()
    composer = _composer(mock_llm_client, mock_tool_registry, capture)
    mock_llm_client.set_error(ValueError("provider rejected the prompt"))
    result = await composer.compose(QUERY)
    assert await drain(timeout=10) == 0
    assert result.success is False
    (finish,) = capture.payloads("composer_record_finish")
    assert finish["p_seed"]["composition_id"] == result.composition_id
    final = finish["p_final"]
    assert (final["status"], final["outcome"], final["failed_phase"]) == (
        "FAILED",
        "failed",
        "decompose",
    )
    assert final["error_type"] and "provider rejected" not in json.dumps(capture.port.payloads)


def _register(registry, name: str, fn: Any) -> None:
    registry.register(
        schema=ToolSchema(
            name=name, description="wiring probe tool.", source_agent="causal_impact", tier=2
        ),
        callable=fn,
    )


def _planning(steps: List[dict]) -> str:
    return json.dumps(
        {
            "reasoning": "t",
            "tool_mappings": [
                {
                    "sub_question_id": s["sub_question_id"],
                    "tool_name": s["tool_name"],
                    "confidence": 0.9,
                }
                for s in steps
            ],
            "execution_steps": steps,
            "parallel_groups": [[s["step_id"]] for s in steps],
        }
    )


async def test_total_failure_records_failed_in_execute_under_the_same_id(
    mock_llm_client, mock_tool_registry
):
    async def failing(**_: Any) -> Any:
        raise RuntimeError("tool down")

    _register(mock_tool_registry, "failing_probe", failing)
    mock_llm_client.set_planning_response(
        _planning(
            [
                {
                    "step_id": "step_1",
                    "sub_question_id": "sq_1",
                    "tool_name": "failing_probe",
                    "input_mapping": {},
                },
                {
                    "step_id": "step_2",
                    "sub_question_id": "sq_2",
                    "tool_name": "failing_probe",
                    "input_mapping": {},
                },
            ]
        )
    )
    capture = Capture()
    composer = _composer(mock_llm_client, mock_tool_registry, capture)
    result = await composer.compose(QUERY)
    assert await drain(timeout=10) == 0
    assert result.success is False
    (finish,) = capture.payloads("composer_record_finish")
    assert finish["p_seed"]["composition_id"] == result.composition_id
    final = finish["p_final"]
    assert (final["status"], final["outcome"], final["failed_phase"]) == (
        "FAILED",
        "failed",
        "execute",
    )
    assert (final["tools_executed"], final["tools_succeeded"]) == (2, 0)


def test_error_and_total_failure_results_carry_the_given_id(mock_llm_client, mock_tool_registry):
    composer = _composer(mock_llm_client, mock_tool_registry, None)
    started = datetime.now(timezone.utc)
    error = composer._create_error_result(
        "q", started, {}, "boom", None, composition_id="comp_given"
    )
    assert error.composition_id == "comp_given"
    total = composer._create_total_failure_result(
        "q",
        error.decomposition,
        error.plan,
        error.execution,
        started,
        {},
        composition_id="comp_given",
    )
    assert total.composition_id == "comp_given"


async def test_cancelled_recorded_then_reraised(mock_llm_client, mock_tool_registry):
    started = asyncio.Event()

    async def slow(**_: Any) -> Any:
        started.set()
        await asyncio.sleep(30)
        return {"late": True}

    async def ok(**_: Any) -> Any:
        return {"value": 1}

    _register(mock_tool_registry, "slow_probe", slow)
    _register(mock_tool_registry, "ok_probe", ok)
    mock_llm_client.set_planning_response(
        _planning(
            [
                {
                    "step_id": "step_1",
                    "sub_question_id": "sq_1",
                    "tool_name": "ok_probe",
                    "input_mapping": {},
                },
                {
                    "step_id": "step_2",
                    "sub_question_id": "sq_2",
                    "tool_name": "slow_probe",
                    "input_mapping": {},
                },
            ]
        )
    )
    capture = Capture()
    composer = _composer(mock_llm_client, mock_tool_registry, capture)
    task = asyncio.create_task(composer.compose(QUERY))
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await drain(timeout=10) == 0
    (finish,) = capture.payloads("composer_record_finish")
    final = finish["p_final"]
    assert (final["status"], final["outcome"], final["failed_phase"]) == (
        "FAILED",
        "cancelled",
        "execute",
    )
    recorded_steps = {
        s["step_number"] for b in capture.payloads("composer_record_steps") for s in b["p_steps"]
    }
    assert recorded_steps == {0}  # the finished step; the cancelled one has no result


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def test_orchestrator_agent_marks_its_compositions(mock_llm_client, mock_tool_registry):
    from src.agents.tool_composer.agent import ToolComposerAgent

    capture = Capture()
    agent = ToolComposerAgent()
    agent._composer = _composer(mock_llm_client, mock_tool_registry, capture)
    output = await agent.run({"query": QUERY, "context": {"session_id": "sess-agent"}})
    assert await drain(timeout=10) == 0
    assert output.success is True
    assert capture.seeds[0]["entry_point"] == "orchestrator_agent"
    assert capture.seeds[0]["session_id"] == "sess-agent"


def test_chat_tool_marks_its_compositions():
    from src.api.routes import chatbot_tools

    context = chatbot_tools._composer_context(
        brand="Kisqali", region="NE", session_id="s", max_parallel=3
    )
    assert context["entry_point"] == "chat_tool"
    assert (
        context["brand"],
        context["region"],
        context["session_id"],
        context["max_parallel"],
    ) == (
        "Kisqali",
        "NE",
        "s",
        3,
    )
    # The tool builds its composer context through that helper (and nowhere else).
    source = inspect.getsource(chatbot_tools.tool_composer_tool.coroutine)
    calls = [
        node.func.id
        for node in ast.walk(ast.parse(source.lstrip()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "_composer_context" in calls


# ---------------------------------------------------------------------------
# A failing audit start
# ---------------------------------------------------------------------------


async def test_a_failing_audit_start_yields_a_null_id_and_the_composition_proceeds(
    mock_llm_client, mock_tool_registry, caplog
):
    """A real audit service whose database is unreachable records no id and fails nothing.

    The client and its transport are real (supabase-py over http); only the endpoint is a
    closed local port, so ``start_workflow``'s insert raises the way a dropped database does.
    The service is installed for the whole composition, and the caller's context carries a
    STALE audit id: the seed must still be null, never that value.
    """
    from supabase import create_client

    from src.agents.base.audit_chain_mixin import set_audit_chain_service
    from src.utils.audit_chain import AuditChainService

    # Not a secret: a syntactically valid key shape for a port nothing listens on.
    service = AuditChainService(create_client("http://127.0.0.1:9", "aaaa.bbbb.cccc"))
    stale = uuid.uuid4()
    capture = Capture()
    composer = _composer(mock_llm_client, mock_tool_registry, capture)

    set_audit_chain_service(service)
    try:
        with caplog.at_level("WARNING"):
            result = await composer.compose(QUERY, {"audit_workflow_id": stale})
    finally:
        set_audit_chain_service(None)
    assert await drain(timeout=10) == 0

    assert result.success is True
    assert any("audit workflow" in record.message for record in caplog.records)
    assert capture.seeds[0]["audit_workflow_id"] is None
    for _name, params in capture.port.payloads:
        assert params.get("p_seed", {}).get("audit_workflow_id") is None
        assert str(stale) not in json.dumps(params, default=str)


# ---------------------------------------------------------------------------
# Recording never fails a composition
# ---------------------------------------------------------------------------


class ThrowingRecorder:
    """A recorder whose every call raises, as a broken or misconfigured one would."""

    def __init__(self, composition_id: str) -> None:
        self.composition_id = composition_id
        self.calls: List[str] = []

    def _raise(self, name: str) -> None:
        self.calls.append(name)
        raise RuntimeError(f"recorder {name} is broken")

    def start(self) -> None:
        self._raise("start")

    def decomposed(self, decomposition: Any, *, latency_ms: float) -> None:
        self._raise("decomposed")

    def planned(self, plan: Any, *, latency_ms: float, plan_source: Any) -> None:
        self._raise("planned")

    def step(self, step_number: int, result: Any) -> None:
        self._raise("step")

    def executed(self, *, latency_ms: float) -> None:
        self._raise("executed")

    def finish(self, **fields: Any) -> None:
        self._raise("finish")

    def cancelled(self, phase: str) -> None:
        self._raise("cancelled")


async def test_a_recorder_that_raises_everywhere_does_not_fail_the_composition(
    mock_llm_client, mock_tool_registry
):
    broken: List[ThrowingRecorder] = []

    def factory(composition_id: str, seed: Dict[str, Any]) -> ThrowingRecorder:
        broken.append(ThrowingRecorder(composition_id))
        return broken[-1]

    composer = ToolComposer(
        llm_client=mock_llm_client,
        tool_registry=mock_tool_registry,
        enable_memory_contribution=False,
        recorder_factory=factory,
        config={"phases": {"execute": {"max_retries": 0}}},
    )
    result = await composer.compose(QUERY)

    assert result.success is True
    # Every hook was reached and every one raised; none of it reached the caller. `step` is
    # named explicitly, and counted, so removing per-step recording would fail this too.
    assert {"start", "decomposed", "planned", "step", "executed", "finish"} <= set(broken[0].calls)
    assert broken[0].calls.count("step") == 2


async def test_a_recorder_that_raises_does_not_swallow_a_cancel(
    mock_llm_client, mock_tool_registry
):
    started = asyncio.Event()

    async def slow(**_: Any) -> Any:
        started.set()
        await asyncio.sleep(30)
        return {"late": True}

    _register(mock_tool_registry, "slow_probe", slow)
    mock_llm_client.set_planning_response(
        _planning(
            [
                {
                    "step_id": "step_1",
                    "sub_question_id": "sq_1",
                    "tool_name": "slow_probe",
                    "input_mapping": {},
                }
            ]
        )
    )
    composer = ToolComposer(
        llm_client=mock_llm_client,
        tool_registry=mock_tool_registry,
        enable_memory_contribution=False,
        recorder_factory=lambda cid, seed: ThrowingRecorder(cid),
        config={"phases": {"execute": {"max_retries": 0}}},
    )
    task = asyncio.create_task(composer.compose(QUERY))
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
