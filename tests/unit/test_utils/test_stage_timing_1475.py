"""#1475 WI1: per-request agent stage-timing ledger.

The chatbot span instrumentation (#1454 / PR #1471) attributes wall time down
to the chatbot graph's node boundary; the orchestrator node inside it is a
~14.9s warm black box. ``src/utils/stage_timing.py`` is the seam that opens
it: a contextvar-scoped ledger the chatbot's orchestrator node activates
around ``orchestrator.run``, into which the SHARED ``audited_node`` wrapper
(already timing every orchestrator graph node with perf_counter) records
``{agent_name}.{node_name}`` wall times. No ledger active -> recording is a
no-op, so nothing changes for any other caller of any audited graph.

Same discipline as #1454:
- timings ride a CONTEXTVAR, never a state channel (checkpointer replay
  class #1442);
- everything is time.perf_counter() — no time.time() mixing.
"""

import asyncio
from uuid import uuid4

import pytest

from src.utils.stage_timing import (
    activate_stage_ledger,
    deactivate_stage_ledger,
    get_active_stage_ledger,
    record_stage_wall_time,
)

# =============================================================================
# 1. Ledger primitives
# =============================================================================


class TestStageLedger:
    def test_no_ledger_by_default_and_record_is_a_noop(self):
        assert get_active_stage_ledger() is None
        record_stage_wall_time("orchestrator.classify", 12.0)  # must not raise
        assert get_active_stage_ledger() is None

    def test_activate_record_deactivate(self):
        ledger, token = activate_stage_ledger()
        try:
            assert get_active_stage_ledger() is ledger
            record_stage_wall_time("orchestrator.classify", 12.5)
            record_stage_wall_time("orchestrator.dispatch", 100.0)
            assert ledger == {
                "orchestrator.classify": 12.5,
                "orchestrator.dispatch": 100.0,
            }
        finally:
            deactivate_stage_ledger(token)
        assert get_active_stage_ledger() is None

    def test_repeat_stages_accumulate(self):
        """Repeat visits (retries, loops) must sum, mirroring node_wall_ms."""
        ledger, token = activate_stage_ledger()
        try:
            record_stage_wall_time("agent.step", 10.0)
            record_stage_wall_time("agent.step", 5.5)
            assert ledger["agent.step"] == pytest.approx(15.5)
        finally:
            deactivate_stage_ledger(token)

    def test_nested_activation_restores_outer_ledger(self):
        outer, outer_token = activate_stage_ledger()
        inner, inner_token = activate_stage_ledger()
        try:
            record_stage_wall_time("s", 1.0)
            assert inner == {"s": 1.0}
            assert outer == {}
        finally:
            deactivate_stage_ledger(inner_token)
            assert get_active_stage_ledger() is outer
            deactivate_stage_ledger(outer_token)

    @pytest.mark.asyncio
    async def test_ledger_propagates_into_spawned_tasks(self):
        """The dispatcher fans agents out via tasks; create_task copies the
        context, so their recordings must land on the SAME ledger object."""
        ledger, token = activate_stage_ledger()
        try:

            async def dispatched_agent():
                record_stage_wall_time("causal_impact.analyze", 42.0)

            await asyncio.gather(dispatched_agent(), dispatched_agent())
            assert ledger["causal_impact.analyze"] == pytest.approx(84.0)
        finally:
            deactivate_stage_ledger(token)

    @pytest.mark.asyncio
    async def test_concurrent_tasks_have_isolated_ledgers(self):
        """Two concurrent requests must never cross-attribute stages."""

        async def request(stage: str, ms: float):
            ledger, token = activate_stage_ledger()
            try:
                await asyncio.sleep(0.01)
                record_stage_wall_time(stage, ms)
                await asyncio.sleep(0.01)
                return dict(ledger)
            finally:
                deactivate_stage_ledger(token)

        a, b = await asyncio.gather(request("orchestrator.a", 1.0), request("orchestrator.b", 2.0))
        assert a == {"orchestrator.a": 1.0}
        assert b == {"orchestrator.b": 2.0}


# =============================================================================
# 2. audited_node records into the active ledger
# =============================================================================


from src.agents.base.audit_chain_mixin import audited_node, set_audit_chain_service  # noqa: E402
from src.utils.audit_chain import AgentTier  # noqa: E402


class TestAuditedNodeStageRecording:
    @pytest.mark.asyncio
    async def test_wrapped_node_records_agent_dot_node_wall_time(self):
        async def slow_node(state):
            await asyncio.sleep(0.02)
            return {"status": "ok"}

        wrapped = audited_node(
            slow_node,
            agent_name="orchestrator",
            agent_tier=AgentTier.COORDINATION,
            node_name="classify",
        )

        ledger, token = activate_stage_ledger()
        try:
            result = await wrapped({"audit_workflow_id": None})
        finally:
            deactivate_stage_ledger(token)

        assert result == {"status": "ok"}
        assert ledger["orchestrator.classify"] >= 15.0

    @pytest.mark.asyncio
    async def test_raising_node_still_records_wall_time(self):
        """A hung-then-failed dispatch is exactly the case attribution must
        survive — record in finally, then re-raise."""

        async def bad_node(state):
            await asyncio.sleep(0.02)
            raise ValueError("dispatch exploded")

        wrapped = audited_node(
            bad_node,
            agent_name="orchestrator",
            agent_tier=AgentTier.COORDINATION,
            node_name="dispatch",
        )

        ledger, token = activate_stage_ledger()
        try:
            with pytest.raises(ValueError, match="dispatch exploded"):
                await wrapped({"audit_workflow_id": None})
        finally:
            deactivate_stage_ledger(token)

        assert ledger["orchestrator.dispatch"] >= 15.0

    @pytest.mark.asyncio
    async def test_no_ledger_means_no_recording_and_no_error(self):
        async def node(state):
            return {"status": "ok"}

        wrapped = audited_node(
            node,
            agent_name="orchestrator",
            agent_tier=AgentTier.COORDINATION,
            node_name="route",
        )
        assert await wrapped({"audit_workflow_id": None}) == {"status": "ok"}
        assert get_active_stage_ledger() is None

    @pytest.mark.asyncio
    async def test_repeat_visits_accumulate_through_the_wrapper(self):
        async def node(state):
            await asyncio.sleep(0.01)
            return {}

        wrapped = audited_node(
            node,
            agent_name="orchestrator",
            agent_tier=AgentTier.COORDINATION,
            node_name="synthesize",
        )

        ledger, token = activate_stage_ledger()
        try:
            await wrapped({"audit_workflow_id": None})
            await wrapped({"audit_workflow_id": None})
        finally:
            deactivate_stage_ledger(token)

        assert ledger["orchestrator.synthesize"] >= 15.0

    @pytest.mark.asyncio
    async def test_audit_entry_behaviour_is_unchanged(self):
        """The stage recording must not disturb the audit-chain contract this
        wrapper exists for (tests/unit/test_audited_node_timing.py pins it in
        depth; this is the collision check under an ACTIVE ledger)."""

        class _RecordingService:
            def __init__(self):
                self.entries = []

            def add_entry(self, **kwargs):
                self.entries.append(kwargs)

        svc = _RecordingService()
        set_audit_chain_service(svc)  # type: ignore[arg-type]
        try:

            async def node(state):
                await asyncio.sleep(0.01)
                return {"status": "ok"}

            wrapped = audited_node(
                node,
                agent_name="gap_analyzer",
                agent_tier=AgentTier.CAUSAL_ANALYTICS,
                node_name="gap_detector",
            )
            ledger, token = activate_stage_ledger()
            try:
                await wrapped({"audit_workflow_id": uuid4()})
            finally:
                deactivate_stage_ledger(token)
        finally:
            set_audit_chain_service(None)  # type: ignore[arg-type]

        assert len(svc.entries) == 1
        assert svc.entries[0]["action_type"] == "gap_detector"
        assert svc.entries[0]["duration_ms"] >= 5
        assert ledger["gap_analyzer.gap_detector"] >= 5.0


# =============================================================================
# 3. Structural pin: every orchestrator graph node is stage-timed
# =============================================================================


def test_every_orchestrator_graph_node_is_stage_timed():
    """Adding an orchestrator node without the audited_node wrapper would
    silently reopen the attribution hole INSIDE orchestrator.run — the exact
    class of hole #1454 closed at the chatbot graph level."""
    from src.agents.orchestrator.graph import create_orchestrator_graph

    compiled = create_orchestrator_graph(agent_registry=None, allow_mock=True)
    node_names = set(compiled.builder.nodes) - {"__start__", "__end__"}
    # audit_init is the genesis workflow marker, not a stage.
    assert node_names == {
        "audit_init",
        "classify",
        "rag_context",
        "route",
        "dispatch",
        "synthesize",
    }

    for name in node_names - {"audit_init"}:
        runnable = compiled.builder.nodes[name].runnable
        fn = getattr(runnable, "afunc", None) or getattr(runnable, "func", None)
        assert getattr(fn, "__stage_timed__", False), (
            f"orchestrator node {name!r} is not wrapped by the stage-timing "
            "audited_node wrapper — its wall time would vanish from "
            "orchestrator_stage_ms"
        )
