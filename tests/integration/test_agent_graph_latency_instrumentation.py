"""End-to-end: instrumented agent graphs record a REAL per-node duration_ms.

This is the faithful counterpart to the ``audited_node`` unit tests. It builds a
real LangGraph (the same ``add_audited_node`` wiring the agent graphs now use),
installs a recording audit service, invokes the compiled graph, and asserts that
a timed audit entry with a real ``duration_ms`` was recorded for the wrapped node.

Problem B regression target: before this change, the ~11 non-causal_impact agent
graphs only emitted a genesis ``workflow_start`` entry (no ``duration_ms``), so
``/analytics/summary`` averaged an empty latency list and returned a fake
``avg_latency_ms = 0.0`` once those agents ran.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from src.agents.base.audit_chain_mixin import (
    add_audited_node,
    create_workflow_initializer,
    set_audit_chain_service,
)
from src.utils.audit_chain import AgentTier


class _RecordingService:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def start_workflow(self, **kwargs: Any) -> Any:
        wid = uuid4()

        class _Entry:
            workflow_id = wid

        return _Entry()

    def add_entry(self, **kwargs: Any) -> Any:
        self.entries.append(kwargs)
        return object()


class _State(TypedDict, total=False):
    audit_workflow_id: Any
    status: str


def test_compiled_graph_records_real_node_duration() -> None:
    """A compiled graph with an add_audited_node business node records duration_ms."""
    rec = _RecordingService()
    set_audit_chain_service(rec)  # type: ignore[arg-type]
    try:
        initializer = create_workflow_initializer("gap_analyzer", AgentTier.CAUSAL_ANALYTICS)

        async def gap_detector(state: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(0.025)
            return {"status": "ok"}

        workflow: Any = StateGraph(_State)
        workflow.add_node("audit_init", initializer)
        add_audited_node(
            workflow,
            "gap_detector",
            gap_detector,
            agent_name="gap_analyzer",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
        )
        workflow.set_entry_point("audit_init")
        workflow.add_edge("audit_init", "gap_detector")
        workflow.add_edge("gap_detector", END)
        graph = workflow.compile()

        final = asyncio.run(graph.ainvoke({}))
        assert final.get("status") == "ok"

        # Genesis entry has no duration; the gap_detector node entry must.
        timed = [e for e in rec.entries if e.get("action_type") == "gap_detector"]
        assert len(timed) == 1, f"expected one timed gap_detector entry, got {rec.entries}"
        entry = timed[0]
        assert entry["agent_name"] == "gap_analyzer"
        assert entry["agent_tier"] == AgentTier.CAUSAL_ANALYTICS
        # REAL measurement: slept 25ms -> at least ~15ms recorded.
        assert isinstance(entry["duration_ms"], int)
        assert entry["duration_ms"] >= 15
    finally:
        set_audit_chain_service(None)  # type: ignore[arg-type]
