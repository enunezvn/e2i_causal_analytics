"""#2238: explainer nodes must not echo the ``operator.add`` channels.

``ExplainerState`` declares ``errors`` and ``warnings`` as
``Annotated[List, operator.add]``. assemble / reason / generate and the
error handler each returned ``{**state, ...}`` (or ``state``), so LangGraph
re-appended the accumulated lists at every node.

Measured before the fix (compiled graph, deterministic reasoning, memory
hooks offline): success path seeds x8; failure path seed warning x4 and the
assembler's own "No analysis results provided" error x2.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from src.agents.explainer.graph import build_explainer_graph

SEED_W = "seed-warning"
SEED_E = {"node": "seed", "error": "seed-error"}


def _base_state(**overrides: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "query": "explain the causal impact result",
        "analysis_results": [
            {
                "agent": "causal_impact",
                "analysis_type": "causal_impact",
                "result": {"ate": 0.1, "confidence": 0.9},
                "key_findings": ["engagement lifts conversions"],
                "confidence": 0.9,
            }
        ],
        "user_expertise": "analyst",
        "output_format": "narrative",
        "focus_areas": [],
        "session_id": None,
        "memory_config": {},
        "errors": [SEED_E],
        "warnings": [SEED_W],
        "status": "pending",
    }
    state.update(overrides)
    return state


@pytest.fixture
def offline_graph():
    with (
        patch("src.agents.base.audit_chain_mixin.get_audit_chain_service", return_value=None),
        patch(
            "src.agents.explainer.memory_hooks.get_explanation_memory_hooks",
            side_effect=RuntimeError("offline"),
        ),
    ):
        yield build_explainer_graph(use_llm=False, use_default_checkpointer=False)


@pytest.mark.asyncio
async def test_success_path_seeds_come_out_once(offline_graph):
    final = await offline_graph.ainvoke(_base_state())

    assert final["status"] == "completed"
    assert final["warnings"] == [SEED_W]
    assert final["errors"] == [SEED_E]


@pytest.mark.asyncio
async def test_failure_path_seeds_and_own_error_come_out_once(offline_graph):
    """assemble (no analysis results) -> error_handler."""
    final = await offline_graph.ainvoke(_base_state(analysis_results=[]))

    assert final["status"] == "failed"
    assert final["warnings"] == [SEED_W]
    assert final["errors"].count(SEED_E) == 1
    own = [e for e in final["errors"] if e.get("node") == "context_assembler"]
    assert len(own) == 1, final["errors"]
