"""Regression tests for #1358: AIMessage.content list-vs-str crash.

Sibling of #1350. ChatAnthropic on current Claude models returns
content-block LISTS; _parse_design_response/_parse_audit_response ran
re.search over the raw content and raised TypeError ("expected string or
bytes-like object, got 'list'"), so every experiment_designer dispatch
died as "Experiment design failed" once the #1353 budget let it run long
enough to reach the parse.
"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.experiment_designer.graph import create_initial_state
from src.agents.experiment_designer.nodes.design_reasoning import DesignReasoningNode
from src.agents.experiment_designer.nodes.validity_audit import ValidityAuditNode


class ContentBlockLLM:
    """Anthropic-style client: ainvoke returns content-block LIST, not str."""

    def __init__(self, text: str):
        self._text = text

    async def ainvoke(self, prompt: Any) -> Any:
        return SimpleNamespace(
            content=[
                {"type": "thinking", "thinking": "chain of thought..."},
                {"type": "text", "text": self._text},
            ],
            response_metadata={},
        )


DESIGN_JSON = {
    "design_type": "RCT",
    "design_rationale": "Randomization is feasible and eliminates confounding.",
}

AUDIT_JSON = {
    "overall_validity_score": 0.8,
    "validity_confidence": "high",
    "redesign_needed": False,
    "proceed_recommendation": "proceed",
    "internal_validity_threats": [],
    "external_validity_limits": [],
}


@pytest.mark.xdist_group(name="design_reasoning")
@pytest.mark.asyncio
async def test_design_reasoning_with_content_block_list():
    node = DesignReasoningNode()
    node.llm = ContentBlockLLM(f"```json\n{json.dumps(DESIGN_JSON)}\n```")

    state = create_initial_state(business_question="Does visit frequency improve engagement?")
    state["status"] = "designing"

    result = await node.execute(state)

    assert result["status"] == "calculating"
    assert result["design_type"] == "RCT"
    # The primary LLM must have succeeded — no fallback warning
    assert not any("fallback" in w.lower() for w in result.get("warnings", []))


@pytest.mark.asyncio
async def test_validity_audit_with_content_block_list():
    node = ValidityAuditNode()
    node.llm = ContentBlockLLM(f"```json\n{json.dumps(AUDIT_JSON)}\n```")

    state = create_initial_state(business_question="Test validity audit")
    state["status"] = "auditing"
    state["design_type"] = "RCT"
    state["treatments"] = [{"name": "Treatment", "description": "Test"}]
    state["outcomes"] = [{"name": "Outcome", "metric_type": "continuous"}]

    result = await node.execute(state)

    assert result["status"] in ["generating", "redesigning"]
    assert result["validity_confidence"] == "high"
    assert not any("timed out" in w.lower() for w in result.get("warnings", []))
