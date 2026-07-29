"""Regression tests for #1350: AIMessage.content list-vs-str crash.

ChatAnthropic on current Claude models returns content-block LISTS
([{"type": "thinking", ...}, {"type": "text", ...}]) instead of plain str.
The decomposer/planner/synthesizer cast response.content to str and fed it
to json.loads, which raised TypeError ("the JSON object must be str, bytes
or bytearray, not list") on every multi_faceted orchestrator dispatch.
"""

from types import SimpleNamespace
from typing import Any, List

import pytest

from src.agents.tool_composer.decomposer import QueryDecomposer
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    ExecutionPlan,
)
from src.agents.tool_composer.planner import ToolPlanner
from src.agents.tool_composer.synthesizer import ResponseSynthesizer


class ContentBlockLLM:
    """Wraps a MockLLMClient but returns Anthropic-style content-block lists.

    Routes through the wrapped client so the same decomposition/planning/
    synthesis response detection applies; only the content SHAPE changes.
    """

    def __init__(self, inner: Any):
        self._inner = inner

    async def ainvoke(self, messages: List[Any]) -> Any:
        resp = await self._inner.ainvoke(messages)
        return SimpleNamespace(
            content=[
                {"type": "thinking", "thinking": "chain of thought..."},
                {"type": "text", "text": resp.content},
            ],
            response_metadata={},
        )


@pytest.mark.asyncio
async def test_decompose_with_content_block_list(mock_llm_client, sample_query):
    decomposer = QueryDecomposer(llm_client=ContentBlockLLM(mock_llm_client))
    result = await decomposer.decompose(sample_query)

    assert isinstance(result, DecompositionResult)
    assert len(result.sub_questions) >= 2


@pytest.mark.asyncio
async def test_plan_with_content_block_list(
    mock_llm_client, mock_tool_registry, sample_decomposition
):
    planner = ToolPlanner(
        llm_client=ContentBlockLLM(mock_llm_client), tool_registry=mock_tool_registry
    )
    plan = await planner.plan(sample_decomposition)

    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) > 0


@pytest.mark.asyncio
async def test_synthesize_with_content_block_list(mock_llm_client, sample_synthesis_input):
    synthesizer = ResponseSynthesizer(llm_client=ContentBlockLLM(mock_llm_client))
    response = await synthesizer.synthesize(sample_synthesis_input)

    # Must be the LLM-parsed answer, not the silent exception-fallback response
    assert response.answer == (
        "Rep visits show a 15% causal lift in Rx volume with regional variation."
    )
    assert response.confidence == 0.85
    assert response.synthesis_reasoning == "Combined causal analysis with regional breakdown"
