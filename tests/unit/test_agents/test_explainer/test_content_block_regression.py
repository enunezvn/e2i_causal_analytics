"""Regression: DeepReasonerNode must handle AIMessage.content block lists (#1350/#1358 sweep).

The explainer's LLM path is only reachable when an ``llm`` is injected (all
current prod constructors pass ``llm=None``), so this site is latent — but the
injected client would be the standard tier (claude-sonnet-5, adaptive
thinking), which returns content-block LISTS. Pre-fix, ``_parse_reasoning``
ran ``re.search`` on the list → TypeError → swallowed → silent deterministic
fallback.
"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.explainer.nodes.deep_reasoner import DeepReasonerNode

REASONING_JSON = {
    "insights": [
        {
            "insight_id": "I1",
            "statement": "Rep visits show a 15% causal lift in Rx volume.",
            "importance": 0.9,
        }
    ],
    "structure": ["Summary", "Findings", "Recommendations"],
    "themes": ["field-force effectiveness"],
}


class ContentBlockLLM:
    """Stub LLM returning Anthropic-style content-block lists."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def ainvoke(self, prompt: Any) -> Any:
        return SimpleNamespace(
            content=[
                {"type": "thinking", "thinking": "chain of thought..."},
                {"type": "text", "text": self._text},
            ],
            response_metadata={},
        )


@pytest.mark.asyncio
async def test_reason_with_llm_content_block_list():
    text = f"```json\n{json.dumps(REASONING_JSON)}\n```"
    node = DeepReasonerNode(use_llm=True, llm=ContentBlockLLM(text))

    parsed = await node._reason_with_llm({"analysis_context": [], "status": "reasoning"})

    # Pre-fix this silently fell back to _reason_deterministic — assert the
    # parsed payload, which only the LLM path can produce.
    assert parsed["themes"] == ["field-force effectiveness"]
    assert parsed["insights"][0]["statement"] == ("Rep visits show a 15% causal lift in Rx volume.")
