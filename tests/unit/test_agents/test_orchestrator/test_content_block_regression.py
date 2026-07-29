"""Regression: orchestrator nodes must handle AIMessage.content block lists (#1350/#1358 sweep).

ChatAnthropic on adaptive-thinking models returns ``content`` as a list of
blocks (``[{"type": "thinking", ...}, {"type": "text", ...}]``) instead of a
plain str. The fast tier (haiku-4.5) returns str today, so these sites are
latent, not live-broken — but a model-tier upgrade flips the content shape,
which is exactly how the tool_composer (#1350) and experiment_designer (#1358)
crashes shipped. These tests pin the normalized behavior.

Pre-fix failure modes reproduced here:
- ``IntentClassifierNode._llm_classify``: ``json.loads(list)`` raises TypeError,
  swallowed by the broad except → silent misroute to ``general`` @ 0.3.
- ``SynthesizerNode._synthesize_multiple``: the block LIST lands in
  ``result["response"]`` with NO exception — user-visible garbage.
"""

import json
from types import SimpleNamespace
from typing import Any, List

import pytest

from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode
from src.agents.orchestrator.nodes.synthesizer import SynthesizerNode


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


class TestIntentClassifierContentBlocks:
    @pytest.mark.asyncio
    async def test_llm_classify_with_content_block_list(self):
        payload = json.dumps(
            {"primary_intent": "causal_effect", "confidence": 0.9, "requires_multi_agent": False}
        )
        node = IntentClassifierNode()
        node.llm = ContentBlockLLM(payload)

        classification = await node._llm_classify("what moved kisqali trx and why")

        # Pre-fix this silently fell back to general @ 0.3 — assert the parsed
        # values, not just "a classification came back".
        assert classification["primary_intent"] == "causal_effect"
        assert classification["confidence"] == 0.9
        assert classification["requires_multi_agent"] is False


class TestSynthesizerContentBlocks:
    @pytest.mark.asyncio
    async def test_synthesize_multiple_with_content_block_list(self):
        synthesis_text = "KEY FINDING: Kisqali TRx grew 12% in the West region."
        node = SynthesizerNode()
        node.llm = ContentBlockLLM(synthesis_text)

        results: List[Any] = [
            {
                "agent_name": "causal_impact",
                "success": True,
                "result": {
                    "narrative": "Rep visits drove a measurable TRx lift.",
                    "confidence": 0.8,
                    "recommendations": [],
                },
            },
            {
                "agent_name": "gap_analyzer",
                "success": True,
                "result": {
                    "narrative": "West region has the largest untapped opportunity.",
                    "confidence": 0.7,
                    "recommendations": [],
                },
            },
        ]

        out = await node._synthesize_multiple(results)

        # Pre-fix the raw block LIST landed in out["response"] without any
        # exception — the strictest honest assertion is exact-str equality.
        assert out["response"] == synthesis_text
        assert isinstance(out["response"], str)
