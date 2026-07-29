"""Regression: feedback_learner nodes must handle AIMessage.content block lists (#1350/#1358 sweep).

Both nodes' LLM paths are latent today (``use_llm=True`` callers never inject
an ``llm`` — see the dspy_optimization_tasks intent gap), but the moment one is
injected it will be an adaptive-thinking model returning content-block LISTS.
Pre-fix, ``_parse_recommendations``/``_parse_patterns`` ran ``re.search`` on
the list → TypeError → swallowed → silent deterministic fallback.
"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.feedback_learner.nodes.learning_extractor import LearningExtractorNode
from src.agents.feedback_learner.nodes.pattern_analyzer import PatternAnalyzerNode

EXTRACTION_JSON = {
    "recommendations": [
        {
            "recommendation_id": "R1",
            "category": "data_update",
            "description": "Tighten the KPI prompt to cite TRx sources.",
            "affected_agents": ["explainer"],
            "expected_impact": "Fewer unsourced KPI answers",
            "implementation_effort": "low",
            "priority": 1,
        }
    ]
}

PATTERNS_JSON = {
    "patterns": [
        {
            "pattern_id": "P1",
            "pattern_type": "accuracy_issue",
            "description": "Wrong TRx figures for West-region queries.",
            "severity": "high",
            "frequency": 3,
        }
    ]
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
async def test_extract_with_llm_content_block_list():
    text = f"```json\n{json.dumps(EXTRACTION_JSON)}\n```"
    node = LearningExtractorNode(use_llm=True, llm=ContentBlockLLM(text))
    state: dict[str, Any] = {
        "detected_patterns": [
            {
                "pattern_id": "P1",
                "description": "Wrong TRx figures",
                "pattern_type": "accuracy_issue",
                "severity": "high",
                "affected_agents": ["explainer"],
                "root_cause_hypothesis": "Stale TRx snapshot in the KPI cache",
            }
        ],
        "status": "extracting",
    }

    result = await node._extract_with_llm(state)

    # Pre-fix this silently fell back to _extract_deterministic — assert the
    # parsed recommendation, which only the LLM path can produce.
    assert [r["description"] for r in result["recommendations"]] == [
        "Tighten the KPI prompt to cite TRx sources."
    ]
    assert result["recommendations"][0]["recommendation_id"] == "R1"


@pytest.mark.asyncio
async def test_analyze_with_llm_content_block_list():
    text = f"```json\n{json.dumps(PATTERNS_JSON)}\n```"
    node = PatternAnalyzerNode(use_llm=True, llm=ContentBlockLLM(text), prefer_optimized=False)
    state: dict[str, Any] = {
        "feedback_items": [
            {
                "feedback_type": "thumbs_down",
                "source_agent": "explainer",
                "query": "What is TRx for Kisqali in the West?",
                "agent_response": "TRx is 9,999.",
                "user_feedback": "Numbers look wrong",
            }
        ],
        "feedback_summary": {"total_count": 1},
        "status": "analyzing",
    }

    result = await node._analyze_with_llm(state)

    # Pre-fix this silently fell back to _analyze_deterministic — assert the
    # parsed pattern, which only the LLM path can produce.
    assert [p["description"] for p in result["patterns"]] == [
        "Wrong TRx figures for West-region queries."
    ]
