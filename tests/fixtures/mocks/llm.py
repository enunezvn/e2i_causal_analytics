"""Centralized LLM mocking for E2I tests.

Provides consistent mock implementations for LLM clients used across
agent tests. Two levels of sophistication:
- SimpleMockLLM: Basic mock for simple tests
- MockLLMClient: Advanced mock with phase detection for complex workflows

Usage:
    # Simple mock (for basic unit tests)
    from tests.fixtures.mocks.llm import mock_simple_llm

    def test_something(mock_simple_llm):
        mock_simple_llm.return_value = "expected response"
        ...

    # Advanced mock (for workflow tests)
    from tests.fixtures.mocks.llm import MockLLMClient

    client = MockLLMClient(responses={
        "decomposition": "decomposed result",
        "planning": "planned result",
        "synthesis": "synthesized result",
    })
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock

import pytest


class SimpleMockLLM:
    """Simple mock LLM for basic unit tests.

    Provides a lightweight mock that returns configurable responses
    via ainvoke(). Suitable for tests that don't need sophisticated
    phase detection or response routing.
    """

    def __init__(self, default_response: str = "Mock LLM response"):
        self.default_response = default_response
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []
        self._ainvoke = AsyncMock(side_effect=self._handle_invoke)

    async def _handle_invoke(self, *args, **kwargs) -> str:
        self.call_count += 1
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.default_response

    async def ainvoke(self, *args, **kwargs) -> str:
        """Async invoke that returns the configured response."""
        return await self._ainvoke(*args, **kwargs)

    def set_response(self, response: str) -> None:
        """Set the response for subsequent calls."""
        self.default_response = response
        self._ainvoke.return_value = response

    def set_responses(self, responses: List[str]) -> None:
        """Set multiple responses (returned in order)."""
        self._ainvoke.side_effect = responses


class MockLLMClient:
    """Advanced mock LLM client with phase detection.

    Detects the current workflow phase from system prompts and returns
    appropriate responses. Supports:
    - Decomposition phase
    - Planning phase
    - Execution phase
    - Synthesis phase

    Usage:
        client = MockLLMClient(responses={
            "decomposition": json.dumps({"sub_questions": [...]}),
            "planning": json.dumps({"execution_plan": [...]}),
            "synthesis": json.dumps({"final_answer": "..."}),
        })
    """

    # Phase detection patterns
    PHASE_PATTERNS = {
        "decomposition": [
            r"decompos",
            r"sub.?question",
            r"break.?down",
            r"split.?query",
        ],
        "planning": [
            r"plan",
            r"orchestrat",
            r"sequence",
            r"order.?of.?execution",
        ],
        "execution": [
            r"execut",
            r"tool.?call",
            r"invoke",
            r"run.?tool",
        ],
        "synthesis": [
            r"synthes",
            r"combin",
            r"aggregat",
            r"final.?answer",
            r"summar",
        ],
        "analysis": [
            r"analyz",
            r"assess",
            r"evaluat",
            r"review",
        ],
    }

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "Mock response",
    ):
        self.responses = responses or {}
        self.default_response = default_response
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []
        self.phase_calls: Dict[str, int] = {}

    def _detect_phase(self, messages: List[Dict[str, Any]]) -> str:
        """Detect the workflow phase from message content."""
        # Combine all message content for pattern matching
        text = " ".join(
            str(m.get("content", "")) for m in messages
        ).lower()

        # Check each phase's patterns
        for phase, patterns in self.PHASE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return phase

        return "unknown"

    async def ainvoke(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Async invoke with phase detection."""
        self.call_count += 1

        # Detect phase and get response
        phase = self._detect_phase(messages)
        self.phase_calls[phase] = self.phase_calls.get(phase, 0) + 1

        # Store call for inspection
        self.calls.append({
            "messages": messages,
            "phase": phase,
            "kwargs": kwargs,
        })

        # Get response for detected phase
        response_content = self.responses.get(phase, self.default_response)

        return {
            "content": response_content,
            "role": "assistant",
        }

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Sync invoke (wraps ainvoke for sync tests)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(messages, **kwargs)
        )

    def set_phase_response(self, phase: str, response: str) -> None:
        """Set response for a specific phase."""
        self.responses[phase] = response

    def get_calls_for_phase(self, phase: str) -> List[Dict[str, Any]]:
        """Get all calls that were detected as a specific phase."""
        return [c for c in self.calls if c["phase"] == phase]

    def reset(self) -> None:
        """Reset call tracking."""
        self.call_count = 0
        self.calls = []
        self.phase_calls = {}


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_simple_llm() -> SimpleMockLLM:
    """Fixture providing a simple mock LLM for basic tests."""
    return SimpleMockLLM()


@pytest.fixture
def mock_advanced_llm() -> MockLLMClient:
    """Fixture providing an advanced mock LLM with phase detection."""
    return MockLLMClient(
        responses={
            "decomposition": json.dumps({
                "sub_questions": [
                    {"id": "q1", "question": "Sub question 1"},
                    {"id": "q2", "question": "Sub question 2"},
                ],
            }),
            "planning": json.dumps({
                "execution_plan": [
                    {"step": 1, "tool": "tool_a", "question_id": "q1"},
                    {"step": 2, "tool": "tool_b", "question_id": "q2"},
                ],
            }),
            "synthesis": json.dumps({
                "final_answer": "Synthesized answer from sub-question results.",
                "confidence": 0.85,
            }),
        }
    )


@pytest.fixture
def mock_llm(request) -> Union[SimpleMockLLM, MockLLMClient]:
    """Smart fixture that provides appropriate mock based on test needs.

    Use via marker:
        @pytest.mark.parametrize("mock_llm_type", ["simple"])
        def test_basic(mock_llm):
            # Gets SimpleMockLLM

        @pytest.mark.parametrize("mock_llm_type", ["advanced"])
        def test_complex(mock_llm):
            # Gets MockLLMClient

    Default: SimpleMockLLM
    """
    # Check for marker or parameter
    mock_type = getattr(request, "param", "simple")
    if hasattr(request, "node"):
        marker = request.node.get_closest_marker("mock_llm_type")
        if marker:
            mock_type = marker.args[0] if marker.args else "simple"

    if mock_type == "advanced":
        return MockLLMClient()
    return SimpleMockLLM()
