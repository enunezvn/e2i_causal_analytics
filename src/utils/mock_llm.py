"""Opt-in MARKED mock LLMs for keyless test / harness contexts (issue #606).

Extends the #471 anti-mock discipline (see
``src/agents/experiment_designer/nodes/validity_audit.py``) to the other
LLM-dependent agents the Tier 1-5 harness exercises. The harness CI deliberately
has no LLM API key; rather than add a real key (cost, per #504) or fake agents
green silently, agents may construct a **clearly-marked** mock LLM, but ONLY:

  * when the explicit opt-in flag ``E2I_ALLOW_MOCK_LLM`` is set, AND
  * a real provider key is absent (the factory would otherwise raise).

Production (no flag) stays **fail-loud**: a missing key still raises, so no
silent-mock value ever reaches a real request. Every mock response carries an
in-band ``mock_response_for_dev_only=True`` marker so downstream audits /
``DataSourceValidator`` can distinguish synthetic output from real LLM output.

Usage at an LLM construction site (preserves fail-loud)::

    from src.utils.mock_llm import MarkedMockChatLLM, mock_llm_allowed
    try:
        self.llm = get_chat_llm(model_tier="reasoning")
    except ValueError:
        if not mock_llm_allowed():
            raise  # prod / no opt-in -> fail loud (missing key)
        self.llm = MarkedMockChatLLM(CANNED_CONTENT)
"""

from __future__ import annotations

import os
from typing import Any

#: Opt-in flag. Set ONLY in CI/test workflow env (never production deploy config).
MOCK_LLM_FLAG = "E2I_ALLOW_MOCK_LLM"

#: In-band marker stamped on every mock response.
MOCK_MARKER = "mock_response_for_dev_only"

_TRUTHY = {"1", "true", "yes", "on"}


def mock_llm_allowed(flag: str = MOCK_LLM_FLAG) -> bool:
    """True only if the explicit opt-in flag is set to a truthy value."""
    return os.environ.get(flag, "").strip().lower() in _TRUTHY


class MarkedMockResponse:
    """Minimal stand-in for a LangChain chat message (``.content`` + metadata).

    Carries the dev-only marker in ``response_metadata`` and
    ``additional_kwargs`` so it is never mistaken for real model output.
    """

    def __init__(self, content: str):
        self.content = content
        self.response_metadata = {
            MOCK_MARKER: True,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        self.additional_kwargs = {MOCK_MARKER: True}

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"MarkedMockResponse(mock_response_for_dev_only=True, content={self.content[:40]!r}...)"
        )


class MarkedMockChatLLM:
    """A LangChain-chat-compatible marked mock that returns canned content.

    Implements the small surface the agents actually call: ``ainvoke`` /
    ``invoke`` (returning a :class:`MarkedMockResponse`) plus chainable
    ``bind`` / ``with_structured_output`` no-ops so call sites that wrap the LLM
    keep working. It is NOT a real model — it always returns ``canned_content``.
    """

    mock_response_for_dev_only = True

    def __init__(self, canned_content: str, model_name: str = "marked-mock-llm"):
        self._content = canned_content
        self.model_name = model_name

    async def ainvoke(self, *args: Any, **kwargs: Any) -> MarkedMockResponse:
        return MarkedMockResponse(self._content)

    def invoke(self, *args: Any, **kwargs: Any) -> MarkedMockResponse:
        return MarkedMockResponse(self._content)

    def bind(self, **kwargs: Any) -> "MarkedMockChatLLM":
        return self

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "MarkedMockChatLLM":
        return self
