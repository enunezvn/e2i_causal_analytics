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

    def __init__(
        self,
        canned_content: str,
        model_name: str = "marked-mock-llm",
        phase_responses: "list[tuple[str, str]] | None" = None,
    ):
        self._content = canned_content
        self.model_name = model_name
        # Optional multi-phase support: an ordered list of (keyword, content).
        # When set, ainvoke/invoke scan the call's message text (LangChain
        # SystemMessage/HumanMessage ``.content``) and return the content for the
        # FIRST matching keyword, else ``canned_content``. This lets multi-call
        # agents (e.g. tool_composer's decompose -> plan -> synthesize) receive a
        # phase-appropriate canned payload from a single mock. Order matters
        # (e.g. match "synth" before "tool", since synthesis prompts mention tools).
        self._phase_responses = phase_responses or []

    def _select(self, args: Any, kwargs: Any) -> str:
        if not self._phase_responses:
            return self._content
        text_parts: list[str] = []
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, (list, tuple)):
                for item in value:
                    text_parts.append(str(getattr(item, "content", "")))
            else:
                text_parts.append(str(getattr(value, "content", value)))
        text = " ".join(text_parts).lower()
        for keyword, content in self._phase_responses:
            if keyword.lower() in text:
                return content
        return self._content

    async def ainvoke(self, *args: Any, **kwargs: Any) -> MarkedMockResponse:
        return MarkedMockResponse(self._select(args, kwargs))

    def invoke(self, *args: Any, **kwargs: Any) -> MarkedMockResponse:
        return MarkedMockResponse(self._select(args, kwargs))

    def bind(self, **kwargs: Any) -> "MarkedMockChatLLM":
        return self

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "MarkedMockChatLLM":
        return self


def llm_or_marked_mock(factory: Any, canned: str, **factory_kwargs: Any) -> Any:
    """Call an ``llm_factory`` function; on a missing-key ``ValueError`` return a
    MARKED mock IFF ``E2I_ALLOW_MOCK_LLM`` is set, else re-raise (fail-loud).

    ``factory`` is e.g. ``get_chat_llm`` / ``get_fast_llm`` / ``get_standard_llm``;
    ``canned`` is the agent-appropriate, parser-valid content the mock returns.
    Passing the factory in keeps this util free of llm_factory imports.
    """
    try:
        return factory(**factory_kwargs)
    except ValueError:
        if not mock_llm_allowed():
            raise
        return MarkedMockChatLLM(canned)
