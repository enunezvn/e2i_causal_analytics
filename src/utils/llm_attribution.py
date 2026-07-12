"""Per-run LLM attribution contextvar (admin observability, spec 2026-07-12).

Chat entrypoints call set_chat_attribution() at run start; both capture hooks
(the llm_factory LangChain callback and the global litellm logger) read it to
attribute usage rows to a user/session. Unset => honest platform-level rows
(NULL user/session, surface fallback 'other'), never a guessed attribution.

Also carries the per-run token accumulator that message persistence drains
into chatbot_messages.tokens_used / model_used. Drain = read-and-reset, so
each assistant row carries tokens accrued since the previous drained row and
sums across a session never double-count.
"""

import contextvars
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Optional

ANONYMOUS_USER_ID = "00000000-0000-0000-0000-000000000000"


@dataclass
class RunUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    last_model: Optional[str] = None


@dataclass
class LLMAttribution:
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    surface: str = "other"
    component: Optional[str] = None
    request_id: Optional[str] = None
    usage: RunUsage = field(default_factory=RunUsage)


_attribution: contextvars.ContextVar[Optional[LLMAttribution]] = contextvars.ContextVar(
    "llm_attribution", default=None
)


def user_id_from_session(session_id: Optional[str]) -> Optional[str]:
    """'<user_id>~<uuid>' -> user_id. Anonymous, malformed, or non-UUID
    prefixes -> None (honest NULL, never fabricated)."""
    if not session_id or "~" not in session_id:
        return None
    prefix = session_id.split("~", 1)[0]
    try:
        _uuid.UUID(prefix)
    except ValueError:
        return None
    return None if prefix == ANONYMOUS_USER_ID else prefix


def set_chat_attribution(session_id: str, request_id: Optional[str] = None) -> LLMAttribution:
    attr = LLMAttribution(
        user_id=user_id_from_session(session_id),
        session_id=session_id,
        surface="chat",
        request_id=request_id,
    )
    _attribution.set(attr)
    return attr


def set_platform_attribution(surface: str, component: Optional[str] = None) -> LLMAttribution:
    attr = LLMAttribution(surface=surface, component=component)
    _attribution.set(attr)
    return attr


def get_attribution() -> Optional[LLMAttribution]:
    return _attribution.get()


def clear_attribution() -> None:
    _attribution.set(None)


def record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """Accumulate into the current run; no-op when no attribution is set."""
    attr = _attribution.get()
    if attr is None:
        return
    attr.usage.input_tokens += input_tokens
    attr.usage.output_tokens += output_tokens
    attr.usage.last_model = model


def drain_run_usage() -> Optional[RunUsage]:
    """Return-and-reset the run accumulator; None when nothing was recorded."""
    attr = _attribution.get()
    if attr is None or (attr.usage.input_tokens == 0 and attr.usage.output_tokens == 0):
        return None
    drained = attr.usage
    attr.usage = RunUsage()
    return drained
