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
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

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

# JWT-verified user id for the current request context. The CopilotKit
# runtime mints bare-UUID threadIds (no user~ prefix on any real chat
# session), so session-prefix derivation alone leaves every chat row
# unattributed; the auth gate stashes the verified identity here and
# set_chat_attribution falls back to it. Verified-or-nothing: never a
# guessed or fabricated id.
_authenticated_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "llm_authenticated_user_id", default=None
)


def set_authenticated_user(user_id: Optional[str]) -> None:
    """Stash the token-verified user id for attribution fallback.

    Rejects non-UUID ids (TESTING_MODE's 'test-user-id') and the anonymous
    UUID — llm_usage_events.user_id is a UUID column and attribution is
    honest-only.
    """
    if user_id and user_id != ANONYMOUS_USER_ID:
        try:
            _uuid.UUID(user_id)
        except ValueError:
            user_id = None
    else:
        user_id = None
    _authenticated_user_id.set(user_id)


def get_authenticated_user_id() -> Optional[str]:
    """The token-verified user of the current request context, or None.

    The public read of the fallback channel above. It is set in the REQUEST
    task, upstream of the SSE keepalive wrapper, so it has always been readable
    from inside a chat graph's nodes (#2077) — including while the attribution
    contextvar was not, because the wrapper pulled each frame in a fresh task
    that discarded its writes. #2100 gave those pulls one shared context, so
    both channels now survive; this one is still the honest-or-nothing source,
    and set_chat_attribution derives its user_id from it.
    """
    return _authenticated_user_id.get()


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


def resolve_session_user_id(session_id: Optional[str]) -> Optional[str]:
    """The user a chat turn belongs to — verified identity first, prefix second.

    The ``{user}~`` prefix is a claim the caller makes, not a credential: AG-UI
    reads ``threadId`` straight from the request body and ``/chat`` takes
    ``session_id`` from it, so a prefix-first rule let one caller persist a turn
    under another user's ownership (``chatbot_conversations.user_id`` is written
    from this attribution, and migration 123's trigger turns it into the
    ``computed_user_id`` the RLS policies read). Every chat entry point now binds
    the verified channel, so the prefix answers only where none was bound at all
    — a direct caller, or a background task replaying a session.
    A disagreement is logged, not raised — the request-level guards reject it;
    this is the last honest read, not the gate.
    """
    verified = _authenticated_user_id.get()
    claimed = user_id_from_session(session_id)
    if verified is None:
        return claimed
    if claimed is not None and claimed != verified:
        logger.warning(
            "Chat session prefix names user %s but the verified request user is %s; "
            "attributing to the verified user (session=%s)",
            claimed,
            verified,
            session_id,
        )
    return verified


def set_chat_attribution(session_id: str, request_id: Optional[str] = None) -> LLMAttribution:
    user_id = resolve_session_user_id(session_id)
    attr = LLMAttribution(
        user_id=user_id,
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
