"""#1336 D5 conversational bridge: /chat/stream fallback to the AG-UI chat brain.

The platform has two chat brains (2026-07-29 two-brain finding):

- ``/api/copilotkit/chat/stream`` — classify → orchestrator → generate. The
  orchestrator's agents demand structured inputs; when none can bind real data
  the turn fails closed by design (#883) and the user gets an error summary.
- ``/api/copilotkit/agent/default`` (AG-UI) — chat_node + tools. Answers the
  same questions with real grounded data pulled through the bound tools.

Owner decision on #1336: BRIDGE. When the orchestrator fails completely (zero
successful agents), route the turn through the AG-UI brain and stream its
grounded answer behind an honest preamble instead of the bare error summary.

Discipline preserved:
- Fires ONLY on complete failure — partial and full successes are untouched.
- Fails open to the status quo: any bridge error/timeout returns ``None`` and
  the caller keeps the original fail-closed summary. The bridge can only
  improve on the status quo, never mask it.
- A fresh graph instance is created per call: the AG-UI module singleton
  carries a ``MemorySaver`` checkpointer, which in a long-lived API process
  would accumulate every bridged turn. (Explicit checkpointer also means no
  parent-checkpointer inheritance — the #1391 subgraph class does not apply.)
"""

import asyncio
import logging
import os
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.utils.llm_content import normalize_llm_content
from src.utils.redaction import redact_query

logger = logging.getLogger(__name__)

# Honest preamble: the bridged answer is real tool-grounded data, but it is
# NOT the full multi-agent analysis the router chose — say so (#883 honesty).
BRIDGE_PREAMBLE = (
    "The full analysis pipeline couldn't complete for this question, "
    "so here's what I can tell you from the data available:"
)

_BRIDGE_HISTORY_CAP = 8


def _bridge_enabled() -> bool:
    # Read at call time (not import time) so the kill switch works in tests
    # and per-container without module reloads.
    return os.getenv("E2I_CHAT_BRIDGE_ENABLED", "true").lower() == "true"


_DEFAULT_BRIDGE_TIMEOUT_S = 90.0


def _bridge_timeout_s() -> float:
    raw = os.getenv("E2I_CHAT_BRIDGE_TIMEOUT_S", "")
    try:
        return float(raw) if raw else _DEFAULT_BRIDGE_TIMEOUT_S
    except ValueError:
        logger.warning(
            "Invalid E2I_CHAT_BRIDGE_TIMEOUT_S=%r; using default %.0fs",
            raw,
            _DEFAULT_BRIDGE_TIMEOUT_S,
        )
        return _DEFAULT_BRIDGE_TIMEOUT_S


def _prepare_bridge_messages(query: str, history: Optional[list[Any]]) -> list[BaseMessage]:
    """Last few conversation messages, guaranteed to end with the user query.

    The AG-UI brain expects the full message history resent each turn; the
    chatbot-graph state carries it. Cap it so a long session doesn't blow the
    bridge model's context, and append the query when the history doesn't
    already end on it (e.g. empty history, or trailing assistant message).
    """
    msgs = [m for m in (history or []) if isinstance(m, BaseMessage)]
    msgs = msgs[-_BRIDGE_HISTORY_CAP:]
    if not msgs or not isinstance(msgs[-1], HumanMessage):
        msgs = [*msgs[-(_BRIDGE_HISTORY_CAP - 1) :], HumanMessage(content=query)]
    return msgs


async def run_conversational_bridge(
    query: str,
    session_id: str,
    history: Optional[list[Any]] = None,
    timeout_s: Optional[float] = None,
) -> Optional[str]:
    """Run the query through the AG-UI chat brain; return its answer text.

    Returns ``None`` on any failure (disabled, timeout, provider error, no
    assistant text) — the caller must then keep its existing fail-closed
    response. Never raises.
    """
    # The entire body is guarded: an escaped exception would be swallowed by
    # orchestrator_node's broad except, dropping the orchestrator result and
    # falling through to generate_node — a behavior change, not the intended
    # keep-the-fail-closed-summary fallback.
    try:
        if not _bridge_enabled():
            return None

        # Lazy import: copilotkit.py imports chatbot_graph lazily in the other
        # direction; importing it at module level here would load the whole
        # CopilotKit SDK surface on chatbot_graph import.
        from src.api.routes.copilotkit import _session_id_context, create_e2i_chat_agent

        graph = create_e2i_chat_agent()
        effective_timeout = timeout_s if timeout_s is not None else _bridge_timeout_s()
        # Shadow session: chat_node persists messages keyed on this contextvar
        # (observed in a real local bridge run). Under the real session id a
        # bridged turn would double-write it — chat_node's raw answer plus
        # finalize's preambled answer. The '~bridge' suffix keeps the user
        # prefix (computed_user_id splits on the first '~') while keeping the
        # real session's UI-loaded history clean; bridge rows remain audited
        # under the shadow session.
        bridge_session_id = f"{session_id}~bridge"
        token = _session_id_context.set(bridge_session_id)
        try:
            final_state = await asyncio.wait_for(
                graph.ainvoke(
                    {
                        "messages": _prepare_bridge_messages(query, history),
                        "session_id": bridge_session_id,
                    },
                    config={"configurable": {"thread_id": f"bridge~{session_id}"}},
                ),
                timeout=effective_timeout,
            )
        except TimeoutError:
            logger.warning(
                "Chat bridge timed out after %.0fs for query=%s",
                effective_timeout,
                redact_query(query),
            )
            return None
        finally:
            _session_id_context.reset(token)

        for msg in reversed(final_state.get("messages") or []):
            if isinstance(msg, AIMessage):
                text = normalize_llm_content(msg.content).strip()
                if text:
                    return text
        logger.warning("Chat bridge produced no assistant text for query=%s", redact_query(query))
        return None
    except Exception as e:
        logger.warning("Chat bridge failed (%s) for query=%s", e, redact_query(query))
        return None
