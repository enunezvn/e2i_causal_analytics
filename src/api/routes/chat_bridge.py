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
- The preamble never overstates the answer (#1451): it discloses that the
  deeper multi-agent analysis did not run, and it claims live platform data
  ONLY when a tool actually executed on the bridged turn.
- A fresh graph instance is created per call: the AG-UI module singleton
  carries a ``MemorySaver`` checkpointer, which in a long-lived API process
  would accumulate every bridged turn. (Explicit checkpointer also means no
  parent-checkpointer inheritance — the #1391 subgraph class does not apply.)
"""

import asyncio
import logging
import os
from typing import Any, NamedTuple, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from src.utils.llm_content import normalize_llm_content
from src.utils.redaction import redact_query
from src.utils.tool_evidence import payload_carries_evidence

logger = logging.getLogger(__name__)

# Honest preamble: the bridged answer is real tool-grounded data, but it is
# NOT the full multi-agent analysis the router chose — say so (#883 honesty).
#
# #1451: the honesty stays, the self-flagellation goes. The old wording
# ("The full analysis pipeline couldn't complete for this question…") led with
# the INTERNAL pipeline's outcome in the pipeline's own terms and buried a
# correct, grounded answer beneath an apology — three measured 2026-08-03 turns
# (TRx = 12,867; a refutation answer; a scoping answer) all read as a failed
# system apologising. Lead with what the answer IS, then disclose what did not
# run. Used ONLY when a tool actually executed — see BRIDGE_PREAMBLE_UNGROUNDED.
BRIDGE_PREAMBLE = (
    "Answered from live platform data, pulled through the analytics tools just now. "
    "The deeper multi-agent analysis did not run for this question."
)

# #1451: the bridged turn answered without executing a tool (e.g. a scoping
# answer). Claiming live platform data there would be a fabricated provenance
# claim, so the ungrounded variant states only what is actually true.
BRIDGE_PREAMBLE_UNGROUNDED = (
    "Answered directly by the platform assistant. "
    "The deeper multi-agent analysis did not run for this question."
)


class BridgeAnswer(NamedTuple):
    """The bridged answer plus the evidence needed to describe it honestly.

    ``tool_grounded`` is EVIDENCE, not a guess: it is True only when the AG-UI
    run executed a tool (a ``ToolMessage`` came back) AND at least one result
    was not a fail-closed ``{"success": false, ...}`` envelope (#1458, same
    rule as copilotkit's ``_evidence_tool_count`` for #1257). The preamble
    may only claim live platform data when it is True (#1451).
    """

    text: str
    tool_grounded: bool


def _first_user_action(failure_details: Optional[Sequence[Any]]) -> Optional[str]:
    """The primary failed agent's user-facing invitation, if the dispatcher wrote one.

    ``failure_details`` follows ``agent_results`` order, so the first entry
    carrying a ``user_action`` is the primary agent's. Only ONE is surfaced —
    two invitations for a single turn contradict each other (causal_impact's
    "name a treatment and an outcome" vs explainer's "run an analysis first").
    The shape is orchestrator-supplied, so anything unexpected degrades to no
    invitation rather than raising on the rescue path.
    """
    for detail in failure_details or []:
        if not isinstance(detail, dict):
            continue
        action = detail.get("user_action")
        if isinstance(action, str) and action.strip():
            return action.strip()
    return None


def build_bridge_preamble(
    *,
    tool_grounded: bool,
    failure_details: Optional[Sequence[Any]] = None,
) -> str:
    """Compose the preamble for a bridged turn (#1451).

    Leads with what the answer IS (its provenance), then discloses that the
    deeper multi-agent analysis did not run, then — when the dispatcher wrote
    one — the actionable invitation naming exactly what it would need. Before
    #1451 that invitation was discarded and replaced with a generic apology.
    """
    preamble = BRIDGE_PREAMBLE if tool_grounded else BRIDGE_PREAMBLE_UNGROUNDED
    action = _first_user_action(failure_details)
    return f"{preamble} {action}" if action else preamble


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
) -> Optional[BridgeAnswer]:
    """Run the query through the AG-UI chat brain; return its answer.

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

        # #1475: fast tool-selection leg. Measured 2026-08-04 (live probes +
        # same-box/key/prompt experiments): the bridge's chat leg is purely a
        # tool router on bridged turns (0 content chars, ~86-token tool call
        # every run) and sonnet-5 spent 3.1-5.8s on it; haiku-4-5 selected the
        # IDENTICAL tool with equivalent args on 3/3 real bridged queries in
        # 1.17-1.33s. Thinking (medium vs none: ~0.5s) and prompt caching
        # (cache-hit TTFT unchanged) were both measured immaterial. The
        # synthesize leg — the user-facing prose author — stays on the
        # standard tier, and the AG-UI brain keeps its defaults (two-brain
        # separation). Residual tradeoff, stated: a bridged turn that answers
        # WITHOUT a tool call gets fast-tier prose behind the ungrounded
        # preamble.
        graph = create_e2i_chat_agent(
            chat_llm_tier="fast",
            chat_llm_reasoning_effort="none",
        )
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

        messages = final_state.get("messages") or []
        # #1451: a ToolMessage is the only real evidence a tool EXECUTED. The
        # preamble's "live platform data" claim is gated on it — an answer the
        # model produced without touching a tool must not be dressed up as a
        # data lookup.
        #
        # #1458: execution alone is NOT evidence. E2I tools fail closed with a
        # {"success": false, ...} envelope that is still a ToolMessage, so
        # presence-gating stamped all-tools-errored turns with the live-data
        # preamble — a stronger trust signal for a weaker answer, persisted as
        # top-reward training rows (the #1257 defect class). Apply the same
        # rule as copilotkit's _evidence_tool_count: only payloads not
        # positively marked failed count; non-envelope payloads still count.
        tool_grounded = any(
            isinstance(m, ToolMessage) and payload_carries_evidence(m.content) for m in messages
        )
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                text = normalize_llm_content(msg.content).strip()
                if text:
                    return BridgeAnswer(text=text, tool_grounded=tool_grounded)
        logger.warning("Chat bridge produced no assistant text for query=%s", redact_query(query))
        return None
    except Exception as e:
        logger.warning("Chat bridge failed (%s) for query=%s", e, redact_query(query))
        return None
