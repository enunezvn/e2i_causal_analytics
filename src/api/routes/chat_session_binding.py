"""Session binding for the chat graphs' tools node (#2064, PR #2088).

``SessionBoundToolNode`` is the tools node both chat graphs mount. It binds the
turn's session into ``chat_session_id_context`` for the duration of its tool
calls and resets it afterwards, so ``orchestrator_tool`` and
``tool_composer_tool`` record the real conversation instead of inventing a
``chatbot-<ts>`` id; with no session to bind it binds nothing and the caller's
binding, if any, still applies. An outer binding could not be relied on for the
browser route: AG-UI's ``execute()`` binds the session while its first frame is
pulled, and ``with_sse_keepalive`` used to pull each later frame in a fresh task
that dropped it, so the graph ran without it. #2100 gave those pulls one shared
context and the binding now arrives as well, but state stays what this node
reads — one channel, and the one the graph owns. The node lives
here rather than in ``chatbot_tools`` because it belongs to neither graph in
particular and to the tools module only by way of the context variable it sets.

#2077: the session a turn PERSISTS under is not always the one its tools belong
to. The chat bridge runs a whole AG-UI turn under ``{session}~bridge`` (#1394's
shadow, which keeps the bridged raw answer out of the real session's UI history);
its tools, whose compositions a rating on the real conversation must be able to
match, belong to the unsuffixed session. A caller that needs that split names the
tools' session in ``config["configurable"]["tool_session_id"]`` — measured
2026-09-14: LangGraph drops input keys that are not declared state channels, and
``configurable`` reaches this node's ``config`` untouched. Absent that key the
node binds ``state["session_id"]`` exactly as before, so nothing changes for the
AG-UI browser route or ``/chat/stream``.
"""

import contextvars
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from src.api.routes.chatbot_tools import reset_chat_session_id, set_chat_session_id


def _bind_tool_session(
    state: Any, config: Optional[RunnableConfig] = None
) -> "Optional[contextvars.Token[Optional[str]]]":
    # The caller's explicit "these tools belong to X" wins over the session the
    # turn persists under; see the module docstring for the bridge's split.
    configurable = (config or {}).get("configurable") or {}
    session_id = configurable.get("tool_session_id")
    if not session_id:
        # Only a plain graph-state dict is bound; ToolCallWithContext / Send and
        # list-of-ToolCall inputs pass through unbound (no chat graph routes tools via Send).
        session_id = state.get("session_id") if isinstance(state, dict) else None
    return set_chat_session_id(session_id) if session_id else None


class SessionBoundToolNode(ToolNode):
    """A ToolNode that runs its tools with the turn's own session bound (#2064).

    Both chat graphs use it. ``config["configurable"]["tool_session_id"]`` if the
    caller set one, else ``state["session_id"]``, is bound for the duration of the
    tool calls and reset afterwards; with neither, nothing is bound here and the
    caller's binding (if any) applies. See the module docstring for why an outer
    binding could not be relied on to reach the tools on the AG-UI route, and for
    the bridge's persisted-vs-tools session split (#2077).
    """

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        token = _bind_tool_session(input, config)
        try:
            return await super().ainvoke(input, config, **kwargs)
        finally:
            if token is not None:
                reset_chat_session_id(token)

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        token = _bind_tool_session(input, config)
        try:
            return super().invoke(input, config, **kwargs)
        finally:
            if token is not None:
                reset_chat_session_id(token)
