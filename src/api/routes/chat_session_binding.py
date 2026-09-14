"""Session binding for the chat graphs' tools node (#2064, PR #2088).

``SessionBoundToolNode`` is the tools node both chat graphs mount. It binds
``state["session_id"]`` into ``chat_session_id_context`` for the duration of a
turn's tool calls and resets it afterwards, so ``orchestrator_tool`` and
``tool_composer_tool`` record the real conversation instead of inventing a
``chatbot-<ts>`` id; when state carries no session it binds nothing and the
caller's binding, if any, still applies. A binding made outside the graph is not
enough on the browser route: AG-UI's ``execute()`` binds the session while its
first frame is pulled, and the handler's ``with_sse_keepalive`` pulls every frame
in a fresh task with a copy of that context, so the graph runs without it. The
node lives here rather than in ``chatbot_tools`` because it belongs to neither
graph in particular and to the tools module only by way of the context variable
it sets.
"""

import contextvars
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from src.api.routes.chatbot_tools import reset_chat_session_id, set_chat_session_id


def _bind_state_session(state: Any) -> "Optional[contextvars.Token[Optional[str]]]":
    # Only a plain graph-state dict is bound; ToolCallWithContext / Send and
    # list-of-ToolCall inputs pass through unbound (no chat graph routes tools via Send).
    session_id = state.get("session_id") if isinstance(state, dict) else None
    return set_chat_session_id(session_id) if session_id else None


class SessionBoundToolNode(ToolNode):
    """A ToolNode that runs its tools with the turn's session from graph state (#2064).

    Both chat graphs use it. ``state["session_id"]`` is bound for the duration of
    the tool calls and reset afterwards; when state has no session, nothing is
    bound here and the caller's binding (if any) applies. See the module
    docstring for why an outer binding does not reach the tools on the AG-UI
    route.
    """

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        token = _bind_state_session(input)
        try:
            return await super().ainvoke(input, config, **kwargs)
        finally:
            if token is not None:
                reset_chat_session_id(token)

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        token = _bind_state_session(input)
        try:
            return super().invoke(input, config, **kwargs)
        finally:
            if token is not None:
                reset_chat_session_id(token)
