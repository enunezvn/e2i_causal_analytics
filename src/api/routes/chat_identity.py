"""Who a chat tool call belongs to (#2077).

``_resolve_session_id`` (chatbot_tools) answers *which conversation*; this
module answers *which user*, from the same kind of channel: one the tools can
actually read at the moment they run.

The issue proposed ``llm_attribution.get_attribution().user_id``. Measured false
on the browser route: the AG-UI handler streams ``execute()`` through
``with_sse_keepalive``, which pulls every frame in a fresh task, so the
attribution set while one frame is produced is gone by the next pull — which is
why chat LLM usage has been recording ``surface='other', user_id=NULL`` since
2026-08-16. Two channels do survive into the tools:

* the session itself, for the ``{user}~{session}`` ids ``/chat/stream`` mints
  (and their ``~bridge`` shadow, which splits the same way ``computed_user_id``
  does); and
* the auth gate's verified id, set in the REQUEST task by
  ``_require_auth_for_copilotkit_execution`` — the only one that names the user
  behind an AG-UI turn, whose thread ids are bare uuids with no prefix.

Honest-or-nothing: with neither, the episode records NULL rather than an id that
would make an unattributable composition look attributed.
``composer_episodes.user_id`` is uuid-typed and the feedback linker compares it
as text, so a non-uuid must never be returned — both sources reject non-uuids and
the anonymous sentinel already.
"""

from typing import Any, Dict, Optional

from src.utils.llm_attribution import get_authenticated_user_id, user_id_from_session


def resolve_tool_user_id(session_id: Optional[str]) -> Optional[str]:
    """The user this tool call belongs to, or None — never a fabricated id."""
    return user_id_from_session(session_id) or get_authenticated_user_id()


def _composer_context(
    *,
    brand: Optional[str],
    region: Optional[str],
    session_id: Optional[str],
    user_id: Optional[str],
    max_parallel: int,
) -> Dict[str, Any]:
    """The context the chat tool hands the Tool Composer, marked with who called (spec §5.3).

    It lives beside the resolvers because what it carries IS the episode's
    identity: composer_episodes keys its session and owner on these two values.
    """
    return {
        "brand": brand,
        "region": region,
        # #2064/#2077: absent stays None, so an unattributable composition
        # records NULL rather than looking attributed.
        "session_id": session_id,
        "user_id": user_id,
        "max_parallel": max_parallel,
        "entry_point": "chat_tool",
    }
