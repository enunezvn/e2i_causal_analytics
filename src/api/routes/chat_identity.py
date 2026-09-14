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

The verified id OUTRANKS the session prefix. A prefix is a claim the caller
makes, not a credential: AG-UI reads ``threadId`` straight from the request body
(``copilotkit.py:~4706``) and copies it into session state (``:~1192``), so
without that precedence user A could send ``{B}~anything`` and have B recorded as
the owner of A's compositions. A uuid-shaped prefix proves syntax, never
ownership. When both channels name a user and disagree, the verified one is used
and the mismatch is logged — a claim that loses is worth seeing, but it is not an
error the turn should die on.

Honest-or-nothing: with neither channel, the episode records NULL rather than an
id that would make an unattributable composition look attributed.
``composer_episodes.user_id`` is VARCHAR(100) (``database/ml/013_tool_composer_tables.sql:275``),
so the database will NOT reject a malformed id for us; the feedback linker
compares it as text, so a wrong or invented value silently mis-attributes instead
of failing loudly. Both sources already reject non-uuids and the anonymous
sentinel, and nothing here may loosen that.
"""

import logging
from typing import Any, Dict, Optional

from src.utils.llm_attribution import (
    get_authenticated_user_id,
    set_authenticated_user,
    user_id_from_session,
)

logger = logging.getLogger(__name__)


def resolve_tool_user_id(session_id: Optional[str]) -> Optional[str]:
    """The user this tool call belongs to, or None — never a fabricated id.

    Verified identity first; the session prefix only when no verified identity
    exists. See the module docstring for why the order is not the other way.
    """
    verified = get_authenticated_user_id()
    claimed = user_id_from_session(session_id)
    if verified is None:
        return claimed
    if claimed is not None and claimed != verified:
        logger.warning(
            "Chat session prefix names user %s but the verified request user is %s; "
            "recording the verified user (session=%s)",
            claimed,
            verified,
            session_id,
        )
    return verified


def bind_verified_request_user(request: Any) -> bool:
    """Bind the identity channel from a request the middleware already verified.

    ``JWTAuthMiddleware`` attaches the verified user to ``request.state``
    (``auth_middleware.py:~335``) but sets no contextvar, and CopilotKit's SDK
    sub-path skips its own auth gate — the only thing that used to populate the
    channel — precisely when that state is already there. On a bare-uuid thread
    the tools then had nothing to read and the composition recorded a NULL owner.
    Returns whether an identity was already established, so the caller can still
    decide to run its gate; the id itself is re-validated by the setter.
    """
    state = getattr(request, "state", None)
    user = getattr(state, "user", None) if state is not None else None
    if user is None:
        return False
    set_authenticated_user(user.get("id") if isinstance(user, dict) else None)
    return True


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
