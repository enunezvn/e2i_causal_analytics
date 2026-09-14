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
  (and their ``~bridge`` shadow, which splits on the first ``~`` the way every
  consumer of the prefix does); and
* the verified id, set in the REQUEST task — by the AG-UI auth gate
  (``_require_auth_for_copilotkit_execution``), by ``authorize_chat_identity``
  on ``/chat`` and ``/chat/stream``, and by ``bind_verified_request_user`` on the
  SDK sub-paths. It is the only channel that names the user behind an AG-UI turn,
  whose thread ids are bare uuids with no prefix.

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

import json
import logging
import uuid
from typing import Any, Dict, Optional, Protocol

from fastapi import HTTPException, status

from src.utils.llm_attribution import (
    resolve_session_user_id,
    set_authenticated_user,
)

logger = logging.getLogger(__name__)


def resolve_tool_user_id(session_id: Optional[str]) -> Optional[str]:
    """The user this tool call belongs to, or None — never a fabricated id.

    One rule, shared with message/usage attribution so the two can never drift:
    verified identity first, the session prefix only when there is none. See the
    module docstring for why the order is not the other way.
    """
    return resolve_session_user_id(session_id)


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
    if not isinstance(user, dict):
        # Fail safe: an absent — or unrecognisably shaped — user is NOT an
        # established identity, so say so and let the caller run its gate rather
        # than clearing the channel on the strength of something unreadable.
        return False
    set_authenticated_user(user.get("id"))
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


def reject_identity_mismatch(
    token_user_id: str,
    body_user_id: Optional[str],
    session_id: Optional[str],
    testing_mode: bool,
) -> None:
    """Reject a chat request that claims to belong to someone other than its caller.

    Finding 1 [HIGH IDOR] made ``ChatRequest.user_id`` non-authoritative and
    rejected a mismatching one with 403. #2077: the supplied ``session_id``
    carries an owner claim of exactly the same weight, and it was unguarded.
    Since migration 123 a message's ``computed_user_id`` is inherited from its
    parent conversation's ``user_id``
    (``database/migrations/123_chatbot_message_owner_inherit.sql:34-46``) rather
    than split from the prefix, and the RLS policies read ONLY that column
    (``database/chat/030_chatbot_rls_policies.sql:133``) — so supplying
    ``{victim}~{uuid}`` either opens a conversation owned by the victim or writes
    into the victim's existing one, and the turn's messages inherit that owner.
    The prefix is compared raw, the way every consumer splits it.

    Skipped in TESTING_MODE, which deliberately bypasses real auth — the same
    exemption the body-``user_id`` check has always had.

    Raises:
        HTTPException: 403 when either claim disagrees with the token identity.
    """
    if testing_mode:
        return
    if body_user_id and body_user_id != token_user_id:
        logger.warning(
            "[Chatbot] Rejected user_id mismatch (possible impersonation): "
            "body user_id does not match authenticated identity"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request user_id does not match the authenticated user.",
        )
    claimed_owner = session_id.split("~", 1)[0] if session_id and "~" in session_id else None
    if claimed_owner and claimed_owner != token_user_id:
        logger.warning(
            "[Chatbot] Rejected session_id mismatch (possible impersonation): "
            "session_id owner prefix does not match authenticated identity"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request session_id does not belong to the authenticated user.",
        )


class ChatClaims(Protocol):
    """The two owner claims a chat request body can carry (see ``ChatRequest``)."""

    user_id: Optional[str]
    session_id: Optional[str]


def authorize_chat_identity(token_user_id: str, claims: ChatClaims, testing_mode: bool) -> str:
    """Vet a chat request's owner claims, bind the verified identity, return it.

    The binding is the point #2077 was missing on ``/chat`` and ``/chat/stream``:
    both handed the authenticated id to the graph but left the channel the chat
    TOOLS read empty, so on a bare-uuid session every composition recorded a NULL
    owner and #2095's owner gate took its absent-pass branch. Binding here — in
    the request task, before the graph or the SSE body starts — is the same
    channel the AG-UI route uses, and it survives the keepalive wrapper's
    per-frame tasks because they copy the context that already carries it.
    """
    reject_identity_mismatch(token_user_id, claims.user_id, claims.session_id, testing_mode)
    set_authenticated_user(token_user_id)
    return str(token_user_id)


def _thread_ids(body: Any) -> list:
    """Every ``threadId`` a CopilotKit body can carry, across the shapes in use.

    ``agent/run`` reads it top level or nested under ``body``; the SDK's
    ``agent/{name}`` and ``agents/execute`` read it top level
    (``copilotkit/integrations/fastapi.py:107,195``). Any of them is a claim, so
    all of them are checked.
    """
    if not isinstance(body, dict):
        return []
    nested = body.get("body")
    candidates = [body.get("threadId")]
    if isinstance(nested, dict):
        candidates.append(nested.get("threadId"))
    return [c for c in candidates if isinstance(c, str) and c]


def _claims_another_owner(body: Any, request: Any, testing_mode: bool) -> bool:
    """Does this body name a thread whose owner prefix is not the caller?

    The thread id becomes the session the turn PERSISTS under. Since migration
    123 a message inherits ``computed_user_id`` from its parent conversation, and
    an EXISTING conversation is accepted without an owner check, so a foreign
    prefix either opens a conversation owned by someone else or writes into
    theirs. CopilotKit mints bare uuids and the frontend passes
    ``CopilotContext.threadId`` through untouched, so no legitimate caller sends
    a prefix at all. TESTING_MODE is exempt, as every other claim check is.
    """
    if testing_mode:
        return False
    state = getattr(request, "state", None)
    user = getattr(state, "user", None) if state is not None else None
    token_user_id = user.get("id") if isinstance(user, dict) else None
    if not token_user_id:
        return False
    for thread_id in _thread_ids(body):
        claimed = thread_id.split("~", 1)[0] if "~" in thread_id else None
        if claimed and claimed != token_user_id:
            logger.warning(
                "[CopilotKit] Rejected threadId mismatch (possible impersonation): "
                "threadId owner prefix does not match authenticated identity"
            )
            return True
    return False


def owned_thread_id(body_json: Dict[str, Any], request: Any, testing_mode: bool) -> Optional[str]:
    """The AG-UI turn's thread id, or None when it claims someone else's ownership.

    None means reject; the caller answers 403 rather than raising, because an
    exception here would be swallowed by the handler's broad except into the
    ungated SDK fallthrough.
    """
    if _claims_another_owner(body_json, request, testing_mode):
        return None
    nested = body_json.get("body") if isinstance(body_json.get("body"), dict) else {}
    return nested.get("threadId") or body_json.get("threadId") or str(uuid.uuid4())


def sdk_thread_denied(body_bytes: bytes, request: Any, testing_mode: bool) -> bool:
    """Whether the SDK sub-path body claims a thread the caller does not own.

    The root branch's check never ran here: ``agent/{name}`` and
    ``agents/execute`` reach the third-party handler through the fallthrough,
    which delegates the body verbatim. Read the same bytes the SDK will read —
    the stream is already buffered at this point, so nothing is consumed. An
    unparseable body names no thread and is left to the SDK to reject.
    """
    try:
        body = json.loads(body_bytes) if body_bytes else None
    except (ValueError, TypeError):
        return False
    return _claims_another_owner(body, request, testing_mode)
