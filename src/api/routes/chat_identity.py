"""Who a chat tool call belongs to (#2077).

``_resolve_session_id`` (chatbot_tools) answers *which conversation*; this
module answers *which user*, from the same kind of channel: one the tools can
actually read at the moment they run.

The issue proposed ``llm_attribution.get_attribution().user_id``. That was
measured dead on the browser route when this module was written: the AG-UI
handler streams ``execute()`` through ``with_sse_keepalive``, which pulled every
frame in a fresh task, so the attribution set while one frame was produced was
gone by the next pull — which is why chat LLM usage recorded ``surface='other',
user_id=NULL`` from 2026-08-16 until #2100 gave those pulls one shared context.

It is readable again, and this module still does not read it. Attribution is a
DERIVED value, and derived by this very module: ``set_chat_attribution`` resolves
its ``user_id`` through the same ``resolve_session_user_id`` that
``resolve_tool_user_id`` below delegates to, so the two cannot disagree and
neither can promote the session prefix over a verified id. Reading the
attribution here would be a hop to an answer already computed, and would make
tool identity depend on whether some entry point called ``set_chat_attribution``
first. Both channels survive into the tools; this module reads them directly:

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
from typing import Any, Dict, List, Optional, Protocol

from fastapi import HTTPException, status

from src.memory.services.factories import get_async_supabase_client
from src.repositories.chatbot_conversation import ChatbotConversationRepository
from src.utils.llm_attribution import (
    ANONYMOUS_USER_ID,
    get_authenticated_user_id,
    resolve_session_user_id,
    set_authenticated_user,
    set_chat_attribution,
)

logger = logging.getLogger(__name__)


async def thread_owner_denied(thread_id: Optional[str], token_user_id: Optional[str]) -> bool:
    """True when an EXISTING conversation belongs to someone other than the caller.

    #2077 refused a foreign ``{owner}~{uuid}`` PREFIX, which is a claim the body
    carries. This answers the other half (#2107): the stored owner of a thread
    that already exists. CopilotKit mints bare uuids with no prefix, so the
    prefix check never saw them — token X could send Y's thread id and the turn
    would run inside Y's conversation, taking Y's last ten messages into the
    prompt (``chatbot_graph.py:~1006``) and writing messages that migration
    123's trigger stamps with Y's ``user_id``. Service-role clients bypass RLS
    (``supabase_client.py:~36``, ``factories.py:~759``), so an application-side
    check is the only gate there can be.

    **Raises nothing a failing lookup can produce.** The AG-UI call site sits
    inside the handler's broad ``except``, which falls through to the *ungated*
    SDK path — an ordinary exception here would defeat the very check it
    performs. Returns a bool; each caller answers 403 itself. Cancellation is
    the deliberate exception: ``CancelledError`` is a ``BaseException`` and is
    left to propagate, because a turn that is being torn down should not be
    reported as allowed.

    Rules, in order:

    1. nothing to compare (no thread, or no verified caller) -> allow, no query;
    2. no such conversation -> allow. This is #1405's arbitrary-thread support
       and it is what keeps every NEW bare thread working;
    3. the caller owns it -> allow;
    4. the anonymous sentinel owns it -> allow, at INFO. The sentinel records
       the ABSENCE of an identity: those rows predate #1405's JWT attribution,
       so there is nobody to protect and nothing to back-fill from;
    5. anyone else owns it -> deny, at WARNING, naming no ids;
    6. the lookup failed -> allow, at WARNING. Fail-OPEN is deliberate: the
       write and the read that constitute the harm go through this same
       database, so an outage that blinds the check equally disables what it
       protects, while fail-closed would 403 every chat user during a hiccup.
    """
    if not thread_id or not token_user_id:
        return False
    try:
        client = await get_async_supabase_client()
        if not client:
            # Without a client the repository answers None, which is also how it
            # reports "no such conversation" — so a check that never RAN would
            # otherwise be indistinguishable in the log from one that ran and
            # allowed. Say which happened.
            logger.warning(
                "[Chat] No conversation store client; the owner check did not run "
                "and the turn is allowed (fail-open, #2107)."
            )
            return False
        repository = ChatbotConversationRepository(supabase_client=client)
        conversation = await repository.get_by_session_id(thread_id)
    except Exception:
        logger.warning(
            "[Chat] Thread owner lookup failed; allowing the turn (fail-open, #2107). "
            "The conversation store this check reads is the same one the turn writes.",
            exc_info=True,
        )
        return False
    if not conversation:
        return False
    return _owner_denies(conversation.get("user_id"), token_user_id)


def _owner_denies(owner: Optional[str], token_user_id: Optional[str]) -> bool:
    """Does this stored owner refuse this caller? The comparison, without the lookup.

    Split out so the route seams and the in-turn tool seam decide ownership by
    the same rules while each spends only ONE primary-key read: the tool needs
    the conversation row itself, and re-deriving the verdict from a second
    lookup would be both slower and a chance for the two to drift.
    """
    if not owner or not token_user_id or owner == token_user_id:
        return False
    if owner == ANONYMOUS_USER_ID:
        logger.info(
            "[Chat] Existing conversation has the anonymous sentinel owner; "
            "treating it as unowned and allowing the turn (#2107)."
        )
        return False
    logger.warning(
        "[Chatbot] Rejected threadId ownership (possible IDOR): the conversation "
        "already exists and is owned by another user"
    )
    return True


async def refuse_foreign_thread(thread_id: Optional[str], token_user_id: Optional[str]) -> None:
    """Raise 403 when ``thread_owner_denied`` says the thread is someone else's; otherwise return.

    The raising form of the gate, for a route whose broad ``except`` RETURNS a
    200 error body rather than falling through to another handler (the #2107
    ``owned_thread_id`` case): ``submit_feedback`` (#2109). Its rules are the
    bool helper's — no row, own thread and the anonymous sentinel allow; a
    lookup failure allows with a WARNING; a foreign owner refuses — with the
    same status and detail string as the AG-UI gates (their body key is
    ``error``, this one's is ``detail``).

    The caller's ``except HTTPException: raise`` clause, placed ahead of its
    broad ``except Exception``, is what lets the 403 out instead of a 200
    ``{"success": false}`` body.

    Existence oracle, accepted: a refused caller learns the thread exists. On
    the ``session_id`` path that costs guessing a v4 uuid; on the ``message_id``
    path the id is a sequential integer, so "exists and foreign" (403) is
    distinguishable from "not found" (200 body) by counting — strictly less
    than before, when the same call rated the row and returned success.
    """
    if await thread_owner_denied(thread_id, token_user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="threadId not yours")


async def owned_conversation(
    conversation_repository: Any, session_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """The conversation a chat TOOL may read, or None when it is not the caller's.

    The route seams close the boundary at the edge of a turn, but the guarantee
    has a hole INSIDE an allowed turn: ``conversation_memory_tool`` takes a
    session id the MODEL supplies, so a caller sitting in their own conversation
    can steer it into naming someone else's and read that history back. Being
    inside an authorized conversation does not make a second conversation yours.

    The caller is the bound verified identity — the channel the route set and
    the one the other tools resolve from — never anything in the tool argument,
    which is a claim. No verified caller means no comparison is possible, so the
    read still proceeds (the unauthenticated and TESTING_MODE paths keep their
    behaviour) — but it WARNS, like every fail-open path on the route seams. An
    empty channel inside the AG-UI graph is a real regression class, and an
    operator must be able to tell a check that passed from one that never ran.

    A refusal returns None, which is the tool's existing "not found" shape: deny
    and nonexistent are deliberately indistinguishable, so the tool is not an
    existence oracle for other people's sessions. A lookup FAILURE still raises
    into the tool's own ``except``, which is today's behaviour for a store that
    is down — unlike the route seams, there is a handler here already.

    #2105 made the argument optional, defaulting to the bound session; this is
    what keeps a model that still names one honest.
    """
    if not session_id:
        # #2105: nothing named (or an empty id) and nothing bound for this turn.
        # An empty session channel inside the graph is the #2100 regression
        # class, and the tool answers "not found" without looking, so the
        # operator has to be able to see that this is why.
        logger.warning(
            "[Chat] conversation_memory_tool called with no conversation named and none "
            "bound for this turn; answering not-found without a lookup (#2105)."
        )
        return None
    conversation: Optional[Dict[str, Any]] = await conversation_repository.get_by_session_id(
        session_id
    )
    if not conversation:
        return None
    caller = get_authenticated_user_id()
    if not caller:
        logger.warning(
            "[Chat] No verified caller bound; the conversation owner check did not "
            "run and the history read is allowed (fail-open, #2107)."
        )
        return conversation
    if _owner_denies(conversation.get("user_id"), caller):
        return None
    return conversation


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
    """The two owner claims a chat request body can carry (see ``ChatRequest``).

    Read-only properties, not plain attributes: a mutable protocol attribute is
    invariant, so ``ChatRequest.user_id`` (a required ``str``) would not satisfy
    an ``Optional[str]`` attribute. Nothing here writes to a claim, so the
    covariant read-only form is both correct and what the call sites need.
    """

    @property
    def user_id(self) -> Optional[str]: ...

    @property
    def session_id(self) -> Optional[str]: ...


async def authorize_chat_identity(
    token_user_id: str, claims: ChatClaims, testing_mode: bool
) -> str:
    """Vet a chat request's owner claims, bind the verified identity, return it.

    The binding is the point #2077 was missing on ``/chat`` and ``/chat/stream``:
    both handed the authenticated id to the graph but left the channel the chat
    TOOLS read empty, so on a bare-uuid session every composition recorded a NULL
    owner and #2095's owner gate took its absent-pass branch. Binding here — in
    the request task, before the graph or the SSE body starts — is the same
    channel the AG-UI route uses, and it survives the keepalive wrapper, whose
    frame pulls all share one context copied from the task that binds it here.

    #2107 adds the stored-owner half of the claim check. Raising is correct
    HERE, unlike on the AG-UI seam: both callers resolve the identity OUTSIDE
    their try/except, precisely so a 403 propagates instead of being swallowed
    into a 200 error body — and on ``/chat/stream`` that also puts the refusal
    ahead of the ``StreamingResponse``, so the caller gets a real 403 rather
    than an SSE error frame inside a 200.

    Raises:
        HTTPException: 403 when a claim, or the conversation's stored owner,
            disagrees with the token identity.
    """
    reject_identity_mismatch(token_user_id, claims.user_id, claims.session_id, testing_mode)
    if not testing_mode and await thread_owner_denied(claims.session_id, token_user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request session_id does not belong to the authenticated user.",
        )
    set_authenticated_user(token_user_id)
    return str(token_user_id)


class ChatTurn(Protocol):
    """A plain chat request whose turn ids the route seam finalises IN PLACE.

    ``ChatClaims`` is read-only because the claim checks only read. This is the
    writable view the same ``ChatRequest`` satisfies: ``session_id`` and
    ``request_id`` are ``Optional[str]`` on the model, so the invariant mutable
    members match exactly, and ``user_id`` stays a read-only property because
    nothing here writes it.
    """

    @property
    def user_id(self) -> Optional[str]: ...

    session_id: Optional[str]
    request_id: Optional[str]


async def bind_plain_chat_turn(
    token_user_id: str,
    chat_request: ChatTurn,
    testing_mode: bool,
    fallback_request_id: Optional[str],
) -> str:
    """Authorize a plain chat turn, finalise its ids on the request, bind attribution.

    #2119: ``set_chat_attribution`` had one caller, inside the AG-UI
    ``execute()``. ``/chat`` and ``/chat/stream`` bound the identity channel
    (``authorize_chat_identity``) and nothing else, so every metered LLM call
    those turns made (zero-token calls are skipped, ``llm_usage_callback.py:~63``)
    recorded ``surface='other'`` with a NULL user and session, and their
    assistant rows drained no token usage.

    The attribution needs the turn's session id, and on these routes the id did
    not exist yet where the binding has to be made: the REQUEST task, before the
    ``StreamingResponse`` is built or ``run_chatbot`` is awaited — the context
    the keepalive wrapper copies once for every frame pull (#2100) and the
    dispatcher's thread offload of a sync agent now copies too (#2119 Part D).
    ``_stream_chat_response`` and ``create_initial_state`` each minted the id
    later, downstream of the graph's creation. So the session is minted HERE
    now, in the exact ``{user}~{uuid4}`` shape both of those used, and written
    back onto the request: the stream route's mint is deleted, and
    ``create_initial_state``'s cannot fire from these routes because the request
    always carries a session by then. ``request_id`` is finalised the same way,
    with the expression both routes already use, so the value bound here is the
    value the routes log and pass to the graph; the response's ``X-Request-ID``
    header is the tracing middleware's own id when tracing is on
    (``tracing.py:~253``), unchanged from before.

    Returns the verified identity, as ``authorize_chat_identity`` does; its 403s
    propagate unchanged, and nothing is bound when it refuses.
    """
    identity = await authorize_chat_identity(token_user_id, chat_request, testing_mode)
    session_id = chat_request.session_id or f"{identity}~{uuid.uuid4()}"
    request_id = chat_request.request_id or fallback_request_id or "unknown"
    chat_request.session_id = session_id
    chat_request.request_id = request_id
    set_chat_attribution(session_id, request_id)
    return identity


def _thread_ids(body: Any) -> List[str]:
    """Every ``threadId`` a CopilotKit body can carry, in precedence order.

    ``agent/run`` reads it nested under ``body`` first, then top level; the SDK's
    ``agent/{name}`` and ``agents/execute`` read it top level
    (``copilotkit/integrations/fastapi.py:107,195``). Any of them is a claim, so
    all of them are checked — and a body that is not a mapping, at either level,
    names no thread at all.
    """
    if not isinstance(body, dict):
        return []
    nested = body.get("body")
    candidates = [nested.get("threadId") if isinstance(nested, dict) else None]
    candidates.append(body.get("threadId"))
    # De-duplicated, order preserved: AG-UI commonly sends the SAME id at both
    # levels, and that is one claim, not two. Precedence still decides which id
    # the turn runs under; each DISTINCT id now costs one lookup, not two.
    return list(dict.fromkeys(c for c in candidates if isinstance(c, str) and c))


def _token_user_id(request: Any) -> Optional[str]:
    """The verified id the middleware or the auth gate attached, or None.

    ``request.state.user`` is the one identity on a chat request that the caller
    did not supply; everything else in the body is a claim.
    """
    state = getattr(request, "state", None)
    user = getattr(state, "user", None) if state is not None else None
    return user.get("id") if isinstance(user, dict) else None


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
    token_user_id = _token_user_id(request)
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


async def _claims_a_foreign_conversation(body: Any, request: Any, testing_mode: bool) -> bool:
    """Does this body name an EXISTING conversation the caller does not own (#2107)?

    The companion to ``_claims_another_owner``, which answers the same question
    about the ``{owner}~`` prefix — syntax the caller supplies. This one reads
    the stored owner, so it is the half that covers the bare uuids CopilotKit
    actually mints. Same TESTING_MODE exemption, and the same lookup-failure
    contract: ``thread_owner_denied`` swallows its own failures.
    """
    if testing_mode:
        return False
    token_user_id = _token_user_id(request)
    if not token_user_id:
        return False
    for thread_id in _thread_ids(body):
        if await thread_owner_denied(thread_id, token_user_id):
            return True
    return False


async def owned_thread_id(
    body_json: Dict[str, Any], request: Any, testing_mode: bool
) -> Optional[str]:
    """The AG-UI turn's thread id, or None when it claims someone else's ownership.

    None means reject; the caller answers 403 rather than raising, because an
    exception here would be swallowed by the handler's broad except into the
    ungated SDK fallthrough.

    Two claims, both refused: a foreign ``{owner}~`` prefix (#2077) and, since
    #2107, a bare id that already names someone else's conversation. A thread
    nobody has opened yet is still accepted — that is the arbitrary-thread
    support #1405 documented, and every new browser turn depends on it.
    """
    if _claims_another_owner(body_json, request, testing_mode):
        return None
    claimed = _thread_ids(body_json)
    if not claimed:
        return str(uuid.uuid4())
    if await _claims_a_foreign_conversation(body_json, request, testing_mode):
        return None
    return claimed[0]


async def sdk_thread_denied(
    body_bytes: bytes, request: Any, testing_mode: bool, method: str = "POST"
) -> bool:
    """Whether the SDK sub-path body claims a thread the caller does not own.

    The root branch's check never ran here: ``agent/{name}``, ``agents/execute``
    and ``agent/{name}/state`` reach the third-party handler through the
    fallthrough, which delegates the body verbatim. All three read ``threadId``
    from the top level of that body (``copilotkit/integrations/fastapi.py``
    :107,127,195), which is what ``_thread_ids`` reads — so one seam covers
    every sub-path, ``/state`` included: it returns the conversation's agent
    state, and disclosure is the same harm as execution.

    Read the same bytes the SDK will read — the stream is already buffered at
    this point, so nothing is consumed. An unparseable body names no thread and
    is left to the SDK to reject.

    ``method`` carries the preflight exemption that used to sit inline at the
    call site: OPTIONS is a CORS negotiation with no turn behind it, so it must
    not cost a database round trip. Taking it here keeps the call site one line
    wide, which the module-size ratchet on ``copilotkit.py`` requires.
    """
    if method == "OPTIONS":
        return False
    try:
        body = json.loads(body_bytes) if body_bytes else None
    except (ValueError, TypeError):
        return False
    if _claims_another_owner(body, request, testing_mode):
        return True
    return await _claims_a_foreign_conversation(body, request, testing_mode)
