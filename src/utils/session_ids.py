"""Shared coercion for the platform's composite chat session-id format.

The chat surfaces use a composite ``{user_uuid}~{session_uuid}`` session id -- the
DB even derives ``chatbot_messages.computed_user_id`` from
``split_part(session_id, '~', 1)``. Several tables key session context on a
``uuid``-typed ``session_id`` column (``audit_chain_entries.session_id``,
``episodic_memories.session_id``); passing the composite string straight into one
of those columns fails with PostgreSQL 22P02 ("invalid input syntax for type
uuid") and -- in code paths that swallow the write error -- silently drops the
row (#1335 audit-chain genesis; #1393 orchestrator episodic memory).

Since PR #1394 the conversational bridge also appends a ``~bridge`` marker to the
(already composite) id, producing ``{user}~{session}~bridge`` whose trailing
segment is NOT a uuid (codex flagged the old ``rsplit('~', 1)[-1]`` tail
assumption as a LOW on #1394). This helper recovers the session's own uuid from
any of those shapes.
"""

from __future__ import annotations

from typing import Optional, Union
from uuid import UUID

# The conversational bridge (PR #1394, ``src/api/routes/chat_bridge.py``) appends
# this marker to the session id of its shadow session (``{session_id}~bridge``).
# It is the ONLY documented ``~``-suffix marker, so it is the only trailing
# non-uuid segment the coercion strips.
_BRIDGE_MARKER = "bridge"


def coerce_session_uuid(value: Optional[Union[UUID, str]]) -> Optional[UUID]:
    """Recover a plain ``UUID`` from a chat session identifier for a uuid column.

    Handles the platform's session-id shapes:

    - a plain uuid (or ``UUID`` object) -> returned unchanged;
    - a composite ``{user_uuid}~{session_uuid}`` -> the trailing session uuid;
    - a bridge-suffixed ``{...}~bridge`` (PR #1394) -> the session uuid the marker
      was appended to (works for both ``{user}~{session}~bridge`` and a bare
      ``{plain_uuid}~bridge``).

    The rule is: strip a single trailing ``~bridge`` marker, then take the (now)
    trailing segment IF it parses as a uuid, else return ``None``. This preserves
    the historical ``rsplit('~', 1)[-1]`` semantics for the bare
    ``{user}~{session}`` composite (audit_chain #1335) while hardening genesis
    against the bridge suffix. Crucially, a MALFORMED id such as
    ``{user_uuid}~garbage`` yields ``None`` (honest null) rather than silently
    mis-associating to the ``user_uuid`` -- the trailing-segment check rejects the
    non-uuid ``garbage`` instead of scanning left into the user segment.

    Returns ``None`` (an honest null for the nullable column) when the recovered
    segment is not a uuid -- never a fabricated id. Callers that need user
    attribution must carry ``user_id`` in its own column; this only recovers the
    session id.
    """
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    segments = str(value).split("~")
    if len(segments) > 1 and segments[-1] == _BRIDGE_MARKER:
        segments.pop()
    try:
        return UUID(segments[-1])
    except (ValueError, AttributeError):
        return None
