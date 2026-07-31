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


def coerce_session_uuid(value: Optional[Union[UUID, str]]) -> Optional[UUID]:
    """Recover a plain ``UUID`` from a chat session identifier for a uuid column.

    Handles the platform's session-id shapes:

    - a plain uuid (or ``UUID`` object) -> returned unchanged;
    - a composite ``{user_uuid}~{session_uuid}`` -> the trailing session uuid;
    - a bridge-suffixed ``{user}~{session}~bridge`` (PR #1394) -> the session
      uuid (the non-uuid ``bridge`` marker is skipped).

    The recovery rule is "the rightmost ``~``-delimited segment that parses as a
    uuid". This generalizes the historical ``rsplit('~', 1)[-1]``
    (audit_chain #1335): it still recovers the trailing session uuid of a bare
    ``{user}~{session}`` id, but also tolerates a trailing non-uuid marker such
    as ``~bridge``. Real session ids always carry uuid segments for both user and
    session, so the rule unambiguously lands on the session uuid.

    Returns ``None`` (an honest null for the nullable column) when no segment
    parses as a uuid -- never a fabricated id. Callers that need user attribution
    must carry ``user_id`` in its own column; this only recovers the session id.
    """
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    for segment in reversed(str(value).split("~")):
        try:
            return UUID(segment)
        except (ValueError, AttributeError):
            continue
    return None
