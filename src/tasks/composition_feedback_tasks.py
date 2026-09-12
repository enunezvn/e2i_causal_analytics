"""Nightly: label a composition with the verdict its user already gave (spec §5, owner gate O1).

A user who thumbs-downs a chat answer has told us the composition behind it did not help. That
signal already exists in ``chatbot_message_feedback``; nothing carried it to ``composer_episodes``,
whose ``success`` column has meant *user feedback* since ml/013 and stayed NULL.

What this may and may not do:

- it LABELS, it never changes behaviour. Nothing here feeds the planner, the circuit breaker or
  the tool offer; the reliability rule reads outcomes, not this column;
- thumbs up sets ``success = true``, thumbs down ``success = false``, and ``feedback_at`` records
  when the rating arrived;
- ``feedback_text`` is NEVER written. The comment is free user text, and §5.5 keeps free text out
  of the learning tables. The rating is the signal; the prose is not;
- it matches inside ONE session and a bounded window, so a rating attaches to the composition it
  followed, not to an older one in the same conversation;
- it is idempotent: an episode that already carries feedback is skipped, so the beat can run
  every night and re-runs change nothing.

Mirrors ``routing_label_tasks``'s explicit-feedback matcher, which solves the same problem for
routing decisions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: How far back to look for unlabelled episodes.
LOOKBACK_DAYS = 7

#: A rating attaches to a composition that finished within this window before it. Chat feedback
#: arrives while the answer is on screen; a rating hours later is not about this composition.
MATCH_WINDOW = timedelta(minutes=30)

#: The enum values chatbot_message_feedback stores (src/agents/feedback_learner/rating_utils.py).
_SUCCESS_BY_RATING = {"thumbs_up": True, "thumbs_down": False}

_EPISODE_COLUMNS = "episode_id, composition_id, session_id, created_at, success, feedback_at"
#: computed_user_id is selected because _same_owner reads it; a column the matcher uses but the
#: query never asks for is how the T14 ordering defect happened.
_FEEDBACK_COLUMNS = "session_id, computed_user_id, rating, created_at"


def _parse(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _owner(value: Any) -> Optional[str]:
    """The user a session belongs to. Chat session ids are ``<user-uuid>~<session>``."""
    if not isinstance(value, str) or not value:
        return None
    return value.split("~", 1)[0] or None


def _same_owner(episode: Dict[str, Any], rating: Dict[str, Any]) -> bool:
    """Defence in depth: never label across users when both sides name one.

    On the chat path this cannot differ — a session has exactly one owner
    (``chatbot_conversations.user_id`` is NOT NULL) and the feedback table derives
    ``computed_user_id`` from the session id itself — but the composer is reachable from entry
    points where ``user_id`` is whatever the caller passed, so the check is worth its two lines.
    """
    rating_owner = rating.get("computed_user_id") or _owner(rating.get("session_id"))
    episode_owner = _owner(episode.get("session_id"))
    if rating_owner is None or episode_owner is None:
        return True
    return str(rating_owner) == str(episode_owner)


def match_episodes(
    episodes: List[Dict[str, Any]], ratings: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Assign each rating to the one composition it followed, by composition_id.

    Driven by the RATINGS, not the episodes: a rating is one verdict about one answer, so it may
    label at most one composition. Matching per episode instead gave two compositions minutes
    apart the same rating — the ordinary case of a user asking twice and then rating.

    A rating claims the nearest eligible composition that started at or before it, within the
    window, in the same session and owned by the same user. A composition already claimed by an
    earlier rating is not reconsidered.
    """
    claimed: Dict[str, Dict[str, Any]] = {}
    taken: set = set()

    for rating in sorted(ratings, key=lambda r: str(r.get("created_at") or "")):
        given = _parse(rating.get("created_at"))
        if given is None:
            continue

        best: Optional[tuple] = None
        for episode in episodes:
            composition_id = episode.get("composition_id")
            if not composition_id or composition_id in taken:
                continue
            if rating.get("session_id") != episode.get("session_id"):
                continue
            if not _same_owner(episode, rating):
                continue
            started = _parse(episode.get("created_at"))
            if started is None or given < started or given - started > MATCH_WINDOW:
                continue
            distance = given - started
            if best is None or distance < best[0]:
                best = (distance, composition_id)

        if best is not None:
            taken.add(best[1])
            claimed[best[1]] = rating

    return claimed


def link_composition_feedback(
    client: Any = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> Dict[str, Any]:
    """Label unlabelled episodes from the chat ratings their sessions carry. Never raises."""
    if client is None:
        from src.memory.services.factories import get_supabase_client

        client = get_supabase_client()
    if client is None:
        logger.warning("composition feedback linker: no database client; nothing labelled")
        return {"labelled": 0, "considered": 0}

    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    try:
        episodes = (
            client.table("composer_episodes")
            .select(_EPISODE_COLUMNS)
            .gte("created_at", since)
            .order("episode_id")
            .execute()
        ).data or []
    except Exception as e:  # noqa: BLE001 - a labelling pass never fails the beat
        logger.warning(
            f"composition feedback linker: episode read failed ({type(e).__name__}: {e})"
        )
        return {"labelled": 0, "considered": 0}

    # Only unlabelled episodes are WRITTEN. Matching still sees every episode (below), because a
    # rating already spent on a labelled composition must not become available again tomorrow.
    pending = [
        episode
        for episode in episodes
        if episode.get("success") is None and episode.get("feedback_at") is None
    ]
    if not pending:
        return {"labelled": 0, "considered": 0}

    try:
        ratings = (
            client.table("chatbot_message_feedback")
            .select(_FEEDBACK_COLUMNS)
            .gte("created_at", since)
            .order("created_at")
            .execute()
        ).data or []
    except Exception as e:  # noqa: BLE001
        logger.warning(f"composition feedback linker: rating read failed ({type(e).__name__}: {e})")
        return {"labelled": 0, "considered": len(pending)}

    # Matched against EVERY recent episode, not just the unlabelled ones: a rating that already
    # labelled a composition is spent, and stays spent. Matching only the unlabelled ones made
    # attribution hold within a pass but not across them — the nightly rerun then handed the same
    # rating to the next-oldest composition, which is both a wrong label and a rerun that changed
    # something it promised not to.
    claimed = match_episodes(episodes, ratings)

    labelled = 0
    failed = 0
    for episode in pending:
        rating = claimed.get(episode.get("composition_id"))
        if rating is None:
            continue
        success = _SUCCESS_BY_RATING.get(str(rating.get("rating")))
        if success is None:
            continue
        try:
            (
                client.table("composer_episodes")
                .update(
                    {
                        "success": success,
                        # feedback_at only: the rating is the signal, the comment is free text.
                        "feedback_at": rating.get("created_at"),
                    }
                )
                .eq("composition_id", episode.get("composition_id"))
                .execute()
            )
        except Exception as e:  # noqa: BLE001 - one row never loses the rest
            # Counted, not just logged: without this a run whose every write failed returns
            # {"labelled": 0} and reads exactly like a run that found nothing to label.
            failed += 1
            logger.warning(
                f"composition feedback linker: could not label "
                f"{episode.get('composition_id')} ({type(e).__name__}: {e})"
            )
            continue
        labelled += 1

    if failed:
        logger.warning(
            "composition feedback linker: %d write(s) failed; those compositions stay unlabelled",
            failed,
        )
    logger.info(
        "composition feedback linker: labelled %d of %d unlabelled compositions",
        labelled,
        len(pending),
    )
    return {"labelled": labelled, "considered": len(pending), "failed": failed}


try:  # pragma: no cover - the Celery app is not importable in every test context
    from src.workers.celery_app import celery_app

    @celery_app.task(bind=True, name="src.tasks.link_composition_feedback")
    def link_composition_feedback_task(
        self: Any, lookback_days: int = LOOKBACK_DAYS
    ) -> Dict[str, Any]:
        """Beat entry point; the work is the pure function above."""
        return link_composition_feedback(lookback_days=lookback_days)

except Exception:  # pragma: no cover - import-time safety, mirrors the sibling task modules
    logger.debug(
        "celery app unavailable; composition feedback linker registered as a plain function"
    )
