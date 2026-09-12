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
_FEEDBACK_COLUMNS = "session_id, rating, created_at"


def _parse(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def match_rating(
    episode: Dict[str, Any], ratings: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """The rating this composition earned: same session, at or after it, inside the window.

    The nearest qualifying rating wins, so in a session with several compositions each rating
    labels the one it actually followed.
    """
    started = _parse(episode.get("created_at"))
    if started is None:
        return None

    candidates = []
    for rating in ratings:
        if rating.get("session_id") != episode.get("session_id"):
            continue
        given = _parse(rating.get("created_at"))
        if given is None or given < started or given - started > MATCH_WINDOW:
            continue
        candidates.append((given - started, rating))
    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[0])[1]


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

    # Already labelled episodes are skipped here, which is what makes a re-run a no-op.
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

    labelled = 0
    failed = 0
    for episode in pending:
        rating = match_rating(episode, ratings)
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
