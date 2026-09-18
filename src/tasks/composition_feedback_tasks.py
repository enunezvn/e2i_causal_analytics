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
from typing import Any, Dict, List, Mapping, Optional

from src.utils.llm_attribution import ANONYMOUS_USER_ID

logger = logging.getLogger(__name__)

#: How far back to look for unlabelled episodes.
LOOKBACK_DAYS = 7

#: A rating attaches to a composition that finished within this window before it. Chat feedback
#: arrives while the answer is on screen; a rating hours later is not about this composition.
MATCH_WINDOW = timedelta(minutes=30)

#: The enum values chatbot_message_feedback stores (src/agents/feedback_learner/rating_utils.py).
_SUCCESS_BY_RATING = {"thumbs_up": True, "thumbs_down": False}

#: user_id is selected because _same_owner reads it (#2062): it is the episode side of the owner
#: gate, and a column the matcher uses but the query never asks for is how the T14 ordering
#: defect happened.
_EPISODE_COLUMNS = (
    "episode_id, composition_id, session_id, user_id, created_at, success, feedback_at, feedback_id"
)
#: computed_user_id is selected for the same reason, on the rating side. id is the value that
#: BECOMES the claim: it is what gets written to composer_episodes.feedback_id (ml/044).
_FEEDBACK_COLUMNS = "id, session_id, computed_user_id, rating, created_at"


def _parse(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _same_owner(episode: Dict[str, Any], rating: Dict[str, Any]) -> bool:
    """Defence in depth: never label across users when both sides name one.

    The two values are intended to name the same person, by different write paths:

    - the rating's ``computed_user_id`` is a trigger-set copy of
      ``chatbot_conversations.user_id`` (migration ``123_chatbot_message_owner_inherit.sql``);
    - the episode's ``user_id`` is the verified chat caller the orchestrator threads into the
      composer's context when it dispatches (#2069).

    The session id is NOT consulted. Until #2062 this compared ``computed_user_id`` with the text
    before ``~`` in the episode's session id, from a time when the column WAS
    ``SPLIT_PART(session_id,'~',1)``. Migration 123 retired that expression, and its own header
    records why the prefix is not an owner: the chat UI mints bare-uuid thread ids, so the prefix
    was the thread's own random uuid for 836 of 874 message rows. Scored against the live
    conversations, the prefix comparison rejected 526 of 635. A prefix that most sessions do not
    carry is not evidence of an owner, so it is not kept as a fallback either.

    Absent on either side returns True: the check could not run, which is not the same as the
    check failing. The anonymous sentinel on the rating side means the same thing: it is the
    fallback owner required by ``chatbot_conversations.user_id NOT NULL`` when creation has no
    usable identity (or the real user's profile FK is unavailable), not a real principal.
    Historical/fallback conversations retain that placeholder even when a later authenticated
    turn records its real caller on the composition, because the lost conversation owner cannot
    be reconstructed safely.

    The sentinel bypass is intentionally rating-side only. Production episode writers call
    ``resolve_tool_user_id``/``resolve_session_user_id``, which normalize the sentinel to None;
    therefore an episode carrying the sentinel is an invalid direct-caller value and must not
    weaken the guard against a rating that names a real owner. This is the SECONDARY guard —
    strict session equality in
    :func:`match_episodes` is the primary one and has already held by the time this is reached.
    Compared as text because the two columns differ in type: ``composer_episodes.user_id`` is
    VARCHAR(100) (ml/013), ``computed_user_id`` is uuid.
    """
    rating_owner = rating.get("computed_user_id")
    episode_owner = episode.get("user_id")
    if rating_owner is None or episode_owner is None or str(rating_owner) == ANONYMOUS_USER_ID:
        return True
    return str(rating_owner) == str(episode_owner)


def match_episodes(
    episodes: List[Dict[str, Any]],
    ratings: List[Dict[str, Any]],
    claims: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Assign each rating to the one composition it followed, by composition_id.

    Driven by the RATINGS, not the episodes: a rating is one verdict about one answer, so it may
    label at most one composition. Matching per episode instead gave two compositions minutes
    apart the same rating — the ordinary case of a user asking twice and then rating.

    A rating claims the nearest eligible composition that started at or before it, within the
    window, in the same session and owned by the same user. A composition already claimed by an
    earlier rating is not reconsidered.

    ``claims`` is the attribution already RECORDED on the episodes — composition_id to the
    ``feedback_id`` (ml/044) written when that composition was labelled. It is what makes the
    assignment survive its rating (#2035). Without it every run rebuilt the claims from the
    ratings that still exist, so deleting one rating handed its composition to another:

        A@10:00 and B@10:10, R1@10:11 and R2@10:12. Run 1 gives B←R1 and A←R2; B's write lands,
        A's fails. R1 is deleted — ratings cascade from both chatbot_messages and
        chatbot_conversations. Run 2 rebuilt gives B←R2, B is skipped as already labelled, and A
        never gets its retry.

    So a recorded claim does two things, and both are needed: its composition is never
    re-offered (even though the label alone would already exclude it from the WRITE, matching
    sees every recent episode), and the rating that made it is SPENT — it cannot slide onto a
    neighbouring composition, whether or not it still exists. Keys and ids are compared as text:
    ``feedback_id`` is a bigint that a driver may return as int or str on either side.

    Pure: it takes no client. That purity is why the defect above was reproducible in seconds.
    """
    claimed: Dict[str, Dict[str, Any]] = {}
    taken: set = {str(key) for key in (claims or {})}
    spent: set = {str(value) for value in (claims or {}).values() if value is not None}

    for rating in sorted(ratings, key=lambda r: str(r.get("created_at") or "")):
        if str(rating.get("id")) in spent:
            continue
        given = _parse(rating.get("created_at"))
        if given is None:
            continue

        best: Optional[tuple] = None
        for episode in episodes:
            composition_id = episode.get("composition_id")
            if not composition_id or str(composition_id) in taken:
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
            taken.add(str(best[1]))
            claimed[best[1]] = rating

    return claimed


def _result(labelled: int, considered: int, failed: int) -> Dict[str, Any]:
    """The task's one result shape.

    ``failed`` used to be absent on all four early exits, so a run that lost a retry returned
    ``{"labelled": 0, "considered": N}`` — byte-identical to a clean no-op, and unusable as the
    "inspect when failed > 0" signal #2035 asks callers to watch.

    ``failed`` counts WRITE ATTEMPTS that did not land — one per composition that had a rating to
    apply and did not receive it. It deliberately does NOT count the two read failures, which
    abort the run before any write is attempted: reporting ``failed=1`` there would name a
    composition that was never written to, and would break the reading that ``failed <=
    considered`` bounds how many labels this run lost. A read failure is a RUN-level fault and is
    reported in the log at WARNING instead, which is the channel that separates it from a clean
    no-op returning the same numbers (``test_every_early_exit_reports_failed`` pins that).
    """
    return {"labelled": labelled, "considered": considered, "failed": failed}


def _warn_on_vanished_claims(claims: Mapping[str, Any], ratings: List[Dict[str, Any]]) -> None:
    """Report a recorded claim whose rating is gone from the window.

    ml/044 deliberately has no foreign key, so the id dangles rather than cascading the episode
    away or being nulled back into the #2035 starvation. Dangling is the intended durable trace,
    but it should never be SILENT: this is the one place that can notice it.

    A surviving rating is always in the window when its episode is — a rating matches at most
    MATCH_WINDOW after the composition it labels, so both sides of a live pair fall inside the
    same lookback. An id with no rating therefore means the rating was deleted, not aged out.
    """
    surviving = {str(r.get("id")) for r in ratings if r.get("id") is not None}
    for composition_id, feedback_id in claims.items():
        if str(feedback_id) not in surviving:
            logger.warning(
                "composition feedback linker: %s is labelled by rating %s, which no longer "
                "exists; the label stands and the claim is kept, but its rating was deleted",
                composition_id,
                feedback_id,
            )


def link_composition_feedback(
    client: Any = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> Dict[str, Any]:
    """Label unlabelled episodes from the chat ratings their sessions carry. Never raises.

    Returns the same three keys on every path, including each early exit:

    - ``labelled`` — compositions that received a verdict on this run;
    - ``considered`` — unlabelled compositions in the lookback that were eligible for one;
    - ``failed`` — WRITE ATTEMPTS that did not land, one per composition that had a rating to
      apply and did not receive it. So ``failed <= considered``, and ``failed > 0`` means that
      many labels were lost and will be retried on the next run (#2035 asks callers to watch
      it). It does NOT count the two read failures: those abort the run before any write is
      attempted, so counting them would name a composition that was never written to. A read
      failure is reported at WARNING and returns the same three numbers a clean no-op does, so
      the log is what separates them — see :func:`_result`.
    """
    if client is None:
        from src.memory.services.factories import get_supabase_client

        client = get_supabase_client()
    if client is None:
        logger.warning("composition feedback linker: no database client; nothing labelled")
        return _result(0, 0, 0)

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
        return _result(0, 0, 0)

    # Only unlabelled episodes are WRITTEN. Matching still sees every episode (below), because a
    # rating already spent on a labelled composition must not become available again tomorrow.
    pending = [
        episode
        for episode in episodes
        if episode.get("success") is None and episode.get("feedback_at") is None
    ]
    # The ratings are read BEFORE the no-pending exit, and the one extra SELECT on an idle night
    # is the price of the divergence check below. The steady state of a healthy window is that
    # every episode is already labelled — pending empty — which is exactly when a claim whose
    # rating has been deleted is the only thing left to notice.
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
        return _result(0, len(pending), 0)

    # The attribution already recorded on the episodes (ml/044). Rebuilding it from the surviving
    # ratings instead is #2035: one deleted rating moved a claim and starved a failed write of
    # its retry.
    claims = {
        episode["composition_id"]: episode["feedback_id"]
        for episode in episodes
        if episode.get("composition_id") and episode.get("feedback_id") is not None
    }
    _warn_on_vanished_claims(claims, ratings)

    if not pending:
        return _result(0, 0, 0)

    # Matched against EVERY recent episode, not just the unlabelled ones: a rating that already
    # labelled a composition is spent, and stays spent. Matching only the unlabelled ones made
    # attribution hold within a pass but not across them — the nightly rerun then handed the same
    # rating to the next-oldest composition, which is both a wrong label and a rerun that changed
    # something it promised not to.
    claimed = match_episodes(episodes, ratings, claims=claims)

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
                        # The claim, in the SAME single-row UPDATE as the label it explains:
                        # with PostgREST each execute() is its own transaction, so writing them
                        # separately could leave a label whose attribution never landed (#2035).
                        "feedback_id": rating.get("id"),
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
    return _result(labelled, len(pending), failed)


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
