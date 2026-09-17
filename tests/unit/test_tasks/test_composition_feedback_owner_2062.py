"""The linker's owner gate compares the conversation owner on both sides (#2062).

The gate used to read the text before ``~`` in the episode's session id and compare it with the
rating's ``computed_user_id``. That encoded a premise which migration
``123_chatbot_message_owner_inherit.sql`` retired: ``chatbot_message_feedback.computed_user_id``
stopped being ``SPLIT_PART(session_id,'~',1)`` and became a trigger-set copy of
``chatbot_conversations.user_id``. Migration 123's own header records why the prefix is not the
owner — the CopilotKit UI mints BARE-uuid thread ids, so the prefix was the thread's random uuid
for 836 of 874 message rows. Scored against the live conversations, the prefix comparison rejected
526 of 635.

These are pure-function pins: ``match_episodes`` takes no client.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.tasks import composition_feedback_tasks as tasks
from src.utils.llm_attribution import ANONYMOUS_USER_ID

OWNER = "11111111-1111-1111-1111-111111111111"
OTHER_OWNER = "22222222-2222-2222-2222-222222222222"

#: What the CopilotKit UI actually mints: a bare uuid, whose text is unrelated to the owner.
BARE_SESSION = "99999999-9999-9999-9999-999999999999"
#: A session id that does carry a ``~``, but whose prefix is the THREAD's uuid, not the owner's.
PREFIXED_SESSION = f"{BARE_SESSION}~sess-1"


def _episode(session_id: str, user_id: Any) -> Dict[str, Any]:
    return {
        "composition_id": "comp_1",
        "session_id": session_id,
        "user_id": user_id,
        "created_at": "2026-09-13T10:00:00+00:00",
        "success": None,
        "feedback_at": None,
        # Unlabelled, so no claim is recorded (ml/044). Present because the double should carry
        # every column _EPISODE_COLUMNS selects, not only the ones this file asserts on.
        "feedback_id": None,
    }


def _rating(session_id: str, computed_user_id: Any) -> Dict[str, Any]:
    return {
        "id": 1,
        "session_id": session_id,
        "computed_user_id": computed_user_id,
        "rating": "thumbs_up",
        "created_at": "2026-09-13T10:05:00+00:00",
    }


def _claims(episodes: List[Dict[str, Any]], ratings: List[Dict[str, Any]]) -> Dict[str, Any]:
    return tasks.match_episodes(episodes, ratings)


@pytest.mark.unit
def test_same_owner_labels_when_the_session_id_carries_no_owner_prefix():
    """The bare-uuid session id the chat UI mints. The owners match; only the prefix disagrees."""
    claimed = _claims(
        [_episode(BARE_SESSION, OWNER)],
        [_rating(BARE_SESSION, OWNER)],
    )
    assert "comp_1" in claimed


@pytest.mark.unit
def test_same_owner_labels_when_the_prefix_is_the_thread_not_the_user():
    """A ``~`` in the session id does not make its prefix an owner (migration 123's header)."""
    claimed = _claims(
        [_episode(PREFIXED_SESSION, OWNER)],
        [_rating(PREFIXED_SESSION, OWNER)],
    )
    assert "comp_1" in claimed


@pytest.mark.unit
def test_a_different_owner_is_still_refused():
    """The gate is defence in depth against labelling one user's composition from another's."""
    claimed = _claims(
        [_episode(BARE_SESSION, OWNER)],
        [_rating(BARE_SESSION, OTHER_OWNER)],
    )
    assert claimed == {}


@pytest.mark.unit
def test_an_episode_without_a_recorded_owner_is_allowed():
    """Absent on either side means the check cannot run, not that it failed. Session equality
    (the primary key of the match) has already held by the time the gate is reached."""
    claimed = _claims(
        [_episode(BARE_SESSION, None)],
        [_rating(BARE_SESSION, OWNER)],
    )
    assert "comp_1" in claimed


@pytest.mark.unit
def test_a_rating_without_a_computed_owner_is_allowed():
    claimed = _claims(
        [_episode(BARE_SESSION, OWNER)],
        [_rating(BARE_SESSION, None)],
    )
    assert "comp_1" in claimed


@pytest.mark.unit
def test_an_anonymous_conversation_owner_is_treated_as_unattributed():
    """The sentinel records that conversation ownership was unavailable, not a user.

    A later authenticated turn can legitimately record its real caller on the composition while
    feedback inherits the historical conversation's sentinel owner. Strict session equality is
    still the primary guard, so the placeholder must take the same pass branch as an absent
    owner instead of permanently starving that composition of its rating (#2161).
    """
    claimed = _claims(
        [_episode(BARE_SESSION, OWNER)],
        [_rating(BARE_SESSION, ANONYMOUS_USER_ID)],
    )
    assert "comp_1" in claimed


@pytest.mark.unit
def test_anonymous_sentinel_on_episode_side_does_not_bypass_a_real_rating_owner():
    """The asymmetry is deliberate: production episode identity resolvers turn the sentinel
    into None before recording, while the NOT-NULL conversation FK must persist it and the
    feedback trigger must inherit it. A sentinel episode is therefore invalid direct-caller
    input, not evidence that a real-owned rating may cross the secondary owner guard.
    """
    claimed = _claims(
        [_episode(BARE_SESSION, ANONYMOUS_USER_ID)],
        [_rating(BARE_SESSION, OWNER)],
    )
    assert claimed == {}


@pytest.mark.unit
def test_owner_ids_compare_as_text_across_the_two_column_types():
    """``composer_episodes.user_id`` is VARCHAR(100); ``computed_user_id`` is uuid. A driver that
    hands back a ``UUID`` object on one side must still match the text on the other."""
    from uuid import UUID

    claimed = _claims(
        [_episode(BARE_SESSION, OWNER)],
        [_rating(BARE_SESSION, UUID(OWNER))],
    )
    assert "comp_1" in claimed


@pytest.mark.unit
def test_the_episode_query_selects_the_owner_column_it_gates_on():
    """A column the matcher reads but the query never asks for is how the T14 ordering defect
    happened; the module's own comment says so on the feedback side."""
    assert "user_id" in tasks._EPISODE_COLUMNS.split(", ")


@pytest.mark.unit
def test_the_session_id_prefix_helper_is_gone():
    """The prefix is not evidence of an owner, so it is not kept as a fallback either: reading it
    when ``user_id`` is absent would resurrect exactly the comparison migration 123 retired."""
    assert not hasattr(tasks, "_owner")
