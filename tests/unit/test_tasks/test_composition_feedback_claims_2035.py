"""A recorded claim survives its rating's deletion, so a failed write keeps its retry (#2035).

The linker used to rebuild the rating-to-composition assignment on every run. Codex's sequence
shows what that costs, and it is reproducible without a database because ``match_episodes`` is a
pure function:

    RUN 1   episodes [A@-20m, B@-10m], ratings [R1@-9m, R2@-8m]  ->  B<-R1, A<-R2
            B's write succeeds; A's write fails, so only B carries a claim.
    R1 is deleted (chatbot_message_feedback cascades from both chatbot_messages and
    chatbot_conversations, so deleting a message or a conversation deletes its ratings).
    RUN 2   rebuilt from the survivors                           ->  B<-R2, A starved
            RUN 2 seeded with the persisted claim                ->  A<-R2, B untouched

The only difference between "A gets its retry" and "A is starved forever" is whether the claim
was recorded.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.tasks import composition_feedback_tasks as tasks

NOW = datetime.now(timezone.utc)
SESSION = "99999999-9999-9999-9999-999999999999"
OWNER = "11111111-1111-1111-1111-111111111111"

R1_ID = 101
R2_ID = 102


def _iso(minutes_ago: int) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def _episode(
    composition_id: str,
    minutes_ago: int,
    *,
    success: Any = None,
    feedback_at: Any = None,
    feedback_id: Any = None,
) -> Dict[str, Any]:
    return {
        "episode_id": composition_id,
        "composition_id": composition_id,
        "session_id": SESSION,
        "user_id": OWNER,
        "created_at": _iso(minutes_ago),
        "success": success,
        "feedback_at": feedback_at,
        "feedback_id": feedback_id,
    }


def _rating(rating_id: int, minutes_ago: int, verdict: str = "thumbs_up") -> Dict[str, Any]:
    return {
        "id": rating_id,
        "session_id": SESSION,
        "computed_user_id": OWNER,
        "rating": verdict,
        "created_at": _iso(minutes_ago),
    }


def _run1_episodes() -> List[Dict[str, Any]]:
    return [_episode("A", 20), _episode("B", 10)]


def _run1_ratings() -> List[Dict[str, Any]]:
    return [_rating(R1_ID, 9), _rating(R2_ID, 8)]


# ---------------------------------------------------------------------------
# The pure matcher
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run1_assigns_the_nearer_composition_to_the_earlier_rating():
    claimed = tasks.match_episodes(_run1_episodes(), _run1_ratings())
    assert claimed["B"]["id"] == R1_ID
    assert claimed["A"]["id"] == R2_ID


@pytest.mark.unit
def test_run2_retries_the_failed_write_when_the_claim_was_recorded():
    """B's write landed and recorded R1; A's failed. R1 is then deleted.

    Without the persisted claim the surviving rating slides onto B and A never gets its retry.
    """
    episodes = [
        _episode("A", 20),  # the failed write: still unlabelled
        _episode("B", 10, success=True, feedback_at=_iso(9), feedback_id=R1_ID),
    ]
    surviving = [_rating(R2_ID, 8)]  # R1 is gone

    claimed = tasks.match_episodes(episodes, surviving, claims={"B": R1_ID})

    assert claimed.get("A", {}).get("id") == R2_ID, "the failed write must keep its retry"
    assert "B" not in claimed, "a composition whose claim is recorded is never re-claimed"


@pytest.mark.unit
def test_without_the_recorded_claim_the_surviving_rating_relabels_the_wrong_composition():
    """The defect, pinned so the fix is not silently reverted: this is run 2 with no claims.

    Deliberately GREEN both before and after #2035: it is the CONTRAST for the test above, which
    runs the same two episodes WITH the claim. It asserts what the no-claims path still does, not
    what is wrong with it. A change that legitimately alters the no-claims path — dropping the
    ``claims=None`` default, or matching on something other than nearest-preceding — should
    DELETE this test rather than update it, because there is then no contrast left to draw.
    """
    episodes = [
        _episode("A", 20),
        _episode("B", 10, success=True, feedback_at=_iso(9), feedback_id=R1_ID),
    ]
    claimed = tasks.match_episodes(episodes, [_rating(R2_ID, 8)])

    assert claimed.get("B", {}).get("id") == R2_ID
    assert "A" not in claimed


@pytest.mark.unit
def test_a_rating_that_already_claimed_a_composition_cannot_claim_another():
    """R1 survives this time. It already labelled B, so it is spent: A belongs to R2.

    Without the spent check R1 would be re-offered to A, because seeding ``taken`` only stops it
    from re-claiming B.
    """
    episodes = [
        _episode("A", 20),
        _episode("B", 10, success=True, feedback_at=_iso(9), feedback_id=R1_ID),
    ]
    claimed = tasks.match_episodes(episodes, _run1_ratings(), claims={"B": R1_ID})

    assert claimed.get("A", {}).get("id") == R2_ID
    assert "B" not in claimed


@pytest.mark.unit
def test_claims_compare_across_the_text_and_integer_forms_of_the_key():
    """The id is a bigint; a driver may hand it back as int or as str on either side."""
    episodes = [
        _episode("A", 20),
        _episode("B", 10, success=True, feedback_at=_iso(9), feedback_id=str(R1_ID)),
    ]
    claimed = tasks.match_episodes(episodes, _run1_ratings(), claims={"B": str(R1_ID)})

    assert claimed.get("A", {}).get("id") == R2_ID
    assert "B" not in claimed


@pytest.mark.unit
def test_claims_defaults_to_none_so_every_existing_caller_is_unchanged():
    assert tasks.match_episodes(_run1_episodes(), _run1_ratings(), claims=None) == (
        tasks.match_episodes(_run1_episodes(), _run1_ratings())
    )


# ---------------------------------------------------------------------------
# The task, through a test double at the PostgREST boundary
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, data: Any) -> None:
        self.data = data


class _FakeQuery:
    """The chainable subset of the sync PostgREST builder the linker uses."""

    def __init__(self, db: "FakeDB", table: str) -> None:
        self._db = db
        self._table = table
        self._mode: Optional[str] = None
        self._payload: Dict[str, Any] = {}
        self._eq: Dict[str, Any] = {}

    def select(self, columns: str) -> "_FakeQuery":
        self._mode = "select"
        self._db.selected[self._table] = columns
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._payload = dict(payload)
        return self

    def gte(self, _column: str, _value: Any) -> "_FakeQuery":
        return self

    def eq(self, column: str, value: Any) -> "_FakeQuery":
        self._eq[column] = value
        return self

    def order(self, _column: str) -> "_FakeQuery":
        return self

    def execute(self) -> _Result:
        if self._mode == "update":
            composition_id = self._eq.get("composition_id")
            if composition_id in self._db.write_fails:
                raise RuntimeError(f"simulated write failure for {composition_id}")
            self._db.updates.append((composition_id, dict(self._payload)))
            return _Result([])
        if self._table in self._db.read_fails:
            raise RuntimeError(f"simulated read failure for {self._table}")
        return _Result(list(self._db.tables.get(self._table, [])))


class FakeDB:
    """An explicit substitution at the database boundary; no production path sees it."""

    def __init__(
        self,
        episodes: Optional[List[Dict[str, Any]]] = None,
        ratings: Optional[List[Dict[str, Any]]] = None,
        *,
        write_fails: Optional[set] = None,
        read_fails: Optional[set] = None,
    ) -> None:
        self.tables = {
            "composer_episodes": [dict(e) for e in (episodes or [])],
            "chatbot_message_feedback": [dict(r) for r in (ratings or [])],
        }
        self.write_fails = write_fails or set()
        self.read_fails = read_fails or set()
        self.updates: List[tuple] = []
        self.selected: Dict[str, str] = {}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)


@pytest.mark.unit
def test_the_claim_is_written_in_the_same_update_as_the_label():
    """One single-row UPDATE: with PostgREST every execute() is its own transaction, so a label
    and its attribution written separately could persist apart."""
    db = FakeDB(_run1_episodes(), _run1_ratings())

    result = tasks.link_composition_feedback(client=db)

    assert result["labelled"] == 2 and result["failed"] == 0
    payloads = dict(db.updates)
    assert payloads["B"]["feedback_id"] == R1_ID
    assert payloads["A"]["feedback_id"] == R2_ID
    for composition_id, payload in payloads.items():
        assert set(payload) == {"success", "feedback_at", "feedback_id"}, composition_id
    # feedback_text is never written: the rating is the signal, the comment is free user text.
    assert all("feedback_text" not in payload for payload in payloads.values())


@pytest.mark.unit
def test_the_episode_query_selects_the_claim_column_it_seeds_from():
    db = FakeDB(_run1_episodes(), _run1_ratings())
    tasks.link_composition_feedback(client=db)
    assert "feedback_id" in db.selected["composer_episodes"].split(", ")
    # And the rating side must return the id that becomes the claim.
    assert "id" in db.selected["chatbot_message_feedback"].split(", ")


@pytest.mark.unit
def test_the_failed_write_is_retried_on_the_next_run_after_its_rating_is_deleted():
    """End to end over two runs: run 1 loses A's write, R1 is deleted, run 2 must still reach A."""
    db = FakeDB(_run1_episodes(), _run1_ratings(), write_fails={"A"})

    first = tasks.link_composition_feedback(client=db)
    assert first == {"labelled": 1, "considered": 2, "failed": 1}
    assert dict(db.updates)["B"]["feedback_id"] == R1_ID

    # Run 2: B carries its recorded claim, A is still pending, and R1 has been deleted.
    second_db = FakeDB(
        [
            _episode("A", 20),
            _episode("B", 10, success=True, feedback_at=_iso(9), feedback_id=R1_ID),
        ],
        [_rating(R2_ID, 8)],
    )
    second = tasks.link_composition_feedback(client=second_db)

    assert second["labelled"] == 1 and second["failed"] == 0
    assert dict(second_db.updates)["A"]["feedback_id"] == R2_ID
    assert "B" not in dict(second_db.updates), "a labelled composition is never rewritten"


@pytest.mark.unit
def test_a_claim_whose_rating_has_vanished_is_logged_at_warning(caplog):
    """The divergence is detectable instead of silent: the dangling id is deliberate, but a
    reader needs to know the rating behind a label is gone."""
    db = FakeDB(
        [
            _episode("A", 20),
            _episode("B", 10, success=True, feedback_at=_iso(9), feedback_id=R1_ID),
        ],
        [_rating(R2_ID, 8)],
    )
    with caplog.at_level(logging.WARNING, logger=tasks.logger.name):
        tasks.link_composition_feedback(client=db)

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(str(R1_ID) in message and "B" in message for message in warnings), warnings


@pytest.mark.unit
def test_no_warning_when_every_recorded_claim_still_has_its_rating(caplog):
    """The ordinary case must be silent, or the warning is noise nobody reads.

    Mutation-checked: making _warn_on_vanished_claims fire unconditionally fails this test.
    """
    db = FakeDB(
        [
            _episode("A", 20),
            _episode("B", 10, success=True, feedback_at=_iso(9), feedback_id=R1_ID),
        ],
        _run1_ratings(),  # R1 is still alive, so B's recorded claim still resolves
    )
    with caplog.at_level(logging.WARNING, logger=tasks.logger.name):
        tasks.link_composition_feedback(client=db)

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.unit
def test_a_dangling_claim_is_warned_even_when_nothing_is_pending(caplog):
    """The steady state is that EVERY episode is already labelled, so pending is empty.

    If the check sits behind the no-pending exit it never runs in exactly the case it is for:
    a fully-labelled window in which one claim's rating has since been deleted.
    """
    db = FakeDB(
        [_episode("B", 10, success=True, feedback_at=_iso(9), feedback_id=R1_ID)],
        [],  # R1 deleted, and nothing else rated
    )
    with caplog.at_level(logging.WARNING, logger=tasks.logger.name):
        result = tasks.link_composition_feedback(client=db)

    assert result == {"labelled": 0, "considered": 0, "failed": 0}
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(str(R1_ID) in message and "B" in message for message in warnings), warnings


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_db, expected_considered, expect_warning",
    [
        (lambda: FakeDB(read_fails={"composer_episodes"}), 0, True),
        (lambda: FakeDB([], []), 0, False),
        (lambda: FakeDB(_run1_episodes(), [], read_fails={"chatbot_message_feedback"}), 2, True),
    ],
    ids=["episode-read-failed", "nothing-pending", "rating-read-failed"],
)
def test_every_early_exit_reports_failed(make_db, expected_considered, expect_warning, caplog):
    """``failed`` absent on an early exit made a starved run byte-identical to a clean no-op, and
    the issue's own mitigation ("treat failed > 0 as the signal to inspect") unusable.

    The doubles are built HERE, not in the parametrize list: constructed at collection time they
    would be shared across the whole session and mutated by whichever test ran first.
    """
    db = make_db()
    with caplog.at_level(logging.WARNING, logger=tasks.logger.name):
        result = tasks.link_composition_feedback(client=db)

    assert result["failed"] == 0
    assert result["considered"] == expected_considered
    assert set(result) == {"labelled", "considered", "failed"}
    # `failed` counts WRITE attempts. A read failure aborts the run before any write is
    # attempted, so it is reported in the log and not in the count — see _result's docstring.
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert bool(warned) is expect_warning, warned


@pytest.mark.unit
def test_the_no_client_exit_reports_failed(monkeypatch):
    import src.memory.services.factories as factories

    monkeypatch.setattr(factories, "get_supabase_client", lambda *a, **k: None)
    result = tasks.link_composition_feedback()
    assert result == {"labelled": 0, "considered": 0, "failed": 0}
