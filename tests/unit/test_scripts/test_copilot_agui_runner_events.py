"""Guard tests for the AG-UI runner's event folding (#1699).

The defect: ``response_text`` concatenated only ``TEXT_MESSAGE_CONTENT``
deltas. Server-side ``copilotkit_manually_emit_message`` payloads (RAW
custom events) were dropped even though the real CopilotKit UI renders
them. Measured on the 2026-08-18 certification run, turn 3.3: the #1691
guard's correction note travelled ONLY as a manual emit + the final
``MESSAGES_SNAPSHOT`` — persisted assistant message 1,481 chars vs
``response_text`` 1,265 chars — and was therefore invisible to all four
graders AND the dispatcher's marker sweep (five blind measurements over
the wrong field).

Fixtures are trimmed verbatim from that run's raw_agui.jsonl
(``docs/demos/results/2026-08-18_post1696_copilot_chat_perf/``):

- ``turn_3_3_guard_note.json``  — 11 in-stream mirror emits + 1 genuinely
  new post-stream emit (the guard note). Snapshot assistant text = streamed
  1,265 chars + 216-char note = 1,481 chars.
- ``turn_1_5_mirror_emits.json`` — 13 manual emits that are ALL mirrors of
  the streamed text, flushed after TEXT_MESSAGE_END. Naively appending
  every manual-emit payload would DOUBLE the answer; measured corpus-wide,
  every one of the 51 turns carries mirror emits, so the naive fold would
  corrupt the whole run.
- ``turn_2_1_two_segments.json`` — two streamed segments (408 + 1,390
  chars, joined with a blank line); mirrors cover only segment 2. Guards
  the snapshot reconciliation against false positives on multi-message
  turns (the snapshot's LAST assistant message alone is 1,390 chars, not
  the turn's full 1,800-char rendered text).
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.demos.copilot_agui_runner import fold_stream_text

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "agui_runner"

GUARD_MARKER = "Automated table cross-check"

# Measured on the 2026-08-18 certification run (issue #1699).
TURN_3_3_STREAMED_LEN = 1265  # what the defective runner recorded
TURN_3_3_RENDERED_LEN = 1481  # what the UI rendered / DB persisted
TURN_1_5_RENDERED_LEN = 1483
TURN_2_1_RENDERED_LEN = 1800


def load_fixture(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def is_manual_emit(event: Dict[str, Any]) -> bool:
    return (
        event.get("type") == "RAW"
        and (event.get("event") or {}).get("name") == "copilotkit_manually_emit_message"
    )


def manual_emit_text(event: Dict[str, Any]) -> str:
    return ((event.get("event") or {}).get("data") or {}).get("message") or ""


def snapshot_turn_text(events: List[Dict[str, Any]]) -> str:
    """Assistant text of THIS turn per the final MESSAGES_SNAPSHOT.

    Non-empty assistant contents after the last user message, joined the way
    the runner joins streamed segments. This is the text the UI renders.
    """
    snapshots = [e for e in events if e.get("type") == "MESSAGES_SNAPSHOT"]
    messages = snapshots[-1]["messages"]
    last_user = max(i for i, m in enumerate(messages) if m.get("role") == "user")
    contents = [
        m.get("content") or "" for m in messages[last_user + 1 :] if m.get("role") == "assistant"
    ]
    return "\n\n".join(c for c in contents if c)


class TestManualEmitFolding:
    """The fix itself: manual-emit content must reach response_text (#1699)."""

    def test_guard_note_folded_into_response_text(self):
        """RED-FIRST: the turn-3.3 guard note must be part of response_text.

        On the defective accumulation this fails exactly the way the five
        blind measurements did: the marker is absent and the text is 1,265
        chars instead of the rendered 1,481.
        """
        fixture = load_fixture("turn_3_3_guard_note.json")
        folded = fold_stream_text(fixture["events"])

        assert GUARD_MARKER in folded["response_text"]
        assert len(folded["response_text"]) == TURN_3_3_RENDERED_LEN
        # The folded text must equal what the UI renders per the snapshot.
        assert folded["response_text"] == snapshot_turn_text(fixture["events"])
        # And reconciliation must be clean once the note is folded in.
        assert folded["snapshot_mismatch"] is None

    def test_guard_note_appended_in_stream_order(self):
        """The note arrived after the stream closed, so it folds at the end."""
        fixture = load_fixture("turn_3_3_guard_note.json")
        folded = fold_stream_text(fixture["events"])

        streamed = fixture["recorded_response_text"]  # old runner's 1,265 chars
        assert len(streamed) == TURN_3_3_STREAMED_LEN
        assert folded["response_text"].startswith(streamed)
        assert folded["response_text"].endswith(
            "Where prose and table disagree, the table values are authoritative."
        )

    def test_messages_out_meaning_unchanged(self):
        """messages_out keeps meaning: closed TEXT_MESSAGE segments only."""
        fixture = load_fixture("turn_3_3_guard_note.json")
        folded = fold_stream_text(fixture["events"])
        assert folded["messages_out"] == fixture["recorded_messages_out"]


class TestMirrorEmitsNotDoubleCounted:
    """The server mirrors every streamed chunk over the manual-emit channel.

    All 51 turns of the 2026-08-18 run carry mirror emits; folding them
    naively would double the answer text corpus-wide. A turn whose manual
    emits are pure mirrors must fold to EXACTLY the old response_text.
    """

    def test_post_stream_mirror_emits_fold_unchanged(self):
        fixture = load_fixture("turn_1_5_mirror_emits.json")
        assert sum(1 for e in fixture["events"] if is_manual_emit(e)) == 13
        folded = fold_stream_text(fixture["events"])

        assert folded["response_text"] == fixture["recorded_response_text"]
        assert len(folded["response_text"]) == TURN_1_5_RENDERED_LEN
        assert folded["snapshot_mismatch"] is None

    def test_two_segment_turn_folds_unchanged_and_reconciles(self):
        """Two streamed segments; in-stream mirrors cover only segment 2.

        Also guards reconciliation: comparing against the snapshot's LAST
        assistant message alone would flag this healthy turn as a mismatch
        (1,390 != 1,800); the turn-scoped comparison must not.
        """
        fixture = load_fixture("turn_2_1_two_segments.json")
        folded = fold_stream_text(fixture["events"])

        assert folded["response_text"] == fixture["recorded_response_text"]
        assert len(folded["response_text"]) == TURN_2_1_RENDERED_LEN
        assert folded["messages_out"] == fixture["recorded_messages_out"]
        assert folded["snapshot_mismatch"] is None


class TestPlainTurnUnchanged:
    """A turn with no manual emits at all folds exactly as before."""

    def test_no_manual_emits_folds_identically(self):
        fixture = load_fixture("turn_1_5_mirror_emits.json")
        plain_events = [e for e in fixture["events"] if not is_manual_emit(e)]
        assert len(plain_events) < len(fixture["events"])  # something was removed

        folded = fold_stream_text(plain_events)
        assert folded["response_text"] == fixture["recorded_response_text"]
        assert folded["messages_out"] == fixture["recorded_messages_out"]
        assert folded["snapshot_mismatch"] is None


class TestSnapshotReconciliation:
    """The generic detector: snapshot text != accumulated text is recorded.

    Built by removing the genuine manual emit from the real turn-3.3 events:
    the snapshot still carries the 1,481-char message while the stream only
    delivered 1,265 chars — precisely the harness gap #1699 describes, and
    the shape a future unknown emission channel would produce.
    """

    def test_mismatch_recorded_not_silently_trusted(self):
        fixture = load_fixture("turn_3_3_guard_note.json")
        events = [
            e
            for e in fixture["events"]
            if not (is_manual_emit(e) and GUARD_MARKER in manual_emit_text(e))
        ]
        folded = fold_stream_text(events)

        # response_text stays what the stream delivered — neither side is
        # silently trusted; the discrepancy is recorded instead.
        assert folded["response_text"] == fixture["recorded_response_text"]
        mismatch = folded["snapshot_mismatch"]
        assert mismatch is not None
        assert mismatch["response_text_len"] == TURN_3_3_STREAMED_LEN
        assert mismatch["snapshot_len"] == TURN_3_3_RENDERED_LEN
        assert mismatch["divergence_at"] == TURN_3_3_STREAMED_LEN
        assert GUARD_MARKER in mismatch["snapshot_at_divergence"]
