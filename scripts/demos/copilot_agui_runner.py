"""Copilot demo-question runner for the AG-UI runtime surface (the real UI brain).

The browser copilot does NOT call /api/copilotkit/chat/stream — it speaks the
CopilotKit remote-endpoint protocol against POST /api/copilotkit/agent/default,
which runs the chat_node + E2I_CHATBOT_TOOLS + synthesize_node graph defined in
src/api/routes/copilotkit.py. That graph is a different brain from the
classify->orchestrator->generate graph behind /chat/stream (which fails closed
on conversational queries — see the shadow-pass results). This runner records
the answers the demo audience would actually see.

Protocol (derived from copilotkit/integrations/fastapi.py handler):
  POST {api_base}/copilotkit/agent/default
  {"threadId": ..., "state": {}, "messages": [CopilotKit message dicts], "actions": []}
The frontend resends the FULL message history each turn; threadId is the
persistent session identifier the server uses for DB persistence. Follow-up
turns therefore reuse the session's threadId and append to its history.

Measured per turn:
- ttfb_ms           — first text-content delta
- first_progress_ms — first CoAgent state-sync event carrying progress_steps /
                      progress_percent (the AgentProgressRenderer feed)
- total_ms          — request start -> run-finished (or stream end)
- tools invoked (recorded in agents_dispatched), full answer text, raw events.

Auth mirrors copilot_chat_perf_runner (GoTrue password grant). Routing fields
(routing_pattern etc.) stay blank on this surface unless orchestrator_tool
fires — they are joined later from classification_logs by session_id.
"""

import argparse
import csv
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from copilot_chat_perf_runner import (  # noqa: E402
    CSV_COLUMNS,
    jwt_sub,
    mint_token,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("copilot_agui_runner")


def _norm(event_type: Optional[str]) -> str:
    """Normalize AG-UI event type names (TEXT_MESSAGE_CONTENT == TextMessageContent)."""
    return (event_type or "").replace("_", "").lower()


def _contains_progress(obj: Any) -> bool:
    """True if a (possibly nested) event payload carries CoAgent progress state."""
    if isinstance(obj, dict):
        if "progress_steps" in obj or "progress_percent" in obj:
            return True
        return any(_contains_progress(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_progress(v) for v in obj)
    return False


def _manual_emit_text(event: Dict[str, Any]) -> Optional[str]:
    """Assistant text carried by a ``copilotkit_manually_emit_message`` RAW event.

    Measured shape (2026-08-18 certification run, turn 3.3 events[103]):

        {"type": "RAW", "event": {"event": "on_custom_event",
         "name": "copilotkit_manually_emit_message",
         "data": {"message": "...", "message_id": "...", "role": "assistant"},
         ...}, ...}

    Returns None for anything that is not an assistant-role manual emit.
    """
    if _norm(event.get("type")) != "raw":
        return None
    inner = event.get("event")
    if not isinstance(inner, dict) or inner.get("name") != "copilotkit_manually_emit_message":
        return None
    data = inner.get("data")
    if not isinstance(data, dict):
        return None
    if data.get("role") not in (None, "assistant"):
        return None
    message = data.get("message")
    return message if isinstance(message, str) else None


def fold_stream_text(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold a turn's recorded AG-UI events into the text the UI renders (#1699).

    Pure function over the recorded ``events`` list (the same dicts stored in
    the raw JSONL), extracted from ``run_turn`` so it can be tested against
    real captured streams.

    Two channels deliver assistant text, and only their union matches what the
    CopilotKit UI renders (measured 2026-08-18, turn 3.3: streamed 1,265 chars
    vs 1,481 persisted — the missing 216-char guard note travelled ONLY as a
    manual emit and was invisible to all four graders + a marker sweep):

    - ``TEXT_MESSAGE_CONTENT`` deltas — closed into segments on
      ``TEXT_MESSAGE_END``, joined with a blank line (unchanged behaviour);
    - ``copilotkit_manually_emit_message`` RAW events. CAUTION: the server
      also MIRRORS every streamed chunk over this channel (measured: all 51
      turns of the 2026-08-18 run carry mirror emits, sometimes flushed only
      after ``TEXT_MESSAGE_END``), so payloads are deduplicated against the
      streamed text and only genuinely-new content is folded in, at its
      stream position.

    Reconciliation: the final ``MESSAGES_SNAPSHOT``'s assistant messages for
    THIS turn (after its user message; the last assistant message alone is
    wrong for multi-segment turns, e.g. 2.1's 408+1,390-char pair) are compared
    against the folded text. On divergence neither side is silently trusted:
    ``snapshot_mismatch`` records the discrepancy.

    Returns a dict with:
    - ``messages_out``   — closed TEXT_MESSAGE segments, in stream order
      (meaning unchanged: manual-emit text is folded into ``response_text``
      only);
    - ``response_text``  — the rendered turn text;
    - ``snapshot_mismatch`` — None when consistent (or no snapshot arrived),
      else ``{response_text_len, snapshot_len, divergence_at,
      response_text_at_divergence, snapshot_at_divergence}``.
    """
    # ---- pass 1: collect segments, manual emits (in order), final snapshot ----
    pieces: List[tuple] = []  # ("segment"|"manual", text) in stream order
    messages_out: List[str] = []
    snapshot_messages: Optional[List[Any]] = None
    current = ""
    for event in events:
        etype = _norm(event.get("type"))
        if etype == "textmessagecontent":
            current += event.get("delta") or event.get("content") or ""
        elif etype == "textmessageend":
            if current:
                pieces.append(("segment", current))
                messages_out.append(current)
            current = ""
        elif etype == "messagessnapshot":
            messages = event.get("messages")
            if isinstance(messages, list):
                snapshot_messages = messages  # keep the LAST snapshot
        else:
            manual = _manual_emit_text(event)
            if manual:
                pieces.append(("manual", manual))
    if current:  # unterminated stream (transport error mid-message)
        pieces.append(("segment", current))
        messages_out.append(current)

    # ---- pass 2: drop mirror emits, keep genuinely-new manual content ----
    # Mirror emits reproduce the streamed delta chunks in order (a contiguous
    # run within the concatenated streamed text); they can lead or lag the
    # deltas, so classification happens after the stream completes. A payload
    # that neither continues the mirror run nor anchors anywhere in the
    # streamed text is genuinely new.
    streamed_concat = "".join(messages_out)
    folded_pieces: List[tuple] = []
    cursor = -1  # mirror cursor into streamed_concat; -1 = not anchored yet
    for kind, text in pieces:
        if kind == "segment":
            folded_pieces.append((kind, text))
            continue
        if cursor >= 0 and streamed_concat.startswith(text, cursor):
            cursor += len(text)  # mirror continuation
            continue
        anchor = streamed_concat.find(text)
        if anchor >= 0:
            cursor = anchor + len(text)  # mirror (re)anchor
            continue
        folded_pieces.append(("manual", text))

    # ---- pass 3: assemble. Segments keep the old blank-line join; genuine
    # manual content concatenates directly at its stream position (measured:
    # the server persists it appended to the assistant message verbatim). ----
    parts: List[str] = []
    seen_segment = False
    for kind, text in folded_pieces:
        if kind == "segment":
            if seen_segment:
                parts.append("\n\n")
            seen_segment = True
        parts.append(text)
    response_text = "".join(parts)

    # ---- pass 4: reconcile against the final MESSAGES_SNAPSHOT ----
    snapshot_mismatch: Optional[Dict[str, Any]] = None
    if snapshot_messages:
        last_user = -1
        for i, message in enumerate(snapshot_messages):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user = i
        contents = [
            (message.get("content") or "")
            for message in snapshot_messages[last_user + 1 :]
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        snapshot_text = "\n\n".join(c for c in contents if c)
        if snapshot_text != response_text:
            limit = min(len(snapshot_text), len(response_text))
            n = 0
            while n < limit and snapshot_text[n] == response_text[n]:
                n += 1
            snapshot_mismatch = {
                "response_text_len": len(response_text),
                "snapshot_len": len(snapshot_text),
                "divergence_at": n,
                "response_text_at_divergence": response_text[n : n + 400],
                "snapshot_at_divergence": snapshot_text[n : n + 400],
            }

    return {
        "messages_out": messages_out,
        "response_text": response_text,
        "snapshot_mismatch": snapshot_mismatch,
    }


def user_message(text: str) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": "TextMessage",
        "role": "user",
        "content": text,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_turn(
    api_base: str,
    token: str,
    question: Dict[str, Any],
    thread_id: str,
    history: List[Dict[str, Any]],
    label: str,
    timeout: int,
) -> Dict[str, Any]:
    """Send one question over the AG-UI protocol; parse the event stream. Never raises."""
    request_id = f"perf-{label}-{question['question_id']}"
    history.append(user_message(question["text"]))
    payload = {
        "threadId": thread_id,
        "state": {},
        "messages": history,
        "actions": [],
    }
    record: Dict[str, Any] = {
        "question_id": question["question_id"],
        "session": question["session"],
        "tier": question.get("tier"),
        "intent_expected": question.get("intent_expected"),
        "label": label,
        "request_id": request_id,
        "ts_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session_id": thread_id,
        "events": [],
        "response_text": "",
        "messages_out": [],
        "tools_invoked": [],
        "dispatch_info": {},
        "ttfb_ms": None,
        "first_progress_ms": None,
        "total_ms": None,
        "error": None,
        "http_status": None,
        # Set when the final MESSAGES_SNAPSHOT's assistant text for this turn
        # diverges from the accumulated response_text (#1699). Additive field;
        # None when consistent (or when no snapshot arrived).
        "snapshot_mismatch": None,
        # How many frames actually reached us. Recorded in the raw jsonl rather
        # than the CSV because CSV_COLUMNS is shared with copilot_chat_perf_runner;
        # the CSV already surfaces the outcome via `error` + `response_chars`.
        "stream_frames": 0,
    }
    t0 = time.monotonic()
    try:
        with httpx.stream(
            "POST",
            f"{api_base}/copilotkit/agent/default",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=httpx.Timeout(timeout, connect=30),
        ) as resp:
            record["http_status"] = resp.status_code
            if resp.status_code != 200:
                body = resp.read().decode(errors="replace")[:300]
                record["error"] = f"HTTP {resp.status_code}: {body}"
                history.pop()  # keep history consistent on failure
                return record
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                now_ms = (time.monotonic() - t0) * 1000
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    record["events"].append({"t_ms": round(now_ms, 1), "raw": line[:500]})
                    continue
                record["events"].append({"t_ms": round(now_ms, 1), **event})
                etype = _norm(event.get("type"))
                if etype == "textmessagecontent":
                    if record["ttfb_ms"] is None:
                        record["ttfb_ms"] = round(now_ms, 1)
                elif etype in ("toolcallstart", "actionexecutionstart"):
                    name = event.get("toolCallName") or event.get("actionName") or event.get("name")
                    if name:
                        record["tools_invoked"].append(name)
                elif etype in ("runerror", "error"):
                    record["error"] = event.get("message") or event.get("data") or "run error"
                elif etype == "runfinished":
                    break
                if record["first_progress_ms"] is None and _contains_progress(event):
                    record["first_progress_ms"] = round(now_ms, 1)
        record["total_ms"] = round((time.monotonic() - t0) * 1000, 1)
    except Exception as exc:  # noqa: BLE001 - fail-soft per turn; the run must survive
        record["total_ms"] = round((time.monotonic() - t0) * 1000, 1)
        record["error"] = f"{type(exc).__name__}: {exc}"
    folded = fold_stream_text(record["events"])
    record["messages_out"] = folded["messages_out"]
    record["response_text"] = folded["response_text"]
    record["snapshot_mismatch"] = folded["snapshot_mismatch"]
    if record["snapshot_mismatch"]:
        logger.warning(
            "[%s] snapshot mismatch: response_text=%d chars, snapshot=%d chars (#1699)",
            request_id,
            record["snapshot_mismatch"]["response_text_len"],
            record["snapshot_mismatch"]["snapshot_len"],
        )
    record["stream_frames"] = len(record["events"])
    _grade_stream_health(record)
    # Mirror the frontend: the assistant's reply joins the resent history.
    if record["response_text"]:
        history.append(
            {
                "id": str(uuid.uuid4()),
                "type": "TextMessage",
                "role": "assistant",
                "content": record["response_text"],
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    return record


def client_fatal_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Frames the browser's ``@ag-ui/core`` Zod schema rejects outright.

    The frontend aborts the WHOLE CopilotKit run at the first invalid event —
    measured live 2026-08-19 (session no90vkf): ``TEXT_MESSAGE_CONTENT`` with
    ``delta: ""`` ("Delta must not be an empty string") killed every browser
    tool+synthesis turn while every server-side artifact stayed green. This
    runner's lenient parser keeps reading past such frames, so certification
    must mirror the client's rule explicitly.

    Mirrored rule (the one that has bitten): ``TextMessageContentEventSchema``
    requires ``delta`` to be a NON-EMPTY STRING. ``TOOL_CALL_ARGS``'s delta is
    unconstrained in the TS schema — deliberately not checked here.
    """
    fatal: List[Dict[str, Any]] = []
    for event in events:
        if _norm(event.get("type")) != "textmessagecontent":
            continue
        delta = event.get("delta")
        if not isinstance(delta, str) or len(delta) == 0:
            fatal.append(event)
    return fatal


def _grade_stream_health(record: Dict[str, Any]) -> None:
    """Fail a turn that returned HTTP 200 and delivered nothing (#1667).

    ``StreamingResponse`` commits the status line before the body generator is
    iterated, so an exception inside ``LangGraphAgent.execute`` produces a
    well-formed **HTTP 200 with an empty body**. #1662 did exactly that to every
    AG-UI run for a full day, and because this runner only ever set ``error``
    from a transport exception or an explicit ``RUN_ERROR`` frame, a 51-turn
    sweep would have reported **51/51 OK with every answer empty** — our
    headline quality metric, green throughout.

    Two distinct failures, graded separately because they mean different things:

    * **zero frames** — the stream never produced anything. This is the #1662
      signature and is unambiguous.
    * **frames but no answer text** — the run streamed machinery and no prose.
      Safe to fail HERE because this runner sends ``"actions": []``: with no
      frontend actions registered there is no legitimate way for a turn to
      deliver its answer other than ``TEXT_MESSAGE_CONTENT`` or a
      ``copilotkit_manually_emit_message`` payload (both folded into
      ``response_text`` since #1699). A runner that starts registering
      actions must revisit this.

    Never overwrites an existing ``error`` — a transport failure or ``RUN_ERROR``
    is the more specific diagnosis and already fails the turn.
    """
    if record.get("error"):
        return
    if record.get("http_status") != 200:
        return
    if not record.get("stream_frames"):
        record["error"] = "empty stream: HTTP 200 with 0 frames — the run is dead (#1667)"
        return
    fatal = client_fatal_events(record.get("events") or [])
    if fatal:
        first = fatal[0]
        record["error"] = (
            f"client-fatal stream: {len(fatal)} TEXT_MESSAGE_CONTENT frame(s) with "
            f"empty/non-string delta (first at t={first.get('t_ms')}ms) — the browser's "
            "@ag-ui/core Zod validation aborts the whole run at this frame even though "
            "the rest of the stream is healthy (session no90vkf, 2026-08-19)"
        )
        return
    if not (record.get("response_text") or "").strip():
        record["error"] = (
            f"no answer delivered: HTTP 200 with {record['stream_frames']} frames but "
            "zero rendered text (TEXT_MESSAGE_CONTENT + manual emits, #1667/#1699)"
        )


def to_csv_row(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question_id": record["question_id"],
        "session": record["session"],
        "tier": record.get("tier"),
        "intent_expected": record.get("intent_expected"),
        "intent_actual": "",  # joined post-run from chatbot_training_signals
        "routing_pattern": "",  # joined post-run from classification_logs (orchestrator_tool turns)
        "agents_dispatched": "|".join(record.get("tools_invoked") or []),
        "routed_agent": "",
        "orchestrator_used": "orchestrator_tool" in (record.get("tools_invoked") or []),
        "ttfb_ms": record.get("ttfb_ms"),
        "first_progress_ms": record.get("first_progress_ms"),
        "total_ms": record.get("total_ms"),
        "classification_latency_ms": "",
        "used_llm_layer": "",
        "execution_time_ms": "",
        "intent_confidence": "",
        "response_confidence": "",
        "response_chars": len(record.get("response_text") or ""),
        "session_id": record.get("session_id"),
        "request_id": record.get("request_id"),
        "error": record.get("error") or "",
        "answer_correct": "",  # graded post-run
        "suggestion_pills_relevant": "",  # UI pass only
        "notes": "",
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Copilot AG-UI (real UI brain) question runner.")
    parser.add_argument(
        "--api-base",
        default="https://eznomics.site/api",
        help="API base URL (default: https://eznomics.site/api)",
    )
    parser.add_argument(
        "--questions",
        default="scripts/demos/copilot_demo_questions.json",
        help="Path to the question-set JSON",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/demos/results/2026-07-29_copilot_chat_perf",
        help="Directory for raw_<label>.jsonl and measurements_<label>.csv",
    )
    parser.add_argument("--label", default="agui", help="Run label — tags request ids and files")
    parser.add_argument(
        "--filter",
        default=None,
        help="Comma-separated question_id prefixes to run",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions")
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds between turns")
    parser.add_argument("--timeout", type=int, default=300, help="Per-turn stream timeout (s)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; send nothing")
    args = parser.parse_args(argv)

    questions = json.loads(Path(args.questions).read_text())["questions"]
    if args.filter:
        wanted = {p.strip() for p in args.filter.split(",") if p.strip()}
        questions = [q for q in questions if any(q["question_id"].startswith(w) for w in wanted)]
    if args.limit is not None:
        questions = questions[: args.limit]

    if args.dry_run:
        for q in questions:
            fu = " (follow-up)" if q.get("follow_up") else ""
            print(f"[dry-run] {q['question_id']:12} session={q['session']:4}{fu} {q['text'][:70]}")
        print(f"[dry-run] {len(questions)} questions -> {args.api_base}/copilotkit/agent/default")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"raw_{args.label}.jsonl"
    csv_path = out_dir / f"measurements_{args.label}.csv"

    token = mint_token()
    logger.info("minted token for user %s", jwt_sub(token))
    threads: Dict[str, str] = {}
    histories: Dict[str, List[Dict[str, Any]]] = {}
    ok = failed = 0

    with raw_path.open("a") as raw_f, csv_path.open("w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, q in enumerate(questions, 1):
            session_key = q["session"]
            thread_id = threads.setdefault(session_key, str(uuid.uuid4()))
            history = histories.setdefault(session_key, [])
            record = run_turn(args.api_base, token, q, thread_id, history, args.label, args.timeout)
            if record.get("error") and str(record.get("http_status")) == "401":
                logger.info(
                    "[%d/%d] %s got 401 — re-minting token", i, len(questions), q["question_id"]
                )
                token = mint_token()
                record = run_turn(
                    args.api_base, token, q, thread_id, history, args.label, args.timeout
                )
            raw_f.write(json.dumps(record) + "\n")
            raw_f.flush()
            writer.writerow(to_csv_row(record))
            csv_f.flush()
            if record.get("error"):
                failed += 1
                logger.warning(
                    "[%d/%d] %s FAILED %s", i, len(questions), q["question_id"], record["error"]
                )
            else:
                ok += 1
                logger.info(
                    "[%d/%d] %s OK tools=%s ttfb=%sms progress=%sms total=%sms chars=%d",
                    i,
                    len(questions),
                    q["question_id"],
                    ",".join(record.get("tools_invoked") or []) or "-",
                    record.get("ttfb_ms"),
                    record.get("first_progress_ms"),
                    record.get("total_ms"),
                    len(record.get("response_text") or ""),
                )
            if i < len(questions):
                time.sleep(args.sleep)

    print(f"run complete ({args.label}): {ok} ok, {failed} failed of {len(questions)}")
    print(f"raw:  {raw_path}")
    print(f"csv:  {csv_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
