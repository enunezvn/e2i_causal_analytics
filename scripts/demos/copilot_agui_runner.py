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
    }
    current_text = ""
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
                    current_text += event.get("delta") or event.get("content") or ""
                elif etype == "textmessageend":
                    if current_text:
                        record["messages_out"].append(current_text)
                    current_text = ""
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
    if current_text:
        record["messages_out"].append(current_text)
    record["response_text"] = "\n\n".join(record["messages_out"])
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
