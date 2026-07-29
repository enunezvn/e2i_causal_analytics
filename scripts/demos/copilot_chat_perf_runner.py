#!/usr/bin/env python3
"""Run the Copilot Chat performance/demo question set and record everything.

Drives ``POST /api/copilotkit/chat/stream`` (SSE) for each question in
``scripts/demos/copilot_demo_questions.json`` (transcribed from
docs/demos/COPILOT_CHAT_DEMO_SCENARIOS.md), recording per question:

- ``ttfb_ms``   — time to the first ``text`` SSE event. Caveat: the stream
  diffs node-level ``response_text`` updates, so text arrives in coarse
  chunks and ttfb typically lands close to total; the browser UI pass is
  the source for progress-render timing.
- ``total_ms``  — request start -> ``done`` event (or stream end).
- ``dispatch_info`` — intent, agents_dispatched, routing_pattern,
  classification_latency_ms, used_llm_layer, execution_time_ms.
- the complete answer text and the full raw SSE event list.

Outputs ``raw_<label>.jsonl`` (full capture) and ``measurements_<label>.csv``
(Appendix B sheet; answer_correct / suggestion_pills_relevant / first_progress_ms
are filled by the grading + UI passes).

Session protocol: the first turn of each session group sends an empty
``session_id`` and adopts the server-generated ``user_id~uuid`` from the
stream's first event; follow-up turns reuse it (T5 memory). ``request_id``
is ``perf-<label>-<question_id>`` for DB correlation (classification_logs /
llm_usage_events / chatbot_messages).

Auth mirrors scripts/replay_golden_set.py: GoTrue password grant with
SUPABASE_URL / SUPABASE_ANON_KEY / E2I_ADMIN_EMAIL / E2I_ADMIN_PASSWORD,
body user_id = the JWT ``sub`` (endpoint 403s on mismatch), re-mint on 401.

Usage (from repo root, strictly sequential — never parallelize against prod):
    .venv/bin/python scripts/demos/copilot_chat_perf_runner.py --dry-run
    .venv/bin/python scripts/demos/copilot_chat_perf_runner.py --limit 2      # smoke
    .venv/bin/python scripts/demos/copilot_chat_perf_runner.py --label shadow
    .venv/bin/python scripts/demos/copilot_chat_perf_runner.py --label active \
        --filter 1.4,1.5,1.6,2.5,3.4,4.3,5.7,6.1,6.2,6.4
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("copilot_chat_perf_runner")

CSV_COLUMNS = [
    "question_id",
    "session",
    "tier",
    "intent_expected",
    "intent_actual",
    "routing_pattern",
    "agents_dispatched",
    "routed_agent",
    "orchestrator_used",
    "ttfb_ms",
    "first_progress_ms",
    "total_ms",
    "classification_latency_ms",
    "used_llm_layer",
    "execution_time_ms",
    "intent_confidence",
    "response_confidence",
    "response_chars",
    "session_id",
    "request_id",
    "error",
    "answer_correct",
    "suggestion_pills_relevant",
    "notes",
]


def mint_token() -> str:
    """Mint a JWT via the GoTrue password grant."""
    su = os.environ["SUPABASE_URL"]
    anon = os.environ["SUPABASE_ANON_KEY"]
    email = os.environ.get("E2I_ADMIN_EMAIL", "admin@e2i.local")
    pw = os.environ["E2I_ADMIN_PASSWORD"]
    resp = httpx.post(
        f"{su}/auth/v1/token?grant_type=password",
        headers={"apikey": anon, "Content-Type": "application/json"},
        json={"email": email, "password": pw},
        timeout=30,
    )
    resp.raise_for_status()
    return str(resp.json()["access_token"])


def jwt_sub(token: str) -> str:
    """Decode the JWT payload's ``sub`` claim (the authoritative user id)."""
    payload_b64 = token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    return str(json.loads(base64.urlsafe_b64decode(payload_b64).decode())["sub"])


def stream_one(
    api_base: str,
    token: str,
    user_id: str,
    question: Dict[str, Any],
    session_id: str,
    label: str,
    timeout: int,
) -> Dict[str, Any]:
    """POST one question to /chat/stream; parse SSE with timings. Never raises."""
    request_id = f"perf-{label}-{question['question_id']}"
    payload = {
        "query": question["text"],
        "user_id": user_id,
        "request_id": request_id,
        "session_id": session_id,
    }
    record: Dict[str, Any] = {
        "question_id": question["question_id"],
        "session": question["session"],
        "tier": question.get("tier"),
        "intent_expected": question.get("intent_expected"),
        "label": label,
        "request_id": request_id,
        "ts_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session_id": session_id or None,
        "events": [],
        "response_text": "",
        "dispatch_info": {},
        "ttfb_ms": None,
        "total_ms": None,
        "error": None,
        "http_status": None,
    }
    t0 = time.monotonic()
    try:
        with httpx.stream(
            "POST",
            f"{api_base}/copilotkit/chat/stream",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=httpx.Timeout(timeout, connect=30),
        ) as resp:
            record["http_status"] = resp.status_code
            if resp.status_code != 200:
                body = resp.read().decode(errors="replace")[:300]
                record["error"] = f"HTTP {resp.status_code}: {body}"
                return record
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                now_ms = (time.monotonic() - t0) * 1000
                try:
                    event = json.loads(line[len("data: ") :])
                except json.JSONDecodeError:
                    record["events"].append({"t_ms": round(now_ms, 1), "raw": line[:500]})
                    continue
                etype = event.get("type")
                record["events"].append({"t_ms": round(now_ms, 1), **event})
                if etype == "session_id":
                    record["session_id"] = event.get("data")
                elif etype == "text":
                    if record["ttfb_ms"] is None:
                        record["ttfb_ms"] = round(now_ms, 1)
                    record["response_text"] += event.get("data") or ""
                elif etype == "dispatch_info":
                    record["dispatch_info"] = event.get("data") or {}
                elif etype == "error":
                    record["error"] = event.get("data")
                elif etype == "done":
                    break
        record["total_ms"] = round((time.monotonic() - t0) * 1000, 1)
    except Exception as exc:  # noqa: BLE001 - fail-soft per turn; the run must survive
        record["total_ms"] = round((time.monotonic() - t0) * 1000, 1)
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def to_csv_row(record: Dict[str, Any]) -> Dict[str, Any]:
    info = record.get("dispatch_info") or {}
    return {
        "question_id": record["question_id"],
        "session": record["session"],
        "tier": record.get("tier"),
        "intent_expected": record.get("intent_expected"),
        "intent_actual": info.get("intent"),
        "routing_pattern": info.get("routing_pattern"),
        "agents_dispatched": "|".join(info.get("agents_dispatched") or []),
        "routed_agent": info.get("routed_agent"),
        "orchestrator_used": info.get("orchestrator_used"),
        "ttfb_ms": record.get("ttfb_ms"),
        "first_progress_ms": "",  # UI pass only — /chat/stream has no progress events
        "total_ms": record.get("total_ms"),
        "classification_latency_ms": info.get("classification_latency_ms"),
        "used_llm_layer": info.get("used_llm_layer"),
        "execution_time_ms": info.get("execution_time_ms"),
        "intent_confidence": info.get("intent_confidence"),
        "response_confidence": info.get("response_confidence"),
        "response_chars": len(record.get("response_text") or ""),
        "session_id": record.get("session_id"),
        "request_id": record.get("request_id"),
        "error": record.get("error") or "",
        "answer_correct": "",  # graded post-run
        "suggestion_pills_relevant": "",  # UI pass only
        "notes": "",
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Copilot Chat perf/demo question runner.")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("E2I_API_BASE", "https://eznomics.site/api"),
        help="API base URL (default: E2I_API_BASE or https://eznomics.site/api)",
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
    parser.add_argument(
        "--label",
        default="shadow",
        help="Run label (shadow|active) — tags request ids and output filenames",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Comma-separated question_id prefixes to run (e.g. the active subset)",
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
        print(f"[dry-run] {len(questions)} questions -> {args.api_base}/copilotkit/chat/stream")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"raw_{args.label}.jsonl"
    csv_path = out_dir / f"measurements_{args.label}.csv"

    token = mint_token()
    user_id = jwt_sub(token)
    session_ids: Dict[str, str] = {}
    ok = failed = 0

    with raw_path.open("a") as raw_f, csv_path.open("w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, q in enumerate(questions, 1):
            session_key = q["session"]
            record = stream_one(
                args.api_base,
                token,
                user_id,
                q,
                session_ids.get(session_key, ""),
                args.label,
                args.timeout,
            )
            if record.get("error") and str(record.get("http_status")) == "401":
                logger.info(
                    "[%d/%d] %s got 401 — re-minting token", i, len(questions), q["question_id"]
                )
                token = mint_token()
                user_id = jwt_sub(token)
                record = stream_one(
                    args.api_base,
                    token,
                    user_id,
                    q,
                    session_ids.get(session_key, ""),
                    args.label,
                    args.timeout,
                )
            # Adopt the server-generated session id for this group's follow-ups
            if record.get("session_id"):
                session_ids[session_key] = record["session_id"]
            raw_f.write(json.dumps(record) + "\n")
            raw_f.flush()
            writer.writerow(to_csv_row(record))
            csv_f.flush()
            info = record.get("dispatch_info") or {}
            if record.get("error"):
                failed += 1
                logger.warning(
                    "[%d/%d] %s FAILED %s", i, len(questions), q["question_id"], record["error"]
                )
            else:
                ok += 1
                logger.info(
                    "[%d/%d] %s OK intent=%s pattern=%s agents=%s ttfb=%sms total=%sms chars=%d",
                    i,
                    len(questions),
                    q["question_id"],
                    info.get("intent"),
                    info.get("routing_pattern"),
                    ",".join(info.get("agents_dispatched") or []) or "-",
                    record.get("ttfb_ms"),
                    record.get("total_ms"),
                    len(record.get("response_text") or ""),
                )
            if i < len(questions):
                time.sleep(args.sleep)

    print(f"run complete ({args.label}): {ok} ok, {failed} failed of {len(questions)}")
    print(f"raw:  {raw_path}")
    print(f"csv:  {csv_path}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
