"""T5 paraphrase-repeat probe (#1339) — the cheapest disproof for "semantic answer reuse".

Issue #1339 reframes the old T5 criterion (verbatim repeat faster than cold) into
a PARAPHRASE-repeat question, and asks — cheapest-disproof-first — whether the
AG-UI chat brain's episodic recap ALREADY satisfies users on a paraphrase repeat
before we build any pre-LLM semantic answer-reuse / bypass layer.

This driver runs a small, faithful experiment on the real UI brain
(POST /api/copilotkit/agent/default, the same surface the browser copilot uses),
reusing the AG-UI runner's auth + event-stream machinery. It measures, per turn:
  - latency (ttfb_ms, total_ms)
  - which tools re-executed (tools_invoked) — did the paraphrase re-run the chain?
  - the full answer text — so the earlier analysis's reuse/acknowledgement and the
    consistency of grounded numbers can be judged from the transcript.

Design (12 real turns, sequential — each is real LLM spend):
  3 in-session pairs, each in its own probe session:
      cold data question -> one intervening off-topic turn -> PARAPHRASE of the cold ask
  3 cold-baseline turns: the SAME paraphrase text asked in a FRESH session (no history),
      isolating the effect of session memory from the wording.

Probe sessions use "t5probe-" thread ids so the prod-write rows are clearly marked
(session_id is character varying; the prefix is a valid key and never a real user's).
Never prints credentials or tokens.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# The main-repo .env (this runs from a worktree whose cwd has no .env of its own).
_MAIN_ENV = "/home/enunez/Projects/e2i_causal_analytics/.env"
load_dotenv(_MAIN_ENV)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from copilot_agui_runner import run_turn  # noqa: E402
from copilot_chat_perf_runner import jwt_sub, mint_token  # noqa: E402

# Recap / reuse phrasing — a coarse heuristic flag only. The authoritative judgment
# is a human read of the transcript; this just surfaces candidates.
RECAP_MARKERS = [
    "as i mentioned",
    "as noted",
    "as i noted",
    "earlier",
    "previously",
    "as before",
    "same as",
    "as i said",
    "to recap",
    "recap",
    "already",
    "reiterat",
    "as we discussed",
    "restat",
    "as reported",
    "as i reported",
    "like i said",
    "we established",
    "i shared",
    "i already",
]


def q(question_id: str, session: str, tier: str, text: str, condition: str) -> Dict[str, Any]:
    return {
        "question_id": question_id,
        "session": session,
        "tier": tier,
        "intent_expected": "",
        "text": text,
        "condition": condition,
    }


# The turn plan. Cold data questions reuse the known-grounded 1.1 / 1.4 phrasings so
# tool-grounding is expected; paraphrases are genuine restatements (different words,
# same referent), each prefaced with a "remind me / again" cue a real user would use.
TURN_PLAN: List[Dict[str, Any]] = [
    # ---- Pair 1: TRx value (Kisqali) ----
    q("t5p1-cold", "s1", "T2", "What is TRx for Kisqali?", "cold"),
    q("t5p1-mid", "s1", "T2", "And what is NRx for Fabhalta?", "intervening"),
    q(
        "t5p1-para",
        "s1",
        "T5",
        "Remind me — what was Kisqali's total prescription count again?",
        "paraphrase",
    ),
    # ---- Pair 2: TRx share (Fabhalta) ----
    q("t5p2-cold", "s2", "T2", "What is the TRx share for Fabhalta?", "cold"),
    q("t5p2-mid", "s2", "T2", "What about Remibrutinib's NRx?", "intervening"),
    q(
        "t5p2-para",
        "s2",
        "T5",
        "Can you tell me again what portion of total scripts Fabhalta holds?",
        "paraphrase",
    ),
    # ---- Pair 3: causal driver (Kisqali NE decline) ----
    q("t5p3-cold", "s3", "T3", "Why did Kisqali TRx drop in Q1 in the northeast region?", "cold"),
    q("t5p3-mid", "s3", "T2", "What is TRx for Remibrutinib?", "intervening"),
    q(
        "t5p3-para",
        "s3",
        "T5",
        "Circle back to Kisqali — what was driving that Northeast decline again?",
        "paraphrase",
    ),
    # ---- Cold baselines: the SAME paraphrase text, fresh session, no history ----
    q(
        "t5b1-baseline",
        "b1",
        "T5",
        "Remind me — what was Kisqali's total prescription count again?",
        "baseline",
    ),
    q(
        "t5b2-baseline",
        "b2",
        "T5",
        "Can you tell me again what portion of total scripts Fabhalta holds?",
        "baseline",
    ),
    q(
        "t5b3-baseline",
        "b3",
        "T5",
        "Circle back to Kisqali — what was driving that Northeast decline again?",
        "baseline",
    ),
]


def recap_hits(text: str) -> List[str]:
    low = (text or "").lower()
    return [m for m in RECAP_MARKERS if m in low]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T5 paraphrase-repeat probe (#1339).")
    parser.add_argument("--api-base", default="https://eznomics.site/api")
    parser.add_argument(
        "--out-dir",
        default="docs/demos/results/2026-07-31_t5_paraphrase_repeat",
        help="Directory for raw_t5probe.jsonl and the summary",
    )
    parser.add_argument("--sleep", type=float, default=6.0, help="Seconds between turns")
    parser.add_argument("--timeout", type=int, default=240, help="Per-turn stream timeout (s)")
    parser.add_argument(
        "--auth-check",
        action="store_true",
        help="Mint a token and exit (no LLM turns, no cost)",
    )
    args = parser.parse_args(argv)

    token = mint_token()
    print(f"minted token for user {jwt_sub(token)}", flush=True)
    if args.auth_check:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_t5probe.jsonl"

    # Per-session thread id (clearly-marked probe) + resent history.
    threads: Dict[str, str] = {}
    histories: Dict[str, List[Dict[str, Any]]] = {}
    records: List[Dict[str, Any]] = []

    with raw_path.open("w") as raw_f:
        for i, question in enumerate(TURN_PLAN, 1):
            skey = question["session"]
            thread_id = threads.setdefault(skey, f"t5probe-{skey}-{uuid.uuid4()}")
            history = histories.setdefault(skey, [])
            record = run_turn(
                args.api_base, token, question, thread_id, history, "t5probe", args.timeout
            )
            if record.get("error") and str(record.get("http_status")) == "401":
                print(f"[{i}] {question['question_id']} 401 — re-minting", flush=True)
                token = mint_token()
                record = run_turn(
                    args.api_base, token, question, thread_id, history, "t5probe", args.timeout
                )
            record["condition"] = question["condition"]
            hits = recap_hits(record.get("response_text") or "")
            record["recap_marker_hits"] = hits
            records.append(record)
            raw_f.write(json.dumps(record) + "\n")
            raw_f.flush()
            print(
                f"[{i}/{len(TURN_PLAN)}] {question['question_id']:15} "
                f"cond={question['condition']:11} "
                f"tools={','.join(record.get('tools_invoked') or []) or '-':22} "
                f"ttfb={record.get('ttfb_ms')}ms total={record.get('total_ms')}ms "
                f"chars={len(record.get('response_text') or '')} "
                f"recap={hits if hits else '-'} "
                f"err={record.get('error') or '-'}",
                flush=True,
            )
            if i < len(TURN_PLAN):
                time.sleep(args.sleep)

    # Compact summary for quick reading; full text lives in the jsonl + the doc.
    summary_path = out_dir / "summary_raw.json"
    summary = [
        {
            "question_id": r["question_id"],
            "condition": r.get("condition"),
            "tools_invoked": r.get("tools_invoked"),
            "ttfb_ms": r.get("ttfb_ms"),
            "total_ms": r.get("total_ms"),
            "chars": len(r.get("response_text") or ""),
            "recap_marker_hits": r.get("recap_marker_hits"),
            "error": r.get("error"),
            "response_text": r.get("response_text"),
        }
        for r in records
    ]
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nraw: {raw_path}\nsummary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
