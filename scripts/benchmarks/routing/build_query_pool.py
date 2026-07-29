#!/usr/bin/env python3
"""Assemble the #1337 Step 0 routing-benchmark query pool.

Step 0 benchmarks three routing-classifier candidates (4-stage pipeline with
prototype LLM stage / single LLM-call classifier / legacy routing) against
judged gold routing. Its prerequisite is a ~200-300 query benchmark set; this
script builds the raw pool from the two free sources:

1. **historical** — distinct real user queries from ``chatbot_messages``
   before the 2026-07-29 demo-recording runs (messy, real distribution:
   fragments, typos, pronoun follow-ups). Each query carries its immediate
   conversation context (previous user + assistant turns in the session) so
   context-dependent fragments ("run both", "is that above baseline?") can be
   gold-labeled meaningfully.
2. **demo** — the 51 doc-authored questions from
   ``scripts/demos/copilot_demo_questions.json`` (clean, tier/intent-annotated;
   also the corpus the recorded legacy-routing baseline covers).

Later pipeline stages (separate scripts) add LLM perturbations of the pool,
authored genuinely-ambiguous queries (CLARIFICATION-is-correct cells), and
gold routing labels via LLM judge + human review. ``gold_pattern`` /
``gold_agents`` are therefore null here.

Usage (from repo root, droplet)::

    SUPABASE_URL=http://localhost:54321 .venv/bin/python \
        scripts/benchmarks/routing/build_query_pool.py

Output: ``scripts/benchmarks/routing/data/query_pool.jsonl`` + a composition
summary on stdout. Deterministic given identical DB state (no randomness;
stable sort by first occurrence).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

DEMO_QUESTIONS = Path("scripts/demos/copilot_demo_questions.json")
DEFAULT_OUTPUT = Path("scripts/benchmarks/routing/data/query_pool.jsonl")
DEMO_CUTOFF = "2026-07-29T00:00:00+00:00"
_FETCH_LIMIT = 5000
_CONTEXT_TRUNCATE = 400


def normalize(text: str) -> str:
    """Dedup key: casefolded, whitespace-collapsed, NFKC."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


async def fetch_messages(cutoff: str) -> List[Dict[str, Any]]:
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    result = await (
        client.table("chatbot_messages")
        .select("id,session_id,role,content,created_at")
        .lt("created_at", cutoff)
        .order("created_at", desc=False)
        .limit(_FETCH_LIMIT)
        .execute()
    )
    rows = list(result.data or [])
    if len(rows) >= _FETCH_LIMIT:
        raise SystemExit(
            f"fetch hit _FETCH_LIMIT={_FETCH_LIMIT}; add pagination before trusting the pool"
        )
    return rows


def build_historical(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Distinct user queries, each with its immediate session context."""
    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for m in messages:
        by_session.setdefault(m.get("session_id") or "", []).append(m)

    pool: Dict[str, Dict[str, Any]] = {}
    for session_id, turns in by_session.items():
        for i, m in enumerate(turns):
            if m.get("role") != "user" or not (m.get("content") or "").strip():
                continue
            key = normalize(m["content"])
            if key in pool:
                pool[key]["n_occurrences"] += 1
                continue
            prev_user: Optional[str] = None
            prev_assistant: Optional[str] = None
            for prior in reversed(turns[:i]):
                if prior.get("role") == "assistant" and prev_assistant is None:
                    prev_assistant = (prior.get("content") or "")[:_CONTEXT_TRUNCATE]
                elif prior.get("role") == "user" and prev_user is None:
                    prev_user = (prior.get("content") or "")[:_CONTEXT_TRUNCATE]
                if prev_user is not None and prev_assistant is not None:
                    break
            is_followup = i > 0
            pool[key] = {
                "text": m["content"].strip(),
                "source": "historical",
                "session_id": session_id,
                "created_at": m.get("created_at"),
                "is_followup": is_followup,
                "context": (
                    {"prev_user": prev_user, "prev_assistant": prev_assistant}
                    if is_followup
                    else None
                ),
                "n_occurrences": 1,
                "demo_meta": None,
                "gold_pattern": None,
                "gold_agents": None,
            }
    return sorted(pool.values(), key=lambda r: (r["created_at"] or "", r["text"]))


def build_demo() -> List[Dict[str, Any]]:
    doc = json.loads(DEMO_QUESTIONS.read_text())
    questions = doc["questions"]
    entries: List[Dict[str, Any]] = []
    prev_in_session: Dict[str, str] = {}
    for q in questions:
        session = q.get("session") or ""
        prev_text = prev_in_session.get(session)
        entries.append(
            {
                "text": q["text"],
                "source": "demo",
                "session_id": f"demo:{session}",
                "created_at": None,
                "is_followup": prev_text is not None,
                "context": {"prev_user": prev_text, "prev_assistant": None}
                if prev_text is not None
                else None,
                "n_occurrences": 1,
                "demo_meta": {
                    k: q.get(k) for k in ("question_id", "tier", "intent_expected", "expected_path")
                },
                "gold_pattern": None,
                "gold_agents": None,
            }
        )
        prev_in_session[session] = q["text"]
    return entries


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default=DEMO_CUTOFF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    load_dotenv()
    messages = await fetch_messages(args.cutoff)
    historical = build_historical(messages)
    demo = build_demo()

    # Demo questions win on text collision (they carry tier/intent annotations).
    demo_keys = {normalize(e["text"]) for e in demo}
    overlap = [e for e in historical if normalize(e["text"]) in demo_keys]
    historical = [e for e in historical if normalize(e["text"]) not in demo_keys]

    entries = demo + historical
    for idx, e in enumerate(entries):
        e["query_id"] = f"pool-{idx:04d}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    followups = sum(1 for e in entries if e["is_followup"])
    print(f"pool: {len(entries)} queries -> {args.output}")
    print(
        f"  demo: {len(demo)}  historical: {len(historical)} (dropped {len(overlap)} demo-duplicates)"
    )
    print(f"  follow-ups with context: {followups}")
    print(
        f"  historical raw user messages scanned: {sum(1 for m in messages if m.get('role') == 'user')}"
    )


if __name__ == "__main__":
    asyncio.run(main())
