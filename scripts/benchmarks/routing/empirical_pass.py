#!/usr/bin/env python3
"""Empirical pass over the 22 disputed TOOL_COMPOSER-correction queries (#1337/#1341).

The review sheet's proposed verdicts are contract-based HYPOTHESES. This runner
collects the disproving evidence: for each disputed query, execute the PROPOSED
candidate route through the production dispatch machinery and capture the real
response. Combined with a live AG-UI pass (scripts/demos/copilot_agui_runner.py
over the emitted questions file), each proposal gets response evidence instead
of contract reasoning alone.

Mechanism (verified 2026-07-29 explore-dispatch report): OrchestratorAgent.run()
has no forced-intent override — the graph classifies unconditionally — but
RouterNode and DispatcherNode are separately callable production nodes. Preset
``state["intent"]`` and call RouterNode.execute → DispatcherNode.execute against
a real agent registry: only the classifier is bypassed (deliberately — WE choose
the route under test), while INTENT_TO_AGENTS plans, per-agent INPUT_RESOLVERS,
fail-closed semantics, real agents, real DB, and real LLMs all stay production.
A fail-closed NeedsStructuredInput result IS evidence, not a harness bug.

Subcommands::

    SUPABASE_URL=http://localhost:54321 .venv/bin/python \
        scripts/benchmarks/routing/empirical_pass.py manifest
    SUPABASE_URL=http://localhost:54321 .venv/bin/python \
        scripts/benchmarks/routing/empirical_pass.py forced [--only q01,q05] [--limit N]

Run from repo root on the droplet with the real .env (NEVER E2I_TESTING_MODE).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_judge_review import (  # noqa: E402
    fetch_corrections,
    group_rows,
    load_proposals,
    normalize,
)

REVIEW_DIR = Path("scripts/benchmarks/routing/review")
MANIFEST = REVIEW_DIR / "empirical_manifest.json"
LIVE_QUESTIONS = REVIEW_DIR / "empirical_live_questions.json"
RESULTS_DIR = REVIEW_DIR / "empirical_results"
FORCED_SUMMARY = RESULTS_DIR / "forced_summary.jsonl"

# Proposal agent → legacy router intent (INTENT_TO_AGENTS key). The forced run
# tests the PROPOSED route, so `agree` (TOOL_COMPOSER stands) maps to
# multi_faceted → tool_composer.
INTENT_FOR_AGENT = {
    "causal_impact": "causal_effect",
    "gap_analyzer": "performance_gap",
    "experiment_designer": "experiment_design",
    "explainer": "explanation",
    "cohort_profiler": "cohort_definition",
    "resource_optimizer": "resource_allocation",
    "prediction_synthesizer": "prediction",
    "heterogeneous_optimizer": "segment_analysis",
    "tool_composer": "multi_faceted",
}


def route_for(proposed: str) -> tuple[str, str]:
    """(intent, candidate_agent) for a proposed verdict string."""
    if proposed == "agree":
        return "multi_faceted", "tool_composer"
    if proposed.startswith(("single_agent:", "extend:")):
        agent = proposed.split(":", 1)[1].split("—")[0].split()[0].strip()
        return INTENT_FOR_AGENT[agent], agent
    raise ValueError(f"unroutable proposal: {proposed!r}")


async def build_manifest() -> None:
    """Fetch the disputed queries (same grouping/order as the review sheet),
    join the proposals, and emit the forced-run manifest + AG-UI questions."""
    proposals = load_proposals()
    groups = group_rows(await fetch_corrections("TOOL_COMPOSER"))
    entries: List[Dict[str, Any]] = []
    for i, g in enumerate(groups, 1):
        prop = proposals.get(normalize(g["query_text"]))
        if prop is None:
            print(f"WARNING: no proposal for query {i}: {g['query_text'][:60]!r}")
            continue
        intent, agent = route_for(prop["proposed"])
        entries.append(
            {
                "qid": f"q{i:02d}",
                "query_text": g["query_text"].strip(),
                "proposed": prop["proposed"],
                "reasoning": prop["reasoning"],
                "forced_intent": intent,
                "candidate_agent": agent,
                "n_rows": g["n_rows"],
            }
        )
    MANIFEST.write_text(json.dumps({"entries": entries}, indent=2) + "\n")
    # AG-UI questions file: one session per query (no shared history — q07 is
    # follow-up-shaped and will run without prior context; caveat in analysis).
    LIVE_QUESTIONS.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": e["qid"],
                        "session": e["qid"],
                        "text": e["query_text"],
                        "intent_expected": e["forced_intent"],
                    }
                    for e in entries
                ]
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(entries)} entries -> {MANIFEST} + {LIVE_QUESTIONS}")


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


async def run_forced(only: set[str] | None, limit: int | None) -> None:
    from src.agents.factory import create_agent_registry
    from src.agents.orchestrator.nodes.dispatcher import DispatcherNode
    from src.agents.orchestrator.nodes.router import RouterNode
    from src.agents.orchestrator.nodes.synthesizer import SynthesizerNode

    entries = json.loads(MANIFEST.read_text())["entries"]
    if only:
        entries = [e for e in entries if e["qid"] in only]
    if limit is not None:
        entries = entries[:limit]

    needed = sorted({e["candidate_agent"] for e in entries})
    # Fallback agents named in INTENT_TO_AGENTS plans for the routes under test
    # must be registered, or a primary failure would fail closed on the fallback
    # dispatch itself instead of exercising the production fallback path.
    needed = sorted(set(needed) | {"explainer", "gap_analyzer"})
    print(f"registry agents: {needed}")
    registry = create_agent_registry(include_agents=needed)
    missing = [a for a in needed if a not in registry]
    if missing:
        print(f"WARNING: registry is PARTIAL, missing {missing} — their routes fail closed")
    router = RouterNode()
    dispatcher = DispatcherNode(registry)
    synthesizer = SynthesizerNode()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(entries, 1):
        record: Dict[str, Any] = {
            "qid": e["qid"],
            "query_text": e["query_text"],
            "proposed": e["proposed"],
            "forced_intent": e["forced_intent"],
            "candidate_agent": e["candidate_agent"],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        t0 = time.monotonic()
        try:
            state: Dict[str, Any] = {
                "query": e["query_text"],
                "query_id": f"empirical-{e['qid']}",
                "session_id": f"empirical-forced-{e['qid']}",
                "user_context": {},
                "parsed_query": None,
                "agent_results": [],
                "intent": {
                    "primary_intent": e["forced_intent"],
                    "confidence": 1.0,
                    "secondary_intents": [],
                    "requires_multi_agent": False,
                },
            }
            state = await router.execute(state)
            record["dispatch_plan"] = [
                {
                    "agent_name": d.get("agent_name"),
                    "priority": d.get("priority"),
                    "timeout_ms": d.get("timeout_ms"),
                    "fallback_agent": d.get("fallback_agent"),
                }
                for d in state.get("dispatch_plan", [])
            ]
            state = await dispatcher.execute(state)
            record["agent_results"] = [
                {k: _jsonable(v) for k, v in r.items()} for r in state.get("agent_results", [])
            ]
            try:
                state = await synthesizer.execute(state)
                record["synthesized_response"] = _jsonable(state.get("synthesized_response"))
                record["response_confidence"] = state.get("response_confidence")
                record["status"] = state.get("status")
            except Exception as exc:  # noqa: BLE001 - synthesis is best-effort evidence
                record["synthesis_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - fail-soft per query; the pass must survive
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()[-3000:]
        record["total_ms"] = round((time.monotonic() - t0) * 1000, 1)
        out = RESULTS_DIR / f"forced_{e['qid']}_{e['candidate_agent']}.json"
        out.write_text(json.dumps(record, indent=2, default=repr) + "\n")
        with FORCED_SUMMARY.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "qid": e["qid"],
                        "candidate_agent": e["candidate_agent"],
                        "n_results": len(record.get("agent_results", [])),
                        "success": [
                            r.get("success") for r in record.get("agent_results", [])
                        ],
                        "status": record.get("status"),
                        "error": record.get("error"),
                        "total_ms": record["total_ms"],
                        "response_chars": len(record.get("synthesized_response") or ""),
                    }
                )
                + "\n"
            )
        ok = record.get("error") is None
        print(
            f"[{i}/{len(entries)}] {e['qid']} {e['candidate_agent']}"
            f" {'OK' if ok else 'ERROR'} {record['total_ms']}ms"
            f" results={len(record.get('agent_results', []))}"
            f" chars={len(record.get('synthesized_response') or '')}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest", help="Build manifest + AG-UI questions from DB + proposals")
    forced = sub.add_parser("forced", help="Run forced-route dispatches per the manifest")
    forced.add_argument("--only", default=None, help="Comma-separated qids to run")
    forced.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    if args.cmd == "manifest":
        await build_manifest()
    else:
        only = {q.strip() for q in args.only.split(",")} if args.only else None
        await run_forced(only, args.limit)


if __name__ == "__main__":
    asyncio.run(main())
