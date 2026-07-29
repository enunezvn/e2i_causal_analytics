#!/usr/bin/env python3
"""Emit the human-review worksheet for judge corrections (#1337 / #1341).

The nightly routing labeler's LLM judge (PR #1342) corrected 35/66 rows to
TOOL_COMPOSER on the first full drain. Contract review (5-agent team,
2026-07-29) verified the platform's own routing semantics against
``.claude/contracts/``, per-agent ``CONTRACT_VALIDATION.md``, and the
classifier/router implementation: TOOL_COMPOSER requires the query to span
>= 2 DISTINCT agent-capability domains AND to have dependencies between the
cross-domain sub-questions. "Needs multiple dependent steps" alone is NOT
sufficient — single-domain queries route SINGLE_AGENT before dependency
analysis is even consulted (pattern_selector.py Rule 3), no matter how many
internal steps the owning agent runs.

Contract text comes from ``data/agent_contracts.json`` — the verified
registry produced by that review — never from hand-distilled summaries.

This script groups the corrected rows by distinct query text (the drain
labeled many duplicates), attaches the demo doc's expected path where the
query is one of the 51 authored questions, shows the candidate single
agents' verified contracts, and emits a markdown worksheet with a blank
verdict per query. The human verdicts feed the #1341 judge-prompt fix and
the #1337 Step 0 gold protocol.

Usage (repo root, droplet)::

    SUPABASE_URL=http://localhost:54321 .venv/bin/python \
        scripts/benchmarks/routing/gen_judge_review.py [--correct-pattern TOOL_COMPOSER]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

DEMO_QUESTIONS = Path("scripts/demos/copilot_demo_questions.json")
CONTRACTS = Path("scripts/benchmarks/routing/data/agent_contracts.json")
PROPOSALS = Path("scripts/benchmarks/routing/review/proposed_verdicts.json")
DEFAULT_OUTPUT = Path("scripts/benchmarks/routing/review/tool_composer_corrections.md")
EMPIRICAL_MANIFEST = Path("scripts/benchmarks/routing/review/empirical_manifest.json")
EMPIRICAL_RESULTS = Path("scripts/benchmarks/routing/review/empirical_results")


def load_proposals() -> Dict[str, Dict[str, str]]:
    """Contract-based proposed verdicts (hypotheses, keyed by normalized text)."""
    if not PROPOSALS.exists():
        return {}
    return json.loads(PROPOSALS.read_text()).get("proposals", {})


def load_empirical() -> Dict[str, Dict[str, Any]]:
    """Response evidence from the empirical pass (empirical_pass.py + the live
    AG-UI run), keyed by normalized query text. Empty when the pass hasn't run."""
    if not EMPIRICAL_MANIFEST.exists():
        return {}
    live_by_qid: Dict[str, Dict[str, Any]] = {}
    live_path = EMPIRICAL_RESULTS / "raw_empirical22.jsonl"
    if live_path.exists():
        for line in live_path.read_text().splitlines():
            r = json.loads(line)
            live_by_qid[r["question_id"]] = r
    evidence: Dict[str, Dict[str, Any]] = {}
    for e in json.loads(EMPIRICAL_MANIFEST.read_text())["entries"]:
        forced_path = EMPIRICAL_RESULTS / f"forced_{e['qid']}_{e['candidate_agent']}.json"
        evidence[normalize(e["query_text"])] = {
            "qid": e["qid"],
            "candidate_agent": e["candidate_agent"],
            "forced": json.loads(forced_path.read_text()) if forced_path.exists() else None,
            "live": live_by_qid.get(e["qid"]),
        }
    return evidence


def _excerpt(text: Any, limit: int = 450) -> str:
    flat = " ".join(str(text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def render_empirical(ev: Dict[str, Any]) -> List[str]:
    lines = [
        f"- empirical evidence ({ev['qid']}, 2026-07-29 pass; full records in"
        " review/empirical_results/):"
    ]
    forced = ev.get("forced")
    if forced:
        results = forced.get("agent_results") or []
        succeeded = [r for r in results if r.get("success")]
        if forced.get("error"):
            detail = f"HARNESS ERROR: {_excerpt(forced['error'], 200)}"
        elif succeeded:
            detail = (
                f"SUCCEEDED ({forced.get('total_ms')}ms);"
                f" response: “{_excerpt(forced.get('synthesized_response'))}”"
            )
        elif results:
            errs = "; ".join(
                f"{r.get('agent_name')}: {_excerpt(r.get('error'), 220)}" for r in results
            )
            detail = f"FAILED CLOSED — {errs}"
        else:
            detail = "no agent results recorded"
        lines.append(
            f"  - forced route `{ev['candidate_agent']}`"
            f" (intent `{forced.get('forced_intent')}`): {detail}"
        )
    else:
        lines.append(f"  - forced route `{ev['candidate_agent']}`: (not yet run)")
    live = ev.get("live")
    if live:
        if live.get("error"):
            detail = f"ERROR: {_excerpt(live['error'], 200)}"
        else:
            tools = ",".join(live.get("tools_invoked") or []) or "none"
            detail = (
                f"tools=[{tools}] ({live.get('total_ms')}ms,"
                f" {len(live.get('response_text') or '')} chars);"
                f" response: “{_excerpt(live.get('response_text'))}”"
            )
        lines.append(f"  - live AG-UI response (real UI brain): {detail}")
    else:
        lines.append("  - live AG-UI response: (not yet run)")
    return lines


def load_registry() -> Dict[str, Any]:
    return json.loads(CONTRACTS.read_text())


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


async def fetch_corrections(correct_pattern: str) -> List[Dict[str, Any]]:
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    result = await (
        client.table("classification_logs")
        .select(
            "classification_id,query_text,routing_pattern,target_agents,"
            "confidence,feedback_notes,created_at"
        )
        .eq("was_correct", False)
        .eq("correct_pattern", correct_pattern)
        .order("created_at", desc=False)
        .limit(1000)
        .execute()
    )
    return list(result.data or [])


def demo_expectations() -> Dict[str, Dict[str, Any]]:
    doc = json.loads(DEMO_QUESTIONS.read_text())
    return {normalize(q["text"]): q for q in doc["questions"]}


def group_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = normalize(r["query_text"])
        notes = r.get("feedback_notes")
        try:
            verdict = json.loads(notes) if notes else {}
        except (TypeError, ValueError):
            verdict = {}
        g = groups.setdefault(
            key,
            {
                "query_text": r["query_text"],
                "pipeline_decisions": {},
                "target_agents": set(),
                "n_rows": 0,
                "judge_confidence": verdict.get("confidence"),
                "judge_rationale": verdict.get("rationale", ""),
            },
        )
        g["n_rows"] += 1
        g["target_agents"].update(r.get("target_agents") or [])
        pattern = r.get("routing_pattern")
        g["pipeline_decisions"][pattern] = g["pipeline_decisions"].get(pattern, 0) + 1
        conf = verdict.get("confidence")
        if conf is not None and (g["judge_confidence"] is None or conf > g["judge_confidence"]):
            g["judge_confidence"] = conf
            g["judge_rationale"] = verdict.get("rationale", "")
    return sorted(groups.values(), key=lambda g: -g["n_rows"])


def agents_in_text(text: str, agent_names: List[str]) -> List[str]:
    """Agent names mentioned in free text (demo expected_path etc.)."""
    return [a for a in agent_names if a in text]


def render_ruling(registry: Dict[str, Any], correct_pattern: str) -> List[str]:
    ruling = registry["composition_ruling"]
    lines = [
        "## The platform's own composition ruling (verified from code, 2026-07-29)",
        "",
        ruling["answer"],
        "",
        "Key evidence:",
        "",
    ]
    lines += [f"- {e}" for e in ruling["evidence"][:4]]
    lines += [
        "",
        "Ambiguity resolutions (settled 2026-07-29 with code/git/measured evidence"
        " — review under these rules; measurement in"
        " empirical_results/classifier_measurement_22.json):",
        "",
    ]
    lines += [f"- {a}" for a in ruling["ambiguities"]]
    lines += [
        "",
        "Boundary notes:",
        "",
    ]
    lines += [f"- {n}" for n in registry["_meta"]["boundary_notes"]]
    return lines


def render(
    groups: List[Dict[str, Any]],
    correct_pattern: str,
    registry: Dict[str, Any],
    proposals: Dict[str, Dict[str, str]],
    empirical: Dict[str, Dict[str, Any]],
) -> str:
    demo = demo_expectations()
    agents: Dict[str, Any] = registry["agents"]
    agent_names = list(agents)
    lines = [
        f"# Judge corrections to {correct_pattern} — human review worksheet",
        "",
        f"{sum(g['n_rows'] for g in groups)} labeled rows, {len(groups)} distinct queries.",
        "",
        "**The verdict question is contract coverage, not step-counting.** Per the"
        " platform ruling below, a query is TOOL_COMPOSER only if it spans >= 2"
        " distinct agent contracts AND the cross-agent parts depend on each other."
        " A single agent's internal multi-step pipeline is one SINGLE_AGENT"
        " dispatch. Fill **Verdict** per query with one of:",
        "",
        f"- `agree` — the ask genuinely spans multiple agent contracts with"
        f" dependencies; {correct_pattern} stands.",
        "- `single_agent:<name>` — that agent's contract covers the ask end-to-end;"
        " judge label is wrong.",
        "- `extend:<name> — <what to add>` — product intent is single-agent"
        " but the named agent's contract needs extending (design input, not"
        " just a label).",
        "- `clarify` — the query is genuinely ambiguous; CLARIFICATION_NEEDED was right after all.",
        "",
        "Verdicts feed the #1341 judge-prompt fix and the #1337 Step 0 gold"
        " protocol; `extend:` verdicts become contract-change work items.",
        "",
        "**Proposed verdicts**: each query below carries a contract-based proposed"
        " verdict with reasoning (from review/proposed_verdicts.json). These are"
        " HYPOTHESES derived from the verified contracts only — not validated"
        " against actual agent responses. The planned empirical pass (execute each"
        " disputed query, capture the real response per candidate route) is the"
        " cheapest disproof; overriding a proposal needs no justification, and"
        " confirming one is best done with response evidence in hand. To accept a"
        " proposal, write `accept` in the Verdict slot; otherwise write your own.",
        "",
    ]
    lines += render_ruling(registry, correct_pattern)
    lines += [
        "",
        "## Agent contract reference (verified registry: data/agent_contracts.json)",
        "",
    ]
    for name, c in agents.items():
        lines.append(f"- **{name}**: {c['purpose']}")
        lines.append(f"  - covers: {'; '.join(c['covers'])}")
        lines.append(f"  - not: {'; '.join(c['does_not_cover'])}")
    lines.append("")
    for i, g in enumerate(groups, 1):
        exp = demo.get(normalize(g["query_text"]))
        decisions = ", ".join(f"{k}×{v}" for k, v in sorted(g["pipeline_decisions"].items()))
        lines += [
            f"## {i}. {g['query_text'].strip()}",
            "",
            f"- rows: {g['n_rows']}  |  pipeline said: {decisions}"
            f"  |  judge: {correct_pattern} @ {g['judge_confidence']}",
            f"- judge rationale: {g['judge_rationale']}",
        ]
        candidates = sorted(g["target_agents"])
        if exp:
            lines.append(
                f"- demo doc expectation: intent `{exp.get('intent_expected')}`"
                f" → {exp.get('expected_path')} (question {exp.get('question_id')})"
            )
            candidates = sorted(
                set(candidates) | set(agents_in_text(exp.get("expected_path") or "", agent_names))
            )
        else:
            lines.append("- demo doc expectation: — (not one of the 51 authored questions)")
        if candidates:
            lines.append("- contract check — candidate single agent(s):")
            for a in candidates:
                c = agents.get(a)
                if c is None:
                    lines.append(f"  - `{a}`: (not in the verified registry — unknown agent)")
                else:
                    lines.append(f"  - `{a}`: {c['purpose']}")
        else:
            lines.append(
                "- contract check: no candidate agent on record (pipeline"
                " dispatched none; pick from the reference above)"
            )
        prop = proposals.get(normalize(g["query_text"]))
        if prop:
            validated = "empirical" in prop
            lines += [
                f"- proposed verdict (contract-based"
                f"{'' if validated else ' hypothesis, unvalidated'}):"
                f" `{prop['proposed']}`",
                f"  - reasoning: {prop['reasoning']}",
            ]
            if validated:
                lines.append(f"  - empirical status: {prop['empirical']}")
        ev = empirical.get(normalize(g["query_text"]))
        if ev:
            lines += render_empirical(ev)
        lines += ["- **Verdict**: _____", ""]
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct-pattern", default="TOOL_COMPOSER")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    load_dotenv()
    registry = load_registry()
    rows = await fetch_corrections(args.correct_pattern)
    groups = group_rows(rows)
    output = args.output or (
        DEFAULT_OUTPUT
        if args.correct_pattern == "TOOL_COMPOSER"
        else DEFAULT_OUTPUT.with_name(f"{args.correct_pattern.lower()}_corrections.md")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render(groups, args.correct_pattern, registry, load_proposals(), load_empirical())
    )
    print(f"{len(rows)} rows -> {len(groups)} distinct queries -> {output}")


if __name__ == "__main__":
    asyncio.run(main())
