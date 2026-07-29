#!/usr/bin/env python3
"""#1337 Step 0 — gold-judge stage (axis 1: routing gold over all 337 queries).

Labels every row of ``data/benchmark_queries.jsonl`` with a gold routing
pattern + owning agent(s), keyed by a precedence cascade:

1. **human-ratified** — the row's normalized text matches one of the 22
   verdicts in ``review/proposed_verdicts.json`` (recorded in the 2026-07-29
   interactive review). The human ``final`` is carried directly, no LLM call.
2. **authored-ground-truth** — ``source == 'authored'`` rows carry
   ``authored_gold_pattern`` / ``authored_gold_agents`` (the benchmark
   author's designed probe-cell labels). Those become gold directly; the LLM
   judge still labels them so we can measure judge-vs-author agreement.
3. **llm-judge** — everything else (historical / perturbation / non-anchor
   demo) is labeled by a strong Anthropic model against the verified contract
   registry (``data/agent_contracts.json``, the SSOT) plus human few-shot
   anchors and the normative composition rules.

Normative rules (override any judge instinct):
  * Contract covers/does_not_cover OWNERSHIP decides the label. Classifier
    reach, resolver operability, dispatch budgets, and known crashes never
    constrain gold.
  * TOOL_COMPOSER gate: the query must span >=2 DISTINCT agent-capability
    domains AND the cross-domain sub-questions must be dependency-linked.
    Step-counting is WRONG — a single-domain multi-step query is SINGLE_AGENT.
  * Follow-up-shaped rows are labeled in-context (the prior turn travels with
    the row).

Judge confidence < 0.6 is still recorded (label + confidence) but the row is
appended to ``review/gold_low_confidence.md`` for human follow-up.

The run is idempotent/resumable: each completed judge verdict is checkpointed
to ``review/empirical_results/gold_progress.jsonl`` keyed by query_id, so a
crash never re-bills a completed row.

Usage (repo root, droplet)::

    .venv/bin/python scripts/benchmarks/routing/gold_judge.py
    .venv/bin/python scripts/benchmarks/routing/gold_judge.py --limit 5   # smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
BENCH = HERE / "data" / "benchmark_queries.jsonl"
CONTRACTS = HERE / "data" / "agent_contracts.json"
PROPOSALS = HERE / "review" / "proposed_verdicts.json"
MANIFEST = HERE / "review" / "empirical_manifest.json"
OUT_GOLD = HERE / "data" / "benchmark_queries_gold.jsonl"
OUT_LOWCONF = HERE / "review" / "gold_low_confidence.md"
OUT_SUMMARY = HERE / "review" / "gold_axis1_summary.json"
PROGRESS = HERE / "review" / "empirical_results" / "gold_progress.jsonl"

VALID_PATTERNS = {"SINGLE_AGENT", "PARALLEL_DELEGATION", "TOOL_COMPOSER", "CLARIFICATION_NEEDED"}
GOLD_CONFIDENCE_FLOOR = 0.6
JUDGE_MODEL = os.getenv("GOLD_JUDGE_MODEL", "claude-sonnet-5")
CONCURRENCY = int(os.getenv("GOLD_JUDGE_CONCURRENCY", "5"))
MAX_RETRIES = 6


# ---------------------------------------------------------------------------
# keying + loaders
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())


def load_rows() -> List[Dict[str, Any]]:
    return [json.loads(line) for line in BENCH.read_text().splitlines() if line.strip()]


def load_registry() -> Dict[str, Any]:
    return json.loads(CONTRACTS.read_text())


def parse_final(final: str) -> Tuple[str, List[str], Optional[str]]:
    """Parse a human ``final`` verdict string into (pattern, agents, extend_note).

    The verdict is the last ``->`` target; the trailing parenthetical (which
    can name a *declined* alternative, e.g. q20's 'extend:prediction_synthesizer
    ... considered and declined') is dropped before classification.
    """
    seg = final.split("->")[-1].split("(")[0]
    m = re.search(r"single_agent:([a-z_]+)", seg)
    if m:
        return "SINGLE_AGENT", [m.group(1)], None
    m = re.search(r"extend:([a-z_]+)", seg)
    if m:
        return "SINGLE_AGENT", [m.group(1)], f"extend:{m.group(1)}"
    if "TOOL_COMPOSER" in seg:
        return "TOOL_COMPOSER", ["tool_composer"], None
    if "PARALLEL" in seg.upper():
        return "PARALLEL_DELEGATION", [], None
    if "clarif" in seg.lower():
        return "CLARIFICATION_NEEDED", [], None
    raise ValueError(f"unparseable human final: {final!r}")


def load_human_anchors() -> Dict[str, Dict[str, Any]]:
    """normalized query text -> {pattern, agents, extend, final} (22 human verdicts)."""
    proposals = json.loads(PROPOSALS.read_text())["proposals"]
    manifest = {e["qid"]: e for e in json.loads(MANIFEST.read_text())["entries"]}
    n2qid = {normalize(e["query_text"]): qid for qid, e in manifest.items()}
    anchors: Dict[str, Dict[str, Any]] = {}
    for q, info in proposals.items():
        pattern, agents, extend = parse_final(info["final"])
        anchors[normalize(q)] = {
            "qid": n2qid.get(normalize(q)),
            "pattern": pattern,
            "agents": agents,
            "extend": extend,
            "final": info["final"],
        }
    return anchors


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------
def contract_digest(registry: Dict[str, Any]) -> str:
    lines = []
    for name, c in registry["agents"].items():
        covers = "; ".join(c["covers"])
        does_not = "; ".join(c["does_not_cover"])
        lines.append(f"- {name}: {c['purpose']}\n    COVERS: {covers}\n    NOT: {does_not}")
    ruling = registry["composition_ruling"]["answer"]
    return "AGENT CONTRACT REGISTRY (single source of truth):\n" + "\n".join(lines) + (
        f"\n\nCOMPOSITION RULING (verified from platform code):\n{ruling}"
    )


# few-shot anchors: drawn ONLY from the 22 human-ratified finals (2026-07-29),
# chosen to cover the label space and the two subtle rulings (q02 breadth, q15 extend).
FEWSHOT = [
    {
        "q": "What is the causal impact of rep visits on TRx for Kisqali?",
        "gold_pattern": "SINGLE_AGENT",
        "gold_agents": ["causal_impact"],
        "why": "Textbook ATE ask — first entry in causal_impact's covers. Sales-vs-clinical data sources are NOT separate agent-capability domains; one domain.",
    },
    {
        "q": "Design an experiment to measure whether speaker programs increase Fabhalta NRx",
        "gold_pattern": "SINGLE_AGENT",
        "gold_agents": ["experiment_designer"],
        "why": "End-to-end experiment design (metrics, controls, power, validity) is experiment_designer's single-dispatch contract; the internal phases are not separate domains.",
    },
    {
        "q": "Show me the KPI summary for Kisqali",
        "gold_pattern": "SINGLE_AGENT",
        "gold_agents": ["explainer"],
        "why": "A KPI lookup/summary — no second domain, no dependency. Catch-all narration owner is explainer; 'clinical/safety/market' facets are not in the query.",
    },
    {
        "q": "Build a patient cohort for Remibrutinib CSU with inclusion criteria for adults over 18 diagnosed in 2024",
        "gold_pattern": "SINGLE_AGENT",
        "gold_agents": ["cohort_profiler"],
        "why": "One domain (COHORT_DEFINITION). cohort_profiler owns the chat cohort ask; brand/indication/temporal criteria are filters inside that one dispatch, not extra domains.",
    },
    {
        "q": "How can I optimize resource allocation for Remibrutinib in the northeast region?",
        "gold_pattern": "SINGLE_AGENT",
        "gold_agents": ["resource_optimizer"],
        "why": "Verbatim resource_optimizer territory (allocate reps/budget across a region). Region is a filter; one domain. (Classifier-unmapped ≠ not-gold.)",
    },
    {
        "q": "Build a cohort of high-value HCPs who prescribed more than 50 TRx last quarter",
        "gold_pattern": "SINGLE_AGENT",
        "gold_agents": ["cohort_profiler"],
        "why": "Still single-domain (COHORT_DEFINITION), so never TOOL_COMPOSER. cohort_profiler owns it but its contract covers PATIENT populations, not HCP-by-TRx cohorts — extend:cohort_profiler (contract-gap; label stays SINGLE_AGENT/cohort_profiler).",
    },
    {
        "q": "Where are the biggest untapped opportunities to grow Remibrutinib market share?",
        "gold_pattern": "TOOL_COMPOSER",
        "gold_agents": ["tool_composer"],
        "why": "'Untapped opportunities to grow market share' is broader than pure gap ROI: it depends on competitive-landscape + clinical/regulatory context (no single agent covers those), so it spans >=2 domains with a dependency. (A plain 'biggest ROI opportunities for <brand>' would instead be SINGLE_AGENT/gap_analyzer.)",
    },
    {
        "q": "If conversion rate in the west is below 15%, which patient segments should we prioritize?",
        "gold_pattern": "TOOL_COMPOSER",
        "gold_agents": ["tool_composer"],
        "why": "Conditional composition: a KPI/gap check GATES a segment-prioritization step — two distinct domains with a hard dependency.",
    },
    {
        "q": "Forecast Kisqali TRx volume for the next two quarters and tell me the biggest risk to that forecast",
        "gold_pattern": "TOOL_COMPOSER",
        "gold_agents": ["tool_composer"],
        "why": "Prediction + risk-to-that-forecast (drift/causal context), with the risk facet depending on the forecast output — >=2 domains, dependency-linked.",
    },
    {
        "q": ("Our Kisqali TRx dropped in the northeast last quarter while conversion rates for "
              "Remibrutinib stayed flat, and I need to understand several things: what actually "
              "caused the Kisqali decline, whether biologic-experienced segments were "
              "disproportionately affected, what the models predict for both brands next quarter, "
              "whether any data drift could be confounding these reads, and what experiment we "
              "should run to test whether adding rep capacity would recover the trend."),
        "gold_pattern": "TOOL_COMPOSER",
        "gold_agents": ["tool_composer"],
        "why": "Five explicit facets across >=4 domains (causal, heterogeneity, prediction, drift, experiment design) with stated dependencies (drift confounds the reads; the experiment depends on the causal finding).",
    },
]


def build_prompt(registry_digest: str, row: Dict[str, Any], parent_hint: Optional[str]) -> str:
    fewshot_lines = []
    for ex in FEWSHOT:
        fewshot_lines.append(
            f'Query: "{ex["q"]}"\n  -> {{"gold_pattern": "{ex["gold_pattern"]}", '
            f'"gold_agents": {json.dumps(ex["gold_agents"])}}}  ({ex["why"]})'
        )
    fewshot = "\n".join(fewshot_lines)

    ctx_lines = []
    if row.get("is_followup") and row.get("context"):
        prev_user = (row["context"] or {}).get("prev_user")
        prev_asst = (row["context"] or {}).get("prev_assistant")
        if prev_user:
            ctx_lines.append(f'  FOLLOW-UP CONTEXT — previous user turn: "{prev_user}"')
        if prev_asst:
            ctx_lines.append(f'  FOLLOW-UP CONTEXT — previous assistant turn: "{str(prev_asst)[:300]}"')
    if row.get("perturbation_type"):
        note = (f"  PERTURBATION: this is a '{row['perturbation_type']}' variant of a parent "
                f"query. Label the text AS WRITTEN, in-context.")
        if parent_hint:
            note += (f" Parent query's gold was {parent_hint} — confirm if the variant preserves "
                     f"the intent, or override if the perturbation changed/obscured it "
                     f"(e.g. a fragment that lost the brand/metric may be CLARIFICATION_NEEDED).")
        ctx_lines.append(note)
    context_block = ("\n" + "\n".join(ctx_lines)) if ctx_lines else ""

    return f"""You are the GOLD ROUTING JUDGE for a pharmaceutical causal-analytics chatbot's routing benchmark (#1337 Step 0). Assign the ONE correct route (the "gold" label) for the user query, judged by CONTRACT OWNERSHIP — not by what any classifier can currently reach, nor by whether a route currently works.

ROUTING PATTERNS (choose exactly one):
- SINGLE_AGENT: exactly one agent-capability domain owns the ask (even if that agent runs many internal steps). gold_agents = [that one agent].
- PARALLEL_DELEGATION: the ask spans >=2 DISTINCT agent-capability domains that are INDEPENDENT (no cross-domain dependency). gold_agents = the independent agents.
- TOOL_COMPOSER: the ask spans >=2 DISTINCT agent-capability domains AND the cross-domain sub-questions are DEPENDENCY-LINKED. gold_agents = ["tool_composer"].
- CLARIFICATION_NEEDED: a reasonable pharma analyst could NOT infer a routable intent (genuinely ambiguous, or a fragment with no brand/metric/domain). gold_agents = [].

HARD RULES (override any instinct):
1. Contract covers/does_not_cover ownership decides the label. Classifier reach, resolver operability, dispatch budgets, and known crashes NEVER constrain gold.
2. TOOL_COMPOSER GATE: >=2 DISTINCT capability domains AND dependency-linked. STEP-COUNTING IS WRONG — a single-domain multi-step query is SINGLE_AGENT no matter how many internal steps the owning agent runs. Data sources (sales vs clinical CRM) are NOT capability domains.
3. A capability that sits entirely inside ONE agent's COVERS list is single-domain. The ask spans domains only when a needed capability appears in that agent's NOT list with a handoff to another agent.
4. Follow-up queries: label in-context (the provided previous turn travels with the query).
5. gold_agents must be chosen from these 14 agents ONLY: cohort_constructor, cohort_profiler, tool_composer, causal_impact, gap_analyzer, heterogeneous_optimizer, experiment_designer, experiment_monitor, drift_monitor, health_score, prediction_synthesizer, resource_optimizer, explainer, feedback_learner. (cohort_constructor is pipeline-only; chat cohort asks are owned by cohort_profiler.)

{registry_digest}

FEW-SHOT CALIBRATION ANCHORS (human-ratified 2026-07-29 — authoritative):
{fewshot}

NOW LABEL THIS QUERY:
Query: "{row['text']}"{context_block}

Respond with ONLY a JSON object (no prose, no code fence):
{{"gold_pattern": "SINGLE_AGENT"|"PARALLEL_DELEGATION"|"TOOL_COMPOSER"|"CLARIFICATION_NEEDED", "gold_agents": ["..."], "gold_confidence": 0.0-1.0, "gold_rationale": "<= 40 words citing the owning contract"}}
If genuinely unsure between two routes, pick the better one but LOWER the confidence (< 0.6)."""


VALID_AGENTS = {
    "cohort_constructor", "cohort_profiler", "tool_composer", "causal_impact", "gap_analyzer",
    "heterogeneous_optimizer", "experiment_designer", "experiment_monitor", "drift_monitor",
    "health_score", "prediction_synthesizer", "resource_optimizer", "explainer", "feedback_learner",
}


def extract_text(resp: Any) -> str:
    """Concatenate all text blocks (sonnet-5 may emit a ThinkingBlock first)."""
    parts = []
    for block in getattr(resp, "content", None) or []:
        if getattr(block, "type", None) == "text" or hasattr(block, "text"):
            parts.append(getattr(block, "text", "") or "")
    return "\n".join(p for p in parts if p)


def parse_verdict(text: str) -> Optional[Dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    # tolerate leading/trailing prose by grabbing the first {...} block
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(0)
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    pattern = payload.get("gold_pattern")
    if pattern not in VALID_PATTERNS:
        return None
    agents = payload.get("gold_agents") or []
    if not isinstance(agents, list):
        agents = []
    agents = [a for a in agents if a in VALID_AGENTS]
    if pattern == "TOOL_COMPOSER":
        agents = ["tool_composer"]
    elif pattern == "CLARIFICATION_NEEDED":
        agents = []
    elif pattern == "SINGLE_AGENT" and len(agents) != 1:
        # keep the first valid agent if the judge over-listed; drop to None if empty
        agents = agents[:1]
    try:
        conf = float(payload.get("gold_confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "gold_pattern": pattern,
        "gold_agents": agents,
        "gold_confidence": min(1.0, max(0.0, conf)),
        "gold_rationale": str(payload.get("gold_rationale") or "")[:400],
    }


# ---------------------------------------------------------------------------
# async judge with retries + checkpointing
# ---------------------------------------------------------------------------
async def judge_one(client, sem, digest, row, parent_hint, lock) -> Tuple[str, Optional[Dict]]:
    import anthropic

    qid = row["query_id"]
    prompt = build_prompt(digest, row, parent_hint)
    retryable_status = {429, 500, 502, 503, 529}
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=2000,  # room for sonnet-5 thinking + the small JSON verdict
                    messages=[{"role": "user", "content": prompt}],
                )
                verdict = parse_verdict(extract_text(resp))
                usage = getattr(resp, "usage", None)
                result = {
                    "query_id": qid,
                    "verdict": verdict,
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "model": JUDGE_MODEL,
                }
                async with lock:
                    with PROGRESS.open("a") as fh:
                        fh.write(json.dumps(result) + "\n")
                return qid, result
            except (anthropic.RateLimitError, anthropic.APIConnectionError,
                    anthropic.APITimeoutError) as e:
                wait = min(2 ** attempt, 30)
                print(f"  [{qid}] retry {attempt + 1}/{MAX_RETRIES} after {type(e).__name__}: "
                      f"waiting {wait}s", flush=True)
                await asyncio.sleep(wait)
            except anthropic.APIStatusError as e:
                if getattr(e, "status_code", None) in retryable_status:
                    wait = min(2 ** attempt, 30)
                    print(f"  [{qid}] retry {attempt + 1}/{MAX_RETRIES} after "
                          f"{type(e).__name__} {e.status_code}: waiting {wait}s", flush=True)
                    await asyncio.sleep(wait)
                    continue
                print(f"  [{qid}] FATAL non-retryable {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
                return qid, None
            except Exception as e:  # noqa: BLE001 — one bad row must not abort the gather
                print(f"  [{qid}] FATAL {type(e).__name__}: {str(e)[:200]}", flush=True)
                return qid, None
        print(f"  [{qid}] FAILED after {MAX_RETRIES} retries", flush=True)
        return qid, None


def load_progress() -> Dict[str, Dict[str, Any]]:
    if not PROGRESS.exists():
        return {}
    done: Dict[str, Dict[str, Any]] = {}
    for line in PROGRESS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("verdict") is not None:
            done[rec["query_id"]] = rec
    return done


async def run_judge(rows_to_judge: List[Tuple[Dict[str, Any], Optional[str]]]) -> None:
    if not rows_to_judge:
        return
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set — cannot run gold judge")
    digest = contract_digest(load_registry())
    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    tasks = [judge_one(client, sem, digest, row, hint, lock) for row, hint in rows_to_judge]
    total = len(tasks)
    done_n = 0
    for coro in asyncio.as_completed(tasks):
        qid, _ = await coro
        done_n += 1
        if done_n % 20 == 0 or done_n == total:
            print(f"  judged {done_n}/{total}", flush=True)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def assemble(rows: List[Dict[str, Any]], anchors: Dict[str, Dict[str, Any]],
             progress: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    low_conf: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []  # gold vs nightly TOOL_COMPOSER (on the 22 anchors)
    authored_disagree: List[Dict[str, Any]] = []  # judge vs authored gold

    for row in rows:
        r = dict(row)
        nkey = normalize(row["text"])
        anchor = anchors.get(nkey)
        prog = progress.get(row["query_id"])
        judge_v = prog["verdict"] if prog else None

        # record the independent judge verdict wherever we have one (cross-check field)
        if judge_v:
            r["judge_pattern"] = judge_v["gold_pattern"]
            r["judge_agents"] = judge_v["gold_agents"]
            r["judge_confidence"] = judge_v["gold_confidence"]

        if anchor:
            r["gold_pattern"] = anchor["pattern"]
            r["gold_agents"] = anchor["agents"]
            r["gold_confidence"] = 1.0
            rationale = "human-ratified 2026-07-29"
            if anchor["extend"]:
                rationale += f" ({anchor['extend']} — contract-gap noted; label stays SINGLE_AGENT)"
            r["gold_rationale"] = rationale
            r["gold_source"] = "human-ratified"
            # the nightly judge corrected all 22 disputed queries to TOOL_COMPOSER
            r["nightly_pattern"] = "TOOL_COMPOSER"
            if anchor["pattern"] != "TOOL_COMPOSER":
                disagreements.append({
                    "query_id": row["query_id"], "qid": anchor["qid"], "text": row["text"],
                    "gold_pattern": anchor["pattern"], "gold_agents": anchor["agents"],
                    "nightly_pattern": "TOOL_COMPOSER",
                })
        elif row.get("source") == "authored" and row.get("authored_gold_pattern"):
            r["gold_pattern"] = row["authored_gold_pattern"]
            r["gold_agents"] = row.get("authored_gold_agents") or []
            r["gold_confidence"] = 1.0
            r["gold_rationale"] = f"authored ground-truth (cell {row.get('cell')})"
            r["gold_source"] = "authored-ground-truth"
            if judge_v and judge_v["gold_pattern"] != row["authored_gold_pattern"]:
                authored_disagree.append({
                    "query_id": row["query_id"], "text": row["text"],
                    "authored": row["authored_gold_pattern"],
                    "authored_agents": row.get("authored_gold_agents"),
                    "judge": judge_v["gold_pattern"], "judge_agents": judge_v["gold_agents"],
                    "judge_conf": judge_v["gold_confidence"],
                })
        elif judge_v:
            r["gold_pattern"] = judge_v["gold_pattern"]
            r["gold_agents"] = judge_v["gold_agents"]
            r["gold_confidence"] = judge_v["gold_confidence"]
            r["gold_rationale"] = judge_v["gold_rationale"]
            r["gold_source"] = "llm-judge"
            if judge_v["gold_confidence"] < GOLD_CONFIDENCE_FLOOR:
                low_conf.append(r)
        else:
            r["gold_pattern"] = None
            r["gold_agents"] = None
            r["gold_confidence"] = None
            r["gold_rationale"] = "UNJUDGED (judge did not complete this row)"
            r["gold_source"] = "unjudged"
        out_rows.append(r)

    summary = _summarize(out_rows, low_conf, disagreements, authored_disagree, progress)
    _write_lowconf(low_conf)
    return out_rows, summary


def _summarize(out_rows, low_conf, disagreements, authored_disagree, progress) -> Dict[str, Any]:
    by_source_pattern: Dict[str, Counter] = {}
    overall = Counter()
    source_counts = Counter()
    src_gold_conf: Dict[str, Counter] = {}
    for r in out_rows:
        src = r["source"]
        source_counts[src] += 1
        by_source_pattern.setdefault(src, Counter())[r["gold_pattern"]] += 1
        overall[r["gold_pattern"]] += 1
        src_gold_conf.setdefault(r["gold_source"], Counter())["n"] += 1
    # judge cost
    in_tok = sum((p.get("input_tokens") or 0) for p in progress.values())
    out_tok = sum((p.get("output_tokens") or 0) for p in progress.values())
    n_calls = len(progress)
    # sonnet-5 pricing assumption (USD / 1M tokens): input 3, output 15
    est_cost = in_tok * 3 / 1e6 + out_tok * 15 / 1e6
    return {
        "total_rows": len(out_rows),
        "overall_label_distribution": dict(overall),
        "by_source": {s: dict(by_source_pattern[s]) for s in by_source_pattern},
        "source_counts": dict(source_counts),
        "gold_source_counts": {k: v["n"] for k, v in src_gold_conf.items()},
        "low_confidence_count": len(low_conf),
        "human_anchor_matches": sum(1 for r in out_rows if r["gold_source"] == "human-ratified"),
        "authored_ground_truth": sum(1 for r in out_rows if r["gold_source"] == "authored-ground-truth"),
        "llm_judged_gold": sum(1 for r in out_rows if r["gold_source"] == "llm-judge"),
        "unjudged": sum(1 for r in out_rows if r["gold_source"] == "unjudged"),
        "nightly_disagreements": disagreements,
        "authored_vs_judge_disagreements": authored_disagree,
        "judge_calls": n_calls,
        "judge_input_tokens": in_tok,
        "judge_output_tokens": out_tok,
        "judge_est_cost_usd": round(est_cost, 2),
        "judge_model": JUDGE_MODEL,
    }


def _write_lowconf(low_conf: List[Dict[str, Any]]) -> None:
    lines = [
        "# Gold-stage low-confidence rows (judge confidence < 0.6) — human follow-up",
        "",
        f"{len(low_conf)} rows. The judge recorded its best label + confidence for each; these",
        "did not clear the 0.6 floor and want a human ruling before they anchor anything.",
        "",
    ]
    for r in sorted(low_conf, key=lambda x: x.get("gold_confidence") or 0.0):
        ctx = ""
        if r.get("is_followup") and r.get("context"):
            ctx = f" [follow-up of: \"{(r['context'] or {}).get('prev_user')}\"]"
        lines.append(
            f"- **{r['query_id']}** ({r['source']}"
            f"{'/' + r['perturbation_type'] if r.get('perturbation_type') else ''}"
            f", conf {r.get('gold_confidence')}): \"{r['text']}\"{ctx}\n"
            f"  - judge: {r['gold_pattern']} / {r['gold_agents']} — {r.get('gold_rationale')}"
        )
    OUT_LOWCONF.write_text("\n".join(lines) + "\n")


def write_outputs(out_rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    with OUT_GOLD.open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="only judge the first N judge-eligible rows (smoke test)")
    parser.add_argument("--assemble-only", action="store_true",
                        help="skip judging; rebuild gold outputs from existing progress")
    args = parser.parse_args()

    load_dotenv()
    rows = load_rows()
    anchors = load_human_anchors()

    # pool_id -> provisional parent gold hint (from authored gold or human anchor)
    parent_gold: Dict[str, str] = {}
    by_pool: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = r.get("source_query_id")
        if sid:
            by_pool.setdefault(sid, []).append(r)
    for r in rows:
        sid = r.get("source_query_id")
        if r.get("source") == "authored" and r.get("authored_gold_pattern") and sid:
            parent_gold[sid] = f"{r['authored_gold_pattern']}/{r.get('authored_gold_agents')}"
    for r in rows:
        if normalize(r["text"]) in anchors and r.get("source_query_id"):
            a = anchors[normalize(r["text"])]
            parent_gold[r["source_query_id"]] = f"{a['pattern']}/{a['agents']}"

    progress = load_progress()

    # decide which rows need the LLM judge (everything except human anchors)
    to_judge: List[Tuple[Dict[str, Any], Optional[str]]] = []
    for r in rows:
        if normalize(r["text"]) in anchors:
            continue  # human anchor — no LLM
        if r["query_id"] in progress:
            continue  # already judged (resume)
        hint = parent_gold.get(r.get("parent_query_id") or "")
        to_judge.append((r, hint))

    if args.limit is not None:
        to_judge = to_judge[: args.limit]

    if not args.assemble_only:
        print(f"gold judge: {len(rows)} rows total | {len(anchors)} human anchors | "
              f"{len(progress)} already judged | {len(to_judge)} to judge now "
              f"(model={JUDGE_MODEL}, concurrency={CONCURRENCY})", flush=True)
        await run_judge(to_judge)
        progress = load_progress()

    out_rows, summary = assemble(rows, anchors, progress)
    write_outputs(out_rows, summary)
    print("\n=== AXIS-1 SUMMARY ===", flush=True)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("nightly_disagreements", "authored_vs_judge_disagreements")},
                     indent=2), flush=True)
    print(f"\nnightly disagreements: {len(summary['nightly_disagreements'])}"
          f" | authored-vs-judge disagreements: {len(summary['authored_vs_judge_disagreements'])}",
          flush=True)
    print(f"outputs: {OUT_GOLD.name}, {OUT_LOWCONF.name}, {OUT_SUMMARY.name}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
