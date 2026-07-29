"""Step 0 scoring run: routing-classifier candidates vs the 337-query gold (#1337).

Runs each candidate over data/benchmark_queries_gold.jsonl, scores predicted
(routing_pattern, target_agents) against gold, and writes:

- review/step0_scores/predictions_<candidate>.jsonl   (checkpoint; resumable)
- review/step0_scores/STEP0_SCORES.json               (all aggregates)
- review/step0_scores/STEP0_RESULTS.md                (readable results + decision readout)
- review/step0_scores/disagreements.md                (human-review worksheet)

Decision rule (#1337): if single_llm >= pipeline_llm on routing accuracy at
comparable latency/cost, the 4-stage design does not merit the async-LLM-stage
investment — replace rather than extend.

Usage:
    .venv/bin/python scripts/benchmarks/routing/score_candidates.py \
        [--candidates legacy,pipeline_rules,pipeline_llm,single_llm] \
        [--limit N] [--concurrency 6] [--no-resume]

Env: ANTHROPIC_API_KEY from .env (fail-fast if missing when an LLM candidate
is requested). ORCHESTRATOR_CLASSIFIER_MODE is forced to "off" in-process so
the legacy node's embedded shadow pipeline (and its classification_logs
writer) stays out of the measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Must be set before any src import triggers a lazy env read.
os.environ["ORCHESTRATOR_CLASSIFIER_MODE"] = "off"

from dotenv import load_dotenv  # noqa: E402

from scripts.benchmarks.routing.step0_scoring import (  # noqa: E402
    aggregate,
    disagreement_rows,
    score_row,
)

DATA = Path(__file__).parent / "data" / "benchmark_queries_gold.jsonl"
OUT_DIR = Path(__file__).parent / "review" / "step0_scores"
ALL_CANDIDATES = ["legacy", "pipeline_rules", "pipeline_llm", "single_llm"]


def load_rows(limit: int | None) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in DATA.open()]
    return rows[:limit] if limit else rows


def load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        for line in path.open():
            rec = json.loads(line)
            done[rec["query_id"]] = rec
    return done


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = min(len(vs) - 1, max(0, int(round(q * (len(vs) - 1)))))
    return vs[idx]


async def run_candidate(
    name: str,
    candidate: Any,
    rows: List[Dict[str, Any]],
    out_path: Path,
    concurrency: int,
    resume: bool,
) -> List[Dict[str, Any]]:
    done = load_checkpoint(out_path) if resume else {}
    todo = [r for r in rows if r["query_id"] not in done]
    print(f"[{name}] {len(done)} cached, {len(todo)} to run")

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    completed = 0

    async def one(row: Dict[str, Any]) -> None:
        nonlocal completed
        async with sem:
            try:
                pred = await candidate.predict(row)
            except Exception as e:  # candidate crash = real finding, not a skip
                pred = None
                err = f"{type(e).__name__}: {e}"
            rec = {
                "query_id": row["query_id"],
                "text": row["text"],
                "source": row.get("source"),
                "is_followup": bool(row.get("is_followup")),
                "gold_pattern": row["gold_pattern"],
                "gold_agents": row.get("gold_agents") or [],
                "gold_source": row.get("gold_source"),
                "gold_confidence": row.get("gold_confidence"),
            }
            if pred is None:
                rec.update(
                    pred_pattern="CANDIDATE_ERROR",
                    pred_agents=[],
                    pred_confidence=0.0,
                    latency_ms=0.0,
                    llm_used=False,
                    parse_failed=False,
                    error=err,
                )
            else:
                rec.update(
                    pred_pattern=pred.routing_pattern,
                    pred_agents=pred.target_agents,
                    pred_confidence=pred.confidence,
                    latency_ms=round(pred.latency_ms, 2),
                    llm_used=pred.llm_used,
                    parse_failed=pred.parse_failed,
                )
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                completed += 1
                if completed % 25 == 0:
                    print(f"[{name}] {completed}/{len(todo)}")

    await asyncio.gather(*(one(r) for r in todo))
    return list(load_checkpoint(out_path).values())


def scored(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for rec in records:
        out.append(
            {
                **rec,
                "score": score_row(
                    rec["gold_pattern"],
                    rec["gold_agents"],
                    rec["pred_pattern"],
                    rec["pred_agents"],
                ),
            }
        )
    return out


def summarize(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [r["latency_ms"] for r in rows]
    summary = {
        "candidate": name,
        **aggregate(rows),
        "by_source": aggregate(rows, slice_key="source")["slices"],
        "by_followup": aggregate(rows, slice_key="is_followup")["slices"],
        "by_gold_source": aggregate(rows, slice_key="gold_source")["slices"],
        "latency_ms_p50": round(percentile(latencies, 0.50), 1),
        "latency_ms_p95": round(percentile(latencies, 0.95), 1),
        "llm_share": sum(1 for r in rows if r["llm_used"]) / len(rows) if rows else 0.0,
        "parse_failures": sum(1 for r in rows if r["parse_failed"]),
        "candidate_errors": sum(1 for r in rows if r["pred_pattern"] == "CANDIDATE_ERROR"),
    }
    return summary


def render_md(summaries: Dict[str, Dict[str, Any]], n_rows: int, elapsed_s: float) -> str:
    lines = [
        "# Step 0 candidate scores (#1337)",
        "",
        f"n = {n_rows} gold-labeled queries; run wall-clock {elapsed_s:.0f}s.",
        "",
        "| candidate | pattern acc (95% CI) | agents exact | jaccard | p50 ms | p95 ms | LLM share | parse fails | errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        lo, hi = s["pattern_accuracy_ci95"]
        lines.append(
            f"| {name} | {s['pattern_accuracy']:.3f} ({lo:.3f}–{hi:.3f}) "
            f"| {s['agents_exact_rate']:.3f} | {s['agents_jaccard_mean']:.3f} "
            f"| {s['latency_ms_p50']} | {s['latency_ms_p95']} "
            f"| {s['llm_share']:.2f} | {s['parse_failures']} | {s['candidate_errors']} |"
        )
    lines.append("")
    for name, s in summaries.items():
        lines.append(f"## {name} — per-pattern")
        lines.append("")
        lines.append("| pattern | gold n | recall | precision | f1 |")
        lines.append("|---|---|---|---|---|")
        for pat, st in s["per_pattern"].items():
            lines.append(
                f"| {pat} | {int(st['gold_n'])} | {st['recall']:.3f} "
                f"| {st['precision']:.3f} | {st['f1']:.3f} |"
            )
        lines.append("")
        lines.append(f"Confusion: `{json.dumps(s['confusion'])}`")
        lines.append("")

    # Decision readout (#1337 Step 0)
    if "pipeline_llm" in summaries and "single_llm" in summaries:
        a, b = summaries["pipeline_llm"], summaries["single_llm"]
        verdict = (
            "single_llm >= pipeline_llm → the 4-stage design does NOT merit the "
            "async-LLM-stage investment (replace rather than extend)."
            if b["pattern_accuracy"] >= a["pattern_accuracy"]
            else "pipeline_llm > single_llm → the staged design earns its keep "
            "(extend with the async LLM stage)."
        )
        lines += [
            "## Decision readout",
            "",
            f"- (a) pipeline_llm accuracy: {a['pattern_accuracy']:.3f} "
            f"(p95 {a['latency_ms_p95']} ms, LLM share {a['llm_share']:.2f})",
            f"- (b) single_llm accuracy:   {b['pattern_accuracy']:.3f} "
            f"(p95 {b['latency_ms_p95']} ms, LLM share {b['llm_share']:.2f})",
            f"- {verdict}",
            "",
            "CI overlap and per-pattern cells above qualify this readout — a",
            "difference inside the 95% CIs is not decision-grade on its own.",
            "",
        ]
    return "\n".join(lines)


def render_disagreements(rows_by_candidate: Dict[str, List[Dict[str, Any]]]) -> str:
    dis = disagreement_rows(rows_by_candidate)
    lines = [
        "# Step 0 disagreement worksheet",
        "",
        f"{len(dis)} of {max(len(v) for v in rows_by_candidate.values())} rows have "
        "at least one candidate missing gold. Protocol: human review of "
        "candidate disagreements.",
        "",
    ]
    for d in dis:
        lines.append(f"## {d['query_id']} — gold: {d['gold_pattern']}")
        lines.append(f"> {d['text']}")
        for cand, verdict in sorted(d["candidates"].items()):
            mark = "✅" if verdict["correct"] else "❌"
            lines.append(f"- {mark} {cand}: {verdict['pred_pattern']}")
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=",".join(ALL_CANDIDATES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    os.environ["ORCHESTRATOR_CLASSIFIER_MODE"] = "off"
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider != "anthropic":
        print(
            f"WARNING: LLM_PROVIDER={provider!r} — legacy fallback would not use "
            "the prod fast tier; results may not be faithful."
        )

    from scripts.benchmarks.routing.step0_candidates import build_candidates

    names = [c.strip() for c in args.candidates.split(",") if c.strip()]
    rows = load_rows(args.limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(names)

    t0 = time.perf_counter()
    results: Dict[str, List[Dict[str, Any]]] = {}
    for name in names:  # sequential per candidate; bounded concurrency within
        out_path = OUT_DIR / f"predictions_{name}.jsonl"
        if args.no_resume and out_path.exists():
            out_path.unlink()
        records = await run_candidate(
            name, candidates[name], rows, out_path, args.concurrency, not args.no_resume
        )
        wanted = {r["query_id"] for r in rows}
        results[name] = scored([r for r in records if r["query_id"] in wanted])
    elapsed = time.perf_counter() - t0

    summaries = {name: summarize(name, results[name]) for name in names}
    (OUT_DIR / "STEP0_SCORES.json").write_text(json.dumps(summaries, indent=1))
    (OUT_DIR / "STEP0_RESULTS.md").write_text(render_md(summaries, len(rows), elapsed))
    (OUT_DIR / "disagreements.md").write_text(render_disagreements(results))

    for name, s in summaries.items():
        lo, hi = s["pattern_accuracy_ci95"]
        print(
            f"{name}: acc={s['pattern_accuracy']:.3f} ({lo:.3f}-{hi:.3f}) "
            f"p95={s['latency_ms_p95']}ms llm_share={s['llm_share']:.2f} "
            f"errors={s['candidate_errors']}"
        )
    print(f"artifacts -> {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
