#!/usr/bin/env python3
"""Deterministic legacy-routing diff over the 337-query gold (#1337 rule tuning).

The #1366 guard method: score the *deterministic* legacy path — the real
``IntentClassifierNode._pattern_classify`` → ``RouterNode.execute`` →
``derive_legacy_pattern`` chain, with ZERO LLM calls — over every gold row, and
diff two runs (before/after a rule change) to see exactly which rows flip. Free,
noise-free, reproducible; the money-and-variance full haiku scorer
(score_candidates.py) is only needed for rows the pattern layer leaves ambiguous.

Rows whose pattern confidence is < 0.8 escalate to the haiku fallback in the
real ``execute()``; this scorer cannot reproduce those deterministically, so it
marks them ``escalate=True`` and reports the deterministic subset separately
from the escalating subset. A rule change is judged on:
  1. the deterministic-subset before/after diff (every flip justified vs gold), and
  2. whether it moves rows across the escalate boundary (changing LLM share) —
     if it does, only those rows need the expensive re-score.

Usage::

    PYTHONPATH=$PWD .venv/bin/python scripts/benchmarks/routing/pattern_diff.py \
        --out /tmp/after.jsonl                 # dump per-row deterministic preds
    PYTHONPATH=$PWD .venv/bin/python scripts/benchmarks/routing/pattern_diff.py \
        --diff /tmp/before.jsonl /tmp/after.jsonl   # flip report between two dumps

No network, no DB, no secrets. ORCHESTRATOR_CLASSIFIER_MODE is forced off so the
router's active-mode branch is inert and the chain is byte-identical to the
incumbent's LLM-free path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Must precede any src import that lazily reads the env.
os.environ["ORCHESTRATOR_CLASSIFIER_MODE"] = "off"
os.environ.setdefault("E2I_ALLOW_MOCK_LLM", "1")
os.environ.setdefault("E2I_TESTING_MODE", "1")

from scripts.benchmarks.routing.step0_scoring import (  # noqa: E402
    aggregate,
    derive_legacy_pattern,
    score_row,
)

DATA = Path(__file__).parent / "data" / "benchmark_queries_gold.jsonl"

# Pattern confidence at/above which the real execute() trusts the pattern layer
# and never calls the LLM (intent_classifier.execute line ~337).
PATTERN_TRUST_FLOOR = 0.8


def _load_gold() -> List[Dict[str, Any]]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


class _DeterministicLegacy:
    """The LLM-free incumbent chain, instantiated once."""

    def __init__(self) -> None:
        from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode
        from src.agents.orchestrator.nodes.router import RouterNode

        # __new__ skips __init__'s fast-LLM construction: _pattern_classify is a
        # pure method over the class-level INTENT_PATTERNS and needs no client.
        self._clf = IntentClassifierNode.__new__(IntentClassifierNode)
        self._router = RouterNode()

    async def predict(self, query: str) -> Tuple[str, List[str], bool, str]:
        intent = self._clf._pattern_classify(query.lower())
        escalate = intent["confidence"] < PATTERN_TRUST_FLOOR
        state: Dict[str, Any] = {"query": query, "intent": dict(intent)}
        routed = await self._router.execute(state)  # type: ignore[arg-type]
        names = [d["agent_name"] for d in routed.get("dispatch_plan") or []]
        pattern = derive_legacy_pattern(intent["primary_intent"], names)
        return pattern, sorted(set(names)), escalate, intent["primary_intent"]


async def _run() -> List[Dict[str, Any]]:
    gold = _load_gold()
    engine = _DeterministicLegacy()
    out: List[Dict[str, Any]] = []
    for row in gold:
        pattern, agents, escalate, primary = await engine.predict(row["text"])
        out.append(
            {
                "query_id": row["query_id"],
                "text": row["text"],
                "gold_pattern": row["gold_pattern"],
                "gold_agents": sorted(row.get("gold_agents") or []),
                "pred_pattern": pattern,
                "pred_agents": agents,
                "primary_intent": primary,
                "escalate": escalate,
            }
        )
    return out


def _score(rows: List[Dict[str, Any]], subset: str) -> Dict[str, Any]:
    """Aggregate over ``subset`` in {all, deterministic, escalate}."""
    if subset == "deterministic":
        rows = [r for r in rows if not r["escalate"]]
    elif subset == "escalate":
        rows = [r for r in rows if r["escalate"]]
    scored = [
        {
            "gold_pattern": r["gold_pattern"],
            "pred_pattern": r["pred_pattern"],
            "score": score_row(
                r["gold_pattern"], r["gold_agents"], r["pred_pattern"], r["pred_agents"]
            ),
        }
        for r in rows
    ]
    return aggregate(scored)


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    n = len(rows)
    n_esc = sum(1 for r in rows if r["escalate"])
    print(f"total gold rows: {n}  deterministic: {n - n_esc}  escalate(<0.8): {n_esc}\n")
    for subset in ("all", "deterministic"):
        agg = _score(rows, subset)
        print(f"### subset={subset}  n={agg['n']}")
        print(
            f"  pattern_acc={agg['pattern_accuracy']:.3f} "
            f"agents_exact={agg['agents_exact_rate']:.3f} "
            f"jaccard={agg['agents_jaccard_mean']:.3f}"
        )
        for pat, s in agg["per_pattern"].items():
            print(
                f"    {pat:22s} gold_n={int(s['gold_n']):3d} "
                f"recall={s['recall']:.3f} precision={s['precision']:.3f} f1={s['f1']:.3f}"
            )
        print(f"  confusion: {agg['confusion']}\n")


def _diff(before_path: Path, after_path: Path) -> None:
    before = {r["query_id"]: r for r in map(json.loads, before_path.read_text().splitlines())}
    after = {r["query_id"]: r for r in map(json.loads, after_path.read_text().splitlines())}
    flips: List[Dict[str, Any]] = []
    esc_flips: List[Dict[str, Any]] = []
    for qid, a in after.items():
        b = before.get(qid)
        if b is None:
            continue
        if b["escalate"] != a["escalate"]:
            esc_flips.append({"qid": qid, "b": b, "a": a})
        if (b["pred_pattern"], b["pred_agents"]) != (a["pred_pattern"], a["pred_agents"]):
            flips.append({"qid": qid, "b": b, "a": a})
    print(f"=== {len(flips)} row(s) changed deterministic prediction ===")
    gained, lost = 0, 0
    for f in flips:
        b, a = f["b"], f["a"]
        b_ok = b["pred_pattern"] == b["gold_pattern"] and b["pred_agents"] == b["gold_agents"]
        a_ok = a["pred_pattern"] == a["gold_pattern"] and a["pred_agents"] == a["gold_agents"]
        mark = "  "
        if a_ok and not b_ok:
            mark, gained = "++", gained + 1
        elif b_ok and not a_ok:
            mark, lost = "--", lost + 1
        print(
            f"{mark} {f['qid']} gold={b['gold_pattern']}/{b['gold_agents']}\n"
            f"     before: {b['pred_pattern']}/{b['pred_agents']} (esc={b['escalate']})\n"
            f"     after:  {a['pred_pattern']}/{a['pred_agents']} (esc={a['escalate']})\n"
            f"     Q: {b['text']!r}"
        )
    print(f"\nagent-exact gained: {gained}  lost: {lost}  net: {gained - lost}")
    print(f"=== {len(esc_flips)} row(s) crossed the escalate(<0.8) boundary ===")
    for f in esc_flips:
        b, a = f["b"], f["a"]
        print(f"  {f['qid']} escalate {b['escalate']}->{a['escalate']}  Q: {b['text']!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="dump per-row deterministic predictions as JSONL")
    ap.add_argument("--diff", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()

    if args.diff:
        _diff(args.diff[0], args.diff[1])
        return

    rows = asyncio.run(_run())
    _print_summary(rows)
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
