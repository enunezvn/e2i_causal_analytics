#!/usr/bin/env python3
"""Generate LLM perturbations of the benchmark pool's demo queries (#1337).

Expansion item 2 from issue #1337: the 51 doc-authored questions are all
clean, well-formed, self-contained English — biased *toward* rule-based
classifiers. Real traffic is messy. This script produces two variants per
demo question with haiku:

- **paraphrase** (every question): same ask, different wording/register.
- one rotating style per question index: **typo** (realistic typos, casual
  casing), **fragment** (telegraphic shorthand), **pronoun_followup**
  (a context-dependent follow-up assuming the original was just answered;
  the original text is attached as ``context.prev_user``).

Variants inherit the parent's ``demo_meta`` as provenance only — intent may
legitimately shift (a fragment can become genuinely ambiguous); gold labels
are assigned later by the Step 0 judge + human review, so ``gold_pattern``
stays null here.

The generator is not deterministic (LLM output); the committed JSONL is the
canonical artifact — re-run only to regenerate the set deliberately.

Usage (repo root)::

    .venv/bin/python scripts/benchmarks/routing/gen_perturbations.py
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

POOL = Path("scripts/benchmarks/routing/data/query_pool.jsonl")
DEFAULT_OUTPUT = Path("scripts/benchmarks/routing/data/perturbations.jsonl")
MODEL = os.environ.get("ROUTING_LABEL_JUDGE_MODEL", "claude-haiku-4-5-20251001")
STYLES = ["typo", "fragment", "pronoun_followup"]

STYLE_INSTRUCTIONS = {
    "paraphrase": "Rewrite with the same meaning but different wording and register.",
    "typo": (
        "Rewrite as a hurried user would type it: 1-3 realistic typos or"
        " misspellings, casual casing, sloppy punctuation. Same meaning."
    ),
    "fragment": (
        "Rewrite as telegraphic shorthand (drop function words, keep key"
        " terms), like a terse search-box query. Meaning still recoverable."
    ),
    "pronoun_followup": (
        "Assume the original question was just asked and answered in the"
        " chat. Write a natural FOLLOW-UP turn that depends on that context:"
        " use pronouns/ellipsis ('why is that?', 'and for the west?'),"
        " drilling into or extending the original ask."
    ),
}

PROMPT = """You generate benchmark variants of user queries for a pharma commercial-analytics chat assistant (brands: Kisqali, Fabhalta, Remibrutinib; KPIs like TRx/NRx; causal analysis, cohorts, experiments).

Original query:
{original}

Produce two variants:
1. "paraphrase": {paraphrase_instr}
2. "{style}": {style_instr}

Return STRICT JSON only, no code fences: {{"paraphrase": "...", "{style}": "..."}}"""


def generate_for(client: Any, entry: Dict[str, Any], style: str) -> List[Dict[str, Any]]:
    prompt = PROMPT.format(
        original=entry["text"],
        paraphrase_instr=STYLE_INSTRUCTIONS["paraphrase"],
        style=style,
        style_instr=STYLE_INSTRUCTIONS[style],
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    payload = json.loads(text)
    out = []
    for ptype in ("paraphrase", style):
        variant: Optional[str] = payload.get(ptype)
        if not variant or not variant.strip():
            continue
        out.append(
            {
                "text": variant.strip(),
                "source": "perturbation",
                "perturbation_type": ptype,
                "parent_query_id": entry["query_id"],
                "session_id": None,
                "created_at": None,
                "is_followup": ptype == "pronoun_followup",
                "context": (
                    {"prev_user": entry["text"], "prev_assistant": None}
                    if ptype == "pronoun_followup"
                    else None
                ),
                "n_occurrences": 1,
                "demo_meta": entry.get("demo_meta"),
                "gold_pattern": None,
                "gold_agents": None,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    load_dotenv()
    import anthropic

    client = anthropic.Anthropic()
    demo = [
        json.loads(line)
        for line in POOL.read_text().splitlines()
        if json.loads(line)["source"] == "demo"
    ]

    results: List[List[Dict[str, Any]]] = [[] for _ in demo]
    errors = 0

    def worker(i: int) -> None:
        nonlocal errors
        try:
            results[i] = generate_for(client, demo[i], STYLES[i % len(STYLES)])
        except Exception as e:  # noqa: BLE001 — one bad generation must not sink the batch
            errors += 1
            print(f"  ERROR {demo[i]['query_id']}: {e}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, range(len(demo))))

    entries = [e for batch in results for e in batch]
    for idx, e in enumerate(entries):
        e["query_id"] = f"pert-{idx:04d}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_type: Dict[str, int] = {}
    for e in entries:
        by_type[e["perturbation_type"]] = by_type.get(e["perturbation_type"], 0) + 1
    print(f"perturbations: {len(entries)} from {len(demo)} demo queries -> {args.output}")
    print(f"  by type: {by_type}  errors: {errors}")


if __name__ == "__main__":
    main()
