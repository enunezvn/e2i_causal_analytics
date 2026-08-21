#!/usr/bin/env python3
"""Assemble the full #1337 Step 0 benchmark set from its three sources.

Merges (in stable order) ``data/query_pool.jsonl`` (demo + historical),
``data/perturbations.jsonl``, and ``data/authored_queries.jsonl`` into
``data/benchmark_queries.jsonl`` with sequential ``bench-NNNN`` ids
(original per-source ids preserved as ``source_query_id``), deduplicated on
normalized text (first occurrence wins). Prints the composition report
against the issue #1337 expansion targets.

Gold labels: only authored entries carry ``authored_gold_pattern`` (author
proposal, seeds the gold protocol); everything else is unlabeled until the
gold-judge + human-review stage.

Usage (repo root)::

    .venv/bin/python scripts/benchmarks/routing/assemble_benchmark.py
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

DATA = Path("scripts/benchmarks/routing/data")
SOURCES = ["query_pool.jsonl", "perturbations.jsonl", "authored_queries.jsonl"]
OUTPUT = DATA / "benchmark_queries.jsonl"


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main() -> None:
    entries = []
    seen = set()
    dropped = 0
    for name in SOURCES:
        for line in (DATA / name).read_text().splitlines():
            e = json.loads(line)
            key = normalize(e["text"])
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            e["source_query_id"] = e.pop("query_id")
            entries.append(e)

    for idx, e in enumerate(entries):
        e["query_id"] = f"bench-{idx:04d}"

    with OUTPUT.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    sources = Counter(e["source"] for e in entries)
    cells = Counter(e.get("cell") for e in entries if e.get("cell"))
    proposed = Counter(
        e.get("authored_gold_pattern") for e in entries if e.get("authored_gold_pattern")
    )
    followups = sum(1 for e in entries if e.get("is_followup"))
    print(f"benchmark: {len(entries)} queries -> {OUTPUT} (dropped {dropped} duplicates)")
    print(f"  by source: {dict(sources)}")
    print(f"  authored cells: {dict(cells)}")
    print(f"  author-proposed gold patterns: {dict(proposed)}")
    print(f"  follow-ups with context: {followups}")
    print("  targets (#1337): ~200-300 total; at least ten queries per unmapped agent;")
    print("  ambiguous CLARIFICATION cell populated; gold labels pending judge+human stage")


if __name__ == "__main__":
    main()
