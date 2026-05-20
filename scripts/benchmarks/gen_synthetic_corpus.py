#!/usr/bin/env python3
"""Generate synthetic corpus JSONL for BM25 rebuild-time benchmark.

Produces ``tests/benchmarks/data/synthetic_corpus.jsonl`` with ~1500
synthetic documents (~50 tokens each). The benchmark slices via repetition
to produce 1k / 5k / 10k doc-equivalents, measuring index build wall-clock at
each slice for a build-time curve (not a single point per issue #391 Box 3).

The vocabulary is intentionally drawn from the same pharma-commercial
domain as the real corpus (Kisqali, Fabhalta, TRx, NRx, HCP, oncologist,
etc.) so token distributions roughly match production — although the
absolute token frequencies + zero-cardinality terms are obviously synthetic.

Deterministic generator: re-run produces identical output (seed pinned).

Usage::

    python scripts/benchmarks/gen_synthetic_corpus.py

See ``tests/benchmarks/data/CURATION_PERF.md`` for the schema + curation
policy.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

_SEED = 391_2026_05_20_2

# Pharma-commercial-shaped vocabulary. Mix of brand names, KPI shortcodes,
# clinical specialty terms, and common English connective tokens.
_VOCAB = (
    "Kisqali Fabhalta Remibrutinib Cosentyx Entresto Pluvicto"
    + " TRx NRx market share growth decline conversion lift"
    + " HCP oncologist cardiologist nephrologist dermatologist"
    + " patient cohort eligibility prescribing pattern adherence"
    + " brand region quarter q1 q2 q3 q4 north south east west"
    + " causal effect treatment outcome endpoint confidence interval"
    + " adoption decline plateau acceleration trajectory hazard ratio"
    + " breast cancer HR-positive HER2-negative PNH paroxysmal hemoglobinuria"
    + " CSU urticaria psoriasis plaque heart failure HFrEF HFpEF"
    + " adherence persistence switching prescriber portfolio coverage"
    + " confidence score 0.85 0.90 0.95 trigger reason evaluator"
    + " sentinel drift monitor agent activity recommended action"
).split()

_DOC_COUNT = 1500
_TOKENS_PER_DOC = 50

_OUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "benchmarks"
    / "data"
    / "synthetic_corpus.jsonl"
)


def main() -> None:
    rng = random.Random(_SEED)
    docs: list[dict[str, str]] = []
    for i in range(_DOC_COUNT):
        tokens = [rng.choice(_VOCAB) for _ in range(_TOKENS_PER_DOC)]
        docs.append(
            {
                "doc_id": f"doc-bench-{i:05d}",
                "content": " ".join(tokens),
            }
        )

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUT_PATH.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Synthetic corpus for BM25 rebuild-time benchmark (issue #391, Box 3).\n"
        )
        fh.write("# See tests/benchmarks/data/CURATION_PERF.md for schema + policy.\n")
        fh.write(
            "# Deterministic generator: scripts/benchmarks/gen_synthetic_corpus.py "
            f"(seed={_SEED}).\n"
        )
        fh.write(
            f"# Doc count: {_DOC_COUNT}, tokens per doc: {_TOKENS_PER_DOC} "
            f"(~{_DOC_COUNT * _TOKENS_PER_DOC} tokens total).\n"
        )
        for doc in docs:
            fh.write(json.dumps(doc, separators=(",", ":")) + "\n")

    print(f"Wrote {len(docs)} docs ({_DOC_COUNT * _TOKENS_PER_DOC} tokens) to {_OUT_PATH}")


if __name__ == "__main__":
    main()
