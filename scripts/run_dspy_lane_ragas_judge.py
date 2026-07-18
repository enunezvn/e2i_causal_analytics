#!/usr/bin/env python
"""RAGAS judge step for the DSPy-lane provider A/B.

Scores each candidate's REAL e2e-generated answers (built by
``build_ragas_samples`` from the e2e replay records) with the frozen gpt-4o
judge via the production ``RAGASEvaluator``. Only faithfulness and
answer_relevancy feed the flip gates; context precision/recall need a
ground-truth reference the replay samples deliberately do not fabricate.

Run INSIDE the prod container (ragas + OPENAI_API_KEY live there); the script
goes in via ``python -c`` so stdin can carry the samples::

    docker exec -i e2i_api python -c "$(cat scripts/run_dspy_lane_ragas_judge.py)" \
        < samples.json

stdin JSON shape: ``{"model": "<litellm id>", "samples": [EvaluationSample...]}``

Opik tracing is disabled (the Opik stack is intentionally stopped).
"""

from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, "/app")


def main() -> None:
    payload = json.load(sys.stdin)
    model = payload["model"]
    from src.rag.evaluation import EvaluationSample, RAGASEvaluator

    samples = [EvaluationSample(**s) for s in payload["samples"]]
    evaluator = RAGASEvaluator(llm_provider="openai", enable_opik_tracing=False)

    async def _run() -> list:
        results = []
        for i, sample in enumerate(samples):
            res = await evaluator.evaluate_sample(sample)
            results.append(
                {
                    "query_id": sample.metadata.get("query_id"),
                    "faithfulness": res.faithfulness,
                    "answer_relevancy": res.answer_relevancy,
                }
            )
            print(f"[{i + 1}/{len(samples)}] judged", file=sys.stderr)
        return results

    per_sample = asyncio.run(_run())

    def _mean(key: str):
        vals = [r[key] for r in per_sample if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else None

    out = {
        "model": model,
        "n_samples": len(per_sample),
        "faithfulness": _mean("faithfulness"),
        "answer_relevancy": _mean("answer_relevancy"),
        "per_sample": per_sample,
    }
    print("RESULTS_JSON_BEGIN")
    print(json.dumps(out))
    print("RESULTS_JSON_END")


main()
