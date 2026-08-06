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
                    "n_contexts": len(sample.retrieved_contexts),
                    "faithfulness": res.faithfulness,
                    "answer_relevancy": res.answer_relevancy,
                    # ADD-ONLY (#1485). _evaluate_with_ragas ends in a broad
                    # `except Exception: return await self._evaluate_with_fallback(...)`
                    # (src/rag/evaluation.py:1188), so a quota error or rate
                    # limit mid-run silently degrades a sample to word-overlap
                    # heuristics while the process still exits 0. The fallback
                    # stamps evaluation_method="fallback_heuristic" (:1270);
                    # without carrying it out here, no consumer can tell a
                    # judged score from a heuristic one. The judged path stamps
                    # nothing, so None means judged. Consumers that do not read
                    # this key are unaffected.
                    "evaluation_method": (res.metadata or {}).get("evaluation_method"),
                }
            )
            print(f"[{i + 1}/{len(samples)}] judged", file=sys.stderr)
        return results

    per_sample = asyncio.run(_run())

    def _mean(rows: list, key: str):
        vals = [r[key] for r in rows if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else None

    # Faithfulness measures answer-vs-contexts; on a run that retrieved no
    # evidence the score is an artifact (NaN->0), so it averages only over
    # samples with contexts. How often retrieval finds evidence is itself
    # model-influenced (hop decisions ride the LM) - n_faithfulness exposes it.
    with_ctx = [r for r in per_sample if r["n_contexts"] > 0]
    out = {
        "model": model,
        "n_samples": len(per_sample),
        "n_faithfulness": len(with_ctx),
        "faithfulness": _mean(with_ctx, "faithfulness"),
        "answer_relevancy": _mean(per_sample, "answer_relevancy"),
        "per_sample": per_sample,
    }
    print("RESULTS_JSON_BEGIN")
    print(json.dumps(out))
    print("RESULTS_JSON_END")


main()
