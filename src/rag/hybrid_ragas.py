"""Fail-closed runtime-integrity checks for hybrid-pipeline RAGAS runs."""

from __future__ import annotations

from typing import Any, Sequence


def hybrid_runtime_validation_errors(
    samples: Sequence[Any],
    results: Sequence[Any],
) -> list[str]:
    """Return reasons a run cannot be called optimizer+reranker RAGAS.

    Metric thresholds answer a quality question only after runtime provenance
    proves the requested components actually ran and the scores came from the
    real judge rather than the evaluator's heuristic fallback.
    """
    errors: list[str] = []
    if len(results) != len(samples):
        errors.append(
            f"evaluation result count mismatch: {len(results)} results for {len(samples)} samples"
        )
    for index, sample in enumerate(samples):
        metadata = getattr(sample, "metadata", {}) or {}
        runtime = metadata.get("pipeline_metadata")
        label = str(metadata.get("id") or getattr(sample, "query", index))[:120]
        if not isinstance(runtime, dict):
            errors.append(f"{label}: missing pipeline runtime metadata")
            continue
        if runtime.get("query_optimizer_enabled") is not True:
            errors.append(f"{label}: query optimizer was not enabled")
        if runtime.get("reranker_enabled") is not True:
            errors.append(f"{label}: reranker was not enabled")
        if runtime.get("query_optimizer_error"):
            errors.append(f"{label}: query optimizer failed ({runtime['query_optimizer_error']})")
        if runtime.get("reranker_error"):
            errors.append(f"{label}: reranker failed ({runtime['reranker_error']})")
        if runtime.get("generation_error"):
            errors.append(f"{label}: answer generation failed ({runtime['generation_error']})")

        retrieval_hit = runtime.get("retrieval_hit")
        retrieved_contexts = getattr(sample, "retrieved_contexts", None) or []
        if retrieval_hit is True:
            if not retrieved_contexts:
                errors.append(f"{label}: retrieval metadata reports a hit without contexts")
            if runtime.get("reranked") is not True:
                errors.append(f"{label}: context-bearing retrieval was not reranked")
            if runtime.get("generation_attempted") is not True:
                errors.append(f"{label}: answer generation was not attempted after a retrieval hit")
            if runtime.get("generation_succeeded") is not True:
                errors.append(f"{label}: answer generation did not succeed")
        elif retrieval_hit is False:
            if retrieved_contexts:
                errors.append(f"{label}: retrieval metadata reports a miss with contexts")
        else:
            errors.append(f"{label}: retrieval outcome metadata is missing")

    for index, result in enumerate(results):
        metadata = getattr(result, "metadata", {}) or {}
        if metadata.get("evaluation_method") == "fallback_heuristic":
            errors.append(f"result[{index}]: heuristic fallback is not a RAGAS judge score")
    return errors
