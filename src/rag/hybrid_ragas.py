"""Fail-closed runtime-integrity checks for hybrid-pipeline RAGAS runs."""

from __future__ import annotations

import math
from typing import Any, Sequence

CORPUS_DATASET_PROVENANCE = "live_non_synthetic_rag_document_chunks"
RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "overall_score",
)


def corpus_grounding_validation_errors(
    samples: Sequence[Any],
    corpus_rows: Sequence[dict[str, Any]],
) -> list[str]:
    """Verify every benchmark reference against a live non-synthetic chunk.

    A provenance label alone is not evidence. Each sample must identify a row
    returned from the live ``rag_document_chunks`` table, and its document ID,
    ground truth, and reference context must match that row.
    """
    errors: list[str] = []
    rows_by_chunk_id = {str(row.get("chunk_id")): row for row in corpus_rows if row.get("chunk_id")}
    for index, sample in enumerate(samples):
        metadata = getattr(sample, "metadata", {}) or {}
        label = str(metadata.get("id") or getattr(sample, "query", index))[:120]
        if metadata.get("dataset_provenance") != CORPUS_DATASET_PROVENANCE:
            errors.append(f"{label}: unsupported dataset provenance")

        chunk_id = metadata.get("source_chunk_id")
        document_id = metadata.get("source_document_id")
        if not chunk_id:
            errors.append(f"{label}: missing source chunk ID")
            continue
        if not document_id:
            errors.append(f"{label}: missing source document ID")

        row = rows_by_chunk_id.get(str(chunk_id))
        if row is None:
            errors.append(f"{label}: source chunk was not found in the live corpus")
            continue
        if row.get("is_synthetic") is not False:
            errors.append(f"{label}: source chunk is synthetic or lacks real-data provenance")
        if str(row.get("document_id") or "") != str(document_id or ""):
            errors.append(f"{label}: source document ID does not match the live corpus")

        content = row.get("content")
        if not isinstance(content, str) or not content:
            errors.append(f"{label}: live source chunk has no content")
            continue
        if getattr(sample, "ground_truth", None) != content:
            errors.append(f"{label}: ground truth does not exactly match live source content")
        contexts = getattr(sample, "contexts", None) or []
        if content not in contexts:
            errors.append(f"{label}: reference contexts do not contain live source content")
    return errors


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
        method = metadata.get("evaluation_method")
        if method != "ragas":
            if method == "fallback_heuristic":
                errors.append(f"result[{index}]: heuristic fallback is not a RAGAS judge score")
            else:
                errors.append(f"result[{index}]: RAGAS judgment is not explicitly attested")
        unmeasured = metadata.get("unmeasured_metrics") or []
        if unmeasured:
            errors.append(
                f"result[{index}]: unmeasured RAGAS metrics: {', '.join(map(str, unmeasured))}"
            )
        for metric in RAGAS_METRICS:
            value = getattr(result, metric, None)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                errors.append(f"result[{index}]: {metric} is not a finite score")
    return errors
