#!/usr/bin/env python3
"""Run RAGAS against the live hybrid retrieval + generation pipeline.

Unlike ``run_ragas_eval.py`` (the frozen fixture/judge-drift sentinel) and
``run_real_pipeline_ragas.py`` (recorded cognitive-RAG HTTP replays), this
manual lane directly exercises ``RAGService.query``:

    QueryOptimizer -> HybridRetriever -> CrossEncoderReranker -> InsightEnricher

It requires a corpus-grounded evaluation dataset, live Supabase/FalkorDB,
Anthropic generation, an OpenAI RAGAS judge, and the opt-in retrieval flags.
The default evaluator fixture is deliberately refused because its simulated
facts need not exist in the live corpus and would make context metrics
meaningless. This lane is manual because it spends judge budget and may
download the cross-encoder into ``HF_HOME``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.dependencies.supabase_client import get_supabase
from src.api.routes.rag import RAGService
from src.rag.evaluation import EvaluationConfig, RAGEvaluationPipeline
from src.rag.hybrid_ragas import (
    corpus_grounding_validation_errors,
    hybrid_runtime_validation_errors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hybrid_pipeline_ragas")


def load_live_corpus_rows(samples: Sequence[object]) -> list[dict[str, object]]:
    """Load only the live chunks claimed by the evaluation samples."""
    chunk_ids = sorted(
        {
            str(chunk_id)
            for sample in samples
            if (chunk_id := (getattr(sample, "metadata", {}) or {}).get("source_chunk_id"))
        }
    )
    if not chunk_ids:
        return []
    supabase = get_supabase()
    if supabase is None:
        raise SystemExit("Supabase is unavailable; cannot verify dataset corpus grounding.")
    response = (
        supabase.table("rag_document_chunks")
        .select("chunk_id,document_id,content,is_synthetic")
        .in_("chunk_id", chunk_ids)
        .execute()
    )
    return list(response.data or [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Corpus-grounded RAGAS dataset JSON (the simulated default fixture is refused)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Samples to judge (default: 10)")
    parser.add_argument("--output", required=True, help="Evaluation report JSON path")
    parser.add_argument("--fail-on-threshold", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if not args.dataset:
        raise SystemExit(
            "A corpus-grounded --dataset is required; the simulated default fixture "
            "cannot produce meaningful live retrieval metrics."
        )
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        raise SystemExit(
            f"Evaluation dataset does not exist or is not a file: {dataset_path}. "
            "Refusing to fall back to the simulated default fixture."
        )
    missing_keys = [
        key for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.environ.get(key)
    ]
    if missing_keys:
        raise SystemExit(
            f"Missing required API keys: {', '.join(missing_keys)}. The hybrid service "
            "uses Anthropic for answer generation and the production RAGAS path uses "
            "OpenAI embeddings plus the gpt-4o judge; heuristic scores are refused."
        )

    pipeline = RAGEvaluationPipeline(
        config=EvaluationConfig(log_to_mlflow=not args.no_mlflow),
        dataset_path=str(dataset_path),
        enable_opik_tracing=False,
    )
    pipeline.dataset = pipeline.dataset[: args.limit]
    if not pipeline.dataset:
        raise SystemExit(
            "Evaluation dataset contains no samples; refusing a vacuous validation run."
        )
    grounding_errors = corpus_grounding_validation_errors(
        pipeline.dataset, load_live_corpus_rows(pipeline.dataset)
    )
    if grounding_errors:
        details = "; ".join(grounding_errors[:10])
        raise SystemExit(
            "Evaluation dataset is not verified against the live non-synthetic corpus: " + details
        )
    if not pipeline.evaluator.can_judge:
        raise SystemExit(
            "RAGAS judge unavailable: " + "; ".join(pipeline.evaluator.judged_path_blockers)
        )
    pipeline.evaluator.verify_dependencies()

    service = RAGService.get_instance()
    if service.query_optimizer is None or service.reranker is None:
        raise SystemExit(
            "Set RAG_ENABLE_QUERY_OPTIMIZATION=true and RAG_ENABLE_RERANKING=true; "
            "this lane refuses to report a run that skipped either component."
        )

    report = await pipeline.run_evaluation(rag_pipeline=service)
    runtime_errors = hybrid_runtime_validation_errors(pipeline.dataset, report.results)
    # Do not contaminate the normal quality experiment with infrastructure or
    # provenance failures. The JSON artifact below retains the rejected run
    # and its reasons for local diagnosis.
    if not args.no_mlflow and not runtime_errors:
        pipeline.log_to_mlflow(report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": report.model_dump(),
        "runtime_validation": {"passed": not runtime_errors, "errors": runtime_errors},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Hybrid-pipeline RAGAS report written to %s", output)

    if runtime_errors:
        for error in runtime_errors:
            logger.error("Runtime validation: %s", error)
        return 2
    if args.fail_on_threshold and not report.all_thresholds_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
