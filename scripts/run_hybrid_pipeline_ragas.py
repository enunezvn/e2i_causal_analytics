#!/usr/bin/env python3
"""Run RAGAS against the live hybrid retrieval + generation pipeline.

Unlike ``run_ragas_eval.py`` (the frozen fixture/judge-drift sentinel) and
``run_real_pipeline_ragas.py`` (recorded cognitive-RAG HTTP replays), this
manual lane directly exercises ``RAGService.query``:

    QueryOptimizer -> HybridRetriever -> CrossEncoderReranker -> InsightEnricher

It requires live Supabase/FalkorDB, Anthropic generation, OpenAI RAGAS judge,
and the opt-in retrieval flags. It is intentionally manual because it spends
judge budget and may download the cross-encoder into ``HF_HOME``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.routes.rag import RAGService
from src.rag.evaluation import EvaluationConfig, RAGEvaluationPipeline
from src.rag.hybrid_ragas import hybrid_runtime_validation_errors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hybrid_pipeline_ragas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Optional RAGAS dataset JSON")
    parser.add_argument("--limit", type=int, default=10, help="Samples to judge (default: 10)")
    parser.add_argument("--output", required=True, help="Evaluation report JSON path")
    parser.add_argument("--fail-on-threshold", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    missing_keys = [
        key for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.environ.get(key)
    ]
    if missing_keys:
        raise SystemExit(
            f"Missing required API keys: {', '.join(missing_keys)}. The hybrid service "
            "uses Anthropic for answer generation and the production RAGAS path uses "
            "OpenAI embeddings plus the gpt-4o judge; heuristic scores are refused."
        )

    service = RAGService.get_instance()
    if service.query_optimizer is None or service.reranker is None:
        raise SystemExit(
            "Set RAG_ENABLE_QUERY_OPTIMIZATION=true and RAG_ENABLE_RERANKING=true; "
            "this lane refuses to report a run that skipped either component."
        )

    pipeline = RAGEvaluationPipeline(
        config=EvaluationConfig(log_to_mlflow=not args.no_mlflow),
        dataset_path=args.dataset,
        enable_opik_tracing=False,
    )
    pipeline.dataset = pipeline.dataset[: args.limit]
    if not pipeline.dataset:
        raise SystemExit(
            "Evaluation dataset contains no samples; refusing a vacuous validation run."
        )
    if not pipeline.evaluator.can_judge:
        raise SystemExit(
            "RAGAS judge unavailable: " + "; ".join(pipeline.evaluator.judged_path_blockers)
        )
    pipeline.evaluator.verify_dependencies()

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
