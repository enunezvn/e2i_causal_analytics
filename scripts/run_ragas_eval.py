#!/usr/bin/env python3
"""
RAGAS Evaluation Script for E2I RAG Pipeline.

Run RAG quality evaluation and optionally fail CI on threshold violations.

Usage:
    # Run evaluation with default settings
    python scripts/run_ragas_eval.py

    # Run with custom thresholds
    python scripts/run_ragas_eval.py --faithfulness 0.90 --answer-relevancy 0.95

    # Run without MLflow logging (for local testing)
    python scripts/run_ragas_eval.py --no-mlflow

    # Run with custom dataset
    python scripts/run_ragas_eval.py --dataset path/to/dataset.json

    # Fail CI on threshold violations
    python scripts/run_ragas_eval.py --fail-on-threshold
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.evaluation import (
    EvaluationConfig,
    RAGEvaluationPipeline,
    DEFAULT_THRESHOLDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation for E2I RAG pipeline"
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom evaluation dataset (JSON)",
    )

    # Threshold options
    parser.add_argument(
        "--faithfulness",
        type=float,
        default=DEFAULT_THRESHOLDS["faithfulness"],
        help=f"Faithfulness threshold (default: {DEFAULT_THRESHOLDS['faithfulness']})",
    )
    parser.add_argument(
        "--answer-relevancy",
        type=float,
        default=DEFAULT_THRESHOLDS["answer_relevancy"],
        help=f"Answer relevancy threshold (default: {DEFAULT_THRESHOLDS['answer_relevancy']})",
    )
    parser.add_argument(
        "--context-precision",
        type=float,
        default=DEFAULT_THRESHOLDS["context_precision"],
        help=f"Context precision threshold (default: {DEFAULT_THRESHOLDS['context_precision']})",
    )
    parser.add_argument(
        "--context-recall",
        type=float,
        default=DEFAULT_THRESHOLDS["context_recall"],
        help=f"Context recall threshold (default: {DEFAULT_THRESHOLDS['context_recall']})",
    )

    # MLflow options
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="rag-evaluation",
        help="MLflow experiment name",
    )

    # CI options
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with error code if thresholds not met",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation report (JSON)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


async def main() -> int:
    """Run evaluation pipeline."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build configuration
    thresholds = {
        "faithfulness": args.faithfulness,
        "answer_relevancy": args.answer_relevancy,
        "context_precision": args.context_precision,
        "context_recall": args.context_recall,
    }

    config = EvaluationConfig(
        thresholds=thresholds,
        log_to_mlflow=not args.no_mlflow,
        mlflow_experiment=args.mlflow_experiment,
    )

    logger.info("=" * 60)
    logger.info("E2I RAG Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Thresholds: {thresholds}")
    logger.info(f"MLflow logging: {'Enabled' if not args.no_mlflow else 'Disabled'}")

    # Initialize pipeline
    pipeline = RAGEvaluationPipeline(
        config=config,
        dataset_path=args.dataset,
    )

    logger.info(f"Dataset: {len(pipeline.dataset)} samples")

    # Run evaluation
    logger.info("-" * 60)
    logger.info("Running evaluation...")

    report = await pipeline.run_evaluation()

    # Log to MLflow
    if not args.no_mlflow:
        pipeline.log_to_mlflow(report)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2)
        logger.info(f"Report saved to: {output_path}")

    # Print results
    logger.info("-" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("-" * 60)
    logger.info(f"Run ID: {report.run_id}")
    logger.info(f"Total Samples: {report.total_samples}")
    logger.info(f"Passed: {report.passed_samples}")
    logger.info(f"Failed: {report.failed_samples}")
    logger.info(f"Pass Rate: {report.passed_samples / report.total_samples * 100:.1f}%")
    logger.info("-" * 60)
    logger.info("METRICS")
    logger.info("-" * 60)

    if report.avg_faithfulness is not None:
        status = "PASS" if report.avg_faithfulness >= thresholds["faithfulness"] else "FAIL"
        logger.info(
            f"Faithfulness:       {report.avg_faithfulness:.3f} (threshold: {thresholds['faithfulness']}) [{status}]"
        )

    if report.avg_answer_relevancy is not None:
        status = "PASS" if report.avg_answer_relevancy >= thresholds["answer_relevancy"] else "FAIL"
        logger.info(
            f"Answer Relevancy:   {report.avg_answer_relevancy:.3f} (threshold: {thresholds['answer_relevancy']}) [{status}]"
        )

    if report.avg_context_precision is not None:
        status = "PASS" if report.avg_context_precision >= thresholds["context_precision"] else "FAIL"
        logger.info(
            f"Context Precision:  {report.avg_context_precision:.3f} (threshold: {thresholds['context_precision']}) [{status}]"
        )

    if report.avg_context_recall is not None:
        status = "PASS" if report.avg_context_recall >= thresholds["context_recall"] else "FAIL"
        logger.info(
            f"Context Recall:     {report.avg_context_recall:.3f} (threshold: {thresholds['context_recall']}) [{status}]"
        )

    if report.overall_score is not None:
        logger.info(f"Overall Score:      {report.overall_score:.3f}")

    logger.info("-" * 60)
    logger.info(f"Evaluation Time: {report.evaluation_time_seconds:.2f}s")
    logger.info("=" * 60)

    # Check thresholds
    passed, failures = pipeline.check_thresholds(report)

    if not passed:
        logger.warning("THRESHOLD VIOLATIONS:")
        for failure in failures:
            logger.warning(f"  - {failure}")

        if args.fail_on_threshold:
            logger.error("Exiting with error due to threshold violations")
            return 1

    logger.info("All thresholds passed!" if passed else "Some thresholds not met")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
