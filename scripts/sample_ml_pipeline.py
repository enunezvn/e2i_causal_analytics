#!/usr/bin/env python3
"""Sample ML Foundation Pipeline - End-to-End Training Example.

This script demonstrates how to use the MLFoundationPipeline to train
a complete ML model from business objectives to deployment.

Example Use Cases:
1. HCP Conversion Prediction - Predict which HCPs will convert
2. Patient Journey Churn - Predict patient dropout risk
3. Trigger Effectiveness - Predict trigger success rate

Usage:
    python scripts/sample_ml_pipeline.py --use-case hcp_conversion
    python scripts/sample_ml_pipeline.py --use-case churn --brand Remibrutinib
    python scripts/sample_ml_pipeline.py --use-case trigger_effectiveness --dry-run

Requirements:
    - Tier 0 agents initialized
    - Database accessible (or use --sample-data flag)
    - MLflow server running (or use --skip-mlflow flag)
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.tier_0 import MLFoundationPipeline, PipelineConfig, PipelineResult, PipelineStage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# USE CASE DEFINITIONS
# =============================================================================

USE_CASES: Dict[str, Dict[str, Any]] = {
    "hcp_conversion": {
        "problem_description": (
            "Predict which Healthcare Professionals (HCPs) are likely to convert "
            "from non-prescribers to active prescribers within the next 90 days. "
            "This enables targeted engagement strategies for field teams."
        ),
        "business_objective": (
            "Increase HCP conversion rate by 15% through data-driven targeting. "
            "Prioritize HCPs with high conversion probability for sales rep visits."
        ),
        "target_outcome": "hcp_conversion",
        "data_source": "business_metrics",
        "problem_type_hint": "binary_classification",
        "target_variable": "converted_90d",
        "candidate_features": [
            "total_interactions",
            "time_since_last_contact",
            "specialty_alignment",
            "sample_requests",
            "email_engagement_rate",
            "prior_rx_volume",
            "territory_potential",
        ],
        "use_case": "commercial_targeting",
        "prediction_horizon_days": 90,
    },
    "churn": {
        "problem_description": (
            "Identify patients at risk of discontinuing therapy within 30 days. "
            "Early intervention can improve adherence and patient outcomes."
        ),
        "business_objective": (
            "Reduce patient churn rate by 20% through proactive engagement. "
            "Target patients showing early warning signs for adherence programs."
        ),
        "target_outcome": "patient_churn",
        "data_source": "patient_journeys",
        "problem_type_hint": "binary_classification",
        "target_variable": "churned_30d",
        "candidate_features": [
            "refill_frequency",
            "days_since_last_refill",
            "therapy_duration",
            "side_effect_reports",
            "support_program_enrollment",
            "insurance_coverage_changes",
        ],
        "use_case": "patient_retention",
        "prediction_horizon_days": 30,
    },
    "trigger_effectiveness": {
        "problem_description": (
            "Predict the effectiveness of marketing triggers for different HCP segments. "
            "Optimize trigger selection to maximize response rates."
        ),
        "business_objective": (
            "Improve trigger response rate by 25% through personalized trigger selection. "
            "Ensure efficient resource allocation for marketing campaigns."
        ),
        "target_outcome": "trigger_response",
        "data_source": "triggers",
        "problem_type_hint": "binary_classification",
        "target_variable": "positive_response",
        "candidate_features": [
            "trigger_type",
            "hcp_segment",
            "time_of_day",
            "channel",
            "previous_trigger_responses",
            "engagement_history",
            "brand_affinity",
        ],
        "use_case": "marketing_optimization",
        "prediction_horizon_days": 14,
    },
    "roi_prediction": {
        "problem_description": (
            "Predict the expected ROI of marketing investments across different channels "
            "and territories to optimize budget allocation."
        ),
        "business_objective": (
            "Maximize marketing ROI by allocating budget to highest-return channels. "
            "Achieve 10% improvement in marketing efficiency."
        ),
        "target_outcome": "roi_value",
        "data_source": "business_metrics",
        "problem_type_hint": "regression",
        "target_variable": "actual_roi",
        "candidate_features": [
            "channel",
            "territory",
            "investment_amount",
            "historical_performance",
            "market_size",
            "competitive_activity",
            "seasonality_index",
        ],
        "use_case": "budget_optimization",
        "prediction_horizon_days": 90,
    },
}


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================


def create_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Create pipeline configuration from command line arguments.

    Args:
        args: Command line arguments

    Returns:
        PipelineConfig instance
    """
    return PipelineConfig(
        # Deployment options
        skip_deployment=args.skip_deployment or args.dry_run,
        target_environment=args.environment,

        # HPO options
        enable_hpo=not args.skip_hpo,
        hpo_trials=args.hpo_trials,
        hpo_timeout_hours=args.hpo_timeout,
        early_stopping=True,

        # Feature analysis
        skip_feature_analysis=args.skip_shap,

        # MLflow options
        skip_mlflow=args.skip_mlflow or args.dry_run,

        # Data options
        use_sample_data=args.sample_data,
        skip_leakage_check=args.skip_leakage_check,

        # Observability
        enable_observability=not args.skip_observability,
        sample_rate=0.1 if args.environment == "production" else 1.0,

        # Callbacks
        on_stage_complete=on_stage_complete_callback if args.verbose else None,
        on_error=on_error_callback,
    )


def on_stage_complete_callback(stage: PipelineStage, output: Dict[str, Any]) -> None:
    """Callback for stage completion.

    Args:
        stage: Completed stage
        output: Stage output
    """
    logger.info(f"=== Stage Complete: {stage.value} ===")

    # Log key metrics for each stage
    if stage == PipelineStage.SCOPE_DEFINITION:
        logger.info(f"  Experiment ID: {output.get('experiment_id')}")
        logger.info(f"  Problem Type: {output.get('scope_spec', {}).get('problem_type')}")

    elif stage == PipelineStage.DATA_PREPARATION:
        qc = output.get("qc_report", {})
        logger.info(f"  QC Score: {qc.get('overall_score', 0):.2f}")
        logger.info(f"  Row Count: {qc.get('row_count', 0)}")
        logger.info(f"  Gate Passed: {output.get('gate_passed', False)}")

    elif stage == PipelineStage.MODEL_SELECTION:
        candidate = output.get("model_candidate", {})
        logger.info(f"  Algorithm: {candidate.get('algorithm_name')}")
        logger.info(f"  Selection Score: {candidate.get('selection_score', 0):.3f}")

    elif stage == PipelineStage.MODEL_TRAINING:
        logger.info(f"  Model URI: {output.get('model_uri')}")
        logger.info(f"  Success Criteria Met: {output.get('success_criteria_met')}")
        val_metrics = output.get("validation_metrics", {})
        if "auc" in val_metrics:
            logger.info(f"  AUC-ROC: {val_metrics['auc']:.4f}")
        if "rmse" in val_metrics:
            logger.info(f"  RMSE: {val_metrics['rmse']:.4f}")

    elif stage == PipelineStage.FEATURE_ANALYSIS:
        analysis = output.get("shap_analysis", {})
        top_features = analysis.get("top_features", [])[:5]
        logger.info(f"  Top 5 Features: {', '.join(top_features)}")

    elif stage == PipelineStage.MODEL_DEPLOYMENT:
        logger.info(f"  Deployment Status: {output.get('status')}")
        logger.info(f"  Endpoint: {output.get('endpoint_url')}")


def on_error_callback(stage: PipelineStage, error: Exception) -> None:
    """Callback for pipeline errors.

    Args:
        stage: Stage where error occurred
        error: The exception
    """
    logger.error(f"Pipeline error at {stage.value}: {error}")


async def run_pipeline(args: argparse.Namespace) -> Optional[PipelineResult]:
    """Run the ML Foundation Pipeline.

    Args:
        args: Command line arguments

    Returns:
        PipelineResult if successful, None otherwise
    """
    # Get use case configuration
    use_case = USE_CASES.get(args.use_case)
    if not use_case:
        logger.error(f"Unknown use case: {args.use_case}")
        logger.info(f"Available use cases: {', '.join(USE_CASES.keys())}")
        return None

    # Create input data
    input_data = {
        **use_case,
        "brand": args.brand,
        "region": args.region,
        "algorithm_preferences": args.algorithms.split(",") if args.algorithms else None,
    }

    # Log configuration
    logger.info("=" * 60)
    logger.info("ML FOUNDATION PIPELINE - SAMPLE TRAINING")
    logger.info("=" * 60)
    logger.info(f"Use Case: {args.use_case}")
    logger.info(f"Brand: {args.brand}")
    logger.info(f"Region: {args.region}")
    logger.info(f"Environment: {args.environment}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - Pipeline will not execute actual agents")
        logger.info(f"Input data:\n{input_data}")
        return None

    # Create pipeline
    config = create_pipeline_config(args)
    pipeline = MLFoundationPipeline(config=config)

    # Run pipeline
    start_time = datetime.now()
    logger.info(f"Starting pipeline at {start_time.isoformat()}")

    result = await pipeline.run(input_data)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Log results
    logger.info("=" * 60)
    logger.info("PIPELINE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Status: {result.status}")
    logger.info(f"Duration: {duration:.2f}s")
    logger.info(f"Stages Completed: {', '.join(result.stages_completed)}")

    if result.status == "completed":
        logger.info("\n--- SUCCESS ---")
        if result.training_result:
            logger.info(f"Model URI: {result.training_result.get('model_uri')}")
            metrics = result.training_result.get("validation_metrics", {})
            for name, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {name}: {value:.4f}")

        if result.deployment_result:
            logger.info(f"Endpoint: {result.deployment_result.get('endpoint_url')}")

    elif result.status == "failed":
        logger.error("\n--- FAILED ---")
        for error in result.errors:
            logger.error(f"  Stage: {error.get('stage')}")
            logger.error(f"  Error: {error.get('error')}")
            logger.error(f"  Type: {error.get('error_type')}")

    # Log warnings
    if result.warnings:
        logger.warning("\nWarnings:")
        for warning in result.warnings:
            logger.warning(f"  - {warning}")

    return result


def print_summary(result: PipelineResult) -> None:
    """Print a formatted summary of pipeline results.

    Args:
        result: Pipeline result
    """
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Run ID: {result.pipeline_run_id}")
    print(f"Status: {result.status.upper()}")
    print(f"Duration: {result.total_duration_seconds:.2f}s")
    print("-" * 60)

    if result.scope_spec:
        print("\nSCOPE:")
        print(f"  Experiment: {result.experiment_id}")
        print(f"  Problem Type: {result.scope_spec.get('problem_type')}")
        print(f"  Target: {result.scope_spec.get('prediction_target')}")

    if result.qc_report:
        print("\nDATA QUALITY:")
        print(f"  Score: {result.qc_report.get('overall_score', 0):.2%}")
        print(f"  Rows: {result.qc_report.get('row_count', 0):,}")
        print(f"  QC Passed: {result.qc_report.get('qc_passed', False)}")

    if result.model_candidate:
        print("\nMODEL:")
        print(f"  Algorithm: {result.model_candidate.get('algorithm_name')}")
        print(f"  Family: {result.model_candidate.get('algorithm_family')}")

    if result.training_result:
        print("\nTRAINING:")
        metrics = result.training_result.get("validation_metrics", {})
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
        print(f"  Success Criteria Met: {result.training_result.get('success_criteria_met')}")

    if result.shap_analysis:
        print("\nFEATURE IMPORTANCE:")
        top_features = result.shap_analysis.get("top_features", [])[:5]
        for i, feature in enumerate(top_features, 1):
            print(f"  {i}. {feature}")

    if result.deployment_result:
        print("\nDEPLOYMENT:")
        print(f"  Status: {result.deployment_result.get('status')}")
        print(f"  Endpoint: {result.deployment_result.get('endpoint_url')}")

    print("=" * 60)


# =============================================================================
# CLI INTERFACE
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sample ML Foundation Pipeline - Train end-to-end ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train HCP conversion model with defaults
  python scripts/sample_ml_pipeline.py --use-case hcp_conversion

  # Train churn model for specific brand
  python scripts/sample_ml_pipeline.py --use-case churn --brand Remibrutinib

  # Quick test with sample data, no deployment
  python scripts/sample_ml_pipeline.py --use-case hcp_conversion --sample-data --skip-deployment

  # Production training with full HPO
  python scripts/sample_ml_pipeline.py --use-case trigger_effectiveness \\
      --environment production --hpo-trials 100

  # Dry run to see configuration
  python scripts/sample_ml_pipeline.py --use-case roi_prediction --dry-run

Available Use Cases:
  hcp_conversion        - Predict HCP conversion likelihood
  churn                 - Predict patient therapy discontinuation
  trigger_effectiveness - Predict marketing trigger success
  roi_prediction        - Predict ROI for budget allocation
        """,
    )

    # Required arguments
    parser.add_argument(
        "--use-case",
        type=str,
        required=True,
        choices=list(USE_CASES.keys()),
        help="Use case to run (required)",
    )

    # Business context
    parser.add_argument(
        "--brand",
        type=str,
        default="Remibrutinib",
        help="Brand to train model for (default: Remibrutinib)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="US",
        help="Region for model (default: US)",
    )

    # Environment
    parser.add_argument(
        "--environment",
        type=str,
        default="staging",
        choices=["staging", "production", "development"],
        help="Target deployment environment (default: staging)",
    )

    # Pipeline options
    parser.add_argument(
        "--skip-deployment",
        action="store_true",
        help="Skip model deployment stage",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP feature analysis",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip MLflow experiment tracking",
    )
    parser.add_argument(
        "--skip-observability",
        action="store_true",
        help="Skip Opik observability",
    )
    parser.add_argument(
        "--skip-hpo",
        action="store_true",
        help="Skip hyperparameter optimization",
    )
    parser.add_argument(
        "--skip-leakage-check",
        action="store_true",
        help="Skip data leakage validation",
    )

    # HPO options
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=50,
        help="Number of HPO trials (default: 50)",
    )
    parser.add_argument(
        "--hpo-timeout",
        type=float,
        default=2.0,
        help="HPO timeout in hours (default: 2.0)",
    )

    # Data options
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Use synthetic sample data instead of real data",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        help="Comma-separated list of preferred algorithms (e.g., XGBoost,LightGBM)",
    )

    # Output options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with stage callbacks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running pipeline",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_args()

    try:
        result = asyncio.run(run_pipeline(args))

        if result is None:
            # Dry run or error before execution
            return 0 if args.dry_run else 1

        # Print summary
        print_summary(result)

        # Return exit code based on status
        return 0 if result.status == "completed" else 1

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
