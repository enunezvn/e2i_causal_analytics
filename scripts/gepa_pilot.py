#!/usr/bin/env python3
"""GEPA Pilot Script - Phase 1: Causal Impact Agent Optimization.

This script runs GEPA optimization on the Causal Impact agent as the initial
pilot for validating the GEPA migration. Causal Impact is chosen because:
1. It's a Hybrid agent with clear optimization objectives
2. Has well-defined DoWhy/EconML tool optimization targets
3. Medium budget allows meaningful optimization in reasonable time

Usage:
    # Dry run (no actual optimization, just validation)
    python scripts/gepa_pilot.py --dry-run

    # Light optimization (quick test)
    python scripts/gepa_pilot.py --budget light

    # Medium optimization (recommended for pilot)
    python scripts/gepa_pilot.py --budget medium

    # Full optimization with MLflow tracking
    python scripts/gepa_pilot.py --budget medium --track-mlflow

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def load_training_data(
    agent_name: str,
    limit: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load training and validation data for the agent.

    Args:
        agent_name: Name of the agent
        limit: Maximum examples to load

    Returns:
        Tuple of (trainset, valset)
    """
    logger.info(f"Loading training data for {agent_name}...")

    # Try to load from database
    try:
        from src.repositories.agent_activity import AgentActivityRepository

        repo = AgentActivityRepository()

        # Get agent activities with outcomes for training
        activities = await repo.get_activities_with_outcomes(
            agent_name=agent_name,
            limit=limit,
        )

        if activities and len(activities) >= 20:
            # Split 80/20 for train/val
            split_idx = int(len(activities) * 0.8)
            trainset = [
                {
                    "question": a.get("query", ""),
                    "context": a.get("context", {}),
                    "ground_truth": a.get("outcome", {}),
                }
                for a in activities[:split_idx]
            ]
            valset = [
                {
                    "question": a.get("query", ""),
                    "context": a.get("context", {}),
                    "ground_truth": a.get("outcome", {}),
                }
                for a in activities[split_idx:]
            ]
            logger.info(f"Loaded {len(trainset)} train, {len(valset)} val examples from database")
            return trainset, valset

    except Exception as e:
        logger.warning(f"Could not load from database: {e}")

    # Fall back to synthetic examples
    logger.info("Using synthetic training examples")
    trainset = _generate_synthetic_examples(agent_name, count=40)
    valset = _generate_synthetic_examples(agent_name, count=10)
    return trainset, valset


def _generate_synthetic_examples(agent_name: str, count: int) -> list[dict[str, Any]]:
    """Generate synthetic training examples for testing.

    Args:
        agent_name: Name of the agent
        count: Number of examples to generate

    Returns:
        List of synthetic training examples
    """
    examples = []

    if agent_name == "causal_impact":
        templates = [
            {
                "question": "What is the causal impact of {treatment} on {outcome}?",
                "treatment": ["email campaigns", "sales calls", "webinars", "samples"],
                "outcome": ["TRx", "NRx", "market share", "conversion rate"],
            },
            {
                "question": "How did {intervention} affect {metric} in {region}?",
                "intervention": ["Q1 campaign", "new messaging", "digital outreach"],
                "metric": ["prescriptions", "HCP engagement", "brand awareness"],
                "region": ["Northeast", "South", "Midwest", "West"],
            },
        ]

        import random

        for i in range(count):
            template = random.choice(templates)
            question = template["question"]
            for key, values in template.items():
                if key != "question":
                    question = question.replace(f"{{{key}}}", random.choice(values))

            examples.append(
                {
                    "question": question,
                    "context": {
                        "brand": random.choice(["Remibrutinib", "Fabhalta", "Kisqali"]),
                        "time_period": random.choice(["Q1", "Q2", "Q3", "Q4"]),
                    },
                    "ground_truth": {
                        "ate": round(random.uniform(0.05, 0.25), 3),
                        "confidence": round(random.uniform(0.8, 0.99), 2),
                    },
                }
            )

    return examples


async def run_pilot(
    budget: str = "medium",
    dry_run: bool = False,
    track_mlflow: bool = False,
    track_opik: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Run GEPA pilot optimization on Causal Impact agent.

    Args:
        budget: GEPA budget preset (light, medium, heavy)
        dry_run: If True, validate setup without running optimization
        track_mlflow: Enable MLflow experiment tracking
        track_opik: Enable Opik observability tracing
        seed: Random seed for reproducibility

    Returns:
        Dict with optimization results
    """
    agent_name = "causal_impact"
    logger.info(f"Starting GEPA pilot for {agent_name}")
    logger.info(f"Budget: {budget}, Dry run: {dry_run}")

    results: dict[str, Any] = {
        "agent_name": agent_name,
        "budget": budget,
        "dry_run": dry_run,
        "started_at": datetime.now().isoformat(),
        "status": "pending",
    }

    try:
        # Load training data
        trainset, valset = await load_training_data(agent_name)
        results["trainset_size"] = len(trainset)
        results["valset_size"] = len(valset)

        if dry_run:
            logger.info("Dry run mode - validating setup only")

            # Validate GEPA imports
            from src.optimization.gepa import (
                CausalImpactGEPAMetric,
                create_gepa_optimizer,
                get_metric_for_agent,
            )

            # Validate metric
            metric = get_metric_for_agent(agent_name)
            logger.info(f"Metric loaded: {type(metric).__name__}")

            # Validate we can create optimizer config
            logger.info("GEPA setup validated successfully")
            results["status"] = "validated"
            results["validation"] = {
                "metric_class": type(metric).__name__,
                "trainset_loaded": len(trainset) > 0,
                "valset_loaded": len(valset) > 0,
            }
            return results

        # Full optimization run
        from src.optimization.gepa import (
            create_gepa_optimizer,
            get_metric_for_agent,
            save_optimized_module,
        )
        from src.optimization.gepa.integration import GEPAMLflowCallback, GEPAOpikTracer

        # Get the agent's DSPy module
        from src.agents.causal_impact.dspy_integration import get_causal_impact_module

        student = get_causal_impact_module()

        # Create metric
        metric = get_metric_for_agent(agent_name)

        # Create optimizer
        optimizer = create_gepa_optimizer(
            metric=metric,
            trainset=trainset,
            valset=valset,
            budget=budget,
            enable_tool_optimization=True,
            seed=seed,
        )

        # Set up callbacks
        callbacks = []

        if track_mlflow:
            mlflow_callback = GEPAMLflowCallback(
                experiment_name=f"gepa_pilot_{agent_name}",
                run_name=f"pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            callbacks.append(mlflow_callback)

        # Set up Opik tracer for observability
        opik_tracer = None
        if track_opik:
            opik_tracer = GEPAOpikTracer(
                project_name=f"gepa_pilot_{agent_name}",
                tags={
                    "pilot": "true",
                    "budget": budget,
                },
            )
            if opik_tracer.enabled:
                logger.info("Opik tracing enabled for optimization run")
            else:
                logger.warning("Opik tracer created but connector not available")

        # Run optimization
        logger.info("Starting GEPA optimization...")
        start_time = datetime.now()

        # Run with Opik tracing if enabled
        if opik_tracer and opik_tracer.enabled:
            async with opik_tracer.trace_run(
                agent_name=agent_name,
                budget=budget,
                enable_tool_optimization=True,
            ):
                optimized = optimizer.compile(
                    student,
                    trainset=trainset,
                )
                elapsed = (datetime.now() - start_time).total_seconds()

                # Log completion to Opik
                await opik_tracer.log_optimization_complete(
                    best_score=getattr(optimizer, "best_score", 0.0),
                    total_generations=getattr(optimizer, "total_generations", 0),
                    total_metric_calls=getattr(optimizer, "total_metric_calls", 0),
                    total_seconds=elapsed,
                )
        else:
            optimized = optimizer.compile(
                student,
                trainset=trainset,
            )
            elapsed = (datetime.now() - start_time).total_seconds()

        # Extract results
        results["status"] = "completed"
        results["elapsed_seconds"] = elapsed
        results["completed_at"] = datetime.now().isoformat()
        results["opik_enabled"] = track_opik and opik_tracer and opik_tracer.enabled

        # Get final score if available
        if hasattr(optimizer, "best_score"):
            results["best_score"] = optimizer.best_score

        # Save optimized module
        version_id = await save_optimized_module(
            agent_name=agent_name,
            optimized_module=optimized,
            budget=budget,
            score=results.get("best_score", 0.0),
        )
        results["version_id"] = version_id

        logger.info(f"Optimization complete: {results}")
        return results

    except Exception as e:
        logger.error(f"Pilot failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        return results


def main() -> None:
    """Main entry point for GEPA pilot script."""
    parser = argparse.ArgumentParser(
        description="GEPA Pilot - Causal Impact Agent Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate setup (no optimization)
  python scripts/gepa_pilot.py --dry-run

  # Quick pilot run
  python scripts/gepa_pilot.py --budget light

  # Full pilot with tracking
  python scripts/gepa_pilot.py --budget medium --track-mlflow
        """,
    )

    parser.add_argument(
        "--budget",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="GEPA budget preset (default: medium)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running optimization",
    )

    parser.add_argument(
        "--track-mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )

    parser.add_argument(
        "--track-opik",
        action="store_true",
        help="Enable Opik observability tracing",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Run pilot
    results = asyncio.run(
        run_pilot(
            budget=args.budget,
            dry_run=args.dry_run,
            track_mlflow=args.track_mlflow,
            track_opik=args.track_opik,
            seed=args.seed,
        )
    )

    # Print summary
    print("\n" + "=" * 60)
    print("GEPA PILOT RESULTS")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # Exit with appropriate code
    if results.get("status") == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
