#!/usr/bin/env python3
"""GEPA Phase 2 Script - All Hybrid Agent Optimization.

This script runs GEPA optimization on all Hybrid agents in parallel:
1. causal_impact - DoWhy/EconML causal inference (Tier 2)
2. experiment_designer - A/B test design with Digital Twin (Tier 3)
3. feature_analyzer - SHAP-based feature analysis (Tier 0)

Hybrid agents benefit most from GEPA's joint tool optimization feature,
which evolves both instructions and tool descriptions together.

Usage:
    # Dry run all agents
    python scripts/gepa_phase2_hybrid.py --dry-run

    # Optimize all hybrid agents sequentially
    python scripts/gepa_phase2_hybrid.py --budget medium

    # Optimize specific agent only
    python scripts/gepa_phase2_hybrid.py --agent causal_impact --budget medium

    # Parallel optimization (requires more resources)
    python scripts/gepa_phase2_hybrid.py --parallel --budget medium

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Hybrid agents with their configurations
HYBRID_AGENTS = {
    "causal_impact": {
        "tier": 2,
        "description": "DoWhy/EconML causal inference",
        "default_budget": "medium",
        "enable_tool_optimization": True,
        "metric_class": "CausalImpactGEPAMetric",
    },
    "experiment_designer": {
        "tier": 3,
        "description": "A/B test design with Digital Twin pre-screening",
        "default_budget": "medium",
        "enable_tool_optimization": True,
        "metric_class": "ExperimentDesignerGEPAMetric",
    },
    "feature_analyzer": {
        "tier": 0,
        "description": "SHAP-based feature importance analysis",
        "default_budget": "medium",
        "enable_tool_optimization": False,
        "metric_class": "StandardAgentGEPAMetric",
    },
}


async def load_agent_training_data(
    agent_name: str,
    limit: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load training data for a specific agent.

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
        activities = await repo.get_activities_with_outcomes(
            agent_name=agent_name,
            limit=limit,
        )

        if activities and len(activities) >= 20:
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
            logger.info(f"Loaded {len(trainset)} train, {len(valset)} val from database")
            return trainset, valset

    except Exception as e:
        logger.warning(f"Could not load from database: {e}")

    # Generate synthetic examples
    return _generate_agent_examples(agent_name)


def _generate_agent_examples(
    agent_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate synthetic examples for an agent.

    Args:
        agent_name: Name of the agent

    Returns:
        Tuple of (trainset, valset)
    """
    import random

    examples = []

    if agent_name == "causal_impact":
        for i in range(50):
            examples.append(
                {
                    "question": f"What caused the change in TRx for brand {i % 3}?",
                    "context": {"brand": ["Remibrutinib", "Fabhalta", "Kisqali"][i % 3]},
                    "ground_truth": {"ate": round(random.uniform(0.05, 0.25), 3)},
                }
            )
    elif agent_name == "experiment_designer":
        for i in range(50):
            examples.append(
                {
                    "question": f"Design an A/B test for {['email', 'call', 'webinar'][i % 3]} intervention",
                    "context": {"target_metric": "conversion_rate"},
                    "ground_truth": {"power": 0.8, "sample_size": 1000 + i * 10},
                }
            )
    elif agent_name == "feature_analyzer":
        for i in range(50):
            examples.append(
                {
                    "question": f"What features drive {['churn', 'conversion', 'engagement'][i % 3]}?",
                    "context": {"model_id": f"model_{i}"},
                    "ground_truth": {"top_features": ["feature_1", "feature_2"]},
                }
            )
    else:
        # Generic examples
        for i in range(50):
            examples.append(
                {
                    "question": f"Query {i} for {agent_name}",
                    "context": {},
                    "ground_truth": {},
                }
            )

    split_idx = int(len(examples) * 0.8)
    return examples[:split_idx], examples[split_idx:]


async def optimize_agent(
    agent_name: str,
    budget: str = "medium",
    dry_run: bool = False,
    track_mlflow: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Optimize a single hybrid agent.

    Args:
        agent_name: Name of the agent to optimize
        budget: GEPA budget preset
        dry_run: If True, validate only
        track_mlflow: Enable MLflow tracking
        seed: Random seed

    Returns:
        Dict with optimization results
    """
    config = HYBRID_AGENTS.get(agent_name)
    if not config:
        return {"agent_name": agent_name, "status": "error", "error": "Unknown agent"}

    logger.info(f"Optimizing {agent_name} (Tier {config['tier']})")

    results: dict[str, Any] = {
        "agent_name": agent_name,
        "tier": config["tier"],
        "budget": budget,
        "dry_run": dry_run,
        "started_at": datetime.now().isoformat(),
        "status": "pending",
    }

    try:
        # Load training data
        trainset, valset = await load_agent_training_data(agent_name)
        results["trainset_size"] = len(trainset)
        results["valset_size"] = len(valset)

        if dry_run:
            logger.info(f"Dry run for {agent_name} - validating setup")

            from src.optimization.gepa import get_metric_for_agent

            metric = get_metric_for_agent(agent_name)
            results["status"] = "validated"
            results["metric_class"] = type(metric).__name__
            return results

        # Full optimization
        from src.optimization.gepa import (
            create_optimizer_for_agent,
            save_optimized_module,
        )

        # Get agent module
        student = _get_agent_module(agent_name)

        # Create and run optimizer
        optimizer = create_optimizer_for_agent(
            agent_name=agent_name,
            trainset=trainset,
            valset=valset,
            budget=budget,
            seed=seed,
        )

        start_time = datetime.now()
        optimized = optimizer.compile(student, trainset=trainset)
        elapsed = (datetime.now() - start_time).total_seconds()

        results["status"] = "completed"
        results["elapsed_seconds"] = elapsed
        results["completed_at"] = datetime.now().isoformat()

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

        return results

    except Exception as e:
        logger.error(f"Failed to optimize {agent_name}: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        return results


def _get_agent_module(agent_name: str) -> Any:
    """Get the DSPy module for an agent.

    Args:
        agent_name: Name of the agent

    Returns:
        DSPy module for the agent
    """
    if agent_name == "causal_impact":
        from src.agents.causal_impact.dspy_integration import get_causal_impact_module

        return get_causal_impact_module()
    elif agent_name == "experiment_designer":
        from src.agents.experiment_designer.dspy_integration import (
            get_experiment_designer_module,
        )

        return get_experiment_designer_module()
    elif agent_name == "feature_analyzer":
        from src.agents.tier_0.feature_analyzer.dspy_integration import (
            get_feature_analyzer_module,
        )

        return get_feature_analyzer_module()
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


async def run_phase2(
    agents: list[str] | None = None,
    budget: str = "medium",
    dry_run: bool = False,
    parallel: bool = False,
    track_mlflow: bool = False,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Run Phase 2 optimization on hybrid agents.

    Args:
        agents: List of agent names (None = all hybrid agents)
        budget: GEPA budget preset
        dry_run: If True, validate only
        parallel: Run agents in parallel
        track_mlflow: Enable MLflow tracking
        seed: Random seed

    Returns:
        Dict mapping agent names to their results
    """
    if agents is None:
        agents = list(HYBRID_AGENTS.keys())

    logger.info(f"Phase 2: Optimizing {len(agents)} hybrid agents")
    logger.info(f"Agents: {agents}")
    logger.info(f"Mode: {'parallel' if parallel else 'sequential'}")

    all_results: dict[str, dict[str, Any]] = {}

    if parallel:
        # Run all agents concurrently
        tasks = [
            optimize_agent(
                agent_name=agent,
                budget=budget,
                dry_run=dry_run,
                track_mlflow=track_mlflow,
                seed=seed,
            )
            for agent in agents
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            all_results[result["agent_name"]] = result
    else:
        # Run sequentially
        for agent in agents:
            result = await optimize_agent(
                agent_name=agent,
                budget=budget,
                dry_run=dry_run,
                track_mlflow=track_mlflow,
                seed=seed,
            )
            all_results[agent] = result

    return all_results


def main() -> None:
    """Main entry point for Phase 2 hybrid optimization."""
    parser = argparse.ArgumentParser(
        description="GEPA Phase 2 - Hybrid Agent Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all hybrid agents
  python scripts/gepa_phase2_hybrid.py --dry-run

  # Optimize all hybrid agents
  python scripts/gepa_phase2_hybrid.py --budget medium

  # Optimize specific agent
  python scripts/gepa_phase2_hybrid.py --agent causal_impact

  # Parallel optimization
  python scripts/gepa_phase2_hybrid.py --parallel --budget medium
        """,
    )

    parser.add_argument(
        "--agent",
        choices=list(HYBRID_AGENTS.keys()),
        help="Specific agent to optimize (default: all)",
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
        "--parallel",
        action="store_true",
        help="Run agents in parallel (requires more resources)",
    )

    parser.add_argument(
        "--track-mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Determine which agents to optimize
    agents = [args.agent] if args.agent else None

    # Run Phase 2
    results = asyncio.run(
        run_phase2(
            agents=agents,
            budget=args.budget,
            dry_run=args.dry_run,
            parallel=args.parallel,
            track_mlflow=args.track_mlflow,
            seed=args.seed,
        )
    )

    # Print summary
    print("\n" + "=" * 70)
    print("GEPA PHASE 2 - HYBRID AGENT OPTIMIZATION RESULTS")
    print("=" * 70)

    for agent_name, result in results.items():
        status = result.get("status", "unknown")
        status_icon = {"completed": "[OK]", "validated": "[OK]", "failed": "[X]"}.get(
            status, "[?]"
        )
        print(f"\n{status_icon} {agent_name} (Tier {result.get('tier', '?')})")
        print(f"    Status: {status}")
        if "best_score" in result:
            print(f"    Best Score: {result['best_score']:.4f}")
        if "elapsed_seconds" in result:
            print(f"    Time: {result['elapsed_seconds']:.1f}s")
        if "error" in result:
            print(f"    Error: {result['error']}")

    print("\n" + "=" * 70)

    # Exit with error if any failed
    if any(r.get("status") == "failed" for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
