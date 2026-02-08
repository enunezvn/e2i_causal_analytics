"""GEPA MLflow Integration for Experiment Tracking.

This module provides MLflow integration for GEPA optimization runs,
enabling experiment tracking, metric logging, and model artifact storage.

Integrates with:
- E2I MLflow connector (src/mlops/mlflow_connector.py)
- GEPA optimizer runs
- Agent optimization experiments

Usage:
    from src.optimization.gepa.integration import GEPAMLflowCallback, log_optimization_run

    # Use as callback during optimization
    callback = GEPAMLflowCallback(experiment_name="causal_impact_gepa")
    optimizer.compile(student, trainset, callbacks=[callback])

    # Or log a completed run
    await log_optimization_run(
        agent_name="causal_impact",
        best_score=0.85,
        generations=5,
        pareto_frontier=[...],
    )

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, cast

logger = logging.getLogger(__name__)


@dataclass
class GEPAMLflowCallback:
    """MLflow callback for GEPA optimization runs.

    Tracks:
    - Optimization parameters (budget, generations, seed)
    - Per-generation metrics (best score, Pareto frontier size)
    - Final optimized instructions (as artifacts)
    - Tool descriptions (if tool optimization enabled)

    Example:
        >>> callback = GEPAMLflowCallback(
        ...     experiment_name="causal_impact_gepa",
        ...     run_name="v1_pilot",
        ... )
        >>> # Called automatically by GEPA during optimization
        >>> callback.on_optimization_start(config)
        >>> callback.on_generation_complete(gen_num, metrics)
        >>> callback.on_optimization_complete(results)
    """

    experiment_name: str = "gepa_optimization"
    run_name: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)
    log_instructions: bool = True
    log_tool_descriptions: bool = True

    # Internal state
    _run_id: Optional[str] = None
    _experiment_id: Optional[str] = None
    _mlflow_connector: Any = None
    _started: bool = False

    def __post_init__(self) -> None:
        """Initialize MLflow connector."""
        try:
            from src.mlops.mlflow_connector import get_mlflow_connector

            self._mlflow_connector = get_mlflow_connector()
            logger.debug("GEPAMLflowCallback initialized with MLflow connector")
        except ImportError:
            logger.warning("MLflow connector not available")
            self._mlflow_connector = None

    @property
    def enabled(self) -> bool:
        """Check if MLflow is enabled."""
        return self._mlflow_connector is not None and self._mlflow_connector.enabled

    async def on_optimization_start(
        self,
        agent_name: str,
        budget: str,
        max_metric_calls: int,
        trainset_size: int,
        valset_size: int,
        enable_tool_optimization: bool = False,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Called when GEPA optimization starts.

        Args:
            agent_name: Name of the agent being optimized
            budget: Budget preset (light, medium, heavy)
            max_metric_calls: Maximum metric calls allowed
            trainset_size: Size of training set
            valset_size: Size of validation set
            enable_tool_optimization: Whether tool optimization is enabled
            seed: Random seed
            **kwargs: Additional parameters to log
        """
        if not self.enabled:
            return

        try:
            # Create or get experiment
            self._experiment_id = await self._mlflow_connector.get_or_create_experiment(
                name=self.experiment_name,
                tags={
                    "agent": agent_name,
                    "optimizer": "gepa",
                    **self.tags,
                },
            )

            # Generate run name if not provided
            run_name = self.run_name or f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Start run
            async with self._mlflow_connector.start_run(
                experiment_id=self._experiment_id,
                run_name=run_name,
                tags={
                    "agent_name": agent_name,
                    "budget": budget,
                    "tool_optimization": str(enable_tool_optimization),
                },
            ) as run:
                self._run_id = run.run_id

                # Log parameters
                await run.log_params(
                    {
                        "agent_name": agent_name,
                        "budget": budget,
                        "max_metric_calls": max_metric_calls,
                        "trainset_size": trainset_size,
                        "valset_size": valset_size,
                        "enable_tool_optimization": enable_tool_optimization,
                        "seed": seed,
                        **{k: str(v) for k, v in kwargs.items()},
                    }
                )

            self._started = True
            logger.info(f"GEPA MLflow run started: {self._run_id}")

        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            self._started = False

    async def on_generation_complete(
        self,
        generation: int,
        best_score: float,
        pareto_size: int,
        candidate_count: int,
        metric_calls: int,
        elapsed_seconds: float,
        **kwargs: Any,
    ) -> None:
        """Called after each generation completes.

        Args:
            generation: Generation number (0-indexed)
            best_score: Best score achieved so far
            pareto_size: Number of candidates on Pareto frontier
            candidate_count: Total candidates evaluated
            metric_calls: Total metric calls so far
            elapsed_seconds: Time elapsed since start
            **kwargs: Additional metrics to log
        """
        if not self.enabled or not self._started:
            return

        try:
            async with self._mlflow_connector.start_run(
                experiment_id=self._experiment_id,
                run_name=self.run_name,
                nested=True,
            ) as run:
                await run.log_metrics(
                    {
                        "best_score": best_score,
                        "pareto_size": pareto_size,
                        "candidate_count": candidate_count,
                        "metric_calls": metric_calls,
                        "elapsed_seconds": elapsed_seconds,
                        **kwargs,
                    },
                    step=generation,
                )

            logger.debug(f"GEPA generation {generation} logged: score={best_score:.4f}")

        except Exception as e:
            logger.warning(f"Failed to log generation metrics: {e}")

    async def on_optimization_complete(
        self,
        best_score: float,
        total_generations: int,
        total_metric_calls: int,
        total_seconds: float,
        optimized_instructions: Optional[str] = None,
        optimized_tools: Optional[list[dict[str, Any]]] = None,
        pareto_frontier: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when GEPA optimization completes.

        Args:
            best_score: Final best score
            total_generations: Total generations run
            total_metric_calls: Total metric evaluations
            total_seconds: Total optimization time
            optimized_instructions: Optimized instruction text
            optimized_tools: Optimized tool descriptions (if enabled)
            pareto_frontier: Final Pareto frontier candidates
            **kwargs: Additional final metrics
        """
        if not self.enabled or not self._started:
            return

        try:
            async with self._mlflow_connector.start_run(
                experiment_id=self._experiment_id,
                run_name=self.run_name,
                nested=True,
            ) as run:
                # Log final metrics
                await run.log_metrics(
                    {
                        "final_best_score": best_score,
                        "total_generations": total_generations,
                        "total_metric_calls": total_metric_calls,
                        "total_seconds": total_seconds,
                        "improvement_rate": best_score / total_metric_calls
                        if total_metric_calls > 0
                        else 0,
                        **kwargs,
                    }
                )

                # Log optimized instructions as artifact
                if self.log_instructions and optimized_instructions:
                    import os
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix="_instructions.txt", delete=False
                    ) as f:
                        f.write(optimized_instructions)
                        temp_path = f.name

                    try:
                        await run.log_artifact(temp_path, "optimized_instructions")
                    finally:
                        os.unlink(temp_path)

                # Log optimized tool descriptions
                if self.log_tool_descriptions and optimized_tools:
                    import json
                    import os
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix="_tools.json", delete=False
                    ) as f:
                        json.dump(optimized_tools, f, indent=2)
                        temp_path = f.name

                    try:
                        await run.log_artifact(temp_path, "optimized_tools")
                    finally:
                        os.unlink(temp_path)

                # Log Pareto frontier
                if pareto_frontier:
                    import json
                    import os
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix="_pareto.json", delete=False
                    ) as f:
                        json.dump(pareto_frontier, f, indent=2)
                        temp_path = f.name

                    try:
                        await run.log_artifact(temp_path, "pareto_frontier")
                    finally:
                        os.unlink(temp_path)

            logger.info(
                f"GEPA optimization complete: score={best_score:.4f}, "
                f"generations={total_generations}, time={total_seconds:.1f}s"
            )

        except Exception as e:
            logger.error(f"Failed to log optimization results: {e}")


async def log_optimization_run(
    agent_name: str,
    best_score: float,
    generations: int,
    metric_calls: int,
    elapsed_seconds: float,
    budget: str = "medium",
    pareto_frontier: Optional[list[dict[str, Any]]] = None,
    experiment_name: str = "gepa_optimization",
    tags: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Convenience function to log a completed GEPA optimization run.

    Use this for one-shot logging after optimization completes,
    rather than using the callback during optimization.

    Args:
        agent_name: Name of the optimized agent
        best_score: Best achieved score
        generations: Total generations run
        metric_calls: Total metric evaluations
        elapsed_seconds: Total time in seconds
        budget: Budget preset used
        pareto_frontier: Final Pareto frontier (optional)
        experiment_name: MLflow experiment name
        tags: Additional tags

    Returns:
        Run ID if successful, None otherwise

    Example:
        >>> run_id = await log_optimization_run(
        ...     agent_name="causal_impact",
        ...     best_score=0.85,
        ...     generations=5,
        ...     metric_calls=500,
        ...     elapsed_seconds=900.0,
        ... )
    """
    try:
        from src.mlops.mlflow_connector import get_mlflow_connector

        connector = get_mlflow_connector()
        if not connector.enabled:
            logger.warning("MLflow not enabled, skipping optimization logging")
            return None

        # Get or create experiment
        experiment_id = await connector.get_or_create_experiment(
            name=experiment_name,
            tags={
                "agent": agent_name,
                "optimizer": "gepa",
                **(tags or {}),
            },
        )

        run_name = f"{agent_name}_gepa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        async with connector.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags={
                "agent_name": agent_name,
                "budget": budget,
            },
        ) as run:
            # Log parameters
            await run.log_params(
                {
                    "agent_name": agent_name,
                    "budget": budget,
                    "optimizer": "gepa",
                }
            )

            # Log metrics
            await run.log_metrics(
                {
                    "best_score": best_score,
                    "total_generations": generations,
                    "total_metric_calls": metric_calls,
                    "total_seconds": elapsed_seconds,
                    "pareto_size": len(pareto_frontier) if pareto_frontier else 0,
                }
            )

            # Log Pareto frontier as artifact
            if pareto_frontier:
                import json
                import os
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix="_pareto.json", delete=False
                ) as f:
                    json.dump(pareto_frontier, f, indent=2)
                    temp_path = f.name

                try:
                    await run.log_artifact(temp_path, "pareto_frontier")
                finally:
                    os.unlink(temp_path)

            logger.info(f"Logged GEPA run: {run.run_id}")
            return cast(str, run.run_id)

    except Exception as e:
        logger.error(f"Failed to log optimization run: {e}")
        return None


__all__ = [
    "GEPAMLflowCallback",
    "log_optimization_run",
]
