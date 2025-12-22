"""MLflow registrar for model_selector.

This module handles registration of model selection decisions
in MLflow for experiment tracking and model registry.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


async def register_selection_in_mlflow(state: Dict[str, Any]) -> Dict[str, Any]:
    """Register model selection decision in MLflow.

    Creates a run in MLflow to track:
    - Selected algorithm and configuration
    - Selection rationale and scores
    - Benchmark results
    - Alternative candidates considered

    Args:
        state: ModelSelectorState with selection details

    Returns:
        Dictionary with mlflow_run_id, registered_in_mlflow
    """
    experiment_id = state.get("experiment_id", "")
    primary_candidate = state.get("primary_candidate", {})
    selection_rationale = state.get("selection_rationale", "")
    benchmark_results = state.get("benchmark_results", {})

    if not primary_candidate:
        return {
            "registered_in_mlflow": False,
            "mlflow_registration_error": "No primary candidate to register",
        }

    try:
        from src.mlops.mlflow_connector import MLflowConnector, ModelStage

        connector = MLflowConnector()

        # Set or create experiment
        mlflow_experiment_id = await _ensure_experiment(connector, experiment_id)

        # Start run for model selection
        run = await connector.start_run(
            experiment_id=mlflow_experiment_id,
            run_name=f"model_selection_{primary_candidate['name']}",
            tags={
                "agent": "model_selector",
                "algorithm": primary_candidate["name"],
                "algorithm_family": primary_candidate.get("family", "unknown"),
                "selection_type": "automated",
            },
        )

        if not run:
            return {
                "registered_in_mlflow": False,
                "mlflow_registration_error": "Failed to start MLflow run",
            }

        run_id = run.info.run_id

        # Log parameters
        await _log_selection_params(connector, primary_candidate, state)

        # Log metrics
        await _log_selection_metrics(connector, primary_candidate, benchmark_results)

        # Log artifacts
        await _log_selection_artifacts(
            connector, selection_rationale, state.get("alternatives_considered", [])
        )

        # End run
        await connector.end_run()

        return {
            "registered_in_mlflow": True,
            "mlflow_run_id": run_id,
            "mlflow_experiment_id": mlflow_experiment_id,
            "stage": ModelStage.DEVELOPMENT.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "model_selector",
        }

    except ImportError:
        return {
            "registered_in_mlflow": False,
            "mlflow_registration_error": "MLflow connector not available",
        }
    except Exception as e:
        return {
            "registered_in_mlflow": False,
            "mlflow_registration_error": str(e),
        }


async def _ensure_experiment(
    connector: Any,
    experiment_id: str,
) -> Optional[str]:
    """Ensure MLflow experiment exists.

    Args:
        connector: MLflowConnector instance
        experiment_id: Experiment ID from scope

    Returns:
        MLflow experiment ID
    """
    experiment_name = f"e2i_model_selection_{experiment_id}"

    mlflow_experiment_id = await connector.get_or_create_experiment(
        name=experiment_name,
        tags={
            "source": "model_selector",
            "e2i_experiment_id": experiment_id,
        },
    )

    return mlflow_experiment_id


async def _log_selection_params(
    connector: Any,
    candidate: Dict[str, Any],
    state: Dict[str, Any],
) -> None:
    """Log selection parameters to MLflow.

    Args:
        connector: MLflowConnector instance
        candidate: Selected algorithm
        state: Full state
    """
    params = {
        "algorithm_name": candidate.get("name", "unknown"),
        "algorithm_family": candidate.get("family", "unknown"),
        "framework": candidate.get("framework", "unknown"),
        "problem_type": state.get("problem_type", "unknown"),
        "interpretability_required": str(state.get("interpretability_required", False)),
        "row_count": state.get("row_count", 0),
        "column_count": state.get("column_count", 0),
    }

    # Add default hyperparameters
    default_params = candidate.get("default_hyperparameters", {})
    for key, value in default_params.items():
        params[f"default_{key}"] = str(value)

    await connector.log_params(params)


async def _log_selection_metrics(
    connector: Any,
    candidate: Dict[str, Any],
    benchmark_results: Dict[str, Dict[str, Any]],
) -> None:
    """Log selection metrics to MLflow.

    Args:
        connector: MLflowConnector instance
        candidate: Selected algorithm
        benchmark_results: Benchmark results
    """
    algo_name = candidate.get("name", "")

    metrics = {
        "selection_score": candidate.get("selection_score", 0.0),
        "interpretability_score": candidate.get("interpretability_score", 0.0),
        "scalability_score": candidate.get("scalability_score", 0.0),
        "inference_latency_ms": float(candidate.get("inference_latency_ms", 0)),
        "memory_requirement_gb": float(candidate.get("memory_gb", 0)),
    }

    # Add benchmark metrics if available
    if algo_name in benchmark_results:
        bench = benchmark_results[algo_name]
        if "error" not in bench:
            metrics["benchmark_cv_mean"] = bench.get("cv_score_mean", 0.0)
            metrics["benchmark_cv_std"] = bench.get("cv_score_std", 0.0)
            metrics["benchmark_time_seconds"] = bench.get("training_time_seconds", 0.0)

    # Add combined score if available
    combined_score = candidate.get("combined_score")
    if combined_score is not None:
        metrics["combined_score"] = combined_score

    await connector.log_metrics(metrics)


async def _log_selection_artifacts(
    connector: Any,
    rationale: str,
    alternatives: list,
) -> None:
    """Log selection artifacts to MLflow.

    Args:
        connector: MLflowConnector instance
        rationale: Selection rationale text
        alternatives: List of alternatives considered
    """
    import json
    import tempfile
    import os

    # Log rationale as text artifact
    if rationale:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(rationale)
            rationale_path = f.name

        try:
            await connector.log_artifact(rationale_path, "selection_rationale.txt")
        finally:
            os.unlink(rationale_path)

    # Log alternatives as JSON artifact
    if alternatives:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(alternatives, f, indent=2, default=str)
            alternatives_path = f.name

        try:
            await connector.log_artifact(alternatives_path, "alternatives_considered.json")
        finally:
            os.unlink(alternatives_path)


async def log_benchmark_comparison(state: Dict[str, Any]) -> Dict[str, Any]:
    """Log detailed benchmark comparison to MLflow.

    Creates a comparison chart/table of all benchmarked algorithms.

    Args:
        state: ModelSelectorState with benchmark_results

    Returns:
        Dictionary with benchmark_logged status
    """
    benchmark_results = state.get("benchmark_results", {})
    mlflow_run_id = state.get("mlflow_run_id")

    if not benchmark_results or not mlflow_run_id:
        return {"benchmark_logged": False}

    try:
        from src.mlops.mlflow_connector import MLflowConnector

        connector = MLflowConnector()

        # Log each algorithm's benchmark as nested metrics
        for algo_name, results in benchmark_results.items():
            if "error" not in results:
                prefix = f"benchmark_{algo_name.lower()}"
                metrics = {
                    f"{prefix}_cv_mean": results.get("cv_score_mean", 0.0),
                    f"{prefix}_cv_std": results.get("cv_score_std", 0.0),
                    f"{prefix}_time": results.get("training_time_seconds", 0.0),
                }
                await connector.log_metrics(metrics)

        return {"benchmark_logged": True}

    except Exception as e:
        return {
            "benchmark_logged": False,
            "benchmark_log_error": str(e),
        }


async def create_selection_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of the selection decision for database storage.

    Prepares a structured summary that can be stored in the
    ml_model_selections table.

    Args:
        state: ModelSelectorState with all selection details

    Returns:
        Dictionary with selection_summary for database storage
    """
    primary = state.get("primary_candidate", {})
    alternatives = state.get("alternative_candidates", [])

    summary = {
        "experiment_id": state.get("experiment_id"),
        "algorithm_name": primary.get("name"),
        "algorithm_family": primary.get("family"),
        "algorithm_class": state.get("algorithm_class"),
        "selection_score": primary.get("selection_score"),
        "combined_score": primary.get("combined_score"),
        "benchmark_score": primary.get("benchmark_score"),
        "selection_rationale": state.get("selection_rationale"),
        "primary_reason": state.get("primary_reason"),
        "supporting_factors": state.get("supporting_factors", []),
        "default_hyperparameters": primary.get("default_hyperparameters", {}),
        "hyperparameter_search_space": primary.get("hyperparameter_space", {}),
        "alternatives_considered": [
            {
                "algorithm_name": alt.get("name"),
                "selection_score": alt.get("selection_score"),
                "reason_not_selected": alt.get("reason_not_selected"),
            }
            for alt in alternatives[:3]
        ],
        "benchmark_results": state.get("benchmark_results", {}),
        "historical_success_rates": state.get("historical_success_rates", {}),
        "constraint_compliance": state.get("constraint_compliance", {}),
        "mlflow_run_id": state.get("mlflow_run_id"),
        "mlflow_experiment_id": state.get("mlflow_experiment_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "model_selector",
    }

    return {
        "selection_summary": summary,
    }
