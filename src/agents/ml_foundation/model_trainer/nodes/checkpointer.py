"""Model checkpointing for model_trainer.

This module saves trained models to disk with metadata for:
- Model persistence and recovery
- Deployment artifact generation
- Model versioning

Version: 1.0.0
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = os.environ.get(
    "E2I_CHECKPOINT_DIR",
    "/tmp/e2i_model_checkpoints",
)


async def save_checkpoint(state: Dict[str, Any]) -> Dict[str, Any]:
    """Save trained model to checkpoint.

    Saves the trained model along with metadata including:
    - Hyperparameters
    - Training configuration
    - Evaluation metrics
    - Model signature

    Args:
        state: ModelTrainerState with trained_model and metadata

    Returns:
        Dictionary with checkpoint_path, checkpoint_metadata_path,
        checkpoint_status, model_hash
    """
    # Check if checkpointing is enabled
    enable_checkpointing = state.get("enable_checkpointing", True)

    if not enable_checkpointing:
        logger.info("Model checkpointing disabled")
        return {
            "checkpoint_status": "disabled",
            "checkpoint_path": None,
        }

    # Check if we have a trained model
    trained_model = state.get("trained_model")
    if trained_model is None:
        logger.warning("No trained model to checkpoint")
        return {
            "checkpoint_status": "skipped",
            "checkpoint_path": None,
            "error": "No trained model available",
        }

    # Extract configuration
    experiment_id = state.get("experiment_id", "unknown")
    algorithm_name = state.get("algorithm_name", "unknown")
    state.get("problem_type", "unknown")
    framework = state.get("framework", _get_framework(algorithm_name))

    # Checkpoint directory
    checkpoint_dir = state.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)
    checkpoint_dir = Path(checkpoint_dir)

    try:
        # Create checkpoint directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Generate checkpoint name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{algorithm_name.lower()}_{experiment_id}_{timestamp}"
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pkl"
        metadata_path = checkpoint_dir / f"{checkpoint_name}_metadata.json"

        # Save model
        model_hash = _save_model(trained_model, checkpoint_path, framework)

        # Prepare metadata
        metadata = _prepare_metadata(state, checkpoint_name, model_hash)

        # Save metadata
        _save_metadata(metadata, metadata_path)

        logger.info(f"Model checkpointed to: {checkpoint_path}")

        return {
            "checkpoint_status": "success",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_metadata_path": str(metadata_path),
            "checkpoint_name": checkpoint_name,
            "model_hash": model_hash,
            "checkpoint_timestamp": timestamp,
        }

    except Exception as e:
        logger.error(f"Checkpointing failed: {e}")
        return {
            "checkpoint_status": "failed",
            "checkpoint_path": None,
            "error": f"Checkpointing failed: {e}",
        }


async def load_checkpoint(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load model from checkpoint.

    Args:
        state: State with checkpoint_path or checkpoint_name

    Returns:
        Dictionary with loaded_model, checkpoint_metadata, load_status
    """
    checkpoint_path = state.get("checkpoint_path")
    checkpoint_name = state.get("checkpoint_name")
    checkpoint_dir = state.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)

    # Resolve checkpoint path
    if checkpoint_path is None and checkpoint_name:
        checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_name}.pkl"
    elif checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
    else:
        logger.error("No checkpoint path or name provided")
        return {
            "load_status": "failed",
            "error": "No checkpoint path or name provided",
        }

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return {
            "load_status": "failed",
            "error": f"Checkpoint not found: {checkpoint_path}",
        }

    try:
        # Load model
        model = _load_model(checkpoint_path)

        # Load metadata
        metadata_path = checkpoint_path.with_name(checkpoint_path.stem + "_metadata.json")
        metadata = _load_metadata(metadata_path) if metadata_path.exists() else {}

        # Verify hash if available
        if metadata.get("model_hash"):
            current_hash = _compute_model_hash(model)
            if current_hash != metadata["model_hash"]:
                logger.warning(
                    f"Model hash mismatch: expected {metadata['model_hash']}, got {current_hash}"
                )

        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")

        return {
            "load_status": "success",
            "loaded_model": model,
            "checkpoint_metadata": metadata,
            "checkpoint_path": str(checkpoint_path),
        }

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return {
            "load_status": "failed",
            "error": f"Failed to load checkpoint: {e}",
        }


def _save_model(model: Any, path: Path, framework: str) -> str:
    """Save model to disk.

    Args:
        model: Trained model object
        path: Output path
        framework: ML framework

    Returns:
        Model hash for verification
    """
    import joblib

    # Use joblib for sklearn-compatible models
    joblib.dump(model, path)

    # Compute hash
    return _compute_model_hash(model)


def _load_model(path: Path) -> Any:
    """Load model from disk.

    Args:
        path: Checkpoint path

    Returns:
        Loaded model
    """
    import joblib

    return joblib.load(path)


def _compute_model_hash(model: Any) -> str:
    """Compute hash of model for verification.

    Args:
        model: Model object

    Returns:
        SHA256 hash string
    """
    import pickle

    try:
        model_bytes = pickle.dumps(model)
        return hashlib.sha256(model_bytes).hexdigest()[:16]
    except Exception:
        return "unknown"


def _prepare_metadata(
    state: Dict[str, Any],
    checkpoint_name: str,
    model_hash: str,
) -> Dict[str, Any]:
    """Prepare checkpoint metadata.

    Args:
        state: Training state
        checkpoint_name: Checkpoint name
        model_hash: Model hash

    Returns:
        Metadata dictionary
    """
    # Extract evaluation metrics
    evaluation_metrics = state.get("evaluation_metrics", {})
    test_metrics = evaluation_metrics.get("test_metrics", {})
    validation_metrics = evaluation_metrics.get("validation_metrics", {})

    return {
        # Identification
        "checkpoint_name": checkpoint_name,
        "model_hash": model_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Configuration
        "experiment_id": state.get("experiment_id"),
        "algorithm_name": state.get("algorithm_name"),
        "problem_type": state.get("problem_type"),
        "framework": state.get("framework"),
        # Hyperparameters
        "best_hyperparameters": state.get("best_hyperparameters", {}),
        # HPO info
        "hpo_completed": state.get("hpo_completed", False),
        "hpo_best_value": state.get("hpo_best_value"),
        "hpo_trials_run": state.get("hpo_trials_run"),
        # Training info
        "training_duration_seconds": state.get("training_duration_seconds"),
        "early_stopped": state.get("early_stopped", False),
        "final_epoch": state.get("final_epoch"),
        # Key metrics
        "test_metrics": _filter_serializable(test_metrics),
        "validation_metrics": _filter_serializable(validation_metrics),
        # Model info
        "success_criteria_met": state.get("success_criteria_met", False),
        "optimal_threshold": state.get("optimal_threshold"),
        # MLflow reference
        "mlflow_run_id": state.get("mlflow_run_id"),
        "mlflow_experiment_id": state.get("mlflow_experiment_id"),
    }


def _filter_serializable(data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter dictionary to only JSON-serializable values.

    Args:
        data: Input dictionary

    Returns:
        Filtered dictionary
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                pass
        elif isinstance(value, dict):
            result[key] = _filter_serializable(value)
    return result


def _save_metadata(metadata: Dict[str, Any], path: Path) -> None:
    """Save metadata to JSON file.

    Args:
        metadata: Metadata dictionary
        path: Output path
    """
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _load_metadata(path: Path) -> Dict[str, Any]:
    """Load metadata from JSON file.

    Args:
        path: Metadata file path

    Returns:
        Metadata dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


def _get_framework(algorithm_name: str) -> str:
    """Get framework name for algorithm.

    Args:
        algorithm_name: Algorithm name

    Returns:
        Framework name
    """
    framework_map = {
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "RandomForest": "sklearn",
        "ExtraTrees": "sklearn",
        "LogisticRegression": "sklearn",
        "Ridge": "sklearn",
        "Lasso": "sklearn",
        "GradientBoosting": "sklearn",
        "SVM": "sklearn",
    }
    return framework_map.get(algorithm_name, "sklearn")


def list_checkpoints(
    checkpoint_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
    algorithm_name: Optional[str] = None,
) -> list:
    """List available checkpoints.

    Args:
        checkpoint_dir: Directory to search
        experiment_id: Filter by experiment
        algorithm_name: Filter by algorithm

    Returns:
        List of checkpoint metadata dictionaries
    """
    checkpoint_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)

    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for metadata_path in checkpoint_dir.glob("*_metadata.json"):
        try:
            metadata = _load_metadata(metadata_path)

            # Apply filters
            if experiment_id and metadata.get("experiment_id") != experiment_id:
                continue
            if algorithm_name and metadata.get("algorithm_name") != algorithm_name:
                continue

            # Add path info
            metadata["metadata_path"] = str(metadata_path)
            metadata["model_path"] = str(
                metadata_path.with_name(metadata_path.name.replace("_metadata.json", ".pkl"))
            )

            checkpoints.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to load metadata {metadata_path}: {e}")

    # Sort by creation time (newest first)
    checkpoints.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return checkpoints


def delete_checkpoint(
    checkpoint_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> bool:
    """Delete a checkpoint.

    Args:
        checkpoint_name: Checkpoint name
        checkpoint_path: Direct path to checkpoint
        checkpoint_dir: Checkpoint directory

    Returns:
        True if deleted successfully
    """
    checkpoint_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)

    if checkpoint_path:
        model_path = Path(checkpoint_path)
    elif checkpoint_name:
        model_path = checkpoint_dir / f"{checkpoint_name}.pkl"
    else:
        return False

    metadata_path = model_path.with_name(model_path.stem + "_metadata.json")

    deleted = False
    if model_path.exists():
        model_path.unlink()
        deleted = True
    if metadata_path.exists():
        metadata_path.unlink()
        deleted = True

    return deleted
