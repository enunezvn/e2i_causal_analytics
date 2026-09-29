"""MLflow registry calls for the model_deployer (register a version, transition a stage).

Split out of ``registry_manager`` (module-size ratchet, #2242); behaviour unchanged.
``registry_manager`` re-exports these names, and its ``register_model`` /
``promote_stage`` call them through its own module globals, so patching
``registry_manager._register_model_mlflow`` / ``._transition_stage_mlflow`` still works.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

logger = logging.getLogger(__name__)


def _get_mlflow_connector() -> Optional[Any]:
    """Get MLflow connector singleton if available.

    Returns:
        MLflowConnector instance or None if unavailable
    """
    try:
        from src.mlops.mlflow_connector import MLflowConnector

        connector = MLflowConnector()
        return connector if connector.enabled else None
    except ImportError:
        logger.warning("MLflowConnector not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get MLflow connector: {e}")
        return None


async def _register_model_mlflow(
    model_uri: str, deployment_name: str
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Register model with MLflow via MLflowConnector.

    Args:
        model_uri: MLflow model URI (runs:/<run_id>/model)
        deployment_name: Name to register model under

    Returns:
        Tuple of (registered_name, version, stage) or (None, None, None) on failure
    """
    connector = _get_mlflow_connector()
    if not connector:
        return None, None, None

    try:
        # Extract run_id and model_path from model_uri
        # MLflow 3.x returns models:/m-<hash> format; legacy uses runs:/<run_id>/<path>
        if model_uri.startswith("runs:/"):
            parts = model_uri[6:].split("/", 1)
            run_id = parts[0]
            model_path = parts[1] if len(parts) > 1 else "model"
        elif model_uri.startswith("models:/"):
            # MLflow 3.x model URI — register directly via mlflow.register_model()
            try:
                import mlflow

                result = mlflow.register_model(model_uri, deployment_name)
                logger.info(
                    f"Registered model from models:/ URI: {deployment_name} v{result.version}"
                )
                return deployment_name, int(result.version), "None"
            except Exception as e:
                logger.warning(f"Direct registration from models:/ URI failed: {e}")
                return None, None, None
        else:
            logger.warning(f"Unexpected model_uri format: {model_uri}")
            return None, None, None

        # Use MLflowConnector's async register_model method

        model_version = await connector.register_model(
            run_id=run_id,
            model_name=deployment_name,
            model_path=model_path,
        )

        if model_version:
            # A newly registered MLflow version is at registry stage "None". The connector's
            # ModelVersion carries ModelStage.DEVELOPMENT ("development"), which is not an
            # MLflow stage and which validate_promotion's paths do not know (#2310 codex r1).
            return model_version.name, int(model_version.version), "None"
        return None, None, None

    except Exception as e:
        logger.warning(f"MLflow registration failed via connector: {e}")
        return None, None, None


async def _transition_stage_mlflow(model_name: str, version: int, target_stage: str) -> bool:
    """Transition model stage via MLflowConnector.

    Args:
        model_name: Registered model name
        version: Model version
        target_stage: Target stage name (Staging, Production, Archived)

    Returns:
        True if successful, False otherwise
    """
    connector = _get_mlflow_connector()
    if not connector:
        return False

    try:
        from src.mlops.mlflow_connector import ModelStage

        # Map MLflow stage names to our enum
        stage_map = {
            "None": ModelStage.DEVELOPMENT,
            "Staging": ModelStage.STAGING,
            "Shadow": ModelStage.SHADOW,
            "Production": ModelStage.PRODUCTION,
            "Archived": ModelStage.ARCHIVED,
        }

        stage = stage_map.get(target_stage, ModelStage.DEVELOPMENT)

        # Use MLflowConnector's async transition_model_stage method
        success = await connector.transition_model_stage(
            model_name=model_name,
            version=str(version),
            stage=stage,
            archive_existing=(target_stage == "Production"),
        )

        if success:
            logger.info(f"MLflow: Transitioned {model_name} v{version} to {target_stage}")
        return cast(bool, success)

    except Exception as e:
        logger.warning(f"MLflow stage transition failed: {e}")
        return False


async def _tag_model_version_mlflow(model_name: str, version: int, tags: Dict[str, str]) -> bool:
    """Set ``tags`` on one MLflow model version (#2310); True only when every tag is set.

    The version must exist: MLflow refuses a tag on a missing version, and that refusal is
    the answer (False), never a silent success. A disabled connector is False too.
    """
    connector = _get_mlflow_connector()
    client = getattr(connector, "_client", None) if connector else None
    if client is None:
        return False
    try:
        for key, value in tags.items():
            client.set_model_version_tag(
                name=model_name, version=str(version), key=key, value=value
            )
    except Exception as e:
        logger.warning(f"MLflow tag of {model_name} v{version} failed: {e}")
        return False
    return True


async def _model_versions_for_run(model_name: str, run_id: str) -> Optional[List[int]]:
    """The MLflow versions of ``model_name`` registered from ``run_id`` (#2310 codex r1).

    ``None`` when MLflow cannot be asked (disabled connector, search failure): the caller
    then cannot prove there is no earlier version and registers as before.
    """
    connector = _get_mlflow_connector()
    client = getattr(connector, "_client", None) if connector else None
    if client is None:
        return None
    try:
        found = client.search_model_versions(f"name = '{model_name}' and run_id = '{run_id}'")
    except Exception as e:
        logger.warning(f"MLflow version search for {model_name} run {run_id} failed: {e}")
        return None
    return sorted(int(v.version) for v in found)


async def _model_version_stage(model_name: str, version: int) -> Optional[str]:
    """The MLflow registry stage of one version ("None", "Staging", ...), or None when
    MLflow cannot be asked or the version does not exist (#2310 codex r5)."""
    connector = _get_mlflow_connector()
    client = getattr(connector, "_client", None) if connector else None
    if client is None:
        return None
    try:
        return str(client.get_model_version(model_name, str(version)).current_stage)
    except Exception as e:
        logger.warning(f"MLflow stage read of {model_name} v{version} failed: {e}")
        return None
