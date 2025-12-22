"""BentoML Model Packaging Utilities.

This module provides utilities for packaging ML models into Bento bundles
ready for deployment. It handles:
- Model serialization and registration
- Bento build configuration
- Containerization
- Deployment validation

Version: 1.0.0
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import bentoml
    from bentoml import Model
    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False
    bentoml = None
    Model = None

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class BentoConfig:
    """Configuration for Bento build."""

    service_name: str
    service_file: str = "service.py"
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    include: List[str] = field(default_factory=lambda: ["*.py"])
    exclude: List[str] = field(default_factory=lambda: ["__pycache__/", "*.pyc", ".git/"])
    python_version: str = "3.11"
    python_packages: List[str] = field(default_factory=list)
    docker_base_image: Optional[str] = None
    docker_env: Dict[str, str] = field(default_factory=dict)

    def to_yaml(self) -> str:
        """Convert config to bentofile.yaml content."""
        config = {
            "service": f"{self.service_file}:{self.service_name}",
            "description": self.description,
            "labels": self.labels,
            "include": self.include,
            "exclude": self.exclude,
            "python": {
                "packages": self.python_packages,
            },
        }

        if self.docker_base_image:
            config["docker"] = {
                "base_image": self.docker_base_image,
                "env": self.docker_env,
            }

        if yaml:
            return yaml.dump(config, default_flow_style=False)
        else:
            # Fallback to simple string formatting
            return str(config)


@dataclass
class ContainerConfig:
    """Configuration for container deployment."""

    image_name: str
    image_tag: str = "latest"
    registry: Optional[str] = None
    port: int = 3000
    workers: int = 1
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    env_vars: Dict[str, str] = field(default_factory=dict)
    health_check_path: str = "/health"
    health_check_interval: int = 30

    @property
    def full_image_name(self) -> str:
        """Get full image name with registry and tag."""
        if self.registry:
            return f"{self.registry}/{self.image_name}:{self.image_tag}"
        return f"{self.image_name}:{self.image_tag}"


# ============================================================================
# Model Packaging Functions
# ============================================================================


def register_sklearn_model(
    model: Any,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    signatures: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, str]] = None,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> str:
    """Register a scikit-learn model with BentoML.

    Args:
        model: Trained sklearn model
        model_name: Name for the model in BentoML store
        metadata: Model metadata (metrics, params, etc.)
        signatures: API signatures for the model
        labels: Labels for organization
        custom_objects: Additional objects (preprocessors, etc.)

    Returns:
        Model tag string
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    metadata = metadata or {}
    metadata["framework"] = "sklearn"
    metadata["registered_at"] = datetime.now(timezone.utc).isoformat()

    tag = bentoml.sklearn.save_model(
        model_name,
        model,
        metadata=metadata,
        signatures=signatures or {"predict": {"batchable": True}},
        labels=labels,
        custom_objects=custom_objects,
    )

    logger.info(f"Registered sklearn model: {tag}")
    return str(tag)


def register_xgboost_model(
    model: Any,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    signatures: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, str]] = None,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> str:
    """Register an XGBoost model with BentoML.

    Args:
        model: Trained XGBoost model
        model_name: Name for the model
        metadata: Model metadata
        signatures: API signatures
        labels: Labels for organization
        custom_objects: Additional objects

    Returns:
        Model tag string
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    metadata = metadata or {}
    metadata["framework"] = "xgboost"
    metadata["registered_at"] = datetime.now(timezone.utc).isoformat()

    tag = bentoml.xgboost.save_model(
        model_name,
        model,
        metadata=metadata,
        signatures=signatures or {"predict": {"batchable": True}},
        labels=labels,
        custom_objects=custom_objects,
    )

    logger.info(f"Registered XGBoost model: {tag}")
    return str(tag)


def register_lightgbm_model(
    model: Any,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    signatures: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, str]] = None,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> str:
    """Register a LightGBM model with BentoML.

    Args:
        model: Trained LightGBM model
        model_name: Name for the model
        metadata: Model metadata
        signatures: API signatures
        labels: Labels for organization
        custom_objects: Additional objects

    Returns:
        Model tag string
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    metadata = metadata or {}
    metadata["framework"] = "lightgbm"
    metadata["registered_at"] = datetime.now(timezone.utc).isoformat()

    tag = bentoml.lightgbm.save_model(
        model_name,
        model,
        metadata=metadata,
        signatures=signatures or {"predict": {"batchable": True}},
        labels=labels,
        custom_objects=custom_objects,
    )

    logger.info(f"Registered LightGBM model: {tag}")
    return str(tag)


def register_causal_model(
    model: Any,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, str]] = None,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> str:
    """Register a causal inference model (EconML/DoWhy) with BentoML.

    Args:
        model: Trained causal model
        model_name: Name for the model
        metadata: Model metadata
        labels: Labels for organization
        custom_objects: Additional objects

    Returns:
        Model tag string
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    metadata = metadata or {}
    metadata["framework"] = "econml"
    metadata["registered_at"] = datetime.now(timezone.utc).isoformat()

    # Causal models are pickled
    tag = bentoml.picklable_model.save_model(
        model_name,
        model,
        metadata=metadata,
        labels=labels,
        custom_objects=custom_objects,
    )

    logger.info(f"Registered causal model: {tag}")
    return str(tag)


def register_model_auto(
    model: Any,
    model_name: str,
    framework: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """Auto-detect framework and register model.

    Args:
        model: Trained model
        model_name: Name for the model
        framework: Explicit framework override
        metadata: Model metadata
        **kwargs: Additional arguments passed to register function

    Returns:
        Model tag string
    """
    if framework:
        framework_lower = framework.lower()
    else:
        # Auto-detect framework
        model_type = type(model).__module__

        if "sklearn" in model_type:
            framework_lower = "sklearn"
        elif "xgboost" in model_type:
            framework_lower = "xgboost"
        elif "lightgbm" in model_type:
            framework_lower = "lightgbm"
        elif "econml" in model_type or "dowhy" in model_type:
            framework_lower = "econml"
        else:
            framework_lower = "pickle"

    register_funcs = {
        "sklearn": register_sklearn_model,
        "xgboost": register_xgboost_model,
        "lightgbm": register_lightgbm_model,
        "econml": register_causal_model,
        "causal": register_causal_model,
    }

    if framework_lower in register_funcs:
        return register_funcs[framework_lower](model, model_name, metadata=metadata, **kwargs)

    # Fallback to pickle
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    metadata = metadata or {}
    metadata["framework"] = framework_lower
    metadata["registered_at"] = datetime.now(timezone.utc).isoformat()

    tag = bentoml.picklable_model.save_model(
        model_name,
        model,
        metadata=metadata,
        **kwargs,
    )

    logger.info(f"Registered model (pickle): {tag}")
    return str(tag)


# ============================================================================
# Bento Build Functions
# ============================================================================


def create_bentofile(
    output_dir: Union[str, Path],
    config: BentoConfig,
) -> Path:
    """Create a bentofile.yaml for building a Bento.

    Args:
        output_dir: Directory to write the bentofile
        config: Bento configuration

    Returns:
        Path to created bentofile.yaml
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bentofile_path = output_dir / "bentofile.yaml"
    bentofile_path.write_text(config.to_yaml())

    logger.info(f"Created bentofile at: {bentofile_path}")
    return bentofile_path


def build_bento(
    service_dir: Union[str, Path],
    bento_name: Optional[str] = None,
    version: Optional[str] = None,
) -> str:
    """Build a Bento from a service directory.

    Args:
        service_dir: Directory containing service and bentofile.yaml
        bento_name: Optional name override
        version: Optional version override

    Returns:
        Bento tag string
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    service_dir = Path(service_dir)

    # Build using bentoml CLI
    cmd = ["bentoml", "build", str(service_dir)]

    if bento_name:
        cmd.extend(["--name", bento_name])
    if version:
        cmd.extend(["--version", version])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Bento build failed: {result.stderr}")
        raise RuntimeError(f"Bento build failed: {result.stderr}")

    # Extract tag from output
    for line in result.stdout.split("\n"):
        if "Successfully built Bento" in line:
            # Parse tag from output
            tag = line.split(":")[-1].strip()
            logger.info(f"Built Bento: {tag}")
            return tag

    logger.warning("Could not parse Bento tag from output")
    return "unknown"


def containerize_bento(
    bento_tag: str,
    config: ContainerConfig,
    push: bool = False,
) -> str:
    """Containerize a Bento into a Docker image.

    Args:
        bento_tag: Tag of the Bento to containerize
        config: Container configuration
        push: Whether to push to registry

    Returns:
        Docker image name
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    # Build Docker image
    cmd = [
        "bentoml", "containerize",
        bento_tag,
        "-t", config.full_image_name,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Containerization failed: {result.stderr}")
        raise RuntimeError(f"Containerization failed: {result.stderr}")

    logger.info(f"Created container: {config.full_image_name}")

    # Push to registry if requested
    if push and config.registry:
        push_cmd = ["docker", "push", config.full_image_name]
        push_result = subprocess.run(push_cmd, capture_output=True, text=True)

        if push_result.returncode != 0:
            logger.error(f"Push failed: {push_result.stderr}")
            raise RuntimeError(f"Push failed: {push_result.stderr}")

        logger.info(f"Pushed to registry: {config.full_image_name}")

    return config.full_image_name


# ============================================================================
# Service Generation Functions
# ============================================================================


def generate_service_file(
    model_tag: str,
    service_type: str,
    output_path: Union[str, Path],
    service_name: str = "prediction_service",
    **kwargs,
) -> Path:
    """Generate a service.py file for a model.

    Args:
        model_tag: BentoML model tag
        service_type: Type of service (classification, regression, causal)
        output_path: Path to write service file
        service_name: Name for the service
        **kwargs: Additional arguments for the service template

    Returns:
        Path to generated service file
    """
    output_path = Path(output_path)

    if service_type == "classification":
        template_import = "from src.mlops.bentoml_templates import ClassificationServiceTemplate"
        template_class = "ClassificationServiceTemplate"
    elif service_type == "regression":
        template_import = "from src.mlops.bentoml_templates import RegressionServiceTemplate"
        template_class = "RegressionServiceTemplate"
    elif service_type == "causal":
        template_import = "from src.mlops.bentoml_templates import CausalInferenceServiceTemplate"
        template_class = "CausalInferenceServiceTemplate"
    else:
        raise ValueError(f"Unknown service type: {service_type}")

    # Format kwargs for service creation
    kwargs_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in kwargs.items())

    service_code = f'''"""Auto-generated BentoML service for {model_tag}.

Generated: {datetime.now(timezone.utc).isoformat()}
Service Type: {service_type}
"""

{template_import}

# Create the service class
{service_name} = {template_class}.create(
    model_tag="{model_tag}",
    service_name="{service_name}",
    {kwargs_str}
)
'''

    output_path.write_text(service_code)
    logger.info(f"Generated service file: {output_path}")
    return output_path


def generate_docker_compose(
    services: List[Dict[str, Any]],
    output_path: Union[str, Path],
    network_name: str = "bentoml-network",
) -> Path:
    """Generate a docker-compose.yaml for multiple services.

    Args:
        services: List of service configurations
        output_path: Path to write docker-compose file
        network_name: Docker network name

    Returns:
        Path to generated docker-compose file
    """
    compose_config = {
        "version": "3.8",
        "services": {},
        "networks": {
            network_name: {
                "driver": "bridge",
            }
        },
    }

    for svc in services:
        svc_name = svc["name"]
        compose_config["services"][svc_name] = {
            "image": svc["image"],
            "container_name": svc_name,
            "ports": [f"{svc.get('port', 3000)}:{svc.get('internal_port', 3000)}"],
            "environment": svc.get("env", {}),
            "networks": [network_name],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD", "curl", "-f", f"http://localhost:{svc.get('internal_port', 3000)}/health"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
            },
        }

        if "cpu_limit" in svc or "memory_limit" in svc:
            compose_config["services"][svc_name]["deploy"] = {
                "resources": {
                    "limits": {
                        "cpus": svc.get("cpu_limit", "2"),
                        "memory": svc.get("memory_limit", "4G"),
                    }
                }
            }

    output_path = Path(output_path)

    if yaml:
        output_path.write_text(yaml.dump(compose_config, default_flow_style=False))
    else:
        # Fallback
        output_path.write_text(str(compose_config))

    logger.info(f"Generated docker-compose: {output_path}")
    return output_path


# ============================================================================
# Validation Functions
# ============================================================================


def validate_bento(bento_tag: str) -> Dict[str, Any]:
    """Validate a Bento is ready for deployment.

    Args:
        bento_tag: Tag of the Bento to validate

    Returns:
        Validation result with status and details
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    result = {
        "valid": False,
        "bento_tag": bento_tag,
        "checks": {},
        "errors": [],
    }

    try:
        bento = bentoml.get(bento_tag)
        result["checks"]["bento_exists"] = True
    except Exception as e:
        result["errors"].append(f"Bento not found: {e}")
        result["checks"]["bento_exists"] = False
        return result

    # Check for required files
    try:
        bento_path = bento.path
        required_files = ["service.py", "bentofile.yaml"]
        for f in required_files:
            file_exists = (Path(bento_path) / f).exists()
            result["checks"][f"has_{f}"] = file_exists
            if not file_exists:
                result["errors"].append(f"Missing required file: {f}")
    except Exception as e:
        result["errors"].append(f"Could not access Bento path: {e}")

    # Check models
    try:
        models = bento.info.models
        result["checks"]["has_models"] = len(models) > 0
        result["model_count"] = len(models)
        if len(models) == 0:
            result["errors"].append("No models found in Bento")
    except Exception as e:
        result["errors"].append(f"Could not list models: {e}")

    result["valid"] = len(result["errors"]) == 0
    return result


def test_service_locally(
    service_module: str,
    test_input: Dict[str, Any],
    timeout: int = 30,
) -> Dict[str, Any]:
    """Test a service locally before deployment.

    Args:
        service_module: Python module path to service
        test_input: Test input data
        timeout: Timeout in seconds

    Returns:
        Test result with response and timing
    """
    import asyncio
    import importlib

    result = {
        "success": False,
        "response": None,
        "error": None,
        "duration_ms": 0,
    }

    try:
        # Import service module
        module = importlib.import_module(service_module)
        service_class = getattr(module, "service", None)

        if service_class is None:
            result["error"] = "No 'service' found in module"
            return result

        # Instantiate and test
        start_time = datetime.now()
        service = service_class()

        # Call predict method
        if hasattr(service, "predict"):
            response = asyncio.run(service.predict(test_input))
            result["response"] = response
            result["success"] = True
        else:
            result["error"] = "Service has no 'predict' method"

        result["duration_ms"] = (datetime.now() - start_time).total_seconds() * 1000

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================================
# Cleanup Functions
# ============================================================================


def cleanup_old_models(
    model_name: str,
    keep_versions: int = 3,
) -> List[str]:
    """Clean up old model versions from BentoML store.

    Args:
        model_name: Base name of the model
        keep_versions: Number of versions to keep

    Returns:
        List of deleted model tags
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    deleted = []

    try:
        models = bentoml.models.list(model_name)
        models = sorted(models, key=lambda m: m.creation_time, reverse=True)

        for model in models[keep_versions:]:
            bentoml.models.delete(model.tag)
            deleted.append(str(model.tag))
            logger.info(f"Deleted old model: {model.tag}")

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

    return deleted


def cleanup_old_bentos(
    bento_name: str,
    keep_versions: int = 3,
) -> List[str]:
    """Clean up old Bento versions.

    Args:
        bento_name: Base name of the Bento
        keep_versions: Number of versions to keep

    Returns:
        List of deleted Bento tags
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    deleted = []

    try:
        bentos = bentoml.list(bento_name)
        bentos = sorted(bentos, key=lambda b: b.creation_time, reverse=True)

        for bento in bentos[keep_versions:]:
            bentoml.delete(bento.tag)
            deleted.append(str(bento.tag))
            logger.info(f"Deleted old Bento: {bento.tag}")

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

    return deleted
