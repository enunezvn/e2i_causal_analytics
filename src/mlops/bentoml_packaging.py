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
import subprocess
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
    bentoml = None  # type: ignore[assignment]
    Model = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


def _get_bentoml_executable() -> str:
    """Get the path to the bentoml executable.

    Returns the full path to bentoml in the current Python environment's
    bin directory, falling back to 'bentoml' if not found.
    """
    import sys

    venv_bin = Path(sys.executable).parent
    bentoml_path = venv_bin / "bentoml"
    if bentoml_path.exists():
        return str(bentoml_path)
    return "bentoml"


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
        return register_funcs[framework_lower](model, model_name, metadata=metadata, **kwargs)  # type: ignore[no-any-return]

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


def _bento_exists(bento_name: str, version: Optional[str] = None) -> bool:
    """Check if a Bento with given name and version exists.

    Args:
        bento_name: Name of the bento
        version: Optional specific version to check

    Returns:
        True if the bento exists, False otherwise
    """
    try:
        check_target = f"{bento_name}:{version}" if version else bento_name
        result = subprocess.run(
            [_get_bentoml_executable(), "list", check_target, "-o", "json", "--quiet"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse JSON output to verify it's not empty
            import json

            bentos = json.loads(result.stdout)
            return len(bentos) > 0
        return False
    except Exception as e:
        logger.debug(f"Error checking bento existence: {e}")
        return False


def build_bento(
    service_dir: Union[str, Path],
    bento_name: Optional[str] = None,
    version: Optional[str] = None,
    force_unique_version: bool = True,
) -> str:
    """Build a Bento from a service directory.

    Args:
        service_dir: Directory containing service and bentofile.yaml
        bento_name: Optional name override
        version: Optional version override
        force_unique_version: If True and bento already exists, append timestamp
            to version to ensure uniqueness (default: True)

    Returns:
        Bento tag string
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    service_dir = Path(service_dir)

    # Check if bento with this name+version already exists
    if force_unique_version and bento_name and version:
        if _bento_exists(bento_name, version):
            # Append timestamp to make version unique
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            original_version = version
            version = f"{version}_{timestamp}"
            logger.info(
                f"Bento '{bento_name}:{original_version}' already exists. "
                f"Using unique version: {version}"
            )

    # Build using bentoml CLI
    cmd = [_get_bentoml_executable(), "build", str(service_dir)]

    if bento_name:
        cmd.extend(["--name", bento_name])
    if version:
        cmd.extend(["--version", version])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr.strip()

        # Check if it's an "already exists" error despite our check
        if "already exists" in error_msg.lower():
            # Try one more time with a more unique version
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            version = f"{version or 'v1'}_{timestamp}"
            cmd = [_get_bentoml_executable(), "build", str(service_dir)]
            if bento_name:
                cmd.extend(["--name", bento_name])
            cmd.extend(["--version", version])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Bento build failed (retry): {result.stderr}")
                raise RuntimeError(f"Bento build failed: {result.stderr}")
        else:
            logger.error(f"Bento build failed: {error_msg}")
            raise RuntimeError(f"Bento build failed: {error_msg}")

    # Extract tag from output using multiple patterns
    stdout = result.stdout
    stderr = result.stderr

    # Pattern 1: "Successfully built Bento(name:version)"
    import re

    for line in stdout.split("\n"):
        if "Successfully built Bento" in line:
            # Try to extract tag from line
            # Format: "Successfully built Bento(name:version)."
            match = re.search(r"Bento\(([^)]+)\)", line)
            if match:
                tag = match.group(1).strip("'\"")  # Strip any quotes
                logger.info(f"Built Bento: {tag}")
                return tag
            # Fallback: split on Bento and take what's after
            parts = line.split("Bento")
            if len(parts) > 1:
                tag = parts[-1].strip(" (.)'\"").strip()  # Strip quotes
                if ":" in tag:
                    logger.info(f"Built Bento: {tag}")
                    return tag

    # Pattern 2: Look for tag in any output containing name:version
    # Only allow alphanumeric, underscore, hyphen, period in version
    combined_output = stdout + "\n" + stderr
    if bento_name:
        import re

        pattern = rf"{re.escape(bento_name)}:[a-zA-Z0-9_.-]+"
        match = re.search(pattern, combined_output)
        if match:
            tag = match.group(0).rstrip(".")
            logger.info(f"Built Bento (parsed): {tag}")
            return tag

    # Pattern 3: Construct tag from inputs if build succeeded
    if result.returncode == 0 and bento_name and version:
        # Sanitize version to only allowed characters
        clean_version = re.sub(r"[^a-zA-Z0-9_.-]", "", version)
        tag = f"{bento_name}:{clean_version}"
        logger.info(f"Built Bento (constructed): {tag}")
        return tag

    logger.warning("Could not parse Bento tag from output, build may have succeeded")
    if bento_name and version:
        clean_version = re.sub(r"[^a-zA-Z0-9_.-]", "", version)
        return f"{bento_name}:{clean_version}"
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
        _get_bentoml_executable(),
        "containerize",
        bento_tag,
        "-t",
        config.full_image_name,
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
    """Generate a self-contained service.py file for a model.

    This function generates a fully self-contained BentoML service file that
    does NOT require any imports from the `src` package. This is critical
    because BentoML builds run in isolation and cannot resolve project imports.

    Args:
        model_tag: BentoML model tag
        service_type: Type of service (classification, regression, causal)
        output_path: Path to write service file
        service_name: Name for the service
        **kwargs: Additional arguments:
            - n_classes (int): Number of classes for classification (default: 2)
            - class_names (list): Class label names
            - default_threshold (float): Classification threshold (default: 0.5)
            - target_name (str): Target variable name for regression
            - treatment_name (str): Treatment variable name for causal
            - outcome_name (str): Outcome variable name for causal
            - cpu (str): CPU resource limit (default: "1")
            - memory (str): Memory resource limit (default: "2Gi")
            - timeout (int): Request timeout in seconds (default: 60)

    Returns:
        Path to generated service file
    """
    output_path = Path(output_path)

    # Extract common kwargs with defaults
    cpu = kwargs.get("cpu", "1")
    memory = kwargs.get("memory", "2Gi")
    timeout = kwargs.get("timeout", 60)

    if service_type == "classification":
        service_code = _generate_classification_service(
            model_tag=model_tag,
            service_name=service_name,
            n_classes=kwargs.get("n_classes", 2),
            class_names=kwargs.get("class_names"),
            default_threshold=kwargs.get("default_threshold", 0.5),
            cpu=cpu,
            memory=memory,
            timeout=timeout,
        )
    elif service_type == "regression":
        service_code = _generate_regression_service(
            model_tag=model_tag,
            service_name=service_name,
            target_name=kwargs.get("target_name", "target"),
            cpu=cpu,
            memory=memory,
            timeout=timeout,
        )
    elif service_type == "causal":
        service_code = _generate_causal_service(
            model_tag=model_tag,
            service_name=service_name,
            treatment_name=kwargs.get("treatment_name", "treatment"),
            outcome_name=kwargs.get("outcome_name", "outcome"),
            cpu=cpu,
            memory=memory,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown service type: {service_type}")

    output_path.write_text(service_code)
    logger.info(f"Generated self-contained service file: {output_path}")
    return output_path


def _generate_classification_service(
    model_tag: str,
    service_name: str,
    n_classes: int = 2,
    class_names: Optional[List[str]] = None,
    default_threshold: float = 0.5,
    cpu: str = "1",
    memory: str = "2Gi",
    timeout: int = 60,
) -> str:
    """Generate self-contained classification service code."""
    class_names_str = (
        repr(class_names) if class_names else f"[f'class_{{i}}' for i in range({n_classes})]"
    )

    return f'''"""Auto-generated BentoML Classification Service.

Model: {model_tag}
Service: {service_name}
Generated: {datetime.now(timezone.utc).isoformat()}

This is a self-contained service file with no external project dependencies.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import bentoml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class ClassificationInput(BaseModel):
    """Input schema for classification requests."""
    features: List[List[float]] = Field(..., description="Feature matrix (samples x features)", min_length=1)
    threshold: float = Field(default={default_threshold}, ge=0.0, le=1.0, description="Classification threshold")
    return_all_classes: bool = Field(default=False, description="Return probabilities for all classes")


class ClassificationOutput(BaseModel):
    """Output schema for classification responses."""
    predictions: List[int] = Field(..., description="Predicted class labels")
    probabilities: List[float] = Field(..., description="Prediction probabilities")
    all_probabilities: Optional[List[List[float]]] = Field(default=None, description="Probabilities for all classes")
    confidence_scores: List[float] = Field(..., description="Confidence scores")
    model_id: str = Field(..., description="Model identifier")
    prediction_time_ms: float = Field(..., description="Prediction time in milliseconds")


# ============================================================================
# Service Definition
# ============================================================================


@bentoml.service(
    name="{service_name}",
    resources={{"cpu": "{cpu}", "memory": "{memory}"}},
    traffic={{"timeout": {timeout}}},
)
class {service_name.title().replace("_", "")}Service:
    """BentoML classification service for {model_tag}."""

    def __init__(self):
        """Initialize service and load model."""
        self.model_tag = "{model_tag}"
        self.n_classes = {n_classes}
        self.class_names = {class_names_str}
        self.default_threshold = {default_threshold}
        self._model = None
        self._prediction_count = 0
        self._total_latency_ms = 0.0

        # Load model - auto-detect framework from metadata
        model_ref = bentoml.models.get(self.model_tag)
        framework = model_ref.info.metadata.get("framework", "sklearn")

        if framework == "sklearn":
            self._model = bentoml.sklearn.load_model(self.model_tag)
        elif framework == "xgboost":
            self._model = bentoml.xgboost.load_model(self.model_tag)
        elif framework == "lightgbm":
            self._model = bentoml.lightgbm.load_model(self.model_tag)
        else:
            self._model = bentoml.picklable_model.load_model(self.model_tag)

        logger.info(f"Loaded classification model: {{self.model_tag}}")

    @bentoml.api
    async def predict(self, input_data: ClassificationInput) -> ClassificationOutput:
        """Run classification prediction."""
        start_time = time.time()

        features = np.array(input_data.features)
        predictions = self._model.predict(features)

        # Get probabilities if available
        if hasattr(self._model, 'predict_proba'):
            all_proba = self._model.predict_proba(features)
            if self.n_classes == 2:
                probabilities = all_proba[:, 1].tolist()
                predictions = (np.array(probabilities) >= input_data.threshold).astype(int).tolist()
            else:
                probabilities = np.max(all_proba, axis=1).tolist()
        else:
            all_proba = None
            probabilities = [1.0] * len(predictions)

        confidence_scores = np.max(all_proba, axis=1).tolist() if all_proba is not None else probabilities
        elapsed_ms = (time.time() - start_time) * 1000

        self._prediction_count += len(predictions)
        self._total_latency_ms += elapsed_ms

        return ClassificationOutput(
            predictions=list(map(int, predictions)),
            probabilities=probabilities,
            all_probabilities=all_proba.tolist() if input_data.return_all_classes and all_proba is not None else None,
            confidence_scores=confidence_scores,
            model_id=self.model_tag,
            prediction_time_ms=elapsed_ms,
        )

    @bentoml.api
    async def health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {{
            "status": "healthy",
            "model_id": self.model_tag,
            "model_loaded": self._model is not None,
            "n_classes": self.n_classes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }}

    @bentoml.api
    async def metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        avg_latency = self._total_latency_ms / self._prediction_count if self._prediction_count > 0 else 0.0
        return {{
            "prediction_count": self._prediction_count,
            "total_latency_ms": self._total_latency_ms,
            "average_latency_ms": avg_latency,
        }}
'''


def _generate_regression_service(
    model_tag: str,
    service_name: str,
    target_name: str = "target",
    cpu: str = "1",
    memory: str = "2Gi",
    timeout: int = 60,
) -> str:
    """Generate self-contained regression service code."""
    return f'''"""Auto-generated BentoML Regression Service.

Model: {model_tag}
Service: {service_name}
Generated: {datetime.now(timezone.utc).isoformat()}

This is a self-contained service file with no external project dependencies.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import bentoml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class RegressionInput(BaseModel):
    """Input schema for regression requests."""
    features: List[List[float]] = Field(..., description="Feature matrix (samples x features)", min_length=1)
    return_intervals: bool = Field(default=False, description="Return prediction intervals")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for intervals")


class RegressionOutput(BaseModel):
    """Output schema for regression responses."""
    predictions: List[float] = Field(..., description="Predicted values")
    lower_bounds: Optional[List[float]] = Field(default=None, description="Lower bounds of prediction intervals")
    upper_bounds: Optional[List[float]] = Field(default=None, description="Upper bounds of prediction intervals")
    model_id: str = Field(..., description="Model identifier")
    prediction_time_ms: float = Field(..., description="Prediction time in milliseconds")


# ============================================================================
# Service Definition
# ============================================================================


@bentoml.service(
    name="{service_name}",
    resources={{"cpu": "{cpu}", "memory": "{memory}"}},
    traffic={{"timeout": {timeout}}},
)
class {service_name.title().replace("_", "")}Service:
    """BentoML regression service for {model_tag}."""

    def __init__(self):
        """Initialize service and load model."""
        self.model_tag = "{model_tag}"
        self.target_name = "{target_name}"
        self._model = None
        self._prediction_count = 0
        self._total_latency_ms = 0.0

        # Load model - auto-detect framework from metadata
        model_ref = bentoml.models.get(self.model_tag)
        framework = model_ref.info.metadata.get("framework", "sklearn")

        if framework == "sklearn":
            self._model = bentoml.sklearn.load_model(self.model_tag)
        elif framework == "xgboost":
            self._model = bentoml.xgboost.load_model(self.model_tag)
        elif framework == "lightgbm":
            self._model = bentoml.lightgbm.load_model(self.model_tag)
        else:
            self._model = bentoml.picklable_model.load_model(self.model_tag)

        logger.info(f"Loaded regression model: {{self.model_tag}}")

    @bentoml.api
    async def predict(self, input_data: RegressionInput) -> RegressionOutput:
        """Run regression prediction."""
        start_time = time.time()

        features = np.array(input_data.features)
        predictions = self._model.predict(features)

        elapsed_ms = (time.time() - start_time) * 1000

        self._prediction_count += len(predictions)
        self._total_latency_ms += elapsed_ms

        return RegressionOutput(
            predictions=predictions.tolist(),
            lower_bounds=None,
            upper_bounds=None,
            model_id=self.model_tag,
            prediction_time_ms=elapsed_ms,
        )

    @bentoml.api
    async def health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {{
            "status": "healthy",
            "model_id": self.model_tag,
            "model_loaded": self._model is not None,
            "target_name": self.target_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }}

    @bentoml.api
    async def metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        avg_latency = self._total_latency_ms / self._prediction_count if self._prediction_count > 0 else 0.0
        return {{
            "prediction_count": self._prediction_count,
            "total_latency_ms": self._total_latency_ms,
            "average_latency_ms": avg_latency,
        }}
'''


def _generate_causal_service(
    model_tag: str,
    service_name: str,
    treatment_name: str = "treatment",
    outcome_name: str = "outcome",
    cpu: str = "2",
    memory: str = "4Gi",
    timeout: int = 120,
) -> str:
    """Generate self-contained causal inference service code."""
    return f'''"""Auto-generated BentoML Causal Inference Service.

Model: {model_tag}
Service: {service_name}
Generated: {datetime.now(timezone.utc).isoformat()}

This is a self-contained service file with no external project dependencies.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import bentoml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class CausalInput(BaseModel):
    """Input schema for causal inference requests."""
    features: List[List[float]] = Field(..., description="Feature matrix (samples x features)", min_length=1)
    treatment: Optional[List[int]] = Field(default=None, description="Treatment assignment (0 or 1)")
    return_intervals: bool = Field(default=True, description="Return confidence intervals for CATE")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level")


class CATEOutput(BaseModel):
    """Output schema for CATE estimation."""
    cate: List[float] = Field(..., description="Conditional Average Treatment Effect estimates")
    lower_bounds: Optional[List[float]] = Field(default=None, description="Lower bounds of confidence intervals")
    upper_bounds: Optional[List[float]] = Field(default=None, description="Upper bounds of confidence intervals")
    ate: float = Field(..., description="Average Treatment Effect (mean of CATE)")
    ate_std: float = Field(..., description="Standard deviation of ATE")
    model_id: str = Field(..., description="Model identifier")
    prediction_time_ms: float = Field(..., description="Prediction time in milliseconds")


# ============================================================================
# Service Definition
# ============================================================================


@bentoml.service(
    name="{service_name}",
    resources={{"cpu": "{cpu}", "memory": "{memory}"}},
    traffic={{"timeout": {timeout}}},
)
class {service_name.title().replace("_", "")}Service:
    """BentoML causal inference service for {model_tag}."""

    def __init__(self):
        """Initialize service and load model."""
        self.model_tag = "{model_tag}"
        self.treatment_name = "{treatment_name}"
        self.outcome_name = "{outcome_name}"
        self._model = None
        self._prediction_count = 0
        self._total_latency_ms = 0.0

        # EconML/DoWhy models are typically pickled
        self._model = bentoml.picklable_model.load_model(self.model_tag)
        logger.info(f"Loaded causal model: {{self.model_tag}}")

    def _estimate_cate(self, features: np.ndarray) -> np.ndarray:
        """Estimate CATE using the loaded model."""
        if hasattr(self._model, 'effect'):
            return self._model.effect(features)
        elif hasattr(self._model, 'const_marginal_effect'):
            return self._model.const_marginal_effect(features)
        elif hasattr(self._model, 'predict'):
            return self._model.predict(features)
        else:
            raise ValueError("Model does not support CATE estimation")

    @bentoml.api
    async def estimate_cate(self, input_data: CausalInput) -> CATEOutput:
        """Estimate Conditional Average Treatment Effect."""
        start_time = time.time()

        features = np.array(input_data.features)
        cate = self._estimate_cate(features)
        if cate.ndim > 1:
            cate = cate.flatten()

        ate = float(np.mean(cate))
        ate_std = float(np.std(cate))
        elapsed_ms = (time.time() - start_time) * 1000

        self._prediction_count += len(cate)
        self._total_latency_ms += elapsed_ms

        return CATEOutput(
            cate=cate.tolist(),
            lower_bounds=None,
            upper_bounds=None,
            ate=ate,
            ate_std=ate_std,
            model_id=self.model_tag,
            prediction_time_ms=elapsed_ms,
        )

    @bentoml.api
    async def health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {{
            "status": "healthy",
            "model_id": self.model_tag,
            "model_loaded": self._model is not None,
            "treatment_name": self.treatment_name,
            "outcome_name": self.outcome_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }}

    @bentoml.api
    async def metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        avg_latency = self._total_latency_ms / self._prediction_count if self._prediction_count > 0 else 0.0
        return {{
            "prediction_count": self._prediction_count,
            "total_latency_ms": self._total_latency_ms,
            "average_latency_ms": avg_latency,
        }}
'''


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
    compose_config: Dict[str, Any] = {
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
                "test": [
                    "CMD",
                    "curl",
                    "-f",
                    f"http://localhost:{svc.get('internal_port', 3000)}/health",
                ],
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


def validate_bento(bento_tag: str, strict: bool = False) -> Dict[str, Any]:
    """Validate a Bento is ready for deployment.

    Modern BentoML (v1.4+) structures bentos differently, so this validation
    focuses on essential checks (bento exists and is loadable) rather than
    specific file layouts.

    Args:
        bento_tag: Tag of the Bento to validate
        strict: If True, require all traditional files to be present.
                If False (default), only check if bento exists and is valid.

    Returns:
        Validation result with status and details
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    result: Dict[str, Any] = {
        "valid": False,
        "bento_tag": bento_tag,
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    # Primary check: Bento exists and can be loaded
    try:
        bento = bentoml.get(bento_tag)
        result["checks"]["bento_exists"] = True
        result["bento_info"] = {
            "tag": str(bento.tag),
            "path": str(bento.path) if hasattr(bento, "path") else None,
        }
    except Exception as e:
        result["errors"].append(f"Bento not found: {e}")
        result["checks"]["bento_exists"] = False
        return result

    # Secondary check: Look for service files (warning if missing, not error)
    # Modern BentoML stores these in various locations
    try:
        bento_path = Path(bento.path) if hasattr(bento, "path") else None
        if bento_path:
            # Check multiple possible locations for service files
            possible_service_files = [
                "service.py",
                "src/service.py",
                "apis/service.py",
            ]
            service_found = False
            for f in possible_service_files:
                if (bento_path / f).exists():
                    service_found = True
                    result["checks"]["has_service"] = True
                    result["service_file"] = f
                    break

            if not service_found:
                if strict:
                    result["errors"].append("No service.py found in Bento")
                else:
                    result["warnings"].append("No service.py found (may be embedded in Bento)")
                result["checks"]["has_service"] = False

            # Check for bentofile.yaml (optional in modern BentoML)
            bentofile_exists = (bento_path / "bentofile.yaml").exists()
            result["checks"]["has_bentofile"] = bentofile_exists
            if not bentofile_exists and strict:
                result["errors"].append("Missing bentofile.yaml")
    except Exception as e:
        result["warnings"].append(f"Could not check Bento files: {e}")

    # Check models (informational - some bentos load models dynamically)
    try:
        if hasattr(bento, "info") and hasattr(bento.info, "models"):
            models = bento.info.models
            result["checks"]["has_models"] = len(models) > 0
            result["model_count"] = len(models)
            if len(models) == 0:
                if strict:
                    result["errors"].append("No models found in Bento")
                else:
                    result["warnings"].append("No models embedded (may be loaded dynamically)")
        else:
            result["checks"]["has_models"] = False
            result["model_count"] = 0
    except Exception as e:
        result["warnings"].append(f"Could not list models: {e}")

    # Bento is valid if it exists and we have no blocking errors
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

    result: Dict[str, Any] = {
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
