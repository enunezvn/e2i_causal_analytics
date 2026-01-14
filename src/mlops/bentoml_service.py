"""BentoML Model Serving for E2I Causal Analytics.

This module provides BentoML integration for model serving:
- Model packaging and registration
- Prediction service creation
- Health checks and monitoring
- Model versioning and deployment

Version: 1.0.0
"""

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

try:
    import bentoml
    # Note: bentoml.io module deprecated in v1.4+
    # Use Pydantic models with @bentoml.api decorator instead
    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False
    bentoml = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object
    Field = lambda *args, **kwargs: None

logger = logging.getLogger(__name__)

# Default model store path
DEFAULT_MODEL_STORE_PATH = os.environ.get(
    "E2I_BENTOML_MODEL_STORE",
    None,  # Use BentoML default
)


# ============================================================================
# Pydantic models for API
# ============================================================================


class PredictionInput(BaseModel):
    """Input schema for prediction requests."""

    features: List[List[float]] = Field(
        ...,
        description="Feature matrix (samples x features)",
        min_length=1,
    )
    return_proba: bool = Field(
        default=False,
        description="Return class probabilities (classification only)",
    )


class PredictionOutput(BaseModel):
    """Output schema for prediction responses."""

    predictions: List[Any] = Field(
        ...,
        description="Model predictions",
    )
    probabilities: Optional[List[List[float]]] = Field(
        default=None,
        description="Class probabilities (if return_proba=True)",
    )
    model_id: str = Field(
        ...,
        description="Model identifier used for prediction",
    )
    prediction_time_ms: float = Field(
        ...,
        description="Time taken for prediction in milliseconds",
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(
        ...,
        description="Service status",
    )
    model_id: str = Field(
        ...,
        description="Loaded model identifier",
    )
    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded",
    )
    timestamp: str = Field(
        ...,
        description="Health check timestamp",
    )


class ModelMetadata(BaseModel):
    """Model metadata schema."""

    model_id: str
    algorithm_name: str
    problem_type: str
    framework: str
    created_at: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    feature_names: List[str] = Field(default_factory=list)
    n_features: int = 0


# ============================================================================
# BentoML Model Manager
# ============================================================================


class BentoMLModelManager:
    """Manages BentoML model registration and retrieval.

    This class provides utilities for:
    - Saving sklearn/xgboost/lightgbm models to BentoML store
    - Loading models for serving
    - Model versioning and tagging
    """

    def __init__(self, model_store_path: Optional[str] = None):
        """Initialize BentoML model manager.

        Args:
            model_store_path: Optional custom model store path
        """
        if not BENTOML_AVAILABLE:
            raise ImportError(
                "BentoML is not installed. "
                "Install with: pip install bentoml"
            )

        self.model_store_path = model_store_path or DEFAULT_MODEL_STORE_PATH

        # Configure model store if custom path provided
        if self.model_store_path:
            os.environ["BENTOML_HOME"] = self.model_store_path

    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        framework: str = "sklearn",
        signatures: Optional[Dict[str, Any]] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """Save a trained model to BentoML model store.

        Args:
            model: Trained model object
            model_name: Name for the model (e.g., "churn_classifier")
            metadata: Optional metadata dictionary
            framework: ML framework ("sklearn", "xgboost", "lightgbm")
            signatures: Optional model signatures for inference
            custom_objects: Optional custom objects to serialize with model
            labels: Optional labels for model organization

        Returns:
            Model tag string (e.g., "churn_classifier:abc123")
        """
        metadata = metadata or {}
        labels = labels or {}

        # Add timestamp to metadata
        metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
        metadata["framework"] = framework

        try:
            # Save based on framework
            if framework == "sklearn":
                saved_model = bentoml.sklearn.save_model(
                    model_name,
                    model,
                    signatures=signatures,
                    labels=labels,
                    custom_objects=custom_objects,
                    metadata=metadata,
                )
            elif framework == "xgboost":
                saved_model = bentoml.xgboost.save_model(
                    model_name,
                    model,
                    signatures=signatures,
                    labels=labels,
                    custom_objects=custom_objects,
                    metadata=metadata,
                )
            elif framework == "lightgbm":
                saved_model = bentoml.lightgbm.save_model(
                    model_name,
                    model,
                    signatures=signatures,
                    labels=labels,
                    custom_objects=custom_objects,
                    metadata=metadata,
                )
            else:
                # Fallback to pickle-based storage
                saved_model = bentoml.picklable_model.save_model(
                    model_name,
                    model,
                    signatures=signatures,
                    labels=labels,
                    custom_objects=custom_objects,
                    metadata=metadata,
                )

            model_tag = str(saved_model.tag)
            logger.info(f"Saved model to BentoML: {model_tag}")
            return model_tag

        except Exception as e:
            logger.error(f"Failed to save model to BentoML: {e}")
            raise

    def save_model_with_preprocessor(
        self,
        model: Any,
        preprocessor: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        framework: str = "sklearn",
    ) -> str:
        """Save model along with its preprocessor.

        Args:
            model: Trained model object
            preprocessor: Fitted preprocessor/pipeline
            model_name: Name for the model
            metadata: Optional metadata dictionary
            framework: ML framework

        Returns:
            Model tag string
        """
        custom_objects = {"preprocessor": preprocessor}
        return self.save_model(
            model=model,
            model_name=model_name,
            metadata=metadata,
            framework=framework,
            custom_objects=custom_objects,
        )

    def get_model(
        self,
        model_tag: str,
    ) -> Any:
        """Load a model from BentoML store.

        Args:
            model_tag: Model tag (e.g., "churn_classifier:latest")

        Returns:
            Loaded model object
        """
        try:
            model_ref = bentoml.models.get(model_tag)

            # Determine framework from metadata
            framework = model_ref.info.metadata.get("framework", "sklearn")

            if framework == "sklearn":
                return bentoml.sklearn.load_model(model_tag)
            elif framework == "xgboost":
                return bentoml.xgboost.load_model(model_tag)
            elif framework == "lightgbm":
                return bentoml.lightgbm.load_model(model_tag)
            else:
                return bentoml.picklable_model.load_model(model_tag)

        except Exception as e:
            logger.error(f"Failed to load model {model_tag}: {e}")
            raise

    def get_model_info(self, model_tag: str) -> Dict[str, Any]:
        """Get model information from BentoML store.

        Args:
            model_tag: Model tag

        Returns:
            Dictionary with model info
        """
        try:
            model_ref = bentoml.models.get(model_tag)
            return {
                "tag": str(model_ref.tag),
                "path": str(model_ref.path),
                "creation_time": model_ref.info.creation_time.isoformat(),
                "metadata": model_ref.info.metadata,
                "labels": model_ref.info.labels,
            }
        except Exception as e:
            logger.error(f"Failed to get model info for {model_tag}: {e}")
            raise

    def list_models(
        self,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List models in BentoML store.

        Args:
            model_name: Optional filter by model name

        Returns:
            List of model info dictionaries
        """
        try:
            models = bentoml.models.list(model_name)
            return [
                {
                    "tag": str(m.tag),
                    "creation_time": m.info.creation_time.isoformat(),
                    "metadata": m.info.metadata,
                    "labels": m.info.labels,
                }
                for m in models
            ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def delete_model(self, model_tag: str) -> bool:
        """Delete a model from BentoML store.

        Args:
            model_tag: Model tag to delete

        Returns:
            True if deleted successfully
        """
        try:
            bentoml.models.delete(model_tag)
            logger.info(f"Deleted model: {model_tag}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_tag}: {e}")
            return False


# ============================================================================
# BentoML Service Factory
# ============================================================================


def create_prediction_service(
    model_tag: str,
    service_name: str = "e2i_prediction_service",
    enable_preprocessing: bool = True,
) -> Type:
    """Create a BentoML prediction service class.

    This factory creates a service class that can be used to serve
    predictions from a trained model.

    Args:
        model_tag: BentoML model tag
        service_name: Name for the service
        enable_preprocessing: Whether to apply preprocessing

    Returns:
        BentoML service class
    """
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML is not installed")

    # Get model reference
    model_ref = bentoml.models.get(model_tag)
    framework = model_ref.info.metadata.get("framework", "sklearn")

    # Create service class dynamically
    @bentoml.service(
        name=service_name,
        resources={"cpu": "1", "memory": "2Gi"},
        traffic={"timeout": 60},
    )
    class E2IPredictionService:
        """BentoML prediction service for E2I models."""

        def __init__(self):
            """Initialize service with model and preprocessor."""
            self.model_tag = model_tag
            self.framework = framework
            self._model = None
            self._preprocessor = None
            self._load_model()

        def _load_model(self):
            """Load model and preprocessor from BentoML store."""
            try:
                model_ref = bentoml.models.get(self.model_tag)

                # Load model based on framework
                if self.framework == "sklearn":
                    self._model = bentoml.sklearn.load_model(self.model_tag)
                elif self.framework == "xgboost":
                    self._model = bentoml.xgboost.load_model(self.model_tag)
                elif self.framework == "lightgbm":
                    self._model = bentoml.lightgbm.load_model(self.model_tag)
                else:
                    self._model = bentoml.picklable_model.load_model(self.model_tag)

                # Try to load preprocessor from custom objects
                if enable_preprocessing and hasattr(model_ref.info, 'custom_objects'):
                    self._preprocessor = model_ref.info.custom_objects.get('preprocessor')

                logger.info(f"Loaded model: {self.model_tag}")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

        @bentoml.api
        async def predict(
            self,
            input_data: np.ndarray,
        ) -> np.ndarray:
            """Run prediction on input data.

            Args:
                input_data: Input feature array (samples x features)

            Returns:
                Predictions array
            """
            import time
            start_time = time.time()

            # Apply preprocessing if available
            if self._preprocessor is not None:
                input_data = self._preprocessor.transform(input_data)

            # Run prediction
            predictions = self._model.predict(input_data)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Prediction completed in {elapsed_ms:.2f}ms")

            return predictions

        @bentoml.api
        async def predict_proba(
            self,
            input_data: np.ndarray,
        ) -> np.ndarray:
            """Get prediction probabilities (classification only).

            Args:
                input_data: Input feature array

            Returns:
                Probability array (samples x classes)
            """
            # Apply preprocessing if available
            if self._preprocessor is not None:
                input_data = self._preprocessor.transform(input_data)

            # Check if model supports predict_proba
            if not hasattr(self._model, 'predict_proba'):
                raise ValueError("Model does not support probability predictions")

            return self._model.predict_proba(input_data)

        @bentoml.api
        async def health(self) -> Dict[str, Any]:
            """Health check endpoint.

            Returns:
                Health status dictionary
            """
            return {
                "status": "healthy",
                "model_id": self.model_tag,
                "model_loaded": self._model is not None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        @bentoml.api
        async def metadata(self) -> Dict[str, Any]:
            """Get model metadata.

            Returns:
                Model metadata dictionary
            """
            model_ref = bentoml.models.get(self.model_tag)
            return {
                "tag": str(model_ref.tag),
                "framework": self.framework,
                "metadata": model_ref.info.metadata,
                "creation_time": model_ref.info.creation_time.isoformat(),
            }

    return E2IPredictionService


# ============================================================================
# Model Packaging Utilities
# ============================================================================


class BentoPackager:
    """Packages models for deployment with BentoML.

    This class handles:
    - Creating Bento bundles from models
    - Configuring dependencies
    - Building Docker images
    """

    def __init__(self, model_manager: Optional[BentoMLModelManager] = None):
        """Initialize packager.

        Args:
            model_manager: Optional model manager instance
        """
        if not BENTOML_AVAILABLE:
            raise ImportError("BentoML is not installed")

        self.model_manager = model_manager or BentoMLModelManager()

    def create_bento(
        self,
        service_class: Type,
        bento_name: str,
        version: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        python_packages: Optional[List[str]] = None,
        docker_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a Bento bundle from a service.

        Args:
            service_class: BentoML service class
            bento_name: Name for the Bento
            version: Optional version string
            include: Files to include
            exclude: Files to exclude
            python_packages: Python dependencies
            docker_options: Docker build options

        Returns:
            Bento tag string
        """
        try:
            # Build Bento
            bento = bentoml.bentos.build(
                service=service_class,
                name=bento_name,
                version=version,
                include=include or [],
                exclude=exclude or [],
                python={
                    "packages": python_packages or self._get_default_packages(),
                },
                docker=docker_options or self._get_default_docker_options(),
            )

            bento_tag = str(bento.tag)
            logger.info(f"Created Bento: {bento_tag}")
            return bento_tag

        except Exception as e:
            logger.error(f"Failed to create Bento: {e}")
            raise

    def _get_default_packages(self) -> List[str]:
        """Get default Python packages for Bento."""
        return [
            "scikit-learn>=1.3.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "joblib>=1.3.0",
        ]

    def _get_default_docker_options(self) -> Dict[str, Any]:
        """Get default Docker options for Bento."""
        return {
            "python_version": "3.11",
            "distro": "debian",
        }

    def containerize(
        self,
        bento_tag: str,
        image_tag: Optional[str] = None,
        push: bool = False,
        registry: Optional[str] = None,
    ) -> str:
        """Build Docker container from Bento.

        Args:
            bento_tag: Bento tag to containerize
            image_tag: Optional custom image tag
            push: Whether to push to registry
            registry: Optional registry URL

        Returns:
            Docker image tag
        """
        try:
            # Build container image
            image = bentoml.container.build(
                bento_tag,
                image_tag=image_tag,
            )

            image_tag = image.tag
            logger.info(f"Built container image: {image_tag}")

            # Push if requested
            if push and registry:
                full_tag = f"{registry}/{image_tag}"
                bentoml.container.push(image_tag, repository=full_tag)
                logger.info(f"Pushed image to: {full_tag}")
                return full_tag

            return image_tag

        except Exception as e:
            logger.error(f"Failed to containerize Bento: {e}")
            raise

    def export_bento(
        self,
        bento_tag: str,
        output_path: str,
    ) -> str:
        """Export Bento to file system.

        Args:
            bento_tag: Bento tag to export
            output_path: Directory to export to

        Returns:
            Path to exported Bento
        """
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            export_path = bentoml.bentos.export_bento(
                bento_tag,
                output_path,
            )

            logger.info(f"Exported Bento to: {export_path}")
            return str(export_path)

        except Exception as e:
            logger.error(f"Failed to export Bento: {e}")
            raise


# ============================================================================
# High-Level Deployment Functions
# ============================================================================


async def register_model_for_serving(
    model: Any,
    model_name: str,
    metadata: Dict[str, Any],
    preprocessor: Optional[Any] = None,
    framework: str = "sklearn",
) -> Dict[str, Any]:
    """Register a trained model for BentoML serving.

    This is the main entry point for registering models trained by
    the model_trainer agent.

    Args:
        model: Trained model object
        model_name: Name for the model
        metadata: Model metadata (from training)
        preprocessor: Optional preprocessor
        framework: ML framework

    Returns:
        Dictionary with model_tag, service_class, registration_status
    """
    if not BENTOML_AVAILABLE:
        return {
            "registration_status": "unavailable",
            "error": "BentoML not installed",
        }

    try:
        manager = BentoMLModelManager()

        # Save model (with preprocessor if provided)
        if preprocessor is not None:
            model_tag = manager.save_model_with_preprocessor(
                model=model,
                preprocessor=preprocessor,
                model_name=model_name,
                metadata=metadata,
                framework=framework,
            )
        else:
            model_tag = manager.save_model(
                model=model,
                model_name=model_name,
                metadata=metadata,
                framework=framework,
            )

        # Create service class
        service_class = create_prediction_service(
            model_tag=model_tag,
            service_name=f"{model_name}_service",
        )

        return {
            "registration_status": "success",
            "model_tag": model_tag,
            "service_class": service_class,
            "model_info": manager.get_model_info(model_tag),
        }

    except Exception as e:
        logger.error(f"Failed to register model for serving: {e}")
        return {
            "registration_status": "failed",
            "error": str(e),
        }


async def deploy_model(
    model_tag: str,
    deployment_name: str,
    replicas: int = 1,
    resources: Optional[Dict[str, str]] = None,
    environment: str = "staging",
    model_registry_id: Optional[str] = None,
    deployed_by: Optional[str] = None,
    supabase_client=None,
) -> Dict[str, Any]:
    """Deploy a registered model to production.

    Args:
        model_tag: BentoML model tag
        deployment_name: Name for the deployment
        replicas: Number of replicas
        resources: Resource limits
        environment: Deployment environment (staging, production)
        model_registry_id: UUID of model in ml_model_registry
        deployed_by: Username of deployer
        supabase_client: Optional Supabase client for DB persistence

    Returns:
        Deployment status dictionary
    """
    if not BENTOML_AVAILABLE:
        return {
            "deployment_status": "unavailable",
            "error": "BentoML not installed",
        }

    resources = resources or {"cpu": "1", "memory": "2Gi"}

    try:
        # Create service
        service_class = create_prediction_service(
            model_tag=model_tag,
            service_name=deployment_name,
        )

        # Package into Bento
        packager = BentoPackager()
        bento_tag = packager.create_bento(
            service_class=service_class,
            bento_name=deployment_name,
        )

        # Build container
        image_tag = packager.containerize(bento_tag)

        result = {
            "deployment_status": "success",
            "bento_tag": bento_tag,
            "image_tag": image_tag,
            "deployment_name": deployment_name,
            "replicas": replicas,
            "resources": resources,
            "environment": environment,
        }

        # Persist deployment record to database
        deployment_record = await _persist_deployment(
            deployment_name=deployment_name,
            model_registry_id=model_registry_id,
            environment=environment,
            deployed_by=deployed_by,
            deployment_config={
                "bento_tag": bento_tag,
                "image_tag": image_tag,
                "replicas": replicas,
                "resources": resources,
            },
            supabase_client=supabase_client,
        )
        if deployment_record:
            result["deployment_id"] = str(deployment_record.id)

        return result

    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        return {
            "deployment_status": "failed",
            "error": str(e),
        }


async def _persist_deployment(
    deployment_name: str,
    model_registry_id: Optional[str] = None,
    environment: str = "staging",
    deployed_by: Optional[str] = None,
    deployment_config: Optional[Dict[str, Any]] = None,
    supabase_client=None,
) -> Optional[Any]:
    """Persist deployment record to ml_deployments table.

    Args:
        deployment_name: Name of the deployment
        model_registry_id: UUID of model in ml_model_registry
        environment: Deployment environment
        deployed_by: Username of deployer
        deployment_config: Configuration details
        supabase_client: Supabase client

    Returns:
        MLDeployment record or None if persistence failed
    """
    try:
        from uuid import UUID
        from src.repositories.deployment import (
            MLDeploymentRepository,
            DeploymentStatus,
        )

        repo = MLDeploymentRepository(supabase_client)

        # Convert model_registry_id if provided
        registry_uuid = UUID(model_registry_id) if model_registry_id else None

        deployment = await repo.create_deployment(
            model_registry_id=registry_uuid,
            deployment_name=deployment_name,
            environment=environment,
            deployed_by=deployed_by,
            deployment_config=deployment_config or {},
        )

        # Update status to active
        if deployment and deployment.id:
            await repo.update_status(deployment.id, DeploymentStatus.ACTIVE)
            logger.info(f"Persisted deployment: {deployment_name} (ID: {deployment.id})")

        return deployment

    except Exception as e:
        logger.warning(f"Failed to persist deployment record: {e}")
        return None


def get_model_serving_status(model_tag: str) -> Dict[str, Any]:
    """Get serving status for a model.

    Args:
        model_tag: BentoML model tag

    Returns:
        Status dictionary
    """
    if not BENTOML_AVAILABLE:
        return {
            "status": "unavailable",
            "error": "BentoML not installed",
        }

    try:
        manager = BentoMLModelManager()
        model_info = manager.get_model_info(model_tag)

        return {
            "status": "registered",
            "model_info": model_info,
        }

    except Exception as e:
        return {
            "status": "not_found",
            "error": str(e),
        }


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Models
    "PredictionInput",
    "PredictionOutput",
    "HealthResponse",
    "ModelMetadata",
    # Classes
    "BentoMLModelManager",
    "BentoPackager",
    # Factory
    "create_prediction_service",
    # High-level functions
    "register_model_for_serving",
    "deploy_model",
    "get_model_serving_status",
    # Constants
    "BENTOML_AVAILABLE",
]
