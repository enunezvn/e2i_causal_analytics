"""
E2I Real-Time SHAP Explainer
=============================
Optimized SHAP computation for real-time API responses.

Key Optimizations:
1. Explainer caching (one explainer per model version)
2. Background data sampling for KernelExplainer
3. TreeExplainer for tree-based models (fastest)
4. Async-compatible interface

Integration:
- Called by api/routes/explain.py
- Uses BentoML-served models
- Stores results in ml_shap_analyses

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, cast

import mlflow
import numpy as np
import shap

logger = logging.getLogger(__name__)

# Environment configuration for model loading
# Set E2I_USE_MOCK_MODELS=true for development/testing without MLflow
_USE_MOCK_MODELS = os.environ.get("E2I_USE_MOCK_MODELS", "false").lower() == "true"

# Thread pool for CPU-bound SHAP computations
_executor = ThreadPoolExecutor(max_workers=4)


class ExplainerType(str, Enum):
    """Supported SHAP explainer types."""

    TREE = "TreeExplainer"
    KERNEL = "KernelExplainer"
    LINEAR = "LinearExplainer"
    DEEP = "DeepExplainer"
    GRADIENT = "GradientExplainer"


@dataclass
class ExplainerConfig:
    """Configuration for SHAP explainer."""

    explainer_type: ExplainerType
    background_sample_size: int = 100  # For KernelExplainer
    max_features: Optional[int] = None  # Limit features for speed
    cache_ttl_seconds: int = 3600  # How long to cache explainer
    feature_names: List[str] = field(default_factory=list)


@dataclass
class SHAPResult:
    """Result of SHAP computation."""

    shap_values: Dict[str, float]
    base_value: float
    expected_value: float
    computation_time_ms: float
    explainer_type: ExplainerType
    feature_count: int
    model_version_id: str
    features_anonymized: bool = False  # L5: Indicates if feature names were anonymized
    feature_name_mapping: Optional[Dict[str, str]] = (
        None  # L5: Mapping from anonymized to original names
    )


class RealTimeSHAPExplainer:
    """
    Real-time SHAP explainer with optimization for production use.

    Features:
    - Explainer caching per model version
    - Background data management
    - Async-compatible interface
    - Multiple explainer type support
    """

    def __init__(
        self,
        bentoml_client=None,
        background_data_cache_size: int = 1000,
        default_background_sample: int = 100,
    ):
        self.bentoml_client = bentoml_client
        self.background_data_cache_size = background_data_cache_size
        self.default_background_sample = default_background_sample

        # Cache for explainers (keyed by model_version_id)
        self._explainer_cache: Dict[str, Tuple[Any, datetime, ExplainerConfig]] = {}

        # Cache for background data (for KernelExplainer)
        self._background_cache: Dict[str, np.ndarray] = {}

    def _get_explainer_type_from_model(self, model: Any) -> ExplainerType:
        """
        Determine optimal explainer type by inspecting the actual model object.

        This is the robust approach that works with any sklearn-compatible model.
        Uses the same logic as shap_computer.py for consistency.

        Args:
            model: The trained model object

        Returns:
            Appropriate ExplainerType for the model
        """
        model_class = model.__class__.__name__

        # Tree-based models (fastest SHAP computation)
        tree_models = {
            "RandomForestClassifier",
            "RandomForestRegressor",
            "GradientBoostingClassifier",
            "GradientBoostingRegressor",
            "XGBClassifier",
            "XGBRegressor",
            "LGBMClassifier",
            "LGBMRegressor",
            "CatBoostClassifier",
            "CatBoostRegressor",
            "DecisionTreeClassifier",
            "DecisionTreeRegressor",
            "ExtraTreesClassifier",
            "ExtraTreesRegressor",
            "AdaBoostClassifier",
            "AdaBoostRegressor",
        }

        if model_class in tree_models:
            return ExplainerType.TREE

        # Linear models
        linear_models = {
            "LinearRegression",
            "LogisticRegression",
            "Ridge",
            "RidgeClassifier",
            "Lasso",
            "ElasticNet",
            "SGDClassifier",
            "SGDRegressor",
            "Perceptron",
            "PassiveAggressiveClassifier",
            "PassiveAggressiveRegressor",
        }

        if model_class in linear_models:
            return ExplainerType.LINEAR

        # Deep learning models (PyTorch/TensorFlow)
        deep_models = {
            "Sequential",  # Keras
            "Module",  # PyTorch base class
            "MLPClassifier",
            "MLPRegressor",
        }

        if model_class in deep_models:
            return ExplainerType.DEEP

        # Check for common deep learning patterns in class hierarchy
        model_module = model.__class__.__module__
        if "torch" in model_module or "tensorflow" in model_module or "keras" in model_module:
            return ExplainerType.DEEP

        # Fallback to KernelExplainer (model-agnostic, slower)
        return ExplainerType.KERNEL

    def _get_explainer_type(self, model_type: str, model: Any = None) -> ExplainerType:
        """
        Determine optimal explainer type.

        Primary method: Inspect the actual model object (robust, works with any model).
        Fallback method: Use business model name hints (for backward compatibility).

        Args:
            model_type: Business model name (e.g., "propensity")
            model: Optional trained model object for accurate detection

        Returns:
            Appropriate ExplainerType for the model
        """
        # Primary: Use model object inspection if available
        if model is not None:
            return self._get_explainer_type_from_model(model)

        # Fallback: Use business model name hints (backward compatibility)
        # These are E2I-specific conventions for when model object isn't available
        tree_model_hints = {
            "propensity",
            "risk_stratification",
            "churn_prediction",
            "xgboost",
            "lightgbm",
            "random_forest",
            "gradient_boosting",
        }
        linear_model_hints = {"baseline_logistic", "logistic", "linear", "ridge", "lasso"}
        deep_model_hints = {"deep_nba", "neural_propensity", "neural", "mlp", "deep"}

        model_type_lower = model_type.lower()

        if any(hint in model_type_lower for hint in tree_model_hints):
            return ExplainerType.TREE
        elif any(hint in model_type_lower for hint in linear_model_hints):
            return ExplainerType.LINEAR
        elif any(hint in model_type_lower for hint in deep_model_hints):
            return ExplainerType.DEEP
        else:
            return ExplainerType.KERNEL

    async def get_explainer(
        self,
        model_type: str,
        model_version_id: str,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        model_uri: Optional[str] = None,
    ) -> Tuple[Any, ExplainerConfig]:
        """
        Get or create SHAP explainer for a model version.

        Uses caching to avoid recreating explainers on every request.

        Args:
            model_type: Business model type name
            model_version_id: Model version identifier
            feature_names: List of feature names
            background_data: Optional background data for KernelExplainer
            model_uri: Optional MLflow model URI for production use

        Returns:
            Tuple of (explainer, config)
        """
        cache_key = f"{model_type}:{model_version_id}"

        # Check cache
        if cache_key in self._explainer_cache:
            explainer, created_at, config = self._explainer_cache[cache_key]
            cache_age = (datetime.now(timezone.utc) - created_at).total_seconds()
            if cache_age < config.cache_ttl_seconds:
                logger.debug(f"Using cached explainer for {cache_key}")
                return explainer, config

        # Get model for SHAP computation
        # In production: loads from MLflow
        # In development (E2I_USE_MOCK_MODELS=true): uses mock model
        model = self._get_model(model_type, model_version_id, model_uri)

        # Determine explainer type by inspecting the actual model object
        # This is more robust than relying on business model names
        explainer_type = self._get_explainer_type(model_type, model=model)
        config = ExplainerConfig(explainer_type=explainer_type, feature_names=feature_names)

        # Create appropriate explainer
        loop = asyncio.get_event_loop()
        explainer = await loop.run_in_executor(
            _executor, self._create_explainer, model, explainer_type, background_data, feature_names
        )

        # Cache it
        self._explainer_cache[cache_key] = (explainer, datetime.now(timezone.utc), config)

        return explainer, config

    def _create_explainer(
        self,
        model: Any,
        explainer_type: ExplainerType,
        background_data: Optional[np.ndarray],
        feature_names: List[str],
    ) -> Any:
        """
        Create SHAP explainer (CPU-bound, runs in thread pool).

        Raises:
            ValueError: If explainer type is unknown or model is incompatible.
            RuntimeError: If explainer creation fails due to SHAP library errors.
        """
        try:
            if explainer_type == ExplainerType.TREE:
                # TreeExplainer is fastest for tree-based models
                return shap.TreeExplainer(model)

            elif explainer_type == ExplainerType.LINEAR:
                if background_data is None:
                    raise ValueError(
                        "LinearExplainer requires background_data but None was provided"
                    )
                return shap.LinearExplainer(model, background_data)

            elif explainer_type == ExplainerType.KERNEL:
                # KernelExplainer needs background data
                if background_data is None:
                    # Generate synthetic background if not provided
                    background_data = self._generate_synthetic_background(feature_names)

                # Sample background for speed
                if len(background_data) > self.default_background_sample:
                    indices = np.random.choice(
                        len(background_data), self.default_background_sample, replace=False
                    )
                    background_data = background_data[indices]

                return shap.KernelExplainer(model.predict_proba, background_data)

            elif explainer_type == ExplainerType.DEEP:
                if background_data is None:
                    raise ValueError("DeepExplainer requires background_data but None was provided")
                return shap.DeepExplainer(model, background_data)

            elif explainer_type == ExplainerType.GRADIENT:
                if background_data is None:
                    raise ValueError(
                        "GradientExplainer requires background_data but None was provided"
                    )
                return shap.GradientExplainer(model, background_data)

            else:
                raise ValueError(f"Unknown explainer type: {explainer_type}")

        except ValueError:
            # Re-raise ValueError (validation errors) as-is
            raise
        except AttributeError as e:
            # Model doesn't have expected interface (e.g., missing predict_proba)
            raise ValueError(
                f"Model incompatible with {explainer_type.value} explainer: {e}"
            ) from e
        except Exception as e:
            # Wrap SHAP library errors in RuntimeError with context
            logger.error(
                f"Failed to create {explainer_type.value} explainer: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"SHAP explainer creation failed for {explainer_type.value}: {e}"
            ) from e

    def _generate_synthetic_background(
        self,
        feature_names: List[str],
        feast_client: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Generate background data for SHAP explainers.

        Attempts to use real data from Feast feature store first. Falls back to
        domain-aware synthetic data based on feature name patterns.

        Args:
            feature_names: List of feature names
            feast_client: Optional Feast client for retrieving real feature statistics

        Returns:
            Background data array of shape (n_samples, n_features)
        """
        n_samples = self.default_background_sample
        len(feature_names)

        # Try to get real data from Feast first
        if feast_client is not None:
            try:
                background = self._get_background_from_feast(feast_client, feature_names, n_samples)
                if background is not None:
                    logger.info("Using real feature data from Feast for SHAP background")
                    return background
            except Exception as e:
                logger.warning(f"Failed to get Feast data, using synthetic: {e}")

        # Generate domain-aware synthetic data based on feature name patterns
        logger.info("Using domain-aware synthetic data for SHAP background")
        return self._generate_domain_aware_background(feature_names, n_samples)

    def _get_background_from_feast(
        self,
        feast_client: Any,
        feature_names: List[str],
        n_samples: int,
    ) -> Optional[np.ndarray]:
        """
        Attempt to retrieve background data from Feast feature store.

        Args:
            feast_client: Feast client instance
            feature_names: Features to retrieve
            n_samples: Number of samples to retrieve

        Returns:
            Background array or None if retrieval fails
        """
        try:
            # Try to get historical features for background
            if hasattr(feast_client, "get_historical_features"):
                # Create entity DataFrame with sample entity IDs
                import pandas as pd

                entity_df = pd.DataFrame(
                    {
                        "patient_id": [f"sample_{i}" for i in range(n_samples)],
                    }
                )

                features = feast_client.get_historical_features(
                    entity_df=entity_df,
                    features=[f"patient_features:{f}" for f in feature_names],
                )
                df = features.to_df()

                # Extract feature columns and convert to numpy
                feature_cols = [c for c in df.columns if c in feature_names]
                if feature_cols:
                    return cast(np.ndarray, df[feature_cols].fillna(0).values)

            return None
        except Exception as e:
            logger.debug(f"Feast retrieval failed: {e}")
            return None

    def _generate_domain_aware_background(
        self,
        feature_names: List[str],
        n_samples: int,
    ) -> np.ndarray:
        """
        Generate synthetic background data using domain-aware distributions.

        Uses feature name patterns to infer appropriate distributions:
        - Binary features (is_*, has_*): Bernoulli(0.3)
        - Count features (*_count, num_*): Poisson(5)
        - Rate/score features (*_rate, *_score): Beta(2, 5) scaled
        - Age features (*_age*, *_days*): Gamma(5, 10)
        - Default: Standard normal

        Args:
            feature_names: List of feature names
            n_samples: Number of samples to generate

        Returns:
            Background array of shape (n_samples, n_features)
        """
        n_features = len(feature_names)
        background = np.zeros((n_samples, n_features))

        for i, name in enumerate(feature_names):
            name_lower = name.lower()

            if name_lower.startswith(("is_", "has_", "flag_")):
                # Binary features: Bernoulli distribution
                background[:, i] = np.random.binomial(1, 0.3, n_samples)

            elif any(x in name_lower for x in ["_count", "num_", "total_", "n_"]):
                # Count features: Poisson distribution
                background[:, i] = np.random.poisson(5, n_samples)

            elif any(x in name_lower for x in ["_rate", "_ratio", "_pct", "_percent"]):
                # Rate features: Beta distribution scaled to [0, 1]
                background[:, i] = np.random.beta(2, 5, n_samples)

            elif any(x in name_lower for x in ["_score", "score_"]):
                # Score features: Beta distribution scaled to [0, 100]
                background[:, i] = np.random.beta(2, 2, n_samples) * 100

            elif any(x in name_lower for x in ["_age", "age_", "_days", "days_"]):
                # Age/duration features: Gamma distribution (always positive)
                background[:, i] = np.random.gamma(5, 10, n_samples)

            elif any(x in name_lower for x in ["_amount", "amount_", "_value", "value_"]):
                # Monetary/value features: Log-normal distribution
                background[:, i] = np.random.lognormal(3, 1, n_samples)

            else:
                # Default: Standard normal (mean=0, std=1)
                background[:, i] = np.random.randn(n_samples)

        return background

    def _load_model_from_mlflow(self, model_uri: str) -> Any:
        """
        Load model from MLflow model registry.

        Uses the same pattern as shap_computer.py for consistency.

        Args:
            model_uri: MLflow model URI (e.g., "runs:/abc123/model" or
                       "models:/my-model/Production")

        Returns:
            Loaded sklearn-compatible model object
        """
        try:
            # Try loading as sklearn model first (most common)
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from MLflow: {model_uri}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load as sklearn model, trying pyfunc: {e}")
            try:
                # Fallback to pyfunc loader for other model types
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Loaded model as pyfunc from MLflow: {model_uri}")
                return model
            except Exception as e2:
                logger.error(f"Failed to load model from MLflow: {e2}")
                raise RuntimeError(f"Could not load model from MLflow: {model_uri}") from e2

    def _get_mock_model(self, model_type: str) -> Any:
        """
        Get a mock model for development/testing purposes.

        IMPORTANT: This method should ONLY be used when E2I_USE_MOCK_MODELS=true.
        In production, models should be loaded from MLflow via _load_model_from_mlflow().

        Args:
            model_type: Business model type name (used for selecting appropriate mock)

        Returns:
            A fitted mock model for SHAP computation

        Raises:
            RuntimeError: If called in production without E2I_USE_MOCK_MODELS=true
        """
        if not _USE_MOCK_MODELS:
            raise RuntimeError(
                "Mock models are disabled in production. "
                "Set E2I_USE_MOCK_MODELS=true for development/testing, "
                "or provide a valid MLflow model_uri."
            )

        logger.warning(
            f"Using mock model for {model_type}. "
            "This is only for development/testing. Do NOT use in production."
        )

        from sklearn.ensemble import GradientBoostingClassifier

        # Create a simple mock model that matches the expected interface
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        return model

    def _get_model(
        self, model_type: str, model_version_id: str, model_uri: Optional[str] = None
    ) -> Any:
        """
        Get model for SHAP computation.

        In production: Loads from MLflow using model_uri.
        In development: Uses mock model when E2I_USE_MOCK_MODELS=true.

        Args:
            model_type: Business model type name
            model_version_id: Model version identifier
            model_uri: Optional MLflow model URI. If None and not in mock mode,
                       constructs URI from model_type and model_version_id.

        Returns:
            Model object suitable for SHAP computation
        """
        # If mock models are enabled, use mock (for development/testing)
        if _USE_MOCK_MODELS:
            return self._get_mock_model(model_type)

        # Production: Load from MLflow
        if model_uri:
            return self._load_model_from_mlflow(model_uri)

        # Construct MLflow URI from model type and version
        # Convention: models:/{model_type}/{model_version_id}
        constructed_uri = f"models:/{model_type}/{model_version_id}"
        logger.info(f"Constructed MLflow URI: {constructed_uri}")
        return self._load_model_from_mlflow(constructed_uri)

    async def compute_shap_values(
        self,
        features: Dict[str, Any],
        model_type: str,
        model_version_id: str,
        top_k: Optional[int] = None,
        model_uri: Optional[str] = None,
        anonymize_features: bool = False,
    ) -> SHAPResult:
        """
        Compute SHAP values for a single instance.

        Args:
            features: Feature dict for the instance
            model_type: Type of model
            model_version_id: Specific model version
            top_k: If provided, return only top K features by importance
            model_uri: Optional MLflow model URI for production use
            anonymize_features: If True, anonymize feature names in output (L5 security)

        Returns:
            SHAPResult with SHAP values and metadata
        """
        import time

        start_time = time.time()

        feature_names = list(features.keys())
        feature_values = np.array([list(features.values())])

        # L5: Apply feature name anonymization if requested
        feature_name_mapping = None
        output_feature_names = feature_names
        if anonymize_features:
            output_feature_names, feature_name_mapping = self._anonymize_feature_names(
                feature_names
            )

        # Get explainer (uses original feature names internally)
        explainer, config = await self.get_explainer(
            model_type=model_type,
            model_version_id=model_version_id,
            feature_names=feature_names,  # Use original names for computation
            model_uri=model_uri,
        )

        # Compute SHAP values (CPU-bound, run in thread pool)
        loop = asyncio.get_event_loop()
        shap_values, expected_value = await loop.run_in_executor(
            _executor, self._compute_shap_sync, explainer, feature_values, config.explainer_type
        )

        # Convert to dict (use output feature names which may be anonymized)
        shap_dict = dict(zip(output_feature_names, shap_values.flatten(), strict=False))

        # Optionally filter to top K
        if top_k is not None:
            sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[
                :top_k
            ]
            shap_dict = dict(sorted_features)

        computation_time_ms = (time.time() - start_time) * 1000

        return SHAPResult(
            shap_values=shap_dict,
            base_value=float(expected_value),
            expected_value=float(expected_value),
            computation_time_ms=round(computation_time_ms, 2),
            explainer_type=config.explainer_type,
            feature_count=len(feature_names),
            model_version_id=model_version_id,
            features_anonymized=anonymize_features,
            feature_name_mapping=feature_name_mapping,
        )

    def _compute_shap_sync(
        self, explainer: Any, features: np.ndarray, explainer_type: ExplainerType
    ) -> Tuple[np.ndarray, float]:
        """
        Synchronous SHAP computation (runs in thread pool).
        """
        shap_values = explainer.shap_values(features)

        # Handle different return formats
        if isinstance(shap_values, list):
            # Multi-class: take positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Get expected value
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

        return shap_values, float(expected_value)

    async def compute_batch_shap(
        self,
        features_batch: List[Dict[str, Any]],
        model_type: str,
        model_version_id: str,
        top_k: Optional[int] = None,
        model_uri: Optional[str] = None,
        anonymize_features: bool = False,
    ) -> List[SHAPResult]:
        """
        Compute SHAP values for a batch of instances.

        More efficient than calling compute_shap_values in a loop
        because explainer is reused.

        Args:
            features_batch: List of feature dicts for each instance
            model_type: Type of model
            model_version_id: Specific model version
            top_k: If provided, return only top K features by importance
            model_uri: Optional MLflow model URI for production use
            anonymize_features: If True, anonymize feature names in output (L5 security)

        Returns:
            List of SHAPResult with SHAP values and metadata
        """
        if not features_batch:
            return []

        # All instances should have same features
        feature_names = list(features_batch[0].keys())
        feature_values = np.array([list(f.values()) for f in features_batch])

        # L5: Apply feature name anonymization if requested
        feature_name_mapping = None
        output_feature_names = feature_names
        if anonymize_features:
            output_feature_names, feature_name_mapping = self._anonymize_feature_names(
                feature_names
            )

        # Get explainer (uses original feature names internally)
        explainer, config = await self.get_explainer(
            model_type=model_type,
            model_version_id=model_version_id,
            feature_names=feature_names,  # Use original names for computation
            model_uri=model_uri,
        )

        # Batch compute
        import time

        start_time = time.time()

        loop = asyncio.get_event_loop()
        shap_values, expected_value = await loop.run_in_executor(
            _executor, self._compute_shap_sync, explainer, feature_values, config.explainer_type
        )

        total_time_ms = (time.time() - start_time) * 1000
        per_instance_time = total_time_ms / len(features_batch)

        # Convert to individual results (use output feature names which may be anonymized)
        results = []
        for i, _features in enumerate(features_batch):
            instance_shap = shap_values[i] if len(shap_values.shape) > 1 else shap_values
            shap_dict = dict(zip(output_feature_names, instance_shap.flatten(), strict=False))

            if top_k is not None:
                sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[
                    :top_k
                ]
                shap_dict = dict(sorted_features)

            results.append(
                SHAPResult(
                    shap_values=shap_dict,
                    base_value=float(expected_value),
                    expected_value=float(expected_value),
                    computation_time_ms=round(per_instance_time, 2),
                    explainer_type=config.explainer_type,
                    feature_count=len(feature_names),
                    model_version_id=model_version_id,
                    features_anonymized=anonymize_features,
                    feature_name_mapping=feature_name_mapping,
                )
            )

        return results

    def clear_cache(self, model_version_id: Optional[str] = None):
        """
        Clear explainer cache.

        Args:
            model_version_id: If provided, clear only that version. Otherwise clear all.
        """
        if model_version_id:
            keys_to_remove = [k for k in self._explainer_cache if model_version_id in k]
            for key in keys_to_remove:
                del self._explainer_cache[key]
            logger.info(f"Cleared cache for {len(keys_to_remove)} explainers")
        else:
            self._explainer_cache.clear()
            logger.info("Cleared all explainer cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the explainer cache."""
        return {
            "cached_explainers": len(self._explainer_cache),
            "cached_models": list(self._explainer_cache.keys()),
            "background_data_cached": len(self._background_cache),
        }

    @staticmethod
    def _anonymize_feature_names(
        feature_names: List[str],
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Anonymize feature names to prevent schema information leakage.

        Args:
            feature_names: Original feature names

        Returns:
            Tuple of (anonymized_names, mapping from anonymous to original)
        """
        anonymized = [f"feature_{i}" for i in range(len(feature_names))]
        mapping = dict(zip(anonymized, feature_names, strict=False))
        return anonymized, mapping


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================


class SHAPVisualization:
    """
    Generate SHAP visualization data for frontend rendering.

    Note: This generates data for visualization, not the actual plots.
    Frontend should use this data with Chart.js or similar.
    """

    @staticmethod
    def generate_waterfall_data(
        shap_result: SHAPResult, features: Dict[str, Any], top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate data for SHAP waterfall plot.

        Returns data structure that frontend can render as waterfall chart.
        """
        # Sort by absolute SHAP value
        sorted_items = sorted(
            shap_result.shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )[:top_k]

        waterfall_data: Dict[str, Any] = {
            "base_value": shap_result.base_value,
            "final_value": shap_result.base_value + sum(shap_result.shap_values.values()),
            "features": [],
        }

        cumulative = shap_result.base_value
        for feature_name, shap_value in sorted_items:
            feature_value = features.get(feature_name, "N/A")
            waterfall_data["features"].append(
                {
                    "name": feature_name,
                    "value": feature_value,
                    "shap_value": shap_value,
                    "direction": "positive" if shap_value > 0 else "negative",
                    "cumulative_start": cumulative,
                    "cumulative_end": cumulative + shap_value,
                }
            )
            cumulative += shap_value

        return waterfall_data

    @staticmethod
    def generate_force_plot_data(
        shap_result: SHAPResult, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate data for SHAP force plot.
        """
        positive_features = []
        negative_features = []

        for feature_name, shap_value in shap_result.shap_values.items():
            feature_info = {
                "name": feature_name,
                "value": features.get(feature_name, "N/A"),
                "shap_value": abs(shap_value),
            }
            if shap_value > 0:
                positive_features.append(feature_info)
            else:
                negative_features.append(feature_info)

        # Sort by absolute SHAP value
        positive_features.sort(key=lambda x: x["shap_value"], reverse=True)
        negative_features.sort(key=lambda x: x["shap_value"], reverse=True)

        return {
            "base_value": shap_result.base_value,
            "output_value": shap_result.base_value + sum(shap_result.shap_values.values()),
            "positive_features": positive_features,
            "negative_features": negative_features,
            "link": "identity",  # or "logit" for classification
        }

    @staticmethod
    def generate_bar_chart_data(shap_result: SHAPResult, top_k: int = 10) -> Dict[str, Any]:
        """
        Generate data for feature importance bar chart.
        """
        sorted_items = sorted(
            shap_result.shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )[:top_k]

        return {
            "features": [item[0] for item in sorted_items],
            "values": [item[1] for item in sorted_items],
            "absolute_values": [abs(item[1]) for item in sorted_items],
            "colors": ["#ff6b6b" if item[1] > 0 else "#4dabf7" for item in sorted_items],
        }
