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

import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timezone
from functools import lru_cache
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

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
        default_background_sample: int = 100
    ):
        self.bentoml_client = bentoml_client
        self.background_data_cache_size = background_data_cache_size
        self.default_background_sample = default_background_sample
        
        # Cache for explainers (keyed by model_version_id)
        self._explainer_cache: Dict[str, Tuple[Any, datetime, ExplainerConfig]] = {}
        
        # Cache for background data (for KernelExplainer)
        self._background_cache: Dict[str, np.ndarray] = {}
    
    def _get_explainer_type(self, model_type: str) -> ExplainerType:
        """
        Determine optimal explainer type based on model type.
        
        Tree-based models (XGBoost, LightGBM, RandomForest) → TreeExplainer (fast)
        Neural networks → DeepExplainer or GradientExplainer
        Linear models → LinearExplainer
        Others → KernelExplainer (model-agnostic, slower)
        """
        tree_models = {"propensity", "risk_stratification", "churn_prediction"}
        linear_models = {"baseline_logistic"}
        deep_models = {"deep_nba", "neural_propensity"}
        
        if model_type.lower() in tree_models:
            return ExplainerType.TREE
        elif model_type.lower() in linear_models:
            return ExplainerType.LINEAR
        elif model_type.lower() in deep_models:
            return ExplainerType.DEEP
        else:
            return ExplainerType.KERNEL
    
    async def get_explainer(
        self,
        model_type: str,
        model_version_id: str,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None
    ) -> Tuple[Any, ExplainerConfig]:
        """
        Get or create SHAP explainer for a model version.
        
        Uses caching to avoid recreating explainers on every request.
        """
        cache_key = f"{model_type}:{model_version_id}"
        
        # Check cache
        if cache_key in self._explainer_cache:
            explainer, created_at, config = self._explainer_cache[cache_key]
            cache_age = (datetime.now(timezone.utc) - created_at).total_seconds()
            if cache_age < config.cache_ttl_seconds:
                logger.debug(f"Using cached explainer for {cache_key}")
                return explainer, config
        
        # Create new explainer
        explainer_type = self._get_explainer_type(model_type)
        config = ExplainerConfig(
            explainer_type=explainer_type,
            feature_names=feature_names
        )
        
        # Get model from BentoML
        # In production: model = await self.bentoml_client.get_model(model_type, model_version_id)
        model = self._get_mock_model(model_type)
        
        # Create appropriate explainer
        loop = asyncio.get_event_loop()
        explainer = await loop.run_in_executor(
            _executor,
            self._create_explainer,
            model,
            explainer_type,
            background_data,
            feature_names
        )
        
        # Cache it
        self._explainer_cache[cache_key] = (explainer, datetime.now(timezone.utc), config)
        
        return explainer, config
    
    def _create_explainer(
        self,
        model: Any,
        explainer_type: ExplainerType,
        background_data: Optional[np.ndarray],
        feature_names: List[str]
    ) -> Any:
        """
        Create SHAP explainer (CPU-bound, runs in thread pool).
        """
        if explainer_type == ExplainerType.TREE:
            # TreeExplainer is fastest for tree-based models
            return shap.TreeExplainer(model)
        
        elif explainer_type == ExplainerType.LINEAR:
            return shap.LinearExplainer(model, background_data)
        
        elif explainer_type == ExplainerType.KERNEL:
            # KernelExplainer needs background data
            if background_data is None:
                # Generate synthetic background if not provided
                background_data = self._generate_synthetic_background(feature_names)
            
            # Sample background for speed
            if len(background_data) > self.default_background_sample:
                indices = np.random.choice(
                    len(background_data), 
                    self.default_background_sample, 
                    replace=False
                )
                background_data = background_data[indices]
            
            return shap.KernelExplainer(model.predict_proba, background_data)
        
        elif explainer_type == ExplainerType.DEEP:
            return shap.DeepExplainer(model, background_data)
        
        elif explainer_type == ExplainerType.GRADIENT:
            return shap.GradientExplainer(model, background_data)
        
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
    
    def _generate_synthetic_background(self, feature_names: List[str]) -> np.ndarray:
        """
        Generate synthetic background data when real data isn't available.
        Uses domain knowledge for realistic distributions.
        """
        n_samples = self.default_background_sample
        n_features = len(feature_names)
        
        # Generate random data with reasonable distributions
        # In production, this would use actual data statistics from Feast
        return np.random.randn(n_samples, n_features)
    
    def _get_mock_model(self, model_type: str):
        """Mock model for demonstration. Replace with BentoML client call."""
        from sklearn.ensemble import GradientBoostingClassifier
        import numpy as np
        
        # Create a simple mock model
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        return model
    
    async def compute_shap_values(
        self,
        features: Dict[str, Any],
        model_type: str,
        model_version_id: str,
        top_k: Optional[int] = None
    ) -> SHAPResult:
        """
        Compute SHAP values for a single instance.
        
        Args:
            features: Feature dict for the instance
            model_type: Type of model
            model_version_id: Specific model version
            top_k: If provided, return only top K features by importance
        
        Returns:
            SHAPResult with SHAP values and metadata
        """
        import time
        start_time = time.time()
        
        feature_names = list(features.keys())
        feature_values = np.array([list(features.values())])
        
        # Get explainer
        explainer, config = await self.get_explainer(
            model_type=model_type,
            model_version_id=model_version_id,
            feature_names=feature_names
        )
        
        # Compute SHAP values (CPU-bound, run in thread pool)
        loop = asyncio.get_event_loop()
        shap_values, expected_value = await loop.run_in_executor(
            _executor,
            self._compute_shap_sync,
            explainer,
            feature_values,
            config.explainer_type
        )
        
        # Convert to dict
        shap_dict = dict(zip(feature_names, shap_values.flatten()))
        
        # Optionally filter to top K
        if top_k is not None:
            sorted_features = sorted(
                shap_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_k]
            shap_dict = dict(sorted_features)
        
        computation_time_ms = (time.time() - start_time) * 1000
        
        return SHAPResult(
            shap_values=shap_dict,
            base_value=float(expected_value),
            expected_value=float(expected_value),
            computation_time_ms=round(computation_time_ms, 2),
            explainer_type=config.explainer_type,
            feature_count=len(feature_names),
            model_version_id=model_version_id
        )
    
    def _compute_shap_sync(
        self,
        explainer: Any,
        features: np.ndarray,
        explainer_type: ExplainerType
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
        top_k: Optional[int] = None
    ) -> List[SHAPResult]:
        """
        Compute SHAP values for a batch of instances.
        
        More efficient than calling compute_shap_values in a loop
        because explainer is reused.
        """
        if not features_batch:
            return []
        
        # All instances should have same features
        feature_names = list(features_batch[0].keys())
        feature_values = np.array([list(f.values()) for f in features_batch])
        
        # Get explainer
        explainer, config = await self.get_explainer(
            model_type=model_type,
            model_version_id=model_version_id,
            feature_names=feature_names
        )
        
        # Batch compute
        import time
        start_time = time.time()
        
        loop = asyncio.get_event_loop()
        shap_values, expected_value = await loop.run_in_executor(
            _executor,
            self._compute_shap_sync,
            explainer,
            feature_values,
            config.explainer_type
        )
        
        total_time_ms = (time.time() - start_time) * 1000
        per_instance_time = total_time_ms / len(features_batch)
        
        # Convert to individual results
        results = []
        for i, features in enumerate(features_batch):
            instance_shap = shap_values[i] if len(shap_values.shape) > 1 else shap_values
            shap_dict = dict(zip(feature_names, instance_shap.flatten()))
            
            if top_k is not None:
                sorted_features = sorted(
                    shap_dict.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:top_k]
                shap_dict = dict(sorted_features)
            
            results.append(SHAPResult(
                shap_values=shap_dict,
                base_value=float(expected_value),
                expected_value=float(expected_value),
                computation_time_ms=round(per_instance_time, 2),
                explainer_type=config.explainer_type,
                feature_count=len(feature_names),
                model_version_id=model_version_id
            ))
        
        return results
    
    def clear_cache(self, model_version_id: Optional[str] = None):
        """
        Clear explainer cache.
        
        Args:
            model_version_id: If provided, clear only that version. Otherwise clear all.
        """
        if model_version_id:
            keys_to_remove = [
                k for k in self._explainer_cache 
                if model_version_id in k
            ]
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
            "background_data_cached": len(self._background_cache)
        }


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
        shap_result: SHAPResult,
        features: Dict[str, Any],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate data for SHAP waterfall plot.
        
        Returns data structure that frontend can render as waterfall chart.
        """
        # Sort by absolute SHAP value
        sorted_items = sorted(
            shap_result.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
        
        waterfall_data = {
            "base_value": shap_result.base_value,
            "final_value": shap_result.base_value + sum(shap_result.shap_values.values()),
            "features": []
        }
        
        cumulative = shap_result.base_value
        for feature_name, shap_value in sorted_items:
            feature_value = features.get(feature_name, "N/A")
            waterfall_data["features"].append({
                "name": feature_name,
                "value": feature_value,
                "shap_value": shap_value,
                "direction": "positive" if shap_value > 0 else "negative",
                "cumulative_start": cumulative,
                "cumulative_end": cumulative + shap_value
            })
            cumulative += shap_value
        
        return waterfall_data
    
    @staticmethod
    def generate_force_plot_data(
        shap_result: SHAPResult,
        features: Dict[str, Any]
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
                "shap_value": abs(shap_value)
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
            "link": "identity"  # or "logit" for classification
        }
    
    @staticmethod
    def generate_bar_chart_data(
        shap_result: SHAPResult,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate data for feature importance bar chart.
        """
        sorted_items = sorted(
            shap_result.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
        
        return {
            "features": [item[0] for item in sorted_items],
            "values": [item[1] for item in sorted_items],
            "absolute_values": [abs(item[1]) for item in sorted_items],
            "colors": ["#ff6b6b" if item[1] > 0 else "#4dabf7" for item in sorted_items]
        }
