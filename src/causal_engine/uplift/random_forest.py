"""
E2I Causal Analytics - Uplift Random Forest
============================================

Uplift Random Forest implementation using CausalML.

Provides heterogeneous treatment effect estimation using
tree-based ensemble methods optimized for uplift modeling.

Author: E2I Causal Analytics Team
"""

from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import BaseUpliftModel, UpliftModelType


class UpliftRandomForest(BaseUpliftModel):
    """Uplift Random Forest for heterogeneous treatment effect estimation.

    Uses CausalML's UpliftRandomForestClassifier which builds an ensemble
    of uplift trees that directly optimize for treatment effect heterogeneity.

    Key Features:
    - Directly optimizes uplift criteria (KL divergence, Chi-squared, etc.)
    - Handles multiple treatment groups
    - Provides feature importance for interpretability
    - Supports honest estimation for valid inference

    Example:
        >>> from src.causal_engine.uplift import UpliftRandomForest, UpliftConfig
        >>> config = UpliftConfig(n_estimators=100, max_depth=5)
        >>> model = UpliftRandomForest(config)
        >>> result = model.estimate(X, treatment, y)
        >>> print(f"ATE: {result.ate:.4f}")
    """

    @property
    def model_type(self) -> UpliftModelType:
        """Return the model type."""
        return UpliftModelType.UPLIFT_RANDOM_FOREST

    def _create_model(self) -> Any:
        """Create CausalML UpliftRandomForestClassifier.

        Returns:
            Configured UpliftRandomForestClassifier instance
        """
        try:
            from causalml.inference.tree import UpliftRandomForestClassifier
        except ImportError:
            raise ImportError(
                "CausalML is required for UpliftRandomForest. Install with: pip install causalml"
            )

        return UpliftRandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            min_samples_treatment=self.config.min_samples_treatment,
            n_reg=self.config.n_reg,
            control_name=self.config.control_name,
            random_state=self.config.random_state,
            honesty=self.config.honesty,
            evaluationFunction=self.config.evaluationFunction,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
        treatment: NDArray[np.int_],
        y: NDArray[np.float64],
        **kwargs,
    ) -> "UpliftRandomForest":
        """Fit the Uplift Random Forest model.

        Args:
            X: Feature matrix (n_samples, n_features)
            treatment: Treatment assignment array (n_samples,).
                       Values should be treatment group identifiers.
            y: Binary outcome array (n_samples,). Must be 0 or 1.
            **kwargs: Additional arguments passed to underlying model

        Returns:
            Self for method chaining
        """
        super().fit(X, treatment, y, **kwargs)
        return cast("UpliftRandomForest", self)

    def predict(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Predict uplift scores.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Uplift scores array (n_samples, n_treatment_groups)
            Each column corresponds to the uplift for a treatment vs control.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # UpliftRandomForestClassifier returns dict with treatment names as keys
        uplift_dict = self.model.predict(X)

        # Convert to array format
        if isinstance(uplift_dict, dict):
            # Stack predictions for each treatment group
            uplift_scores = np.column_stack(
                [uplift_dict[group] for group in sorted(uplift_dict.keys())]
            )
        else:
            uplift_scores = uplift_dict

        # Normalize if configured
        if self.config.normalize_scores:
            uplift_scores = self._normalize_scores(uplift_scores)

        return uplift_scores

    def _get_feature_importances(self) -> Optional[dict]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or self.model is None:
            return None

        try:
            # CausalML provides feature_importances_ attribute
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_

                if self._feature_names is not None and len(self._feature_names) == len(importances):
                    return {
                        name: float(imp)
                        for name, imp in zip(self._feature_names, importances, strict=False)
                    }
                return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
        except Exception:
            pass

        return None


class UpliftTree(BaseUpliftModel):
    """Single Uplift Decision Tree.

    A single tree model for uplift estimation. Useful for interpretability
    or as a baseline before using ensemble methods.

    Example:
        >>> from src.causal_engine.uplift import UpliftTree, UpliftConfig
        >>> config = UpliftConfig(max_depth=5)
        >>> model = UpliftTree(config)
        >>> result = model.estimate(X, treatment, y)
    """

    @property
    def model_type(self) -> UpliftModelType:
        """Return the model type."""
        return UpliftModelType.UPLIFT_TREE

    def _create_model(self) -> Any:
        """Create CausalML UpliftTreeClassifier.

        Returns:
            Configured UpliftTreeClassifier instance
        """
        try:
            from causalml.inference.tree import UpliftTreeClassifier
        except ImportError:
            raise ImportError(
                "CausalML is required for UpliftTree. Install with: pip install causalml"
            )

        return UpliftTreeClassifier(
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            min_samples_treatment=self.config.min_samples_treatment,
            n_reg=self.config.n_reg,
            control_name=self.config.control_name,
            random_state=self.config.random_state,
            evaluationFunction=self.config.evaluationFunction,
        )

    def predict(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Predict uplift scores.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Uplift scores array
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # UpliftTreeClassifier returns dict with treatment names as keys
        uplift_dict = self.model.predict(X)

        # Convert to array format
        if isinstance(uplift_dict, dict):
            uplift_scores = np.column_stack(
                [uplift_dict[group] for group in sorted(uplift_dict.keys())]
            )
        else:
            uplift_scores = uplift_dict

        # Normalize if configured
        if self.config.normalize_scores:
            uplift_scores = self._normalize_scores(uplift_scores)

        return uplift_scores
