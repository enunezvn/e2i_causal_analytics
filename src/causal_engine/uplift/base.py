"""
E2I Causal Analytics - Uplift Modeling Base Classes
====================================================

Base classes and interfaces for CausalML uplift modeling.

Provides:
- BaseUpliftModel: Abstract base for all uplift models
- UpliftResult: Standardized result container
- UpliftConfig: Configuration for uplift models

Author: E2I Causal Analytics Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class UpliftModelType(str, Enum):
    """Supported uplift model types."""

    UPLIFT_RANDOM_FOREST = "uplift_random_forest"
    UPLIFT_TREE = "uplift_tree"
    UPLIFT_GRADIENT_BOOSTING = "uplift_gradient_boosting"
    CAUSAL_TREE = "causal_tree"


class UpliftNormalization(str, Enum):
    """Normalization methods for uplift scores."""

    NONE = "none"
    MINMAX = "minmax"
    ZSCORE = "zscore"


@dataclass
class UpliftConfig:
    """Configuration for uplift model training.

    Attributes:
        n_estimators: Number of trees in the forest (for ensemble methods)
        max_depth: Maximum depth of trees
        min_samples_leaf: Minimum samples required at leaf node
        min_samples_treatment: Minimum samples per treatment group at leaf
        n_reg: Regularization parameter
        control_name: Name/identifier for control group
        random_state: Random seed for reproducibility
        normalize_scores: Whether to normalize uplift scores
        normalization_method: Method for score normalization
    """

    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_leaf: int = 100
    min_samples_treatment: int = 10
    n_reg: int = 10
    control_name: str = "control"
    random_state: int = 42
    normalize_scores: bool = False
    normalization_method: UpliftNormalization = UpliftNormalization.MINMAX
    honesty: bool = False
    inference: bool = False
    evaluationFunction: str = "KL"  # KL divergence, ED, Chi, CTS, DDP

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_treatment": self.min_samples_treatment,
            "n_reg": self.n_reg,
            "control_name": self.control_name,
            "random_state": self.random_state,
            "normalize_scores": self.normalize_scores,
            "normalization_method": self.normalization_method.value,
            "honesty": self.honesty,
            "inference": self.inference,
            "evaluationFunction": self.evaluationFunction,
        }


@dataclass
class UpliftResult:
    """Standardized result container for uplift models.

    Attributes:
        model_type: Type of uplift model used
        success: Whether estimation succeeded
        uplift_scores: Individual treatment effect predictions (ITE/CATE)
        ate: Average Treatment Effect
        att: Average Treatment Effect on Treated
        atc: Average Treatment Effect on Control
        ate_std: Standard deviation of ATE
        ate_ci_lower: Lower bound of ATE confidence interval
        ate_ci_upper: Upper bound of ATE confidence interval
        treatment_groups: Names of treatment groups
        feature_importances: Feature importance scores
        error_message: Error message if estimation failed
        estimation_time_ms: Time taken for estimation in milliseconds
        metadata: Additional metadata
    """

    model_type: UpliftModelType
    success: bool
    uplift_scores: Optional[NDArray[np.float64]] = None
    ate: Optional[float] = None
    att: Optional[float] = None
    atc: Optional[float] = None
    ate_std: Optional[float] = None
    ate_ci_lower: Optional[float] = None
    ate_ci_upper: Optional[float] = None
    treatment_groups: List[str] = field(default_factory=list)
    feature_importances: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    estimation_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result consistency."""
        if self.success and self.uplift_scores is None:
            raise ValueError("Successful result must include uplift_scores")
        if not self.success and self.error_message is None:
            self.error_message = "Unknown error"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "model_type": self.model_type.value,
            "success": self.success,
            "uplift_scores": (
                self.uplift_scores.tolist()
                if self.uplift_scores is not None
                else None
            ),
            "ate": self.ate,
            "att": self.att,
            "atc": self.atc,
            "ate_std": self.ate_std,
            "ate_ci_lower": self.ate_ci_lower,
            "ate_ci_upper": self.ate_ci_upper,
            "treatment_groups": self.treatment_groups,
            "feature_importances": self.feature_importances,
            "error_message": self.error_message,
            "estimation_time_ms": self.estimation_time_ms,
            "metadata": self.metadata,
        }


class BaseUpliftModel(ABC):
    """Abstract base class for uplift models.

    All uplift model implementations must inherit from this class
    and implement the required abstract methods.

    Attributes:
        config: Model configuration
        model: Underlying CausalML model instance
        is_fitted: Whether the model has been fitted
    """

    def __init__(self, config: Optional[UpliftConfig] = None):
        """Initialize uplift model.

        Args:
            config: Model configuration. If None, uses defaults.
        """
        self.config = config or UpliftConfig()
        self.model: Any = None
        self.is_fitted: bool = False
        self._feature_names: Optional[List[str]] = None
        self._treatment_groups: List[str] = []

    @property
    @abstractmethod
    def model_type(self) -> UpliftModelType:
        """Return the type of uplift model."""
        pass

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying CausalML model instance.

        Returns:
            CausalML model instance
        """
        pass

    def fit(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
        treatment: NDArray[np.int_],
        y: NDArray[np.float64],
        **kwargs,
    ) -> "BaseUpliftModel":
        """Fit the uplift model.

        Args:
            X: Feature matrix (n_samples, n_features)
            treatment: Treatment assignment array (n_samples,)
            y: Outcome array (n_samples,)
            **kwargs: Additional arguments passed to underlying model

        Returns:
            Self for method chaining
        """
        import time

        start_time = time.time()

        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.values

        # Identify treatment groups
        self._treatment_groups = [
            str(g) for g in sorted(np.unique(treatment)) if str(g) != self.config.control_name
        ]

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X=X, treatment=treatment, y=y, **kwargs)
        self.is_fitted = True

        elapsed_ms = (time.time() - start_time) * 1000
        self._fit_time_ms = elapsed_ms

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Predict uplift scores for new data.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Uplift scores array (n_samples, n_treatment_groups)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        uplift_scores = self.model.predict(X)

        # Normalize if configured
        if self.config.normalize_scores:
            uplift_scores = self._normalize_scores(uplift_scores)

        return uplift_scores

    def estimate(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
        treatment: NDArray[np.int_],
        y: NDArray[np.float64],
        X_test: Optional[Union[pd.DataFrame, NDArray[np.float64]]] = None,
        treatment_test: Optional[NDArray[np.int_]] = None,
        y_test: Optional[NDArray[np.float64]] = None,
        **kwargs,
    ) -> UpliftResult:
        """Fit model and estimate treatment effects.

        Args:
            X: Training feature matrix
            treatment: Training treatment assignments
            y: Training outcomes
            X_test: Test feature matrix (optional, uses training if not provided)
            treatment_test: Test treatment assignments (optional)
            y_test: Test outcomes (optional)
            **kwargs: Additional arguments

        Returns:
            UpliftResult with estimated effects
        """
        import time

        start_time = time.time()

        try:
            # Fit the model
            self.fit(X, treatment, y, **kwargs)

            # Use test data if provided, else training data
            X_pred = X_test if X_test is not None else X
            treatment_eval = treatment_test if treatment_test is not None else treatment
            y_eval = y_test if y_test is not None else y

            if isinstance(X_pred, pd.DataFrame):
                X_pred_arr = X_pred.values
            else:
                X_pred_arr = X_pred

            # Get uplift predictions
            uplift_scores = self.predict(X_pred)

            # Calculate aggregate treatment effects
            ate, att, atc, ate_std = self._calculate_ate(
                uplift_scores, treatment_eval, y_eval
            )

            # Calculate confidence interval (assuming normal distribution)
            n = len(uplift_scores)
            if ate_std is not None and n > 0:
                z = 1.96  # 95% CI
                margin = z * ate_std / np.sqrt(n)
                ate_ci_lower = ate - margin if ate is not None else None
                ate_ci_upper = ate + margin if ate is not None else None
            else:
                ate_ci_lower = None
                ate_ci_upper = None

            # Get feature importances
            feature_importances = self._get_feature_importances()

            elapsed_ms = (time.time() - start_time) * 1000

            return UpliftResult(
                model_type=self.model_type,
                success=True,
                uplift_scores=uplift_scores,
                ate=ate,
                att=att,
                atc=atc,
                ate_std=ate_std,
                ate_ci_lower=ate_ci_lower,
                ate_ci_upper=ate_ci_upper,
                treatment_groups=self._treatment_groups,
                feature_importances=feature_importances,
                estimation_time_ms=elapsed_ms,
                metadata={
                    "n_samples_train": len(X) if hasattr(X, "__len__") else 0,
                    "n_samples_test": len(X_pred_arr),
                    "n_features": X_pred_arr.shape[1] if len(X_pred_arr.shape) > 1 else 1,
                    "config": self.config.to_dict(),
                },
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return UpliftResult(
                model_type=self.model_type,
                success=False,
                error_message=str(e),
                estimation_time_ms=elapsed_ms,
            )

    def _calculate_ate(
        self,
        uplift_scores: NDArray[np.float64],
        treatment: NDArray[np.int_],
        y: NDArray[np.float64],
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate ATE, ATT, ATC from uplift scores.

        Args:
            uplift_scores: Predicted uplift scores
            treatment: Treatment assignments
            y: Observed outcomes

        Returns:
            Tuple of (ATE, ATT, ATC, ATE_std)
        """
        try:
            # Handle multi-dimensional uplift scores (multiple treatments)
            if len(uplift_scores.shape) > 1:
                # Use first treatment group for ATE calculation
                scores = uplift_scores[:, 0]
            else:
                scores = uplift_scores

            # ATE: Average of all uplift scores
            ate = float(np.mean(scores))
            ate_std = float(np.std(scores))

            # ATT: Average uplift among treated
            treatment_binary = treatment.astype(bool)
            if np.sum(treatment_binary) > 0:
                att = float(np.mean(scores[treatment_binary]))
            else:
                att = None

            # ATC: Average uplift among control
            if np.sum(~treatment_binary) > 0:
                atc = float(np.mean(scores[~treatment_binary]))
            else:
                atc = None

            return ate, att, atc, ate_std

        except Exception:
            return None, None, None, None

    def _normalize_scores(
        self, scores: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Normalize uplift scores.

        Args:
            scores: Raw uplift scores

        Returns:
            Normalized scores
        """
        method = self.config.normalization_method

        if method == UpliftNormalization.MINMAX:
            min_val = np.min(scores)
            max_val = np.max(scores)
            if max_val - min_val > 0:
                return (scores - min_val) / (max_val - min_val)
            return scores

        elif method == UpliftNormalization.ZSCORE:
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            if std_val > 0:
                return (scores - mean_val) / std_val
            return scores - mean_val

        return scores

    def _get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores from fitted model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or self.model is None:
            return None

        try:
            # Try to get feature importances from model
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_

                # Map to feature names if available
                if self._feature_names is not None and len(self._feature_names) == len(
                    importances
                ):
                    return dict(zip(self._feature_names, importances.tolist()))
                else:
                    return {
                        f"feature_{i}": float(imp)
                        for i, imp in enumerate(importances)
                    }
        except Exception:
            pass

        return None

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.to_dict()
