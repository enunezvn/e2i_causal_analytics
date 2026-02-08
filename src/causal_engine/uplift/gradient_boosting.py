"""
E2I Causal Analytics - Uplift Gradient Boosting
================================================

Gradient Boosting-based uplift modeling using CausalML meta-learners.

Uses T-Learner and X-Learner architectures with gradient boosting
base learners for heterogeneous treatment effect estimation.

Author: E2I Causal Analytics Team
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import BaseUpliftModel, UpliftConfig, UpliftModelType, UpliftResult


class GradientBoostingMetaLearner(str, Enum):
    """Meta-learner architecture for gradient boosting uplift."""

    T_LEARNER = "t_learner"
    X_LEARNER = "x_learner"
    S_LEARNER = "s_learner"


@dataclass
class GradientBoostingUpliftConfig(UpliftConfig):
    """Configuration for Gradient Boosting Uplift model.

    Extends UpliftConfig with gradient boosting specific parameters.

    Attributes:
        meta_learner: Which meta-learner architecture to use
        learning_rate: Learning rate for gradient boosting
        subsample: Subsample ratio for stochastic gradient boosting
        use_xgboost: Whether to use XGBoost (faster) or sklearn
        early_stopping_rounds: Early stopping rounds for XGBoost
    """

    meta_learner: GradientBoostingMetaLearner = GradientBoostingMetaLearner.T_LEARNER
    learning_rate: float = 0.1
    subsample: float = 0.8
    use_xgboost: bool = True
    early_stopping_rounds: Optional[int] = None
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "meta_learner": self.meta_learner.value,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "use_xgboost": self.use_xgboost,
                "early_stopping_rounds": self.early_stopping_rounds,
                "colsample_bytree": self.colsample_bytree,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
            }
        )
        return base_dict


class UpliftGradientBoosting(BaseUpliftModel):
    """Gradient Boosting Uplift Model using CausalML meta-learners.

    Combines gradient boosting classifiers/regressors with meta-learner
    architectures (T-Learner, X-Learner, S-Learner) for uplift estimation.

    Key Features:
    - Flexible meta-learner architecture selection
    - XGBoost support for faster training
    - Regularization for preventing overfitting
    - Feature importance from gradient boosting

    Example:
        >>> from src.causal_engine.uplift import UpliftGradientBoosting
        >>> from src.causal_engine.uplift.gradient_boosting import GradientBoostingUpliftConfig
        >>> config = GradientBoostingUpliftConfig(
        ...     n_estimators=100,
        ...     max_depth=5,
        ...     meta_learner=GradientBoostingMetaLearner.X_LEARNER
        ... )
        >>> model = UpliftGradientBoosting(config)
        >>> result = model.estimate(X, treatment, y)
    """

    def __init__(self, config: Optional[GradientBoostingUpliftConfig] = None):
        """Initialize Gradient Boosting Uplift model.

        Args:
            config: Model configuration. If None, uses defaults.
        """
        if config is None:
            config = GradientBoostingUpliftConfig()
        super().__init__(config)
        self.config: GradientBoostingUpliftConfig = config
        self._base_learner: Any = None

    @property
    def model_type(self) -> UpliftModelType:
        """Return the model type."""
        return UpliftModelType.UPLIFT_GRADIENT_BOOSTING

    def _create_base_learner(self) -> Any:
        """Create the base gradient boosting learner.

        Returns:
            XGBoost or sklearn gradient boosting model
        """
        if self.config.use_xgboost:
            try:
                from xgboost import XGBClassifier

                return XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth or 6,
                    learning_rate=self.config.learning_rate,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    reg_alpha=self.config.reg_alpha,
                    reg_lambda=self.config.reg_lambda,
                    random_state=self.config.random_state,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                )
            except ImportError:
                pass  # Fall through to sklearn

        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth or 3,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )

    def _create_model(self) -> Any:
        """Create CausalML meta-learner with gradient boosting base.

        Returns:
            Configured meta-learner instance
        """
        self._base_learner = self._create_base_learner()

        meta_learner = self.config.meta_learner

        if meta_learner == GradientBoostingMetaLearner.T_LEARNER:
            try:
                from causalml.inference.meta import BaseTClassifier

                return BaseTClassifier(
                    learner=self._base_learner,
                    control_name=self.config.control_name,
                )
            except ImportError:
                raise ImportError(
                    "CausalML is required for UpliftGradientBoosting. "
                    "Install with: pip install causalml"
                )

        elif meta_learner == GradientBoostingMetaLearner.X_LEARNER:
            try:
                from causalml.inference.meta import BaseXClassifier

                return BaseXClassifier(
                    learner=self._base_learner,
                    control_name=self.config.control_name,
                )
            except ImportError:
                raise ImportError(
                    "CausalML is required for UpliftGradientBoosting. "
                    "Install with: pip install causalml"
                )

        elif meta_learner == GradientBoostingMetaLearner.S_LEARNER:
            try:
                from causalml.inference.meta import BaseSClassifier

                return BaseSClassifier(
                    learner=self._base_learner,
                    control_name=self.config.control_name,
                )
            except ImportError:
                raise ImportError(
                    "CausalML is required for UpliftGradientBoosting. "
                    "Install with: pip install causalml"
                )

        raise ValueError(f"Unknown meta-learner: {meta_learner}")

    def fit(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
        treatment: NDArray[np.int_],
        y: NDArray[np.float64],
        p: Optional[NDArray[np.float64]] = None,
        **kwargs,
    ) -> "UpliftGradientBoosting":
        """Fit the Gradient Boosting Uplift model.

        Args:
            X: Feature matrix (n_samples, n_features)
            treatment: Treatment assignment array (n_samples,)
            y: Outcome array (n_samples,). Can be binary or continuous.
            p: Propensity scores (n_samples,). Optional.
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
        unique_treatments = sorted(np.unique(treatment))
        self._treatment_groups = [
            str(g) for g in unique_treatments if str(g) != self.config.control_name
        ]

        # Create and fit model
        self.model = self._create_model()

        # CausalML meta-learners expect specific signature
        if p is not None:
            self.model.fit(X=X, treatment=treatment, y=y, p=p, **kwargs)
        else:
            self.model.fit(X=X, treatment=treatment, y=y, **kwargs)

        self.is_fitted = True
        self._fit_time_ms = (time.time() - start_time) * 1000

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
        p: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Predict uplift scores.

        Args:
            X: Feature matrix (n_samples, n_features)
            p: Propensity scores (optional, used by some meta-learners)

        Returns:
            Uplift scores array (n_samples,) or (n_samples, n_treatments)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # CausalML meta-learners return (n_samples, n_treatments) array
        if p is not None:
            uplift_scores = self.model.predict(X=X, p=p)
        else:
            uplift_scores = self.model.predict(X=X)

        # Ensure 2D array
        if len(uplift_scores.shape) == 1:
            uplift_scores = uplift_scores.reshape(-1, 1)

        # Normalize if configured
        if self.config.normalize_scores:
            uplift_scores = self._normalize_scores(uplift_scores)

        return cast(NDArray[np.float64], uplift_scores)

    def estimate(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
        treatment: NDArray[np.int_],
        y: NDArray[np.float64],
        X_test: Optional[Union[pd.DataFrame, NDArray[np.float64]]] = None,
        treatment_test: Optional[NDArray[np.int_]] = None,
        y_test: Optional[NDArray[np.float64]] = None,
        p: Optional[NDArray[np.float64]] = None,
        p_test: Optional[NDArray[np.float64]] = None,
        **kwargs,
    ) -> UpliftResult:
        """Fit model and estimate treatment effects.

        Args:
            X: Training feature matrix
            treatment: Training treatment assignments
            y: Training outcomes
            X_test: Test feature matrix (optional)
            treatment_test: Test treatment assignments (optional)
            y_test: Test outcomes (optional)
            p: Training propensity scores (optional)
            p_test: Test propensity scores (optional)
            **kwargs: Additional arguments

        Returns:
            UpliftResult with estimated effects
        """
        import time

        start_time = time.time()

        try:
            # Fit the model
            self.fit(X, treatment, y, p=p, **kwargs)

            # Use test data if provided, else training data
            X_pred = X_test if X_test is not None else X
            treatment_eval = treatment_test if treatment_test is not None else treatment
            y_eval = y_test if y_test is not None else y
            p_pred = p_test if p_test is not None else p

            if isinstance(X_pred, pd.DataFrame):
                X_pred_arr = X_pred.values
            else:
                X_pred_arr = X_pred

            # Get uplift predictions
            uplift_scores = self.predict(X_pred, p=p_pred)

            # Calculate aggregate treatment effects
            ate, att, atc, ate_std = self._calculate_ate(uplift_scores, treatment_eval, y_eval)

            # Calculate confidence interval
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
                    "meta_learner": self.config.meta_learner.value,
                    "use_xgboost": self.config.use_xgboost,
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

    def _get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importance from base learner(s).

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or self._base_learner is None:
            return None

        try:
            # Try to get from base learner
            if hasattr(self._base_learner, "feature_importances_"):
                importances = self._base_learner.feature_importances_

                if self._feature_names is not None and len(self._feature_names) == len(importances):
                    return {
                        name: float(imp)
                        for name, imp in zip(self._feature_names, importances, strict=False)
                    }
                return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}

            # For T-Learner and X-Learner, try to get from internal models
            if hasattr(self.model, "models_t") and self.model.models_t:
                # Get average importance across treatment models
                all_importances = []
                for model in self.model.models_t.values():
                    if hasattr(model, "feature_importances_"):
                        all_importances.append(model.feature_importances_)

                if all_importances:
                    avg_importances = np.mean(all_importances, axis=0)
                    if self._feature_names is not None and len(self._feature_names) == len(
                        avg_importances
                    ):
                        return {
                            name: float(imp)
                            for name, imp in zip(self._feature_names, avg_importances, strict=False)
                        }
                    return {f"feature_{i}": float(imp) for i, imp in enumerate(avg_importances)}

        except Exception:
            pass

        return None
