"""Preprocessing for model_trainer.

This module fits preprocessing pipelines on training data ONLY to prevent leakage.

The model_trainer preprocessor is designed to work in two modes:
1. Standalone: Full preprocessing when data_preparer hasn't run
2. Pipeline: Light validation when data is already preprocessed

CRITICAL DATA LEAKAGE PREVENTION:
- Preprocessing (scaling, encoding, imputation) MUST be fit ONLY on train data
- Validation/test/holdout are transformed using train-fit preprocessor
- NEVER fit on validation, test, or holdout
- Statistics (mean, std, etc.) come ONLY from training set

Version: 2.0.0 - Full sklearn implementation
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class ModelTrainerPreprocessor:
    """
    Sklearn-compatible preprocessor for model training.

    Handles:
    - Numeric feature scaling (StandardScaler)
    - Categorical feature encoding (OneHotEncoder)
    - Missing value imputation (SimpleImputer)

    Always fits on training data ONLY.
    """

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scaling_method: str = "standard",
        imputation_strategy: str = "mean",
    ):
        """
        Initialize preprocessor.

        Args:
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names
            scaling_method: "standard" or "none"
            imputation_strategy: "mean", "median", or "most_frequent"
        """
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.scaling_method = scaling_method
        self.imputation_strategy = imputation_strategy

        self._pipeline: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None
        self.train_statistics_: Dict[str, Any] = {}
        self._is_fitted = False

    def _build_pipeline(self) -> ColumnTransformer:
        """Build sklearn ColumnTransformer pipeline."""
        transformers = []

        # Numeric transformer: impute then scale
        if self.numeric_features:
            numeric_steps = [("imputer", SimpleImputer(strategy=self.imputation_strategy))]
            if self.scaling_method == "standard":
                numeric_steps.append(("scaler", StandardScaler()))

            numeric_transformer = Pipeline(steps=numeric_steps)
            transformers.append(("numeric", numeric_transformer, self.numeric_features))

        # Categorical transformer: impute then encode
        if self.categorical_features:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("categorical", categorical_transformer, self.categorical_features))

        return ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",  # Keep other columns unchanged
            verbose_feature_names_out=False,
        )

    def fit(self, X: pd.DataFrame, y=None) -> "ModelTrainerPreprocessor":
        """
        Fit preprocessing pipeline on training data ONLY.

        Args:
            X: Training feature DataFrame
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        # Auto-detect feature types if not specified
        if not self.numeric_features and not self.categorical_features:
            self._detect_feature_types(X)

        # Record input features
        self.n_features_in_ = X.shape[1]

        # Build and fit pipeline
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X)
        self._is_fitted = True

        # Get output feature names
        try:
            self.feature_names_out_ = list(self._pipeline.get_feature_names_out())
        except Exception:
            self.feature_names_out_ = None

        # Compute statistics from training data ONLY
        self._compute_train_statistics(X)

        logger.info(
            f"Preprocessor fitted: {len(self.numeric_features)} numeric, "
            f"{len(self.categorical_features)} categorical features"
        )

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.

        Args:
            X: Feature DataFrame to transform

        Returns:
            Transformed numpy array
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        return self._pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _detect_feature_types(self, X: pd.DataFrame) -> None:
        """Auto-detect numeric and categorical features from DataFrame."""
        self.numeric_features = []
        self.categorical_features = []

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numeric_features.append(col)
            elif pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
                # Only treat as categorical if cardinality is reasonable
                n_unique = X[col].nunique()
                if n_unique <= 50:
                    self.categorical_features.append(col)
                else:
                    logger.warning(f"Skipping high-cardinality column: {col} ({n_unique} unique)")

    def _compute_train_statistics(self, X: pd.DataFrame) -> None:
        """Compute statistics from training data ONLY."""
        stats = {}

        if self.numeric_features:
            numeric_data = X[self.numeric_features]
            stats["numeric"] = {
                "mean": numeric_data.mean().to_dict(),
                "std": numeric_data.std().to_dict(),
                "min": numeric_data.min().to_dict(),
                "max": numeric_data.max().to_dict(),
                "missing_rate": numeric_data.isnull().mean().to_dict(),
            }

        if self.categorical_features:
            cat_data = X[self.categorical_features]
            stats["categorical"] = {
                "unique_counts": {col: cat_data[col].nunique() for col in cat_data.columns},
                "missing_rate": cat_data.isnull().mean().to_dict(),
            }

        self.train_statistics_ = stats


def _is_already_preprocessed(X: pd.DataFrame) -> bool:
    """
    Check if data appears to already be preprocessed.

    Heuristics:
    - No missing values
    - Numeric columns are roughly scaled (mean near 0, std near 1)
    - No object dtype columns (all encoded)
    """
    # Check for object columns (suggests not encoded)
    if X.select_dtypes(include=["object"]).shape[1] > 0:
        return False

    # Check for missing values
    if X.isnull().any().any():
        return False

    # Check if numeric data is scaled
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        means = X[numeric_cols].mean()
        stds = X[numeric_cols].std()

        # If means are close to 0 and stds close to 1, likely already scaled
        mean_centered = (means.abs() < 1.0).mean() > 0.5
        std_scaled = ((stds > 0.5) & (stds < 2.0)).mean() > 0.5

        if mean_centered and std_scaled:
            return True

    return False


async def fit_preprocessing(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fit preprocessing pipeline on training data ONLY.

    CRITICAL DATA LEAKAGE PREVENTION:
    - Preprocessing (scaling, encoding, imputation) MUST be fit ONLY on train data
    - Validation/test/holdout are transformed using train-fit preprocessor
    - NEVER fit on validation, test, or holdout
    - Statistics (mean, std, etc.) come ONLY from training set

    This node is smart about detecting if data is already preprocessed
    (e.g., from data_preparer) and will skip redundant transformations.

    Args:
        state: ModelTrainerState with train_data, validation_data, test_data

    Returns:
        Dictionary with preprocessor, X_train_preprocessed,
        X_validation_preprocessed, X_test_preprocessed,
        preprocessing_statistics

    Raises:
        No exceptions - returns error in state if preprocessing fails
    """
    try:
        # Extract split data
        train_data = state.get("train_data", {})
        validation_data = state.get("validation_data", {})
        test_data = state.get("test_data", {})

        # Extract feature matrices
        X_train = train_data.get("X")
        X_validation = validation_data.get("X")
        X_test = test_data.get("X")

        # Validation checks
        if X_train is None:
            return {
                "error": "X_train is None - cannot fit preprocessing",
                "error_type": "missing_training_data",
            }

        if X_validation is None:
            return {
                "error": "X_validation is None - cannot transform validation data",
                "error_type": "missing_validation_data",
            }

        if X_test is None:
            return {
                "error": "X_test is None - cannot transform test data",
                "error_type": "missing_test_data",
            }

        # Convert to DataFrame if numpy arrays
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
        if isinstance(X_validation, np.ndarray):
            X_validation = pd.DataFrame(X_validation)
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)

        # Check if data is already preprocessed (from data_preparer)
        already_preprocessed = _is_already_preprocessed(X_train)

        if already_preprocessed:
            logger.info(
                "Data appears already preprocessed (from data_preparer). "
                "Applying light validation only."
            )
            preprocessing_type = "passthrough"
        else:
            preprocessing_type = "full"

        # Get preprocessing config from state if available
        preprocessing_config = state.get("preprocessing_config", {})
        scaling_method = preprocessing_config.get("scaling_method", "standard")
        imputation_strategy = preprocessing_config.get("imputation_strategy", "mean")

        # Create preprocessor
        preprocessor = ModelTrainerPreprocessor(
            scaling_method=scaling_method if not already_preprocessed else "none",
            imputation_strategy=imputation_strategy,
        )

        # Fit on TRAIN ONLY, transform all splits
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_validation_preprocessed = preprocessor.transform(X_validation)
        X_test_preprocessed = preprocessor.transform(X_test)

        # Build statistics
        preprocessing_statistics = {
            "n_features_in": preprocessor.n_features_in_,
            "n_features_out": (
                X_train_preprocessed.shape[1] if X_train_preprocessed.ndim > 1 else 1
            ),
            "feature_names_out": preprocessor.feature_names_out_,
            "numeric_features": preprocessor.numeric_features,
            "categorical_features": preprocessor.categorical_features,
            "train_statistics": preprocessor.train_statistics_,
            "preprocessing_type": preprocessing_type,
            "scaling_method": scaling_method if not already_preprocessed else "none",
            "imputation_strategy": imputation_strategy,
        }

        logger.info(
            f"Preprocessing complete: {preprocessing_type} mode, "
            f"{preprocessing_statistics['n_features_in']} -> "
            f"{preprocessing_statistics['n_features_out']} features"
        )

        return {
            "preprocessor": preprocessor,
            "X_train_preprocessed": X_train_preprocessed,
            "X_validation_preprocessed": X_validation_preprocessed,
            "X_test_preprocessed": X_test_preprocessed,
            "preprocessing_statistics": preprocessing_statistics,
        }

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": "preprocessing_error",
        }
