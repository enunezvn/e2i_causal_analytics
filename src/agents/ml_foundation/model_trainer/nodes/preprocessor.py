"""Preprocessing for model_trainer.

This module fits preprocessing pipelines on training data ONLY to prevent leakage.
"""

from typing import Any, Dict

import numpy as np


async def fit_preprocessing(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fit preprocessing pipeline on training data ONLY.

    CRITICAL DATA LEAKAGE PREVENTION:
    - Preprocessing (scaling, encoding, imputation) MUST be fit ONLY on train data
    - Validation/test/holdout are transformed using train-fit preprocessor
    - NEVER fit on validation, test, or holdout
    - Statistics (mean, std, etc.) come ONLY from training set

    Args:
        state: ModelTrainerState with train_data, validation_data, test_data

    Returns:
        Dictionary with preprocessor, X_train_preprocessed,
        X_validation_preprocessed, X_test_preprocessed,
        preprocessing_statistics

    Raises:
        No exceptions - returns error in state if preprocessing fails
    """
    # Extract split data
    train_data = state.get("train_data", {})
    validation_data = state.get("validation_data", {})
    test_data = state.get("test_data", {})

    # Extract feature matrices
    X_train = train_data.get("X")
    X_validation = validation_data.get("X")
    X_test = test_data.get("X")

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

    # TODO: Extract preprocessing config from model_candidate
    # For now, implement basic preprocessing
    # In production, this should:
    # 1. Get preprocessing_config from model_candidate
    # 2. Build sklearn Pipeline based on config
    # 3. Handle categorical encoding, scaling, imputation

    # PLACEHOLDER: Identity preprocessing (no transformation)
    # This will be replaced with actual sklearn Pipeline
    class IdentityPreprocessor:
        """Placeholder preprocessor that passes data through unchanged."""

        def __init__(self):
            self.feature_names_ = None
            self.n_features_in_ = None
            self.train_statistics_ = {}

        def fit(self, X, y=None):
            """Fit on training data and compute statistics."""
            # Record feature information
            if hasattr(X, "shape"):
                self.n_features_in_ = X.shape[1]
            if hasattr(X, "columns"):
                self.feature_names_ = list(X.columns)

            # Compute basic statistics from TRAIN ONLY
            if hasattr(X, "describe"):
                # Pandas DataFrame
                self.train_statistics_ = {
                    "mean": X.mean().to_dict(),
                    "std": X.std().to_dict(),
                    "min": X.min().to_dict(),
                    "max": X.max().to_dict(),
                }
            elif isinstance(X, np.ndarray):
                # NumPy array
                self.train_statistics_ = {
                    "mean": float(np.mean(X)),
                    "std": float(np.std(X)),
                    "min": float(np.min(X)),
                    "max": float(np.max(X)),
                }
            else:
                self.train_statistics_ = {}

            return self

        def transform(self, X):
            """Transform data (identity transformation)."""
            return X

        def fit_transform(self, X, y=None):
            """Fit and transform in one step."""
            return self.fit(X, y).transform(X)

    # Create and fit preprocessor on TRAIN ONLY
    preprocessor = IdentityPreprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    # Transform validation and test using train-fit preprocessor
    X_validation_preprocessed = preprocessor.transform(X_validation)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Extract preprocessing statistics (computed from TRAIN ONLY)
    preprocessing_statistics = {
        "n_features": preprocessor.n_features_in_,
        "feature_names": preprocessor.feature_names_,
        "train_statistics": preprocessor.train_statistics_,
        "preprocessing_type": "identity",  # TODO: Update when real pipeline added
    }

    # TODO: Implement actual preprocessing pipeline
    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import StandardScaler, OneHotEncoder
    # from sklearn.compose import ColumnTransformer
    #
    # preprocessor = ColumnTransformer([
    #     ('num', StandardScaler(), numeric_features),
    #     ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    # ])

    return {
        "preprocessor": preprocessor,
        "X_train_preprocessed": X_train_preprocessed,
        "X_validation_preprocessed": X_validation_preprocessed,
        "X_test_preprocessed": X_test_preprocessed,
        "preprocessing_statistics": preprocessing_statistics,
    }
