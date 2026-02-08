"""
Twin Generator
==============

ML-based generation of digital twins from historical data.
Uses Random Forest / Gradient Boosting to model entity behavior
and generate synthetic populations for simulation.

Integration:
    - MLflow for model tracking and registry
    - Feast feature store for consistent features
    - DoWhy for causal-aware feature selection
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .models.twin_models import (
    Brand,
    DigitalTwin,
    TwinModelConfig,
    TwinModelMetrics,
    TwinPopulation,
    TwinType,
)

logger = logging.getLogger(__name__)


class TwinGenerator:
    """
    Generates digital twins based on historical entity data.

    The generator learns patterns from real HCPs, patients, or territories
    to create synthetic populations that simulate response to interventions.

    Attributes:
        twin_type: Type of entity to generate (hcp, patient, territory)
        brand: Pharmaceutical brand context
        model: Trained sklearn model
        config: Model configuration
        metrics: Performance metrics from training

    Example:
        >>> generator = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.KISQALI)
        >>> generator.train(training_data, target_col="prescribing_change")
        >>> twins = generator.generate(n=10000)
    """

    # Minimum samples required for training
    MIN_TRAINING_SAMPLES = 1000

    # Default feature columns by twin type
    DEFAULT_FEATURES = {
        TwinType.HCP: [
            "specialty",
            "years_experience",
            "practice_type",
            "practice_size",
            "region",
            "decile",
            "priority_tier",
            "total_patient_volume",
            "target_patient_volume",
            "digital_engagement_score",
            "preferred_channel",
            "last_interaction_days",
            "interaction_frequency",
            "adoption_stage",
            "peer_influence_score",
        ],
        TwinType.PATIENT: [
            "age_group",
            "gender",
            "geographic_region",
            "socioeconomic_index",
            "primary_diagnosis_code",
            "comorbidity_count",
            "risk_score",
            "journey_complexity_score",
            "insurance_type",
            "insurance_coverage_flag",
            "journey_stage",
            "journey_duration_days",
            "treatment_line",
        ],
        TwinType.TERRITORY: [
            "region",
            "state_count",
            "zip_count",
            "total_hcps",
            "covered_hcps",
            "coverage_rate",
            "total_patient_volume",
            "market_share",
            "growth_rate",
            "competitor_presence",
        ],
    }

    def __init__(
        self,
        twin_type: TwinType,
        brand: Brand,
        config: Optional[TwinModelConfig] = None,
    ):
        """
        Initialize twin generator.

        Args:
            twin_type: Type of twins to generate
            brand: Brand context
            config: Optional model configuration
        """
        self.twin_type = twin_type
        self.brand = brand
        self.config = config

        # Model components (set during training)
        self.model: Optional[Any] = None
        self.model_id: Optional[UUID] = None
        self.feature_columns: List[str] = []
        self.target_column: Optional[str] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.metrics: Optional[TwinModelMetrics] = None

        # Feature statistics for generation
        self._feature_stats: Dict[str, Dict] = {}
        self._categorical_distributions: Dict[str, Dict] = {}

        logger.info(f"Initialized TwinGenerator for {twin_type.value} twins, brand={brand.value}")

    def train(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        algorithm: str = "random_forest",
        **kwargs,
    ) -> TwinModelMetrics:
        """
        Train the twin generator model on historical data.

        Args:
            data: Training DataFrame with entity features and outcomes
            target_col: Column name for outcome variable
            feature_cols: Feature columns to use (defaults to type-specific)
            algorithm: "random_forest" or "gradient_boosting"
            **kwargs: Additional model parameters

        Returns:
            TwinModelMetrics with performance metrics

        Raises:
            ValueError: If insufficient training data
        """
        start_time = datetime.now(timezone.utc)

        # Validate data
        if len(data) < self.MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Insufficient training data: {len(data)} < {self.MIN_TRAINING_SAMPLES}"
            )

        # Set features
        self.feature_columns = feature_cols or self.DEFAULT_FEATURES.get(self.twin_type, [])
        self.target_column = target_col

        # Validate columns exist
        missing_cols = set(self.feature_columns) - set(data.columns)
        if missing_cols:
            logger.warning(f"Missing columns, will skip: {missing_cols}")
            self.feature_columns = [c for c in self.feature_columns if c in data.columns]

        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not in data")

        logger.info(f"Training on {len(data)} samples with {len(self.feature_columns)} features")

        # Prepare features
        X, y = self._prepare_training_data(data)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model = self._create_model(algorithm, **kwargs)
        assert self.model is not None, "Model must be created successfully"
        self.model.fit(X_train, y_train)

        # Calculate metrics
        y_pred = self.model.predict(X_val)
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="r2")

        training_duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Feature importances
        importances = dict(
            zip(self.feature_columns, self.model.feature_importances_.tolist(), strict=False)
        )
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        self.model_id = uuid4()
        self.metrics = TwinModelMetrics(
            model_id=self.model_id,
            r2_score=float(r2_score(y_val, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_val, y_pred))),
            mae=float(mean_absolute_error(y_val, y_pred)),
            cv_scores=cv_scores.tolist(),
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            feature_importances=importances,
            top_features=[f[0] for f in sorted_features[:10]],
            training_samples=len(data),
            training_duration_seconds=training_duration,
        )

        logger.info(
            f"Training complete: R²={self.metrics.r2_score:.3f}, "
            f"RMSE={self.metrics.rmse:.4f}, CV mean={self.metrics.cv_mean:.3f}"
        )

        return self.metrics

    def generate(
        self,
        n: int,
        constraints: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> TwinPopulation:
        """
        Generate a population of digital twins.

        Args:
            n: Number of twins to generate
            constraints: Optional constraints on feature values
            seed: Random seed for reproducibility

        Returns:
            TwinPopulation containing generated twins

        Raises:
            RuntimeError: If model not trained
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if seed is not None:
            np.random.seed(seed)

        logger.info(f"Generating {n} {self.twin_type.value} twins")

        twins = []
        features_list = []

        for _ in range(n):
            # Generate features
            features = self._generate_features(constraints)
            features_list.append(features)

            # Predict baseline outcome
            X = self._features_to_array(features)
            baseline_outcome = float(self.model.predict(X.reshape(1, -1))[0])

            # Calculate propensity (simplified)
            propensity = self._calculate_propensity(features)

            twin = DigitalTwin(
                twin_type=self.twin_type,
                brand=self.brand,
                features=features,
                baseline_outcome=baseline_outcome,
                baseline_propensity=propensity,
            )
            twins.append(twin)

        # Calculate feature summary
        feature_summary = self._calculate_feature_summary(features_list)

        population = TwinPopulation(
            twin_type=self.twin_type,
            brand=self.brand,
            twins=twins,
            size=n,
            feature_summary=feature_summary,
            model_id=self.model_id,
            generation_config={
                "constraints": constraints,
                "seed": seed,
            },
        )

        logger.info(f"Generated population of {len(population)} twins")
        return population

    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training."""

        # Store feature statistics for generation
        for col in self.feature_columns:
            if col not in data.columns:
                continue

            if data[col].dtype in ["object", "category", "bool"]:
                # Categorical: store distribution
                value_counts = data[col].value_counts(normalize=True)
                self._categorical_distributions[col] = value_counts.to_dict()

                # Encode for training
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Numerical: store mean and std
                self._feature_stats[col] = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                }

        X = data[self.feature_columns].values
        y = data[self.target_column].values

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        return X, y

    def _create_model(self, algorithm: str, **kwargs):
        """Create sklearn model based on algorithm."""
        if algorithm == "random_forest":
            return RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1,
            )
        elif algorithm == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=kwargs.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _generate_features(self, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate feature values for a single twin."""
        features = {}
        constraints = constraints or {}

        for col in self.feature_columns:
            if col in constraints:
                # Use constrained value
                features[col] = constraints[col]
            elif col in self._categorical_distributions:
                # Sample from categorical distribution
                dist = self._categorical_distributions[col]
                features[col] = np.random.choice(list(dist.keys()), p=list(dist.values()))
            elif col in self._feature_stats:
                # Sample from numerical distribution (truncated normal)
                stats = self._feature_stats[col]
                value = np.random.normal(stats["mean"], stats["std"])
                value = np.clip(value, stats["min"], stats["max"])
                features[col] = value
            else:
                features[col] = 0  # Default

        return features

    def _features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dict to numpy array for prediction."""
        values: List[Any] = []
        for col in self.feature_columns:
            val = features.get(col, 0)

            if col in self.label_encoders:
                # Encode categorical
                try:
                    val = self.label_encoders[col].transform([str(val)])[0]
                except ValueError:
                    val = 0  # Unknown category

            values.append(val)

        arr: np.ndarray = np.array(values, dtype=float)

        if self.scaler is not None:
            arr = self.scaler.transform(arr.reshape(1, -1))[0]

        return arr

    def _calculate_propensity(self, features: Dict[str, Any]) -> float:
        """Calculate treatment propensity score for twin."""
        # Simplified propensity based on key features
        propensity = 0.5  # Base propensity

        if self.twin_type == TwinType.HCP:
            # Higher propensity for high-decile, engaged HCPs
            decile = features.get("decile", 5)
            engagement = features.get("digital_engagement_score", 0.5)
            propensity = 0.3 + 0.05 * (11 - decile) + 0.2 * engagement

        return float(np.clip(propensity, 0.1, 0.9))

    def _calculate_feature_summary(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for generated population."""
        if not features_list:
            return {}

        df = pd.DataFrame(features_list)
        summary = {}

        for col in df.columns:
            if df[col].dtype in ["object", "category"]:
                summary[col] = df[col].value_counts().to_dict()
            else:
                summary[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }

        return summary

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging/storage."""
        return {
            "model_id": str(self.model_id) if self.model_id else None,
            "twin_type": self.twin_type.value,
            "brand": self.brand.value,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "is_trained": self.model is not None,
        }
