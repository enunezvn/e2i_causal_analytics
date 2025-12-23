"""Mock data connector for testing drift detection.

This module provides a mock implementation of BaseDataConnector that generates
synthetic data for testing drift detection algorithms without database access.

The mock generates reproducible data using seeds, with configurable drift
scenarios for validating detection logic.

Example:
    connector = MockDataConnector(drift_magnitude=0.3)
    data = await connector.query_features(
        feature_names=["age", "income"],
        time_window=TimeWindow(start=..., end=..., label="baseline"),
    )
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from src.agents.drift_monitor.connectors.base import (
    BaseDataConnector,
    FeatureData,
    PredictionData,
    TimeWindow,
)


class MockDataConnector(BaseDataConnector):
    """Mock data connector that generates synthetic data for testing.

    This connector simulates drift scenarios by generating baseline data
    from one distribution and current data from a shifted distribution.
    The magnitude of shift is configurable.

    Attributes:
        drift_magnitude: How much to shift current distribution (0=no drift)
        sample_size: Number of samples to generate
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        drift_magnitude: float = 0.2,
        sample_size: int = 1000,
        seed: int = 42,
    ):
        """Initialize mock connector.

        Args:
            drift_magnitude: Amount of distribution shift for current data
            sample_size: Number of samples to generate
            seed: Random seed for reproducibility
        """
        self.drift_magnitude = drift_magnitude
        self.sample_size = sample_size
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    async def query_features(
        self,
        feature_names: list[str],
        time_window: TimeWindow,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, FeatureData]:
        """Generate synthetic feature data.

        Baseline data is drawn from N(0, 1).
        Current data is drawn from N(drift_magnitude, 1 + drift_magnitude/5).

        Args:
            feature_names: List of feature names
            time_window: Time window (used to determine baseline vs current)
            filters: Optional filters (ignored in mock)

        Returns:
            Dictionary mapping feature name to FeatureData
        """
        is_baseline = time_window.label == "baseline"

        result = {}
        for i, feature in enumerate(feature_names):
            # Use feature index for reproducible per-feature variation
            feature_seed = self.seed + i

            if is_baseline:
                # Baseline: standard normal distribution
                values = self._generate_baseline(feature_seed)
            else:
                # Current: shifted distribution
                values = self._generate_current(feature_seed)

            # Generate timestamps
            timestamps = self._generate_timestamps(time_window, len(values))

            result[feature] = FeatureData(
                feature_name=feature,
                values=values,
                timestamps=timestamps,
                time_window=time_window,
            )

        return result

    async def query_predictions(
        self,
        model_id: str,
        time_window: TimeWindow,
        filters: dict[str, Any] | None = None,
    ) -> PredictionData:
        """Generate synthetic prediction data.

        Args:
            model_id: Model identifier
            time_window: Time window
            filters: Optional filters (ignored)

        Returns:
            PredictionData with synthetic predictions
        """
        is_baseline = time_window.label == "baseline"

        if is_baseline:
            # Baseline: predictions centered around 0.5
            scores = self._rng.beta(5, 5, size=self.sample_size)
        else:
            # Current: shifted predictions (higher or lower depending on drift)
            alpha = 5 + self.drift_magnitude * 2
            scores = self._rng.beta(alpha, 5, size=self.sample_size)

        # Generate binary labels from scores
        threshold = 0.5
        labels = (scores > threshold).astype(int)

        # Generate timestamps
        timestamps = self._generate_timestamps(time_window, len(scores))

        return PredictionData(
            model_id=model_id,
            scores=scores,
            labels=labels,
            timestamps=timestamps,
            time_window=time_window,
        )

    async def query_labeled_predictions(
        self,
        model_id: str,
        time_window: TimeWindow,
        filters: dict[str, Any] | None = None,
    ) -> PredictionData:
        """Generate synthetic labeled prediction data for concept drift.

        In baseline, predictions match actual labels well.
        In current, there's some degradation in prediction quality.

        Args:
            model_id: Model identifier
            time_window: Time window
            filters: Optional filters (ignored)

        Returns:
            PredictionData with predictions and actual labels
        """
        is_baseline = time_window.label == "baseline"

        # Generate actual labels (ground truth)
        actual_rate = 0.3  # 30% positive class
        actual_labels = self._rng.binomial(1, actual_rate, size=self.sample_size)

        if is_baseline:
            # Good predictions in baseline (low noise)
            noise_level = 0.1
        else:
            # Degraded predictions in current (higher noise - concept drift)
            noise_level = 0.1 + self.drift_magnitude

        # Generate scores based on actual labels with noise
        scores = np.clip(
            actual_labels * 0.8 + (1 - actual_labels) * 0.2 + self._rng.normal(0, noise_level, self.sample_size),
            0,
            1,
        )

        # Predicted labels
        labels = (scores > 0.5).astype(int)

        timestamps = self._generate_timestamps(time_window, len(scores))

        return PredictionData(
            model_id=model_id,
            scores=scores,
            labels=labels,
            actual_labels=actual_labels,
            timestamps=timestamps,
            time_window=time_window,
        )

    async def get_available_features(
        self,
        source_table: str | None = None,
    ) -> list[str]:
        """Return mock available features.

        Args:
            source_table: Ignored in mock

        Returns:
            List of mock feature names
        """
        return [
            "age",
            "income",
            "tenure_months",
            "prescriptions_30d",
            "visits_30d",
            "engagement_score",
            "market_share",
            "growth_rate",
        ]

    async def get_available_models(
        self,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return mock available models.

        Args:
            stage: Optional stage filter

        Returns:
            List of mock model metadata
        """
        models = [
            {
                "id": "mock-model-001",
                "name": "churn_predictor_v1",
                "stage": "production",
                "version": "1.0.0",
            },
            {
                "id": "mock-model-002",
                "name": "conversion_predictor_v2",
                "stage": "production",
                "version": "2.1.0",
            },
            {
                "id": "mock-model-003",
                "name": "engagement_classifier",
                "stage": "staging",
                "version": "1.5.0",
            },
        ]

        if stage:
            models = [m for m in models if m["stage"] == stage]

        return models

    async def health_check(self) -> dict[str, bool]:
        """Mock health check always returns healthy.

        Returns:
            Health status dictionary
        """
        return {
            "connected": True,
            "database": True,
            "predictions_table": True,
            "features_table": True,
        }

    async def close(self) -> None:
        """No-op for mock connector."""
        pass

    def _generate_baseline(self, seed: int) -> np.ndarray:
        """Generate baseline distribution data.

        Args:
            seed: Random seed

        Returns:
            Numpy array of baseline values
        """
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, size=self.sample_size)

    def _generate_current(self, seed: int) -> np.ndarray:
        """Generate current distribution data with drift.

        Args:
            seed: Random seed

        Returns:
            Numpy array of current values (shifted from baseline)
        """
        rng = np.random.default_rng(seed + 1000)  # Different seed
        # Shift mean and slightly increase variance
        return rng.normal(
            self.drift_magnitude,
            1 + self.drift_magnitude / 5,
            size=self.sample_size,
        )

    def _generate_timestamps(
        self,
        time_window: TimeWindow,
        count: int,
    ) -> np.ndarray:
        """Generate evenly spaced timestamps within the time window.

        Args:
            time_window: Time window for timestamps
            count: Number of timestamps to generate

        Returns:
            Numpy array of datetime objects
        """
        duration = time_window.end - time_window.start
        step = duration / count

        timestamps = [time_window.start + step * i for i in range(count)]
        return np.array(timestamps)

    def set_drift_magnitude(self, magnitude: float) -> None:
        """Update drift magnitude for testing different scenarios.

        Args:
            magnitude: New drift magnitude (0 = no drift)
        """
        self.drift_magnitude = magnitude

    def reset_rng(self) -> None:
        """Reset random number generator to seed for reproducibility."""
        self._rng = np.random.default_rng(self.seed)
