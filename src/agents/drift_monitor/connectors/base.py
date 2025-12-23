"""Base data connector interface for drift detection.

This module defines the abstract interface that all data connectors must implement.
The interface is designed for efficient drift detection queries with support for:
- Time-windowed feature data retrieval
- Prediction data retrieval for model drift
- Baseline vs current period comparisons

Contract: .claude/contracts/tier3-contracts.md
Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class TimeWindow:
    """Represents a time window for data queries.

    Attributes:
        start: Start of the time window
        end: End of the time window
        label: Human-readable label (e.g., "baseline", "current")
    """

    start: datetime
    end: datetime
    label: str = "custom"


@dataclass
class FeatureData:
    """Container for feature data retrieved from a connector.

    Attributes:
        feature_name: Name of the feature
        values: Numpy array of feature values
        timestamps: Optional array of corresponding timestamps
        entity_ids: Optional array of entity identifiers
        sample_count: Number of samples
        time_window: Time window for this data
    """

    feature_name: str
    values: np.ndarray
    timestamps: np.ndarray | None = None
    entity_ids: np.ndarray | None = None
    sample_count: int = 0
    time_window: TimeWindow | None = None

    def __post_init__(self):
        """Calculate sample count from values."""
        self.sample_count = len(self.values)


@dataclass
class PredictionData:
    """Container for prediction data retrieved from a connector.

    Attributes:
        model_id: Identifier of the model
        scores: Predicted probabilities/scores
        labels: Predicted class labels (for classification)
        actual_labels: Actual labels (if available, for concept drift)
        timestamps: Prediction timestamps
        entity_ids: Entity identifiers
        sample_count: Number of predictions
        time_window: Time window for this data
    """

    model_id: str
    scores: np.ndarray
    labels: np.ndarray | None = None
    actual_labels: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    entity_ids: np.ndarray | None = None
    sample_count: int = 0
    time_window: TimeWindow | None = None

    def __post_init__(self):
        """Calculate sample count from scores."""
        self.sample_count = len(self.scores)


class BaseDataConnector(ABC):
    """Abstract base class for data connectors.

    All data connectors for drift detection must implement this interface.
    The interface provides methods for:
    - Querying feature data for data drift detection
    - Querying prediction data for model drift detection
    - Querying labeled data for concept drift detection

    Implementations:
    - SupabaseDataConnector: Production connector using Supabase
    - MockDataConnector: Mock connector for testing

    Example:
        class MyConnector(BaseDataConnector):
            async def query_features(...) -> dict[str, FeatureData]:
                # Implementation
                pass
    """

    @abstractmethod
    async def query_features(
        self,
        feature_names: list[str],
        time_window: TimeWindow,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, FeatureData]:
        """Query feature values for drift detection.

        Args:
            feature_names: List of feature names to retrieve
            time_window: Time window for the query
            filters: Optional filters (e.g., {"brand": "remibrutinib"})

        Returns:
            Dictionary mapping feature name to FeatureData

        Raises:
            ConnectionError: If database connection fails
            ValueError: If invalid feature names provided
        """
        pass

    @abstractmethod
    async def query_predictions(
        self,
        model_id: str,
        time_window: TimeWindow,
        filters: dict[str, Any] | None = None,
    ) -> PredictionData:
        """Query prediction data for model drift detection.

        Args:
            model_id: Identifier of the model
            time_window: Time window for the query
            filters: Optional filters (e.g., {"segment": "high_value"})

        Returns:
            PredictionData containing predictions in the time window

        Raises:
            ConnectionError: If database connection fails
            ValueError: If invalid model_id provided
        """
        pass

    @abstractmethod
    async def query_labeled_predictions(
        self,
        model_id: str,
        time_window: TimeWindow,
        filters: dict[str, Any] | None = None,
    ) -> PredictionData:
        """Query predictions with actual labels for concept drift detection.

        This requires ground truth labels to be available, which may have
        a delay compared to predictions. Use for concept drift detection.

        Args:
            model_id: Identifier of the model
            time_window: Time window for the query
            filters: Optional filters

        Returns:
            PredictionData with both predicted and actual labels

        Raises:
            ConnectionError: If database connection fails
            ValueError: If no labeled data available
        """
        pass

    @abstractmethod
    async def get_available_features(
        self,
        source_table: str | None = None,
    ) -> list[str]:
        """Get list of available features for drift monitoring.

        Args:
            source_table: Optional table name to filter features

        Returns:
            List of available feature names
        """
        pass

    @abstractmethod
    async def get_available_models(
        self,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get list of available models for drift monitoring.

        Args:
            stage: Optional stage filter (e.g., "production", "staging")

        Returns:
            List of model metadata dictionaries
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, bool]:
        """Check connector health and connectivity.

        Returns:
            Dictionary with health status for each component
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connector and release resources."""
        pass
