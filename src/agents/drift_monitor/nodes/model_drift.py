"""Model Drift Detection Node.

This node detects drift in model prediction distributions by comparing
current predictions against baseline predictions.

Drift Types Detected:
1. Prediction Score Drift: Changes in predicted probability/score distribution (KS test)
2. Prediction Class Drift: Changes in predicted class distribution (Chi-square test)

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md lines 312-461
Contract: .claude/contracts/tier3-contracts.md lines 349-562
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Literal, cast

import numpy as np
from scipy import stats

from src.agents.drift_monitor.connectors import get_connector
from src.agents.drift_monitor.connectors.base import BaseDataConnector, TimeWindow
from src.agents.drift_monitor.state import DriftMonitorState, DriftResult, ErrorDetails


class ModelDriftNode:
    """Detects drift in model predictions.

    Drift Detection Strategy:
    1. Fetch baseline and current predictions
    2. Check prediction score distribution (KS test)
    3. Check prediction class distribution (Chi-square test)
    4. Determine drift severity

    Performance Target: <2s for predictions check
    """

    def __init__(self, connector: BaseDataConnector | None = None):
        """Initialize model drift node.

        Args:
            connector: Data connector instance. If None, uses factory.
        """
        self.data_connector = connector or get_connector()
        self._min_samples = 30

    async def execute(self, state: DriftMonitorState) -> DriftMonitorState:
        """Execute model drift detection.

        Args:
            state: Current agent state with model_id

        Returns:
            Updated state with model_drift_results
        """
        start_time = time.time()

        # Check if model drift detection is enabled
        if not state.get("check_model_drift", True):
            state["model_drift_results"] = []
            state["warnings"] = state.get("warnings", []) + [
                "Model drift detection skipped (disabled)"
            ]
            return state

        # Skip if no model_id provided
        if not state.get("model_id"):
            state["model_drift_results"] = []
            state["warnings"] = state.get("warnings", []) + [
                "Model drift detection skipped (no model_id provided)"
            ]
            return state

        # Skip if status is failed
        if state.get("status") == "failed":
            state["model_drift_results"] = []
            return state

        # Priority 1: Use tier0_data passthrough if available
        tier0_data = state.get("tier0_data")
        if tier0_data is not None:
            try:
                drift_results = self._create_model_drift_from_tier0(
                    tier0_data,
                    state["features_to_monitor"],
                    state["significance_level"],
                    state.get("psi_threshold", 0.1),
                )
                state["model_drift_results"] = drift_results
                latency_ms = int((time.time() - start_time) * 1000)
                state["total_latency_ms"] = state.get("total_latency_ms", 0) + latency_ms
                return state
            except Exception as e:
                error: ErrorDetails = {
                    "node": "model_drift",
                    "error": f"tier0 passthrough failed: {e}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                state["errors"] = state.get("errors", []) + [error]
                state["model_drift_results"] = []
                state["warnings"] = state.get("warnings", []) + [
                    "Model drift: tier0 passthrough error, returning empty results"
                ]
                return state

        try:
            # Parse time window
            days = int(state["time_window"].replace("d", ""))
            now = datetime.now(timezone.utc)

            # Baseline period: from 2*days ago to 1*days ago
            baseline_window = TimeWindow(
                start=now - timedelta(days=days * 2),
                end=now - timedelta(days=days),
                label="baseline",
            )

            # Current period: from 1*days ago to now
            current_window = TimeWindow(
                start=now - timedelta(days=days),
                end=now,
                label="current",
            )

            # Build filters
            filters = {}
            if state.get("brand"):
                filters["brand"] = state["brand"]

            # Fetch baseline and current predictions
            baseline_preds = await self.data_connector.query_predictions(
                model_id=state["model_id"],
                time_window=baseline_window,
                filters=filters if filters else None,
            )

            current_preds = await self.data_connector.query_predictions(
                model_id=state["model_id"],
                time_window=current_window,
                filters=filters if filters else None,
            )

            # Detect drift
            drift_results = []

            # 1. Prediction score drift (KS test)
            score_drift = self._detect_score_drift(
                baseline_preds.scores,
                current_preds.scores,
                state["significance_level"],
                state["psi_threshold"],
            )
            if score_drift:
                drift_results.append(score_drift)

            # 2. Prediction class drift (Chi-square test)
            if baseline_preds.labels is not None and current_preds.labels is not None:
                class_drift = self._detect_class_drift(
                    baseline_preds.labels,
                    current_preds.labels,
                    state["significance_level"],
                )
                if class_drift:
                    drift_results.append(class_drift)

            # Update state
            state["model_drift_results"] = drift_results

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + latency_ms

        except Exception as e:
            error_details: ErrorDetails = {
                "node": "model_drift",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error_details]
            state["status"] = "failed"
            state["model_drift_results"] = []

        return state

    def _create_model_drift_from_tier0(
        self,
        tier0_data,
        features_to_monitor: list[str],
        significance_level: float,
        psi_threshold: float,
    ) -> list[DriftResult]:
        """Create model drift results from tier0 DataFrame passthrough.

        Simulates prediction score drift by splitting the DataFrame into
        baseline/current halves and computing per-feature pseudo-prediction
        distribution shift via KS-test. This follows the same pattern as
        data_drift.py's _create_drift_splits_from_tier0().

        Args:
            tier0_data: pandas DataFrame from tier0 pipeline
            features_to_monitor: Feature names to analyze
            significance_level: Statistical significance level
            psi_threshold: PSI warning threshold

        Returns:
            List of DriftResult for model drift
        """
        import pandas as pd

        if not isinstance(tier0_data, pd.DataFrame) or len(tier0_data) < self._min_samples * 2:
            return []

        rng = np.random.default_rng(42)
        n_rows = len(tier0_data)
        mid_point = n_rows // 2

        # Select numeric features to create pseudo-prediction scores
        numeric_features = [
            f
            for f in features_to_monitor
            if f in tier0_data.columns and np.issubdtype(tier0_data[f].dtype, np.number)
        ][:5]

        if not numeric_features:
            return []

        # Create pseudo-prediction scores by averaging normalized numeric features
        feature_data = tier0_data[numeric_features].copy()
        # Normalize each feature to 0-1 range
        for col in numeric_features:
            col_min, col_max = feature_data[col].min(), feature_data[col].max()
            if col_max > col_min:
                feature_data[col] = (feature_data[col] - col_min) / (col_max - col_min)
            else:
                feature_data[col] = 0.5

        pseudo_scores = feature_data.mean(axis=1).values

        # Split into baseline/current and apply drift
        baseline_scores = pseudo_scores[:mid_point]
        current_scores = pseudo_scores[mid_point:].copy()
        # Apply shift to simulate model drift
        drift_shift = 0.15 * np.std(baseline_scores)
        current_scores += rng.normal(
            drift_shift, np.std(baseline_scores) * 0.05, size=len(current_scores)
        )
        current_scores = np.clip(current_scores, 0.0, 1.0)

        drift_results = []

        # Score drift (KS test on pseudo-predictions)
        score_drift = self._detect_score_drift(
            baseline_scores, current_scores, significance_level, psi_threshold
        )
        if score_drift:
            drift_results.append(score_drift)

        # Class drift (binarize scores at 0.5 threshold)
        baseline_classes = (baseline_scores > 0.5).astype(int)
        current_classes = (current_scores > 0.5).astype(int)
        class_drift = self._detect_class_drift(
            baseline_classes, current_classes, significance_level
        )
        if class_drift:
            drift_results.append(class_drift)

        return drift_results

    def _detect_score_drift(
        self,
        baseline_scores: np.ndarray,
        current_scores: np.ndarray,
        significance: float,
        psi_threshold: float,
    ) -> DriftResult | None:
        """Detect drift in prediction score distribution.

        Uses KS test to compare continuous prediction score distributions.

        Args:
            baseline_scores: Baseline prediction scores
            current_scores: Current prediction scores
            significance: Statistical significance level
            psi_threshold: PSI warning threshold

        Returns:
            DriftResult or None if insufficient data
        """
        # Check minimum sample size
        if len(baseline_scores) < self._min_samples or len(current_scores) < self._min_samples:
            return None

        # Run KS test
        ks_stat, p_value = stats.ks_2samp(baseline_scores, current_scores)

        # Calculate PSI for severity determination
        psi = self._calculate_psi(baseline_scores, current_scores)

        # Determine severity
        severity, drift_detected = self._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        result: DriftResult = {
            "feature": "prediction_scores",
            "drift_type": "model",
            "test_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "severity": cast(Literal["none", "low", "medium", "high", "critical"], severity),
            "baseline_period": "baseline",
            "current_period": "current",
        }

        return result

    def _detect_class_drift(
        self, baseline_classes: np.ndarray, current_classes: np.ndarray, significance: float
    ) -> DriftResult | None:
        """Detect drift in prediction class distribution.

        Uses Chi-square test to compare categorical prediction class distributions.

        Args:
            baseline_classes: Baseline prediction classes
            current_classes: Current prediction classes
            significance: Statistical significance level

        Returns:
            DriftResult or None if insufficient data
        """
        # Check minimum sample size
        if len(baseline_classes) < self._min_samples or len(current_classes) < self._min_samples:
            return None

        # Create contingency table
        unique_classes = np.unique(np.concatenate([baseline_classes, current_classes]))

        baseline_counts = np.array([np.sum(baseline_classes == c) for c in unique_classes])
        current_counts = np.array([np.sum(current_classes == c) for c in unique_classes])

        # Chi-square test
        contingency_table = np.array([baseline_counts, current_counts])
        chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)

        # Determine severity based on p-value
        if p_value < significance / 10:
            severity = "critical"
            drift_detected = True
        elif p_value < significance:
            severity = "high"
            drift_detected = True
        elif p_value < significance * 2:
            severity = "medium"
            drift_detected = True
        else:
            severity = "none"
            drift_detected = False

        result: DriftResult = {
            "feature": "prediction_classes",
            "drift_type": "model",
            "test_statistic": float(chi2_stat),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "severity": cast(Literal["none", "low", "medium", "high", "critical"], severity),
            "baseline_period": "baseline",
            "current_period": "current",
        }

        return result

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index for continuous predictions.

        Args:
            expected: Baseline distribution
            actual: Current distribution
            bins: Number of bins

        Returns:
            PSI value
        """
        # Create bins from expected distribution
        _, bin_edges = np.histogram(expected, bins=bins)

        # Calculate proportions
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Avoid division by zero
        expected_pct = np.clip(expected_pct, 0.0001, None)
        actual_pct = np.clip(actual_pct, 0.0001, None)

        # PSI formula
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return float(psi)

    def _determine_severity(
        self, psi: float, p_value: float, significance: float, psi_threshold: float
    ) -> tuple[str, bool]:
        """Determine drift severity.

        Args:
            psi: Population Stability Index
            p_value: Statistical test p-value
            significance: Significance level
            psi_threshold: PSI warning threshold

        Returns:
            (severity, drift_detected) tuple
        """
        psi_critical = 0.25

        if psi >= psi_critical or p_value < significance / 10:
            return "critical", True
        elif psi >= psi_threshold or p_value < significance:
            severity = "high" if psi >= 0.2 else "medium"
            return severity, True
        elif psi >= 0.05:
            return "low", True
        else:
            return "none", False
