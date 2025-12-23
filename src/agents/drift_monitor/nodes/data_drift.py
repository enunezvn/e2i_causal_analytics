"""Data Drift Detection Node.

This node detects drift in feature distributions by comparing current data
against baseline distributions using PSI (Population Stability Index) and
KS (Kolmogorov-Smirnov) test.

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md lines 160-310
Contract: .claude/contracts/tier3-contracts.md lines 349-562
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from scipy import stats

from src.agents.drift_monitor.connectors import get_connector
from src.agents.drift_monitor.connectors.base import BaseDataConnector, TimeWindow
from src.agents.drift_monitor.state import DriftMonitorState, DriftResult, ErrorDetails


class DataDriftNode:
    """Detects drift in feature distributions.

    Drift Detection Strategy:
    1. Fetch baseline and current data in parallel
    2. For each feature:
       - Calculate PSI (Population Stability Index)
       - Run KS test for continuous features
       - Determine drift severity
    3. Return structured drift results

    Performance Target: <8s for 50 features
    """

    def __init__(self, connector: BaseDataConnector | None = None):
        """Initialize data drift node.

        Args:
            connector: Data connector instance. If None, uses factory.
        """
        self.data_connector = connector or get_connector()
        self._min_samples = 30  # Minimum samples required for drift detection

    async def execute(self, state: DriftMonitorState) -> DriftMonitorState:
        """Execute data drift detection.

        Args:
            state: Current agent state with features_to_monitor

        Returns:
            Updated state with data_drift_results
        """
        start_time = time.time()

        # Check if data drift detection is enabled
        if not state.get("check_data_drift", True):
            state["data_drift_results"] = []
            state["warnings"] = state.get("warnings", []) + [
                "Data drift detection skipped (disabled)"
            ]
            return state

        # Skip if status is failed
        if state.get("status") == "failed":
            state["data_drift_results"] = []
            return state

        try:
            # Update status
            state["status"] = "detecting"

            # Fetch data
            baseline_data, current_data = await self._fetch_data(state)

            # Store timestamps
            state["baseline_timestamp"] = self._get_baseline_timestamp(state["time_window"])
            state["current_timestamp"] = datetime.now(timezone.utc).isoformat()

            # Detect drift for all features in parallel
            drift_results = await self._detect_drift_parallel(
                state["features_to_monitor"],
                baseline_data,
                current_data,
                state["significance_level"],
                state["psi_threshold"],
            )

            # Update state
            state["data_drift_results"] = drift_results
            state["features_checked"] = len(state["features_to_monitor"])

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["detection_latency_ms"] = state.get("detection_latency_ms", 0) + latency_ms

        except Exception as e:
            error: ErrorDetails = {
                "node": "data_drift",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            state["status"] = "failed"
            state["data_drift_results"] = []

        return state

    async def _fetch_data(
        self, state: DriftMonitorState
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Fetch baseline and current data in parallel.

        Args:
            state: Current agent state

        Returns:
            (baseline_data, current_data) tuple
        """
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

        # Fetch in parallel using new connector interface
        baseline_task = self.data_connector.query_features(
            feature_names=state["features_to_monitor"],
            time_window=baseline_window,
            filters=filters if filters else None,
        )

        current_task = self.data_connector.query_features(
            feature_names=state["features_to_monitor"],
            time_window=current_window,
            filters=filters if filters else None,
        )

        baseline_result, current_result = await asyncio.gather(baseline_task, current_task)

        # Convert FeatureData to numpy arrays for compatibility
        baseline_data = {name: data.values for name, data in baseline_result.items()}
        current_data = {name: data.values for name, data in current_result.items()}

        return baseline_data, current_data

    async def _detect_drift_parallel(
        self,
        features: list[str],
        baseline_data: dict[str, np.ndarray],
        current_data: dict[str, np.ndarray],
        significance_level: float,
        psi_threshold: float,
    ) -> list[DriftResult]:
        """Detect drift for all features in parallel.

        Args:
            features: List of feature names
            baseline_data: Baseline distributions
            current_data: Current distributions
            significance_level: Statistical significance level
            psi_threshold: PSI warning threshold

        Returns:
            List of drift results
        """
        tasks = [
            self._detect_feature_drift(
                feature,
                baseline_data[feature],
                current_data[feature],
                significance_level,
                psi_threshold,
            )
            for feature in features
        ]

        results = await asyncio.gather(*tasks)

        # Filter out None results (insufficient data)
        return [r for r in results if r is not None]

    async def _detect_feature_drift(
        self,
        feature: str,
        baseline: np.ndarray,
        current: np.ndarray,
        significance: float,
        psi_threshold: float,
    ) -> DriftResult | None:
        """Detect drift for a single feature.

        Args:
            feature: Feature name
            baseline: Baseline distribution
            current: Current distribution
            significance: Statistical significance level
            psi_threshold: PSI warning threshold

        Returns:
            DriftResult or None if insufficient data
        """
        # Check minimum sample size
        if len(baseline) < self._min_samples or len(current) < self._min_samples:
            return None

        # Calculate PSI
        psi = self._calculate_psi(baseline, current)

        # Run KS test (for continuous features)
        ks_stat, p_value = stats.ks_2samp(baseline, current)

        # Determine severity
        severity, drift_detected = self._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        result: DriftResult = {
            "feature": feature,
            "drift_type": "data",
            "test_statistic": float(psi),  # Use PSI as primary statistic
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "severity": severity,
            "baseline_period": "baseline",
            "current_period": "current",
        }

        return result

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index.

        PSI Formula:
            PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

        Interpretation:
            PSI < 0.1: No significant change
            0.1 <= PSI < 0.2: Moderate change
            PSI >= 0.2: Significant change

        Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md lines 289-309

        Args:
            expected: Baseline distribution
            actual: Current distribution
            bins: Number of bins for bucketing

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

        # PSI formula: sum((actual - expected) * ln(actual / expected))
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return float(psi)

    def _determine_severity(
        self, psi: float, p_value: float, significance: float, psi_threshold: float
    ) -> tuple[str, bool]:
        """Determine drift severity based on PSI and p-value.

        Severity Thresholds:
            - PSI >= 0.25 OR p < significance/10: CRITICAL
            - PSI >= 0.2 OR p < significance: HIGH
            - 0.1 <= PSI < 0.2: MEDIUM
            - 0.05 <= PSI < 0.1: LOW
            - PSI < 0.05: NONE

        Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md lines 254-276

        Args:
            psi: Population Stability Index
            p_value: KS test p-value
            significance: Statistical significance level
            psi_threshold: PSI warning threshold (typically 0.1)

        Returns:
            (severity, drift_detected) tuple
        """
        psi_critical = 0.25

        if psi >= psi_critical or p_value < significance / 10:
            return "critical", True
        elif psi >= psi_threshold or p_value < significance:
            # High or medium based on PSI magnitude
            severity = "high" if psi >= 0.2 else "medium"
            return severity, True
        elif psi >= 0.05:
            return "low", True
        else:
            return "none", False

    def _get_baseline_timestamp(self, time_window: str) -> str:
        """Calculate baseline timestamp from time window.

        Args:
            time_window: Time window string (e.g., '7d', '30d')

        Returns:
            ISO timestamp for baseline period
        """
        # Parse time window
        days = int(time_window.replace("d", ""))
        baseline_time = datetime.now(timezone.utc) - timedelta(days=days)

        return baseline_time.isoformat()
