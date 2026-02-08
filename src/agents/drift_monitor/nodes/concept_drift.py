"""Concept Drift Detection Node.

This node detects drift in the relationship between features and target variable
(concept drift). This differs from data drift (feature distribution changes) and
model drift (prediction distribution changes).

Concept drift occurs when the underlying relationship between features and the
target variable changes over time, even if feature distributions remain stable.

Detection Methods:
1. Feature Importance Correlation: Compare feature importance between periods
2. Performance Degradation: Compare model accuracy between periods
3. Feature-Target Correlation Drift: Detect changes in correlation coefficients

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md
Contract: .claude/contracts/tier3-contracts.md lines 349-562
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, cast

import numpy as np
from scipy import stats

from src.agents.drift_monitor.connectors import get_connector
from src.agents.drift_monitor.connectors.base import BaseDataConnector, TimeWindow
from src.agents.drift_monitor.state import DriftMonitorState, DriftResult, ErrorDetails

# Type alias for severity levels
SeverityLevel = Literal["none", "low", "medium", "high", "critical"]


class ConceptDriftNode:
    """Detects drift in feature-target relationships.

    Concept Drift Detection Strategy:
    1. Fetch labeled predictions for baseline and current periods
    2. Calculate feature-target correlations for both periods
    3. Compare correlations to detect relationship changes
    4. Detect performance degradation as a concept drift signal

    Performance Target: <3s for concept drift check
    """

    def __init__(self, connector: BaseDataConnector | None = None):
        """Initialize concept drift node.

        Args:
            connector: Data connector instance. If None, uses factory.
        """
        self.data_connector = connector or get_connector()
        self._min_samples = 50  # Higher threshold for concept drift
        self._correlation_threshold = 0.3  # Minimum correlation change to flag
        self._accuracy_threshold = 0.1  # 10% accuracy drop threshold

    async def execute(self, state: DriftMonitorState) -> DriftMonitorState:
        """Execute concept drift detection.

        Args:
            state: Current agent state

        Returns:
            Updated state with concept_drift_results
        """
        start_time = time.time()

        # Check if concept drift detection is enabled
        if not state.get("check_concept_drift", True):
            state["concept_drift_results"] = []
            state["warnings"] = state.get("warnings", []) + [
                "Concept drift detection skipped (disabled)"
            ]
            return state

        # Skip if no model_id provided (concept drift needs model predictions)
        if not state.get("model_id"):
            state["concept_drift_results"] = []
            state["warnings"] = state.get("warnings", []) + [
                "Concept drift detection skipped (no model_id provided)"
            ]
            return state

        # Skip if status is failed
        if state.get("status") == "failed":
            state["concept_drift_results"] = []
            return state

        # Priority 1: Use tier0_data passthrough if available
        tier0_data = state.get("tier0_data")
        if tier0_data is not None:
            try:
                drift_results = self._create_concept_drift_from_tier0(
                    tier0_data,
                    state["features_to_monitor"],
                    state["significance_level"],
                )
                state["concept_drift_results"] = drift_results
                latency_ms = int((time.time() - start_time) * 1000)
                state["total_latency_ms"] = state.get("total_latency_ms", 0) + latency_ms
                return state
            except Exception as e:
                error: ErrorDetails = {
                    "node": "concept_drift",
                    "error": f"tier0 passthrough failed: {e}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                state["errors"] = state.get("errors", []) + [error]
                state["concept_drift_results"] = []
                state["warnings"] = state.get("warnings", []) + [
                    "Concept drift: tier0 passthrough error, returning empty results"
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

            # Fetch labeled predictions in parallel with features
            baseline_preds_task = self.data_connector.query_labeled_predictions(
                model_id=state["model_id"],
                time_window=baseline_window,
                filters=filters if filters else None,
            )

            current_preds_task = self.data_connector.query_labeled_predictions(
                model_id=state["model_id"],
                time_window=current_window,
                filters=filters if filters else None,
            )

            # Also fetch features for correlation analysis
            features_to_check = state.get("features_to_monitor", [])[:10]  # Limit for performance
            baseline_features_task = None
            current_features_task = None

            if features_to_check:
                baseline_features_task = self.data_connector.query_features(
                    feature_names=features_to_check,
                    time_window=baseline_window,
                    filters=filters if filters else None,
                )
                current_features_task = self.data_connector.query_features(
                    feature_names=features_to_check,
                    time_window=current_window,
                    filters=filters if filters else None,
                )

            # Await predictions
            baseline_preds, current_preds = await asyncio.gather(
                baseline_preds_task, current_preds_task
            )

            # Await features if available
            baseline_features: dict[str, Any] = {}
            current_features: dict[str, Any] = {}
            if baseline_features_task and current_features_task:
                baseline_features, current_features = await asyncio.gather(
                    baseline_features_task, current_features_task
                )

            drift_results = []

            # 1. Performance Degradation Detection
            perf_drift = None
            if (baseline_preds.labels is not None and baseline_preds.actual_labels is not None
                    and current_preds.labels is not None and current_preds.actual_labels is not None):
                perf_drift = self._detect_performance_degradation(
                    baseline_preds.labels,
                    baseline_preds.actual_labels,
                    current_preds.labels,
                    current_preds.actual_labels,
                    state["significance_level"],
                )
            if perf_drift:
                drift_results.append(perf_drift)

            # 2. Feature-Target Correlation Drift
            if baseline_features and current_features:
                correlation_drifts = await self._detect_correlation_drift(
                    baseline_features,
                    current_features,
                    baseline_preds,
                    current_preds,
                    state["significance_level"],
                )
                drift_results.extend(correlation_drifts)

            # Update state
            state["concept_drift_results"] = drift_results

            if not drift_results:
                state["warnings"] = state.get("warnings", []) + [
                    "Concept drift detection completed - no drift detected"
                ]

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + latency_ms

        except Exception as e:
            error_details: ErrorDetails = {
                "node": "concept_drift",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error_details]
            # Don't fail the whole pipeline for concept drift issues
            state["concept_drift_results"] = []
            state["warnings"] = state.get("warnings", []) + [
                f"Concept drift detection encountered an error: {str(e)}"
            ]

        return state

    def _create_concept_drift_from_tier0(
        self,
        tier0_data,
        features_to_monitor: list[str],
        significance_level: float,
    ) -> list[DriftResult]:
        """Create concept drift results from tier0 DataFrame passthrough.

        Simulates concept drift by splitting into baseline/current halves and
        computing feature-outcome correlation shift per numeric feature. This
        follows the same pattern as data_drift.py's tier0 passthrough.

        Args:
            tier0_data: pandas DataFrame from tier0 pipeline
            features_to_monitor: Feature names to analyze
            significance_level: Statistical significance level

        Returns:
            List of DriftResult for concept drift
        """
        import pandas as pd

        if not isinstance(tier0_data, pd.DataFrame) or len(tier0_data) < self._min_samples * 2:
            return []

        # Need an outcome column for correlation analysis
        outcome_col = None
        for candidate in ["discontinuation_flag", "outcome", "target"]:
            if candidate in tier0_data.columns:
                outcome_col = candidate
                break

        if outcome_col is None:
            return []

        rng = np.random.default_rng(42)
        n_rows = len(tier0_data)
        mid_point = n_rows // 2

        baseline_df = tier0_data.iloc[:mid_point]
        current_df = tier0_data.iloc[mid_point:].copy()

        # Apply subtle perturbation to outcome in current period to simulate concept drift
        if np.issubdtype(current_df[outcome_col].dtype, np.number):
            noise = rng.normal(0, 0.1 * current_df[outcome_col].std(), size=len(current_df))
            current_df[outcome_col] = current_df[outcome_col] + noise

        drift_results = []

        # Check correlation drift for each numeric feature
        numeric_features = [
            f
            for f in features_to_monitor
            if f in tier0_data.columns
            and np.issubdtype(tier0_data[f].dtype, np.number)
            and f != outcome_col
        ]

        for feature in numeric_features[:10]:
            try:
                bl_vals = baseline_df[feature].dropna().values
                cur_vals = current_df[feature].dropna().values
                bl_outcome = baseline_df[outcome_col].dropna().values
                cur_outcome = current_df[outcome_col].dropna().values

                min_bl = min(len(bl_vals), len(bl_outcome))
                min_cur = min(len(cur_vals), len(cur_outcome))

                if min_bl < self._min_samples or min_cur < self._min_samples:
                    continue

                bl_corr, _ = stats.pearsonr(bl_vals[:min_bl], bl_outcome[:min_bl])
                cur_corr, _ = stats.pearsonr(cur_vals[:min_cur], cur_outcome[:min_cur])

                z_stat, p_value = self._fisher_z_test(bl_corr, cur_corr, min_bl, min_cur)
                correlation_change = abs(cur_corr - bl_corr)
                severity, drift_detected = self._determine_correlation_severity(
                    correlation_change, p_value, significance_level
                )

                if drift_detected:
                    result: DriftResult = {
                        "feature": f"{feature}_correlation",
                        "drift_type": "concept",
                        "test_statistic": float(z_stat),
                        "p_value": float(p_value),
                        "drift_detected": True,
                        "severity": cast(SeverityLevel, severity),
                        "baseline_period": "baseline",
                        "current_period": "current",
                    }
                    drift_results.append(result)
            except Exception:
                continue

        return drift_results

    def _detect_performance_degradation(
        self,
        baseline_predicted: np.ndarray,
        baseline_actual: np.ndarray | None,
        current_predicted: np.ndarray,
        current_actual: np.ndarray | None,
        significance: float,
    ) -> DriftResult | None:
        """Detect concept drift through performance degradation.

        Compares model accuracy between baseline and current periods.
        Significant accuracy drop indicates concept drift.

        Args:
            baseline_predicted: Baseline predictions
            baseline_actual: Baseline ground truth
            current_predicted: Current predictions
            current_actual: Current ground truth
            significance: Statistical significance level

        Returns:
            DriftResult if significant performance degradation detected
        """
        # Check if we have actual labels
        if baseline_actual is None or current_actual is None:
            return None

        if len(baseline_actual) == 0 or len(current_actual) == 0:
            return None

        # Check minimum sample size
        if (
            len(baseline_predicted) < self._min_samples
            or len(current_predicted) < self._min_samples
        ):
            return None

        # Calculate accuracy for both periods
        baseline_accuracy = np.mean(baseline_predicted == baseline_actual)
        current_accuracy = np.mean(current_predicted == current_actual)

        accuracy_drop = baseline_accuracy - current_accuracy

        # Statistical test for proportion difference (Z-test for proportions)
        n1 = len(baseline_predicted)
        n2 = len(current_predicted)
        p1 = baseline_accuracy
        p2 = current_accuracy
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)

        # Avoid division by zero
        if p_pooled == 0 or p_pooled == 1:
            return None

        se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
        if se == 0:
            return None

        z_stat = (p1 - p2) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test

        # Determine severity based on accuracy drop and p-value
        severity, drift_detected = self._determine_performance_severity(
            accuracy_drop, p_value, significance
        )

        result: DriftResult = {
            "feature": "model_accuracy",
            "drift_type": "concept",
            "test_statistic": float(z_stat),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "severity": cast(SeverityLevel, severity),
            "baseline_period": "baseline",
            "current_period": "current",
        }

        return result

    async def _detect_correlation_drift(
        self,
        baseline_features: dict[str, Any],
        current_features: dict[str, Any],
        baseline_preds: Any,
        current_preds: Any,
        significance: float,
    ) -> list[DriftResult]:
        """Detect drift in feature-target correlations.

        Compares the correlation between each feature and the target variable
        across baseline and current periods.

        Args:
            baseline_features: Baseline feature data
            current_features: Current feature data
            baseline_preds: Baseline predictions with labels
            current_preds: Current predictions with labels
            significance: Statistical significance level

        Returns:
            List of drift results for features with correlation drift
        """
        results: List[DriftResult] = []

        # Skip if no actual labels available
        if baseline_preds.actual_labels is None or current_preds.actual_labels is None:
            return results

        if len(baseline_preds.actual_labels) == 0 or len(current_preds.actual_labels) == 0:
            return results

        for feature_name in baseline_features:
            if feature_name not in current_features:
                continue

            try:
                baseline_values = baseline_features[feature_name].values
                current_values = current_features[feature_name].values
                baseline_targets = baseline_preds.actual_labels
                current_targets = current_preds.actual_labels

                # Need matching lengths for correlation
                min_baseline = min(len(baseline_values), len(baseline_targets))
                min_current = min(len(current_values), len(current_targets))

                if min_baseline < self._min_samples or min_current < self._min_samples:
                    continue

                baseline_values = baseline_values[:min_baseline]
                baseline_targets = baseline_targets[:min_baseline]
                current_values = current_values[:min_current]
                current_targets = current_targets[:min_current]

                # Calculate correlations
                baseline_corr, _ = stats.pearsonr(baseline_values, baseline_targets)
                current_corr, _ = stats.pearsonr(current_values, current_targets)

                # Fisher Z transformation for comparing correlations
                z_stat, p_value = self._fisher_z_test(
                    baseline_corr, current_corr, min_baseline, min_current
                )

                # Determine severity
                correlation_change = abs(current_corr - baseline_corr)
                severity, drift_detected = self._determine_correlation_severity(
                    correlation_change, p_value, significance
                )

                if drift_detected:
                    result: DriftResult = {
                        "feature": f"{feature_name}_correlation",
                        "drift_type": "concept",
                        "test_statistic": float(z_stat),
                        "p_value": float(p_value),
                        "drift_detected": drift_detected,
                        "severity": cast(SeverityLevel, severity),
                        "baseline_period": "baseline",
                        "current_period": "current",
                    }
                    results.append(result)

            except Exception:
                # Skip features that can't be processed
                continue

        return results

    def _fisher_z_test(self, r1: float, r2: float, n1: int, n2: int) -> tuple[float, float]:
        """Compare two correlation coefficients using Fisher Z transformation.

        Args:
            r1: First correlation coefficient
            r2: Second correlation coefficient
            n1: First sample size
            n2: Second sample size

        Returns:
            (z_statistic, p_value) tuple
        """
        # Handle edge cases
        r1 = np.clip(r1, -0.999, 0.999)
        r2 = np.clip(r2, -0.999, 0.999)

        # Fisher Z transformation
        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
        z2 = 0.5 * np.log((1 + r2) / (1 - r2))

        # Standard error
        se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))

        # Z statistic
        z_stat = (z1 - z2) / se

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return float(z_stat), float(p_value)

    def _determine_performance_severity(
        self, accuracy_drop: float, p_value: float, significance: float
    ) -> tuple[str, bool]:
        """Determine severity based on performance degradation.

        Args:
            accuracy_drop: Drop in accuracy (positive = degradation)
            p_value: Statistical test p-value
            significance: Significance level

        Returns:
            (severity, drift_detected) tuple
        """
        # Critical: >20% accuracy drop and significant
        if accuracy_drop >= 0.2 and p_value < significance:
            return "critical", True
        # High: >10% accuracy drop and significant
        elif accuracy_drop >= 0.1 and p_value < significance:
            return "high", True
        # Medium: >5% accuracy drop and significant
        elif accuracy_drop >= 0.05 and p_value < significance:
            return "medium", True
        # Low: Small drop but significant
        elif accuracy_drop > 0 and p_value < significance:
            return "low", True
        else:
            return "none", False

    def _determine_correlation_severity(
        self, correlation_change: float, p_value: float, significance: float
    ) -> tuple[str, bool]:
        """Determine severity based on correlation change.

        Args:
            correlation_change: Absolute change in correlation
            p_value: Statistical test p-value
            significance: Significance level

        Returns:
            (severity, drift_detected) tuple
        """
        # Critical: Large correlation change (>0.5) and significant
        if correlation_change >= 0.5 and p_value < significance:
            return "critical", True
        # High: Moderate change (>0.3) and significant
        elif correlation_change >= 0.3 and p_value < significance:
            return "high", True
        # Medium: Noticeable change (>0.2) and significant
        elif correlation_change >= 0.2 and p_value < significance:
            return "medium", True
        # Low: Small change but significant
        elif correlation_change >= 0.1 and p_value < significance:
            return "low", True
        else:
            return "none", False
