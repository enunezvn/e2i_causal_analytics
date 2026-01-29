"""Alert Aggregator Node.

This node aggregates drift detection results from all detectors and generates
alerts with recommended actions.

Responsibilities:
1. Calculate composite drift score
2. Identify features with drift
3. Generate critical/warning alerts
4. Create human-readable drift summary
5. Provide recommended actions

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md lines 463-657
Contract: .claude/contracts/tier3-contracts.md lines 349-562
"""

import time
import uuid
from datetime import datetime, timezone

from src.agents.drift_monitor.state import DriftAlert, DriftMonitorState, DriftResult, ErrorDetails

# Severity weights for composite drift score calculation
SEVERITY_WEIGHTS = {"none": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}


class AlertAggregatorNode:
    """Aggregates drift results and generates alerts.

    Aggregation Strategy:
    1. Collect all drift results (data, model, concept)
    2. Calculate composite drift score (weighted average of severity)
    3. Identify features with drift
    4. Generate alerts for critical/high severity drifts
    5. Create drift summary and recommended actions

    Performance Target: <100ms for alert generation
    """

    def __init__(self):
        """Initialize alert aggregator node."""
        pass

    async def execute(self, state: DriftMonitorState) -> DriftMonitorState:
        """Execute alert aggregation.

        Args:
            state: Current agent state with drift results

        Returns:
            Updated state with alerts, drift score, and summary
        """
        start_time = time.time()

        # Skip if status is failed
        if state.get("status") == "failed":
            state["overall_drift_score"] = 0.0
            state["features_with_drift"] = []
            state["alerts"] = []
            state["drift_summary"] = "Drift detection failed"
            state["recommended_actions"] = []
            return state

        try:
            # Update status
            state["status"] = "aggregating"

            # Collect all drift results
            all_results = self._collect_all_results(state)

            # Calculate composite drift score
            drift_score = self._calculate_drift_score(all_results)

            # Identify features with drift
            features_with_drift = self._identify_drifted_features(all_results)

            # Generate alerts
            alerts = self._generate_alerts(all_results)

            # Create drift summary
            drift_summary = self._create_drift_summary(
                all_results, drift_score, features_with_drift
            )

            # Generate recommended actions
            recommended_actions = self._generate_recommendations(all_results, drift_score)

            # Generate drift interpretation with root cause analysis
            interpretation = self._interpret_drift(all_results, drift_score)

            # Enhance drift summary with interpretation
            enhanced_summary = self._enhance_summary_with_interpretation(
                drift_summary, interpretation
            )

            # Update state
            state["overall_drift_score"] = drift_score
            state["features_with_drift"] = features_with_drift
            state["alerts"] = alerts
            state["drift_summary"] = enhanced_summary
            state["recommended_actions"] = recommended_actions
            state["drift_interpretation"] = interpretation
            state["status"] = "completed"

            # VALIDATION: Ensure high drift always has recommendations
            if drift_score > 0.7 and not recommended_actions:
                state["recommended_actions"] = [
                    f"CRITICAL: Drift score {drift_score:.2f} requires immediate investigation",
                    "Review feature distributions for data quality issues",
                    "Consider retraining model with recent data",
                ]

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + latency_ms

        except Exception as e:
            error: ErrorDetails = {
                "node": "alert_aggregator",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            state["status"] = "failed"
            state["overall_drift_score"] = 0.0
            state["features_with_drift"] = []
            state["alerts"] = []
            state["drift_summary"] = "Alert aggregation failed"
            state["recommended_actions"] = []

        return state

    def _collect_all_results(self, state: DriftMonitorState) -> list[DriftResult]:
        """Collect all drift results from all detectors.

        Args:
            state: Current agent state

        Returns:
            Combined list of all drift results
        """
        all_results: list[DriftResult] = []

        # Add data drift results
        if "data_drift_results" in state:
            all_results.extend(state["data_drift_results"])

        # Add model drift results
        if "model_drift_results" in state:
            all_results.extend(state["model_drift_results"])

        # Add concept drift results
        if "concept_drift_results" in state:
            all_results.extend(state["concept_drift_results"])

        return all_results

    def _calculate_drift_score(self, results: list[DriftResult]) -> float:
        """Calculate composite drift score.

        Drift score is weighted average of severity across all features.

        Score Interpretation:
            0.0 - 0.2: No significant drift
            0.2 - 0.4: Low drift
            0.4 - 0.6: Moderate drift
            0.6 - 0.8: High drift
            0.8 - 1.0: Critical drift

        Args:
            results: All drift detection results

        Returns:
            Composite drift score (0.0 to 1.0)
        """
        if not results:
            return 0.0

        # Calculate weighted average
        total_weight = sum(SEVERITY_WEIGHTS[r["severity"]] for r in results)
        drift_score = total_weight / len(results)

        return round(drift_score, 3)

    def _identify_drifted_features(self, results: list[DriftResult]) -> list[str]:
        """Identify features showing drift.

        Args:
            results: All drift detection results

        Returns:
            List of feature names with detected drift
        """
        drifted_features = [r["feature"] for r in results if r["drift_detected"]]

        # Remove duplicates and sort
        return sorted(set(drifted_features))

    def _generate_alerts(self, results: list[DriftResult]) -> list[DriftAlert]:
        """Generate alerts for critical and high severity drifts.

        Alert Generation Rules:
        - Critical severity → Critical alert
        - High severity → Warning alert
        - Medium/Low severity → No alert (just logged in results)

        Args:
            results: All drift detection results

        Returns:
            List of generated alerts
        """
        alerts: list[DriftAlert] = []

        # Group by drift_type and severity
        critical_by_type: dict[str, list[str]] = {"data": [], "model": [], "concept": []}
        high_by_type: dict[str, list[str]] = {"data": [], "model": [], "concept": []}

        for result in results:
            if result["severity"] == "critical":
                critical_by_type[result["drift_type"]].append(result["feature"])
            elif result["severity"] == "high":
                high_by_type[result["drift_type"]].append(result["feature"])

        # Generate critical alerts
        for drift_type, features in critical_by_type.items():
            if features:
                alert = self._create_alert(
                    severity="critical", drift_type=drift_type, affected_features=features
                )
                alerts.append(alert)

        # Generate warning alerts
        for drift_type, features in high_by_type.items():
            if features:
                alert = self._create_alert(
                    severity="warning", drift_type=drift_type, affected_features=features
                )
                alerts.append(alert)

        return alerts

    def _create_alert(
        self, severity: str, drift_type: str, affected_features: list[str]
    ) -> DriftAlert:
        """Create a drift alert.

        Args:
            severity: Alert severity (critical or warning)
            drift_type: Type of drift (data, model, concept)
            affected_features: Features showing drift

        Returns:
            DriftAlert
        """
        # Generate alert message
        feature_list = ", ".join(affected_features[:5])
        if len(affected_features) > 5:
            feature_list += f" and {len(affected_features) - 5} more"

        messages = {
            "critical": {
                "data": f"CRITICAL data drift detected in features: {feature_list}",
                "model": f"CRITICAL model drift detected in predictions: {feature_list}",
                "concept": f"CRITICAL concept drift detected in features: {feature_list}",
            },
            "warning": {
                "data": f"HIGH data drift detected in features: {feature_list}",
                "model": f"HIGH model drift detected in predictions: {feature_list}",
                "concept": f"HIGH concept drift detected in features: {feature_list}",
            },
        }

        # Generate recommended action
        actions = {
            "critical": {
                "data": "Immediate action required: Retrain model with recent data and investigate feature distribution changes",
                "model": "Immediate action required: Investigate model degradation and consider retraining or recalibration",
                "concept": "Immediate action required: Review ground truth labels and feature-target relationships",
            },
            "warning": {
                "data": "Monitor closely: Schedule model retraining if drift persists",
                "model": "Monitor closely: Check prediction accuracy on recent data",
                "concept": "Monitor closely: Validate model performance on current data",
            },
        }

        alert: DriftAlert = {
            "alert_id": str(uuid.uuid4()),
            "severity": severity,  # type: ignore
            "drift_type": drift_type,  # type: ignore
            "affected_features": affected_features,
            "message": messages[severity][drift_type],
            "recommended_action": actions[severity][drift_type],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return alert

    def _create_drift_summary(
        self, results: list[DriftResult], drift_score: float, features_with_drift: list[str]
    ) -> str:
        """Create human-readable drift summary.

        Args:
            results: All drift detection results
            drift_score: Composite drift score
            features_with_drift: Features showing drift

        Returns:
            Drift summary string
        """
        if not results:
            return "No drift detection results available"

        # Count by severity
        severity_counts = {
            "critical": sum(1 for r in results if r["severity"] == "critical"),
            "high": sum(1 for r in results if r["severity"] == "high"),
            "medium": sum(1 for r in results if r["severity"] == "medium"),
            "low": sum(1 for r in results if r["severity"] == "low"),
            "none": sum(1 for r in results if r["severity"] == "none"),
        }

        # Count by drift type
        type_counts = {
            "data": sum(1 for r in results if r["drift_type"] == "data" and r["drift_detected"]),
            "model": sum(1 for r in results if r["drift_type"] == "model" and r["drift_detected"]),
            "concept": sum(
                1 for r in results if r["drift_type"] == "concept" and r["drift_detected"]
            ),
        }

        # Determine overall status
        if drift_score >= 0.8:
            status = "CRITICAL DRIFT"
        elif drift_score >= 0.6:
            status = "HIGH DRIFT"
        elif drift_score >= 0.4:
            status = "MODERATE DRIFT"
        elif drift_score >= 0.2:
            status = "LOW DRIFT"
        else:
            status = "NO SIGNIFICANT DRIFT"

        # Build summary
        summary_parts = [
            f"Drift Detection Summary: {status} (score: {drift_score:.3f})",
            f"Features checked: {len(results)}",
            f"Features with drift: {len(features_with_drift)}",
        ]

        # Add severity breakdown
        if any(count > 0 for count in severity_counts.values() if count > 0):
            severity_breakdown = []
            for sev in ["critical", "high", "medium", "low"]:
                if severity_counts[sev] > 0:
                    severity_breakdown.append(f"{sev}: {severity_counts[sev]}")
            if severity_breakdown:
                summary_parts.append(f"Severity breakdown: {', '.join(severity_breakdown)}")

        # Add drift type breakdown
        type_breakdown = []
        for dtype in ["data", "model", "concept"]:
            if type_counts[dtype] > 0:
                type_breakdown.append(f"{dtype}: {type_counts[dtype]}")
        if type_breakdown:
            summary_parts.append(f"Drift types detected: {', '.join(type_breakdown)}")

        return "\n".join(summary_parts)

    def _enhance_summary_with_interpretation(
        self, base_summary: str, interpretation: dict
    ) -> str:
        """Enhance the drift summary with interpretation insights.

        Args:
            base_summary: Original drift summary
            interpretation: Drift interpretation dictionary

        Returns:
            Enhanced summary with interpretation
        """
        enhanced_parts = [base_summary]

        # Add severity assessment
        if interpretation.get("severity_assessment"):
            enhanced_parts.append(f"\n{interpretation['severity_assessment']}")

        # Add primary drivers
        if interpretation.get("primary_drivers"):
            enhanced_parts.append("\nPrimary Drift Drivers:")
            for driver in interpretation["primary_drivers"][:3]:
                enhanced_parts.append(
                    f"  - {driver['feature']} ({driver['drift_type']}): {driver['change_description']}"
                )

        # Add root cause hypothesis
        if interpretation.get("root_cause_hypothesis"):
            enhanced_parts.append(f"\nRoot Cause Hypothesis: {interpretation['root_cause_hypothesis']}")

        # Add cascading effects
        if interpretation.get("cascading_effects"):
            enhanced_parts.append("\nPotential Cascading Effects:")
            for effect in interpretation["cascading_effects"]:
                enhanced_parts.append(f"  - {effect}")

        return "\n".join(enhanced_parts)

    def _generate_recommendations(
        self, results: list[DriftResult], drift_score: float
    ) -> list[str]:
        """Generate recommended actions based on drift results.

        Args:
            results: All drift detection results
            drift_score: Composite drift score

        Returns:
            List of recommended actions
        """
        recommendations: list[str] = []

        # High-level recommendation based on drift score
        if drift_score >= 0.8:
            recommendations.append("URGENT: Immediate retraining required due to critical drift")
        elif drift_score >= 0.6:
            recommendations.append("Schedule model retraining within 24-48 hours")
        elif drift_score >= 0.4:
            recommendations.append("Monitor model performance closely and plan retraining")
        elif drift_score >= 0.2:
            recommendations.append("Continue monitoring - no immediate action required")
        else:
            recommendations.append("No drift detected - model is stable")

        # Specific recommendations for critical features
        critical_features = [r["feature"] for r in results if r["severity"] == "critical"]
        if critical_features:
            recommendations.append(
                f"Investigate critical drift in: {', '.join(critical_features[:3])}"
            )

        # Data drift specific
        data_drift_count = sum(
            1 for r in results if r["drift_type"] == "data" and r["drift_detected"]
        )
        if data_drift_count > 0:
            recommendations.append(
                f"Review data pipeline - {data_drift_count} features showing distribution changes"
            )

        # Model drift specific
        model_drift_count = sum(
            1 for r in results if r["drift_type"] == "model" and r["drift_detected"]
        )
        if model_drift_count > 0:
            recommendations.append("Check model prediction quality on recent data")

        return recommendations

    def _interpret_drift(self, results: list[DriftResult], drift_score: float) -> dict:
        """Interpret drift results with root cause analysis.

        Provides deeper context about WHAT drifted, WHY it might have happened,
        and the potential business impact.

        Args:
            results: All drift detection results
            drift_score: Composite drift score

        Returns:
            Interpretation dictionary with root cause analysis
        """
        interpretation = {
            "severity_assessment": "",
            "primary_drivers": [],
            "root_cause_hypothesis": "",
            "comparison_to_baseline": "",
            "cascading_effects": [],
            "confidence_in_assessment": "medium",
        }

        if not results:
            interpretation["severity_assessment"] = "No drift data available for interpretation."
            return interpretation

        # Identify primary drift drivers (features with highest severity)
        drifted_results = [r for r in results if r["drift_detected"]]
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
        sorted_results = sorted(
            drifted_results,
            key=lambda r: severity_order.get(r["severity"], 0),
            reverse=True
        )

        for result in sorted_results[:3]:
            feature = result["feature"]
            drift_type = result["drift_type"]
            severity = result["severity"]
            psi = result.get("psi_score", 0)

            driver_info = {
                "feature": feature,
                "drift_type": drift_type,
                "severity": severity,
                "change_description": self._describe_change(result),
                "impact": self._assess_feature_impact(feature, severity, drift_type),
            }
            interpretation["primary_drivers"].append(driver_info)

        # Generate severity assessment with context
        if drift_score >= 0.8:
            interpretation["severity_assessment"] = (
                f"CRITICAL: Drift score of {drift_score:.2f} indicates severe distribution shift. "
                f"Model predictions are likely unreliable. Immediate action required."
            )
            interpretation["confidence_in_assessment"] = "high"
        elif drift_score >= 0.6:
            interpretation["severity_assessment"] = (
                f"HIGH: Drift score of {drift_score:.2f} indicates significant distribution shift. "
                f"Model accuracy may be meaningfully degraded. Recommend validation on recent data."
            )
        elif drift_score >= 0.4:
            interpretation["severity_assessment"] = (
                f"MODERATE: Drift score of {drift_score:.2f} indicates notable changes. "
                f"Monitor model performance closely over the next few days."
            )
        elif drift_score >= 0.2:
            interpretation["severity_assessment"] = (
                f"LOW: Drift score of {drift_score:.2f} indicates minor changes. "
                f"No immediate action needed but continue monitoring."
            )
        else:
            interpretation["severity_assessment"] = (
                f"MINIMAL: Drift score of {drift_score:.2f} indicates stable distributions. "
                f"Model operating within expected parameters."
            )

        # Generate root cause hypothesis based on drift patterns
        interpretation["root_cause_hypothesis"] = self._generate_root_cause_hypothesis(results)

        # Identify cascading effects
        interpretation["cascading_effects"] = self._identify_cascading_effects(results)

        return interpretation

    def _describe_change(self, result: DriftResult) -> str:
        """Describe the nature of the drift in human-readable terms.

        Args:
            result: Single drift result

        Returns:
            Human-readable description of the change
        """
        psi = result.get("psi_score", 0)
        ks_stat = result.get("ks_statistic", 0)
        drift_type = result["drift_type"]

        if drift_type == "data":
            if psi > 0.25:
                return f"Major distribution shift (PSI={psi:.3f}) - new data patterns significantly different from training"
            elif psi > 0.1:
                return f"Moderate distribution shift (PSI={psi:.3f}) - noticeable change in data patterns"
            else:
                return f"Minor variation (PSI={psi:.3f}) - within expected range"
        elif drift_type == "model":
            return "Model output distribution has shifted from baseline behavior"
        elif drift_type == "concept":
            return "Relationship between features and target has changed"
        else:
            return f"Detected {drift_type} drift"

    def _assess_feature_impact(self, feature: str, severity: str, drift_type: str) -> str:
        """Assess the business impact of drift in a specific feature.

        Args:
            feature: Feature name
            severity: Drift severity
            drift_type: Type of drift

        Returns:
            Impact assessment string
        """
        # Common high-impact features in pharma commercial analytics
        high_impact_features = {
            "hcp_visits", "prescription_volume", "market_share", "trx_count",
            "nrx_count", "conversion_rate", "patient_count", "days_on_therapy"
        }

        feature_lower = feature.lower()
        is_high_impact = any(hif in feature_lower for hif in high_impact_features)

        if severity in ("critical", "high"):
            if is_high_impact:
                return f"HIGH IMPACT: {feature} is a key business metric - drift here directly affects model reliability for critical decisions"
            else:
                return f"SIGNIFICANT: {feature} shows major drift - may affect downstream predictions"
        elif severity == "medium":
            return f"MODERATE: {feature} drift may contribute to prediction degradation over time"
        else:
            return f"LOW: {feature} drift is minor and unlikely to significantly impact predictions"

    def _generate_root_cause_hypothesis(self, results: list[DriftResult]) -> str:
        """Generate hypothesis about root cause of observed drift.

        Args:
            results: All drift detection results

        Returns:
            Root cause hypothesis string
        """
        # Analyze patterns in drift
        data_drift_count = sum(1 for r in results if r["drift_type"] == "data" and r["drift_detected"])
        model_drift_count = sum(1 for r in results if r["drift_type"] == "model" and r["drift_detected"])
        concept_drift_count = sum(1 for r in results if r["drift_type"] == "concept" and r["drift_detected"])

        hypotheses = []

        if data_drift_count > 3:
            hypotheses.append(
                "Multiple features showing data drift suggests upstream data pipeline changes "
                "or significant shift in population characteristics."
            )

        if model_drift_count > 0 and data_drift_count == 0:
            hypotheses.append(
                "Model drift without data drift may indicate model degradation "
                "or changes in prediction behavior over time."
            )

        if concept_drift_count > 0:
            hypotheses.append(
                "Concept drift indicates the relationship between features and target has evolved. "
                "Consider external market changes or competitive dynamics."
            )

        if data_drift_count > 0 and model_drift_count > 0:
            hypotheses.append(
                "Combined data and model drift suggests input distribution changes are "
                "propagating through to model outputs - retraining may restore accuracy."
            )

        if not hypotheses:
            hypotheses.append("Drift patterns do not indicate a clear root cause - recommend detailed investigation.")

        return " ".join(hypotheses)

    def _identify_cascading_effects(self, results: list[DriftResult]) -> list[str]:
        """Identify potential cascading effects of detected drift.

        Args:
            results: All drift detection results

        Returns:
            List of potential cascading effect descriptions
        """
        effects = []
        critical_count = sum(1 for r in results if r["severity"] == "critical")
        high_count = sum(1 for r in results if r["severity"] == "high")

        if critical_count > 0:
            effects.append(
                f"Critical drift in {critical_count} feature(s) may cause significant prediction errors"
            )

        if high_count > 2:
            effects.append(
                "Multiple high-severity drifts compound to increase overall model uncertainty"
            )

        # Check for correlated features drifting together
        drifted_features = [r["feature"] for r in results if r["drift_detected"]]
        if len(drifted_features) > 3:
            effects.append(
                f"Drift in {len(drifted_features)} features suggests systemic shift - "
                "all downstream analyses using these features may be affected"
            )

        return effects
