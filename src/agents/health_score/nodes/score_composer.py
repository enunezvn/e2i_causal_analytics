"""
E2I Health Score Agent - Score Composer Node
Version: 4.2
Purpose: Compose overall health score from component scores
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from ..metrics import DEFAULT_GRADES, DEFAULT_WEIGHTS, GradeThresholds, ScoreWeights
from ..state import HealthScoreState

logger = logging.getLogger(__name__)


class ScoreComposerNode:
    """
    Compose overall health score from component scores.
    Pure computation - no LLM calls.
    """

    def __init__(
        self,
        weights: Optional[ScoreWeights] = None,
        grades: Optional[GradeThresholds] = None,
    ):
        """
        Initialize score composer.

        Args:
            weights: Custom weights for score components
            grades: Custom grade thresholds
        """
        self.weights = weights or DEFAULT_WEIGHTS
        self.grades = grades or DEFAULT_GRADES

    async def execute(self, state: HealthScoreState) -> HealthScoreState:
        """Compose overall health score."""
        start_time = time.time()

        try:
            # Collect scores (default to 1.0 if not present)
            scores = {
                "component": state.get("component_health_score", 1.0),
                "model": state.get("model_health_score", 1.0),
                "pipeline": state.get("pipeline_health_score", 1.0),
                "agent": state.get("agent_health_score", 1.0),
            }

            # Calculate weighted average
            weights_dict = self.weights.to_dict()
            overall_score = sum(scores[dim] * weight for dim, weight in weights_dict.items())

            # Convert to 0-100 scale
            overall_score_100 = overall_score * 100

            # Determine grade
            grade = self.grades.get_grade(overall_score)

            # Identify issues
            critical_issues, warnings = self._identify_issues(state)

            # Generate diagnostic reasoning
            diagnosis = self._generate_diagnosis(state, scores)

            # Generate enhanced summary with diagnosis
            summary = self._generate_summary(overall_score_100, grade, critical_issues)

            # Add diagnosis insights to summary if there are issues
            if diagnosis["root_causes"]:
                summary += f"\n\nDiagnostic Analysis:"
                summary += f"\n- Health Trend: {diagnosis['health_trend'].upper()}"
                if diagnosis["priority_fixes"]:
                    top_fix = diagnosis["priority_fixes"][0]
                    summary += f"\n- Top Priority: {top_fix['action']} ({top_fix['component']})"

            check_time = state.get("total_latency_ms", 0) + int((time.time() - start_time) * 1000)

            logger.info(
                f"Score composition complete: score={overall_score_100:.1f}, "
                f"grade={grade}, issues={len(critical_issues)}, warnings={len(warnings)}"
            )

            return {
                **state,
                "overall_health_score": overall_score_100,
                "health_grade": grade,
                "critical_issues": critical_issues,
                "warnings": warnings,
                "health_summary": summary,
                "health_diagnosis": diagnosis,
                "total_latency_ms": check_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Score composition failed: {e}")
            return {
                **state,
                "errors": [{"node": "score_composer", "error": str(e)}],
                "overall_health_score": 0.0,
                "health_grade": "F",
                "critical_issues": [f"Score composition failed: {e}"],
                "warnings": [],
                "health_summary": "Unable to compute health score due to an error.",
                "total_latency_ms": state.get("total_latency_ms", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
            }

    def _identify_issues(self, state: HealthScoreState) -> Tuple[List[str], List[str]]:
        """Identify critical issues and warnings."""
        critical = []
        warnings = []

        # Check components
        for comp in state.get("component_statuses") or []:
            if comp["status"] == "unhealthy":
                critical.append(f"Component '{comp['component_name']}' is unhealthy")
            elif comp["status"] == "degraded":
                warnings.append(f"Component '{comp['component_name']}' is degraded")
            elif comp["status"] == "unknown":
                warnings.append(f"Component '{comp['component_name']}' status is unknown")

        # Check models
        for model in state.get("model_metrics") or []:
            if model["status"] == "unhealthy":
                critical.append(f"Model '{model['model_id']}' is unhealthy")
            elif model["status"] == "degraded":
                warnings.append(f"Model '{model['model_id']}' is degraded")

        # Check pipelines
        for pipeline in state.get("pipeline_statuses") or []:
            if pipeline["status"] == "failed":
                critical.append(f"Pipeline '{pipeline['pipeline_name']}' has failed")
            elif pipeline["status"] == "stale":
                warnings.append(f"Pipeline '{pipeline['pipeline_name']}' data is stale")

        # Check agents
        for agent in state.get("agent_statuses") or []:
            if not agent["available"]:
                critical.append(f"Agent '{agent['agent_name']}' is unavailable")
            elif agent["success_rate"] < 0.9:
                warnings.append(
                    f"Agent '{agent['agent_name']}' has low success rate "
                    f"({agent['success_rate']:.1%})"
                )

        # Check accumulated errors
        for error in state.get("errors") or []:
            node = error.get("node", "unknown")
            msg = error.get("error", "Unknown error")
            warnings.append(f"Error in {node}: {msg}")

        return critical, warnings

    def _generate_summary(self, score: float, grade: str, issues: List[str]) -> str:
        """Generate health summary."""
        status_map = {
            "A": "excellent",
            "B": "good",
            "C": "fair",
            "D": "poor",
            "F": "critical",
        }
        status = status_map.get(grade, "unknown")

        summary = f"System health is {status} (Grade: {grade}, Score: {score:.1f}/100)."

        if issues:
            summary += f" {len(issues)} critical issue(s) detected."
        else:
            summary += " All systems operational."

        return summary

    def _generate_diagnosis(self, state: HealthScoreState, scores: dict) -> dict:
        """Generate diagnostic reasoning for health issues.

        Provides deeper analysis of root causes, cascading effects,
        and prioritized remediation steps.

        Args:
            state: Current health score state
            scores: Component scores dictionary

        Returns:
            Diagnosis dictionary with root causes and remediation
        """
        diagnosis = {
            "root_causes": [],
            "cascading_effects": [],
            "priority_fixes": [],
            "health_trend": "stable",
            "confidence": "high",
        }

        # Analyze each dimension for issues
        component_issues = self._analyze_component_health(state, scores)
        model_issues = self._analyze_model_health(state, scores)
        pipeline_issues = self._analyze_pipeline_health(state, scores)
        agent_issues = self._analyze_agent_health(state, scores)

        # Collect all root causes
        all_issues = component_issues + model_issues + pipeline_issues + agent_issues

        # Sort by impact (critical first)
        all_issues.sort(key=lambda x: x.get("impact_score", 0), reverse=True)
        diagnosis["root_causes"] = all_issues[:5]

        # Identify cascading effects
        diagnosis["cascading_effects"] = self._find_cascading_effects(all_issues)

        # Generate prioritized fixes
        diagnosis["priority_fixes"] = self._prioritize_fixes(all_issues)

        # Determine health trend
        if len(all_issues) > 3:
            diagnosis["health_trend"] = "degrading"
        elif len(all_issues) == 0:
            diagnosis["health_trend"] = "healthy"

        return diagnosis

    def _analyze_component_health(self, state: HealthScoreState, scores: dict) -> List[dict]:
        """Analyze component health issues."""
        issues = []
        component_score = scores.get("component", 1.0)

        if component_score < 0.7:
            for comp in state.get("component_statuses") or []:
                if comp["status"] in ("unhealthy", "degraded"):
                    issue = {
                        "dimension": "component",
                        "component": comp["component_name"],
                        "status": comp["status"],
                        "root_cause": self._infer_component_root_cause(comp),
                        "metrics": {
                            "latency_ms": comp.get("latency_ms"),
                            "error_message": comp.get("error_message"),
                        },
                        "impact_score": 1.0 if comp["status"] == "unhealthy" else 0.5,
                    }
                    issues.append(issue)

        return issues

    def _analyze_model_health(self, state: HealthScoreState, scores: dict) -> List[dict]:
        """Analyze model health issues."""
        issues = []
        model_score = scores.get("model", 1.0)

        if model_score < 0.8:
            for model in state.get("model_metrics") or []:
                if model["status"] in ("unhealthy", "degraded"):
                    accuracy = model.get("accuracy")
                    error_rate = model.get("error_rate", 0)

                    root_cause = "Unknown model issue"
                    if accuracy and accuracy < 0.7:
                        root_cause = f"Model accuracy ({accuracy:.1%}) below threshold"
                    elif error_rate > 0.1:
                        root_cause = f"High error rate ({error_rate:.1%})"

                    issue = {
                        "dimension": "model",
                        "component": model["model_id"],
                        "status": model["status"],
                        "root_cause": root_cause,
                        "metrics": {
                            "accuracy": accuracy,
                            "precision": model.get("precision"),
                            "recall": model.get("recall"),
                            "error_rate": error_rate,
                        },
                        "impact_score": 0.8 if model["status"] == "unhealthy" else 0.4,
                    }
                    issues.append(issue)

        return issues

    def _analyze_pipeline_health(self, state: HealthScoreState, scores: dict) -> List[dict]:
        """Analyze pipeline health issues."""
        issues = []
        pipeline_score = scores.get("pipeline", 1.0)

        if pipeline_score < 0.9:
            for pipeline in state.get("pipeline_statuses") or []:
                if pipeline["status"] in ("failed", "stale"):
                    freshness = pipeline.get("freshness_hours", 0)

                    if pipeline["status"] == "failed":
                        root_cause = "Pipeline execution failed"
                    elif freshness > 24:
                        root_cause = f"Data stale by {freshness:.1f} hours (>24h threshold)"
                    else:
                        root_cause = f"Data freshness degraded ({freshness:.1f} hours)"

                    issue = {
                        "dimension": "pipeline",
                        "component": pipeline["pipeline_name"],
                        "status": pipeline["status"],
                        "root_cause": root_cause,
                        "metrics": {
                            "freshness_hours": freshness,
                            "rows_processed": pipeline.get("rows_processed"),
                            "last_success": pipeline.get("last_success"),
                        },
                        "impact_score": 0.9 if pipeline["status"] == "failed" else 0.3,
                    }
                    issues.append(issue)

        return issues

    def _analyze_agent_health(self, state: HealthScoreState, scores: dict) -> List[dict]:
        """Analyze agent health issues."""
        issues = []
        agent_score = scores.get("agent", 1.0)

        if agent_score < 0.9:
            for agent in state.get("agent_statuses") or []:
                if not agent["available"] or agent["success_rate"] < 0.9:
                    if not agent["available"]:
                        root_cause = "Agent unavailable - may be down or unreachable"
                    else:
                        root_cause = f"Low success rate ({agent['success_rate']:.1%})"

                    issue = {
                        "dimension": "agent",
                        "component": agent["agent_name"],
                        "status": "unavailable" if not agent["available"] else "degraded",
                        "root_cause": root_cause,
                        "metrics": {
                            "tier": agent.get("tier"),
                            "success_rate": agent.get("success_rate"),
                            "avg_latency_ms": agent.get("avg_latency_ms"),
                        },
                        "impact_score": 0.7 if not agent["available"] else 0.3,
                    }
                    issues.append(issue)

        return issues

    def _infer_component_root_cause(self, comp: dict) -> str:
        """Infer root cause for component issues."""
        component_name = comp.get("component_name", "").lower()
        error_msg = comp.get("error_message", "")
        latency = comp.get("latency_ms")

        if "database" in component_name or "db" in component_name:
            if latency and latency > 1000:
                return "Database connection slow - possible connection pool exhaustion"
            elif error_msg:
                return f"Database error: {error_msg[:100]}"
            return "Database connectivity issue"

        elif "cache" in component_name or "redis" in component_name:
            if error_msg and "connection" in error_msg.lower():
                return "Cache server connection refused - may need restart"
            return "Cache service degraded"

        elif "api" in component_name:
            if latency and latency > 5000:
                return "API response time critical - check downstream dependencies"
            return "API service degraded"

        elif "queue" in component_name or "message" in component_name:
            return "Message queue backlog or connectivity issue"

        return f"Component degraded: {error_msg[:100] if error_msg else 'Unknown cause'}"

    def _find_cascading_effects(self, issues: List[dict]) -> List[str]:
        """Identify cascading effects from root causes."""
        effects = []

        # Check for database issues affecting other components
        db_issues = [i for i in issues if "database" in i.get("component", "").lower()]
        if db_issues:
            effects.append(
                "Database issues may cause failures in agents, pipelines, and API endpoints"
            )

        # Check for pipeline issues affecting model freshness
        pipeline_issues = [i for i in issues if i.get("dimension") == "pipeline"]
        if pipeline_issues:
            effects.append(
                "Stale pipelines mean models are operating on outdated data - predictions may be unreliable"
            )

        # Check for model issues affecting agent reliability
        model_issues = [i for i in issues if i.get("dimension") == "model"]
        if model_issues:
            effects.append(
                "Degraded model accuracy affects all downstream agents relying on predictions"
            )

        # Multiple component failures
        if len(issues) > 3:
            effects.append(
                f"Multiple simultaneous issues ({len(issues)}) suggest potential infrastructure problem"
            )

        return effects[:3]

    def _prioritize_fixes(self, issues: List[dict]) -> List[dict]:
        """Generate prioritized list of fixes."""
        fixes = []

        # Define fix templates
        fix_templates = {
            "component": {
                "database": "Check database connection pool and consider restart",
                "cache": "Verify Redis service status and memory usage",
                "api": "Review API logs and check downstream service health",
                "default": "Investigate component logs and restart if necessary",
            },
            "model": "Evaluate model on recent data and consider retraining",
            "pipeline": "Check pipeline logs, verify data sources, and re-run",
            "agent": "Check agent logs, verify dependencies, and restart service",
        }

        for i, issue in enumerate(issues[:5]):
            dimension = issue.get("dimension", "unknown")
            component = issue.get("component", "").lower()

            if dimension == "component":
                if "database" in component:
                    fix_action = fix_templates["component"]["database"]
                elif "cache" in component or "redis" in component:
                    fix_action = fix_templates["component"]["cache"]
                elif "api" in component:
                    fix_action = fix_templates["component"]["api"]
                else:
                    fix_action = fix_templates["component"]["default"]
            else:
                fix_action = fix_templates.get(dimension, "Investigate and remediate")

            fix = {
                "priority": i + 1,
                "component": issue.get("component"),
                "issue": issue.get("root_cause"),
                "action": fix_action,
                "estimated_impact": "high" if issue.get("impact_score", 0) > 0.7 else "medium",
            }
            fixes.append(fix)

        return fixes
