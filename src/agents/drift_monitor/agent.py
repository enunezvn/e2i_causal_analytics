"""Drift Monitor Agent.

This agent detects drift in data, model predictions, and concepts over time.

Tier: 3 (Monitoring)
Agent Type: Standard (Fast Path)
Performance Target: <10s for 50 features

Key Features:
- Data drift detection (PSI + KS test)
- Model drift detection (prediction distribution)
- Concept drift detection (placeholder)
- Alert generation with severity levels
- Composite drift score calculation

Integration:
- Memory: None (stateless statistical computation)
- Observability: Opik tracing with graceful degradation, MLflow tracking
- Data: SupabaseDataConnector (auto-detects based on env)

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md
Contract: .claude/contracts/tier3-contracts.md lines 349-562
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field, field_validator

from src.agents.drift_monitor.graph import drift_monitor_graph
from src.agents.drift_monitor.state import DriftMonitorState

if TYPE_CHECKING:
    from src.agents.drift_monitor.mlflow_tracker import DriftMonitorMLflowTracker

logger = logging.getLogger(__name__)


def _get_opik_connector():
    """Get OpikConnector (lazy import to avoid circular deps)."""
    try:
        from src.mlops.opik_connector import get_opik_connector

        return get_opik_connector()
    except Exception as e:
        logger.warning(f"Could not get Opik connector: {e}")
        return None


# ===== INPUT/OUTPUT MODELS =====


class DriftMonitorInput(BaseModel):
    """Input model for Drift Monitor Agent.

    Contract: .claude/contracts/tier3-contracts.md lines 355-393
    """

    # Required fields
    query: str = Field(..., description="User query or description of drift check")
    features_to_monitor: list[str] = Field(
        ..., min_length=1, description="Features to check for drift"
    )

    # Optional fields
    model_id: Optional[str] = Field(None, description="Model ID for model drift checks")
    time_window: str = Field("7d", description="Time window for comparison (e.g., '7d', '30d')")
    brand: Optional[str] = Field(None, description="Brand filter")

    # Configuration
    significance_level: float = Field(
        0.05, ge=0.01, le=0.10, description="Statistical significance level"
    )
    psi_threshold: float = Field(0.1, ge=0.0, le=1.0, description="PSI warning threshold")
    check_data_drift: bool = Field(True, description="Whether to check data drift")
    check_model_drift: bool = Field(True, description="Whether to check model drift")
    check_concept_drift: bool = Field(True, description="Whether to check concept drift")

    # Tier0 data passthrough for testing
    tier0_data: Optional[Any] = Field(
        None, description="Tier0 DataFrame for testing drift detection with real synthetic data"
    )

    @field_validator("time_window")
    @classmethod
    def validate_time_window(cls, v: str) -> str:
        """Validate time window format."""
        if not v.endswith("d"):
            raise ValueError("time_window must end with 'd' (e.g., '7d', '30d')")
        try:
            days = int(v[:-1])
            if days < 1 or days > 365:
                raise ValueError("time_window must be between 1d and 365d")
        except ValueError as e:
            raise ValueError(f"Invalid time_window format: {e}") from e
        return v


class DriftMonitorOutput(BaseModel):
    """Output model for Drift Monitor Agent.

    Contract: .claude/contracts/tier3-contracts.md lines 395-445
    """

    # Detection results
    data_drift_results: list[dict] = Field(..., description="Data drift detection results")
    model_drift_results: list[dict] = Field(..., description="Model drift detection results")
    concept_drift_results: list[dict] = Field(..., description="Concept drift detection results")

    # Aggregated outputs
    overall_drift_score: float = Field(..., ge=0.0, le=1.0, description="Composite drift score")
    features_with_drift: list[str] = Field(..., description="Features showing drift")
    alerts: list[dict] = Field(..., description="Generated alerts")

    # Summary
    drift_summary: str = Field(..., description="Human-readable summary")
    recommended_actions: list[str] = Field(..., description="Recommended actions")

    # Metadata (Contract-required fields)
    total_latency_ms: int = Field(..., description="Total detection latency")
    timestamp: str = Field(..., description="Completion timestamp")
    features_checked: int = Field(..., description="Number of features checked")
    baseline_timestamp: str = Field(..., description="Baseline period timestamp")
    warnings: list[str] = Field(default_factory=list, description="Warnings")

    # Contract-required fields (v4.3 fix: must be in output model for contract validation)
    errors: list[dict] = Field(default_factory=list, description="Error details from workflow")
    status: str = Field("completed", description="Agent execution status")


# ===== MAIN AGENT =====


class DriftMonitorAgent:
    """Drift Monitor Agent - Detects data, model, and concept drift.

    This agent monitors for distribution changes in:
    1. Feature distributions (data drift)
    2. Model predictions (model drift)
    3. Feature-target relationships (concept drift)

    Usage:
        agent = DriftMonitorAgent()
        result = agent.run(DriftMonitorInput(
            query="Check for drift in key features",
            features_to_monitor=["feature1", "feature2", "feature3"],
            model_id="model_v1",
            time_window="7d"
        ))

    Performance:
        - <10s for 50 features
        - <100ms for alert generation
        - No LLM usage (pure statistical computation)
    """

    # Agent metadata
    tier = 3
    tier_name = "monitoring"
    agent_name = "drift_monitor"
    agent_type = "standard"
    sla_seconds = 10  # <10s for 50 features
    tools = ["scipy", "numpy"]  # Statistical libraries for drift detection

    def __init__(self, enable_mlflow: bool = True):
        """Initialize drift monitor agent.

        Args:
            enable_mlflow: Whether to enable MLflow tracking (default: True)
        """
        self.graph = drift_monitor_graph
        self.enable_mlflow = enable_mlflow
        self._mlflow_tracker: Optional["DriftMonitorMLflowTracker"] = None

    def _get_mlflow_tracker(self) -> Optional["DriftMonitorMLflowTracker"]:
        """Get or create MLflow tracker instance (lazy initialization)."""
        if not self.enable_mlflow:
            return None

        if self._mlflow_tracker is None:
            try:
                from src.agents.drift_monitor.mlflow_tracker import DriftMonitorMLflowTracker

                self._mlflow_tracker = DriftMonitorMLflowTracker()
            except ImportError:
                logger.warning("MLflow tracker not available")
                return None

        return self._mlflow_tracker

    async def run(self, input_data: DriftMonitorInput) -> DriftMonitorOutput:
        """Execute drift detection workflow.

        Args:
            input_data: Validated input parameters

        Returns:
            DriftMonitorOutput with drift results and alerts

        Raises:
            ValueError: If input validation fails
            RuntimeError: If drift detection fails
        """
        start_time = datetime.now(timezone.utc)
        feature_count = len(input_data.features_to_monitor)

        logger.info(
            f"Starting drift detection: {feature_count} features, "
            f"time_window={input_data.time_window}, "
            f"model_id={input_data.model_id}"
        )

        # Create initial state from input
        initial_state = self._create_initial_state(input_data)

        # Get MLflow tracker
        mlflow_tracker = self._get_mlflow_tracker()

        # Execute LangGraph workflow with optional Opik tracing and MLflow tracking
        opik = _get_opik_connector()

        async def execute_workflow():
            """Execute the drift detection workflow."""
            if opik and opik.is_enabled:
                async with opik.trace_agent(
                    agent_name=self.agent_name,
                    operation="detect_drift",
                    metadata={
                        "tier": self.tier,
                        "feature_count": feature_count,
                        "time_window": input_data.time_window,
                        "model_id": input_data.model_id,
                        "check_data_drift": input_data.check_data_drift,
                        "check_model_drift": input_data.check_model_drift,
                        "check_concept_drift": input_data.check_concept_drift,
                    },
                    tags=[self.agent_name, "tier_3", "drift_detection"],
                    input_data={
                        "feature_count": feature_count,
                        "time_window": input_data.time_window,
                        "model_id": input_data.model_id,
                    },
                ) as span:
                    final_state = await self.graph.ainvoke(initial_state)
                    # Set output on span
                    if span and final_state.get("status") != "failed":
                        span.set_output(
                            {
                                "overall_drift_score": final_state.get("overall_drift_score", 0.0),
                                "features_with_drift": final_state.get("features_with_drift", []),
                                "alert_count": len(final_state.get("alerts", [])),
                                "total_latency_ms": final_state.get("total_latency_ms", 0),
                            }
                        )
                    return final_state
            else:
                return await self.graph.ainvoke(initial_state)

        try:
            # Execute with MLflow tracking if available
            if mlflow_tracker:
                async with mlflow_tracker.start_monitoring_run(
                    experiment_name=getattr(input_data, "experiment_name", "default")
                    if hasattr(input_data, "experiment_name")
                    else "default",
                    brand=input_data.brand,
                    model_id=input_data.model_id,
                    time_window=input_data.time_window,
                ):
                    final_state = await execute_workflow()

                    # Convert to output model
                    output = self._create_output(final_state)

                    # Log to MLflow
                    await mlflow_tracker.log_monitoring_result(output, final_state)

                    # Check for failures
                    if final_state["status"] == "failed":
                        error_messages = [e["error"] for e in final_state.get("errors", [])]
                        raise RuntimeError(f"Drift detection failed: {'; '.join(error_messages)}")
            else:
                # Execute without MLflow tracking
                final_state = await execute_workflow()

                # Convert to output model
                output = self._create_output(final_state)

                # Check for failures
                if final_state["status"] == "failed":
                    error_messages = [e["error"] for e in final_state.get("errors", [])]
                    raise RuntimeError(f"Drift detection failed: {'; '.join(error_messages)}")

        except RuntimeError:
            raise
        except Exception as e:
            logger.exception(f"Drift detection failed: {e}")
            raise RuntimeError(f"Drift detection workflow failed: {str(e)}") from e

        # Log execution time and SLA check
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            f"Drift detection complete: {feature_count} features, "
            f"drift_score={output.overall_drift_score:.2f}, "
            f"alerts={len(output.alerts)} in {duration:.2f}s"
        )

        if duration > self.sla_seconds:
            logger.warning(
                f"SLA violation: {duration:.2f}s > {self.sla_seconds}s for {feature_count} features"
            )

        return output

    def _create_initial_state(self, input_data: DriftMonitorInput) -> DriftMonitorState:
        """Create initial state from input.

        Args:
            input_data: Validated input

        Returns:
            Initial DriftMonitorState
        """
        state: DriftMonitorState = {
            # Input fields
            "query": input_data.query,
            "model_id": input_data.model_id or "",
            "features_to_monitor": input_data.features_to_monitor,
            "time_window": input_data.time_window,
            "brand": input_data.brand or "",
            # Tier0 data passthrough for testing
            "tier0_data": input_data.tier0_data,
            # Configuration
            "significance_level": input_data.significance_level,
            "psi_threshold": input_data.psi_threshold,
            "check_data_drift": input_data.check_data_drift,
            "check_model_drift": input_data.check_model_drift,
            "check_concept_drift": input_data.check_concept_drift,
            # Error handling
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        return state

    def _create_output(self, state: DriftMonitorState) -> DriftMonitorOutput:
        """Create output from final state.

        Args:
            state: Final state after graph execution

        Returns:
            DriftMonitorOutput
        """
        # Extract errors as list of dicts (convert ErrorDetails TypedDicts)
        raw_errors = state.get("errors", [])
        errors: list[dict] = [dict(e) if hasattr(e, "keys") else {"error": str(e)} for e in raw_errors]

        return DriftMonitorOutput(
            # Detection results (convert TypedDicts to plain dicts)
            data_drift_results=[dict(r) for r in state.get("data_drift_results", [])],
            model_drift_results=[dict(r) for r in state.get("model_drift_results", [])],
            concept_drift_results=[dict(r) for r in state.get("concept_drift_results", [])],
            # Aggregated outputs
            overall_drift_score=state.get("overall_drift_score", 0.0),
            features_with_drift=state.get("features_with_drift", []),
            alerts=[dict(a) for a in state.get("alerts", [])],
            # Summary
            drift_summary=state.get("drift_summary", "No summary available"),
            recommended_actions=state.get("recommended_actions", []),
            # Metadata (Contract-required fields)
            total_latency_ms=state.get("total_latency_ms", 0),
            timestamp=state.get("timestamp", ""),
            features_checked=state.get("features_checked", 0),
            baseline_timestamp=state.get("baseline_timestamp", ""),
            warnings=state.get("warnings", []),
            # Contract-required fields (v4.3 fix)
            errors=errors,
            status=state.get("status", "completed"),
        )
