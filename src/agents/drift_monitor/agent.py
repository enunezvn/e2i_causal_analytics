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

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md
Contract: .claude/contracts/tier3-contracts.md lines 349-562
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.agents.drift_monitor.graph import drift_monitor_graph
from src.agents.drift_monitor.state import DriftMonitorState

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

    # Metadata
    detection_latency_ms: int = Field(..., description="Total detection latency")
    features_checked: int = Field(..., description="Number of features checked")
    baseline_timestamp: str = Field(..., description="Baseline period timestamp")
    current_timestamp: str = Field(..., description="Current period timestamp")
    warnings: list[str] = Field(default_factory=list, description="Warnings")


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

    def __init__(self):
        """Initialize drift monitor agent."""
        self.graph = drift_monitor_graph

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
        # Create initial state from input
        initial_state = self._create_initial_state(input_data)

        # Execute graph (async)
        final_state = await self.graph.ainvoke(initial_state)

        # Convert to output model
        output = self._create_output(final_state)

        # Check for failures
        if final_state["status"] == "failed":
            error_messages = [e["error"] for e in final_state.get("errors", [])]
            raise RuntimeError(f"Drift detection failed: {'; '.join(error_messages)}")

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
            "model_id": input_data.model_id,
            "features_to_monitor": input_data.features_to_monitor,
            "time_window": input_data.time_window,
            "brand": input_data.brand,
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
        return DriftMonitorOutput(
            # Detection results
            data_drift_results=state.get("data_drift_results", []),
            model_drift_results=state.get("model_drift_results", []),
            concept_drift_results=state.get("concept_drift_results", []),
            # Aggregated outputs
            overall_drift_score=state.get("overall_drift_score", 0.0),
            features_with_drift=state.get("features_with_drift", []),
            alerts=state.get("alerts", []),
            # Summary
            drift_summary=state.get("drift_summary", "No summary available"),
            recommended_actions=state.get("recommended_actions", []),
            # Metadata
            detection_latency_ms=state.get("detection_latency_ms", 0),
            features_checked=state.get("features_checked", 0),
            baseline_timestamp=state.get("baseline_timestamp", ""),
            current_timestamp=state.get("current_timestamp", ""),
            warnings=state.get("warnings", []),
        )
