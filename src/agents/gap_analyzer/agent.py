"""Gap Analyzer Agent - Tier 2 Standard Agent.

Identifies ROI opportunities by detecting performance gaps across segments,
calculating ROI for each gap, and prioritizing opportunities into quick wins
and strategic bets.

Performance Target: <20s total execution time

Architecture:
- Type: Standard Agent (Computational focus, minimal LLM usage)
- Workflow: Linear 4-node pipeline (gap_detector → roi_calculator → prioritizer → formatter)
- Parallelization: Segment analysis runs in parallel for throughput
- Economics: Pharma-specific ROI calculations
"""

import time
import uuid
from typing import Any, Dict, Optional

from .graph import create_gap_analyzer_graph
from .state import GapAnalyzerState


class GapAnalyzerAgent:
    """Gap Analyzer Agent for ROI opportunity detection.

    Tier 2 Standard Agent: Computational focus with minimal LLM usage.
    """

    def __init__(self):
        """Initialize Gap Analyzer Agent."""
        self.graph = create_gap_analyzer_graph()
        self.agent_name = "gap_analyzer"
        self.agent_tier = 2

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute gap analyzer workflow.

        Args:
            input_data: Input dictionary with query, metrics, segments, etc.

        Returns:
            GapAnalyzerOutput dictionary with prioritized opportunities

        Raises:
            ValueError: If required input fields are missing
        """
        # Validate input first (raises ValueError for invalid input)
        self._validate_input(input_data)

        start_time = time.time()

        try:
            # Initialize state
            state = self._initialize_state(input_data)

            # Execute workflow
            final_state = await self.graph.ainvoke(state)

            # Build output
            output = self._build_output(final_state)

            return output

        except Exception as e:
            # Build error output for workflow errors (not validation errors)
            return self._build_error_output(
                error=str(e),
                query_id=input_data.get("query_id", str(uuid.uuid4())),
                total_latency_ms=int((time.time() - start_time) * 1000),
            )

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate required input fields.

        Args:
            input_data: Input dictionary

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Required fields
        required_fields = ["query", "metrics", "segments", "brand"]

        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Validate types
        if not isinstance(input_data["metrics"], list) or not input_data["metrics"]:
            raise ValueError("metrics must be a non-empty list")

        if not isinstance(input_data["segments"], list) or not input_data["segments"]:
            raise ValueError("segments must be a non-empty list")

        # Validate optional fields
        if "gap_type" in input_data:
            valid_gap_types = ["vs_target", "vs_benchmark", "vs_potential", "temporal", "all"]
            if input_data["gap_type"] not in valid_gap_types:
                raise ValueError(
                    f"gap_type must be one of {valid_gap_types}, got {input_data['gap_type']}"
                )

        if "min_gap_threshold" in input_data:
            if not isinstance(input_data["min_gap_threshold"], (int, float)):
                raise ValueError("min_gap_threshold must be a number")

        if "max_opportunities" in input_data:
            if (
                not isinstance(input_data["max_opportunities"], int)
                or input_data["max_opportunities"] <= 0
            ):
                raise ValueError("max_opportunities must be a positive integer")

    def _initialize_state(self, input_data: Dict[str, Any]) -> GapAnalyzerState:
        """Initialize gap analyzer state.

        Args:
            input_data: Validated input dictionary

        Returns:
            Initialized GapAnalyzerState
        """
        state: GapAnalyzerState = {
            # Input
            "query": input_data["query"],
            "metrics": input_data["metrics"],
            "segments": input_data["segments"],
            "brand": input_data["brand"],
            "time_period": input_data.get("time_period", "current_quarter"),
            "filters": input_data.get("filters"),
            # Configuration
            "gap_type": input_data.get("gap_type", "vs_potential"),  # type: ignore
            "min_gap_threshold": input_data.get("min_gap_threshold", 5.0),
            "max_opportunities": input_data.get("max_opportunities", 10),
            # Outputs (initialized as None)
            "gaps_detected": None,
            "gaps_by_segment": None,
            "total_gap_value": None,
            "roi_estimates": None,
            "total_addressable_value": None,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            # Metadata
            "detection_latency_ms": 0,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 0,
            # Error handling
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        return state

    def _build_output(self, state: GapAnalyzerState) -> Dict[str, Any]:
        """Build output from final state.

        Args:
            state: Final gap analyzer state

        Returns:
            GapAnalyzerOutput dictionary
        """
        # Calculate overall confidence based on number of gaps and data quality
        confidence = self._calculate_confidence(state)

        # Determine if further analysis is needed
        requires_further_analysis = self._check_further_analysis(state)

        # Suggest next agent if needed
        suggested_next_agent = self._suggest_next_agent(state)

        output = {
            "prioritized_opportunities": state.get("prioritized_opportunities", []),
            "quick_wins": state.get("quick_wins", []),
            "strategic_bets": state.get("strategic_bets", []),
            "total_addressable_value": state.get("total_addressable_value", 0.0),
            "total_gap_value": state.get("total_gap_value", 0.0),
            "segments_analyzed": state.get("segments_analyzed", 0),
            "executive_summary": state.get("executive_summary", ""),
            "key_insights": state.get("key_insights", []),
            "detection_latency_ms": state.get("detection_latency_ms", 0),
            "roi_latency_ms": state.get("roi_latency_ms", 0),
            "total_latency_ms": state.get("total_latency_ms", 0),
            "confidence": confidence,
            "warnings": state.get("warnings", []),
            "requires_further_analysis": requires_further_analysis,
            "suggested_next_agent": suggested_next_agent,
        }

        return output

    def _calculate_confidence(self, state: GapAnalyzerState) -> float:
        """Calculate overall confidence in results.

        Factors:
        - Number of gaps detected
        - Data quality (no errors)
        - Segment coverage

        Args:
            state: Final state

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.7  # Base confidence

        # Number of gaps factor
        gaps_detected = state.get("gaps_detected", [])
        if len(gaps_detected) > 10:
            confidence += 0.1
        elif len(gaps_detected) < 3:
            confidence -= 0.1

        # Error factor
        errors = state.get("errors", [])
        if errors:
            confidence -= 0.2

        # Segment coverage factor
        segments_analyzed = state.get("segments_analyzed", 0)
        expected_segments = len(state.get("segments", []))
        if segments_analyzed >= expected_segments:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _check_further_analysis(self, state: GapAnalyzerState) -> bool:
        """Check if further analysis is recommended.

        Criteria:
        - High-impact opportunities with uncertainty
        - Large gaps requiring causal validation
        - Complex segment patterns

        Args:
            state: Final state

        Returns:
            True if further analysis recommended
        """
        prioritized_opportunities = state.get("prioritized_opportunities", [])

        if not prioritized_opportunities:
            return False

        # Check for high-ROI opportunities with low confidence
        for opp in prioritized_opportunities[:3]:  # Top 3
            roi_estimate = opp["roi_estimate"]
            if roi_estimate["expected_roi"] > 3.0 and roi_estimate["confidence"] < 0.7:
                return True

        # Check for very large gaps (may need causal validation)
        gaps_detected = state.get("gaps_detected", [])
        for gap in gaps_detected:
            if abs(gap["gap_percentage"]) > 50:
                return True

        return False

    def _suggest_next_agent(self, state: GapAnalyzerState) -> Optional[str]:
        """Suggest next agent for follow-up analysis.

        Args:
            state: Final state

        Returns:
            Agent name or None
        """
        if not self._check_further_analysis(state):
            return None

        prioritized_opportunities = state.get("prioritized_opportunities", [])

        if not prioritized_opportunities:
            return None

        # High-ROI opportunities → Causal validation
        top_opp = prioritized_opportunities[0]
        if top_opp["roi_estimate"]["expected_roi"] > 3.0:
            return "causal_impact"

        # Segment-specific patterns → Heterogeneous optimization
        gaps_by_segment = state.get("gaps_by_segment", {})
        if len(gaps_by_segment) > 5:
            return "heterogeneous_optimizer"

        return None

    def _build_error_output(
        self, error: str, query_id: str, total_latency_ms: int
    ) -> Dict[str, Any]:
        """Build error output.

        Args:
            error: Error message
            query_id: Query identifier
            total_latency_ms: Latency

        Returns:
            Error output dictionary
        """
        return {
            "prioritized_opportunities": [],
            "quick_wins": [],
            "strategic_bets": [],
            "total_addressable_value": 0.0,
            "total_gap_value": 0.0,
            "segments_analyzed": 0,
            "executive_summary": f"Error: {error}",
            "key_insights": [],
            "detection_latency_ms": 0,
            "roi_latency_ms": 0,
            "total_latency_ms": total_latency_ms,
            "confidence": 0.0,
            "warnings": [f"Gap analysis failed: {error}"],
            "requires_further_analysis": False,
            "suggested_next_agent": None,
        }

    # Helper methods for orchestrator

    async def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent for gap analysis.

        Args:
            query: Natural language query

        Returns:
            Intent classification with confidence
        """
        # Simple keyword-based classification (upgrade to LLM if needed)
        query_lower = query.lower()

        gap_keywords = ["gap", "opportunity", "roi", "improvement", "target", "benchmark"]
        metric_keywords = ["trx", "nrx", "market share", "conversion", "engagement"]

        confidence = 0.0

        if any(kw in query_lower for kw in gap_keywords):
            confidence += 0.5

        if any(kw in query_lower for kw in metric_keywords):
            confidence += 0.3

        return {
            "primary_intent": "gap_detection",
            "confidence": min(1.0, confidence),
            "requires_gap_analysis": confidence > 0.5,
        }

    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified interface for orchestrator.

        Args:
            input_data: Input dictionary

        Returns:
            Simplified output for orchestrator consumption
        """
        full_output = await self.run(input_data)

        # Simplify for orchestrator
        return {
            "summary": full_output["executive_summary"],
            "opportunities": full_output["prioritized_opportunities"],
            "quick_wins": full_output["quick_wins"],
            "strategic_bets": full_output["strategic_bets"],
            "confidence": full_output["confidence"],
            "suggested_next_agent": full_output["suggested_next_agent"],
        }
