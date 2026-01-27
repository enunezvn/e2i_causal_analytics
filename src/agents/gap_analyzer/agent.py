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

import logging
import time
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.agents.base import SkillsMixin
from .graph import create_gap_analyzer_graph
from .state import GapAnalyzerState

if TYPE_CHECKING:
    from .mlflow_tracker import GapAnalyzerMLflowTracker
    from .opik_tracer import GapAnalyzerOpikTracer

logger = logging.getLogger(__name__)


class GapAnalyzerAgent(SkillsMixin):
    """Gap Analyzer Agent for ROI opportunity detection.

    Tier 2 Standard Agent: Computational focus with minimal LLM usage.

    Skills Integration:
        - gap-analysis/roi-estimation.md: ROI calculation procedures
        - gap-analysis/opportunity-sizing.md: Opportunity sizing framework
        - pharma-commercial/brand-analytics.md: Brand-specific context
    """

    def __init__(self, enable_mlflow: bool = True, enable_opik: bool = True):
        """Initialize Gap Analyzer Agent.

        Args:
            enable_mlflow: Whether to enable MLflow tracking (default: True)
            enable_opik: Whether to enable Opik distributed tracing (default: True)
        """
        self.graph = create_gap_analyzer_graph()
        self.agent_name = "gap_analyzer"
        self.agent_tier = 2
        self.enable_mlflow = enable_mlflow
        self.enable_opik = enable_opik

        # MLflow tracker (lazy initialization)
        self._mlflow_tracker: Optional["GapAnalyzerMLflowTracker"] = None
        # Opik tracer (lazy initialization)
        self._opik_tracer: Optional["GapAnalyzerOpikTracer"] = None

    def _get_mlflow_tracker(self) -> Optional["GapAnalyzerMLflowTracker"]:
        """Get or create MLflow tracker instance (lazy initialization)."""
        if not self.enable_mlflow:
            return None

        if self._mlflow_tracker is None:
            try:
                from .mlflow_tracker import GapAnalyzerMLflowTracker

                self._mlflow_tracker = GapAnalyzerMLflowTracker()
            except ImportError:
                logger.warning("MLflow tracker not available")
                return None

        return self._mlflow_tracker

    def _get_opik_tracer(self) -> Optional["GapAnalyzerOpikTracer"]:
        """Get or create Opik tracer instance (lazy initialization)."""
        if not self.enable_opik:
            return None

        if self._opik_tracer is None:
            try:
                from .opik_tracer import GapAnalyzerOpikTracer

                self._opik_tracer = GapAnalyzerOpikTracer()
            except ImportError:
                logger.warning("Opik tracer not available")
                return None

        return self._opik_tracer

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute gap analyzer workflow.

        Args:
            input_data: Input dictionary with query, metrics, segments, etc.

        Returns:
            GapAnalyzerOutput dictionary with prioritized opportunities

        Raises:
            ValueError: If required input fields are missing
        """
        # Clear loaded skills from previous invocation
        self.clear_loaded_skills()

        # Load relevant domain skills for gap analysis
        await self._load_analysis_skills(input_data)

        # Validate input first (raises ValueError for invalid input)
        self._validate_input(input_data)

        start_time = time.time()

        # Get trackers
        tracker = self._get_mlflow_tracker()
        opik_tracer = self._get_opik_tracer()

        try:
            # Initialize state
            state = self._initialize_state(input_data)

            # Execute with Opik tracing if available
            if opik_tracer:
                async with opik_tracer.trace_analysis(
                    query=input_data["query"],
                    brand=input_data["brand"],
                    metrics=input_data.get("metrics"),
                    segments=input_data.get("segments"),
                    gap_type=input_data.get("gap_type", "vs_potential"),
                    query_id=input_data.get("query_id"),
                ) as trace_ctx:
                    # Run with MLflow tracking if available
                    if tracker:
                        async with tracker.start_analysis_run(
                            experiment_name=input_data.get("experiment_name", "default"),
                            brand=input_data.get("brand"),
                            region=input_data.get("region"),
                            gap_type=input_data.get("gap_type"),
                            query_id=input_data.get("query_id"),
                        ):
                            # Execute workflow
                            final_state = await self.graph.ainvoke(state)

                            # Build output
                            output = self._build_output(final_state)

                            # Log to MLflow
                            await tracker.log_analysis_result(output, final_state)
                    else:
                        # Execute workflow without MLflow
                        final_state = await self.graph.ainvoke(state)
                        output = self._build_output(final_state)

                    # Log analysis results to Opik
                    trace_ctx.log_analysis_complete(
                        status="success",
                        success=True,
                        total_duration_ms=output.get("total_latency_ms", 0),
                        gaps_detected=len(final_state.get("gaps_detected", [])),
                        opportunities_count=len(output.get("prioritized_opportunities", [])),
                        quick_wins_count=len(output.get("quick_wins", [])),
                        strategic_bets_count=len(output.get("strategic_bets", [])),
                        total_addressable_value=output.get("total_addressable_value", 0.0),
                        confidence=output.get("confidence", 0.0),
                        suggested_next_agent=output.get("suggested_next_agent"),
                    )

                    return output
            else:
                # Run without Opik tracing
                if tracker:
                    async with tracker.start_analysis_run(
                        experiment_name=input_data.get("experiment_name", "default"),
                        brand=input_data.get("brand"),
                        region=input_data.get("region"),
                        gap_type=input_data.get("gap_type"),
                        query_id=input_data.get("query_id"),
                    ):
                        # Execute workflow
                        final_state = await self.graph.ainvoke(state)

                        # Build output
                        output = self._build_output(final_state)

                        # Log to MLflow
                        await tracker.log_analysis_result(output, final_state)

                        return output
                else:
                    # Execute workflow without any tracking
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

    async def _load_analysis_skills(self, input_data: Dict[str, Any]) -> None:
        """Load relevant skills for gap analysis.

        Loads domain-specific procedural knowledge based on the analysis task.
        Skills are optional - analysis proceeds without them if unavailable.

        Args:
            input_data: The gap analysis input parameters.
        """
        try:
            # Load core gap analysis skills
            await self.load_skill("gap-analysis/roi-estimation.md")
            await self.load_skill("gap-analysis/opportunity-sizing.md")

            # Load brand-specific context if brand is specified
            brand = input_data.get("brand")
            if brand:
                await self.load_skill("pharma-commercial/brand-analytics.md")

            loaded_names = self.get_loaded_skill_names()
            if loaded_names:
                logger.info(f"Loaded {len(loaded_names)} analysis skills: {loaded_names}")
        except Exception as e:
            # Skills are optional - log warning and proceed without
            logger.warning(f"Failed to load analysis skills (proceeding without): {e}")

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

    # Orchestrator handoff protocol

    def get_handoff(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate handoff for orchestrator agent coordination.

        Follows standardized handoff protocol for multi-agent orchestration.
        Provides key findings, output availability, and next agent suggestions.

        Args:
            output: GapAnalyzerOutput dictionary from run() method

        Returns:
            Handoff dictionary with:
            - agent: Agent name
            - analysis_type: Type of analysis performed
            - key_findings: Summary metrics and counts
            - outputs: Availability of each output type
            - requires_further_analysis: Whether follow-up is needed
            - suggested_next_agent: Recommended agent for follow-up
            - suggestions: Actionable suggestions based on findings
        """
        opportunities = output.get("prioritized_opportunities", [])
        quick_wins = output.get("quick_wins", [])
        strategic_bets = output.get("strategic_bets", [])

        # Count gaps by severity (if available in opportunities)
        critical_count = len([o for o in opportunities if o.get("priority", 0) == 1])
        high_count = len([o for o in opportunities if o.get("priority", 0) == 2])

        # Build suggestions based on findings
        suggestions = []
        if quick_wins:
            suggestions.append(
                f"Implement {len(quick_wins)} quick wins with combined ROI of "
                f"${sum(qw.get('roi_estimate', {}).get('expected_value', 0) for qw in quick_wins):,.0f}"
            )
        if strategic_bets:
            suggestions.append(
                f"Evaluate {len(strategic_bets)} strategic bets for long-term value"
            )
        if output.get("suggested_next_agent") == "causal_impact":
            suggestions.append("Validate high-ROI opportunities with causal analysis")
        if output.get("suggested_next_agent") == "heterogeneous_optimizer":
            suggestions.append("Optimize allocation across segments with heterogeneous effects")

        handoff = {
            "agent": self.agent_name,
            "analysis_type": "gap_analysis",
            "key_findings": {
                "total_opportunities": len(opportunities),
                "quick_wins_count": len(quick_wins),
                "strategic_bets_count": len(strategic_bets),
                "critical_gaps": critical_count,
                "high_priority_gaps": high_count,
                "total_addressable_value": output.get("total_addressable_value", 0),
                "total_gap_value": output.get("total_gap_value", 0),
                "segments_analyzed": output.get("segments_analyzed", 0),
                "confidence": output.get("confidence", 0),
            },
            "outputs": {
                "opportunities": "available" if opportunities else "unavailable",
                "quick_wins": len(quick_wins),
                "strategic_bets": len(strategic_bets),
                "executive_summary": "available" if output.get("executive_summary") else "unavailable",
                "key_insights": len(output.get("key_insights", [])),
            },
            "requires_further_analysis": output.get("requires_further_analysis", False),
            "suggested_next_agent": output.get("suggested_next_agent"),
            "suggestions": suggestions,
        }

        return handoff

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
