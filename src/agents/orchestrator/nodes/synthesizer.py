"""Synthesizer node for orchestrator agent.

Combine multi-agent results into coherent response.
Uses fast model for synthesis.
"""

import logging
import time
from typing import Any, Dict, List

from src.utils.llm_factory import get_fast_llm, get_llm_provider

from ..state import AgentResult, OrchestratorState

logger = logging.getLogger(__name__)


def _get_opik_connector():
    """Lazy import of OpikConnector to avoid circular imports."""
    try:
        from src.mlops.opik_connector import get_opik_connector

        return get_opik_connector()
    except ImportError:
        logger.debug("OpikConnector not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get OpikConnector: {e}")
        return None


class SynthesizerNode:
    """Combine multi-agent results into coherent response.

    Uses fast model (Haiku) for synthesis.
    """

    def __init__(self):
        """Initialize synthesizer with fast LLM for synthesis."""
        # Use fast LLM (Haiku or gpt-4o-mini based on LLM_PROVIDER env var)
        self.llm = get_fast_llm(max_tokens=1024, timeout=5)
        self._provider = get_llm_provider()

    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute response synthesis.

        Args:
            state: Current orchestrator state

        Returns:
            Updated state with synthesized response
        """
        start_time = time.time()

        results = state.get("agent_results", [])
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]

        # Single agent - return directly
        if len(successful_results) == 1:
            synthesized = self._extract_response(successful_results[0])
        elif len(successful_results) > 1:
            # Multi-agent synthesis
            synthesized = await self._synthesize_multiple(successful_results)
        else:
            # All failed
            synthesized = self._generate_error_response(failed_results)

        synthesis_time = int((time.time() - start_time) * 1000)
        total_latency = (
            state.get("classification_latency_ms", 0)
            + state.get("rag_latency_ms", 0)
            + state.get("routing_latency_ms", 0)
            + state.get("dispatch_latency_ms", 0)
            + synthesis_time
        )

        return {
            **state,
            "synthesized_response": synthesized["response"],
            "response_confidence": synthesized.get("confidence", 0.5),
            "recommendations": synthesized.get("recommendations", []),
            "follow_up_suggestions": synthesized.get("follow_ups", []),
            "synthesis_latency_ms": synthesis_time,
            "total_latency_ms": total_latency,
            "current_phase": "complete",
            "status": "completed" if successful_results else "failed",
        }

    def _extract_response(self, result: AgentResult) -> Dict[str, Any]:
        """Extract and enhance response from single agent result.

        Args:
            result: Agent result

        Returns:
            Enhanced response with strategic interpretation
        """
        agent_output = result.get("result", {})
        agent_name = result.get("agent_name", "unknown")

        # Get raw response
        raw_response = agent_output.get(
            "narrative", agent_output.get("response", str(agent_output))
        )

        # Extract key metrics for strategic interpretation
        interpretation = self._build_strategic_interpretation(agent_name, agent_output)

        # Enhance response with strategic context if we have metrics
        if interpretation["has_metrics"]:
            enhanced_response = f"""{raw_response}

**Strategic Implications:**
{interpretation['strategic_summary']}

**Recommended Actions:**
{chr(10).join(f'- {r}' for r in interpretation['actions'][:3])}"""
        else:
            enhanced_response = raw_response

        # Generate context-specific follow-ups based on agent type
        follow_ups = self._generate_context_follow_ups(agent_name, agent_output)

        return {
            "response": enhanced_response,
            "confidence": agent_output.get("confidence", 0.5),
            "recommendations": agent_output.get("recommendations", []) or interpretation["actions"],
            "follow_ups": follow_ups,
        }

    def _build_strategic_interpretation(self, agent_name: str, output: Dict[str, Any]) -> Dict[str, Any]:
        """Build strategic interpretation from agent output metrics.

        Args:
            agent_name: Name of the agent
            output: Agent output dictionary

        Returns:
            Strategic interpretation with business impact
        """
        interpretation = {
            "has_metrics": False,
            "strategic_summary": "",
            "actions": [],
        }

        # Causal Impact interpretation
        if agent_name == "causal_impact":
            ate = output.get("ate") or output.get("average_treatment_effect")
            p_value = output.get("p_value")
            if ate is not None:
                interpretation["has_metrics"] = True
                effect_direction = "positive" if ate > 0 else "negative"
                significance = "statistically significant" if p_value and p_value < 0.05 else "not statistically significant"
                interpretation["strategic_summary"] = (
                    f"The intervention shows a {effect_direction} effect of {abs(ate):.1%} on the outcome. "
                    f"This effect is {significance}. "
                    f"If scaled across the portfolio, this could translate to meaningful business impact."
                )
                if ate > 0 and (p_value is None or p_value < 0.05):
                    interpretation["actions"] = [
                        "Scale the successful intervention to additional segments",
                        "Allocate resources to replicate this approach in similar markets",
                        "Design A/B test to validate effect in different populations",
                    ]
                else:
                    interpretation["actions"] = [
                        "Investigate why the intervention underperformed",
                        "Consider alternative intervention strategies",
                        "Review targeting criteria for the intervention",
                    ]

        # Gap Analyzer interpretation
        elif agent_name == "gap_analyzer":
            gap_value = output.get("gap_value") or output.get("total_gap")
            roi_potential = output.get("roi_potential") or output.get("estimated_roi")
            if gap_value is not None:
                interpretation["has_metrics"] = True
                interpretation["strategic_summary"] = (
                    f"Identified performance gap of {gap_value:.1%} representing untapped opportunity. "
                    + (f"Estimated ROI potential: {roi_potential:.1%}. " if roi_potential else "")
                    + "Prioritize closing this gap to capture incremental value."
                )
                interpretation["actions"] = [
                    "Develop targeted campaign to address the identified gap",
                    "Reallocate resources from low-performing to high-opportunity segments",
                    "Establish monitoring metrics to track gap closure progress",
                ]

        # Heterogeneous Optimizer interpretation
        elif agent_name == "heterogeneous_optimizer":
            expected_lift = output.get("expected_total_lift", 0)
            if expected_lift:
                interpretation["has_metrics"] = True
                interpretation["strategic_summary"] = (
                    f"Optimal resource allocation would yield {expected_lift:.1f} units of incremental outcome. "
                    f"Segment-specific targeting can significantly outperform uniform approaches."
                )
                interpretation["actions"] = [
                    "Implement segment-specific allocation strategy",
                    "Focus resources on high-CATE segments identified",
                    "Reduce investment in low-response segments",
                ]

        # Prediction Synthesizer interpretation
        elif agent_name == "prediction_synthesizer":
            prediction = output.get("ensemble_prediction", {}).get("point_estimate")
            confidence = output.get("ensemble_prediction", {}).get("confidence_level")
            if prediction is not None:
                interpretation["has_metrics"] = True
                risk_level = "high" if prediction > 0.7 else "moderate" if prediction > 0.4 else "low"
                interpretation["strategic_summary"] = (
                    f"Predicted risk level: {risk_level} ({prediction:.0%}). "
                    + (f"Confidence: {confidence:.0%}. " if confidence else "")
                    + "Use this to prioritize intervention timing and intensity."
                )
                interpretation["actions"] = [
                    f"{'Immediate' if risk_level == 'high' else 'Planned'} intervention for high-risk entities",
                    "Monitor prediction accuracy over time for model calibration",
                    "Integrate prediction into operational workflows",
                ]

        return interpretation

    def _generate_context_follow_ups(self, agent_name: str, output: Dict[str, Any]) -> List[str]:
        """Generate context-specific follow-up suggestions.

        Args:
            agent_name: Name of the agent
            output: Agent output

        Returns:
            List of relevant follow-up suggestions
        """
        follow_ups = []

        # Agent-specific follow-ups
        if agent_name == "causal_impact":
            follow_ups = [
                "Analyze heterogeneous effects across patient segments",
                "Validate findings with a prospective experiment",
                "Explore time-varying treatment effects",
            ]
        elif agent_name == "gap_analyzer":
            follow_ups = [
                "Deep-dive into highest-potential segments",
                "Compare gap drivers across brands",
                "Develop intervention roadmap for gap closure",
            ]
        elif agent_name == "heterogeneous_optimizer":
            follow_ups = [
                "Simulate different budget scenarios",
                "Identify features driving segment differences",
                "Design targeting rules for field implementation",
            ]
        elif agent_name == "drift_monitor":
            follow_ups = [
                "Investigate root causes of detected drift",
                "Schedule model retraining if drift persists",
                "Review data pipeline for potential issues",
            ]
        elif agent_name == "prediction_synthesizer":
            follow_ups = [
                "Compare predictions across different models",
                "Analyze feature importance for predictions",
                "Set up alerting for high-risk predictions",
            ]
        else:
            # Default follow-ups
            follow_ups = [
                "Explore related analyses for deeper insights",
                "Validate findings with additional data sources",
            ]

        return follow_ups[:3]

    async def _synthesize_multiple(self, results: List[AgentResult]) -> Dict[str, Any]:
        """Synthesize multiple agent results with strategic interpretation.

        Args:
            results: List of successful agent results

        Returns:
            Synthesized response with business context
        """
        summaries = []
        all_recommendations = []
        confidences = []
        agent_names = []

        for result in results:
            agent_output = result.get("result", {})
            agent_name = result["agent_name"]
            agent_names.append(agent_name)
            narrative = agent_output.get("narrative", "")[:500]

            # Include key metrics in summary for LLM context
            metrics_str = self._extract_key_metrics_str(agent_name, agent_output)
            summary = f"**{agent_name}**: {narrative}"
            if metrics_str:
                summary += f" [Key metrics: {metrics_str}]"

            summaries.append(summary)
            all_recommendations.extend(agent_output.get("recommendations", []))
            confidences.append(agent_output.get("confidence", 0.5))

        synthesis_prompt = f"""Synthesize these analysis results into actionable strategic insights.

Results from {len(results)} analyses:
{chr(10).join(summaries)}

Provide a response that includes:

1. **KEY FINDING**: What is the single most important discovery? Include specific numbers.

2. **BUSINESS IMPACT**: What does this mean in practical terms?
   - Quantify the opportunity or risk where possible
   - Translate statistical findings into business language

3. **STRATEGIC IMPLICATIONS**: How should this affect decision-making?
   - What should change based on these insights?
   - What are the trade-offs to consider?

4. **RECOMMENDED ACTIONS** (prioritized):
   - Immediate (this week): Most urgent action
   - Short-term (this month): Follow-up actions
   - Long-term (this quarter): Strategic initiatives

5. **CONFIDENCE & CAVEATS**: What are the limitations?
   - Note any assumptions or data limitations
   - Indicate where additional analysis would add value

Be specific and quantitative. Avoid generic statements."""

        try:
            # Get OpikConnector for LLM call tracing
            opik = _get_opik_connector()

            if opik and opik.is_enabled:
                # Trace the LLM call with dynamic provider info
                model_name = "gpt-4o-mini" if self._provider == "openai" else "claude-haiku-4-20250414"
                async with opik.trace_llm_call(
                    model=model_name,
                    provider=self._provider,
                    prompt_template="strategic_multi_agent_synthesis",
                    input_data={"summaries": summaries, "prompt": synthesis_prompt},
                    metadata={"agent": "orchestrator", "operation": "strategic_synthesis"},
                ) as llm_span:
                    response = await self.llm.ainvoke(synthesis_prompt)
                    # Log tokens from response metadata
                    usage = response.response_metadata.get("usage", {})
                    llm_span.log_tokens(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    )
            else:
                # Fallback: no tracing
                response = await self.llm.ainvoke(synthesis_prompt)

            # Calculate weighted confidence
            avg_confidence = round(sum(confidences) / len(confidences) if confidences else 0.5, 2)

            # Generate context-specific follow-ups based on all agents involved
            follow_ups = self._generate_multi_agent_follow_ups(agent_names, results)

            return {
                "response": response.content,
                "confidence": avg_confidence,
                "recommendations": all_recommendations[:5] if all_recommendations else self._extract_recommendations_from_response(response.content),
                "follow_ups": follow_ups,
            }
        except Exception as e:
            logger.warning(f"LLM synthesis failed, using fallback: {e}")
            # Fallback: structured concatenation with interpretation
            fallback_response = self._build_fallback_synthesis(summaries, results)
            return {
                "response": fallback_response,
                "confidence": round(sum(confidences) / len(confidences) if confidences else 0.5, 2),
                "recommendations": all_recommendations[:5],
                "follow_ups": self._generate_multi_agent_follow_ups(agent_names, results),
            }

    def _extract_key_metrics_str(self, agent_name: str, output: Dict[str, Any]) -> str:
        """Extract key metrics as a string for LLM context.

        Args:
            agent_name: Agent name
            output: Agent output

        Returns:
            Metrics string or empty string
        """
        metrics = []
        if agent_name == "causal_impact":
            ate = output.get("ate") or output.get("average_treatment_effect")
            p_value = output.get("p_value")
            if ate is not None:
                metrics.append(f"ATE={ate:.3f}")
            if p_value is not None:
                metrics.append(f"p={p_value:.4f}")
        elif agent_name == "gap_analyzer":
            gap = output.get("gap_value") or output.get("total_gap")
            if gap is not None:
                metrics.append(f"gap={gap:.1%}")
        elif agent_name == "heterogeneous_optimizer":
            lift = output.get("expected_total_lift")
            if lift:
                metrics.append(f"expected_lift={lift:.1f}")
        elif agent_name == "prediction_synthesizer":
            pred = output.get("ensemble_prediction", {}).get("point_estimate")
            if pred is not None:
                metrics.append(f"prediction={pred:.2f}")

        return ", ".join(metrics)

    def _generate_multi_agent_follow_ups(self, agent_names: List[str], results: List[AgentResult]) -> List[str]:
        """Generate follow-ups based on multiple agent results.

        Args:
            agent_names: List of agent names involved
            results: Agent results

        Returns:
            Context-specific follow-up suggestions
        """
        follow_ups = []

        # Combination-specific follow-ups
        if "causal_impact" in agent_names and "heterogeneous_optimizer" in agent_names:
            follow_ups.append("Apply optimal allocation to maximize causal impact")
        if "gap_analyzer" in agent_names and "causal_impact" in agent_names:
            follow_ups.append("Design intervention to close identified gaps based on causal drivers")
        if "prediction_synthesizer" in agent_names:
            follow_ups.append("Validate predictions with prospective monitoring")
        if "drift_monitor" in agent_names:
            follow_ups.append("Address any drift issues before acting on other findings")

        # Add general strategic follow-up if we don't have enough specific ones
        if len(follow_ups) < 2:
            follow_ups.append("Develop implementation plan with success metrics")
            follow_ups.append("Schedule follow-up analysis to track outcomes")

        return follow_ups[:3]

    def _extract_recommendations_from_response(self, response: str) -> List[str]:
        """Extract recommendations from LLM response text.

        Args:
            response: LLM response text

        Returns:
            List of extracted recommendations
        """
        recommendations = []
        lines = response.split("\n")
        in_recommendations = False

        for line in lines:
            line = line.strip()
            if "RECOMMENDED" in line.upper() or "ACTION" in line.upper():
                in_recommendations = True
                continue
            if in_recommendations and line.startswith("-"):
                # Extract the recommendation text
                rec = line.lstrip("- ").strip()
                if rec and len(rec) > 10:
                    recommendations.append(rec)
            if in_recommendations and line.startswith("**") and "CONFIDENCE" in line.upper():
                break

        return recommendations[:5]

    def _build_fallback_synthesis(self, summaries: List[str], results: List[AgentResult]) -> str:
        """Build structured fallback synthesis when LLM fails.

        Args:
            summaries: Agent summary strings
            results: Agent results

        Returns:
            Fallback synthesis response
        """
        parts = ["**Analysis Summary**\n"]

        for summary in summaries:
            # Extract just the narrative part
            if ": " in summary:
                parts.append(summary.split(": ", 1)[1])

        parts.append("\n**Key Takeaways**")
        parts.append("- Multiple analyses converged on actionable insights")
        parts.append("- Review individual agent findings for detailed metrics")
        parts.append("- Consider follow-up analysis to validate conclusions")

        return "\n\n".join(parts)

    def _generate_error_response(self, failed_results: List[AgentResult]) -> Dict[str, Any]:
        """Generate error response when all agents fail.

        Args:
            failed_results: List of failed agent results

        Returns:
            Error response
        """
        errors = [f"- {r['agent_name']}: {r.get('error')}" for r in failed_results]

        return {
            "response": f"I was unable to complete the analysis due to the following errors:\n{chr(10).join(errors)}\n\nPlease try again or rephrase your question.",
            "confidence": 0.0,
            "recommendations": [],
            "follow_ups": ["Simplify your question", "Check system health"],
        }


# Export for use in graph
async def synthesize_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node function for response synthesis.

    Args:
        state: Current state

    Returns:
        Updated state
    """
    synthesizer = SynthesizerNode()
    return await synthesizer.execute(state)
