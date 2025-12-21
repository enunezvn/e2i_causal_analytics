"""Interpretation Node - Natural language interpretation of causal results.

Deep Reasoning node that converts technical results into user-friendly narratives.
"""

import time
from typing import Dict

from src.agents.causal_impact.state import (
    CausalImpactState,
    NaturalLanguageInterpretation,
)


class InterpretationNode:
    """Generates natural language interpretation of causal analysis.

    Performance target: <30s
    Type: Deep Reasoning (LLM-heavy, uses Sonnet/Opus)
    """

    def __init__(self):
        """Initialize interpretation node."""
        pass

    async def execute(self, state: CausalImpactState) -> Dict:
        """Generate natural language interpretation.

        Args:
            state: Current workflow state with all analysis results

        Returns:
            Updated state with interpretation
        """
        start_time = time.time()

        try:
            # Get depth and expertise settings
            depth = state.get("interpretation_depth", "standard")
            user_context = state.get("user_context", {})
            expertise = user_context.get("expertise", "analyst")

            # Skip interpretation if depth is "none"
            if depth == "none":
                interpretation: NaturalLanguageInterpretation = {
                    "narrative": "Interpretation skipped per user request.",
                    "key_findings": [],
                    "effect_magnitude": "N/A",
                    "causal_confidence": "N/A",
                    "assumptions_made": [],
                    "limitations": [],
                    "recommendations": [],
                    "depth_level": "none",
                    "user_expertise_adjusted": False,
                }

                latency_ms = (time.time() - start_time) * 1000

                return {
                    **state,
                    "interpretation": interpretation,
                    "interpretation_latency_ms": latency_ms,
                    "current_phase": "completed",
                    "status": "completed",
                }

            # Generate interpretation based on depth and expertise
            if depth == "minimal":
                interpretation = await self._generate_minimal_interpretation(state, expertise)
            elif depth == "standard":
                interpretation = await self._generate_standard_interpretation(state, expertise)
            elif depth == "deep":
                interpretation = await self._generate_deep_interpretation(state, expertise)
            else:
                raise ValueError(f"Unknown interpretation depth: {depth}")

            latency_ms = (time.time() - start_time) * 1000

            return {
                **state,
                "interpretation": interpretation,
                "interpretation_latency_ms": latency_ms,
                "current_phase": "completed",
                "status": "completed",
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                **state,
                "interpretation_error": str(e),
                "interpretation_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Interpretation failed: {e}",
            }

    async def _generate_minimal_interpretation(
        self, state: CausalImpactState, expertise: str
    ) -> NaturalLanguageInterpretation:
        """Generate minimal interpretation (1-2 sentences).

        Args:
            state: Workflow state with results
            expertise: User expertise level

        Returns:
            Minimal interpretation
        """
        estimation_result = state.get("estimation_result", {})
        ate = estimation_result.get("ate", 0.0)
        effect_size = estimation_result.get("effect_size", "unknown")
        significance = estimation_result.get("statistical_significance", False)

        # Simple narrative
        if significance:
            narrative = (
                f"The analysis found a statistically significant {effect_size} causal effect "
                f"of {ate:.2f}. This effect is likely real and actionable."
            )
        else:
            narrative = (
                f"The analysis found a {effect_size} effect of {ate:.2f}, but it is not "
                f"statistically significant. Interpret with caution."
            )

        interpretation: NaturalLanguageInterpretation = {
            "narrative": narrative,
            "key_findings": [f"Effect size: {effect_size}", f"ATE: {ate:.2f}"],
            "effect_magnitude": effect_size,
            "causal_confidence": "medium" if significance else "low",
            "assumptions_made": ["Standard causal assumptions"],
            "limitations": ["Minimal interpretation - see standard or deep for details"],
            "recommendations": ["Consider running detailed analysis for actionable insights"],
            "depth_level": "minimal",
            "user_expertise_adjusted": True,
        }

        return interpretation

    async def _generate_standard_interpretation(
        self, state: CausalImpactState, expertise: str
    ) -> NaturalLanguageInterpretation:
        """Generate standard interpretation (3-5 paragraphs).

        Args:
            state: Workflow state with results
            expertise: User expertise level

        Returns:
            Standard interpretation
        """
        # Extract results
        estimation_result = state.get("estimation_result", {})
        refutation_results = state.get("refutation_results", {})
        sensitivity_analysis = state.get("sensitivity_analysis", {})

        ate = estimation_result.get("ate", 0.0)
        ate_ci_lower = estimation_result.get("ate_ci_lower", 0.0)
        ate_ci_upper = estimation_result.get("ate_ci_upper", 0.0)
        effect_size = estimation_result.get("effect_size", "unknown")
        significance = estimation_result.get("statistical_significance", False)
        method = estimation_result.get("method", "unknown")

        tests_passed = refutation_results.get("tests_passed", 0)
        total_tests = refutation_results.get("total_tests", 0)
        overall_robust = refutation_results.get("overall_robust", False)

        e_value = sensitivity_analysis.get("e_value", 1.0)
        robust_to_confounding = sensitivity_analysis.get("robust_to_confounding", False)

        # Construct narrative
        narrative_parts = []

        # Effect summary
        if expertise == "executive":
            narrative_parts.append(
                f"Our analysis reveals that the treatment has a {effect_size} impact, "
                f"with an estimated effect of {ate:.2f}. This translates to a measurable "
                f"business outcome with {'strong' if significance else 'moderate'} statistical support."
            )
        else:
            narrative_parts.append(
                f"The causal analysis using {method} estimates an average treatment effect (ATE) "
                f"of {ate:.2f} (95% CI: [{ate_ci_lower:.2f}, {ate_ci_upper:.2f}]). "
                f"This effect is classified as {effect_size} and is "
                f"{'statistically significant' if significance else 'not statistically significant'}."
            )

        # Robustness
        if overall_robust:
            narrative_parts.append(
                f"The effect passed {tests_passed} out of {total_tests} robustness tests, "
                f"indicating the finding is likely genuine and not spurious. "
                f"The E-value of {e_value:.2f} suggests "
                f"{'strong' if e_value > 3 else 'moderate' if e_value > 2 else 'weak'} "
                f"robustness to unmeasured confounding."
            )
        else:
            narrative_parts.append(
                f"However, the effect failed some robustness tests ({tests_passed}/{total_tests} passed), "
                f"suggesting caution in interpretation. The E-value of {e_value:.2f} indicates "
                f"{'limited' if e_value < 2 else 'moderate'} robustness to unmeasured confounding."
            )

        # Recommendations
        if significance and overall_robust:
            narrative_parts.append(
                "Based on these findings, we recommend proceeding with interventions targeting "
                "this treatment variable, with careful monitoring of outcomes to validate "
                "the predicted effects in practice."
            )
        else:
            narrative_parts.append(
                "Given the uncertainty in these results, we recommend further investigation, "
                "potentially with additional data or alternative analytical approaches, before "
                "making major strategic decisions."
            )

        narrative = " ".join(narrative_parts)

        # Key findings
        key_findings = [
            f"Estimated causal effect: {ate:.2f} ({effect_size})",
            f"Statistical significance: {'Yes' if significance else 'No'}",
            f"Robustness tests: {tests_passed}/{total_tests} passed",
            f"E-value: {e_value:.2f}",
        ]

        # Assumptions
        assumptions_made = [
            "No unmeasured confounding (given observed covariates)",
            "Positivity: All subgroups have non-zero treatment probability",
            "SUTVA: No interference between units",
            "Correct causal graph specification",
        ]

        # Limitations
        limitations = [
            "Analysis based on observational data, not randomized experiment",
            "E-value indicates potential for unmeasured confounding",
            "Assumes causal graph accurately represents true relationships",
        ]

        # Recommendations
        recommendations = []
        if significance and overall_robust:
            recommendations.extend(
                [
                    "Implement targeted interventions based on identified causal effect",
                    "Monitor outcomes closely to validate predictions",
                    "Consider heterogeneous effects across segments for optimization",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Collect additional data to improve statistical power",
                    "Conduct sensitivity analyses with alternative model specifications",
                    "Consider randomized experiment to validate findings",
                ]
            )

        # Confidence
        if significance and overall_robust and robust_to_confounding:
            confidence = "high"
        elif significance and overall_robust:
            confidence = "medium"
        else:
            confidence = "low"

        interpretation: NaturalLanguageInterpretation = {
            "narrative": narrative,
            "key_findings": key_findings,
            "effect_magnitude": effect_size,
            "causal_confidence": confidence,
            "assumptions_made": assumptions_made,
            "limitations": limitations,
            "recommendations": recommendations,
            "depth_level": "standard",
            "user_expertise_adjusted": True,
        }

        return interpretation

    async def _generate_deep_interpretation(
        self, state: CausalImpactState, expertise: str
    ) -> NaturalLanguageInterpretation:
        """Generate deep interpretation (5-8 paragraphs with technical details).

        This would use Claude Opus in production for extended reasoning.

        Args:
            state: Workflow state with results
            expertise: User expertise level

        Returns:
            Deep interpretation
        """
        # For deep interpretation, include all technical details
        # In production, this would make an LLM call to Opus

        # Get standard interpretation as base
        standard = await self._generate_standard_interpretation(state, expertise)

        # Enhance with additional technical details
        estimation_result = state.get("estimation_result", {})
        refutation_results = state.get("refutation_results", {})
        causal_graph = state.get("causal_graph", {})

        # Enhanced narrative with graph details
        enhanced_narrative = (
            f"{standard['narrative']}\n\n"
            f"CAUSAL GRAPH STRUCTURE: "
            f"The constructed causal DAG contains {len(causal_graph.get('nodes', []))} nodes "
            f"and {len(causal_graph.get('edges', []))} edges, with "
            f"{len(causal_graph.get('adjustment_sets', [[]]))} valid adjustment sets identified. "
        )

        # Add refutation test details
        individual_tests = refutation_results.get("individual_tests", [])
        if individual_tests:
            enhanced_narrative += "\n\nREFUTATION TESTS: "
            for test in individual_tests:
                test_name = test.get("test_name", "unknown")
                passed = test.get("passed", False)
                details = test.get("details", "")
                enhanced_narrative += (
                    f"\n- {test_name}: {'PASSED' if passed else 'FAILED'} - {details}"
                )

        # Add methodological details
        enhanced_narrative += (
            f"\n\nMETHODOLOGY: "
            f"The analysis employed {estimation_result.get('method', 'unknown')} "
            f"with adjustment for {len(estimation_result.get('covariates_adjusted', []))} covariates. "
        )

        if estimation_result.get("heterogeneity_detected", False):
            enhanced_narrative += (
                "Significant treatment effect heterogeneity was detected across segments, "
                "suggesting the average treatment effect varies meaningfully by subgroup."
            )

        interpretation: NaturalLanguageInterpretation = {
            "narrative": enhanced_narrative,
            "key_findings": standard["key_findings"]
            + [
                f"Causal graph: {len(causal_graph.get('nodes', []))} nodes, {len(causal_graph.get('edges', []))} edges",
                f"Adjustment sets: {len(causal_graph.get('adjustment_sets', [[]]))} valid sets",
            ],
            "effect_magnitude": standard["effect_magnitude"],
            "causal_confidence": standard["causal_confidence"],
            "assumptions_made": standard["assumptions_made"]
            + [
                "Linear effect assumption (for DML methods)",
                "Conditional independence given adjustment set",
            ],
            "limitations": standard["limitations"]
            + [
                "Finite sample bias possible with small sample sizes",
                "Model misspecification could affect estimates",
                "Temporal dynamics not captured in cross-sectional analysis",
            ],
            "recommendations": standard["recommendations"]
            + [
                "Investigate heterogeneous effects for optimization",
                "Consider longitudinal analysis to capture dynamics",
                "Validate findings with external data sources if available",
            ],
            "depth_level": "deep",
            "user_expertise_adjusted": True,
        }

        return interpretation


# Standalone function for LangGraph integration
async def interpret_results(state: CausalImpactState) -> Dict:
    """Generate interpretation (standalone function).

    Args:
        state: Current workflow state

    Returns:
        Updated state with interpretation
    """
    node = InterpretationNode()
    return await node.execute(state)
