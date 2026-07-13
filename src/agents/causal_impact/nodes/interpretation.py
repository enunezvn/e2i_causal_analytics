"""Interpretation Node - Natural language interpretation of causal results.

Deep Reasoning node that converts technical results into user-friendly narratives.

Additionally handles:
- DSPy training signal collection and routing
"""

import asyncio
import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict

from src.agents.causal_impact.state import (
    CausalImpactState,
    NaturalLanguageInterpretation,
)

if TYPE_CHECKING:
    from src.services.clinical_context import ClinicalContextService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_clinical_context_service() -> "ClinicalContextService":
    """Lazily build the shared clinical-context service (one instance; avoids
    import-time client construction in the agent module). The service's fragment
    cache is module-global, so live API results are reused across runs."""
    from src.services.clinical_context import ClinicalContextService

    return ClinicalContextService()


def _clinical_context_sentence(ctx: Dict[str, Any]) -> str:
    """Concise on-/off-label + market sentence from a get_context payload; returns
    '' when there is nothing useful to add (the narrative is only extended when the
    clinical context is real)."""
    inds = ctx.get("approved_indications") or {}
    ind_list = inds.get("indications") or []
    lou = inds.get("limitations_of_use")
    comp = ctx.get("competitor_landscape") or {}
    count = comp.get("count") or 0
    bits = []
    if ind_list:
        bits.append(f"FDA-approved use includes {ind_list[0]}")
    if lou:
        bits.append(f"label limitation of use — {lou}")
    if count:
        bits.append(f"{count} therapeutic competitor(s) occupy this indication")
    if not bits:
        return ""
    return "Clinical/market context: " + "; ".join(bits) + "."


def _humanize_var(name: str) -> str:
    """Turn a raw column like ``acceptance_status`` into ``acceptance status`` for
    prose. Empty/None -> a neutral placeholder so the sentence still reads."""
    if not name:
        return "the treatment"
    return str(name).replace("_", " ").strip() or "the treatment"


def _actionable_recommendations(
    *,
    treatment_var: str,
    outcome_var: str,
    brand: str | None,
    ate: float,
    effect_size: str,
    significant: bool,
    robust: bool,
) -> list[str]:
    """Concrete, GROUNDED next actions for the "Recommended actions" surface.

    The previous static bullets ("Implement targeted interventions…", "Monitor
    outcomes closely…") named no lever, no audience, and no metric — the user
    asked HOW. These reference the actual treatment -> outcome, the estimated
    direction/magnitude, and the brand, so each bullet says WHAT to pull, WHERE
    to focus, and WHICH metric to watch. Grounded only in values already
    computed upstream — never invents a number.
    """
    t = _humanize_var(treatment_var)
    o = _humanize_var(outcome_var)
    who = brand or "this brand"
    # Effect of the treatment on the outcome. A positive ATE means MORE of the
    # treatment raises the outcome, so the lever is to increase exposure; a
    # negative ATE means more of the treatment lowers it, so the lever is to
    # re-target it. Framed on the sign, never asserting desirability.
    raises = ate >= 0

    # NOTE: rendered as PLAIN TEXT in the UI (<li>{r}</li>) — no markdown, so no
    # ** ** emphasis (it would show literal asterisks). Keep the copy clean prose.
    who_poss = f"{who}'s"  # e.g. "Kisqali's" / "this brand's"
    if significant and robust:
        lever = (
            f"Scale up '{t}' for {who} where it is currently low"
            if raises
            else f"Re-target '{t}' for {who} — more of it is associated with a lower {o}"
        )
        return [
            (
                f"{lever}: the estimate is {ate:+.3f} ({effect_size} effect) on '{o}' and "
                f"cleared the robustness gate. Rank {who_poss} HCPs/patients by their current "
                f"{t} and target the segment where changing it moves {o} most."
            ),
            (
                f"Validate before a full rollout: apply {t} to a randomized subset of {who_poss} "
                f"HCPs/patients and track {o} against a matched holdout for ~1–2 follow-up "
                f"cycles to confirm the {ate:+.3f} effect replicates in-market before committing "
                f"budget."
            ),
            (
                f"Spend where it pays off: use the heterogeneous-effects (CATE) view to concentrate "
                f"{t} on {who_poss} sub-cohorts with the largest predicted effect on {o}, rather "
                f"than applying it uniformly."
            ),
        ]

    # Not significant and/or not robust — the honest action is to WITHHOLD the
    # decision and strengthen the evidence, named to this pair/brand.
    return [
        (
            f"Do not act on {t} → {o} for {who} yet: the effect is not yet distinguishable "
            f"from zero and/or did not clear the robustness gate — treating it as real risks "
            f"spending against noise."
        ),
        (
            f"Increase statistical power: enlarge the {who} cohort or extend the observation "
            f"window, then re-estimate {t} → {o} before making a call."
        ),
        (
            f"Pressure-test the design: re-run with an alternative adjustment set and a placebo "
            f"outcome; a genuine {t} → {o} effect should survive, a spurious one will not."
        ),
    ]


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

            # Collect and route DSPy training signal (non-blocking)
            await self._collect_dspy_signal(state, interpretation, latency_ms)

            # MED (fail-open): a sensitivity-node FAILURE (which set
            # state["sensitivity_error"] and status="failed" with a defaulted
            # E-value) must NOT be masked as "completed" here. Surface it: keep
            # the failed status and flag the result as needing review.
            sensitivity_failed = bool(state.get("sensitivity_error"))
            result: Dict[str, Any] = {
                **state,
                "interpretation": interpretation,
                "interpretation_latency_ms": latency_ms,
                "current_phase": "failed" if sensitivity_failed else "completed",
                "status": "failed" if sensitivity_failed else "completed",
            }
            if sensitivity_failed:
                result["needs_review"] = True
            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            # Contract: accumulate errors using operator.add
            errors = [{"phase": "interpretation", "message": str(e)}]
            return {
                **state,
                "interpretation_error": str(e),
                "interpretation_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Interpretation failed: {e}",
                "errors": errors,  # Contract error accumulator
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
        # H2: a REVIEW-band gate is borderline (needs_review), NOT robust. The
        # legacy ``overall_robust`` is True for both PROCEED and REVIEW ("not
        # blocked"), so a REVIEW result would otherwise get "likely genuine /
        # proceed" narrative language. Downgrade it here so REVIEW takes the
        # caution branch and is never described as robust to the user.
        if refutation_results.get("needs_review") or refutation_results.get("gate_decision") == (
            "review"
        ):
            overall_robust = False

        # Humanized treatment/outcome for grounded prose (the "too generic" fix).
        t_name = _humanize_var(str(state.get("treatment_var") or ""))
        o_name = _humanize_var(str(state.get("outcome_var") or ""))

        e_value = sensitivity_analysis.get("e_value", 1.0)
        robust_to_confounding = sensitivity_analysis.get("robust_to_confounding", False)
        # M-fo3: the sensitivity node writes NO sensitivity_analysis when it raises
        # (it sets state["sensitivity_error"] + status="failed"), so e_value above
        # defaults to 1.00. The narrative must NOT present that defaulted value as a
        # real (weak) robustness result.
        sensitivity_failed = bool(state.get("sensitivity_error")) or not sensitivity_analysis

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
            robustness_line = (
                f"The effect passed {tests_passed} out of {total_tests} robustness tests, "
                "indicating the finding is likely genuine and not spurious. "
            )
        else:
            robustness_line = (
                f"However, the effect failed some robustness tests ({tests_passed}/{total_tests} passed), "
                "suggesting caution in interpretation. "
            )

        if sensitivity_failed:
            # M-fo3: do NOT cite the defaulted E-value of 1.00 as a real result.
            robustness_line += (
                "The sensitivity analysis (E-value) could not be completed, so "
                "robustness to unmeasured confounding is UNVERIFIED; do not rely on "
                "any reported E-value."
            )
        elif overall_robust:
            strength = "strong" if e_value > 3 else "moderate" if e_value > 2 else "weak"
            robustness_line += (
                f"The E-value of {e_value:.2f} suggests {strength} robustness to "
                "unmeasured confounding."
            )
        else:
            strength = "limited" if e_value < 2 else "moderate"
            robustness_line += (
                f"The E-value of {e_value:.2f} indicates {strength} robustness to "
                "unmeasured confounding."
            )

        narrative_parts.append(robustness_line)

        # Clinical / market context woven into the narrative (fail-open: any failure
        # leaves the narrative unchanged and NEVER blocks the estimate). Brand+outcome
        # scoped; surfaces FDA-label use, any limitation of use, and competitor count.
        brand = state.get("brand")
        outcome_var = state.get("outcome_var")
        if brand and outcome_var:
            try:
                ctx = await asyncio.to_thread(
                    _get_clinical_context_service().get_context, brand, outcome_var
                )
                sentence = _clinical_context_sentence(ctx)
                if sentence:
                    narrative_parts.append(sentence)
            except Exception as exc:  # noqa: BLE001 — best-effort; never blocks the estimate
                logger.debug("interpretation: clinical context unavailable: %s", exc)

        # Recommendations (grounded in the actual treatment -> outcome).
        if significance and overall_robust:
            _lever = "increasing" if ate >= 0 else "re-targeting"
            narrative_parts.append(
                f"Because this effect is significant and cleared the robustness gate, the practical "
                f"next step is to act on {t_name} — focus on {_lever} it for the segment where it is "
                f"currently lowest — and confirm the lift in {o_name} against a matched holdout "
                f"before scaling."
            )
        else:
            narrative_parts.append(
                f"Because the {t_name} → {o_name} effect is not yet significant and/or robust, hold "
                f"off on acting on it: enlarge the cohort or extend the window to gain power, and "
                f"re-run with an alternative adjustment set and a placebo outcome before making a "
                f"strategic call."
            )

        narrative = " ".join(narrative_parts)

        # Key findings
        key_findings = [
            f"Estimated causal effect: {ate:.2f} ({effect_size})",
            f"Statistical significance: {'Yes' if significance else 'No'}",
            f"Robustness tests: {tests_passed}/{total_tests} passed",
            # M-fo3: do not surface a defaulted E-value when the analysis failed.
            (
                "E-value: unavailable (sensitivity analysis failed)"
                if sensitivity_failed
                else f"E-value: {e_value:.2f}"
            ),
        ]

        # Assumptions / limitations are DESIGN-aware: a genuinely randomized
        # treatment (state.randomized_design, declared by the API layer from
        # the dataset spec) makes "no unmeasured confounding" a design
        # GUARANTEE, not an assumption — and claiming the analysis is
        # observational would be plainly false for the RCT holdout.
        if state.get("randomized_design"):
            assumptions_made = [
                "Randomized treatment assignment (no unmeasured confounding by design)",
                "Positivity: All subgroups have non-zero treatment probability",
                "SUTVA: No interference between units",
                "Correct causal graph specification",
            ]
            limitations = [
                "Randomized experiment: E-value reported for information only "
                "(unmeasured confounding of assignment excluded by design)",
                "Assumes randomization was executed as designed (no assignment leakage)",
            ]
        else:
            assumptions_made = [
                "No unmeasured confounding (given observed covariates)",
                "Positivity: All subgroups have non-zero treatment probability",
                "SUTVA: No interference between units",
                "Correct causal graph specification",
            ]
            limitations = [
                "Analysis based on observational data, not randomized experiment",
                "E-value indicates potential for unmeasured confounding",
                "Assumes causal graph accurately represents true relationships",
            ]

        # Recommendations — GROUNDED in the actual treatment -> outcome, the
        # estimated direction/magnitude, and the brand (the user asked HOW). See
        # _actionable_recommendations: each bullet names the lever, the audience,
        # and the metric to watch rather than generic "implement interventions".
        recommendations = _actionable_recommendations(
            treatment_var=str(state.get("treatment_var") or ""),
            outcome_var=str(state.get("outcome_var") or ""),
            brand=state.get("brand"),
            ate=ate,
            effect_size=effect_size,
            significant=bool(significance),
            robust=bool(overall_robust),
        )

        # Confidence
        if significance and overall_robust and robust_to_confounding:
            confidence = "high"
        elif significance and overall_robust:
            confidence = "medium"
        else:
            confidence = "low"

        # Executive summary (2-3 sentences). The graph (agent-analyze) path never
        # runs the agent's _build_output, which is the only OTHER place an
        # executive summary is generated — so produce it HERE so both paths carry
        # a non-null headline. (When _build_output does run, it overwrites this.)
        # This is the BOLD headline rendered ABOVE the narrative, so significance
        # alone must NOT read as an endorsement: a significant-but-not-robust
        # result (failed/REVIEW gate) is framed cautiously, never "passed N/M".
        _direction = "increases" if ate > 0 else "decreases"
        if significance and overall_robust:
            executive_summary = (
                f"The treatment {_direction} the outcome by an estimated {ate:.3f} "
                f"({effect_size} effect), a statistically significant result that passed "
                f"{tests_passed}/{total_tests} robustness checks. Overall confidence: {confidence}."
            )
        elif significance:
            executive_summary = (
                f"The treatment {_direction} the outcome by an estimated {ate:.3f} "
                f"({effect_size} effect). The effect is statistically significant but did NOT clear "
                f"the robustness gate ({tests_passed}/{total_tests} checks passed) — treat as "
                f"preliminary. Overall confidence: {confidence}."
            )
        else:
            executive_summary = (
                f"The estimated effect is {ate:.3f} ({effect_size}), but it is not "
                f"statistically significant ({tests_passed}/{total_tests} robustness checks passed). "
                f"Treat this as preliminary; overall confidence: {confidence}."
            )

        interpretation: NaturalLanguageInterpretation = {
            "narrative": narrative,
            "executive_summary": executive_summary,
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
        # Contract: individual_tests is Dict with test names as keys
        individual_tests = refutation_results.get("individual_tests", {})
        if individual_tests:
            enhanced_narrative += "\n\nREFUTATION TESTS: "
            for test_key, test in individual_tests.items():
                test_name = test.get("test_name", test_key)
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

        # Deep-specific next-steps, GROUNDED in the actual pair/brand (same HOW
        # discipline as the standard "Recommended actions") — NOT generic filler.
        _t = _humanize_var(str(state.get("treatment_var") or ""))
        _o = _humanize_var(str(state.get("outcome_var") or ""))
        _who = state.get("brand") or "this brand"

        interpretation: NaturalLanguageInterpretation = {
            "narrative": enhanced_narrative,
            "executive_summary": standard.get("executive_summary", ""),
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
                (
                    f"Profile where the '{_t}' → '{_o}' effect concentrates for {_who}: read "
                    f"the CATE / heterogeneity view to find the sub-cohorts driving the average "
                    f"and prioritize them, rather than acting on the pooled effect."
                ),
                (
                    f"Add a temporal read: re-estimate '{_t}' → '{_o}' across successive periods "
                    f"to see whether the effect is building, fading, or stable before committing "
                    f"to a multi-cycle plan."
                ),
                (
                    f"Triangulate the '{_t}' → '{_o}' finding against an external source (claims, "
                    f"registry, or a published study) for {_who}; a real effect should hold its "
                    f"direction outside this cohort."
                ),
            ],
            "depth_level": "deep",
            "user_expertise_adjusted": True,
        }

        return interpretation

    async def _collect_dspy_signal(
        self,
        state: CausalImpactState,
        interpretation: NaturalLanguageInterpretation,
        interpretation_latency_ms: float,
    ) -> None:
        """Collect and route DSPy training signal to feedback_learner.

        Non-blocking: failures are logged but don't affect workflow.

        Args:
            state: Complete workflow state with all phase results
            interpretation: Generated interpretation
            interpretation_latency_ms: Interpretation node latency
        """
        try:
            from src.agents.causal_impact.dspy_integration import (
                get_causal_impact_signal_collector,
            )
            from src.agents.tier2_signal_router import route_causal_impact_signal

            collector = get_causal_impact_signal_collector()

            # Initialize signal with input context
            signal = collector.collect_analysis_signal(
                session_id=state.get("session_id", ""),
                query=state.get("query", ""),
                treatment_var=state.get("treatment_var", ""),
                outcome_var=state.get("outcome_var", ""),
                confounders_count=len(state.get("confounders", [])),
            )

            # Update with graph building phase
            causal_graph = state.get("causal_graph", {})
            collector.update_graph_building(
                signal=signal,
                dag_nodes_count=len(causal_graph.get("nodes", [])),
                dag_edges_count=len(causal_graph.get("edges", [])),
                adjustment_sets_found=len(causal_graph.get("adjustment_sets", [])),
                graph_confidence=causal_graph.get("confidence", 0.0),
            )

            # Update with estimation phase
            estimation = state.get("estimation_result", {})
            collector.update_estimation(
                signal=signal,
                method=estimation.get("method", "unknown"),
                ate_estimate=estimation.get("ate", 0.0),
                ate_ci_lower=estimation.get("ate_ci_lower", 0.0),
                ate_ci_upper=estimation.get("ate_ci_upper", 0.0),
                statistical_significance=estimation.get("statistical_significance", False),
                effect_size=estimation.get("effect_size", "unknown"),
                sample_size=estimation.get("sample_size", 0),
            )

            # Update with energy score (V4.2)
            collector.update_energy_score(
                signal=signal,
                energy_score_enabled=state.get("energy_score_enabled", False),
                selection_strategy=estimation.get("selection_strategy", ""),
                selected_estimator=estimation.get("selected_estimator", ""),
                energy_score=estimation.get("energy_score", 0.0),
                energy_score_gap=estimation.get("energy_score_gap", 0.0),
                n_estimators_evaluated=estimation.get("n_estimators_evaluated", 0),
                n_estimators_succeeded=estimation.get("n_estimators_succeeded", 0),
            )

            # Update with refutation phase
            refutation = state.get("refutation_results", {})
            collector.update_refutation(
                signal=signal,
                tests_passed=refutation.get("tests_passed", 0),
                tests_failed=refutation.get("total_tests", 0) - refutation.get("tests_passed", 0),
                overall_robust=refutation.get("overall_robust", False),
            )

            # Update with sensitivity phase
            sensitivity = state.get("sensitivity_analysis", {})
            collector.update_sensitivity(
                signal=signal,
                e_value=sensitivity.get("e_value", 0.0),
                robust_to_confounding=sensitivity.get("robust_to_confounding", False),
            )

            # Calculate total latency
            total_latency_ms = (
                state.get("graph_builder_latency_ms", 0)
                + state.get("estimation_latency_ms", 0)
                + state.get("refutation_latency_ms", 0)
                + state.get("sensitivity_latency_ms", 0)
                + interpretation_latency_ms
            )

            # Update with interpretation phase (final)
            collector.update_interpretation(
                signal=signal,
                interpretation_depth=interpretation.get("depth_level", "standard"),
                narrative_length=len(interpretation.get("narrative", "")),
                key_findings_count=len(interpretation.get("key_findings", [])),
                recommendations_count=len(interpretation.get("recommendations", [])),
                total_latency_ms=total_latency_ms,
                confidence_score=self._confidence_to_score(
                    interpretation.get("causal_confidence", "medium")
                ),
            )

            # Route to feedback_learner
            await route_causal_impact_signal(signal.to_dict())

            logger.debug(
                f"DSPy signal collected: reward={signal.compute_reward():.3f}, "
                f"robust={refutation.get('overall_robust', False)}"
            )

        except Exception as e:
            logger.warning(f"DSPy signal collection failed (non-fatal): {e}")

    def _confidence_to_score(self, confidence: str) -> float:
        """Convert confidence level string to numeric score."""
        confidence_map = {"low": 0.33, "medium": 0.66, "high": 1.0}
        return confidence_map.get(confidence.lower(), 0.5)


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
