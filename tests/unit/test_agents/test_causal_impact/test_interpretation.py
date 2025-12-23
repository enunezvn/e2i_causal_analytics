"""Tests for interpretation node."""

import pytest

from src.agents.causal_impact.nodes.interpretation import InterpretationNode
from src.agents.causal_impact.state import (
    CausalGraph,
    CausalImpactState,
    EstimationResult,
    RefutationResults,
    SensitivityAnalysis,
)


class TestInterpretationNode:
    """Test InterpretationNode."""

    def _create_full_state(self) -> CausalImpactState:
        """Create complete state with all analysis results."""
        causal_graph: CausalGraph = {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "digraph { ... }",
            "confidence": 0.85,
        }

        estimation_result: EstimationResult = {
            "method": "CausalForestDML",
            "ate": 0.5,
            "ate_ci_lower": 0.4,
            "ate_ci_upper": 0.6,
            "standard_error": 0.05,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": 1000,
            "covariates_adjusted": ["geographic_region"],
            "heterogeneity_detected": True,
        }

        # Contract: individual_tests is Dict with test names as keys
        refutation_results: RefutationResults = {
            "tests_passed": 3,
            "tests_failed": 1,
            "total_tests": 4,
            "overall_robust": True,
            "individual_tests": {
                "placebo_treatment": {
                    "test_name": "placebo_treatment",
                    "passed": True,
                    "new_effect": 0.02,
                    "original_effect": 0.5,
                    "p_value": 0.85,
                    "details": "Placebo effect near zero",
                },
                "random_common_cause": {
                    "test_name": "random_common_cause",
                    "passed": True,
                    "new_effect": 0.48,
                    "original_effect": 0.5,
                    "p_value": 0.02,
                    "details": "Effect stable with random cause",
                },
                "data_subset": {
                    "test_name": "data_subset",
                    "passed": True,
                    "new_effect": 0.52,
                    "original_effect": 0.5,
                    "p_value": 0.03,
                    "details": "Effect stable across subsets",
                },
                "unobserved_common_cause": {
                    "test_name": "unobserved_common_cause",
                    "passed": False,
                    "new_effect": 0.3,
                    "original_effect": 0.5,
                    "p_value": 0.08,
                    "details": "E-value indicates moderate sensitivity",
                },
            },
            "confidence_adjustment": 0.75,
        }

        sensitivity_analysis: SensitivityAnalysis = {
            "e_value": 2.5,
            "e_value_ci": 2.2,
            "interpretation": "Effect is robust to moderate confounding",
            "robust_to_confounding": True,
            "unmeasured_confounder_strength": "moderate",
        }

        state: CausalImpactState = {
            "query": "what is the impact of hcp engagement on conversions?",
            "query_id": "test-1",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": causal_graph,
            "estimation_result": estimation_result,
            "refutation_results": refutation_results,
            "sensitivity_analysis": sensitivity_analysis,
            "interpretation_depth": "standard",
            "user_context": {"expertise": "analyst"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        return state

    @pytest.mark.asyncio
    async def test_standard_interpretation(self):
        """Test standard depth interpretation."""
        node = InterpretationNode()

        state = self._create_full_state()
        state["interpretation_depth"] = "standard"

        result = await node.execute(state)

        assert "interpretation" in result
        interp = result["interpretation"]

        assert interp["depth_level"] == "standard"
        assert len(interp["narrative"]) > 100  # Substantial narrative
        assert len(interp["key_findings"]) >= 3
        assert len(interp["recommendations"]) > 0
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_minimal_interpretation(self):
        """Test minimal depth interpretation."""
        node = InterpretationNode()

        state = self._create_full_state()
        state["interpretation_depth"] = "minimal"

        result = await node.execute(state)

        interp = result["interpretation"]

        assert interp["depth_level"] == "minimal"
        # Minimal should be brief (1-2 sentences)
        assert len(interp["narrative"]) < 300

    @pytest.mark.asyncio
    async def test_deep_interpretation(self):
        """Test deep depth interpretation."""
        node = InterpretationNode()

        state = self._create_full_state()
        state["interpretation_depth"] = "deep"

        result = await node.execute(state)

        interp = result["interpretation"]

        assert interp["depth_level"] == "deep"
        # Deep should be comprehensive
        assert len(interp["narrative"]) > 300
        assert len(interp["key_findings"]) >= 4
        assert len(interp["assumptions_made"]) >= 4

    @pytest.mark.asyncio
    async def test_no_interpretation(self):
        """Test depth=none (skip interpretation)."""
        node = InterpretationNode()

        state = self._create_full_state()
        state["interpretation_depth"] = "none"

        result = await node.execute(state)

        interp = result["interpretation"]

        assert interp["depth_level"] == "none"
        assert "skipped" in interp["narrative"].lower()

    @pytest.mark.asyncio
    async def test_executive_expertise_framing(self):
        """Test interpretation for executive audience."""
        node = InterpretationNode()

        state = self._create_full_state()
        state["user_context"] = {"expertise": "executive"}

        result = await node.execute(state)

        interp = result["interpretation"]

        # Executive framing should focus on business impact
        narrative_lower = interp["narrative"].lower()
        assert (
            "business" in narrative_lower
            or "impact" in narrative_lower
            or "outcome" in narrative_lower
        )

    @pytest.mark.asyncio
    async def test_analyst_expertise_framing(self):
        """Test interpretation for analyst audience."""
        node = InterpretationNode()

        state = self._create_full_state()
        state["user_context"] = {"expertise": "analyst"}

        result = await node.execute(state)

        interp = result["interpretation"]

        # Analyst framing should include technical terms
        narrative_lower = interp["narrative"].lower()
        assert (
            "effect" in narrative_lower
            or "analysis" in narrative_lower
            or "confidence" in narrative_lower
        )

    @pytest.mark.asyncio
    async def test_data_scientist_expertise_framing(self):
        """Test interpretation for data scientist audience."""
        node = InterpretationNode()

        state = self._create_full_state()
        state["user_context"] = {"expertise": "data_scientist"}

        result = await node.execute(state)

        interp = result["interpretation"]

        # Data scientist framing should be most technical
        assert interp["user_expertise_adjusted"] is True

    @pytest.mark.asyncio
    async def test_interpretation_includes_effect_magnitude(self):
        """Test that interpretation includes effect magnitude."""
        node = InterpretationNode()

        state = self._create_full_state()

        result = await node.execute(state)

        interp = result["interpretation"]

        assert "effect_magnitude" in interp
        assert interp["effect_magnitude"] == "medium"  # From estimation result

    @pytest.mark.asyncio
    async def test_interpretation_includes_confidence(self):
        """Test that interpretation includes causal confidence."""
        node = InterpretationNode()

        state = self._create_full_state()

        result = await node.execute(state)

        interp = result["interpretation"]

        assert "causal_confidence" in interp
        assert interp["causal_confidence"] in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_interpretation_includes_assumptions(self):
        """Test that interpretation lists assumptions."""
        node = InterpretationNode()

        state = self._create_full_state()

        result = await node.execute(state)

        interp = result["interpretation"]

        assert "assumptions_made" in interp
        assert len(interp["assumptions_made"]) > 0
        # Should include standard causal assumptions
        assumptions_text = " ".join(interp["assumptions_made"]).lower()
        assert "confound" in assumptions_text or "assumption" in assumptions_text

    @pytest.mark.asyncio
    async def test_interpretation_includes_limitations(self):
        """Test that interpretation lists limitations."""
        node = InterpretationNode()

        state = self._create_full_state()

        result = await node.execute(state)

        interp = result["interpretation"]

        assert "limitations" in interp
        assert len(interp["limitations"]) > 0

    @pytest.mark.asyncio
    async def test_interpretation_includes_recommendations(self):
        """Test that interpretation includes recommendations."""
        node = InterpretationNode()

        state = self._create_full_state()

        result = await node.execute(state)

        interp = result["interpretation"]

        assert "recommendations" in interp
        assert len(interp["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test that interpretation latency is measured."""
        node = InterpretationNode()

        state = self._create_full_state()

        result = await node.execute(state)

        assert "interpretation_latency_ms" in result
        assert result["interpretation_latency_ms"] >= 0
        assert result["interpretation_latency_ms"] < 30000  # Should be < 30s

    @pytest.mark.asyncio
    async def test_high_confidence_interpretation(self):
        """Test interpretation when all signals are positive."""
        node = InterpretationNode()

        state = self._create_full_state()
        # All positive signals
        state["estimation_result"]["statistical_significance"] = True
        state["refutation_results"]["overall_robust"] = True
        state["sensitivity_analysis"]["robust_to_confounding"] = True

        result = await node.execute(state)

        interp = result["interpretation"]

        # Should have high confidence
        assert interp["causal_confidence"] == "high"
        # Recommendations should be actionable
        assert len(interp["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_low_confidence_interpretation(self):
        """Test interpretation when signals are negative."""
        node = InterpretationNode()

        state = self._create_full_state()
        # Negative signals
        state["estimation_result"]["statistical_significance"] = False
        state["refutation_results"]["overall_robust"] = False
        state["sensitivity_analysis"]["robust_to_confounding"] = False

        result = await node.execute(state)

        interp = result["interpretation"]

        # Should have low confidence
        assert interp["causal_confidence"] == "low"
        # Recommendations should be cautious (includes phrases like "collect data", "sensitivity analyses")
        recommendations_text = " ".join(interp["recommendations"]).lower()
        assert (
            "caution" in recommendations_text
            or "further" in recommendations_text
            or "investigate" in recommendations_text
            or "additional data" in recommendations_text
            or "sensitivity" in recommendations_text
            or "validate" in recommendations_text
        )


class TestInterpretationNarrativeQuality:
    """Test quality of generated narratives."""

    @pytest.mark.asyncio
    async def test_narrative_mentions_effect_size(self):
        """Test that narrative mentions the effect size."""
        node = InterpretationNode()

        state = self._create_full_state()
        state["estimation_result"]["effect_size"] = "large"

        result = await node.execute(state)

        narrative = result["interpretation"]["narrative"].lower()

        assert "large" in narrative or "effect" in narrative

    @pytest.mark.asyncio
    async def test_narrative_mentions_robustness(self):
        """Test that narrative mentions robustness results."""
        node = InterpretationNode()

        state = self._create_full_state()
        state["interpretation_depth"] = "standard"

        result = await node.execute(state)

        narrative = result["interpretation"]["narrative"].lower()

        assert "robust" in narrative or "test" in narrative

    @pytest.mark.asyncio
    async def test_narrative_mentions_evalue(self):
        """Test that narrative mentions E-value."""
        node = InterpretationNode()

        state = self._create_full_state()
        state["interpretation_depth"] = "standard"

        result = await node.execute(state)

        narrative = result["interpretation"]["narrative"].lower()

        assert "e-value" in narrative or "confound" in narrative

    def _create_full_state(self) -> CausalImpactState:
        """Create complete state with all analysis results."""
        causal_graph: CausalGraph = {
            "nodes": ["T", "O", "C"],
            "edges": [("C", "T"), ("T", "O")],
            "treatment_nodes": ["T"],
            "outcome_nodes": ["O"],
            "adjustment_sets": [["C"]],
            "dag_dot": "...",
            "confidence": 0.85,
        }

        estimation_result: EstimationResult = {
            "method": "CausalForestDML",
            "ate": 0.5,
            "ate_ci_lower": 0.4,
            "ate_ci_upper": 0.6,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": 1000,
            "covariates_adjusted": ["C"],
            "heterogeneity_detected": True,
        }

        # Contract: individual_tests is Dict with test names as keys
        refutation_results: RefutationResults = {
            "tests_passed": 3,
            "tests_failed": 1,
            "total_tests": 4,
            "overall_robust": True,
            "individual_tests": {},
            "confidence_adjustment": 0.75,
        }

        sensitivity_analysis: SensitivityAnalysis = {
            "e_value": 2.5,
            "e_value_ci": 2.2,
            "interpretation": "...",
            "robust_to_confounding": True,
            "unmeasured_confounder_strength": "moderate",
        }

        return {
            "query": "test query",
            "query_id": "test-1",
            "treatment_var": "T",
            "outcome_var": "O",
            "confounders": ["C"],
            "data_source": "synthetic",
            "causal_graph": causal_graph,
            "estimation_result": estimation_result,
            "refutation_results": refutation_results,
            "sensitivity_analysis": sensitivity_analysis,
            "interpretation_depth": "standard",
            "user_context": {"expertise": "analyst"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }


class TestKeyFindingsGeneration:
    """Test key findings generation."""

    @pytest.mark.asyncio
    async def test_key_findings_include_ate(self):
        """Test that key findings include ATE."""
        node = InterpretationNode()

        state = self._create_full_state()

        result = await node.execute(state)

        key_findings = result["interpretation"]["key_findings"]
        findings_text = " ".join(key_findings).lower()

        assert "effect" in findings_text or "0.5" in findings_text

    @pytest.mark.asyncio
    async def test_key_findings_include_significance(self):
        """Test that key findings include significance."""
        node = InterpretationNode()

        state = self._create_full_state()

        result = await node.execute(state)

        key_findings = result["interpretation"]["key_findings"]
        findings_text = " ".join(key_findings).lower()

        assert "significance" in findings_text or "significant" in findings_text

    def _create_full_state(self) -> CausalImpactState:
        """Create complete state."""
        return {
            "query": "test",
            "query_id": "test-1",
            "treatment_var": "T",
            "outcome_var": "O",
            "confounders": [],
            "data_source": "synthetic",
            "causal_graph": {
                "nodes": ["T", "O"],
                "edges": [("T", "O")],
                "treatment_nodes": ["T"],
                "outcome_nodes": ["O"],
                "adjustment_sets": [[]],
                "dag_dot": "...",
                "confidence": 0.8,
            },
            "estimation_result": {
                "method": "CausalForestDML",
                "ate": 0.5,
                "ate_ci_lower": 0.4,
                "ate_ci_upper": 0.6,
                "standard_error": 0.05,
                "effect_size": "medium",
                "statistical_significance": True,
                "p_value": 0.01,
                "sample_size": 1000,
                "covariates_adjusted": [],
                "heterogeneity_detected": False,
            },
            # Contract: individual_tests is Dict
            "refutation_results": {
                "tests_passed": 3,
                "tests_failed": 1,
                "total_tests": 4,
                "overall_robust": True,
                "individual_tests": {},
                "confidence_adjustment": 0.75,
            },
            "sensitivity_analysis": {
                "e_value": 2.5,
                "e_value_ci": 2.2,
                "interpretation": "...",
                "robust_to_confounding": True,
                "unmeasured_confounder_strength": "moderate",
            },
            "interpretation_depth": "standard",
            "user_context": {"expertise": "analyst"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }
