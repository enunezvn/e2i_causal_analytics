"""Tests for Experiment Designer State Types.

Tests TypedDict definitions and state initialization.
"""

import pytest
from typing import get_type_hints
from src.agents.experiment_designer.state import (
    ExperimentDesignState,
    TreatmentDefinition,
    OutcomeDefinition,
    ValidityThreat,
    MitigationRecommendation,
    PowerAnalysisResult,
    DoWhySpec,
    ExperimentTemplate,
    DesignIteration,
    ErrorDetails,
)
from src.agents.experiment_designer.graph import create_initial_state


class TestTreatmentDefinition:
    """Test TreatmentDefinition TypedDict."""

    def test_create_minimal(self):
        """Test creating minimal treatment definition."""
        treatment: TreatmentDefinition = {
            "name": "Treatment A",
            "description": "Increased visit frequency",
            "implementation_details": "Weekly visits instead of bi-weekly",
            "target_population": "All HCPs in territory"
        }

        assert treatment["name"] == "Treatment A"
        assert treatment["description"] == "Increased visit frequency"

    def test_create_full(self):
        """Test creating full treatment definition."""
        treatment: TreatmentDefinition = {
            "name": "Treatment A",
            "description": "Increased visit frequency",
            "implementation_details": "Weekly visits instead of bi-weekly",
            "target_population": "High-value HCPs in northeast region",
            "dosage_or_intensity": "2x baseline visits",
            "duration": "12 weeks"
        }

        assert treatment["dosage_or_intensity"] == "2x baseline visits"
        assert treatment["duration"] == "12 weeks"


class TestOutcomeDefinition:
    """Test OutcomeDefinition TypedDict."""

    def test_create_continuous_outcome(self):
        """Test creating continuous outcome definition."""
        outcome: OutcomeDefinition = {
            "name": "Engagement Score",
            "metric_type": "continuous",
            "measurement_method": "CRM engagement index",
            "measurement_frequency": "weekly",
            "is_primary": True,
            "baseline_value": 45.0,
            "expected_effect_size": 5.0
        }

        assert outcome["metric_type"] == "continuous"
        assert outcome["is_primary"] is True
        assert outcome["baseline_value"] == 45.0

    def test_create_binary_outcome(self):
        """Test creating binary outcome definition."""
        outcome: OutcomeDefinition = {
            "name": "Conversion",
            "metric_type": "binary",
            "measurement_method": "First prescription written",
            "measurement_frequency": "monthly",
            "is_primary": True,
            "baseline_value": 0.15,
            "expected_effect_size": 0.05
        }

        assert outcome["metric_type"] == "binary"

    def test_create_time_to_event_outcome(self):
        """Test creating time-to-event outcome definition."""
        outcome: OutcomeDefinition = {
            "name": "Time to First Prescription",
            "metric_type": "time_to_event",
            "measurement_method": "Days from first contact to first Rx",
            "measurement_frequency": "continuous",
            "is_primary": True
        }

        assert outcome["metric_type"] == "time_to_event"


class TestValidityThreat:
    """Test ValidityThreat TypedDict."""

    def test_create_internal_threat(self):
        """Test creating internal validity threat."""
        threat: ValidityThreat = {
            "threat_type": "internal",
            "threat_name": "selection_bias",
            "description": "Non-random assignment to treatment groups",
            "severity": "high",
            "mitigation_possible": True,
            "mitigation_strategy": "Use stratified randomization on baseline characteristics"
        }

        assert threat["threat_type"] == "internal"
        assert threat["severity"] == "high"
        assert threat["mitigation_possible"] is True

    def test_create_external_threat(self):
        """Test creating external validity threat."""
        threat: ValidityThreat = {
            "threat_type": "external",
            "threat_name": "generalizability",
            "description": "Results may not generalize to other regions",
            "severity": "medium",
            "mitigation_possible": True,
            "mitigation_strategy": "Include diverse geographic regions in sample"
        }

        assert threat["threat_type"] == "external"

    def test_create_statistical_threat(self):
        """Test creating statistical validity threat."""
        threat: ValidityThreat = {
            "threat_type": "statistical",
            "threat_name": "low_power",
            "description": "Insufficient sample size to detect expected effect",
            "severity": "critical",
            "mitigation_possible": True,
            "mitigation_strategy": "Increase sample size or extend duration"
        }

        assert threat["threat_type"] == "statistical"
        assert threat["severity"] == "critical"

    def test_severity_levels(self):
        """Test all valid severity levels."""
        for severity in ["low", "medium", "high", "critical"]:
            threat: ValidityThreat = {
                "threat_type": "internal",
                "threat_name": "test",
                "description": "Test threat",
                "severity": severity,
                "mitigation_possible": True
            }
            assert threat["severity"] == severity


class TestMitigationRecommendation:
    """Test MitigationRecommendation TypedDict."""

    def test_create_mitigation(self):
        """Test creating mitigation recommendation."""
        mitigation: MitigationRecommendation = {
            "threat_addressed": "selection_bias",
            "recommendation": "Use stratified randomization",
            "effectiveness_rating": "high",
            "implementation_cost": "low",
            "implementation_steps": [
                "Identify key stratification variables",
                "Create balanced strata",
                "Randomize within strata"
            ]
        }

        assert mitigation["threat_addressed"] == "selection_bias"
        assert mitigation["effectiveness_rating"] == "high"
        assert len(mitigation["implementation_steps"]) == 3

    def test_effectiveness_ratings(self):
        """Test all valid effectiveness ratings."""
        for rating in ["low", "medium", "high"]:
            mitigation: MitigationRecommendation = {
                "threat_addressed": "test",
                "recommendation": "Test mitigation",
                "effectiveness_rating": rating,
                "implementation_cost": "medium",
                "implementation_steps": []
            }
            assert mitigation["effectiveness_rating"] == rating


class TestPowerAnalysisResult:
    """Test PowerAnalysisResult TypedDict."""

    def test_create_power_analysis(self):
        """Test creating power analysis result."""
        power: PowerAnalysisResult = {
            "required_sample_size": 500,
            "required_sample_size_per_arm": 250,
            "achieved_power": 0.82,
            "minimum_detectable_effect": 0.25,
            "alpha": 0.05,
            "effect_size_type": "cohens_d",
            "assumptions": [
                "Equal variances assumed",
                "Normal distribution of outcome",
                "No clustering effects"
            ],
            "sensitivity_analysis": {
                "power_at_smaller_effect": 0.65,
                "power_at_larger_effect": 0.92
            }
        }

        assert power["required_sample_size"] == 500
        assert power["achieved_power"] == 0.82
        assert len(power["assumptions"]) == 3

    def test_effect_size_types(self):
        """Test different effect size types."""
        for es_type in ["cohens_d", "odds_ratio", "hazard_ratio", "relative_risk"]:
            power: PowerAnalysisResult = {
                "required_sample_size": 100,
                "required_sample_size_per_arm": 50,
                "achieved_power": 0.8,
                "minimum_detectable_effect": 0.3,
                "alpha": 0.05,
                "effect_size_type": es_type,
                "assumptions": []
            }
            assert power["effect_size_type"] == es_type


class TestDoWhySpec:
    """Test DoWhySpec TypedDict."""

    def test_create_dowhy_spec(self):
        """Test creating DoWhy specification."""
        spec: DoWhySpec = {
            "treatment_variable": "visit_frequency",
            "outcome_variable": "engagement_score",
            "confounders": ["territory_size", "baseline_engagement", "hcp_specialty"],
            "instruments": ["random_assignment"],
            "effect_modifiers": ["hcp_experience", "region"],
            "causal_graph_dot": "digraph { visit_frequency -> engagement_score; territory_size -> engagement_score; }"
        }

        assert spec["treatment_variable"] == "visit_frequency"
        assert len(spec["confounders"]) == 3
        assert "digraph" in spec["causal_graph_dot"]

    def test_minimal_spec(self):
        """Test creating minimal DoWhy specification."""
        spec: DoWhySpec = {
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "confounders": [],
            "instruments": [],
            "effect_modifiers": [],
            "causal_graph_dot": ""
        }

        assert spec["treatment_variable"] == "treatment"


class TestExperimentTemplate:
    """Test ExperimentTemplate TypedDict."""

    def test_create_template(self):
        """Test creating experiment template."""
        template: ExperimentTemplate = {
            "analysis_code": "from dowhy import CausalModel\n# Analysis code...",
            "pre_registration_document": "# Pre-registration\n## Hypothesis...",
            "monitoring_dashboard_spec": {
                "metrics": ["enrollment_rate", "dropout_rate"],
                "alerts": ["low_enrollment", "high_dropout"]
            },
            "randomization_script": "import random\n# Randomization code..."
        }

        assert "CausalModel" in template["analysis_code"]
        assert len(template["monitoring_dashboard_spec"]["metrics"]) == 2

    def test_template_with_empty_monitoring(self):
        """Test template with empty monitoring spec."""
        template: ExperimentTemplate = {
            "analysis_code": "# Code",
            "pre_registration_document": "# Doc",
            "monitoring_dashboard_spec": {},
            "randomization_script": ""
        }

        assert template["monitoring_dashboard_spec"] == {}


class TestDesignIteration:
    """Test DesignIteration TypedDict."""

    def test_create_iteration(self):
        """Test creating design iteration record."""
        iteration: DesignIteration = {
            "iteration_number": 1,
            "design_type": "RCT",
            "validity_threats_identified": 5,
            "critical_threats": 1,
            "power_achieved": 0.75,
            "redesign_reason": "Insufficient power for primary outcome",
            "timestamp": "2024-01-15T10:30:00Z"
        }

        assert iteration["iteration_number"] == 1
        assert iteration["critical_threats"] == 1

    def test_multiple_iterations(self):
        """Test tracking multiple iterations."""
        iterations = []
        for i in range(3):
            iteration: DesignIteration = {
                "iteration_number": i,
                "design_type": "RCT",
                "validity_threats_identified": 5 - i,  # Decreasing threats
                "critical_threats": 1 if i == 0 else 0,
                "power_achieved": 0.75 + (i * 0.05),  # Increasing power
                "redesign_reason": f"Iteration {i} reason",
                "timestamp": f"2024-01-{15 + i}T10:30:00Z"
            }
            iterations.append(iteration)

        assert len(iterations) == 3
        assert iterations[2]["power_achieved"] == 0.85


class TestErrorDetails:
    """Test ErrorDetails TypedDict."""

    def test_create_error(self):
        """Test creating error details."""
        error: ErrorDetails = {
            "node": "design_reasoning",
            "error": "LLM response parsing failed",
            "timestamp": "2024-01-15T10:30:00Z",
            "recoverable": True
        }

        assert error["node"] == "design_reasoning"
        assert error["recoverable"] is True

    def test_non_recoverable_error(self):
        """Test non-recoverable error."""
        error: ErrorDetails = {
            "node": "power_analysis",
            "error": "Invalid effect size",
            "timestamp": "2024-01-15T10:30:00Z",
            "recoverable": False
        }

        assert error["recoverable"] is False


class TestExperimentDesignState:
    """Test ExperimentDesignState TypedDict."""

    def test_create_initial_state(self):
        """Test creating initial state."""
        state = create_initial_state(
            business_question="Does X impact Y?"
        )

        assert state["business_question"] == "Does X impact Y?"
        assert state["constraints"] == {}
        assert state["available_data"] == {}
        assert state["preregistration_formality"] == "medium"
        assert state["max_redesign_iterations"] == 2
        assert state["enable_validity_audit"] is True
        assert state["errors"] == []
        assert state["warnings"] == []
        assert state["status"] == "pending"

    def test_create_initial_state_with_constraints(self):
        """Test creating initial state with constraints."""
        state = create_initial_state(
            business_question="Test question here?",
            constraints={
                "expected_effect_size": 0.25,
                "power": 0.80
            },
            available_data={
                "variables": ["var1", "var2"]
            }
        )

        assert state["constraints"]["expected_effect_size"] == 0.25
        assert state["available_data"]["variables"] == ["var1", "var2"]

    def test_create_initial_state_custom_formality(self):
        """Test creating initial state with custom formality."""
        state = create_initial_state(
            business_question="Test question here?",
            preregistration_formality="heavy"
        )

        assert state["preregistration_formality"] == "heavy"

    def test_create_initial_state_custom_iterations(self):
        """Test creating initial state with custom iterations."""
        state = create_initial_state(
            business_question="Test question here?",
            max_redesign_iterations=5
        )

        assert state["max_redesign_iterations"] == 5

    def test_create_initial_state_no_validity_audit(self):
        """Test creating initial state without validity audit."""
        state = create_initial_state(
            business_question="Test question here?",
            enable_validity_audit=False
        )

        assert state["enable_validity_audit"] is False


class TestStateFieldTypes:
    """Test state field type annotations."""

    def test_state_has_required_fields(self):
        """Test state has all required fields."""
        required_fields = [
            "business_question",
            "constraints",
            "available_data",
            "preregistration_formality",
            "max_redesign_iterations",
            "enable_validity_audit",
            "errors",
            "warnings",
            "status"
        ]

        hints = get_type_hints(ExperimentDesignState)

        for field in required_fields:
            assert field in hints, f"Missing required field: {field}"

    def test_state_has_design_fields(self):
        """Test state has design-related fields."""
        design_fields = [
            "design_type",
            "design_rationale",
            "treatments",
            "outcomes",
            "randomization_unit",
            "randomization_method",
            "stratification_variables",
            "blocking_variables"
        ]

        hints = get_type_hints(ExperimentDesignState)

        for field in design_fields:
            assert field in hints, f"Missing design field: {field}"

    def test_state_has_power_fields(self):
        """Test state has power analysis fields."""
        power_fields = [
            "power_analysis",
            "sample_size_justification",
            "duration_estimate_days"
        ]

        hints = get_type_hints(ExperimentDesignState)

        for field in power_fields:
            assert field in hints, f"Missing power field: {field}"

    def test_state_has_validity_fields(self):
        """Test state has validity audit fields."""
        validity_fields = [
            "validity_threats",
            "mitigations",
            "overall_validity_score",
            "validity_confidence",
            "redesign_needed",
            "redesign_recommendations"
        ]

        hints = get_type_hints(ExperimentDesignState)

        for field in validity_fields:
            assert field in hints, f"Missing validity field: {field}"

    def test_state_has_template_fields(self):
        """Test state has template generation fields."""
        template_fields = [
            "dowhy_spec",
            "causal_graph_dot",
            "analysis_code",
            "experiment_template"
        ]

        hints = get_type_hints(ExperimentDesignState)

        for field in template_fields:
            assert field in hints, f"Missing template field: {field}"

    def test_state_has_execution_fields(self):
        """Test state has execution metadata fields."""
        execution_fields = [
            "current_iteration",
            "iteration_history",
            "node_latencies_ms"
        ]

        hints = get_type_hints(ExperimentDesignState)

        for field in execution_fields:
            assert field in hints, f"Missing execution field: {field}"
