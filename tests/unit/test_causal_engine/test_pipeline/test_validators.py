"""Unit tests for Pipeline Validators.

Tests the inter-library compatibility validation.
"""

import pytest

from src.causal_engine.pipeline import (
    CausalLibrary,
    DoWhyToEconMLValidator,
    EconMLToCausalMLValidator,
    LibraryExecutionResult,
    NetworkXToDoWhyValidator,
    PipelineConfig,
    PipelineStage,
    PipelineState,
    PipelineValidator,
    ValidationResult,
    validate_pipeline_state,
)


def create_empty_state() -> PipelineState:
    """Create an empty pipeline state for testing."""
    return PipelineState(
        query="test query",
        question_type=None,
        treatment_var=None,
        outcome_var=None,
        confounders=None,
        effect_modifiers=None,
        data_source="test",
        filters=None,
        config=PipelineConfig(
            mode="sequential",
            libraries_enabled=["networkx", "dowhy", "econml", "causalml"],
            primary_library="dowhy",
            stage_timeout_ms=30000,
            total_timeout_ms=120000,
            cross_validate=True,
            min_agreement_threshold=0.85,
            max_parallel_libraries=4,
            fail_fast=False,
            segment_by_uplift=False,
            nested_ci_level=0.95,
        ),
        routed_libraries=[],
        routing_confidence=0.0,
        routing_rationale=None,
        networkx_result=None,
        causal_graph=None,
        graph_metrics=None,
        dowhy_result=None,
        causal_effect=None,
        refutation_results=None,
        identification_method=None,
        econml_result=None,
        cate_by_segment=None,
        overall_ate=None,
        heterogeneity_score=None,
        causalml_result=None,
        uplift_scores=None,
        auuc=None,
        qini=None,
        targeting_recommendations=None,
        consensus_effect=None,
        consensus_confidence=None,
        library_agreement=None,
        nested_cate=None,
        segment_confidence_intervals=None,
        executive_summary=None,
        key_insights=None,
        recommended_actions=None,
        current_stage=PipelineStage.PENDING,
        stage_latencies={},
        total_latency_ms=0,
        libraries_executed=[],
        libraries_skipped=[],
        errors=[],
        warnings=[],
        status="pending",
    )


class TestValidationResult:
    """Test ValidationResult class."""

    def test_valid_result_is_truthy(self):
        """Test that valid result is truthy."""
        result = ValidationResult(is_valid=True)
        assert bool(result) is True

    def test_invalid_result_is_falsy(self):
        """Test that invalid result is falsy."""
        result = ValidationResult(is_valid=False)
        assert bool(result) is False

    def test_result_with_errors(self):
        """Test result with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
        )
        assert len(result.errors) == 2
        assert "Error 1" in result.errors

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = ValidationResult(
            is_valid=True,
            warnings=["Warning 1"],
            suggestions=["Suggestion 1"],
        )
        d = result.to_dict()
        assert d["is_valid"] is True
        assert "warnings" in d
        assert "suggestions" in d


class TestNetworkXToDoWhyValidator:
    """Test NetworkX to DoWhy validation."""

    @pytest.fixture
    def validator(self) -> NetworkXToDoWhyValidator:
        return NetworkXToDoWhyValidator()

    def test_validator_libraries(self, validator: NetworkXToDoWhyValidator):
        """Test validator identifies correct libraries."""
        assert validator.source_library == CausalLibrary.NETWORKX
        assert validator.target_library == CausalLibrary.DOWHY

    def test_validates_missing_graph(self, validator: NetworkXToDoWhyValidator):
        """Test validation fails when no graph available."""
        state = create_empty_state()
        result = validator.validate(state)

        assert result.is_valid is False
        assert any("No causal graph" in e for e in result.errors)

    def test_validates_empty_graph(self, validator: NetworkXToDoWhyValidator):
        """Test validation fails with empty graph nodes."""
        state = create_empty_state()
        state["causal_graph"] = {"nodes": [], "edges": []}

        result = validator.validate(state)

        assert result.is_valid is False
        assert any("no nodes" in e for e in result.errors)

    def test_validates_graph_without_edges(self, validator: NetworkXToDoWhyValidator):
        """Test validation warns on graph without edges."""
        state = create_empty_state()
        state["causal_graph"] = {"nodes": ["A", "B"], "edges": []}

        result = validator.validate(state)

        assert result.is_valid is True
        assert any("no edges" in w for w in result.warnings)

    def test_validates_missing_treatment_in_graph(self, validator: NetworkXToDoWhyValidator):
        """Test validation warns when treatment not in graph."""
        state = create_empty_state()
        state["treatment_var"] = "treatment"
        state["outcome_var"] = "outcome"
        state["causal_graph"] = {
            "nodes": ["A", "B"],
            "edges": [{"from": "A", "to": "B"}],
        }

        result = validator.validate(state)

        assert result.is_valid is True
        assert any("treatment" in w.lower() for w in result.warnings)

    def test_validates_complete_graph(self, validator: NetworkXToDoWhyValidator):
        """Test validation passes with complete graph."""
        state = create_empty_state()
        state["treatment_var"] = "treatment"
        state["outcome_var"] = "outcome"
        state["causal_graph"] = {
            "nodes": ["treatment", "outcome", "confounder"],
            "edges": [
                {"from": "treatment", "to": "outcome"},
                {"from": "confounder", "to": "treatment"},
                {"from": "confounder", "to": "outcome"},
            ],
        }

        result = validator.validate(state)

        assert result.is_valid is True
        assert len(result.errors) == 0


class TestDoWhyToEconMLValidator:
    """Test DoWhy to EconML validation."""

    @pytest.fixture
    def validator(self) -> DoWhyToEconMLValidator:
        return DoWhyToEconMLValidator()

    def test_validator_libraries(self, validator: DoWhyToEconMLValidator):
        """Test validator identifies correct libraries."""
        assert validator.source_library == CausalLibrary.DOWHY
        assert validator.target_library == CausalLibrary.ECONML

    def test_validates_missing_dowhy_result(self, validator: DoWhyToEconMLValidator):
        """Test validation fails when no DoWhy result."""
        state = create_empty_state()
        result = validator.validate(state)

        assert result.is_valid is False
        assert any("No DoWhy" in e for e in result.errors)

    def test_validates_failed_dowhy_execution(self, validator: DoWhyToEconMLValidator):
        """Test validation fails when DoWhy execution failed."""
        state = create_empty_state()
        state["dowhy_result"] = LibraryExecutionResult(
            library="dowhy",
            success=False,
            latency_ms=100,
            result=None,
            error="Estimation failed",
            confidence=0.0,
            warnings=[],
        )

        result = validator.validate(state)

        assert result.is_valid is False
        assert any("failed" in e.lower() for e in result.errors)

    def test_validates_missing_identification(self, validator: DoWhyToEconMLValidator):
        """Test validation warns on missing identification."""
        state = create_empty_state()
        state["dowhy_result"] = LibraryExecutionResult(
            library="dowhy",
            success=True,
            latency_ms=100,
            result={"causal_effect": 0.5},
            error=None,
            confidence=0.85,
            warnings=[],
        )

        result = validator.validate(state)

        assert result.is_valid is True
        assert any("identification" in w.lower() for w in result.warnings)

    def test_validates_missing_effect_modifiers(self, validator: DoWhyToEconMLValidator):
        """Test validation warns on missing effect modifiers."""
        state = create_empty_state()
        state["dowhy_result"] = LibraryExecutionResult(
            library="dowhy",
            success=True,
            latency_ms=100,
            result={"causal_effect": 0.5},
            error=None,
            confidence=0.85,
            warnings=[],
        )
        state["identification_method"] = "backdoor"
        state["causal_effect"] = 0.5

        result = validator.validate(state)

        assert result.is_valid is True
        assert any("effect modifiers" in w.lower() for w in result.warnings)

    def test_validates_complete_dowhy_result(self, validator: DoWhyToEconMLValidator):
        """Test validation passes with complete DoWhy result."""
        state = create_empty_state()
        state["dowhy_result"] = LibraryExecutionResult(
            library="dowhy",
            success=True,
            latency_ms=100,
            result={"causal_effect": 0.5},
            error=None,
            confidence=0.85,
            warnings=[],
        )
        state["identification_method"] = "backdoor"
        state["causal_effect"] = 0.5
        state["effect_modifiers"] = ["region", "age"]

        result = validator.validate(state)

        assert result.is_valid is True


class TestEconMLToCausalMLValidator:
    """Test EconML to CausalML validation."""

    @pytest.fixture
    def validator(self) -> EconMLToCausalMLValidator:
        return EconMLToCausalMLValidator()

    def test_validator_libraries(self, validator: EconMLToCausalMLValidator):
        """Test validator identifies correct libraries."""
        assert validator.source_library == CausalLibrary.ECONML
        assert validator.target_library == CausalLibrary.CAUSALML

    def test_validates_missing_econml_result(self, validator: EconMLToCausalMLValidator):
        """Test validation fails when no EconML result."""
        state = create_empty_state()
        result = validator.validate(state)

        assert result.is_valid is False
        assert any("No EconML" in e for e in result.errors)

    def test_validates_failed_econml_execution(self, validator: EconMLToCausalMLValidator):
        """Test validation fails when EconML execution failed."""
        state = create_empty_state()
        state["econml_result"] = LibraryExecutionResult(
            library="econml",
            success=False,
            latency_ms=100,
            result=None,
            error="CATE estimation failed",
            confidence=0.0,
            warnings=[],
        )

        result = validator.validate(state)

        assert result.is_valid is False
        assert any("failed" in e.lower() for e in result.errors)

    def test_validates_low_heterogeneity(self, validator: EconMLToCausalMLValidator):
        """Test validation warns on low heterogeneity."""
        state = create_empty_state()
        state["econml_result"] = LibraryExecutionResult(
            library="econml",
            success=True,
            latency_ms=100,
            result={"ate": 0.5},
            error=None,
            confidence=0.82,
            warnings=[],
        )
        state["heterogeneity_score"] = 0.05

        result = validator.validate(state)

        assert result.is_valid is True
        assert any("heterogeneity" in w.lower() for w in result.warnings)

    def test_validates_negative_cate_segments(self, validator: EconMLToCausalMLValidator):
        """Test validation warns on negative CATE segments."""
        state = create_empty_state()
        state["econml_result"] = LibraryExecutionResult(
            library="econml",
            success=True,
            latency_ms=100,
            result={"ate": 0.2},
            error=None,
            confidence=0.82,
            warnings=[],
        )
        state["cate_by_segment"] = {
            "segment_A": 0.5,
            "segment_B": -0.3,
            "segment_C": 0.1,
        }

        result = validator.validate(state)

        assert result.is_valid is True
        assert any("negative" in w.lower() for w in result.warnings)

    def test_validates_zero_ate(self, validator: EconMLToCausalMLValidator):
        """Test validation warns on zero ATE."""
        state = create_empty_state()
        state["econml_result"] = LibraryExecutionResult(
            library="econml",
            success=True,
            latency_ms=100,
            result={"ate": 0.0},
            error=None,
            confidence=0.82,
            warnings=[],
        )
        state["overall_ate"] = 0.0

        result = validator.validate(state)

        assert result.is_valid is True
        assert any("zero" in w.lower() for w in result.warnings)


class TestPipelineValidator:
    """Test PipelineValidator orchestrator."""

    @pytest.fixture
    def validator(self) -> PipelineValidator:
        return PipelineValidator()

    def test_has_default_validators(self, validator: PipelineValidator):
        """Test that default validators are registered."""
        # Test by validating transitions
        state = create_empty_state()

        # Should not raise
        validator.validate_transition(state, CausalLibrary.NETWORKX, CausalLibrary.DOWHY)
        validator.validate_transition(state, CausalLibrary.DOWHY, CausalLibrary.ECONML)
        validator.validate_transition(state, CausalLibrary.ECONML, CausalLibrary.CAUSALML)

    def test_validate_unknown_transition(self, validator: PipelineValidator):
        """Test validating unknown transition returns valid with warning."""
        state = create_empty_state()

        result = validator.validate_transition(
            state, CausalLibrary.NETWORKX, CausalLibrary.CAUSALML
        )

        assert result.is_valid is True
        assert any("No validator" in w for w in result.warnings)

    def test_validate_full_pipeline(self, validator: PipelineValidator):
        """Test validating full pipeline."""
        state = create_empty_state()
        libraries = [
            CausalLibrary.NETWORKX,
            CausalLibrary.DOWHY,
            CausalLibrary.ECONML,
            CausalLibrary.CAUSALML,
        ]

        results = validator.validate_full_pipeline(state, libraries)

        assert "networkx_to_dowhy" in results
        assert "dowhy_to_econml" in results
        assert "econml_to_causalml" in results

    def test_get_all_warnings(self, validator: PipelineValidator):
        """Test extracting all warnings."""
        results = {
            "networkx_to_dowhy": ValidationResult(True, warnings=["Warning 1"]),
            "dowhy_to_econml": ValidationResult(True, warnings=["Warning 2"]),
        }

        warnings = validator.get_all_warnings(results)

        assert len(warnings) == 2
        assert any("Warning 1" in w for w in warnings)
        assert any("Warning 2" in w for w in warnings)

    def test_get_all_suggestions(self, validator: PipelineValidator):
        """Test extracting all suggestions."""
        results = {
            "networkx_to_dowhy": ValidationResult(True, suggestions=["Suggestion 1"]),
            "dowhy_to_econml": ValidationResult(True, suggestions=["Suggestion 2"]),
        }

        suggestions = validator.get_all_suggestions(results)

        assert len(suggestions) == 2
        assert any("Suggestion 1" in s for s in suggestions)
        assert any("Suggestion 2" in s for s in suggestions)


class TestValidatePipelineStateFunction:
    """Test convenience function validate_pipeline_state."""

    def test_convenience_function_works(self):
        """Test that convenience function works."""
        state = create_empty_state()

        result = validate_pipeline_state(state, CausalLibrary.NETWORKX, CausalLibrary.DOWHY)

        assert isinstance(result, ValidationResult)


class TestCustomValidatorRegistration:
    """Test registering custom validators."""

    def test_register_custom_validator(self):
        """Test registering a custom validator."""

        class CustomValidator(NetworkXToDoWhyValidator):
            def validate(self, state: PipelineState) -> ValidationResult:
                return ValidationResult(True, suggestions=["Custom suggestion"])

        validator = PipelineValidator()
        custom = CustomValidator()
        validator.register_validator(custom)

        state = create_empty_state()
        result = validator.validate_transition(state, CausalLibrary.NETWORKX, CausalLibrary.DOWHY)

        assert "Custom suggestion" in result.suggestions
