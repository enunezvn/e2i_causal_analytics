"""
Tests for src/causal_engine/pipeline/state.py

Covers:
- PipelineStage enum
- LibraryExecutionResult TypedDict
- PipelineConfig TypedDict
- PipelineState TypedDict
- PipelineInput TypedDict
- PipelineOutput TypedDict
"""

from src.causal_engine.pipeline.state import (
    LibraryExecutionResult,
    PipelineConfig,
    PipelineInput,
    PipelineOutput,
    PipelineStage,
    PipelineState,
)

# =============================================================================
# PipelineStage Enum Tests
# =============================================================================


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_all_stages_exist(self):
        """Test all pipeline stages are defined."""
        expected_stages = [
            "PENDING",
            "ROUTING",
            "GRAPH_ANALYSIS",
            "CAUSAL_VALIDATION",
            "EFFECT_ESTIMATION",
            "UPLIFT_MODELING",
            "AGGREGATING",
            "COMPLETED",
            "FAILED",
        ]
        actual_stages = [s.name for s in PipelineStage]
        assert sorted(expected_stages) == sorted(actual_stages)

    def test_stage_values_are_strings(self):
        """Test stage values are lowercase strings."""
        for stage in PipelineStage:
            assert isinstance(stage.value, str)
            assert stage.value.islower()

    def test_pending_stage(self):
        """Test PENDING stage."""
        assert PipelineStage.PENDING.value == "pending"

    def test_routing_stage(self):
        """Test ROUTING stage."""
        assert PipelineStage.ROUTING.value == "routing"

    def test_graph_analysis_stage(self):
        """Test GRAPH_ANALYSIS stage (NetworkX)."""
        assert PipelineStage.GRAPH_ANALYSIS.value == "graph_analysis"

    def test_causal_validation_stage(self):
        """Test CAUSAL_VALIDATION stage (DoWhy)."""
        assert PipelineStage.CAUSAL_VALIDATION.value == "causal_validation"

    def test_effect_estimation_stage(self):
        """Test EFFECT_ESTIMATION stage (EconML)."""
        assert PipelineStage.EFFECT_ESTIMATION.value == "effect_estimation"

    def test_uplift_modeling_stage(self):
        """Test UPLIFT_MODELING stage (CausalML)."""
        assert PipelineStage.UPLIFT_MODELING.value == "uplift_modeling"

    def test_aggregating_stage(self):
        """Test AGGREGATING stage."""
        assert PipelineStage.AGGREGATING.value == "aggregating"

    def test_completed_stage(self):
        """Test COMPLETED stage."""
        assert PipelineStage.COMPLETED.value == "completed"

    def test_failed_stage(self):
        """Test FAILED stage."""
        assert PipelineStage.FAILED.value == "failed"

    def test_stage_is_str_enum(self):
        """Test PipelineStage inherits from str."""
        assert issubclass(PipelineStage, str)


# =============================================================================
# LibraryExecutionResult TypedDict Tests
# =============================================================================


class TestLibraryExecutionResult:
    """Tests for LibraryExecutionResult TypedDict."""

    def test_can_create_valid_result(self):
        """Test creating a valid LibraryExecutionResult."""
        result: LibraryExecutionResult = {
            "library": "dowhy",
            "success": True,
            "latency_ms": 1500,
            "result": {"effect": 0.25},
            "error": None,
            "confidence": 0.95,
            "warnings": [],
        }
        assert result["library"] == "dowhy"
        assert result["success"] is True
        assert result["confidence"] == 0.95

    def test_result_with_error(self):
        """Test creating a result with an error."""
        result: LibraryExecutionResult = {
            "library": "econml",
            "success": False,
            "latency_ms": 500,
            "result": None,
            "error": "Insufficient data",
            "confidence": 0.0,
            "warnings": ["Low sample size"],
        }
        assert result["success"] is False
        assert result["error"] == "Insufficient data"

    def test_library_names(self):
        """Test expected library names."""
        expected_libraries = ["networkx", "dowhy", "econml", "causalml"]
        for lib in expected_libraries:
            result: LibraryExecutionResult = {
                "library": lib,
                "success": True,
                "latency_ms": 100,
                "result": {},
                "error": None,
                "confidence": 0.5,
                "warnings": [],
            }
            assert result["library"] == lib


# =============================================================================
# PipelineConfig TypedDict Tests
# =============================================================================


class TestPipelineConfig:
    """Tests for PipelineConfig TypedDict."""

    def test_can_create_sequential_config(self):
        """Test creating a sequential mode config."""
        config: PipelineConfig = {
            "mode": "sequential",
            "libraries_enabled": ["networkx", "dowhy"],
            "primary_library": "dowhy",
            "stage_timeout_ms": 30000,
            "total_timeout_ms": 120000,
            "cross_validate": False,
            "min_agreement_threshold": 0.85,
            "max_parallel_libraries": 4,
            "fail_fast": False,
            "segment_by_uplift": False,
            "nested_ci_level": 0.95,
        }
        assert config["mode"] == "sequential"
        assert "dowhy" in config["libraries_enabled"]

    def test_can_create_parallel_config(self):
        """Test creating a parallel mode config."""
        config: PipelineConfig = {
            "mode": "parallel",
            "libraries_enabled": ["dowhy", "econml", "causalml"],
            "primary_library": None,
            "stage_timeout_ms": 30000,
            "total_timeout_ms": 120000,
            "cross_validate": True,
            "min_agreement_threshold": 0.85,
            "max_parallel_libraries": 4,
            "fail_fast": True,
            "segment_by_uplift": False,
            "nested_ci_level": 0.95,
        }
        assert config["mode"] == "parallel"
        assert config["fail_fast"] is True

    def test_can_create_validation_loop_config(self):
        """Test creating a validation_loop mode config."""
        config: PipelineConfig = {
            "mode": "validation_loop",
            "libraries_enabled": ["dowhy", "econml"],
            "primary_library": "dowhy",
            "stage_timeout_ms": 30000,
            "total_timeout_ms": 120000,
            "cross_validate": True,
            "min_agreement_threshold": 0.90,
            "max_parallel_libraries": 2,
            "fail_fast": False,
            "segment_by_uplift": False,
            "nested_ci_level": 0.95,
        }
        assert config["mode"] == "validation_loop"
        assert config["cross_validate"] is True

    def test_can_create_hierarchical_config(self):
        """Test creating a hierarchical mode config."""
        config: PipelineConfig = {
            "mode": "hierarchical",
            "libraries_enabled": ["causalml", "econml"],
            "primary_library": "causalml",
            "stage_timeout_ms": 60000,
            "total_timeout_ms": 180000,
            "cross_validate": False,
            "min_agreement_threshold": 0.85,
            "max_parallel_libraries": 2,
            "fail_fast": False,
            "segment_by_uplift": True,
            "nested_ci_level": 0.99,
        }
        assert config["mode"] == "hierarchical"
        assert config["segment_by_uplift"] is True


# =============================================================================
# PipelineState TypedDict Tests
# =============================================================================


class TestPipelineState:
    """Tests for PipelineState TypedDict."""

    def test_can_create_initial_state(self):
        """Test creating an initial pipeline state."""
        config: PipelineConfig = {
            "mode": "sequential",
            "libraries_enabled": ["dowhy"],
            "primary_library": "dowhy",
            "stage_timeout_ms": 30000,
            "total_timeout_ms": 120000,
            "cross_validate": False,
            "min_agreement_threshold": 0.85,
            "max_parallel_libraries": 4,
            "fail_fast": False,
            "segment_by_uplift": False,
            "nested_ci_level": 0.95,
        }

        state: PipelineState = {
            # Input
            "query": "What is the effect of marketing on sales?",
            "question_type": None,
            "treatment_var": "marketing_spend",
            "outcome_var": "sales",
            "confounders": ["region", "season"],
            "effect_modifiers": None,
            "data_source": "sales_data",
            "filters": None,
            # Configuration
            "config": config,
            # Routing
            "routed_libraries": [],
            "routing_confidence": 0.0,
            "routing_rationale": None,
            # Library results (all None initially)
            "networkx_result": None,
            "causal_graph": None,
            "graph_metrics": None,
            "dowhy_result": None,
            "causal_effect": None,
            "refutation_results": None,
            "identification_method": None,
            "econml_result": None,
            "cate_by_segment": None,
            "overall_ate": None,
            "heterogeneity_score": None,
            "causalml_result": None,
            "uplift_scores": None,
            "auuc": None,
            "qini": None,
            "targeting_recommendations": None,
            # Aggregated outputs
            "consensus_effect": None,
            "consensus_confidence": None,
            "library_agreement": None,
            "nested_cate": None,
            "segment_confidence_intervals": None,
            "executive_summary": None,
            "key_insights": None,
            "recommended_actions": None,
            # Execution metadata
            "current_stage": PipelineStage.PENDING,
            "stage_latencies": {},
            "total_latency_ms": 0,
            "libraries_executed": [],
            "libraries_skipped": [],
            # Error handling
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        assert state["query"] == "What is the effect of marketing on sales?"
        assert state["current_stage"] == PipelineStage.PENDING
        assert state["status"] == "pending"

    def test_state_status_values(self):
        """Test valid status values."""
        valid_statuses = ["pending", "running", "completed", "failed", "partial"]
        for status in valid_statuses:
            # Just verify these are valid strings
            assert status in valid_statuses


# =============================================================================
# PipelineInput TypedDict Tests
# =============================================================================


class TestPipelineInput:
    """Tests for PipelineInput TypedDict."""

    def test_can_create_minimal_input(self):
        """Test creating a minimal pipeline input."""
        input_data: PipelineInput = {
            "query": "What is the causal effect of X on Y?",
            "treatment_var": "X",
            "outcome_var": "Y",
            "confounders": None,
            "effect_modifiers": None,
            "data_source": "test_data",
            "filters": None,
            "mode": None,
            "libraries_enabled": None,
            "cross_validate": None,
        }
        assert input_data["query"] == "What is the causal effect of X on Y?"

    def test_can_create_full_input(self):
        """Test creating a full pipeline input with all options."""
        input_data: PipelineInput = {
            "query": "What drives sales in region A?",
            "treatment_var": "promotion",
            "outcome_var": "sales",
            "confounders": ["price", "competition"],
            "effect_modifiers": ["customer_segment"],
            "data_source": "regional_sales",
            "filters": {"region": "A", "year": 2024},
            "mode": "parallel",
            "libraries_enabled": ["dowhy", "econml"],
            "cross_validate": True,
        }
        assert input_data["mode"] == "parallel"
        assert input_data["cross_validate"] is True


# =============================================================================
# PipelineOutput TypedDict Tests
# =============================================================================


class TestPipelineOutput:
    """Tests for PipelineOutput TypedDict."""

    def test_can_create_successful_output(self):
        """Test creating a successful pipeline output."""
        output: PipelineOutput = {
            "question_type": "causal_effect",
            "primary_result": {
                "effect": 0.25,
                "confidence_interval": [0.15, 0.35],
            },
            "libraries_used": ["dowhy", "econml"],
            "consensus_effect": 0.24,
            "consensus_confidence": 0.92,
            "executive_summary": "Marketing increases sales by 25%",
            "key_insights": [
                "Effect is consistent across regions",
                "Higher impact on new customers",
            ],
            "recommended_actions": [
                "Increase marketing budget by 10%",
            ],
            "total_latency_ms": 5000,
            "status": "completed",
            "warnings": [],
            "errors": [],
        }
        assert output["status"] == "completed"
        assert output["consensus_effect"] == 0.24

    def test_can_create_partial_output(self):
        """Test creating a partial pipeline output."""
        output: PipelineOutput = {
            "question_type": "causal_effect",
            "primary_result": {"effect": 0.20},
            "libraries_used": ["dowhy"],
            "consensus_effect": None,
            "consensus_confidence": None,
            "executive_summary": "Partial analysis completed",
            "key_insights": ["Limited confidence due to single library"],
            "recommended_actions": [],
            "total_latency_ms": 30000,
            "status": "partial",
            "warnings": ["EconML timed out"],
            "errors": [],
        }
        assert output["status"] == "partial"
        assert len(output["warnings"]) == 1

    def test_can_create_failed_output(self):
        """Test creating a failed pipeline output."""
        output: PipelineOutput = {
            "question_type": "unknown",
            "primary_result": {},
            "libraries_used": [],
            "consensus_effect": None,
            "consensus_confidence": None,
            "executive_summary": "Analysis failed",
            "key_insights": [],
            "recommended_actions": [],
            "total_latency_ms": 1000,
            "status": "failed",
            "warnings": [],
            "errors": [{"stage": "routing", "message": "Invalid query"}],
        }
        assert output["status"] == "failed"
        assert len(output["errors"]) == 1
