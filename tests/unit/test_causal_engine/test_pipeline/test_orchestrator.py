"""
Tests for src/causal_engine/pipeline/orchestrator.py

Covers:
- NetworkXExecutor: library property, execute(), validate_input()
- DoWhyExecutor: library property, execute(), validate_input()
- EconMLExecutor: library property, execute(), validate_input()
- CausalMLExecutor: library property, execute(), validate_input()
- PipelineOrchestrator: initialization, state management, output creation, routing
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.causal_engine.pipeline.orchestrator import (
    LibraryExecutor,
    NetworkXExecutor,
    DoWhyExecutor,
    EconMLExecutor,
    CausalMLExecutor,
    PipelineOrchestrator,
)
from src.causal_engine.pipeline.router import (
    CausalLibrary,
    QuestionType,
    RoutingDecision,
    LibraryRouter,
)
from src.causal_engine.pipeline.state import (
    LibraryExecutionResult,
    PipelineConfig,
    PipelineInput,
    PipelineOutput,
    PipelineStage,
    PipelineState,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_pipeline_state() -> PipelineState:
    """Create a minimal PipelineState for testing."""
    config: PipelineConfig = {
        "mode": "sequential",
        "libraries_enabled": ["dowhy"],
        "primary_library": "dowhy",
        "stage_timeout_ms": 30000,
        "total_timeout_ms": 120000,
        "cross_validate": True,
        "min_agreement_threshold": 0.85,
        "max_parallel_libraries": 4,
        "fail_fast": False,
        "segment_by_uplift": False,
        "nested_ci_level": 0.95,
    }

    return PipelineState(
        query="Does marketing spend cause sales?",
        question_type="causal_relationship",
        treatment_var="marketing_spend",
        outcome_var="sales",
        confounders=["region", "season"],
        effect_modifiers=None,
        data_source="test_data",
        filters=None,
        config=config,
        routed_libraries=["dowhy"],
        routing_confidence=0.9,
        routing_rationale="Test routing",
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


@pytest.fixture
def minimal_pipeline_config() -> PipelineConfig:
    """Create a minimal PipelineConfig for testing."""
    return {
        "mode": "sequential",
        "libraries_enabled": ["dowhy"],
        "primary_library": "dowhy",
        "stage_timeout_ms": 30000,
        "total_timeout_ms": 120000,
        "cross_validate": True,
        "min_agreement_threshold": 0.85,
        "max_parallel_libraries": 4,
        "fail_fast": False,
        "segment_by_uplift": False,
        "nested_ci_level": 0.95,
    }


@pytest.fixture
def pipeline_input() -> PipelineInput:
    """Create a PipelineInput for testing."""
    return PipelineInput(
        query="Does marketing spend cause sales?",
        treatment_var="marketing_spend",
        outcome_var="sales",
        confounders=["region", "season"],
        effect_modifiers=None,
        data_source="test_data",
        filters=None,
        mode=None,
        libraries_enabled=None,
        cross_validate=None,
    )


@pytest.fixture
def routing_decision() -> RoutingDecision:
    """Create a RoutingDecision for testing."""
    return RoutingDecision(
        question_type=QuestionType.CAUSAL_RELATIONSHIP,
        primary_library=CausalLibrary.DOWHY,
        secondary_libraries=[CausalLibrary.NETWORKX],
        confidence=0.9,
        rationale="Matched causal relationship pattern",
        recommended_mode="validation_loop",
    )


# =============================================================================
# NetworkXExecutor Tests
# =============================================================================


class TestNetworkXExecutor:
    """Tests for NetworkXExecutor class."""

    def test_library_property(self):
        """Test library property returns NETWORKX."""
        executor = NetworkXExecutor()
        assert executor.library == CausalLibrary.NETWORKX

    @pytest.mark.asyncio
    async def test_execute_success_with_treatment_and_outcome(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute returns successful result with treatment and outcome vars."""
        executor = NetworkXExecutor()

        result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["library"] == "networkx"
        assert result["success"] is True
        assert result["latency_ms"] >= 0
        assert result["error"] is None
        assert result["confidence"] == 0.8
        assert "nodes" in result["result"]
        assert "edges" in result["result"]
        # Should include treatment and outcome in nodes
        assert "marketing_spend" in result["result"]["nodes"]
        assert "sales" in result["result"]["nodes"]
        # Should have edge from treatment to outcome
        assert len(result["result"]["edges"]) >= 1

    @pytest.mark.asyncio
    async def test_execute_success_with_confounders_only(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute with only confounders (no treatment/outcome)."""
        executor = NetworkXExecutor()
        state = minimal_pipeline_state.copy()
        state["treatment_var"] = None
        state["outcome_var"] = None
        state["confounders"] = ["region", "season", "market_size"]

        result = await executor.execute(state, minimal_pipeline_config)

        assert result["success"] is True
        assert "region" in result["result"]["nodes"]
        assert "season" in result["result"]["nodes"]
        assert "market_size" in result["result"]["nodes"]

    @pytest.mark.asyncio
    async def test_execute_handles_exception(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute handles exceptions gracefully."""
        executor = NetworkXExecutor()

        # Create state that will cause exception in the logic
        state = minimal_pipeline_state.copy()
        # Set confounders to a non-iterable to cause exception inside try block
        state["confounders"] = 123  # Not a list, will fail when trying to iterate

        result = await executor.execute(state, minimal_pipeline_config)

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0

    def test_validate_input_with_treatment_var(self, minimal_pipeline_state):
        """Test validate_input passes with treatment_var."""
        executor = NetworkXExecutor()

        is_valid, error = executor.validate_input(minimal_pipeline_state)

        assert is_valid is True
        assert error == ""

    def test_validate_input_with_confounders(self, minimal_pipeline_state):
        """Test validate_input passes with confounders only."""
        executor = NetworkXExecutor()
        state = minimal_pipeline_state.copy()
        state["treatment_var"] = None
        state["confounders"] = ["region", "season"]

        is_valid, error = executor.validate_input(state)

        assert is_valid is True
        assert error == ""

    def test_validate_input_fails_without_treatment_or_confounders(
        self, minimal_pipeline_state
    ):
        """Test validate_input fails without treatment_var or confounders."""
        executor = NetworkXExecutor()
        state = minimal_pipeline_state.copy()
        state["treatment_var"] = None
        state["confounders"] = None

        is_valid, error = executor.validate_input(state)

        assert is_valid is False
        assert "NetworkX requires treatment_var or confounders" in error


# =============================================================================
# DoWhyExecutor Tests
# =============================================================================


class TestDoWhyExecutor:
    """Tests for DoWhyExecutor class."""

    def test_library_property(self):
        """Test library property returns DOWHY."""
        executor = DoWhyExecutor()
        assert executor.library == CausalLibrary.DOWHY

    @pytest.mark.asyncio
    async def test_execute_success(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute returns successful result."""
        executor = DoWhyExecutor()

        result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["library"] == "dowhy"
        assert result["success"] is True
        assert result["latency_ms"] >= 0
        assert result["error"] is None
        assert result["confidence"] == 0.85
        assert "identified_estimand" in result["result"]
        assert "causal_effect" in result["result"]
        assert "confidence_interval" in result["result"]
        assert "refutation_results" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_uses_graph_from_networkx(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute notes when causal_graph is available from NetworkX."""
        executor = DoWhyExecutor()
        state = minimal_pipeline_state.copy()
        state["causal_graph"] = {"nodes": ["X", "Y"], "edges": [{"from": "X", "to": "Y"}]}

        result = await executor.execute(state, minimal_pipeline_config)

        assert result["success"] is True
        assert result["result"]["graph_source"] == "networkx"

    @pytest.mark.asyncio
    async def test_execute_handles_exception(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute handles exceptions gracefully."""
        executor = DoWhyExecutor()

        # Create a call counter that raises on second call (inside try block)
        call_count = {"n": 0}

        def time_side_effect():
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ValueError("DoWhy error")
            return 100.0

        with patch(
            "src.causal_engine.pipeline.orchestrator.time.time",
            side_effect=time_side_effect
        ):
            result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["success"] is False
        assert result["error"] == "DoWhy error"
        assert result["confidence"] == 0.0

    def test_validate_input_success(self, minimal_pipeline_state):
        """Test validate_input passes with treatment and outcome vars."""
        executor = DoWhyExecutor()

        is_valid, error = executor.validate_input(minimal_pipeline_state)

        assert is_valid is True
        assert error == ""

    def test_validate_input_fails_without_treatment_var(self, minimal_pipeline_state):
        """Test validate_input fails without treatment_var."""
        executor = DoWhyExecutor()
        state = minimal_pipeline_state.copy()
        state["treatment_var"] = None

        is_valid, error = executor.validate_input(state)

        assert is_valid is False
        assert "DoWhy requires treatment_var" in error

    def test_validate_input_fails_without_outcome_var(self, minimal_pipeline_state):
        """Test validate_input fails without outcome_var."""
        executor = DoWhyExecutor()
        state = minimal_pipeline_state.copy()
        state["outcome_var"] = None

        is_valid, error = executor.validate_input(state)

        assert is_valid is False
        assert "DoWhy requires outcome_var" in error


# =============================================================================
# EconMLExecutor Tests
# =============================================================================


class TestEconMLExecutor:
    """Tests for EconMLExecutor class."""

    def test_library_property(self):
        """Test library property returns ECONML."""
        executor = EconMLExecutor()
        assert executor.library == CausalLibrary.ECONML

    @pytest.mark.asyncio
    async def test_execute_success(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute returns successful result."""
        executor = EconMLExecutor()

        result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["library"] == "econml"
        assert result["success"] is True
        assert result["latency_ms"] >= 0
        assert result["error"] is None
        assert result["confidence"] == 0.82
        assert "estimator" in result["result"]
        assert "ate" in result["result"]
        assert "cate_by_segment" in result["result"]
        assert "heterogeneity_score" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_uses_causal_effect_from_dowhy(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute uses causal_effect from DoWhy when available."""
        executor = EconMLExecutor()
        state = minimal_pipeline_state.copy()
        state["causal_effect"] = 0.15

        result = await executor.execute(state, minimal_pipeline_config)

        assert result["success"] is True
        assert result["result"]["ate"] == 0.15

    @pytest.mark.asyncio
    async def test_execute_handles_exception(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute handles exceptions gracefully."""
        executor = EconMLExecutor()

        # Create a call counter that raises on second call (inside try block)
        call_count = {"n": 0}

        def time_side_effect():
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ValueError("EconML error")
            return 100.0

        with patch(
            "src.causal_engine.pipeline.orchestrator.time.time",
            side_effect=time_side_effect
        ):
            result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["success"] is False
        assert result["error"] == "EconML error"
        assert result["confidence"] == 0.0

    def test_validate_input_success(self, minimal_pipeline_state):
        """Test validate_input passes with treatment and outcome vars."""
        executor = EconMLExecutor()

        is_valid, error = executor.validate_input(minimal_pipeline_state)

        assert is_valid is True
        assert error == ""

    def test_validate_input_fails_without_treatment_var(self, minimal_pipeline_state):
        """Test validate_input fails without treatment_var."""
        executor = EconMLExecutor()
        state = minimal_pipeline_state.copy()
        state["treatment_var"] = None

        is_valid, error = executor.validate_input(state)

        assert is_valid is False
        assert "EconML requires treatment_var" in error

    def test_validate_input_fails_without_outcome_var(self, minimal_pipeline_state):
        """Test validate_input fails without outcome_var."""
        executor = EconMLExecutor()
        state = minimal_pipeline_state.copy()
        state["outcome_var"] = None

        is_valid, error = executor.validate_input(state)

        assert is_valid is False
        assert "EconML requires outcome_var" in error


# =============================================================================
# CausalMLExecutor Tests
# =============================================================================


class TestCausalMLExecutor:
    """Tests for CausalMLExecutor class."""

    def test_library_property(self):
        """Test library property returns CAUSALML."""
        executor = CausalMLExecutor()
        assert executor.library == CausalLibrary.CAUSALML

    @pytest.mark.asyncio
    async def test_execute_success(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute returns successful result."""
        executor = CausalMLExecutor()

        result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["library"] == "causalml"
        assert result["success"] is True
        assert result["latency_ms"] >= 0
        assert result["error"] is None
        assert result["confidence"] == 0.78
        assert "model" in result["result"]
        assert "auuc" in result["result"]
        assert "qini" in result["result"]
        assert "uplift_by_segment" in result["result"]
        assert "targeting_recommendations" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_notes_econml_comparison(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute notes when CATE from EconML is available."""
        executor = CausalMLExecutor()
        state = minimal_pipeline_state.copy()
        state["cate_by_segment"] = {"segment_A": 0.12, "segment_B": 0.08}

        result = await executor.execute(state, minimal_pipeline_config)

        assert result["success"] is True
        assert result["result"]["econml_comparison"] == "available"

    @pytest.mark.asyncio
    async def test_execute_handles_exception(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute handles exceptions gracefully."""
        executor = CausalMLExecutor()

        # Create a call counter that raises on second call (inside try block)
        call_count = {"n": 0}

        def time_side_effect():
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ValueError("CausalML error")
            return 100.0

        with patch(
            "src.causal_engine.pipeline.orchestrator.time.time",
            side_effect=time_side_effect
        ):
            result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["success"] is False
        assert result["error"] == "CausalML error"
        assert result["confidence"] == 0.0

    def test_validate_input_success(self, minimal_pipeline_state):
        """Test validate_input passes with treatment and outcome vars."""
        executor = CausalMLExecutor()

        is_valid, error = executor.validate_input(minimal_pipeline_state)

        assert is_valid is True
        assert error == ""

    def test_validate_input_fails_without_treatment_var(self, minimal_pipeline_state):
        """Test validate_input fails without treatment_var."""
        executor = CausalMLExecutor()
        state = minimal_pipeline_state.copy()
        state["treatment_var"] = None

        is_valid, error = executor.validate_input(state)

        assert is_valid is False
        assert "CausalML requires treatment_var" in error

    def test_validate_input_fails_without_outcome_var(self, minimal_pipeline_state):
        """Test validate_input fails without outcome_var."""
        executor = CausalMLExecutor()
        state = minimal_pipeline_state.copy()
        state["outcome_var"] = None

        is_valid, error = executor.validate_input(state)

        assert is_valid is False
        assert "CausalML requires outcome_var" in error


# =============================================================================
# PipelineOrchestrator Tests
# =============================================================================


class ConcreteOrchestrator(PipelineOrchestrator):
    """Concrete implementation for testing abstract base class."""

    async def execute(self, input_data: PipelineInput) -> PipelineOutput:
        """Simple execute implementation for testing."""
        routing_decision = await self.route(input_data["query"])
        state = self._create_initial_state(input_data, routing_decision)

        # Execute primary library
        primary_lib = routing_decision.primary_library
        if primary_lib in self.executors:
            result = await self.executors[primary_lib].execute(state, state["config"])
            state = self._update_state_with_result(state, primary_lib, result)

        return self._create_output(state)


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator class."""

    def test_init_with_defaults(self):
        """Test initialization with default router and executors."""
        orchestrator = ConcreteOrchestrator()

        assert orchestrator.router is not None
        assert isinstance(orchestrator.router, LibraryRouter)
        assert len(orchestrator.executors) == 4
        assert CausalLibrary.NETWORKX in orchestrator.executors
        assert CausalLibrary.DOWHY in orchestrator.executors
        assert CausalLibrary.ECONML in orchestrator.executors
        assert CausalLibrary.CAUSALML in orchestrator.executors

    def test_init_with_custom_router(self):
        """Test initialization with custom router."""
        custom_router = LibraryRouter()
        orchestrator = ConcreteOrchestrator(router=custom_router)

        assert orchestrator.router is custom_router

    def test_init_with_custom_executors(self):
        """Test initialization with custom executors."""
        custom_executors = {CausalLibrary.DOWHY: DoWhyExecutor()}
        orchestrator = ConcreteOrchestrator(executors=custom_executors)

        assert orchestrator.executors == custom_executors
        assert CausalLibrary.DOWHY in orchestrator.executors
        assert CausalLibrary.NETWORKX not in orchestrator.executors

    def test_default_executors(self):
        """Test _default_executors creates all four executor types."""
        orchestrator = ConcreteOrchestrator()
        executors = orchestrator._default_executors()

        assert len(executors) == 4
        assert isinstance(executors[CausalLibrary.NETWORKX], NetworkXExecutor)
        assert isinstance(executors[CausalLibrary.DOWHY], DoWhyExecutor)
        assert isinstance(executors[CausalLibrary.ECONML], EconMLExecutor)
        assert isinstance(executors[CausalLibrary.CAUSALML], CausalMLExecutor)

    def test_create_initial_state(self, pipeline_input, routing_decision):
        """Test _create_initial_state creates proper PipelineState."""
        orchestrator = ConcreteOrchestrator()

        state = orchestrator._create_initial_state(pipeline_input, routing_decision)

        # Check input fields
        assert state["query"] == pipeline_input["query"]
        assert state["treatment_var"] == pipeline_input["treatment_var"]
        assert state["outcome_var"] == pipeline_input["outcome_var"]
        assert state["confounders"] == pipeline_input["confounders"]
        assert state["data_source"] == pipeline_input["data_source"]

        # Check routing fields
        assert state["question_type"] == routing_decision.question_type.value
        assert state["routing_confidence"] == routing_decision.confidence
        assert state["routing_rationale"] == routing_decision.rationale

        # Check routed libraries
        assert "dowhy" in state["routed_libraries"]
        assert "networkx" in state["routed_libraries"]

        # Check config
        assert state["config"]["mode"] == routing_decision.recommended_mode
        assert state["config"]["primary_library"] == "dowhy"

        # Check initial state
        assert state["current_stage"] == PipelineStage.PENDING
        assert state["status"] == "pending"
        assert state["libraries_executed"] == []
        assert state["errors"] == []
        assert state["warnings"] == []

    def test_update_state_with_networkx_result(self, minimal_pipeline_state):
        """Test _update_state_with_result for NetworkX."""
        orchestrator = ConcreteOrchestrator()
        result: LibraryExecutionResult = {
            "library": "networkx",
            "success": True,
            "latency_ms": 100,
            "result": {
                "nodes": ["X", "Y"],
                "edges": [{"from": "X", "to": "Y"}],
                "centrality": {"X": 0.5, "Y": 0.5},
            },
            "error": None,
            "confidence": 0.8,
            "warnings": [],
        }

        updated_state = orchestrator._update_state_with_result(
            minimal_pipeline_state, CausalLibrary.NETWORKX, result
        )

        assert updated_state["networkx_result"] == result
        assert updated_state["causal_graph"] == result["result"]
        assert updated_state["graph_metrics"] == {"X": 0.5, "Y": 0.5}
        assert "networkx" in updated_state["libraries_executed"]
        assert updated_state["stage_latencies"]["networkx"] == 100

    def test_update_state_with_dowhy_result(self, minimal_pipeline_state):
        """Test _update_state_with_result for DoWhy."""
        orchestrator = ConcreteOrchestrator()
        result: LibraryExecutionResult = {
            "library": "dowhy",
            "success": True,
            "latency_ms": 200,
            "result": {
                "identified_estimand": "backdoor",
                "causal_effect": 0.15,
                "refutation_results": {"placebo": 0.01},
            },
            "error": None,
            "confidence": 0.85,
            "warnings": ["Some warning"],
        }

        updated_state = orchestrator._update_state_with_result(
            minimal_pipeline_state, CausalLibrary.DOWHY, result
        )

        assert updated_state["dowhy_result"] == result
        assert updated_state["causal_effect"] == 0.15
        assert updated_state["refutation_results"] == {"placebo": 0.01}
        assert updated_state["identification_method"] == "backdoor"
        assert "dowhy" in updated_state["libraries_executed"]
        assert "Some warning" in updated_state["warnings"]

    def test_update_state_with_econml_result(self, minimal_pipeline_state):
        """Test _update_state_with_result for EconML."""
        orchestrator = ConcreteOrchestrator()
        result: LibraryExecutionResult = {
            "library": "econml",
            "success": True,
            "latency_ms": 300,
            "result": {
                "ate": 0.15,
                "cate_by_segment": {"A": 0.12, "B": 0.18},
                "heterogeneity_score": 0.3,
            },
            "error": None,
            "confidence": 0.82,
            "warnings": [],
        }

        updated_state = orchestrator._update_state_with_result(
            minimal_pipeline_state, CausalLibrary.ECONML, result
        )

        assert updated_state["econml_result"] == result
        assert updated_state["overall_ate"] == 0.15
        assert updated_state["cate_by_segment"] == {"A": 0.12, "B": 0.18}
        assert updated_state["heterogeneity_score"] == 0.3
        assert "econml" in updated_state["libraries_executed"]

    def test_update_state_with_causalml_result(self, minimal_pipeline_state):
        """Test _update_state_with_result for CausalML."""
        orchestrator = ConcreteOrchestrator()
        result: LibraryExecutionResult = {
            "library": "causalml",
            "success": True,
            "latency_ms": 250,
            "result": {
                "auuc": 0.65,
                "qini": 0.45,
                "uplift_by_segment": {"high": 0.2, "low": 0.05},
                "targeting_recommendations": [{"segment": "high", "action": "target"}],
            },
            "error": None,
            "confidence": 0.78,
            "warnings": [],
        }

        updated_state = orchestrator._update_state_with_result(
            minimal_pipeline_state, CausalLibrary.CAUSALML, result
        )

        assert updated_state["causalml_result"] == result
        assert updated_state["auuc"] == 0.65
        assert updated_state["qini"] == 0.45
        assert updated_state["uplift_scores"] == {"high": 0.2, "low": 0.05}
        assert updated_state["targeting_recommendations"] == [{"segment": "high", "action": "target"}]
        assert "causalml" in updated_state["libraries_executed"]

    def test_update_state_with_failed_result(self, minimal_pipeline_state):
        """Test _update_state_with_result handles failed results."""
        orchestrator = ConcreteOrchestrator()
        result: LibraryExecutionResult = {
            "library": "dowhy",
            "success": False,
            "latency_ms": 50,
            "result": None,
            "error": "Model fitting failed",
            "confidence": 0.0,
            "warnings": [],
        }

        updated_state = orchestrator._update_state_with_result(
            minimal_pipeline_state, CausalLibrary.DOWHY, result
        )

        assert updated_state["dowhy_result"] == result
        assert "dowhy" in updated_state["libraries_executed"]
        assert len(updated_state["errors"]) == 1
        assert updated_state["errors"][0]["library"] == "dowhy"
        assert updated_state["errors"][0]["error"] == "Model fitting failed"

    def test_create_output_completed(self, minimal_pipeline_state):
        """Test _create_output for completed pipeline."""
        orchestrator = ConcreteOrchestrator()

        # Set up successful DoWhy result
        state = minimal_pipeline_state.copy()
        state["dowhy_result"] = {
            "library": "dowhy",
            "success": True,
            "latency_ms": 200,
            "result": {"causal_effect": 0.15},
            "error": None,
            "confidence": 0.85,
            "warnings": [],
        }
        state["libraries_executed"] = ["dowhy"]
        state["total_latency_ms"] = 200
        state["executive_summary"] = "Marketing causes sales increase"
        state["key_insights"] = ["Effect is significant"]
        state["recommended_actions"] = ["Increase marketing budget"]

        output = orchestrator._create_output(state)

        assert output["question_type"] == "causal_relationship"
        assert output["primary_result"] == {"causal_effect": 0.15}
        assert output["libraries_used"] == ["dowhy"]
        assert output["status"] == "completed"
        assert output["total_latency_ms"] == 200
        assert output["errors"] == []

    def test_create_output_failed(self, minimal_pipeline_state):
        """Test _create_output for failed pipeline."""
        orchestrator = ConcreteOrchestrator()

        state = minimal_pipeline_state.copy()
        state["libraries_executed"] = []
        state["errors"] = [{"library": "dowhy", "error": "Failed"}]

        output = orchestrator._create_output(state)

        assert output["status"] == "failed"
        assert len(output["errors"]) == 1

    def test_create_output_partial(self, minimal_pipeline_state):
        """Test _create_output for partial completion (some libraries failed)."""
        orchestrator = ConcreteOrchestrator()

        state = minimal_pipeline_state.copy()
        state["dowhy_result"] = {
            "library": "dowhy",
            "success": True,
            "latency_ms": 200,
            "result": {"causal_effect": 0.15},
            "error": None,
            "confidence": 0.85,
            "warnings": [],
        }
        state["libraries_executed"] = ["dowhy"]
        state["errors"] = [{"library": "econml", "error": "CATE estimation failed"}]

        output = orchestrator._create_output(state)

        assert output["status"] == "partial"
        assert output["libraries_used"] == ["dowhy"]
        assert len(output["errors"]) == 1

    @pytest.mark.asyncio
    async def test_route_delegates_to_router(self):
        """Test route method delegates to router."""
        mock_router = MagicMock(spec=LibraryRouter)
        mock_router.route.return_value = RoutingDecision(
            question_type=QuestionType.CAUSAL_RELATIONSHIP,
            primary_library=CausalLibrary.DOWHY,
        )

        orchestrator = ConcreteOrchestrator(router=mock_router)

        result = await orchestrator.route("Does X cause Y?", force_libraries=["dowhy"])

        mock_router.route.assert_called_once_with("Does X cause Y?", force_libraries=["dowhy"])
        assert result.question_type == QuestionType.CAUSAL_RELATIONSHIP

    @pytest.mark.asyncio
    async def test_execute_full_pipeline(self, pipeline_input):
        """Test full execute workflow."""
        orchestrator = ConcreteOrchestrator()

        output = await orchestrator.execute(pipeline_input)

        assert output["question_type"] in [
            "causal_relationship",
            "effect_heterogeneity",
            "targeting_optimization",
            "impact_flow",
            "comprehensive",
            "unknown",
        ]
        assert output["status"] in ["completed", "partial", "failed"]
        assert isinstance(output["libraries_used"], list)
        assert isinstance(output["total_latency_ms"], int)


# =============================================================================
# Integration Tests
# =============================================================================


class TestOrchestratorIntegration:
    """Integration tests for orchestrator components."""

    @pytest.mark.asyncio
    async def test_executor_chain_propagates_state(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test that results from one executor can be used by another."""
        # Execute NetworkX first
        networkx_executor = NetworkXExecutor()
        nx_result = await networkx_executor.execute(
            minimal_pipeline_state, minimal_pipeline_config
        )

        # Update state with NetworkX result
        orchestrator = ConcreteOrchestrator()
        state = orchestrator._update_state_with_result(
            minimal_pipeline_state, CausalLibrary.NETWORKX, nx_result
        )

        # Execute DoWhy - should see the causal_graph
        dowhy_executor = DoWhyExecutor()
        dw_result = await dowhy_executor.execute(state, minimal_pipeline_config)

        # DoWhy should note that graph came from NetworkX
        assert dw_result["success"] is True
        assert "graph_source" in dw_result["result"]
        assert dw_result["result"]["graph_source"] == "networkx"

    @pytest.mark.asyncio
    async def test_econml_uses_dowhy_effect(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test that EconML uses causal_effect from DoWhy."""
        orchestrator = ConcreteOrchestrator()

        # Simulate DoWhy result
        dowhy_result: LibraryExecutionResult = {
            "library": "dowhy",
            "success": True,
            "latency_ms": 200,
            "result": {"causal_effect": 0.25, "identified_estimand": "backdoor"},
            "error": None,
            "confidence": 0.85,
            "warnings": [],
        }

        state = orchestrator._update_state_with_result(
            minimal_pipeline_state, CausalLibrary.DOWHY, dowhy_result
        )

        # Execute EconML
        econml_executor = EconMLExecutor()
        ecn_result = await econml_executor.execute(state, minimal_pipeline_config)

        # EconML should use the causal_effect value
        assert ecn_result["success"] is True
        assert ecn_result["result"]["ate"] == 0.25

    @pytest.mark.asyncio
    async def test_causalml_sees_econml_cate(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test that CausalML notes CATE from EconML."""
        orchestrator = ConcreteOrchestrator()

        # Simulate EconML result
        econml_result: LibraryExecutionResult = {
            "library": "econml",
            "success": True,
            "latency_ms": 300,
            "result": {
                "ate": 0.15,
                "cate_by_segment": {"high": 0.2, "low": 0.1},
                "heterogeneity_score": 0.3,
            },
            "error": None,
            "confidence": 0.82,
            "warnings": [],
        }

        state = orchestrator._update_state_with_result(
            minimal_pipeline_state, CausalLibrary.ECONML, econml_result
        )

        # Execute CausalML
        causalml_executor = CausalMLExecutor()
        cml_result = await causalml_executor.execute(state, minimal_pipeline_config)

        # CausalML should note EconML comparison is available
        assert cml_result["success"] is True
        assert cml_result["result"]["econml_comparison"] == "available"
