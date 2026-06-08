"""
Tests for src/causal_engine/pipeline/orchestrator.py

Covers:
- NetworkXExecutor: library property, execute(), validate_input()
- DoWhyExecutor: library property, execute(), validate_input()
- EconMLExecutor: library property, execute(), validate_input()
- CausalMLExecutor: library property, execute(), validate_input()
- PipelineOrchestrator: initialization, state management, output creation, routing
"""

from unittest.mock import MagicMock, patch

import pytest

from src.causal_engine.pipeline.orchestrator import (
    CausalMLExecutor,
    DoWhyExecutor,
    EconMLExecutor,
    NetworkXExecutor,
    PipelineOrchestrator,
)
from src.causal_engine.pipeline.router import (
    CausalLibrary,
    LibraryRouter,
    QuestionType,
    RoutingDecision,
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
        """Test execute returns successful result with treatment and outcome vars.

        Updated in phase C-5 of GH #354: confidence is now derived from graph
        structure (1.0 well-formed DAG with treatment-outcome path and >=3 nodes;
        0.5 limited; 0.0 cyclic), NOT the placeholder's hardcoded 0.8.
        See .claude/plans/354_c5_networkx_design_spike.md §2.1 for rationale.
        With treatment=marketing_spend + outcome=sales + 2 confounders, the
        backdoor pattern produces a 4-node DAG with shortest path length 1
        => confidence = 1.0.
        """
        executor = NetworkXExecutor()

        result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["library"] == "networkx"
        assert result["success"] is True
        assert result["latency_ms"] >= 0
        assert result["error"] is None
        # C-5: structural confidence — well-formed DAG with treatment-outcome path
        assert result["confidence"] == 1.0
        assert result["result"]["is_dag"] is True
        assert result["result"]["has_treatment_outcome_path"] is True
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
    async def test_execute_handles_exception(self, minimal_pipeline_state, minimal_pipeline_config):
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

    def test_validate_input_fails_without_treatment_or_confounders(self, minimal_pipeline_state):
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
    async def test_execute_fails_closed_without_dataframe(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Execute fails closed when no DataFrame is in state['filters'].

        UPDATED in C-2 (PR for #354): the prior assertion pinned the C-1
        placeholder body (`success=True`, `confidence=0.85`, hardcoded
        `identified_estimand="backdoor"` / `causal_effect=0.0`). C-2 wires
        the executor to real `dowhy.CausalModel`; without a DataFrame it
        MUST fail-closed instead of returning placeholder values. The
        real-DataFrame success path is asserted in `test_executor_dowhy.py`.
        """
        executor = DoWhyExecutor()

        result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["library"] == "dowhy"
        assert result["success"] is False, (
            "Without a DataFrame in state['filters'], DoWhyExecutor must fail-closed; "
            "the previous success-asserting test pinned C-1's placeholder body."
        )
        assert result["error"] is not None
        # Result payload is None on fail-closed (no placeholder values).
        assert result["result"] is None
        # Confidence is zero on failure path (no fake confidence value).
        assert result["confidence"] == 0.0
        # Latency is still tracked.
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_with_graph_from_networkx_still_fails_without_dataframe(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Even with a NetworkX `causal_graph` in state, execute still needs a DataFrame.

        UPDATED in C-2 (PR for #354): the prior assertion pinned the
        placeholder's `graph_source: "networkx"` side-effect when a graph
        was present. Real DoWhy needs the DataFrame regardless of whether
        an upstream NetworkX graph is available; `graph_source` is now a
        success-path bookkeeping field, not a substitute for real data.
        """
        executor = DoWhyExecutor()
        state = minimal_pipeline_state.copy()
        state["causal_graph"] = {"nodes": ["X", "Y"], "edges": [{"from": "X", "to": "Y"}]}

        result = await executor.execute(state, minimal_pipeline_config)

        # Still fail-closed — graph alone is insufficient; DoWhy needs the data.
        assert result["success"] is False
        assert result["error"] is not None
        assert result["result"] is None

    @pytest.mark.asyncio
    async def test_execute_handles_dowhy_exception(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Execute returns fail-closed structured error when DoWhy raises.

        UPDATED in C-2 (PR for #354): the prior assertion used a
        `time.time`-monkeypatch trick that depended on the placeholder
        body's two-time-call shape. Real wiring uses a different code
        path. We assert the equivalent functional invariant: when any
        underlying DoWhy call raises during identify/estimate, the
        executor catches it and returns `success=False` with the error
        preserved in `result['error']` — never silently substitutes a
        placeholder value.
        """
        executor = DoWhyExecutor()
        # Provide a DataFrame matching the minimal_pipeline_state fixture's
        # treatment_var=marketing_spend, outcome_var=sales so we reach the
        # CausalModel call path, then force CausalModel construction itself
        # to raise by patching it module-level.
        import pandas as pd

        df = pd.DataFrame(
            {
                "marketing_spend": [0.0, 1.0, 0.0, 1.0],
                "sales": [1.0, 2.0, 1.0, 2.0],
                "region": [0, 0, 1, 1],
            }
        )
        state = minimal_pipeline_state.copy()
        state["filters"] = {"estimation_data": df}
        # minimal_pipeline_state's confounders default to ["region", "season"];
        # narrow to ["region"] so the present DataFrame satisfies the column
        # check (season is intentionally absent — separate test covers the
        # missing-column path explicitly).
        state["confounders"] = ["region"]

        # Force CausalModel to raise during construction.
        with patch(
            "src.causal_engine.pipeline.executors.dowhy.CausalModel",
            side_effect=ValueError("DoWhy boom"),
        ):
            result = await executor.execute(state, minimal_pipeline_config)

        assert result["success"] is False
        assert result["error"] is not None
        assert "DoWhy boom" in result["error"], (
            f"Expected underlying exception preserved in error; got {result['error']!r}"
        )
        assert result["confidence"] == 0.0
        assert result["result"] is None

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
    """Tests for EconMLExecutor class.

    Phase C-3 (GH #354) rewired the executor body from a placeholder returning
    hardcoded ``confidence=0.82`` / ``ate=0.0`` (and silently using
    ``state['causal_effect']`` from DoWhy as a substitute ATE) to a real wrap of
    ``src.causal_engine.energy_score.estimator_selector.EstimatorSelector``. The
    happy-path + fail-closed contract details live in
    ``test_executor_econml.py``; these tests pin only the ABC contract surface
    + fail-closed-default behavior on the legacy ``minimal_pipeline_state``
    fixture (which does NOT carry a ``data_cache.estimation_data``, hence the
    executor MUST fail-closed).
    """

    def test_library_property(self):
        """Test library property returns ECONML."""
        executor = EconMLExecutor()
        assert executor.library == CausalLibrary.ECONML

    @pytest.mark.asyncio
    async def test_execute_fails_closed_without_data_backend(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Without ``data_cache.estimation_data`` the executor MUST fail-closed.

        Pre-C-3 shape returned ``success=True`` with ``ate=0.0`` (silent
        fabrication); post-C-3 the executor refuses to invent ATE/CATE without
        a real DataFrame. See ``test_executor_econml.py`` for the dependency-
        injected happy-path coverage.
        """
        executor = EconMLExecutor()

        result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["library"] == "econml"
        assert result["success"] is False
        assert result["latency_ms"] >= 0
        assert result["error"] is not None
        # Confidence is 0.0 on fail-closed (NOT the legacy 0.82 placeholder).
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_execute_ignores_dowhy_causal_effect_when_no_data(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Pre-C-3 the executor copied ``state['causal_effect']`` into its own
        ATE — silent fabrication of heterogeneity from a different stage's
        single ATE. Post-C-3 the executor fails-closed without a real
        DataFrame, regardless of what DoWhy wrote upstream.
        """
        executor = EconMLExecutor()
        state = minimal_pipeline_state.copy()
        state["causal_effect"] = 0.15  # NOT used as ATE substitute anymore

        result = await executor.execute(state, minimal_pipeline_config)

        # No fabrication of ate=0.15; fail-closed instead.
        assert result["success"] is False
        assert result["result"] is None or "ate" not in (result["result"] or {})

    @pytest.mark.asyncio
    async def test_execute_handles_unexpected_exception(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """A truly unexpected exception inside ``execute()`` MUST be caught and
        surfaced as ``success=False`` with the message in ``error``. Replaces
        the legacy ``time.time`` patch trick (which only worked because the
        old body called ``time.time()`` exactly twice — the new body has a
        different control-flow shape).
        """
        executor = EconMLExecutor()
        # Inject a state-extension attribute that makes the executor blow up.
        # We use a property-accessor trick: a dict-like that raises on get.

        class _ExplosiveDict(dict):
            def get(self, key, default=None):  # type: ignore[override]
                if key == "data_cache":
                    raise RuntimeError("EconML error")
                return super().get(key, default)

        bad_state = _ExplosiveDict(minimal_pipeline_state)
        result = await executor.execute(bad_state, minimal_pipeline_config)  # type: ignore[arg-type]

        assert result["success"] is False
        assert "EconML error" in result["error"]
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
    """Tests for CausalMLExecutor class.

    NOTE: phase C-4 of GH #354 (2026-05-22) rewired `CausalMLExecutor.execute()`
    to call the real CausalML library via `src.causal_engine.uplift.*`. The
    pre-C-4 behavior — returning `auuc=0.0, qini=0.0, confidence=0.78` with
    `success=True` regardless of input — was a SCAFFOLDED PLACEHOLDER and is
    no longer the contract. Post-C-4:

    * When `state["filters"]["dataframe"]` is missing or unusable, the executor
      fails closed (`success=False, result=None, confidence=0.0`) — no synthetic
      data fallback. Tests that previously pinned the placeholder all-zero
      success path are updated here to assert the fail-closed semantics.
    * Real-library success-path assertions (auuc/qini/ate finite, model in known
      set) live in `tests/unit/test_causal_engine/test_pipeline/test_executor_causalml.py`
      and are marked `slow` because fitting a real CausalML model on the test
      fixture takes seconds.
    """

    def test_library_property(self):
        """Test library property returns CAUSALML."""
        executor = CausalMLExecutor()
        assert executor.library == CausalLibrary.CAUSALML

    @pytest.mark.asyncio
    async def test_execute_fails_closed_without_dataframe(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Post-C-4: without `filters["dataframe"]`, executor fails closed.

        Replaces the pre-C-4 `test_execute_success` which pinned the stub's
        all-zero success path (`auuc=0.0, qini=0.0, confidence=0.78`). The
        minimal_pipeline_state fixture has `filters=None`, so the executor
        must NOT fall back to synthetic data — it must return success=False.
        """
        executor = CausalMLExecutor()

        result = await executor.execute(minimal_pipeline_state, minimal_pipeline_config)

        assert result["library"] == "causalml"
        # Was: success=True (stub silently returned zeros).
        # Now: success=False, fail-closed on missing data.
        assert result["success"] is False
        assert result["error"] is not None
        assert "data" in (result["error"] or "").lower()
        assert result["result"] is None
        # Was: confidence=0.78 (hardcoded stub).
        # Now: confidence=0.0 on failure.
        assert result["confidence"] == 0.0
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_handles_unexpected_exception(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test execute handles unexpected exceptions gracefully (not data-unavailable).

        Pre-C-4 version of this test patched `time.time` to raise on the second
        call, exercising the catch-all `except Exception` branch of the stub
        body. Post-C-4, the catch-all branch still exists for unexpected
        failures (e.g. an uplift backend bug), distinct from the
        ExecutorDataUnavailable fail-closed path. To exercise it deterministically
        without depending on the order of time.time() calls, we patch the data
        extractor itself to raise a non-ExecutorDataUnavailable exception.
        """
        executor = CausalMLExecutor()

        with patch(
            "src.causal_engine.pipeline.executors.causalml._extract_uplift_inputs_from_state",
            side_effect=ValueError("CausalML error"),
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
        """Test _update_state_with_result for NetworkX.

        Updated phase C-6: also asserts ``graph_quality`` extraction from
        the rich post-C-5 NetworkX payload (n_nodes, n_edges, is_dag,
        has_treatment_outcome_path, structural_quality).
        """
        orchestrator = ConcreteOrchestrator()
        result: LibraryExecutionResult = {
            "library": "networkx",
            "success": True,
            "latency_ms": 100,
            "result": {
                "nodes": ["X", "Y"],
                "edges": [{"from": "X", "to": "Y"}],
                "centrality": {"X": 0.5, "Y": 0.5},
                # C-6: post-C-5 NetworkX payload also includes structural fields
                "n_nodes": 2,
                "n_edges": 1,
                "is_dag": True,
                "has_treatment_outcome_path": False,
                "cycles": [],
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
        # C-6: graph_quality channel populated from structural payload
        assert updated_state["graph_quality"] is not None
        assert updated_state["graph_quality"]["n_nodes"] == 2
        assert updated_state["graph_quality"]["is_dag"] is True
        # n_nodes<3 => structural_quality=0.5 even with DAG
        assert updated_state["graph_quality"]["structural_quality"] == 0.5
        # M-fo2 (precise): a DAG is acyclic, never review-gated.
        assert updated_state["graph_quality"]["structural_identification"] == "acyclic"
        assert updated_state["graph_quality"]["requires_structural_review"] is False
        assert updated_state["graph_quality"]["cycle_affects_identification"] is False

    def _networkx_result_with_cycle(self, *, cycle_affects: bool) -> "LibraryExecutionResult":
        """A NetworkX payload representing a non-DAG (cycle present), parameterized by
        whether the cycle sits on the (T,Y) ancestral subgraph (M-fo2 precise gate)."""
        return {
            "library": "networkx",
            "success": True,
            "latency_ms": 100,
            "result": {
                "nodes": ["T", "Y", "A", "B", "C"],
                "edges": [{"from": "T", "to": "Y"}],
                "centrality": {},
                "n_nodes": 5,
                "n_edges": 5,
                "is_dag": False,
                "has_treatment_outcome_path": True,
                "cycles": [["A", "B", "C"]],
                "cycle_affects_identification": cycle_affects,
                "cycles_on_relevant_subgraph": [["A", "B", "C"]] if cycle_affects else [],
                "orientation_ambiguity_only": False,
            },
            "error": None,
            "confidence": 0.0 if cycle_affects else 1.0,
            "warnings": [],
        }

    def test_graph_quality_irrelevant_cycle_no_penalty(self, minimal_pipeline_state):
        """M-fo2 (precise): a cycle OFF the (T,Y) ancestral subgraph leaves the estimand
        identifiable -> structural_quality stays 1.0 (no haircut) and the result is NOT
        review-gated."""
        orchestrator = ConcreteOrchestrator()
        result = self._networkx_result_with_cycle(cycle_affects=False)
        updated_state = orchestrator._update_state_with_result(
            minimal_pipeline_state, CausalLibrary.NETWORKX, result
        )
        gq = updated_state["graph_quality"]
        assert gq["is_dag"] is False
        assert gq["cycle_affects_identification"] is False
        # has_path + n_nodes>=3 and identification not blocked => full quality.
        assert gq["structural_quality"] == 1.0
        assert gq["structural_identification"] == "cycle_irrelevant"
        assert gq["requires_structural_review"] is False

    def test_graph_quality_relevant_cycle_zeroes_and_flags_review(self, minimal_pipeline_state):
        """M-fo2 (precise): a cycle ON the (T,Y) ancestral subgraph makes backdoor
        adjustment undefined -> structural_quality 0.0 AND a review/quarantine flag."""
        orchestrator = ConcreteOrchestrator()
        result = self._networkx_result_with_cycle(cycle_affects=True)
        updated_state = orchestrator._update_state_with_result(
            minimal_pipeline_state, CausalLibrary.NETWORKX, result
        )
        gq = updated_state["graph_quality"]
        assert gq["is_dag"] is False
        assert gq["cycle_affects_identification"] is True
        assert gq["structural_quality"] == 0.0
        assert gq["structural_identification"] == "undefined_cyclic"
        assert gq["requires_structural_review"] is True

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
        # C-6: DoWhy registers as ATE-track contributor
        assert updated_state["library_metric_types"] is not None
        assert updated_state["library_metric_types"].get("dowhy") == "ate"

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
        # C-6: EconML registers as ATE-track contributor
        assert updated_state["library_metric_types"] is not None
        assert updated_state["library_metric_types"].get("econml") == "ate"

    def test_update_state_with_causalml_result(self, minimal_pipeline_state):
        """Test _update_state_with_result for CausalML.

        Updated phase C-6: reads the post-C-4 key ``uplift_scores_summary``
        (the pre-C-4 key ``uplift_by_segment`` is no longer produced by the
        real CausalML executor — see
        ``executors/causalml.py::_summarize_uplift_scores``). C-6 also
        extracts an ``uplift_summary`` channel and records CausalML's
        metric type in ``library_metric_types``.
        """
        orchestrator = ConcreteOrchestrator()
        result: LibraryExecutionResult = {
            "library": "causalml",
            "success": True,
            "latency_ms": 250,
            "result": {
                "model": "uplift_random_forest",
                "ate": 0.18,
                "ate_ci_lower": 0.10,
                "ate_ci_upper": 0.26,
                "auuc": 0.65,
                "qini": 0.45,
                "uplift_scores_summary": {
                    "n": 1000,
                    "mean": 0.18,
                    "std": 0.30,
                    "min": -0.5,
                    "max": 0.75,
                    "median": 0.18,
                    "p10": -0.2,
                    "p90": 0.55,
                },
                "n_samples": 1000,
                "treatment_groups": ["1"],
                "observed_treatment_groups": ["0", "1"],
                "control_name": "0",
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
        # Post-C-6: uplift_scores reads the post-C-4 key uplift_scores_summary
        assert updated_state["uplift_scores"] is not None
        assert updated_state["uplift_scores"]["mean"] == 0.18
        assert updated_state["uplift_scores"]["n"] == 1000
        assert updated_state["targeting_recommendations"] == [
            {"segment": "high", "action": "target"}
        ]
        assert "causalml" in updated_state["libraries_executed"]
        # C-6: uplift_summary channel must be populated
        assert updated_state["uplift_summary"] is not None
        assert updated_state["uplift_summary"]["auuc"] == 0.65
        assert updated_state["uplift_summary"]["qini"] == 0.45
        assert updated_state["uplift_summary"]["ate"] == 0.18
        # C-6: library_metric_types records CausalML as ATE-track
        assert updated_state["library_metric_types"] is not None
        assert updated_state["library_metric_types"].get("causalml") == "ate"

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
        """Test that results from one executor can be used by another.

        UPDATED in C-2 (PR for #354): the prior assertion pinned DoWhy's
        placeholder `success=True` + `graph_source: "networkx"` stub
        side-effect. Real DoWhy needs a DataFrame to succeed. We now
        provide one and assert the chain produces a real causal_effect
        from DoWhy after NetworkX populates the causal_graph.
        """
        import numpy as np
        import pandas as pd

        # Build a small DataFrame with a known effect so DoWhy can run.
        rng = np.random.default_rng(7)
        n = 200
        region = rng.normal(0.0, 1.0, n)
        treatment = 0.4 * region + rng.normal(0.0, 1.0, n)
        outcome = 1.2 * treatment + 0.5 * region + rng.normal(0.0, 1.0, n)
        df = pd.DataFrame({"marketing_spend": treatment, "sales": outcome, "region": region})

        # Seed the state with the DataFrame for DoWhy.
        state = minimal_pipeline_state.copy()
        state["filters"] = {"estimation_data": df}
        state["confounders"] = ["region"]

        # Execute NetworkX first.
        networkx_executor = NetworkXExecutor()
        nx_result = await networkx_executor.execute(state, minimal_pipeline_config)

        # Update state with NetworkX result.
        orchestrator = ConcreteOrchestrator()
        state = orchestrator._update_state_with_result(state, CausalLibrary.NETWORKX, nx_result)

        # Execute DoWhy - should see the causal_graph + run the real model.
        dowhy_executor = DoWhyExecutor()
        dw_result = await dowhy_executor.execute(state, minimal_pipeline_config)

        # Real DoWhy must succeed with a finite causal_effect; `graph_source`
        # is still a success-path bookkeeping field.
        assert dw_result["success"] is True, (
            f"DoWhy should succeed with real data; got error: {dw_result['error']!r}"
        )
        assert isinstance(dw_result["result"]["causal_effect"], float)
        assert np.isfinite(dw_result["result"]["causal_effect"])
        assert dw_result["result"]["graph_source"] == "networkx"

    @pytest.mark.asyncio
    async def test_econml_does_not_silently_use_dowhy_effect(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Phase C-3 (#354): pre-rewire shape silently copied DoWhy's
        ``causal_effect`` into EconML's ``ate``, fabricating heterogeneity
        from a single ATE. The new shape requires a real DataFrame in
        ``state['data_cache']['estimation_data']`` and fail-closed otherwise.

        This is a regression-guard: do NOT regress to silent DoWhy-leak shape.
        """
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

        # Execute EconML WITHOUT injecting data_cache. The executor must
        # fail-closed; it must NOT use 0.25 as its ATE.
        econml_executor = EconMLExecutor()
        ecn_result = await econml_executor.execute(state, minimal_pipeline_config)

        assert ecn_result["success"] is False
        # No silent fabrication: result body MUST NOT contain ate==0.25.
        body = ecn_result["result"] or {}
        assert body.get("ate") != 0.25

    @pytest.mark.asyncio
    async def test_causalml_fails_closed_without_data_even_with_econml_state(
        self, minimal_pipeline_state, minimal_pipeline_config
    ):
        """Test that CausalML fails closed when no DataFrame is supplied, even
        when an EconML result has populated `cate_by_segment` upstream.

        Replaces pre-C-4 `test_causalml_sees_econml_cate` which pinned the
        stub's "EconML state is sufficient to produce uplift result" behavior
        (set `econml_comparison: "available"` and returned success=True with
        all-zero uplift fields). Post-C-4, EconML state does NOT substitute
        for real uplift modeling — without the DataFrame, executor fails
        closed regardless of upstream state.

        Cross-executor state propagation that DEPENDS on real uplift output
        (e.g., uplift_scores informing C-6's consensus aggregator) is covered
        by the `slow`-marked tests in test_executor_causalml.py.
        """
        orchestrator = ConcreteOrchestrator()

        # Simulate EconML result populating cate_by_segment.
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

        # Execute CausalML — should fail closed because no DataFrame.
        causalml_executor = CausalMLExecutor()
        cml_result = await causalml_executor.execute(state, minimal_pipeline_config)

        # Post-C-4: fail closed; no silent substitution from EconML state.
        assert cml_result["success"] is False
        assert cml_result["result"] is None
        assert cml_result["error"] is not None
