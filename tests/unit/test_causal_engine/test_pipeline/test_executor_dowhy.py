"""Real-DoWhy wiring tests for `DoWhyExecutor` (phase C-2 of GH #354).

This file is RED-FIRST per the TDD protocol in `.claude/dispatch/354_executor_brief_template.md`.
On the placeholder body these assertions FAIL — they assert the executor:

1. CALLS `dowhy.CausalModel` against a real DataFrame from pipeline state
2. RETURNS DoWhy-derived outputs (identified estimand label from `model.identify_effect`,
   numeric `causal_effect` from `model.estimate_effect`) — NOT hardcoded
   `"backdoor"` / `0.0` placeholder values
3. FAILS CLOSED when no DataFrame is provided in state (no synthetic fallback,
   no hardcoded values, no all-zero `LibraryExecutionResult`)
4. FAILS CLOSED when DoWhy is unavailable
5. FAILS CLOSED when treatment/outcome columns are missing in the provided DataFrame
6. PROPAGATES warnings from the DoWhy run path

Mirrors the production-mature wrap-point pattern at
`src/agents/causal_impact/nodes/refutation.py:_reconstruct_dowhy_artifacts`
(`refutation.py:206-247`) and `causal_engine/refutation_runner.py:35`
(`from dowhy import CausalModel` — §0 V-03 of the dispatch plan).

Cross-refs:
- Dispatch plan: .claude/plans/354_dispatch_plan_v1.md §2.2 C-2
- Design plan: .claude/plans/causal_engine_canonical_routing_v4.md §1-§5
- Brief template: .claude/dispatch/354_executor_brief_template.md
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.pipeline.executors.dowhy import DoWhyExecutor
from src.causal_engine.pipeline.router import CausalLibrary
from src.causal_engine.pipeline.state import (
    PipelineConfig,
    PipelineStage,
    PipelineState,
)


# =============================================================================
# Helpers
# =============================================================================


def _build_pipeline_state(
    *,
    treatment_var: str | None = "treatment",
    outcome_var: str | None = "outcome",
    confounders: list[str] | None = None,
    filters: dict | None = None,
    causal_graph: dict | None = None,
) -> PipelineState:
    """Build a minimal PipelineState for DoWhy executor tests.

    The pipeline state contract does NOT carry an in-memory DataFrame field
    (state.py is locked in C-1). The conveyance channel is the existing
    ``filters: Optional[Dict[str, Any]]`` field, which the executor reads as
    its data-passthrough surface (mirrors the agents/causal_impact pattern of
    ``state['data_cache']['estimation_data']`` but routed through the only
    ``Dict[str, Any]`` field the locked TypedDict already exposes).
    """
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
    confounders = confounders if confounders is not None else ["confounder_a"]
    return PipelineState(
        query="Does treatment cause outcome?",
        question_type="causal_relationship",
        treatment_var=treatment_var,
        outcome_var=outcome_var,
        confounders=confounders,
        effect_modifiers=None,
        data_source="test_data",
        filters=filters,
        config=config,
        routed_libraries=["dowhy"],
        routing_confidence=0.9,
        routing_rationale="Test routing",
        networkx_result=None,
        causal_graph=causal_graph,
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


def _build_pipeline_config() -> PipelineConfig:
    """Convenience config matching the state's embedded config."""
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


def _build_real_dataframe(*, n: int = 400, true_ate: float = 1.5, seed: int = 13) -> pd.DataFrame:
    """Build a DataFrame with a known causal effect for asserting DoWhy ATE recovery.

    This is FIXTURE data assembled inside the TEST (not the executor body).
    The forbidden pattern is feeding seeded synthetic data to the real
    estimator from INSIDE the executor body. Tests are allowed (and expected)
    to construct DataFrames that the executor then consumes via state['filters'].
    """
    rng = np.random.default_rng(seed)
    confounder_a = rng.normal(0.0, 1.0, n)
    # Treatment depends on confounder (creates non-trivial estimand)
    treatment = 0.5 * confounder_a + rng.normal(0.0, 1.0, n)
    # Outcome depends linearly on treatment and confounder; ATE = true_ate
    outcome = true_ate * treatment + 0.7 * confounder_a + rng.normal(0.0, 1.0, n)
    return pd.DataFrame(
        {"treatment": treatment, "outcome": outcome, "confounder_a": confounder_a}
    )


# =============================================================================
# Tests
# =============================================================================


class TestDoWhyExecutorRealWiring:
    """C-2 real-DoWhy wrap assertions for `DoWhyExecutor.execute`."""

    @pytest.mark.asyncio
    async def test_execute_runs_real_dowhy_against_dataframe_in_filters(self) -> None:
        """A DataFrame in `state['filters']['estimation_data']` flows into `CausalModel`.

        Asserts the executor:
        - Returns `success=True`
        - Returns numeric finite `causal_effect` (NOT the hardcoded `0.0` placeholder)
        - Returns a non-empty string `identified_estimand` label produced by
          `model.identify_effect` (NOT the hardcoded `"backdoor"` constant)
        - Reports the resolved DoWhy `method_name` it actually used
        - Records latency from the real call (>0ms; DoWhy's identify+estimate is
          multi-millisecond minimum on a 400-row frame)
        """
        df = _build_real_dataframe()
        state = _build_pipeline_state(
            confounders=["confounder_a"],
            filters={"estimation_data": df},
        )
        config = _build_pipeline_config()
        executor = DoWhyExecutor()

        result = await executor.execute(state, config)

        assert result["library"] == "dowhy"
        assert result["success"] is True, f"Expected success=True; got error={result['error']!r}"
        assert result["error"] is None
        assert result["latency_ms"] > 0
        payload = result["result"]
        assert payload is not None, "Result payload must be populated on success"
        ce = payload.get("causal_effect")
        assert isinstance(ce, float)
        assert np.isfinite(ce), f"causal_effect must be finite numeric; got {ce!r}"
        # The DoWhy-estimated ATE should be within reasonable tolerance of the true ATE
        # (1.5). DoWhy with backdoor.linear_regression typically recovers ATE
        # well within 0.5 on this fixture; we use a 0.8 absolute tolerance to
        # avoid flake while still catching catastrophic mis-estimation.
        assert abs(ce - 1.5) < 0.8, f"DoWhy ATE estimate {ce} diverges far from true 1.5"
        # Identified estimand is the DoWhy-derived label, NOT the hardcoded "backdoor".
        estimand = payload.get("identified_estimand")
        assert isinstance(estimand, str)
        assert len(estimand) > 0
        # The executor must report which DoWhy method it actually ran.
        assert payload.get("dowhy_method") is not None
        # Confidence is library-specific; on a successful identify+estimate path it
        # MUST NOT be the hardcoded stub value `0.85`. The wrapper may choose any
        # confidence policy but cannot be the constant the placeholder used.
        assert result["confidence"] != 0.85, (
            "confidence=0.85 is the stub-shape constant; real wiring must derive "
            "confidence from the DoWhy run (or pin a different documented value)"
        )

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_no_dataframe_in_state(self) -> None:
        """No DataFrame anywhere in state ⇒ fail-closed (success=False, descriptive error).

        Forbidden patterns this test catches:
        - All-default / all-zero `LibraryExecutionResult` on data unavailability
          (`causal_effect == 0.0`, `identified_estimand == "backdoor"`)
        - Synthetic-data fabrication inside the executor body
        - Silent substitution of a different signal
        """
        state = _build_pipeline_state(filters=None)  # No data anywhere
        config = _build_pipeline_config()
        executor = DoWhyExecutor()

        result = await executor.execute(state, config)

        assert result["library"] == "dowhy"
        assert result["success"] is False
        assert result["error"] is not None
        # Error must be descriptive about WHY we failed.
        assert "data" in result["error"].lower() or "dataframe" in result["error"].lower()
        # On failure the executor MUST NOT silently return placeholder values.
        # If `result` payload is populated at all, it must not contain the stub
        # constants. Most fail-closed paths set result to None entirely, which
        # is the cleanest signal.
        if result["result"] is not None:
            assert result["result"].get("identified_estimand") != "backdoor"
            assert result["result"].get("causal_effect") != 0.0
        # Latency is still tracked (a meaningful 0+ value).
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_dowhy_unavailable(self, monkeypatch) -> None:
        """If `dowhy` import fails, executor fails closed instead of returning placeholder."""
        # Force the executor to think DoWhy is unavailable. We do this by
        # patching the module-level flag/symbol the executor uses to short-
        # circuit. The exact patch target is up to the GREEN implementation;
        # the assertion is on the OBSERVED behavior (success=False with a
        # descriptive error mentioning DoWhy / import).
        import src.causal_engine.pipeline.executors.dowhy as dowhy_module

        # Hide the dowhy package from the executor by setting a sentinel.
        # The GREEN implementation must consult either a DOWHY_AVAILABLE flag
        # or guard the import within `execute`.
        monkeypatch.setattr(dowhy_module, "DOWHY_AVAILABLE", False, raising=False)
        monkeypatch.setattr(dowhy_module, "CausalModel", None, raising=False)

        df = _build_real_dataframe()
        state = _build_pipeline_state(filters={"estimation_data": df})
        config = _build_pipeline_config()
        executor = DoWhyExecutor()

        result = await executor.execute(state, config)

        assert result["library"] == "dowhy"
        assert result["success"] is False
        assert result["error"] is not None
        assert "dowhy" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_dataframe_missing_treatment_column(self) -> None:
        """DataFrame missing the treatment column ⇒ fail-closed, no silent substitution."""
        df = _build_real_dataframe().drop(columns=["treatment"])
        state = _build_pipeline_state(
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=["confounder_a"],
            filters={"estimation_data": df},
        )
        config = _build_pipeline_config()
        executor = DoWhyExecutor()

        result = await executor.execute(state, config)

        assert result["success"] is False
        assert result["error"] is not None
        # Error must indicate the missing-column condition (not a generic stack trace
        # leak) so callers can act on the structured failure.
        err = result["error"].lower()
        assert "treatment" in err or "column" in err or "missing" in err

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_dataframe_missing_outcome_column(self) -> None:
        """DataFrame missing the outcome column ⇒ fail-closed."""
        df = _build_real_dataframe().drop(columns=["outcome"])
        state = _build_pipeline_state(
            filters={"estimation_data": df},
        )
        config = _build_pipeline_config()
        executor = DoWhyExecutor()

        result = await executor.execute(state, config)

        assert result["success"] is False
        assert result["error"] is not None
        err = result["error"].lower()
        assert "outcome" in err or "column" in err or "missing" in err

    @pytest.mark.asyncio
    async def test_execute_fails_closed_on_missing_treatment_var(self) -> None:
        """State missing `treatment_var` ⇒ fail-closed (matches validate_input semantic)."""
        df = _build_real_dataframe()
        state = _build_pipeline_state(
            treatment_var=None,
            filters={"estimation_data": df},
        )
        config = _build_pipeline_config()
        executor = DoWhyExecutor()

        result = await executor.execute(state, config)

        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_execute_fails_closed_on_missing_outcome_var(self) -> None:
        """State missing `outcome_var` ⇒ fail-closed."""
        df = _build_real_dataframe()
        state = _build_pipeline_state(
            outcome_var=None,
            filters={"estimation_data": df},
        )
        config = _build_pipeline_config()
        executor = DoWhyExecutor()

        result = await executor.execute(state, config)

        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_execute_uses_confounders_from_state_as_common_causes(self) -> None:
        """`state['confounders']` flow into DoWhy `CausalModel(common_causes=...)`.

        We assert this end-to-end by passing a DataFrame whose confounder column
        is required for identification, and confirming the executor produces
        a valid estimand (which DoWhy can only do if it sees the confounders).
        """
        df = _build_real_dataframe()
        state = _build_pipeline_state(
            confounders=["confounder_a"],
            filters={"estimation_data": df},
        )
        config = _build_pipeline_config()
        executor = DoWhyExecutor()

        result = await executor.execute(state, config)

        assert result["success"] is True, f"Got error: {result['error']!r}"
        payload = result["result"]
        assert payload is not None
        common_causes = payload.get("common_causes")
        assert isinstance(common_causes, list)
        assert "confounder_a" in common_causes

    def test_validate_input_unchanged_preserves_fail_closed_on_missing_treatment(self) -> None:
        """`validate_input` semantics preserved (locked by ABC contract)."""
        state = _build_pipeline_state(treatment_var=None)
        executor = DoWhyExecutor()
        is_valid, error = executor.validate_input(state)
        assert is_valid is False
        assert "treatment_var" in error

    def test_validate_input_unchanged_preserves_fail_closed_on_missing_outcome(self) -> None:
        """`validate_input` semantics preserved on missing outcome_var."""
        state = _build_pipeline_state(outcome_var=None)
        executor = DoWhyExecutor()
        is_valid, error = executor.validate_input(state)
        assert is_valid is False
        assert "outcome_var" in error

    def test_validate_input_passes_with_treatment_and_outcome(self) -> None:
        """validate_input passes with treatment + outcome (no DataFrame needed at validate time)."""
        state = _build_pipeline_state()
        executor = DoWhyExecutor()
        is_valid, error = executor.validate_input(state)
        assert is_valid is True
        assert error == ""

    def test_library_property_returns_dowhy_enum(self) -> None:
        """`library` property returns CausalLibrary.DOWHY (locked by ABC contract)."""
        executor = DoWhyExecutor()
        assert executor.library == CausalLibrary.DOWHY


class TestDoWhyExecutorReturnContract:
    """Pins LibraryExecutionResult TypedDict-shape conformance for both paths."""

    @pytest.mark.asyncio
    async def test_success_path_returns_typed_dict_keys(self) -> None:
        """Success path returns all 7 `LibraryExecutionResult` keys."""
        df = _build_real_dataframe()
        state = _build_pipeline_state(filters={"estimation_data": df})
        config = _build_pipeline_config()
        executor = DoWhyExecutor()
        result = await executor.execute(state, config)
        for key in (
            "library",
            "success",
            "latency_ms",
            "result",
            "error",
            "confidence",
            "warnings",
        ):
            assert key in result, f"Missing required LibraryExecutionResult key: {key}"

    @pytest.mark.asyncio
    async def test_fail_closed_path_returns_typed_dict_keys(self) -> None:
        """Fail-closed path also returns all 7 `LibraryExecutionResult` keys."""
        state = _build_pipeline_state(filters=None)
        config = _build_pipeline_config()
        executor = DoWhyExecutor()
        result = await executor.execute(state, config)
        for key in (
            "library",
            "success",
            "latency_ms",
            "result",
            "error",
            "confidence",
            "warnings",
        ):
            assert key in result, f"Missing required LibraryExecutionResult key: {key}"

    @pytest.mark.asyncio
    async def test_module_imports_real_dowhy_at_module_load_time(self) -> None:
        """The executor module imports `dowhy.CausalModel` (V-03 wrap point)."""
        # Reload module to capture current import state.
        mod = importlib.import_module("src.causal_engine.pipeline.executors.dowhy")
        # The module MUST attempt the real DoWhy import (success or graceful
        # ImportError fallback). The marker is the presence of either a
        # `CausalModel` symbol bound or a `DOWHY_AVAILABLE` flag indicating
        # the import was attempted.
        has_causal_model = getattr(mod, "CausalModel", None) is not None
        has_availability_flag = hasattr(mod, "DOWHY_AVAILABLE")
        assert has_causal_model or has_availability_flag, (
            "executors/dowhy.py must import `from dowhy import CausalModel` "
            "(mirroring V-03 wrap point at causal_engine/refutation_runner.py:35)"
        )
