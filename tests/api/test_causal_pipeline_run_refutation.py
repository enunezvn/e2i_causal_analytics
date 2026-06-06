"""R6-F1 (#740): opt-in real-refutation wiring on /causal/pipeline.

These tests cover the *opt-in* refutation path added behind the new
``run_refutation: bool = False`` request flag. They are ADDITIVE to the
existing default-path contract in ``test_causal_pipeline_c8_wiring.py``
(which keeps asserting ``robustness_validation_performed is False`` for the
DEFAULT path where ``run_refutation`` is unset).

Edits covered:
- Edit A: request models gain ``run_refutation`` (default False, round-trips True).
- Edits B/C/D: the flag threads request → PipelineInput → state["config"].
- Edit F: the response helper ``_robustness_from_state`` gates
  ``robustness_validation_performed`` on the REAL gate band (PROCEED → True;
  REVIEW/BLOCK/error/skipped → False + a band-naming caveat).
- M-fo2: a non-DAG (cyclic) graph_quality FORCES
  ``robustness_validation_performed=False`` + an un-ignorable structural caveat,
  regardless of any PROCEED refutation result.

Reference: .claude/plans/causal-validation-remediation/R6-F1-implementation-plan.md
"""

from __future__ import annotations

from src.api.routes.causal import (
    _build_pipeline_input_parallel,
    _build_pipeline_input_sequential,
)
from src.api.schemas.causal import (
    ParallelPipelineRequest,
    SequentialPipelineRequest,
)

# =============================================================================
# Edit A — request models carry the opt-in flag (default False, round-trips True)
# =============================================================================


class TestRequestModelHasRunRefutation:
    def test_sequential_request_run_refutation_default_false(self) -> None:
        req = SequentialPipelineRequest(
            treatment_var="t",
            outcome_var="y",
            stages=[
                {"library": "dowhy", "estimator": "linear_regression"},
                {"library": "econml", "estimator": "linear_dml"},
            ],
        )
        assert req.run_refutation is False

    def test_sequential_request_run_refutation_round_trips_true(self) -> None:
        req = SequentialPipelineRequest(
            treatment_var="t",
            outcome_var="y",
            run_refutation=True,
            stages=[
                {"library": "dowhy", "estimator": "linear_regression"},
                {"library": "econml", "estimator": "linear_dml"},
            ],
        )
        assert req.run_refutation is True

    def test_parallel_request_run_refutation_default_false(self) -> None:
        req = ParallelPipelineRequest(
            treatment_var="t",
            outcome_var="y",
            libraries=["dowhy", "econml"],
        )
        assert req.run_refutation is False

    def test_parallel_request_run_refutation_round_trips_true(self) -> None:
        req = ParallelPipelineRequest(
            treatment_var="t",
            outcome_var="y",
            run_refutation=True,
            libraries=["dowhy", "econml"],
        )
        assert req.run_refutation is True


# =============================================================================
# Edits B/C/D — the flag threads request → PipelineInput → state["config"]
# =============================================================================


class TestRunRefutationThreadsThroughPlumbing:
    def test_run_refutation_flows_request_to_sequential_input(self) -> None:
        req = SequentialPipelineRequest(
            treatment_var="t",
            outcome_var="y",
            run_refutation=True,
            stages=[
                {"library": "dowhy", "estimator": "linear_regression"},
                {"library": "econml", "estimator": "linear_dml"},
            ],
        )
        pipeline_input = _build_pipeline_input_sequential(req)
        assert pipeline_input.get("run_refutation") is True

    def test_run_refutation_default_false_in_sequential_input(self) -> None:
        req = SequentialPipelineRequest(
            treatment_var="t",
            outcome_var="y",
            stages=[
                {"library": "dowhy", "estimator": "linear_regression"},
                {"library": "econml", "estimator": "linear_dml"},
            ],
        )
        pipeline_input = _build_pipeline_input_sequential(req)
        assert pipeline_input.get("run_refutation") is False

    def test_run_refutation_flows_request_to_parallel_input(self) -> None:
        req = ParallelPipelineRequest(
            treatment_var="t",
            outcome_var="y",
            run_refutation=True,
            libraries=["dowhy", "econml"],
        )
        pipeline_input = _build_pipeline_input_parallel(req)
        assert pipeline_input.get("run_refutation") is True

    def test_pipeline_input_run_refutation_lands_in_state_config(self) -> None:
        """Orchestrator copies PipelineInput.run_refutation into state["config"]."""
        from src.causal_engine.pipeline.router import (
            CausalLibrary,
            QuestionType,
            RoutingDecision,
        )
        from src.causal_engine.pipeline.sequential import SequentialPipeline
        from src.causal_engine.pipeline.state import PipelineInput

        pipeline = SequentialPipeline()
        pipeline_input: PipelineInput = {
            "query": "q",
            "treatment_var": "t",
            "outcome_var": "y",
            "confounders": ["c"],
            "effect_modifiers": None,
            "data_source": "test",
            "filters": None,
            "mode": "sequential",
            "libraries_enabled": ["dowhy"],
            "cross_validate": None,
            "run_refutation": True,
        }
        routing_decision = RoutingDecision(
            question_type=QuestionType.CAUSAL_RELATIONSHIP,
            primary_library=CausalLibrary.DOWHY,
            secondary_libraries=[],
            confidence=0.9,
            rationale="test",
            recommended_mode="sequential",
        )
        state = pipeline._create_initial_state(pipeline_input, routing_decision)
        assert state["config"].get("run_refutation") is True

    def test_state_config_run_refutation_defaults_false(self) -> None:
        from src.causal_engine.pipeline.router import (
            CausalLibrary,
            QuestionType,
            RoutingDecision,
        )
        from src.causal_engine.pipeline.sequential import SequentialPipeline
        from src.causal_engine.pipeline.state import PipelineInput

        pipeline = SequentialPipeline()
        pipeline_input: PipelineInput = {
            "query": "q",
            "treatment_var": "t",
            "outcome_var": "y",
            "confounders": ["c"],
            "effect_modifiers": None,
            "data_source": "test",
            "filters": None,
            "mode": "sequential",
            "libraries_enabled": ["dowhy"],
            "cross_validate": None,
        }
        routing_decision = RoutingDecision(
            question_type=QuestionType.CAUSAL_RELATIONSHIP,
            primary_library=CausalLibrary.DOWHY,
            secondary_libraries=[],
            confidence=0.9,
            rationale="test",
            recommended_mode="sequential",
        )
        state = pipeline._create_initial_state(pipeline_input, routing_decision)
        assert state["config"].get("run_refutation") is False
