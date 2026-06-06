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

from typing import Any, Optional

from src.api.routes import causal as causal_module
from src.api.routes.causal import (
    _build_pipeline_input_parallel,
    _build_pipeline_input_sequential,
)
from src.api.schemas.causal import (
    ParallelPipelineRequest,
    PipelineStageConfig,
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


# =============================================================================
# Edit F — _robustness_from_state gates the response flag on the REAL gate band
# =============================================================================


def _make_output(
    *,
    refutation_results: Optional[dict] = None,
    graph_quality: Optional[dict] = None,
) -> dict:
    """Minimal PipelineOutput-shaped dict the response builders consume.

    libraries_used carries a data-required library ("dowhy") that "succeeded"
    (not in errors) so _enforce_data_required_fail_close passes and we reach the
    response constructor — the surface under test. graph_quality, when provided,
    is passed via ``output`` (the existing M-fo2 surface contract).
    """
    out: dict = {
        "question_type": "effect_estimation",
        "primary_result": {"causal_effect": 1.23, "refutation_results": refutation_results or {}},
        "libraries_used": ["dowhy"],
        "errors": [],
        "consensus_effect": 1.23,
        "consensus_confidence": 0.8,
        "library_agreement_score": None,
        "total_latency_ms": 5,
        "warnings": [],
    }
    if graph_quality is not None:
        out["graph_quality"] = graph_quality
    return out


def _make_state(
    *,
    refutation_results: Optional[dict] = None,
    graph_quality: Optional[dict] = None,
) -> dict[str, Any]:
    """Minimal state-shaped mapping carrying refutation_results / graph_quality."""
    st: dict[str, Any] = {
        "refutation_results": refutation_results,
        "library_agreement_score": None,
    }
    if graph_quality is not None:
        st["graph_quality"] = graph_quality
    return st


def _seq_request() -> SequentialPipelineRequest:
    return SequentialPipelineRequest(
        treatment_var="treatment",
        outcome_var="outcome",
        stages=[
            PipelineStageConfig(library="dowhy", estimator="linear_regression"),
            PipelineStageConfig(library="econml", estimator="linear_dml"),
        ],
    )


def _par_request() -> ParallelPipelineRequest:
    return ParallelPipelineRequest(
        treatment_var="treatment",
        outcome_var="outcome",
        libraries=["dowhy", "econml"],
    )


class TestRobustnessFromState:
    def test_proceed_sets_validation_true_no_warning(self) -> None:
        perf, warn = causal_module._robustness_from_state(
            _make_state(refutation_results={"gate_decision": "proceed"})
        )
        assert perf is True
        assert warn is None

    def test_review_sets_false_with_review_caveat(self) -> None:
        perf, warn = causal_module._robustness_from_state(
            _make_state(refutation_results={"gate_decision": "review"})
        )
        assert perf is False
        assert warn is not None
        assert "review" in warn.lower()

    def test_block_sets_false_with_block_caveat(self) -> None:
        perf, warn = causal_module._robustness_from_state(
            _make_state(refutation_results={"gate_decision": "block"})
        )
        assert perf is False
        assert warn is not None
        assert "block" in warn.lower()

    def test_error_sets_false(self) -> None:
        perf, warn = causal_module._robustness_from_state(
            _make_state(refutation_results={"error": "boom", "gate_decision": "block"})
        )
        assert perf is False
        assert warn is not None

    def test_skipped_falls_back_to_unvalidated(self) -> None:
        perf, warn = causal_module._robustness_from_state(
            _make_state(refutation_results={"skipped": True, "reason": "no SE"})
        )
        assert perf is False
        assert warn == causal_module._ROBUSTNESS_UNVALIDATED_WARNING

    def test_empty_falls_back_to_unvalidated(self) -> None:
        perf, warn = causal_module._robustness_from_state(_make_state(refutation_results={}))
        assert perf is False
        assert warn == causal_module._ROBUSTNESS_UNVALIDATED_WARNING

    def test_none_state_falls_back_to_unvalidated(self) -> None:
        perf, warn = causal_module._robustness_from_state(None)
        assert perf is False
        assert warn == causal_module._ROBUSTNESS_UNVALIDATED_WARNING


class TestResponseGatingOnRealSuite:
    def test_sequential_proceed_flips_validation_true(self) -> None:
        resp = causal_module._sequential_output_to_response(
            "pipe_test",
            _seq_request(),
            _make_output(refutation_results={"gate_decision": "proceed"}),
            state=_make_state(refutation_results={"gate_decision": "proceed"}),
        )
        assert resp.robustness_validation_performed is True
        assert resp.robustness_warning is None
        # When validated, the "unvalidated" caveat must NOT pollute warnings.
        assert not any("unvalidated" in w.lower() for w in resp.warnings)

    def test_sequential_review_stays_false_with_caveat(self) -> None:
        resp = causal_module._sequential_output_to_response(
            "pipe_test",
            _seq_request(),
            _make_output(refutation_results={"gate_decision": "review"}),
            state=_make_state(refutation_results={"gate_decision": "review"}),
        )
        assert resp.robustness_validation_performed is False
        assert resp.robustness_warning is not None
        assert "review" in resp.robustness_warning.lower()
        assert any("review" in w.lower() for w in resp.warnings)

    def test_sequential_block_stays_false(self) -> None:
        resp = causal_module._sequential_output_to_response(
            "pipe_test",
            _seq_request(),
            _make_output(refutation_results={"gate_decision": "block"}),
            state=_make_state(refutation_results={"gate_decision": "block"}),
        )
        assert resp.robustness_validation_performed is False
        assert resp.robustness_warning is not None
        assert "block" in resp.robustness_warning.lower()

    def test_sequential_default_empty_stays_unvalidated(self) -> None:
        """No refutation (default path) → False + the R2 unvalidated caveat."""
        resp = causal_module._sequential_output_to_response(
            "pipe_test",
            _seq_request(),
            _make_output(refutation_results={}),
            state=_make_state(refutation_results={}),
        )
        assert resp.robustness_validation_performed is False
        assert resp.robustness_warning == causal_module._ROBUSTNESS_UNVALIDATED_WARNING

    def test_parallel_proceed_flips_validation_true(self) -> None:
        resp = causal_module._parallel_output_to_response(
            "pipe_test",
            _par_request(),
            _make_output(refutation_results={"gate_decision": "proceed"}),
            state=_make_state(refutation_results={"gate_decision": "proceed"}),
        )
        assert resp.robustness_validation_performed is True
        assert resp.robustness_warning is None

    def test_parallel_review_stays_false_with_caveat(self) -> None:
        resp = causal_module._parallel_output_to_response(
            "pipe_test",
            _par_request(),
            _make_output(refutation_results={"gate_decision": "review"}),
            state=_make_state(refutation_results={"gate_decision": "review"}),
        )
        assert resp.robustness_validation_performed is False
        assert resp.robustness_warning is not None
        assert "review" in resp.robustness_warning.lower()


# =============================================================================
# M-fo2 — a non-DAG (cyclic) graph FORCES robustness False + structural caveat,
# even when refutation PROCEEDed.
# =============================================================================


_NON_DAG_GRAPH_QUALITY = {
    "n_nodes": 3,
    "n_edges": 3,
    "is_dag": False,
    "has_treatment_outcome_path": True,
    "structural_quality": 0.0,
    "n_cycles": 1,
}


class TestNonDagStructuralGate:
    def test_sequential_non_dag_forces_false_despite_proceed(self) -> None:
        resp = causal_module._sequential_output_to_response(
            "pipe_test",
            _seq_request(),
            _make_output(
                refutation_results={"gate_decision": "proceed"},
                graph_quality=_NON_DAG_GRAPH_QUALITY,
            ),
            state=_make_state(
                refutation_results={"gate_decision": "proceed"},
                graph_quality=_NON_DAG_GRAPH_QUALITY,
            ),
        )
        # M-fo2: a cyclic graph is NOT identification-valid; refutation PROCEED
        # must NOT be allowed to mark it robust.
        assert resp.robustness_validation_performed is False
        assert resp.robustness_warning is not None
        # The un-ignorable structural caveat appears in BOTH the warning + list.
        assert "cycle" in resp.robustness_warning.lower()
        assert any("cycle" in w.lower() for w in resp.warnings)
        # The structured signal is still surfaced.
        assert resp.graph_is_dag is False
        assert resp.structural_quality == 0.0

    def test_parallel_non_dag_forces_false_despite_proceed(self) -> None:
        resp = causal_module._parallel_output_to_response(
            "pipe_test",
            _par_request(),
            _make_output(
                refutation_results={"gate_decision": "proceed"},
                graph_quality=_NON_DAG_GRAPH_QUALITY,
            ),
            state=_make_state(
                refutation_results={"gate_decision": "proceed"},
                graph_quality=_NON_DAG_GRAPH_QUALITY,
            ),
        )
        assert resp.robustness_validation_performed is False
        assert resp.robustness_warning is not None
        assert "cycle" in resp.robustness_warning.lower()
        assert any("cycle" in w.lower() for w in resp.warnings)
        assert resp.graph_is_dag is False
        assert resp.structural_quality == 0.0

    def test_sequential_dag_proceed_still_validates(self) -> None:
        """A valid DAG with PROCEED must still validate (gate is non-DAG-specific)."""
        dag_quality = {**_NON_DAG_GRAPH_QUALITY, "is_dag": True, "structural_quality": 1.0}
        resp = causal_module._sequential_output_to_response(
            "pipe_test",
            _seq_request(),
            _make_output(
                refutation_results={"gate_decision": "proceed"},
                graph_quality=dag_quality,
            ),
            state=_make_state(
                refutation_results={"gate_decision": "proceed"},
                graph_quality=dag_quality,
            ),
        )
        assert resp.robustness_validation_performed is True
        assert resp.graph_is_dag is True
