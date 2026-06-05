"""M-fo2: the sequential/parallel pipeline response must surface the structured
graph-quality signal (is_dag / structural_quality) so a consumer can programmatically
detect a non-DAG, not only read a free-text warning.

NOTE ON BUILDER SIGNATURE (deviation from the shard's illustrative snippet):
    The real builders are
        _sequential_output_to_response(pipeline_id, request, output, *, state=None)
        _parallel_output_to_response(pipeline_id, request, output, *, state=None)
    (verified against src/api/routes/causal.py), NOT the single-arg form the shard
    sketched. So this test constructs a minimal request and calls them with the real
    3-arg signature. The synthetic ``output`` carries libraries_used=["networkx","dowhy"]
    (dowhy is a data-required library that "succeeded") so the builders' internal
    _enforce_data_required_fail_close passes and we reach the response constructor —
    the only thing under test here. No HTTP, no heavy DoWhy/EconML run.
"""

import pytest

from src.api.routes import causal as causal_module
from src.api.schemas.causal import (
    ParallelPipelineRequest,
    PipelineStageConfig,
    SequentialPipelineRequest,
)

pytestmark = pytest.mark.integration


def _output_with_cyclic_graph():
    # Minimal PipelineOutput-shaped dict the response builder consumes.
    return {
        "question_type": "effect_estimation",
        "primary_result": {"ate": 1.23},
        "libraries_used": ["networkx", "dowhy"],
        "errors": [],
        "consensus_effect": 1.23,
        "consensus_confidence": 0.4,
        "library_agreement_score": None,
        "total_latency_ms": 5,
        "warnings": [
            "NetworkX graph contains cycles; downstream causal-effect "
            "estimators may misidentify backdoor paths"
        ],
        "graph_quality": {
            "n_nodes": 3,
            "n_edges": 3,
            "is_dag": False,
            "has_treatment_outcome_path": True,
            "structural_quality": 0.0,
            "n_cycles": 1,
        },
    }


def _sequential_request():
    return SequentialPipelineRequest(
        treatment_var="treatment",
        outcome_var="outcome",
        stages=[
            PipelineStageConfig(library="networkx"),
            PipelineStageConfig(library="dowhy", estimator="propensity_score_matching"),
        ],
    )


def _parallel_request():
    return ParallelPipelineRequest(
        treatment_var="treatment",
        outcome_var="outcome",
        libraries=["networkx", "dowhy"],
    )


def test_sequential_response_surfaces_graph_quality():
    resp = causal_module._sequential_output_to_response(
        "pipe_test", _sequential_request(), _output_with_cyclic_graph()
    )
    assert resp.graph_is_dag is False
    assert resp.structural_quality == 0.0
    # the existing human-readable warning is still present
    assert any("cycle" in w.lower() for w in resp.warnings)


def test_parallel_response_surfaces_graph_quality():
    resp = causal_module._parallel_output_to_response(
        "pipe_test", _parallel_request(), _output_with_cyclic_graph()
    )
    assert resp.graph_is_dag is False
    assert resp.structural_quality == 0.0
    assert any("cycle" in w.lower() for w in resp.warnings)
