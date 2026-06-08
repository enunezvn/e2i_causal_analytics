"""M-fo2 (precise identifiability gate): the sequential/parallel pipeline response
must distinguish a cycle that actually breaks backdoor identification of the
(treatment, outcome) estimand from one that does not.

- ``undefined_cyclic`` — a directed cycle on the ``An({T,Y}) ∪ {T,Y}`` ancestral
  subgraph makes backdoor adjustment undefined. The response WITHHOLDS the
  consensus effect (consensus_effect=None), sets ``requires_review=True`` and
  ``structural_identification="undefined_cyclic"``, and forces
  ``robustness_validation_performed=False`` even over a PROCEED refutation band.
- ``cycle_irrelevant`` — a cycle exists somewhere in the graph but OFF the
  ancestral subgraph; the estimand stays identifiable, so the consensus effect is
  PRESERVED, ``requires_review=False``, and the cycle does NOT force robustness
  False (it is gated only on the real refutation band).
- ``acyclic`` — a DAG; no structural review.

Conservative fail-closed default: a non-DAG graph_quality dict that LACKS the
precise ``cycle_affects_identification`` field (e.g. injected via the synthetic
``output`` path) is treated as ``undefined_cyclic``.

NOTE ON BUILDER SIGNATURE: the real builders are
    _sequential_output_to_response(pipeline_id, request, output, *, state=None)
    _parallel_output_to_response(pipeline_id, request, output, *, state=None)
(verified against src/api/routes/causal.py). The synthetic ``output`` carries
libraries_used=["networkx","dowhy"] so _enforce_data_required_fail_close passes
and we reach the response constructor — the only thing under test. No HTTP run.
"""

import pytest

from src.api.routes import causal as causal_module
from src.api.schemas.causal import (
    ParallelPipelineRequest,
    PipelineStageConfig,
    SequentialPipelineRequest,
)

pytestmark = pytest.mark.integration

_CYCLE_WARNING = (
    "NetworkX graph contains cycles; downstream causal-effect "
    "estimators may misidentify backdoor paths"
)


def _output(consensus_effect=1.23, warnings=None):
    """Minimal PipelineOutput-shaped dict the response builder consumes."""
    return {
        "question_type": "effect_estimation",
        "primary_result": {"ate": consensus_effect},
        "libraries_used": ["networkx", "dowhy"],
        "errors": [],
        "consensus_effect": consensus_effect,
        "consensus_confidence": 0.4,
        "library_agreement_score": None,
        "total_latency_ms": 5,
        "warnings": list(warnings) if warnings else [],
    }


def _graph_quality(*, is_dag, structural_quality, cycle_affects=None):
    gq = {
        "n_nodes": 3,
        "n_edges": 3,
        "is_dag": is_dag,
        "has_treatment_outcome_path": True,
        "structural_quality": structural_quality,
        "n_cycles": 0 if is_dag else 1,
    }
    if cycle_affects is not None:
        gq["cycle_affects_identification"] = cycle_affects
        gq["structural_identification"] = (
            "acyclic" if is_dag else ("undefined_cyclic" if cycle_affects else "cycle_irrelevant")
        )
        gq["requires_structural_review"] = bool((is_dag is False) and cycle_affects)
    return gq


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


# --------------------------------------------------------------------------- #
# undefined_cyclic — cycle on the ancestral subgraph: withhold + quarantine
# --------------------------------------------------------------------------- #


def test_sequential_undefined_cyclic_withholds_consensus_and_flags_review():
    state = {
        "graph_quality": _graph_quality(is_dag=False, structural_quality=0.0, cycle_affects=True)
    }
    resp = causal_module._sequential_output_to_response(
        "pipe_test",
        _sequential_request(),
        _output(consensus_effect=1.23, warnings=[_CYCLE_WARNING]),
        state=state,
    )
    assert resp.graph_is_dag is False
    assert resp.structural_quality == 0.0
    assert resp.structural_identification == "undefined_cyclic"
    assert resp.requires_review is True
    # Backdoor adjustment is undefined -> the consensus number is WITHHELD.
    assert resp.consensus_effect is None
    assert resp.robustness_validation_performed is False
    assert any("cycle" in w.lower() for w in resp.warnings)


def test_parallel_undefined_cyclic_withholds_consensus_and_flags_review():
    state = {
        "graph_quality": _graph_quality(is_dag=False, structural_quality=0.0, cycle_affects=True)
    }
    resp = causal_module._parallel_output_to_response(
        "pipe_test",
        _parallel_request(),
        _output(consensus_effect=1.23, warnings=[_CYCLE_WARNING]),
        state=state,
    )
    assert resp.graph_is_dag is False
    assert resp.structural_identification == "undefined_cyclic"
    assert resp.requires_review is True
    assert resp.consensus_effect is None
    assert resp.robustness_validation_performed is False


def test_undefined_cyclic_forces_robustness_false_despite_proceed():
    """Even a PROCEED refutation band cannot rescue an unidentified (cyclic) estimand."""
    state = {
        "graph_quality": _graph_quality(is_dag=False, structural_quality=0.0, cycle_affects=True),
        "refutation_results": {"gate_decision": "proceed"},
    }
    resp = causal_module._sequential_output_to_response(
        "pipe_test", _sequential_request(), _output(), state=state
    )
    assert resp.robustness_validation_performed is False
    assert resp.requires_review is True
    assert resp.consensus_effect is None


def test_non_dag_without_precise_field_defaults_to_undefined_cyclic():
    """Conservative fail-closed: a non-DAG graph_quality lacking the precise
    cycle_affects field (synthetic ``output`` path) is treated as undefined_cyclic."""
    resp = causal_module._sequential_output_to_response(
        "pipe_test",
        _sequential_request(),
        {
            **_output(warnings=[_CYCLE_WARNING]),
            "graph_quality": _graph_quality(is_dag=False, structural_quality=0.0),
        },
    )
    assert resp.graph_is_dag is False
    assert resp.structural_identification == "undefined_cyclic"
    assert resp.requires_review is True
    assert resp.consensus_effect is None


# --------------------------------------------------------------------------- #
# cycle_irrelevant — cycle off the ancestral subgraph: preserve the estimate
# --------------------------------------------------------------------------- #


def test_cycle_irrelevant_preserves_consensus_and_robustness():
    """A cycle OFF the ancestral subgraph leaves the estimand identifiable: the
    consensus is preserved and the cycle does NOT force robustness False (a PROCEED
    refutation stands)."""
    state = {
        "graph_quality": _graph_quality(is_dag=False, structural_quality=1.0, cycle_affects=False),
        "refutation_results": {"gate_decision": "proceed"},
    }
    resp = causal_module._sequential_output_to_response(
        "pipe_test", _sequential_request(), _output(consensus_effect=1.23), state=state
    )
    assert resp.graph_is_dag is False
    assert resp.structural_identification == "cycle_irrelevant"
    assert resp.requires_review is False
    # Estimand identifiable -> number preserved, PROCEED robustness stands.
    assert resp.consensus_effect == 1.23
    assert resp.robustness_validation_performed is True


def test_parallel_cycle_irrelevant_preserves_consensus():
    state = {
        "graph_quality": _graph_quality(is_dag=False, structural_quality=1.0, cycle_affects=False),
        "refutation_results": {"gate_decision": "proceed"},
    }
    resp = causal_module._parallel_output_to_response(
        "pipe_test", _parallel_request(), _output(consensus_effect=1.23), state=state
    )
    assert resp.structural_identification == "cycle_irrelevant"
    assert resp.requires_review is False
    assert resp.consensus_effect == 1.23
    assert resp.robustness_validation_performed is True


# --------------------------------------------------------------------------- #
# acyclic — a DAG: no structural review
# --------------------------------------------------------------------------- #


def test_acyclic_dag_no_structural_review():
    state = {
        "graph_quality": _graph_quality(is_dag=True, structural_quality=1.0, cycle_affects=False),
        "refutation_results": {"gate_decision": "proceed"},
    }
    resp = causal_module._sequential_output_to_response(
        "pipe_test", _sequential_request(), _output(consensus_effect=0.21), state=state
    )
    assert resp.graph_is_dag is True
    assert resp.structural_identification == "acyclic"
    assert resp.requires_review is False
    assert resp.consensus_effect == 0.21
    assert resp.robustness_validation_performed is True
