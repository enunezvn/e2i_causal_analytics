"""Tests for phase C-6 aggregation: 4-library consensus + extraction.

Covers:
- ``_update_state_with_result`` extracts new top-level fields from CausalML
  (``uplift_summary``) and NetworkX (``graph_quality``) result payloads
  written by Wave-1 executors.
- ``_aggregate_results`` (sequential) and ``_aggregate_parallel_results``
  (parallel) build cross-library consensus that:
  * Includes CausalML's ``ate`` in the ATE-consensus track when present
  * Keeps CausalML's ``auuc``/``qini`` in a SEPARATE uplift channel
    (uplift is a population-targeting quality signal; not averaged with ATE)
  * Modulates ``consensus_confidence`` by NetworkX graph quality
    (DAG + treatment->outcome path present => no penalty;
     non-DAG => structural-quality penalty)
- The Wave-3 hardcoded ``0.8`` confidence fallback is REMOVED — if a
  library result is missing or its ``confidence`` is non-finite or
  ``None``, that library is EXCLUDED from consensus rather than defaulted.
- The ``_resolve_estimation_dataframe`` helper (Option B from C-6) reads
  state in the canonical priority order documented in the helper docstring;
  the three Wave-1 data-key writers (DoWhy ``filters.estimation_data``;
  EconML ``data_cache.estimation_data``; CausalML ``filters.dataframe``)
  remain accepted for back-compat.

These tests are RED-first per the C-6 dispatch brief: they encode the
NEW post-C-6 invariants and must fail against current `main` until C-6
lands. See `.claude/plans/354_dispatch_plan_v1.md` §2.3 for design context.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pandas as pd

from src.causal_engine.pipeline.data_resolver import resolve_estimation_dataframe
from src.causal_engine.pipeline.orchestrator import PipelineOrchestrator
from src.causal_engine.pipeline.parallel import ParallelPipeline
from src.causal_engine.pipeline.router import (
    CausalLibrary,
    LibraryRouter,
)
from src.causal_engine.pipeline.sequential import SequentialPipeline
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


def _minimal_pipeline_state() -> PipelineState:
    """A minimal valid PipelineState shell for unit tests."""
    config: PipelineConfig = {
        "mode": "sequential",
        "libraries_enabled": ["dowhy", "econml", "causalml", "networkx"],
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

    state: PipelineState = {
        "query": "Does X cause Y?",
        "question_type": "causal_relationship",
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "confounders": ["c1"],
        "effect_modifiers": None,
        "data_source": "test",
        "filters": None,
        "config": config,
        "routed_libraries": ["dowhy"],
        "routing_confidence": 0.9,
        "routing_rationale": "test",
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
        "consensus_effect": None,
        "consensus_confidence": None,
        "library_agreement": None,
        "graph_quality": None,
        "uplift_summary": None,
        "library_metric_types": None,
        "nested_cate": None,
        "segment_confidence_intervals": None,
        "executive_summary": None,
        "key_insights": None,
        "recommended_actions": None,
        "current_stage": PipelineStage.PENDING,
        "stage_latencies": {},
        "total_latency_ms": 0,
        "libraries_executed": [],
        "libraries_skipped": [],
        "errors": [],
        "warnings": [],
        "status": "pending",
    }
    return state


class _ConcreteOrchestrator(PipelineOrchestrator):
    """Minimal concrete orchestrator for testing ``_update_state_with_result``."""

    async def execute(self, input_data: PipelineInput) -> PipelineOutput:
        routing_decision = await self.route(input_data["query"])
        state = self._create_initial_state(input_data, routing_decision)
        return self._create_output(state)


def _real_networkx_payload() -> dict[str, Any]:
    """A NetworkX result payload matching the post-C-5 executor output shape."""
    return {
        "nodes": ["treatment", "outcome", "c1"],
        "edges": [
            {"from": "treatment", "to": "outcome"},
            {"from": "c1", "to": "treatment"},
            {"from": "c1", "to": "outcome"},
        ],
        "centrality": {
            "degree": {"treatment": 0.5, "outcome": 0.5, "c1": 1.0},
            "betweenness": {"treatment": 0.0, "outcome": 0.0, "c1": 0.0},
            "in_degree": {"treatment": 1, "outcome": 2, "c1": 0},
            "out_degree": {"treatment": 1, "outcome": 0, "c1": 2},
        },
        "paths": {
            "treatment_to_outcome": [["treatment", "outcome"]],
            "n_paths_treatment_to_outcome": 1,
            "shortest_path_length": 1,
        },
        "n_nodes": 3,
        "n_edges": 3,
        "is_dag": True,
        "has_treatment_outcome_path": True,
        "cycles": [],
        "graph_source": "symbolic",
        "treatment_var": "treatment",
        "outcome_var": "outcome",
    }


def _real_causalml_payload() -> dict[str, Any]:
    """A CausalML result payload matching the post-C-4 executor output shape.

    Mirrors `_build_causalml_result_payload` in
    ``src/causal_engine/pipeline/executors/causalml.py`` (line 450-475).
    """
    return {
        "model": "uplift_random_forest",
        "ate": 0.18,
        "att": 0.20,
        "atc": 0.15,
        "ate_std": 0.04,
        "ate_ci_lower": 0.10,
        "ate_ci_upper": 0.26,
        "auuc": 0.72,
        "qini": 0.55,
        "uplift_scores_summary": {
            "n": 1000,
            "mean": 0.18,
            "std": 0.30,
            "min": -0.50,
            "max": 0.75,
            "median": 0.18,
            "p10": -0.20,
            "p90": 0.55,
        },
        "feature_importances": {"c1": 0.4, "x1": 0.6},
        "feature_names": ["c1", "x1"],
        "n_samples": 1000,
        "treatment_groups": ["1"],
        "observed_treatment_groups": ["0", "1"],
        "control_name": "0",
    }


def _real_dowhy_payload() -> dict[str, Any]:
    """A DoWhy result payload matching the post-C-2 executor output shape."""
    return {
        "causal_effect": 0.15,
        "identified_estimand": "nonparametric-ate",
        "identified_estimand_repr": "<Estimand>",
        "dowhy_method": "backdoor.linear_regression",
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "common_causes": ["c1"],
        "graph_source": "networkx",
        "refutation_results": {},
    }


def _real_econml_payload() -> dict[str, Any]:
    """An EconML result payload matching the post-C-3 executor output shape."""
    return {
        "estimator": "causal_forest_dml",
        "ate": 0.17,
        "ate_ci_lower": 0.10,
        "ate_ci_upper": 0.24,
        "ate_std": 0.04,
        "cate_by_segment": {"A": 0.15, "B": 0.20},
        "heterogeneity_score": 0.30,
        "energy_score": 0.20,
        "quality_tier": "high",
        "selection_strategy": "best_energy_score",
        "selection_reason": "lowest energy_score",
        "energy_scores": {"causal_forest_dml": 0.20},
        "energy_score_gap": 0.05,
        "n_estimators_evaluated": 4,
        "n_estimators_succeeded": 4,
    }


def _le_result(
    library: str, payload: dict[str, Any] | None, *, confidence: float, success: bool = True
) -> LibraryExecutionResult:
    return LibraryExecutionResult(
        library=library,
        success=success,
        latency_ms=100,
        result=payload,
        error=None if success else "fail",
        confidence=confidence,
        warnings=[],
    )


# =============================================================================
# 1. _update_state_with_result extraction for NetworkX (graph_quality)
# =============================================================================


class TestUpdateStateWithNetworkXResult:
    """C-6 extends `_update_state_with_result` to populate
    `state["graph_quality"]` from the rich post-C-5 NetworkX payload.
    """

    def test_extracts_graph_quality_summary_from_real_networkx_payload(self) -> None:
        orchestrator = _ConcreteOrchestrator()
        state = _minimal_pipeline_state()
        result = _le_result("networkx", _real_networkx_payload(), confidence=1.0)

        updated = orchestrator._update_state_with_result(state, CausalLibrary.NETWORKX, result)

        # New top-level field: structural quality summary
        quality = updated["graph_quality"]
        assert quality is not None, (
            "graph_quality must be populated when NetworkX returns success "
            "with a valid result payload"
        )
        assert quality["n_nodes"] == 3
        assert quality["n_edges"] == 3
        assert quality["is_dag"] is True
        assert quality["has_treatment_outcome_path"] is True
        assert quality["structural_quality"] == 1.0  # DAG + path + n>=3

    def test_graph_quality_reflects_cyclic_graph(self) -> None:
        orchestrator = _ConcreteOrchestrator()
        state = _minimal_pipeline_state()
        payload = _real_networkx_payload()
        payload["is_dag"] = False
        payload["cycles"] = [["a", "b", "a"]]
        result = _le_result("networkx", payload, confidence=0.0)

        updated = orchestrator._update_state_with_result(state, CausalLibrary.NETWORKX, result)

        quality = updated["graph_quality"]
        assert quality is not None
        assert quality["is_dag"] is False
        assert quality["structural_quality"] == 0.0  # non-DAG => zero structural quality
        assert quality["n_cycles"] == 1

    def test_graph_quality_none_when_executor_failed(self) -> None:
        orchestrator = _ConcreteOrchestrator()
        state = _minimal_pipeline_state()
        # Failure path — graph_quality should NOT be populated with fake structure
        result = _le_result("networkx", None, confidence=0.0, success=False)

        updated = orchestrator._update_state_with_result(state, CausalLibrary.NETWORKX, result)

        assert updated["graph_quality"] is None

    def test_preserves_existing_causal_graph_field_for_backcompat(self) -> None:
        """Backward-compat: pre-C-6 readers of `state["causal_graph"]` still
        see the full result dict (no silent rename)."""
        orchestrator = _ConcreteOrchestrator()
        state = _minimal_pipeline_state()
        result = _le_result("networkx", _real_networkx_payload(), confidence=1.0)

        updated = orchestrator._update_state_with_result(state, CausalLibrary.NETWORKX, result)
        # The orchestrator still sets `causal_graph` to result["result"] (the full payload).
        assert updated["causal_graph"] is not None
        assert "nodes" in updated["causal_graph"]


# =============================================================================
# 2. _update_state_with_result extraction for CausalML (uplift_summary)
# =============================================================================


class TestUpdateStateWithCausalMLResult:
    """C-6 extends `_update_state_with_result` to:
    1. Pull CausalML's `ate` into a dedicated `state` field
       (`causalml_ate`, distinct from EconML's `overall_ate`).
    2. Populate `state["uplift_summary"]` with model-quality metrics
       (`auuc`/`qini`/`ate_ci`/`n_samples`/`treatment_groups`).
    3. Use the post-C-4 key name `uplift_scores_summary` (not the pre-C-4
       `uplift_by_segment`).
    """

    def test_extracts_uplift_summary_with_real_payload(self) -> None:
        orchestrator = _ConcreteOrchestrator()
        state = _minimal_pipeline_state()
        result = _le_result("causalml", _real_causalml_payload(), confidence=0.7)

        updated = orchestrator._update_state_with_result(state, CausalLibrary.CAUSALML, result)

        summary = updated["uplift_summary"]
        assert summary is not None
        assert summary["auuc"] == 0.72
        assert summary["qini"] == 0.55
        assert summary["ate"] == 0.18
        assert summary["ate_ci_lower"] == 0.10
        assert summary["ate_ci_upper"] == 0.26
        assert summary["n_samples"] == 1000
        assert summary["treatment_groups"] == ["1"]

    def test_reads_post_c4_key_name_uplift_scores_summary(self) -> None:
        """The orchestrator must read the post-C-4 key `uplift_scores_summary`
        rather than the pre-C-4 key `uplift_by_segment`."""
        orchestrator = _ConcreteOrchestrator()
        state = _minimal_pipeline_state()
        result = _le_result("causalml", _real_causalml_payload(), confidence=0.7)

        updated = orchestrator._update_state_with_result(state, CausalLibrary.CAUSALML, result)

        # `uplift_scores` should pick up the `uplift_scores_summary` summary dict,
        # not the legacy `uplift_by_segment` (which is None in post-C-4 payloads).
        assert updated["uplift_scores"] is not None
        assert updated["uplift_scores"]["n"] == 1000
        assert updated["uplift_scores"]["mean"] == 0.18

    def test_uplift_summary_none_when_executor_failed(self) -> None:
        orchestrator = _ConcreteOrchestrator()
        state = _minimal_pipeline_state()
        result = _le_result("causalml", None, confidence=0.0, success=False)

        updated = orchestrator._update_state_with_result(state, CausalLibrary.CAUSALML, result)

        assert updated["uplift_summary"] is None

    def test_handles_auuc_qini_none_post_metrics_failure(self) -> None:
        """Per C-4 docstring: when auuc/qini metric helpers raise,
        CausalML returns success=True with auuc/qini = None and a warning.
        The orchestrator must propagate None (no silent zero-substitution).
        """
        orchestrator = _ConcreteOrchestrator()
        state = _minimal_pipeline_state()
        payload = _real_causalml_payload()
        payload["auuc"] = None
        payload["qini"] = None
        result = _le_result("causalml", payload, confidence=0.5)

        updated = orchestrator._update_state_with_result(state, CausalLibrary.CAUSALML, result)

        assert updated["uplift_summary"]["auuc"] is None
        assert updated["uplift_summary"]["qini"] is None


# =============================================================================
# 3. Aggregation — 4-library consensus + ATE/uplift channel separation
# =============================================================================


def _seq_pipeline() -> SequentialPipeline:
    return SequentialPipeline(router=LibraryRouter())


def _par_pipeline() -> ParallelPipeline:
    return ParallelPipeline(router=LibraryRouter())


class TestATEConsensusIncludesCausalML:
    """C-6: CausalML's `ate` is part of the ATE-consensus track (alongside
    DoWhy and EconML); CausalML and EconML both report population ATE.
    """

    def test_sequential_consensus_includes_causalml_ate(self) -> None:
        state = _minimal_pipeline_state()
        # All 3 effect-estimating libraries succeeded.
        state["dowhy_result"] = _le_result("dowhy", _real_dowhy_payload(), confidence=1.0)
        state["causal_effect"] = 0.15
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17
        state["causalml_result"] = _le_result("causalml", _real_causalml_payload(), confidence=0.6)

        pipe = _seq_pipeline()
        updated = pipe._aggregate_results(state)

        # Consensus must include CausalML — weighted average of 3 ATEs.
        consensus = updated["consensus_effect"]
        assert consensus is not None
        # Weighted average: (0.15*1.0 + 0.17*0.8 + 0.18*0.6) / (1.0+0.8+0.6)
        # = (0.15 + 0.136 + 0.108) / 2.4 = 0.394 / 2.4 ≈ 0.164
        assert 0.14 < consensus < 0.19, (
            f"consensus_effect={consensus} should average over all 3 ATEs; "
            "CausalML's ate=0.18 must be included"
        )

    def test_parallel_consensus_includes_causalml_ate(self) -> None:
        state = _minimal_pipeline_state()
        state["dowhy_result"] = _le_result("dowhy", _real_dowhy_payload(), confidence=1.0)
        state["causal_effect"] = 0.15
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17
        state["causalml_result"] = _le_result("causalml", _real_causalml_payload(), confidence=0.6)

        pipe = _par_pipeline()
        updated = pipe._aggregate_parallel_results(state)

        consensus = updated["consensus_effect"]
        assert consensus is not None
        assert 0.14 < consensus < 0.19

    def test_pairwise_agreement_includes_all_pairs_with_ate(self) -> None:
        """C-6: agreement dict must include all pairwise combinations of
        effect-estimating libraries that produced an ATE.
        """
        state = _minimal_pipeline_state()
        state["dowhy_result"] = _le_result("dowhy", _real_dowhy_payload(), confidence=1.0)
        state["causal_effect"] = 0.15
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17
        state["causalml_result"] = _le_result("causalml", _real_causalml_payload(), confidence=0.6)

        pipe = _par_pipeline()
        updated = pipe._aggregate_parallel_results(state)

        agreement = updated["library_agreement"]
        assert agreement is not None
        # 3 libraries => 3 pairs: dowhy_econml, dowhy_causalml, econml_causalml
        assert "dowhy_econml" in agreement
        assert "dowhy_causalml" in agreement
        assert "econml_causalml" in agreement


class TestUpliftChannelSeparateFromATE:
    """C-6: CausalML's uplift (auuc/qini) lives in a separate channel from
    ATE consensus. Uplift metrics are population-targeting quality indicators,
    not effect-magnitude estimates.
    """

    def test_uplift_summary_is_populated_when_causalml_succeeds(self) -> None:
        state = _minimal_pipeline_state()
        state["causalml_result"] = _le_result("causalml", _real_causalml_payload(), confidence=0.7)
        # Pre-populate via orchestrator helper.
        orchestrator = _ConcreteOrchestrator()
        state = orchestrator._update_state_with_result(
            state, CausalLibrary.CAUSALML, cast(LibraryExecutionResult, state["causalml_result"])
        )

        pipe = _seq_pipeline()
        updated = pipe._aggregate_results(state)

        # uplift_summary must survive aggregation
        assert updated["uplift_summary"] is not None
        assert updated["uplift_summary"]["auuc"] == 0.72
        assert updated["uplift_summary"]["qini"] == 0.55

    def test_auuc_qini_not_averaged_into_consensus_effect(self) -> None:
        """auuc=0.72 and qini=0.55 must NOT be mistaken for ATE values.
        If only CausalML ran (no DoWhy/EconML), uplift metrics must NOT
        leak into consensus_effect.
        """
        state = _minimal_pipeline_state()
        state["causalml_result"] = _le_result("causalml", _real_causalml_payload(), confidence=0.7)
        orchestrator = _ConcreteOrchestrator()
        state = orchestrator._update_state_with_result(
            state, CausalLibrary.CAUSALML, cast(LibraryExecutionResult, state["causalml_result"])
        )

        pipe = _seq_pipeline()
        updated = pipe._aggregate_results(state)

        # Consensus_effect should equal CausalML's ATE (0.18), NOT auuc/qini.
        # If consensus_effect were 0.72 or 0.55 the auuc/qini leaked in.
        assert updated["consensus_effect"] != 0.72
        assert updated["consensus_effect"] != 0.55
        if updated["consensus_effect"] is not None:
            assert abs(updated["consensus_effect"] - 0.18) < 0.01


# =============================================================================
# 4. Graph quality modulates consensus_confidence
# =============================================================================


class TestGraphQualityModulatesConsensusConfidence:
    """C-6: structural quality of the NetworkX graph modulates the trust
    we place in downstream effect estimates. A cyclic graph (is_dag=False)
    means backdoor adjustment isn't well-defined; consensus confidence is
    penalized. A clean DAG with treatment->outcome path => no penalty.
    """

    def test_dag_with_path_does_not_penalize_confidence(self) -> None:
        state = _minimal_pipeline_state()
        # All libraries succeed
        state["dowhy_result"] = _le_result("dowhy", _real_dowhy_payload(), confidence=1.0)
        state["causal_effect"] = 0.15
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17
        # NetworkX populates graph_quality with structural_quality=1.0
        nx_result = _le_result("networkx", _real_networkx_payload(), confidence=1.0)
        orchestrator = _ConcreteOrchestrator()
        state = orchestrator._update_state_with_result(state, CausalLibrary.NETWORKX, nx_result)

        pipe = _seq_pipeline()
        updated = pipe._aggregate_results(state)

        # With perfect graph quality, consensus confidence is not penalized.
        # Baseline (no penalty): mean of confidences = (1.0+0.8)/2 = 0.9.
        # Allowing slight floating-point drift.
        assert updated["consensus_confidence"] is not None
        assert updated["consensus_confidence"] >= 0.85

    def test_non_dag_penalizes_consensus_confidence(self) -> None:
        state = _minimal_pipeline_state()
        state["dowhy_result"] = _le_result("dowhy", _real_dowhy_payload(), confidence=1.0)
        state["causal_effect"] = 0.15
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17
        # NetworkX reports non-DAG => structural_quality=0.0
        nx_payload = _real_networkx_payload()
        nx_payload["is_dag"] = False
        nx_payload["cycles"] = [["a", "b", "a"]]
        nx_result = _le_result("networkx", nx_payload, confidence=0.0)
        orchestrator = _ConcreteOrchestrator()
        state = orchestrator._update_state_with_result(state, CausalLibrary.NETWORKX, nx_result)

        pipe = _seq_pipeline()
        updated = pipe._aggregate_results(state)

        # Non-DAG => confidence must be penalized to BELOW the no-penalty baseline.
        assert updated["consensus_confidence"] is not None
        assert updated["consensus_confidence"] < 0.85

    def test_no_networkx_result_does_not_penalize(self) -> None:
        """When NetworkX did not run (graph_quality is None), the aggregator
        must NOT apply a penalty. Absence of evidence is not evidence of
        absence; this avoids silent under-rating when NetworkX is skipped.
        """
        state = _minimal_pipeline_state()
        state["dowhy_result"] = _le_result("dowhy", _real_dowhy_payload(), confidence=1.0)
        state["causal_effect"] = 0.15
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17
        # NetworkX skipped — graph_quality stays None
        assert state["graph_quality"] is None

        pipe = _seq_pipeline()
        updated = pipe._aggregate_results(state)

        assert updated["consensus_confidence"] is not None
        assert updated["consensus_confidence"] >= 0.85


# =============================================================================
# 5. 0.8 hardcoded confidence fallback REMOVED
# =============================================================================


class TestZeroEightConfidenceFallbackRemoved:
    """C-6: per Wave-3 anti-mocking pattern #2, the `0.8` hardcoded confidence
    fallback at sequential.py:181/185 and parallel.py:291-300 is REMOVED.
    A missing/None confidence means the library is EXCLUDED from consensus
    (skip, do not silent-default).
    """

    def test_sequential_skips_library_with_none_confidence(self) -> None:
        """When a result dict is missing or its confidence is None, the
        corresponding effect must be DROPPED from the consensus, not
        defaulted to 0.8.
        """
        state = _minimal_pipeline_state()
        # DoWhy has an effect but the result is None (couldn't happen in
        # practice via the executor, but the aggregator MUST guard against it)
        state["dowhy_result"] = None
        state["causal_effect"] = 0.15  # leftover from somewhere
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17

        pipe = _seq_pipeline()
        updated = pipe._aggregate_results(state)

        # When DoWhy result is None, its effect is excluded from consensus.
        # consensus_effect = 0.17 * 0.8 / 0.8 = 0.17 (only EconML contributes)
        assert updated["consensus_effect"] is not None
        assert abs(updated["consensus_effect"] - 0.17) < 0.001
        assert abs(updated["consensus_confidence"] - 0.8) < 0.001

    def test_parallel_skips_library_with_failed_result(self) -> None:
        state = _minimal_pipeline_state()
        state["dowhy_result"] = _le_result("dowhy", None, confidence=0.0, success=False)
        state["causal_effect"] = None
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17

        pipe = _par_pipeline()
        updated = pipe._aggregate_parallel_results(state)

        # Only EconML contributes.
        assert updated["consensus_effect"] is not None
        assert abs(updated["consensus_effect"] - 0.17) < 0.001

    def test_sequential_excludes_executor_with_non_finite_confidence(self) -> None:
        """Non-finite (NaN/inf) confidence values must be EXCLUDED from
        consensus, never silently coerced to 0.8."""
        state = _minimal_pipeline_state()
        state["dowhy_result"] = _le_result("dowhy", _real_dowhy_payload(), confidence=float("nan"))
        state["causal_effect"] = 0.15
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17

        pipe = _seq_pipeline()
        updated = pipe._aggregate_results(state)

        # NaN confidence => DoWhy excluded. consensus is EconML alone.
        assert updated["consensus_effect"] is not None
        assert abs(updated["consensus_effect"] - 0.17) < 0.001

    def test_no_silent_0_8_default_when_no_libraries_contribute(self) -> None:
        """When NO library has a valid (effect, confidence) pair, consensus
        must remain None — never silently default to a 0.8-shaped value.
        """
        state = _minimal_pipeline_state()
        # No effect-producing libraries succeeded
        pipe = _seq_pipeline()
        updated = pipe._aggregate_results(state)

        assert updated["consensus_effect"] is None
        assert updated["consensus_confidence"] is None


# =============================================================================
# 6. resolve_estimation_dataframe helper (Option B data-key harmonization)
# =============================================================================


class TestResolveEstimationDataframe:
    """C-6 Option B: a single data-resolver helper that reads any of the
    three current data-key sites used by Wave-1 executors. Each Wave-1
    executor keeps its current site for back-compat; new code uses this
    helper for forward compatibility with a planned single canonical site.
    """

    def test_resolves_from_data_cache_estimation_data(self) -> None:
        """Canonical priority key (matches the agents/causal_impact convention)."""
        state = _minimal_pipeline_state()
        df = pd.DataFrame({"a": [1, 2, 3]})
        state_cast: dict[str, Any] = cast(dict[str, Any], state)
        state_cast["data_cache"] = {"estimation_data": df}

        resolved = resolve_estimation_dataframe(state)
        assert resolved is df

    def test_resolves_from_filters_estimation_data(self) -> None:
        """DoWhy's data-conveyance key (post-C-2)."""
        state = _minimal_pipeline_state()
        df = pd.DataFrame({"a": [1, 2, 3]})
        state["filters"] = {"estimation_data": df}

        resolved = resolve_estimation_dataframe(state)
        assert resolved is df

    def test_resolves_from_filters_dataframe(self) -> None:
        """CausalML's data-conveyance key (post-C-4)."""
        state = _minimal_pipeline_state()
        df = pd.DataFrame({"a": [1, 2, 3]})
        state["filters"] = {"dataframe": df}

        resolved = resolve_estimation_dataframe(state)
        assert resolved is df

    def test_resolves_from_filters_data(self) -> None:
        """Forward-compat key supported by DoWhy executor."""
        state = _minimal_pipeline_state()
        df = pd.DataFrame({"a": [1, 2, 3]})
        state["filters"] = {"data": df}

        resolved = resolve_estimation_dataframe(state)
        assert resolved is df

    def test_priority_data_cache_over_filters(self) -> None:
        """When BOTH `data_cache.estimation_data` and `filters` carry a
        DataFrame, the canonical `data_cache.estimation_data` wins."""
        state = _minimal_pipeline_state()
        canonical = pd.DataFrame({"a": [1]})
        legacy = pd.DataFrame({"b": [2]})
        state_cast: dict[str, Any] = cast(dict[str, Any], state)
        state_cast["data_cache"] = {"estimation_data": canonical}
        state["filters"] = {"estimation_data": legacy}

        resolved = resolve_estimation_dataframe(state)
        assert resolved is canonical

    def test_returns_none_when_no_dataframe_anywhere(self) -> None:
        state = _minimal_pipeline_state()
        # No data_cache, filters=None
        resolved = resolve_estimation_dataframe(state)
        assert resolved is None

    def test_returns_none_when_filters_is_non_dict(self) -> None:
        state = _minimal_pipeline_state()
        state["filters"] = cast(dict[str, Any], "not a dict")  # type: ignore[arg-type]

        resolved = resolve_estimation_dataframe(state)
        assert resolved is None

    def test_returns_none_when_data_cache_value_is_not_dataframe(self) -> None:
        """A non-DataFrame at the canonical key returns None — never
        substitutes a different signal silently."""
        state = _minimal_pipeline_state()
        state_cast: dict[str, Any] = cast(dict[str, Any], state)
        state_cast["data_cache"] = {"estimation_data": [[1, 2], [3, 4]]}  # list, not DF

        resolved = resolve_estimation_dataframe(state)
        assert resolved is None

    def test_ignores_dowhy_method_string_in_filters(self) -> None:
        """The `dowhy_method` key in `filters` is a string, not a DataFrame —
        helper must skip it (regression-guard)."""
        state = _minimal_pipeline_state()
        state["filters"] = {"dowhy_method": "backdoor.linear_regression"}

        resolved = resolve_estimation_dataframe(state)
        assert resolved is None


# =============================================================================
# 7. End-to-end aggregation invariants (TypedDict + non-finite guarding)
# =============================================================================


class TestAggregationInvariants:
    """Aggregator output invariants: consensus_* fields are always
    finite-or-None; library_agreement values are in [0, 1] or absent.
    """

    def test_consensus_effect_is_finite_or_none(self) -> None:
        state = _minimal_pipeline_state()
        state["dowhy_result"] = _le_result("dowhy", _real_dowhy_payload(), confidence=1.0)
        state["causal_effect"] = 0.15
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17

        pipe = _seq_pipeline()
        updated = pipe._aggregate_results(state)

        if updated["consensus_effect"] is not None:
            assert math.isfinite(updated["consensus_effect"])
        if updated["consensus_confidence"] is not None:
            assert math.isfinite(updated["consensus_confidence"])

    def test_library_agreement_values_in_unit_interval(self) -> None:
        state = _minimal_pipeline_state()
        state["dowhy_result"] = _le_result("dowhy", _real_dowhy_payload(), confidence=1.0)
        state["causal_effect"] = 0.15
        state["econml_result"] = _le_result("econml", _real_econml_payload(), confidence=0.8)
        state["overall_ate"] = 0.17
        state["causalml_result"] = _le_result("causalml", _real_causalml_payload(), confidence=0.6)

        pipe = _par_pipeline()
        updated = pipe._aggregate_parallel_results(state)

        agreement = updated["library_agreement"]
        assert agreement is not None
        for pair_name, score in agreement.items():
            assert 0.0 <= score <= 1.0, f"library_agreement[{pair_name}]={score} must be in [0,1]"


# =============================================================================
# 8. Anti-mocking AST guard for new aggregation code
# =============================================================================


class TestNoSyntheticDataInAggregator:
    """Per CLAUDE.md anti-mocking discipline: the C-6 aggregator must not
    introduce `random.uniform` / `np.random.seed` / hardcoded plausible-but-
    fake constants. Static check on the aggregation modules.
    """

    def test_no_random_uniform_in_sequential(self) -> None:
        from pathlib import Path

        src = Path(__file__).parents[4] / "src/causal_engine/pipeline/sequential.py"
        text = src.read_text(encoding="utf-8")
        assert "random.uniform" not in text
        assert "np.random.seed" not in text

    def test_no_random_uniform_in_parallel(self) -> None:
        from pathlib import Path

        src = Path(__file__).parents[4] / "src/causal_engine/pipeline/parallel.py"
        text = src.read_text(encoding="utf-8")
        assert "random.uniform" not in text
        assert "np.random.seed" not in text

    def test_no_random_uniform_in_orchestrator(self) -> None:
        from pathlib import Path

        src = Path(__file__).parents[4] / "src/causal_engine/pipeline/orchestrator.py"
        text = src.read_text(encoding="utf-8")
        assert "random.uniform" not in text
        assert "np.random.seed" not in text

    def test_no_zero_eight_hardcoded_fallback_in_sequential(self) -> None:
        """The pre-C-6 `else 0.8` fallback pattern is forbidden post-C-6.

        Uses AST analysis (not raw substring) so module docstrings and
        inline comments that REFERENCE the historical anti-pattern in
        their explanatory text don't trigger a false positive. The
        executable form (an ``else`` clause yielding the literal
        constant ``0.8``) is what we forbid.
        """
        import ast
        from pathlib import Path

        src = Path(__file__).parents[4] / "src/causal_engine/pipeline/sequential.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        offenders = list(_find_else_0_8_expressions(tree))
        assert not offenders, (
            f"Wave-3 anti-mocking pattern #2 forbids the `else 0.8` "
            f"silent-fallback executable pattern; found at lines: {offenders}"
        )

    def test_no_zero_eight_hardcoded_fallback_in_parallel(self) -> None:
        import ast
        from pathlib import Path

        src = Path(__file__).parents[4] / "src/causal_engine/pipeline/parallel.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        offenders = list(_find_else_0_8_expressions(tree))
        assert not offenders, (
            f"Wave-3 anti-mocking pattern #2 forbids the `else 0.8` "
            f"silent-fallback executable pattern; found at lines: {offenders}"
        )


def _find_else_0_8_expressions(tree: object) -> list[int]:
    """Walk an AST and yield line numbers of ``X if Y else 0.8`` expressions.

    Catches the precise Wave-3 anti-pattern at sequential.py:181/185:
    ``state["dowhy_result"]["confidence"] if state["dowhy_result"] else 0.8``
    Comments / docstrings that REFERENCE the pattern in prose don't show
    up in the AST so this guard is robust to documentation churn.
    """
    import ast

    offenders: list[int] = []
    for node in ast.walk(tree):  # type: ignore[arg-type]
        if not isinstance(node, ast.IfExp):
            continue
        else_branch = node.orelse
        if isinstance(else_branch, ast.Constant) and else_branch.value == 0.8:
            offenders.append(node.lineno)
    return offenders
