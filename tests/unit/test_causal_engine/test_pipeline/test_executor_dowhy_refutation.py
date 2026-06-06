"""R6-F1 (#740) Edit E: DoWhyExecutor runs the real refutation suite when opted-in.

These tests exercise the *real* RefutationRunner on a small real DataFrame —
NO mocking of the executor or the runner (per the anti-mocking discipline). The
load-bearing test ``test_dowhy_executor_runs_refutation_when_flag_set`` is the
cheapest-disproof gate from the R6-F1 plan §4.3: if a real small-data refutation
does not populate a ``gate_decision`` here, the CI-from-SE approach is wrong and
the design must change before proceeding.

Covered:
- run_refutation absent/False → refutation_results == {} (no-op default).
- run_refutation True (linear_regression) → refutation_results populated with a
  real gate_decision from RefutationRunner.run_all_tests on the LIVE objects.
- run_refutation True but no SE (non-linear method) → honest skip dict, no error.

Reference: .claude/plans/causal-validation-remediation/R6-F1-implementation-plan.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.pipeline.executors.dowhy import DoWhyExecutor
from src.causal_engine.pipeline.state import PipelineConfig, PipelineStage, PipelineState


def _build_pipeline_config(*, run_refutation: bool | None = None) -> PipelineConfig:
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
    if run_refutation is not None:
        config["run_refutation"] = run_refutation
    return config


def _build_state(
    *,
    df: pd.DataFrame,
    confounders: list[str],
    run_refutation: bool | None = None,
    dowhy_method: str | None = None,
) -> PipelineState:
    filters: dict = {"estimation_data": df}
    if dowhy_method is not None:
        filters["dowhy_method"] = dowhy_method
    return PipelineState(
        query="Does treatment cause outcome?",
        question_type="causal_relationship",
        treatment_var="treatment",
        outcome_var="outcome",
        confounders=confounders,
        effect_modifiers=None,
        data_source="test_data",
        filters=filters,
        config=_build_pipeline_config(run_refutation=run_refutation),
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


def _build_real_dataframe(*, n: int = 200, true_ate: float = 1.5, seed: int = 13) -> pd.DataFrame:
    """A tiny real DataFrame with a known causal effect (test-fixture data)."""
    rng = np.random.default_rng(seed)
    confounder_a = rng.normal(0.0, 1.0, n)
    treatment = 0.5 * confounder_a + rng.normal(0.0, 1.0, n)
    outcome = true_ate * treatment + 0.7 * confounder_a + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"treatment": treatment, "outcome": outcome, "confounder_a": confounder_a})


def _build_binary_treatment_dataframe(
    *, n: int = 200, true_ate: float = 1.5, seed: int = 13
) -> pd.DataFrame:
    """A tiny real DataFrame with a BINARY treatment (test-fixture data).

    Used for the non-linear-method skip-path test: propensity-score matching
    estimates successfully on a binary treatment but the executor only computes a
    native SE for ``linear_regression`` — so ``dowhy_se`` is None and refutation
    must honestly skip rather than fabricate a CI.
    """
    rng = np.random.default_rng(seed)
    confounder_a = rng.normal(0.0, 1.0, n)
    propensity = 1.0 / (1.0 + np.exp(-(0.8 * confounder_a)))
    treatment = (rng.uniform(0.0, 1.0, n) < propensity).astype(int)
    outcome = true_ate * treatment + 0.7 * confounder_a + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"treatment": treatment, "outcome": outcome, "confounder_a": confounder_a})


class TestDoWhyExecutorRefutationOptIn:
    @pytest.mark.asyncio
    async def test_dowhy_executor_skips_refutation_by_default(self) -> None:
        """run_refutation absent → refutation_results == {} (no-op fast path)."""
        df = _build_real_dataframe()
        state = _build_state(df=df, confounders=["confounder_a"])  # flag absent
        result = await DoWhyExecutor().execute(state, state["config"])
        assert result["success"] is True, f"error={result['error']!r}"
        payload = result["result"]
        assert payload is not None
        assert payload["refutation_results"] == {}, (
            "default path must NOT run refutation — refutation_results must stay {}"
        )

    @pytest.mark.asyncio
    async def test_dowhy_executor_runs_refutation_when_flag_set(self) -> None:
        """LOAD-BEARING (plan §4.3): real refutation populates a gate_decision.

        Runs the REAL RefutationRunner on the LIVE DoWhy model/estimand/estimate
        built by the executor, with the CI derived from the real linear-regression
        SE. The faithful disproof (run separately) returned gate_decision=PROCEED;
        this asserts the executor surfaces that real gate.
        """
        df = _build_real_dataframe()
        state = _build_state(df=df, confounders=["confounder_a"], run_refutation=True)
        result = await DoWhyExecutor().execute(state, state["config"])
        assert result["success"] is True, f"error={result['error']!r}"
        payload = result["result"]
        assert payload is not None
        rr = payload["refutation_results"]
        assert rr, "refutation_results must be non-empty when run_refutation=True"
        # Real gate band must be present and a valid value.
        assert rr.get("gate_decision") in {"proceed", "review", "block"}, (
            f"refutation must surface a real gate_decision; got {rr!r}"
        )
        # On this clean linear fixture the faithful disproof returned PROCEED.
        assert rr["gate_decision"] == "proceed", (
            f"expected PROCEED on the clean linear fixture; got {rr.get('gate_decision')!r}"
        )
        # Not the honest-skip shape (we DID have a SE here).
        assert rr.get("skipped") is not True
        # Legacy-format keys surface so the dormant consumers can read them.
        assert "individual_tests" in rr
        assert "needs_review" in rr

    @pytest.mark.asyncio
    async def test_dowhy_executor_refutation_skips_without_se(self) -> None:
        """Non-linear method (no native SE) → honest skip dict, NO fabricated CI."""
        df = _build_binary_treatment_dataframe()
        state = _build_state(
            df=df,
            confounders=["confounder_a"],
            run_refutation=True,
            dowhy_method="backdoor.propensity_score_matching",
        )
        result = await DoWhyExecutor().execute(state, state["config"])
        assert result["success"] is True, f"error={result['error']!r}"
        payload = result["result"]
        assert payload is not None
        rr = payload["refutation_results"]
        assert rr.get("skipped") is True, (
            f"non-linear method has no SE → must honestly skip, not fabricate; got {rr!r}"
        )
        assert "reason" in rr
        # No gate_decision should be claimed on a skip (it was not actually run).
        assert "gate_decision" not in rr


class TestParallelInsightWakesOnPopulatedRefutation:
    """Acceptance #4: the dormant parallel.py:394 consumer fires once
    refutation_results is populated (it stayed silent while results were {})."""

    def test_insight_fires_when_refutation_results_populated(self) -> None:
        from src.causal_engine.pipeline.parallel import ParallelPipeline

        df = _build_real_dataframe()
        state = _build_state(df=df, confounders=["confounder_a"])
        state["refutation_results"] = {"gate_decision": "proceed", "tests_passed": 3}
        insights = ParallelPipeline()._generate_parallel_insights(state)
        assert any("refutation" in i.lower() for i in insights), (
            f"populated refutation_results must wake the insight; got {insights!r}"
        )

    def test_insight_silent_when_refutation_results_empty(self) -> None:
        from src.causal_engine.pipeline.parallel import ParallelPipeline

        df = _build_real_dataframe()
        state = _build_state(df=df, confounders=["confounder_a"])
        state["refutation_results"] = {}
        insights = ParallelPipeline()._generate_parallel_insights(state)
        assert not any("refutation" in i.lower() for i in insights)
