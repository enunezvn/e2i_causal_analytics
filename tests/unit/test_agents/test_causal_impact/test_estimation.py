"""Tests for estimation node."""

import pytest

from src.agents.causal_impact.nodes.estimation import EstimationNode
from src.agents.causal_impact.state import CausalGraph, CausalImpactState


class TestEstimationNode:
    """Test EstimationNode."""

    def _create_test_graph(self) -> CausalGraph:
        """Create test causal graph."""
        return {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "digraph { ... }",
            "confidence": 0.85,
        }

    @pytest.mark.asyncio
    async def test_estimate_with_causal_forest(self):
        """Test estimation using CausalForestDML."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-1",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "CausalForestDML"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert "estimation_result" in result
        est = result["estimation_result"]

        assert est["method"] == "CausalForestDML"
        assert "ate" in est
        assert "ate_ci_lower" in est
        assert "ate_ci_upper" in est
        assert est["effect_size"] in ["small", "medium", "large"]
        assert isinstance(est["statistical_significance"], bool)
        assert result["current_phase"] == "refuting"

    @pytest.mark.asyncio
    async def test_estimate_with_linear_dml(self):
        """Test estimation using LinearDML."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-2",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "LinearDML"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        assert est["method"] == "LinearDML"
        assert "ate" in est

    @pytest.mark.asyncio
    async def test_estimate_with_linear_regression(self):
        """Test estimation using linear regression."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-3",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "linear_regression"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        assert est["method"] == "linear_regression"

    @pytest.mark.asyncio
    async def test_estimate_with_propensity_weighting(self):
        """Test estimation using propensity score weighting."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-4",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "propensity_score_weighting"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        assert est["method"] == "propensity_score_weighting"

    @pytest.mark.asyncio
    async def test_confidence_interval_validity(self):
        """Test that confidence intervals are valid."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-5",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        ate = est["ate"]
        ci_lower = est["ate_ci_lower"]
        ci_upper = est["ate_ci_upper"]

        # CI should contain point estimate
        assert ci_lower <= ate <= ci_upper, "ATE not within confidence interval"

    @pytest.mark.asyncio
    async def test_effect_size_classification(self):
        """Test effect size classification."""
        node = EstimationNode()

        # Test small effect
        assert node._classify_effect_size(0.1) == "small"

        # Test medium effect
        assert node._classify_effect_size(0.3) == "medium"

        # Test large effect
        assert node._classify_effect_size(0.8) == "large"

        # Test negative effects
        assert node._classify_effect_size(-0.1) == "small"
        assert node._classify_effect_size(-0.6) == "large"

    @pytest.mark.asyncio
    async def test_heterogeneity_detection(self):
        """Test CATE heterogeneity detection."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-6",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "CausalForestDML"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]

        # CausalForestDML should detect heterogeneity
        if est["method"] == "CausalForestDML":
            assert est["heterogeneity_detected"] is True
            assert "cate_segments" in est
            assert len(est["cate_segments"]) > 0

    @pytest.mark.asyncio
    async def test_covariates_adjusted(self):
        """Test that adjustment set is recorded."""
        node = EstimationNode()

        graph = self._create_test_graph()
        graph["adjustment_sets"] = [["geographic_region", "hcp_specialty"]]

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-7",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region", "hcp_specialty"],
            "data_source": "synthetic",
            "causal_graph": graph,
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        assert "covariates_adjusted" in est
        assert "geographic_region" in est["covariates_adjusted"]

    @pytest.mark.asyncio
    async def test_sample_size_recorded(self):
        """Test that sample size is recorded."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-8",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        assert "sample_size" in est
        assert est["sample_size"] > 0

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test that estimation latency is measured."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-9",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert "estimation_latency_ms" in result
        assert result["estimation_latency_ms"] >= 0
        assert result["estimation_latency_ms"] < 30000  # Should be < 30s

    @pytest.mark.asyncio
    async def test_error_handling_missing_graph(self):
        """Test error handling when causal graph is missing."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-10",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert "estimation_error" in result
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_error_handling_unknown_method(self):
        """Test error handling for unknown estimation method."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-11",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "unknown_method"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert "estimation_error" in result
        assert result["status"] == "failed"


class TestCATESegments:
    """Test CATE segment analysis."""

    @pytest.mark.asyncio
    async def test_cate_segments_structure(self):
        """Test CATE segments have correct structure."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-12",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": [],
            "data_source": "synthetic",
            "causal_graph": {
                "nodes": ["hcp_engagement_level", "patient_conversion_rate"],
                "edges": [("hcp_engagement_level", "patient_conversion_rate")],
                "treatment_nodes": ["hcp_engagement_level"],
                "outcome_nodes": ["patient_conversion_rate"],
                # A CATE forest needs covariates; use the real backdoor from the
                # synthetic frame. (An EMPTY [[]] backdoor would mean zero
                # covariates -> CausalForestDML cannot fit -> fail-closed; that
                # path is covered by TestEmptyBackdoorEstimation.)
                "adjustment_sets": [["geographic_region", "hcp_specialty"]],
                "dag_dot": "...",
                "confidence": 0.8,
            },
            "parameters": {"method": "CausalForestDML"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]

        if "cate_segments" in est:
            for segment in est["cate_segments"]:
                assert "segment" in segment
                assert "cate" in segment
                assert "size" in segment
                assert "description" in segment


class TestStatisticalSignificance:
    """Test statistical significance logic."""

    @pytest.mark.asyncio
    async def test_significance_threshold(self):
        """Test that significance is based on 95% CI."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-13",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": [],
            "data_source": "synthetic",
            "causal_graph": {
                "nodes": ["hcp_engagement_level", "patient_conversion_rate"],
                "edges": [("hcp_engagement_level", "patient_conversion_rate")],
                "treatment_nodes": ["hcp_engagement_level"],
                "outcome_nodes": ["patient_conversion_rate"],
                "adjustment_sets": [[]],
                "dag_dot": "...",
                "confidence": 0.8,
            },
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]

        # If significant, CI should not contain 0
        if est["statistical_significance"]:
            ci_lower = est["ate_ci_lower"]
            ci_upper = est["ate_ci_upper"]

            # Either both positive or both negative
            assert (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)

    @pytest.mark.asyncio
    async def test_p_value_consistency(self):
        """Test that p-value is consistent with significance."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-14",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": [],
            "data_source": "synthetic",
            "causal_graph": {
                "nodes": ["hcp_engagement_level", "patient_conversion_rate"],
                "edges": [("hcp_engagement_level", "patient_conversion_rate")],
                "treatment_nodes": ["hcp_engagement_level"],
                "outcome_nodes": ["patient_conversion_rate"],
                "adjustment_sets": [[]],
                "dag_dot": "...",
                "confidence": 0.8,
            },
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]

        # If significant, p-value should be < 0.05
        if est["statistical_significance"]:
            assert est["p_value"] < 0.05


class TestEnergyScoreReviewGate:
    """M-est3: when the selector flags requires_review (best energy score above
    max_acceptable), the EstimationResult must be marked requires_review=True
    and downgraded to the 'unreliable' quality tier rather than emitted as a
    clean, reliable ATE.
    """

    def _graph(self):
        return {
            "nodes": ["t", "y", "c"],
            "edges": [("c", "t"), ("c", "y"), ("t", "y")],
            "treatment_nodes": ["t"],
            "outcome_nodes": ["y"],
            "adjustment_sets": [["c"]],
            "dag_dot": "digraph { }",
            "confidence": 0.8,
        }

    def test_requires_review_propagates_to_estimation_result(self, monkeypatch):
        import numpy as np

        from src.agents.causal_impact.nodes.estimation import EstimationNode
        from src.causal_engine.energy_score.estimator_selector import (
            EstimatorResult,
            EstimatorType,
            SelectionResult,
            SelectionStrategy,
        )
        from src.causal_engine.energy_score.score_calculator import EnergyScoreResult

        node = EstimationNode()

        # Craft a SUCCESSFUL but high-energy selection (breaches max_acceptable).
        energy = EnergyScoreResult(
            estimator_name="ols",
            energy_score=0.95,  # well above default max_acceptable 0.8
            treatment_balance_score=0.9,
            outcome_fit_score=0.95,
            propensity_calibration=0.9,
            n_samples=100,
            n_treated=50,
            n_control=50,
            computation_time_ms=1.0,
        )
        selected = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=2.0,
            ate_std=0.5,
            ate_ci_lower=1.0,
            ate_ci_upper=3.0,
            cate=np.array([2.0, 2.0]),
            energy_score_result=energy,
        )
        sr = SelectionResult(
            selected=selected,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
            all_results=[selected],
            requires_review=True,
            exceeded_max_energy_score=True,
            energy_scores={"ols": 0.95},
        )

        class _FakeSelector:
            def select(self, treatment, outcome, covariates, **kw):
                return sr

        monkeypatch.setattr(node, "_get_estimator_selector", lambda *a, **k: _FakeSelector())

        result, selection_dict, _latency = node._select_estimator_with_energy_score(
            data=__import__("pandas").DataFrame(
                {"t": [0, 1, 0, 1], "y": [1.0, 2.0, 1.5, 2.5], "c": [0.1, 0.2, 0.3, 0.4]}
            ),
            treatment="t",
            outcome="y",
            adjustment_set=["c"],
            strategy="best_energy",
        )
        assert result["requires_review"] is True
        assert result["energy_score_data"]["quality_tier"] == "unreliable"
        assert selection_dict["requires_review"] is True
        assert selection_dict["quality_tier"] == "unreliable"

    def test_no_review_when_within_threshold(self, monkeypatch):
        import numpy as np
        import pandas as pd

        from src.agents.causal_impact.nodes.estimation import EstimationNode
        from src.causal_engine.energy_score.estimator_selector import (
            EstimatorResult,
            EstimatorType,
            SelectionResult,
            SelectionStrategy,
        )
        from src.causal_engine.energy_score.score_calculator import EnergyScoreResult

        node = EstimationNode()
        energy = EnergyScoreResult(
            estimator_name="ols",
            energy_score=0.20,
            treatment_balance_score=0.2,
            outcome_fit_score=0.2,
            propensity_calibration=0.2,
            n_samples=100,
            n_treated=50,
            n_control=50,
            computation_time_ms=1.0,
        )
        selected = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=2.0,
            ate_std=0.5,
            ate_ci_lower=1.0,
            ate_ci_upper=3.0,
            cate=np.array([2.0, 2.0]),
            energy_score_result=energy,
        )
        sr = SelectionResult(
            selected=selected,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
            all_results=[selected],
            requires_review=False,
            exceeded_max_energy_score=False,
            energy_scores={"ols": 0.20},
        )

        class _FakeSelector:
            def select(self, treatment, outcome, covariates, **kw):
                return sr

        monkeypatch.setattr(node, "_get_estimator_selector", lambda *a, **k: _FakeSelector())
        result, selection_dict, _ = node._select_estimator_with_energy_score(
            data=pd.DataFrame(
                {"t": [0, 1, 0, 1], "y": [1.0, 2.0, 1.5, 2.5], "c": [0.1, 0.2, 0.3, 0.4]}
            ),
            treatment="t",
            outcome="y",
            adjustment_set=["c"],
            strategy="best_energy",
        )
        assert result["requires_review"] is False
        assert result["energy_score_data"]["quality_tier"] == "excellent"


@pytest.mark.asyncio
async def test_energy_score_selection_offloaded_to_thread(monkeypatch):
    """The CPU-bound energy-score estimator selection must run OFF the event
    loop so a multi-minute fit cannot block the gunicorn worker past --timeout
    and get it KILLED mid-run (which orphaned async jobs at status='running').
    Verifies the offload happens AND stays transparent.

    #1601: the off-load target changed from ``asyncio.to_thread`` — i.e. the
    loop's DEFAULT executor, 12 threads and outside every in-process bound — to
    the BOUNDED agent-compute pool. The original guarantee (off the loop,
    transparent) is unchanged; the destination is now bounded, so this test
    pins the pool rather than the bare thread hand-off.
    """
    import asyncio as _aio

    from src.api.dependencies import compute as _compute_mod

    node = EstimationNode()
    offloaded: list = []
    real_offload = _compute_mod.run_in_agent_compute_executor

    async def _spy(func, *args, **kwargs):
        offloaded.append(getattr(func, "__name__", str(func)))
        return await real_offload(func, *args, **kwargs)

    monkeypatch.setattr(_compute_mod, "run_in_agent_compute_executor", _spy)

    # And it must NOT fall back to the unbounded default executor.
    to_thread_calls: list = []
    real_to_thread = _aio.to_thread

    async def _to_thread_spy(func, /, *args, **kwargs):
        to_thread_calls.append(getattr(func, "__name__", str(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(_aio, "to_thread", _to_thread_spy)

    state = {
        "query": "offload test",
        "query_id": "offload-est-1",
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "confounders": ["geographic_region"],
        "data_source": "synthetic",
        "causal_graph": {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "digraph { ... }",
            "confidence": 0.85,
        },
        "parameters": {"method": "CausalForestDML"},
        "status": "pending",
        "errors": [],
        "warnings": [],
    }

    result = await node.execute(state)

    # The heavy selection ran on the BOUNDED agent-compute pool (offloaded)...
    assert "_select_estimator_with_energy_score" in offloaded
    # ...never on the loop's unbounded default executor...
    assert to_thread_calls == [], f"selection escaped to the default executor via {to_thread_calls}"
    # ...and the run still produced a real estimate (offload is transparent).
    assert "ate" in result["estimation_result"]


class TestEmptyBackdoorEstimation:
    """An empty adjustment set (RCT / exogenous treatment) must produce an
    UNADJUSTED estimate via OLS, not fail-closed. See the empty-backdoor path in
    the energy-score OLS wrapper. The estimation-node success mapping flows
    unchanged (covariates_adjusted=[], selected_estimator='ols')."""

    @pytest.mark.heavy_ml
    def test_empty_adjustment_set_yields_unadjusted_ols(self):
        import numpy as np
        import pandas as pd

        from src.agents.causal_impact.nodes.estimation import EstimationNode

        rng = np.random.RandomState(0)
        n = 1500
        t = (rng.rand(n) < 0.4).astype(int)
        y = (rng.rand(n) < (0.3 + 0.3 * t)).astype(float)
        data = pd.DataFrame({"treatment_arm": t, "adopted": y})  # ONLY t + outcome
        naive_diff = float(y[t == 1].mean() - y[t == 0].mean())

        node = EstimationNode()
        result, selection_dict, _latency = node._select_estimator_with_energy_score(
            data=data,
            treatment="treatment_arm",
            outcome="adopted",
            adjustment_set=[],  # empty backdoor
            strategy="best_energy",
        )

        # Unadjusted OLS estimate produced (NOT fail-closed).
        assert result["selected_estimator"] == "ols"
        assert result["method"] == "linear_regression"
        assert result["covariates_adjusted"] == []
        assert result["ate"] == pytest.approx(naive_diff, abs=1e-9)
        assert result["ate_ci_lower"] < result["ate"] < result["ate_ci_upper"]
        # Clean RCT estimate: a finite, good-tier energy score, not 'unreliable'.
        assert result["requires_review"] is False
        assert result["energy_score_data"]["quality_tier"] != "unreliable"
        # The naive-contrast foil agrees with the unadjusted estimate.
        if result.get("naive_ate") is not None:
            assert result["naive_ate"] == pytest.approx(naive_diff, abs=1e-9)

    @pytest.mark.heavy_ml
    def test_explicit_empty_set_not_expanded_on_wide_frame(self):
        """An EXPLICIT empty backdoor ([]) must run UNADJUSTED even when the frame
        carries extra columns — it must NOT be silently expanded to all columns
        (which would adjust an RCT/exogenous question on spurious covariates).
        Only a MISSING set (None) falls back to all-other-columns."""
        import numpy as np
        import pandas as pd

        from src.agents.causal_impact.nodes.estimation import EstimationNode

        rng = np.random.RandomState(2)
        n = 1500
        t = (rng.rand(n) < 0.4).astype(int)
        y = (rng.rand(n) < (0.3 + 0.3 * t)).astype(float)
        # A spurious extra column present in the frame but NOT in the backdoor.
        data = pd.DataFrame({"treatment_arm": t, "adopted": y, "spurious": rng.rand(n)})
        naive_diff = float(y[t == 1].mean() - y[t == 0].mean())

        node = EstimationNode()
        result, _sel, _lat = node._select_estimator_with_energy_score(
            data=data,
            treatment="treatment_arm",
            outcome="adopted",
            adjustment_set=[],  # EXPLICIT empty -> zero covariates, ignore 'spurious'
            strategy="best_energy",
        )
        assert result["selected_estimator"] == "ols"
        assert result["covariates_adjusted"] == []
        # If 'spurious' had been (wrongly) used as a covariate, the OLS coefficient
        # would differ from the pure diff-in-means.
        assert result["ate"] == pytest.approx(naive_diff, abs=1e-9)

    def test_continuous_treatment_binarized_unadjusted(self):
        """A continuous treatment with an empty backdoor is binarized at the
        median, and the unadjusted estimate equals the diff-in-means on the
        binarized arms (the same estimand the adjusted path would report)."""
        import numpy as np
        import pandas as pd

        from src.agents.causal_impact.nodes.estimation import EstimationNode

        rng = np.random.RandomState(1)
        n = 1500
        score = rng.rand(n)  # continuous treatment in [0,1)
        hi = (score > np.median(score)).astype(int)
        y = (rng.rand(n) < (0.3 + 0.3 * hi)).astype(float)
        data = pd.DataFrame({"peer_influence_score": score, "adopted": y})
        naive_diff = float(y[hi == 1].mean() - y[hi == 0].mean())

        node = EstimationNode()
        result, _sel, _lat = node._select_estimator_with_energy_score(
            data=data,
            treatment="peer_influence_score",
            outcome="adopted",
            adjustment_set=[],
            strategy="best_energy",
        )
        assert result["selected_estimator"] == "ols"
        assert result["ate"] == pytest.approx(naive_diff, abs=1e-9)


class TestDegenerateQueryGuard:
    """A degenerate query (treatment == outcome, or a node absent from the DAG)
    yields graph_builder's trivial [[]] backdoor; the estimation node must
    fail-closed rather than route it as a validated empty backdoor (codex r2)."""

    @pytest.mark.asyncio
    async def test_treatment_equals_outcome_fails_closed(self):
        from src.agents.causal_impact.nodes.estimation import EstimationNode

        node = EstimationNode()
        state: dict = {
            "query": "q",
            "query_id": "deg-1",
            "treatment_var": "x",
            "outcome_var": "x",
            "confounders": [],
            "data_source": "synthetic",
            "causal_graph": {
                "nodes": ["x", "y"],
                "edges": [],
                "treatment_nodes": ["x"],
                "outcome_nodes": ["x"],  # treatment == outcome -> degenerate
                "adjustment_sets": [[]],
                "dag_dot": "...",
                "confidence": 0.8,
            },
            "parameters": {},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert result["status"] == "failed"
        blob = f"{result.get('estimation_error', '')} {result.get('error_message', '')}"
        assert "egenerate" in blob  # "Degenerate"/"degenerate"


class TestEfficiencyBaselineEstimation:
    """#1188: an RCT question (empty adjustment set) with curated PRE-TREATMENT
    baselines runs the covariate estimators as EFFICIENCY controls (ANCOVA-style
    variance reduction): estimators fit (not skipped), the result is labeled
    adjustment_type='efficiency' with the baseline list, and the de-confounding
    set stays honestly empty."""

    def _prognostic_frame(self, n=1500, seed=0):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        t = (rng.random(n) < 0.5).astype(int)
        sev = rng.normal(0.0, 1.0, n)
        y = (0.3 * t + 1.0 * sev + rng.normal(0, 0.3, n)).astype(float)
        data = pd.DataFrame(
            {
                "control_group_flag": t,
                "action_taken": y,
                "disease_severity": sev,
                "age_at_diagnosis": rng.integers(18, 86, n).astype(float),
            }
        )
        naive = float(y[t == 1].mean() - y[t == 0].mean())
        return data, naive

    @pytest.mark.heavy_ml
    def test_baselines_flow_as_efficiency_controls_and_label(self):
        from src.agents.causal_impact.nodes.estimation import EstimationNode

        data, naive = self._prognostic_frame()
        node = EstimationNode()
        result, _sel, _lat = node._select_estimator_with_energy_score(
            data=data,
            treatment="control_group_flag",
            outcome="action_taken",
            adjustment_set=[],  # RCT: empty backdoor stays correct
            strategy="best_energy",
            baseline_covariates=["disease_severity", "age_at_diagnosis"],
        )

        assert result["adjustment_type"] == "efficiency"
        assert result["baseline_covariates_adjusted"] == [
            "disease_severity",
            "age_at_diagnosis",
        ]
        # De-confounding set stays EMPTY — baselines are not confounders.
        assert result["covariates_adjusted"] == []
        # Covariate estimators actually fit (not skipped-not-applicable).
        evaluated = result["all_estimators_evaluated"]
        assert not any(e.get("skipped") for e in evaluated), evaluated
        # The naive foil remains the raw randomized contrast.
        if result.get("naive_ate") is not None:
            assert result["naive_ate"] == pytest.approx(naive, abs=1e-9)
        # Unbiasedness: the adjusted ATE recovers the PLANTED tau=0.3. (It can
        # legitimately differ from the raw contrast by the chance-imbalance
        # correction beta*d(sev) — with a strongly prognostic baseline that
        # correction is exactly what ANCOVA is for, so compare to truth.)
        assert result["ate"] == pytest.approx(0.3, abs=0.08)

    @pytest.mark.heavy_ml
    def test_no_baselines_keeps_empty_backdoor_semantics(self):
        from src.agents.causal_impact.nodes.estimation import EstimationNode

        data, naive = self._prognostic_frame(seed=1)
        node = EstimationNode()
        result, _sel, _lat = node._select_estimator_with_energy_score(
            data=data,
            treatment="control_group_flag",
            outcome="action_taken",
            adjustment_set=[],
            strategy="best_energy",
        )
        assert result["adjustment_type"] == "none"
        assert result.get("baseline_covariates_adjusted", []) == []
        assert result["selected_estimator"] == "ols"
        assert result["ate"] == pytest.approx(naive, abs=1e-9)
