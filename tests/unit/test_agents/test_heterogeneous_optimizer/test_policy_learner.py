"""Tests for Policy Learner Node."""

import pytest

from src.agents.heterogeneous_optimizer.nodes.policy_learner import PolicyLearnerNode
from src.agents.heterogeneous_optimizer.state import (
    CATEResult,
    HeterogeneousOptimizerState,
    SegmentProfile,
)


class TestPolicyLearnerNode:
    """Test PolicyLearnerNode."""

    def _create_cate_result(
        self,
        segment_name: str,
        segment_value: str,
        cate: float,
        sample_size: int = 100,
        significant: bool = True,
        treatment_rate: float | None = 0.5,
    ) -> CATEResult:
        """Create test CATE result.

        treatment_rate=0.5 by default so legacy numeric assertions (written
        against the old flat-50% baseline) stay valid; pass None to exercise
        the loud legacy-fallback path in policy_learner.
        """
        result: CATEResult = {
            "segment_name": segment_name,
            "segment_value": segment_value,
            "cate_estimate": cate,
            "cate_ci_lower": cate - 0.1,
            "cate_ci_upper": cate + 0.1,
            "sample_size": sample_size,
            "statistical_significance": significant,
        }
        if treatment_rate is not None:
            result["treatment_rate"] = treatment_rate
        return result

    def _create_segment_profile(
        self, segment_id: str, responder_type: str, cate: float, size: int = 100
    ) -> SegmentProfile:
        """Create test segment profile."""
        return {
            "segment_id": segment_id,
            "responder_type": responder_type,
            "cate_estimate": cate,
            "defining_features": [{"variable": "test", "value": "test", "effect_size": 1.0}],
            "size": size,
            "size_percentage": 10.0,
            "recommendation": "test recommendation",
        }

    def _create_test_state(self, overall_ate: float = 0.25) -> HeterogeneousOptimizerState:
        """Create test state with analysis results."""
        cate_by_segment = {
            "hcp_specialty": [
                self._create_cate_result("hcp_specialty", "Oncology", 0.50, 200, True),  # High
                self._create_cate_result("hcp_specialty", "Cardiology", 0.30, 150, True),
                self._create_cate_result("hcp_specialty", "Primary Care", 0.10, 300, True),  # Low
            ]
        }

        high_responders = [
            self._create_segment_profile("hcp_specialty_Oncology", "high", 0.50, 200)
        ]

        low_responders = [
            self._create_segment_profile("hcp_specialty_Primary Care", "low", 0.10, 300)
        ]

        return {
            "query": "test",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["modifier1"],
            "data_source": "test",
            "filters": None,
            "n_estimators": 100,
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 10,
            "cate_by_segment": cate_by_segment,
            "overall_ate": overall_ate,
            "heterogeneity_score": 0.6,
            "feature_importance": {"modifier1": 0.5},
            "high_responders": high_responders,
            "low_responders": low_responders,
            "segment_comparison": {
                "overall_ate": overall_ate,
                "high_responder_avg_cate": 0.50,
                "low_responder_avg_cate": 0.10,
                "effect_ratio": 5.0,
                "high_responder_count": 1,
                "low_responder_count": 1,
            },
            "policy_recommendations": None,
            "expected_total_lift": None,
            "optimal_allocation_summary": None,
            "cate_plot_data": None,
            "segment_grid_data": None,
            "executive_summary": None,
            "key_insights": None,
            "estimation_latency_ms": 100,
            "analysis_latency_ms": 50,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "optimizing",
        }

    def _mild_heterogeneity_state(self, ate: float = 0.243) -> HeterogeneousOptimizerState:
        """State mirroring the real adherent_180d run: a beneficial, fairly-uniform
        effect. NO segment crosses the strict 1.5x|ATE| (=0.365) or 0.5x|ATE| (=0.121)
        bars, but segment_analyzer's relative fallback still classifies the top/bottom
        segments as high/low responders (the tiers the page displays)."""
        cate_by_segment = {
            "disease_severity_band": [
                self._create_cate_result("disease_severity_band", "high", 0.295, 1354, True),
                self._create_cate_result("disease_severity_band", "medium", 0.252, 4444, True),
                self._create_cate_result("disease_severity_band", "low", 0.199, 3945, True),
            ]
        }
        high_responders = [
            self._create_segment_profile("disease_severity_band_high", "high", 0.295, 1354),
            self._create_segment_profile("disease_severity_band_medium", "high", 0.252, 4444),
        ]
        low_responders = [
            self._create_segment_profile("disease_severity_band_low", "low", 0.199, 3945),
        ]
        state = self._create_test_state(overall_ate=ate)
        state["cate_by_segment"] = cate_by_segment
        state["high_responders"] = high_responders
        state["low_responders"] = low_responders
        return state

    @pytest.mark.asyncio
    async def test_generate_policy_recommendations(self):
        """Test policy recommendation generation."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "policy_recommendations" in result
        assert len(result["policy_recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_policy_recommendation_structure(self):
        """Test structure of policy recommendations."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        if result["policy_recommendations"]:
            rec = result["policy_recommendations"][0]
            assert "segment" in rec
            assert "current_treatment_rate" in rec
            assert "recommended_treatment_rate" in rec
            assert "expected_incremental_outcome" in rec
            assert "confidence" in rec

    @pytest.mark.asyncio
    async def test_high_responder_recommendation(self):
        """Test recommendation for high responder segment."""
        node = PolicyLearnerNode()
        state = self._create_test_state(overall_ate=0.25)

        result = await node.execute(state)

        # Find Oncology recommendation (CATE=0.50, which is 2x ATE)
        oncology_rec = [r for r in result["policy_recommendations"] if "Oncology" in r["segment"]]

        assert oncology_rec, "expected an Oncology recommendation"
        rec = oncology_rec[0]
        # High responder should have increased treatment rate
        assert rec["recommended_treatment_rate"] > rec["current_treatment_rate"]

    @pytest.mark.asyncio
    async def test_low_responder_recommendation(self):
        """A below-average but still-BENEFICIAL low responder is MAINTAINED.

        Clinically-safe policy: we never recommend pulling a drug that still helps
        (positive CATE), even from the low-responder tier. Primary Care has
        CATE=0.10 (positive) -> maintain, not decrease.
        """
        node = PolicyLearnerNode()
        state = self._create_test_state(overall_ate=0.25)

        result = await node.execute(state)

        # Find Primary Care recommendation (CATE=0.10 — positive, below average)
        primary_care_rec = [
            r for r in result["policy_recommendations"] if "Primary Care" in r["segment"]
        ]

        assert primary_care_rec, "expected a Primary Care recommendation"
        rec = primary_care_rec[0]
        # Positive-but-low responder is maintained (never pull a beneficial drug).
        assert rec["recommended_treatment_rate"] == rec["current_treatment_rate"]

    @pytest.mark.asyncio
    async def test_above_ate_gate_homogeneous_yields_zero_lift(self):
        """ABOVE-ATE GATE (user-requested): a beneficial but UNIFORM effect — every
        segment's CATE CI overlaps the overall ATE — has NO segment that responds
        significantly above average, so the honest targeting lift is ~0 (count AND
        pp). Mirrors the real adherent_180d run (ate~0.243, segments 0.199-0.295,
        CIs ~±0.1 all straddling the ATE).
        """
        node = PolicyLearnerNode()
        state = self._mild_heterogeneity_state()  # ci_lower = cate-0.1 < ate for all

        result = await node.execute(state)

        assert result["expected_total_lift"] == 0.0
        assert result["expected_lift_pp"] == 0.0
        assert all(
            r["recommended_treatment_rate"] == r["current_treatment_rate"]
            for r in result["policy_recommendations"]
        )

    @pytest.mark.asyncio
    async def test_above_ate_gate_real_heterogeneity_yields_lift(self):
        """A segment whose CATE CI lies ENTIRELY ABOVE the ATE is a genuine
        differential responder -> increase -> positive lift. A below-average segment
        (CI overlaps the ATE) is maintained.
        """
        node = PolicyLearnerNode()
        # cate 0.50, ci_lower 0.40 > ate 0.25 -> significantly above average.
        cate_by_segment = {
            "disease_severity_band": [
                self._create_cate_result("disease_severity_band", "high", 0.50, 1000, True),
                self._create_cate_result("disease_severity_band", "low", 0.20, 1000, True),
            ]
        }
        state = self._create_test_state(overall_ate=0.25)
        state["cate_by_segment"] = cate_by_segment
        state["high_responders"] = []
        state["low_responders"] = []

        result = await node.execute(state)

        assert result["expected_total_lift"] == pytest.approx(0.2 * 0.50 * 1000, abs=0.5)
        assert result["expected_lift_pp"] > 0
        increases = [
            r
            for r in result["policy_recommendations"]
            if r["recommended_treatment_rate"] > r["current_treatment_rate"]
        ]
        assert len(increases) == 1 and "high" in increases[0]["segment"]

    @pytest.mark.asyncio
    async def test_expected_lift_not_double_counted_across_dimensions(self):
        """REGRESSION (user-reported +1289): cate_by_segment carries MULTIPLE
        dimensions partitioning the SAME cohort. expected_total_lift must be the best
        SINGLE-dimension total, NOT the cross-dimension sum.
        """
        node = PolicyLearnerNode()
        # Both segments clear the above-ATE gate (ci_lower 0.40 > ate 0.25).
        # Each dim lift = 0.2 * 0.50 * 1000 = 100.
        cate_by_segment = {
            "disease_severity_band": [
                self._create_cate_result("disease_severity_band", "high", 0.50, 1000, True),
            ],
            "age_band": [
                self._create_cate_result("age_band", ">65", 0.50, 1000, True),
            ],
        }
        state = self._create_test_state(overall_ate=0.25)
        state["cate_by_segment"] = cate_by_segment
        state["high_responders"] = []
        state["low_responders"] = []

        result = await node.execute(state)

        # Best single axis = 100, NOT the cross-dimension sum 200.
        assert result["expected_total_lift"] == pytest.approx(100.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_pp_lift_is_count_over_cohort(self):
        """expected_lift_pp == best-axis count / cohort size (a percentage-point
        change in the outcome rate). cohort = one dimension's total patients."""
        node = PolicyLearnerNode()
        cate_by_segment = {
            "disease_severity_band": [
                self._create_cate_result("disease_severity_band", "high", 0.50, 1000, True),
                self._create_cate_result("disease_severity_band", "low", 0.20, 1000, True),
            ]
        }
        state = self._create_test_state(overall_ate=0.25)
        state["cate_by_segment"] = cate_by_segment
        state["high_responders"] = []
        state["low_responders"] = []

        result = await node.execute(state)

        cohort = 2000  # 1000 + 1000
        assert result["expected_lift_pp"] == pytest.approx(
            result["expected_total_lift"] / cohort, abs=1e-6
        )

    @pytest.mark.asyncio
    async def test_pp_denominator_is_best_dim_cohort_not_first_dim(self):
        """REGRESSION (review H1): the pp denominator must be the cohort of the dim
        that total_lift comes from (best_dim), NOT the FIRST dim in the dict. With
        real per-variable missingness, dimensions have DIFFERENT patient totals, so
        dividing the best-dim count by the first-dim cohort yields a wrong headline
        number. Here best_dim ('beta_band', second) has 2000 patients while the first
        dim ('alpha_band') has only 500 — so pp must be 200/2000=0.10, not 200/500.
        """
        node = PolicyLearnerNode()
        # Both clear the above-ATE gate (ci_lower 0.40 > ate 0.25).
        # alpha (first): lift = 0.2*0.50*500  = 50,  cohort 500
        # beta  (best) : lift = 0.2*0.50*2000 = 200, cohort 2000
        cate_by_segment = {
            "alpha_band": [
                self._create_cate_result("alpha_band", "a", 0.50, 500, True),
            ],
            "beta_band": [
                self._create_cate_result("beta_band", "b", 0.50, 2000, True),
            ],
        }
        state = self._create_test_state(overall_ate=0.25)
        state["cate_by_segment"] = cate_by_segment
        state["high_responders"] = []
        state["low_responders"] = []

        result = await node.execute(state)

        # best_dim = beta_band (200 > 50); denominator must be beta's 2000.
        assert result["expected_total_lift"] == pytest.approx(200.0, abs=0.5)
        assert result["expected_lift_pp"] == pytest.approx(0.10, abs=1e-6)

    @pytest.mark.asyncio
    async def test_significantly_harmful_segment_decreased(self):
        """A segment whose CATE CI lies ENTIRELY BELOW zero is a genuinely harmful
        subgroup -> decrease (minimise)."""
        node = PolicyLearnerNode()
        # cate -0.30, ci_upper = -0.20 < 0 -> significantly harmful.
        cate_by_segment = {
            "disease_severity_band": [
                self._create_cate_result("disease_severity_band", "high", -0.30, 1000, True),
            ]
        }
        state = self._create_test_state(overall_ate=0.25)
        state["cate_by_segment"] = cate_by_segment
        state["high_responders"] = []
        state["low_responders"] = []

        result = await node.execute(state)

        recs = result["policy_recommendations"]
        assert recs
        assert recs[0]["recommended_treatment_rate"] < recs[0]["current_treatment_rate"]

    @pytest.mark.asyncio
    async def test_negative_ate_uncertain_benefit_not_increased(self):
        """REGRESSION (review H2): under a net-harmful ATE (<0), a segment whose CI
        lies above the ATE but STRADDLES zero has UNCONFIRMED benefit and must NOT be
        increased. The INCREASE gate requires the CI above BOTH the ATE and zero.

        ate=-0.10; segment cate=0.05, 95% CI [-0.05, 0.15]: above the ATE but the
        CI crosses zero, so benefit is not statistically confirmed -> MAINTAIN.
        """
        node = PolicyLearnerNode()
        cate_by_segment = {
            "disease_severity_band": [
                self._create_cate_result("disease_severity_band", "mid", 0.05, 1000, True),
            ]
        }
        state = self._create_test_state(overall_ate=-0.10)
        state["cate_by_segment"] = cate_by_segment
        state["high_responders"] = []
        state["low_responders"] = []

        result = await node.execute(state)

        recs = result["policy_recommendations"]
        assert recs
        # CI straddles zero -> benefit unconfirmed -> maintained, not increased.
        assert recs[0]["recommended_treatment_rate"] == recs[0]["current_treatment_rate"]
        assert result["expected_total_lift"] == 0.0

    # (Harmful-segment decrease is covered by test_significantly_harmful_segment_decreased,
    # which sets a CATE whose CI lies entirely below 0 — the above-ATE gate's harmful
    # branch. Overriding only cate_estimate is insufficient now that the gate reads CIs.)

    @pytest.mark.asyncio
    async def test_treatment_rate_bounds(self):
        """Test treatment rates are bounded between 0.1 and 0.9."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        for rec in result["policy_recommendations"]:
            assert 0.0 <= rec["recommended_treatment_rate"] <= 1.0
            assert 0.0 <= rec["current_treatment_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_expected_lift_calculation(self):
        """Test expected lift calculation."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Expected lift should be sum of incremental outcomes
        expected_total = sum(
            r["expected_incremental_outcome"] for r in result["policy_recommendations"]
        )

        assert abs(result["expected_total_lift"] - expected_total) < 0.01

    @pytest.mark.asyncio
    async def test_confidence_calculation(self):
        """Test confidence calculation."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        for rec in result["policy_recommendations"]:
            assert 0.0 <= rec["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_increases_with_sample_size(self):
        """Test confidence increases with sample size."""
        node = PolicyLearnerNode()

        # Create two similar segments with different sample sizes
        result_small = self._create_cate_result(
            "test", "small", 0.30, sample_size=50, significant=False
        )
        result_large = self._create_cate_result(
            "test", "large", 0.30, sample_size=500, significant=False
        )

        rec_small = node._generate_recommendation(result_small, 0.25)
        rec_large = node._generate_recommendation(result_large, 0.25)

        # Larger sample should have higher confidence
        assert rec_large["confidence"] > rec_small["confidence"]

    @pytest.mark.asyncio
    async def test_confidence_increases_with_significance(self):
        """Test confidence increases with statistical significance."""
        node = PolicyLearnerNode()

        result_sig = self._create_cate_result(
            "test", "sig", 0.30, sample_size=100, significant=True
        )
        result_nonsig = self._create_cate_result(
            "test", "nonsig", 0.30, sample_size=100, significant=False
        )

        rec_sig = node._generate_recommendation(result_sig, 0.25)
        rec_nonsig = node._generate_recommendation(result_nonsig, 0.25)

        # Significant result should have higher confidence
        assert rec_sig["confidence"] > rec_nonsig["confidence"]

    @pytest.mark.asyncio
    async def test_recommendations_sorted_by_lift(self):
        """Test recommendations sorted by expected incremental outcome."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        recs = result["policy_recommendations"]
        for i in range(len(recs) - 1):
            assert (
                recs[i]["expected_incremental_outcome"]
                >= recs[i + 1]["expected_incremental_outcome"]
            )

    @pytest.mark.asyncio
    async def test_top_20_recommendations(self):
        """Test limiting to top 20 recommendations."""
        node = PolicyLearnerNode()

        # Create state with many segments
        cate_by_segment = {
            "segment1": [
                self._create_cate_result("segment1", f"value_{i}", 0.30, 100) for i in range(30)
            ]
        }

        state = self._create_test_state()
        state["cate_by_segment"] = cate_by_segment

        result = await node.execute(state)

        # Should be limited to 20
        assert len(result["policy_recommendations"]) <= 20

    @pytest.mark.asyncio
    async def test_allocation_summary_generation(self):
        """Test allocation summary generation."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "optimal_allocation_summary" in result
        assert isinstance(result["optimal_allocation_summary"], str)
        assert len(result["optimal_allocation_summary"]) > 0

    @pytest.mark.asyncio
    async def test_total_latency_calculation(self):
        """Test total latency calculation."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Should sum all latencies
        assert result["total_latency_ms"] > 0
        assert result["total_latency_ms"] >= result["estimation_latency_ms"]
        assert result["total_latency_ms"] >= result["analysis_latency_ms"]

    @pytest.mark.asyncio
    async def test_status_update(self):
        """Test status update to completed."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_failed_status_passthrough(self):
        """Test failed status is passed through."""
        node = PolicyLearnerNode()
        state = self._create_test_state()
        state["status"] = "failed"

        result = await node.execute(state)

        assert result["status"] == "failed"


class TestPolicyLearnerEdgeCases:
    """Test edge cases for policy learner."""

    def _create_cate_result(self, segment_name: str, segment_value: str, cate: float):
        """Create test CATE result."""
        return {
            "segment_name": segment_name,
            "segment_value": segment_value,
            "cate_estimate": cate,
            "cate_ci_lower": cate - 0.1,
            "cate_ci_upper": cate + 0.1,
            "sample_size": 100,
            "statistical_significance": True,
        }

    def _create_test_state(self, cate_by_segment, overall_ate=0.25):
        """Create test state."""
        return {
            "query": "test",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "segment_vars": list(cate_by_segment.keys()),
            "effect_modifiers": ["modifier1"],
            "data_source": "test",
            "filters": None,
            "n_estimators": 100,
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 10,
            "cate_by_segment": cate_by_segment,
            "overall_ate": overall_ate,
            "heterogeneity_score": 0.6,
            "feature_importance": {},
            "high_responders": [],
            "low_responders": [],
            "segment_comparison": {},
            "policy_recommendations": None,
            "expected_total_lift": None,
            "optimal_allocation_summary": None,
            "cate_plot_data": None,
            "segment_grid_data": None,
            "executive_summary": None,
            "key_insights": None,
            "estimation_latency_ms": 100,
            "analysis_latency_ms": 50,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "optimizing",
        }

    @pytest.mark.asyncio
    async def test_zero_cate(self):
        """Test with zero CATE estimate."""
        node = PolicyLearnerNode()

        cate_by_segment = {"segment1": [self._create_cate_result("segment1", "value1", 0.0)]}
        state = self._create_test_state(cate_by_segment)

        result = await node.execute(state)

        # Should still generate recommendation
        assert len(result["policy_recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_negative_cate(self):
        """A genuinely HARMFUL segment (CATE CI entirely below 0) is minimised."""
        node = PolicyLearnerNode()

        # cate -0.20, ci_upper = -0.10 < 0 -> significantly harmful (above-ATE gate's
        # harmful branch). A merely-noisy negative (CI straddling 0) would maintain.
        cate_by_segment = {"segment1": [self._create_cate_result("segment1", "value1", -0.20)]}
        state = self._create_test_state(cate_by_segment)

        result = await node.execute(state)

        # Should recommend minimal treatment for a genuinely harmful effect.
        assert result["policy_recommendations"]
        rec = result["policy_recommendations"][0]
        assert rec["recommended_treatment_rate"] <= 0.1

    @pytest.mark.asyncio
    async def test_very_high_cate(self):
        """A segment whose CATE CI is far above the ATE is increased."""
        node = PolicyLearnerNode()

        # cate 5.0, ci_lower 4.9 > ate 1.0 -> significantly above average.
        cate_by_segment = {"segment1": [self._create_cate_result("segment1", "value1", 5.0)]}
        state = self._create_test_state(cate_by_segment, overall_ate=1.0)

        result = await node.execute(state)

        # Should recommend high treatment rate
        assert result["policy_recommendations"]
        rec = result["policy_recommendations"][0]
        assert rec["recommended_treatment_rate"] >= 0.7


class TestObservedTreatmentRate:
    """The current rate is the segment's OBSERVED treated share, not a flat 50%."""

    def _result(self, cate: float, treatment_rate: float | None, **kw) -> CATEResult:
        result: CATEResult = {
            "segment_name": "insurance_type",
            "segment_value": "medicaid",
            "cate_estimate": cate,
            "cate_ci_lower": kw.get("ci_lower", cate - 0.1),
            "cate_ci_upper": kw.get("ci_upper", cate + 0.1),
            "sample_size": kw.get("sample_size", 1000),
            "statistical_significance": True,
        }
        if treatment_rate is not None:
            result["treatment_rate"] = treatment_rate
        return result

    def test_maintain_reports_observed_baseline(self):
        """MAINTAIN keeps the segment at its measured rate (CI overlaps the ATE)."""
        node = PolicyLearnerNode()
        rec = node._generate_recommendation(self._result(0.25, 0.35), ate=0.25)
        assert rec["current_treatment_rate"] == pytest.approx(0.35)
        assert rec["recommended_treatment_rate"] == pytest.approx(0.35)
        assert rec["expected_incremental_outcome"] == pytest.approx(0.0)

    def test_increase_steps_up_from_observed_baseline(self):
        """INCREASE = observed + 0.2, and the lift uses the real rate change."""
        node = PolicyLearnerNode()
        # ci_lower 0.40 > ate 0.25 and > 0 -> significantly above average.
        rec = node._generate_recommendation(self._result(0.50, 0.35), ate=0.25)
        assert rec["current_treatment_rate"] == pytest.approx(0.35)
        assert rec["recommended_treatment_rate"] == pytest.approx(0.55)
        assert rec["expected_incremental_outcome"] == pytest.approx(0.2 * 0.50 * 1000)

    def test_increase_capped_at_ceiling(self):
        node = PolicyLearnerNode()
        rec = node._generate_recommendation(self._result(0.50, 0.85), ate=0.25)
        assert rec["recommended_treatment_rate"] == pytest.approx(0.9)

    def test_increase_never_flips_direction_above_ceiling(self):
        """A segment already above the 0.9 cap is maintained, not nudged down."""
        node = PolicyLearnerNode()
        rec = node._generate_recommendation(self._result(0.50, 0.95), ate=0.25)
        assert rec["recommended_treatment_rate"] == pytest.approx(0.95)
        assert rec["expected_incremental_outcome"] == pytest.approx(0.0)

    def test_decrease_lift_scales_with_observed_baseline(self):
        """DECREASE targets the 0.1 floor from the REAL rate: cutting a harmful
        segment from 15% avoids far less exposure than the fabricated 50->10."""
        node = PolicyLearnerNode()
        # ci_upper -0.10 < 0 -> significantly harmful.
        rec = node._generate_recommendation(self._result(-0.20, 0.15), ate=0.25)
        assert rec["current_treatment_rate"] == pytest.approx(0.15)
        assert rec["recommended_treatment_rate"] == pytest.approx(0.1)
        # (0.1 - 0.15) * -0.20 * 1000 = +10 avoided-harm lift.
        assert rec["expected_incremental_outcome"] == pytest.approx(10.0)

    def test_decrease_never_raises_a_segment_below_the_floor(self):
        """A harmful segment already under 10% treated must NOT be raised to 0.1."""
        node = PolicyLearnerNode()
        rec = node._generate_recommendation(self._result(-0.20, 0.05), ate=0.25)
        assert rec["recommended_treatment_rate"] == pytest.approx(0.05)
        assert rec["expected_incremental_outcome"] == pytest.approx(0.0)

    def test_missing_rate_falls_back_loudly_to_legacy_half(self, caplog):
        """Results predating the field use the legacy 0.5 — with a warning."""
        import logging

        node = PolicyLearnerNode()
        with caplog.at_level(logging.WARNING):
            rec = node._generate_recommendation(self._result(0.25, None), ate=0.25)
        assert rec["current_treatment_rate"] == pytest.approx(0.5)
        assert any("treatment_rate" in r.message for r in caplog.records)
