"""Tests for ROI Calculator Node.

Updated to test the new ROICalculationService-integrated implementation.
"""

import pytest

from src.agents.gap_analyzer.nodes.roi_calculator import ROICalculatorNode
from src.agents.gap_analyzer.state import GapAnalyzerState, PerformanceGap
from src.services.roi_calculation import (
    AttributionLevel,
    ROICalculationService,
    ValueDriverType,
)


class TestROICalculatorNode:
    """Test ROICalculatorNode."""

    def _create_test_gap(
        self, metric: str = "trx", gap_size: float = 100.0, gap_percentage: float = 20.0
    ) -> PerformanceGap:
        """Create test performance gap."""
        return {
            "gap_id": f"region_Northeast_{metric}_vs_target",
            "metric": metric,
            "segment": "region",
            "segment_value": "Northeast",
            "current_value": 400.0,
            "target_value": 500.0,
            "gap_size": gap_size,
            "gap_percentage": gap_percentage,
            "gap_type": "vs_target",
        }

    def _create_test_state(self, gaps: list) -> GapAnalyzerState:
        """Create test state with gaps."""
        return {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": "vs_target",
            "min_gap_threshold": 5.0,
            "max_opportunities": 10,
            "gaps_detected": gaps,
            "gaps_by_segment": None,
            "total_gap_value": None,
            "roi_estimates": None,
            "total_addressable_value": None,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 100,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 1,
            "errors": [],
            "warnings": [],
            "status": "calculating",
        }

    @pytest.mark.asyncio
    async def test_calculate_roi_for_trx_gap(self):
        """Test ROI calculation for TRx gap."""
        # Use fewer simulations for faster tests
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0)
        state = self._create_test_state(gaps=[gap])

        result = await node.execute(state)

        assert "roi_estimates" in result
        assert len(result["roi_estimates"]) == 1

        roi = result["roi_estimates"][0]
        assert roi["gap_id"] == gap["gap_id"]
        assert roi["estimated_revenue_impact"] > 0
        assert roi["estimated_cost_to_close"] > 0
        assert roi["expected_roi"] >= 0

    @pytest.mark.asyncio
    async def test_roi_calculation_formula(self):
        """Test ROI calculation uses proper methodology.

        New methodology uses:
        - $850/TRx value driver (not $500)
        - Attribution framework (full for vs_target = 100%)
        - Risk adjustment
        """
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0)

        roi = node._calculate_roi(gap)

        # Revenue = 100 TRx × $850 × 100% attribution (full for vs_target) = $85,000
        # After risk adjustment it may be different
        assert roi["estimated_revenue_impact"] == 85000.0  # Full attribution

        # Cost includes engineering + potentially change management
        # 5 base days × 1.0 scale (gap 10-100) × $2,500 = $12,500 engineering
        assert roi["estimated_cost_to_close"] > 0

        # ROI should be positive for this gap
        assert roi["expected_roi"] > 0

    @pytest.mark.asyncio
    async def test_value_driver_mapping(self):
        """Test different metrics map to correct value drivers."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)

        # TRx maps to TRX_LIFT
        assert node._get_value_driver("trx") == ValueDriverType.TRX_LIFT
        assert node._get_value_driver("nrx") == ValueDriverType.TRX_LIFT

        # Patient identification
        assert node._get_value_driver("patient_count") == ValueDriverType.PATIENT_IDENTIFICATION

        # Action rate
        assert node._get_value_driver("trigger_acceptance") == ValueDriverType.ACTION_RATE

        # ITP
        assert node._get_value_driver("hcp_engagement_score") == ValueDriverType.INTENT_TO_PRESCRIBE

    @pytest.mark.asyncio
    async def test_attribution_by_gap_type(self):
        """Test attribution level based on gap type."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)

        # vs_target = FULL (100%)
        assert node._determine_attribution("vs_target") == AttributionLevel.FULL

        # vs_benchmark = PARTIAL (65%)
        assert node._determine_attribution("vs_benchmark") == AttributionLevel.PARTIAL

        # vs_potential = SHARED (35%)
        assert node._determine_attribution("vs_potential") == AttributionLevel.SHARED

        # temporal = MINIMAL (10%)
        assert node._determine_attribution("temporal") == AttributionLevel.MINIMAL

    @pytest.mark.asyncio
    async def test_payback_period_calculation(self):
        """Test payback period calculation."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0)

        roi = node._calculate_roi(gap)

        assert "payback_period_months" in roi
        assert 1 <= roi["payback_period_months"] <= 24
        assert isinstance(roi["payback_period_months"], int)

    @pytest.mark.asyncio
    async def test_confidence_interval(self):
        """Test confidence interval from bootstrap."""
        service = ROICalculationService(n_simulations=100, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0)

        roi = node._calculate_roi(gap)

        assert "confidence_interval" in roi
        ci = roi["confidence_interval"]
        assert ci is not None

        # CI structure
        assert "lower_bound" in ci
        assert "median" in ci
        assert "upper_bound" in ci
        assert "probability_positive" in ci
        assert "probability_target" in ci

        # Lower < median < upper
        assert ci["lower_bound"] <= ci["median"] <= ci["upper_bound"]

        # Probabilities in [0, 1]
        assert 0.0 <= ci["probability_positive"] <= 1.0
        assert 0.0 <= ci["probability_target"] <= 1.0

    @pytest.mark.asyncio
    async def test_legacy_confidence_calculation(self):
        """Test legacy confidence score calculation."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0, gap_percentage=20.0)

        confidence = node._calculate_legacy_confidence(gap)

        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_legacy_confidence_factors(self):
        """Test legacy confidence factors."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)

        # Large gap should have higher confidence
        large_gap = self._create_test_gap(gap_size=200.0, gap_percentage=30.0)
        small_gap = self._create_test_gap(gap_size=5.0, gap_percentage=5.0)

        large_conf = node._calculate_legacy_confidence(large_gap)
        small_conf = node._calculate_legacy_confidence(small_gap)

        assert large_conf >= small_conf

    @pytest.mark.asyncio
    async def test_gap_type_legacy_confidence(self):
        """Test legacy confidence by gap type."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)

        gap_vs_target = self._create_test_gap(gap_size=100.0)
        gap_vs_target["gap_type"] = "vs_target"

        gap_temporal = self._create_test_gap(gap_size=100.0)
        gap_temporal["gap_type"] = "temporal"

        conf_target = node._calculate_legacy_confidence(gap_vs_target)
        conf_temporal = node._calculate_legacy_confidence(gap_temporal)

        # vs_target should have higher confidence than temporal
        assert conf_target > conf_temporal

    @pytest.mark.asyncio
    async def test_assumptions_generation(self):
        """Test that assumptions are generated."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx")

        roi = node._calculate_roi(gap)

        assert "assumptions" in roi
        assert isinstance(roi["assumptions"], list)
        assert len(roi["assumptions"]) > 0

    @pytest.mark.asyncio
    async def test_assumptions_content(self):
        """Test assumptions content includes methodology details."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx")

        roi = node._calculate_roi(gap)
        assumptions = roi["assumptions"]

        # Should mention value driver
        assert any("value driver" in a.lower() for a in assumptions)

        # Should mention unit value ($850/TRx for trx)
        assert any("$850" in a for a in assumptions)

        # Should mention attribution
        assert any("attribution" in a.lower() for a in assumptions)

        # Should mention bootstrap
        assert any("bootstrap" in a.lower() for a in assumptions)

    @pytest.mark.asyncio
    async def test_total_addressable_value(self):
        """Test total addressable value calculation."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gaps = [
            self._create_test_gap(metric="trx", gap_size=100.0),
            self._create_test_gap(metric="nrx", gap_size=50.0),
        ]
        state = self._create_test_state(gaps=gaps)

        result = await node.execute(state)

        assert "total_addressable_value" in result
        assert result["total_addressable_value"] > 0

        # Should equal sum of individual revenue impacts
        manual_total = sum(est["estimated_revenue_impact"] for est in result["roi_estimates"])
        assert abs(result["total_addressable_value"] - manual_total) < 0.01

    @pytest.mark.asyncio
    async def test_roi_latency_measurement(self):
        """Test ROI latency measurement."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap()
        state = self._create_test_state(gaps=[gap])

        result = await node.execute(state)

        assert "roi_latency_ms" in result
        assert result["roi_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_status_update(self):
        """Test status update to prioritizing."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap()
        state = self._create_test_state(gaps=[gap])

        result = await node.execute(state)

        assert result["status"] == "prioritizing"

    @pytest.mark.asyncio
    async def test_multiple_gaps(self):
        """Test ROI calculation for multiple gaps."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gaps = [
            self._create_test_gap(metric="trx", gap_size=100.0),
            self._create_test_gap(metric="nrx", gap_size=50.0),
            self._create_test_gap(metric="market_share", gap_size=5.0),
        ]
        state = self._create_test_state(gaps=gaps)

        result = await node.execute(state)

        assert len(result["roi_estimates"]) == 3

        # Each gap should have ROI estimate
        gap_ids = {g["gap_id"] for g in gaps}
        roi_gap_ids = {r["gap_id"] for r in result["roi_estimates"]}
        assert gap_ids == roi_gap_ids

    @pytest.mark.asyncio
    async def test_risk_adjusted_roi(self):
        """Test risk-adjusted ROI is calculated."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0)

        roi = node._calculate_roi(gap)

        assert "risk_adjusted_roi" in roi
        assert "total_risk_adjustment" in roi

        # Risk-adjusted should be <= base ROI (risk reduces returns)
        assert roi["risk_adjusted_roi"] <= roi["expected_roi"]

    @pytest.mark.asyncio
    async def test_value_by_driver(self):
        """Test value breakdown by driver is included."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0)

        roi = node._calculate_roi(gap)

        assert "value_by_driver" in roi
        assert roi["value_by_driver"] is not None
        assert "trx_lift" in roi["value_by_driver"]


class TestROICalculatorEdgeCases:
    """Test edge cases for ROI calculator."""

    def _create_test_gap(self, metric: str = "trx", gap_size: float = 100.0) -> PerformanceGap:
        """Create test gap."""
        return {
            "gap_id": f"region_Northeast_{metric}_vs_target",
            "metric": metric,
            "segment": "region",
            "segment_value": "Northeast",
            "current_value": 400.0,
            "target_value": 500.0,
            "gap_size": gap_size,
            "gap_percentage": 20.0,
            "gap_type": "vs_target",
        }

    def _create_test_state(self, gaps: list) -> GapAnalyzerState:
        """Create test state."""
        return {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": "vs_target",
            "min_gap_threshold": 5.0,
            "max_opportunities": 10,
            "gaps_detected": gaps,
            "gaps_by_segment": None,
            "total_gap_value": None,
            "roi_estimates": None,
            "total_addressable_value": None,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 100,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 1,
            "errors": [],
            "warnings": [],
            "status": "calculating",
        }

    @pytest.mark.asyncio
    async def test_zero_size_gap(self):
        """Test gap with zero size still calculates (edge case)."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(gap_size=0.0)

        roi = node._calculate_roi(gap)

        # Zero gap = zero revenue impact
        assert roi["estimated_revenue_impact"] == 0.0
        # But engineering cost still applies (minimum effort)
        assert roi["estimated_cost_to_close"] > 0

    @pytest.mark.asyncio
    async def test_very_large_gap(self):
        """Test very large gap."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(gap_size=10000.0)

        roi = node._calculate_roi(gap)

        assert roi["estimated_revenue_impact"] > 0
        assert roi["estimated_cost_to_close"] > 0
        # Large gap should have meaningful ROI
        assert roi["expected_roi"] > 0

    @pytest.mark.asyncio
    async def test_unknown_metric(self):
        """Test gap with unknown metric uses default (TRX_LIFT)."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="unknown_metric", gap_size=100.0)

        roi = node._calculate_roi(gap)

        # Should use default TRX_LIFT driver
        assert roi["estimated_revenue_impact"] > 0
        assert roi["estimated_cost_to_close"] > 0

    @pytest.mark.asyncio
    async def test_no_gaps_detected(self):
        """Test ROI calculator with no gaps."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        state = self._create_test_state(gaps=[])

        result = await node.execute(state)

        assert result["roi_estimates"] == []
        assert result["total_addressable_value"] == 0.0
        assert "warnings" in result

    @pytest.mark.asyncio
    async def test_extreme_gap_percentage(self):
        """Test gap with extreme percentage reduces legacy confidence."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(gap_size=100.0)
        gap["gap_percentage"] = 200.0  # Extreme gap

        confidence = node._calculate_legacy_confidence(gap)

        # Very large gap % should reduce confidence
        assert confidence < 0.8

    @pytest.mark.asyncio
    async def test_payback_period_capped(self):
        """Test that payback period is capped at 24 months."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)

        # Create gap with very low revenue
        gap = self._create_test_gap(gap_size=1.0)

        roi = node._calculate_roi(gap)

        assert roi["payback_period_months"] <= 24

    @pytest.mark.asyncio
    async def test_different_gap_types_affect_attribution(self):
        """Test that different gap types get different attribution."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)

        # vs_target gets FULL (100%)
        gap_target = self._create_test_gap(gap_size=100.0)
        gap_target["gap_type"] = "vs_target"

        # temporal gets MINIMAL (10%)
        gap_temporal = self._create_test_gap(gap_size=100.0)
        gap_temporal["gap_type"] = "temporal"

        roi_target = node._calculate_roi(gap_target)
        roi_temporal = node._calculate_roi(gap_temporal)

        # Same gap size but different attribution
        # Full attribution: 100% × $85,000 = $85,000
        # Minimal attribution: 10% × $85,000 = $8,500
        assert roi_target["estimated_revenue_impact"] > roi_temporal["estimated_revenue_impact"]
        assert roi_target["attribution_rate"] > roi_temporal["attribution_rate"]


class TestROICalculatorIntegration:
    """Integration tests for ROI calculator with ROICalculationService."""

    def _create_test_gap(self, metric: str = "trx", gap_size: float = 100.0) -> PerformanceGap:
        """Create test gap."""
        return {
            "gap_id": f"region_Northeast_{metric}_vs_target",
            "metric": metric,
            "segment": "region",
            "segment_value": "Northeast",
            "current_value": 400.0,
            "target_value": 500.0,
            "gap_size": gap_size,
            "gap_percentage": 20.0,
            "gap_type": "vs_target",
        }

    @pytest.mark.asyncio
    async def test_reproducible_with_seed(self):
        """Test that results are reproducible with same seed."""
        service1 = ROICalculationService(n_simulations=100, seed=42)
        service2 = ROICalculationService(n_simulations=100, seed=42)

        node1 = ROICalculatorNode(roi_service=service1)
        node2 = ROICalculatorNode(roi_service=service2)

        gap = self._create_test_gap(gap_size=100.0)

        roi1 = node1._calculate_roi(gap)
        roi2 = node2._calculate_roi(gap)

        # With same seed, should get same results
        assert roi1["expected_roi"] == roi2["expected_roi"]
        assert roi1["risk_adjusted_roi"] == roi2["risk_adjusted_roi"]
        if roi1["confidence_interval"] and roi2["confidence_interval"]:
            assert roi1["confidence_interval"]["median"] == roi2["confidence_interval"]["median"]

    @pytest.mark.asyncio
    async def test_custom_simulations(self):
        """Test custom number of simulations."""
        # Fewer simulations for speed
        service = ROICalculationService(n_simulations=10, seed=42)
        node = ROICalculatorNode(roi_service=service)

        gap = self._create_test_gap(gap_size=100.0)
        roi = node._calculate_roi(gap)

        # Should still produce valid results
        assert roi["expected_roi"] > 0
        assert roi["confidence_interval"] is not None

    @pytest.mark.asyncio
    async def test_roi_output_structure(self):
        """Test complete ROI output structure."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)

        gap = self._create_test_gap(gap_size=100.0)
        roi = node._calculate_roi(gap)

        # Required fields
        required_fields = [
            "gap_id",
            "estimated_revenue_impact",
            "estimated_cost_to_close",
            "expected_roi",
            "risk_adjusted_roi",
            "payback_period_months",
            "confidence_interval",
            "attribution_level",
            "attribution_rate",
            "total_risk_adjustment",
            "value_by_driver",
            "confidence",
            "assumptions",
        ]

        for field in required_fields:
            assert field in roi, f"Missing field: {field}"


class TestROICalculatorUpliftIntegration:
    """Tests for ROI calculator uplift integration (Phase B6).

    Tests the integration of CausalML uplift models with ROI calculations,
    enabling targeting optimization value to be included in ROI estimates.
    """

    def _create_test_gap(
        self, metric: str = "trx", gap_size: float = 100.0, gap_percentage: float = 20.0
    ) -> PerformanceGap:
        """Create test performance gap."""
        return {
            "gap_id": f"region_Northeast_{metric}_vs_target",
            "metric": metric,
            "segment": "region",
            "segment_value": "Northeast",
            "current_value": 400.0,
            "target_value": 500.0,
            "gap_size": gap_size,
            "gap_percentage": gap_percentage,
            "gap_type": "vs_target",
        }

    def _create_test_state_with_uplift(
        self,
        gaps: list,
        auuc: float = 0.7,
        qini: float = 0.65,
        efficiency: float = 0.8,
    ) -> GapAnalyzerState:
        """Create test state with uplift context fields."""
        return {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": "vs_target",
            "min_gap_threshold": 5.0,
            "max_opportunities": 10,
            # Uplift context fields
            "uplift_auuc": auuc,
            "uplift_qini": qini,
            "uplift_targeting_efficiency": efficiency,
            "uplift_by_segment": {
                "Northeast": [{"segment_value": "Northeast", "mean_uplift_score": 0.75}]
            },
            # Gap detection
            "gaps_detected": gaps,
            "gaps_by_segment": None,
            "total_gap_value": None,
            "roi_estimates": None,
            "total_addressable_value": None,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 100,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 1,
            "errors": [],
            "warnings": [],
            "status": "calculating",
        }

    def test_extract_uplift_context_with_data(self):
        """Test extracting uplift context from state with full data."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        state = self._create_test_state_with_uplift(
            gaps=[],
            auuc=0.75,
            qini=0.70,
            efficiency=0.85,
        )

        context = node._extract_uplift_context(state)

        assert context is not None
        assert context["auuc"] == 0.75
        assert context["qini_coefficient"] == 0.70
        assert context["targeting_efficiency"] == 0.85
        assert context["uplift_by_segment"] is not None

    def test_extract_uplift_context_none_when_missing(self):
        """Test uplift context returns None when no uplift data in state."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        # State without uplift fields
        state: GapAnalyzerState = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": "vs_target",
            "min_gap_threshold": 5.0,
            "max_opportunities": 10,
            "gaps_detected": [],
            "gaps_by_segment": None,
            "total_gap_value": None,
            "roi_estimates": None,
            "total_addressable_value": None,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 100,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 1,
            "errors": [],
            "warnings": [],
            "status": "calculating",
        }

        context = node._extract_uplift_context(state)

        assert context is None

    def test_extract_uplift_context_defaults(self):
        """Test uplift context fills defaults when partially available."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        # State with only AUUC (no qini or efficiency)
        state = self._create_test_state_with_uplift(gaps=[])
        state["uplift_auuc"] = 0.65
        state["uplift_qini"] = None
        state["uplift_targeting_efficiency"] = None

        context = node._extract_uplift_context(state)

        assert context is not None
        assert context["auuc"] == 0.65
        assert context["qini_coefficient"] is None
        assert context["targeting_efficiency"] == 0.5  # Default

    def test_create_uplift_value_driver_for_targeting_metric(self):
        """Test creating uplift value driver for metrics that benefit from targeting."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0)
        uplift_context = {
            "auuc": 0.7,
            "qini_coefficient": 0.65,
            "targeting_efficiency": 0.8,
            "uplift_by_segment": None,
        }

        driver = node._create_uplift_value_driver(gap, uplift_context)

        assert driver is not None
        assert driver.driver_type == ValueDriverType.UPLIFT_TARGETING
        assert driver.auuc == 0.7
        assert driver.targeting_efficiency == 0.8
        assert driver.baseline_treatment_value == 85000.0  # 100 * 850
        assert driver.targeted_population_size == 100

    def test_create_uplift_value_driver_none_for_non_targeting_metric(self):
        """Test uplift value driver returns None for metrics that don't benefit."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        # data_quality doesn't benefit from targeting
        gap = self._create_test_gap(metric="data_quality", gap_size=100.0)
        uplift_context = {
            "auuc": 0.7,
            "qini_coefficient": 0.65,
            "targeting_efficiency": 0.8,
            "uplift_by_segment": None,
        }

        driver = node._create_uplift_value_driver(gap, uplift_context)

        assert driver is None

    def test_value_driver_mapping_targeting_metrics(self):
        """Test targeting efficiency and uplift score map to UPLIFT_TARGETING."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)

        assert node._get_value_driver("targeting_efficiency") == ValueDriverType.UPLIFT_TARGETING
        assert node._get_value_driver("uplift_score") == ValueDriverType.UPLIFT_TARGETING

    @pytest.mark.asyncio
    async def test_roi_calculation_with_uplift_context(self):
        """Test that ROI calculation includes uplift value when context available."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0)

        # Calculate ROI without uplift context
        roi_without_uplift = node._calculate_roi(gap, uplift_context=None)

        # Calculate ROI with uplift context
        uplift_context = {
            "auuc": 0.7,
            "qini_coefficient": 0.65,
            "targeting_efficiency": 0.8,
            "uplift_by_segment": None,
        }
        roi_with_uplift = node._calculate_roi(gap, uplift_context=uplift_context)

        # With uplift context, revenue impact should be higher
        assert roi_with_uplift["estimated_revenue_impact"] >= roi_without_uplift["estimated_revenue_impact"]

    @pytest.mark.asyncio
    async def test_execute_with_uplift_state(self):
        """Test full execute workflow with uplift context in state."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0)
        state = self._create_test_state_with_uplift(
            gaps=[gap],
            auuc=0.7,
            qini=0.65,
            efficiency=0.8,
        )

        result = await node.execute(state)

        assert "roi_estimates" in result
        assert len(result["roi_estimates"]) == 1
        assert result["roi_estimates"][0]["estimated_revenue_impact"] > 0
        assert result["total_addressable_value"] > 0

    @pytest.mark.asyncio
    async def test_execute_without_uplift_state(self):
        """Test execute workflow still works without uplift context."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gap = self._create_test_gap(metric="trx", gap_size=100.0)
        # State without uplift fields
        state: GapAnalyzerState = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": "vs_target",
            "min_gap_threshold": 5.0,
            "max_opportunities": 10,
            "gaps_detected": [gap],
            "gaps_by_segment": None,
            "total_gap_value": None,
            "roi_estimates": None,
            "total_addressable_value": None,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 100,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 1,
            "errors": [],
            "warnings": [],
            "status": "calculating",
        }

        result = await node.execute(state)

        assert "roi_estimates" in result
        assert len(result["roi_estimates"]) == 1
        assert result["roi_estimates"][0]["estimated_revenue_impact"] == 85000.0  # Base TRx only

    @pytest.mark.asyncio
    async def test_multiple_gaps_with_uplift_context(self):
        """Test multiple gaps all receive uplift context benefit."""
        service = ROICalculationService(n_simulations=50, seed=42)
        node = ROICalculatorNode(roi_service=service)
        gaps = [
            self._create_test_gap(metric="trx", gap_size=100.0),
            self._create_test_gap(metric="patient_count", gap_size=50.0),
            self._create_test_gap(metric="data_quality", gap_size=20.0),  # Won't get uplift
        ]
        state = self._create_test_state_with_uplift(
            gaps=gaps,
            auuc=0.7,
            qini=0.65,
            efficiency=0.8,
        )

        result = await node.execute(state)

        assert len(result["roi_estimates"]) == 3
        # All gaps should have valid ROI estimates
        for roi in result["roi_estimates"]:
            assert roi["estimated_revenue_impact"] >= 0
            assert roi["estimated_cost_to_close"] > 0
