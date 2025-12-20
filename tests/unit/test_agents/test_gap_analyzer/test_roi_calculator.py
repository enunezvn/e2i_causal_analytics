"""Tests for ROI Calculator Node."""

import pytest
from src.agents.gap_analyzer.nodes.roi_calculator import ROICalculatorNode
from src.agents.gap_analyzer.state import GapAnalyzerState, PerformanceGap


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
        node = ROICalculatorNode()
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
        """Test ROI calculation formula."""
        node = ROICalculatorNode()
        gap = self._create_test_gap(metric="trx", gap_size=100.0)

        roi = node._calculate_roi(gap)

        # Revenue = gap_size * metric_multiplier
        expected_revenue = 100.0 * 500.0  # TRx multiplier is $500
        assert abs(roi["estimated_revenue_impact"] - expected_revenue) < 0.01

        # Cost = gap_size * intervention_cost
        expected_cost = 100.0 * 100.0  # TRx intervention is $100
        assert abs(roi["estimated_cost_to_close"] - expected_cost) < 0.01

        # ROI = (revenue - cost) / cost
        expected_roi = (expected_revenue - expected_cost) / expected_cost
        assert abs(roi["expected_roi"] - expected_roi) < 0.01

    @pytest.mark.asyncio
    async def test_metric_multipliers(self):
        """Test different metric multipliers."""
        node = ROICalculatorNode()

        metrics_to_test = ["trx", "nrx", "market_share", "conversion_rate"]

        for metric in metrics_to_test:
            multiplier = node._get_metric_multiplier(metric)
            assert multiplier > 0
            assert isinstance(multiplier, float)

    @pytest.mark.asyncio
    async def test_intervention_costs(self):
        """Test different intervention costs."""
        node = ROICalculatorNode()

        metrics_to_test = ["trx", "nrx", "market_share", "conversion_rate"]

        for metric in metrics_to_test:
            cost = node._get_intervention_cost(metric)
            assert cost > 0
            assert isinstance(cost, float)

    @pytest.mark.asyncio
    async def test_payback_period_calculation(self):
        """Test payback period calculation."""
        node = ROICalculatorNode()
        gap = self._create_test_gap(metric="trx", gap_size=100.0)

        roi = node._calculate_roi(gap)

        assert "payback_period_months" in roi
        assert 1 <= roi["payback_period_months"] <= 24
        assert isinstance(roi["payback_period_months"], int)

    @pytest.mark.asyncio
    async def test_confidence_calculation(self):
        """Test confidence score calculation."""
        node = ROICalculatorNode()
        gap = self._create_test_gap(metric="trx", gap_size=100.0, gap_percentage=20.0)

        confidence = node._calculate_confidence(gap)

        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_factors(self):
        """Test confidence factors."""
        node = ROICalculatorNode()

        # Large gap should have higher confidence
        large_gap = self._create_test_gap(gap_size=200.0, gap_percentage=30.0)
        small_gap = self._create_test_gap(gap_size=5.0, gap_percentage=5.0)

        large_conf = node._calculate_confidence(large_gap)
        small_conf = node._calculate_confidence(small_gap)

        assert large_conf >= small_conf

    @pytest.mark.asyncio
    async def test_gap_type_confidence(self):
        """Test confidence by gap type."""
        node = ROICalculatorNode()

        gap_vs_target = self._create_test_gap(gap_size=100.0)
        gap_vs_target["gap_type"] = "vs_target"

        gap_temporal = self._create_test_gap(gap_size=100.0)
        gap_temporal["gap_type"] = "temporal"

        conf_target = node._calculate_confidence(gap_vs_target)
        conf_temporal = node._calculate_confidence(gap_temporal)

        # vs_target should have higher confidence than temporal
        assert conf_target > conf_temporal

    @pytest.mark.asyncio
    async def test_assumptions_generation(self):
        """Test that assumptions are generated."""
        node = ROICalculatorNode()
        gap = self._create_test_gap(metric="trx")

        roi = node._calculate_roi(gap)

        assert "assumptions" in roi
        assert isinstance(roi["assumptions"], list)
        assert len(roi["assumptions"]) > 0

    @pytest.mark.asyncio
    async def test_assumptions_content(self):
        """Test assumptions content."""
        node = ROICalculatorNode()

        assumptions = node._get_assumptions("trx")

        # Should have base assumptions
        assert any("revenue per trx" in a.lower() for a in assumptions)
        assert any("cost per hcp" in a.lower() for a in assumptions)

        # Should have metric-specific assumptions
        assert any("trx" in a.lower() for a in assumptions)

    @pytest.mark.asyncio
    async def test_total_addressable_value(self):
        """Test total addressable value calculation."""
        node = ROICalculatorNode()
        gaps = [
            self._create_test_gap(metric="trx", gap_size=100.0),
            self._create_test_gap(metric="nrx", gap_size=50.0),
        ]
        state = self._create_test_state(gaps=gaps)

        result = await node.execute(state)

        assert "total_addressable_value" in result
        assert result["total_addressable_value"] > 0

        # Should equal sum of individual revenue impacts
        manual_total = sum(
            est["estimated_revenue_impact"] for est in result["roi_estimates"]
        )
        assert abs(result["total_addressable_value"] - manual_total) < 0.01

    @pytest.mark.asyncio
    async def test_roi_latency_measurement(self):
        """Test ROI latency measurement."""
        node = ROICalculatorNode()
        gap = self._create_test_gap()
        state = self._create_test_state(gaps=[gap])

        result = await node.execute(state)

        assert "roi_latency_ms" in result
        assert result["roi_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_status_update(self):
        """Test status update to prioritizing."""
        node = ROICalculatorNode()
        gap = self._create_test_gap()
        state = self._create_test_state(gaps=[gap])

        result = await node.execute(state)

        assert result["status"] == "prioritizing"

    @pytest.mark.asyncio
    async def test_multiple_gaps(self):
        """Test ROI calculation for multiple gaps."""
        node = ROICalculatorNode()
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
    async def test_zero_cost_gap(self):
        """Test gap with zero intervention cost."""
        node = ROICalculatorNode()
        gap = self._create_test_gap(gap_size=0.0)

        roi = node._calculate_roi(gap)

        # Zero cost should result in infinite ROI or very high ROI
        assert roi["estimated_cost_to_close"] == 0.0

    @pytest.mark.asyncio
    async def test_very_large_gap(self):
        """Test very large gap."""
        node = ROICalculatorNode()
        gap = self._create_test_gap(gap_size=10000.0)

        roi = node._calculate_roi(gap)

        assert roi["estimated_revenue_impact"] > 0
        assert roi["estimated_cost_to_close"] > 0

    @pytest.mark.asyncio
    async def test_unknown_metric(self):
        """Test gap with unknown metric."""
        node = ROICalculatorNode()
        gap = self._create_test_gap(metric="unknown_metric", gap_size=100.0)

        roi = node._calculate_roi(gap)

        # Should use default multiplier and cost
        assert roi["estimated_revenue_impact"] > 0
        assert roi["estimated_cost_to_close"] > 0

    @pytest.mark.asyncio
    async def test_no_gaps_detected(self):
        """Test ROI calculator with no gaps."""
        node = ROICalculatorNode()
        state = self._create_test_state(gaps=[])

        result = await node.execute(state)

        assert result["roi_estimates"] == []
        assert result["total_addressable_value"] == 0.0
        assert "warnings" in result

    @pytest.mark.asyncio
    async def test_extreme_gap_percentage(self):
        """Test gap with extreme percentage."""
        node = ROICalculatorNode()
        gap = self._create_test_gap(gap_size=100.0)
        gap["gap_percentage"] = 200.0  # Extreme gap

        confidence = node._calculate_confidence(gap)

        # Very large gap % should reduce confidence
        assert confidence < 0.8

    @pytest.mark.asyncio
    async def test_payback_period_capped(self):
        """Test that payback period is capped at 24 months."""
        node = ROICalculatorNode()

        # Create gap with very low revenue
        gap = self._create_test_gap(gap_size=1.0)

        roi = node._calculate_roi(gap)

        assert roi["payback_period_months"] <= 24
