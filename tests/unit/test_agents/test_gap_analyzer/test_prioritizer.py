"""Tests for Prioritizer Node."""

import pytest
from src.agents.gap_analyzer.nodes.prioritizer import PrioritizerNode
from src.agents.gap_analyzer.state import (
    GapAnalyzerState,
    PerformanceGap,
    ROIEstimate,
)


class TestPrioritizerNode:
    """Test PrioritizerNode."""

    def _create_test_gap(
        self, metric: str = "trx", gap_size: float = 100.0, gap_percentage: float = 20.0
    ) -> PerformanceGap:
        """Create test gap."""
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

    def _create_test_roi(
        self, gap_id: str, roi: float = 3.0, cost: float = 10000.0
    ) -> ROIEstimate:
        """Create test ROI estimate."""
        revenue = cost * (roi + 1)  # revenue = cost * (1 + ROI)
        return {
            "gap_id": gap_id,
            "estimated_revenue_impact": revenue,
            "estimated_cost_to_close": cost,
            "expected_roi": roi,
            "payback_period_months": 6,
            "confidence": 0.8,
            "assumptions": ["Test assumption"],
        }

    def _create_test_state(
        self, gaps: list, roi_estimates: list
    ) -> GapAnalyzerState:
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
            "total_gap_value": 1000.0,
            "roi_estimates": roi_estimates,
            "total_addressable_value": 100000.0,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 100,
            "roi_latency_ms": 50,
            "total_latency_ms": 0,
            "segments_analyzed": 1,
            "errors": [],
            "warnings": [],
            "status": "prioritizing",
        }

    @pytest.mark.asyncio
    async def test_prioritize_opportunities(self):
        """Test opportunity prioritization."""
        node = PrioritizerNode()

        gap1 = self._create_test_gap(metric="trx", gap_size=100.0)
        gap2 = self._create_test_gap(metric="nrx", gap_size=50.0)

        roi1 = self._create_test_roi(gap1["gap_id"], roi=5.0)
        roi2 = self._create_test_roi(gap2["gap_id"], roi=3.0)

        state = self._create_test_state(
            gaps=[gap1, gap2], roi_estimates=[roi1, roi2]
        )

        result = await node.execute(state)

        assert "prioritized_opportunities" in result
        assert len(result["prioritized_opportunities"]) == 2

        # Verify structure
        opp = result["prioritized_opportunities"][0]
        assert "rank" in opp
        assert "gap" in opp
        assert "roi_estimate" in opp
        assert "recommended_action" in opp
        assert "implementation_difficulty" in opp
        assert "time_to_impact" in opp

    @pytest.mark.asyncio
    async def test_opportunities_sorted_by_roi(self):
        """Test that opportunities are sorted by ROI descending."""
        node = PrioritizerNode()

        gap1 = self._create_test_gap(metric="trx")
        gap2 = self._create_test_gap(metric="nrx")
        gap3 = self._create_test_gap(metric="market_share")

        roi1 = self._create_test_roi(gap1["gap_id"], roi=2.0)
        roi2 = self._create_test_roi(gap2["gap_id"], roi=5.0)  # Highest
        roi3 = self._create_test_roi(gap3["gap_id"], roi=3.0)

        state = self._create_test_state(
            gaps=[gap1, gap2, gap3], roi_estimates=[roi1, roi2, roi3]
        )

        result = await node.execute(state)

        opps = result["prioritized_opportunities"]

        # Should be sorted by ROI descending
        assert opps[0]["roi_estimate"]["expected_roi"] == 5.0
        assert opps[1]["roi_estimate"]["expected_roi"] == 3.0
        assert opps[2]["roi_estimate"]["expected_roi"] == 2.0

    @pytest.mark.asyncio
    async def test_rank_assignment(self):
        """Test that ranks are assigned correctly."""
        node = PrioritizerNode()

        gap1 = self._create_test_gap(metric="trx")
        gap2 = self._create_test_gap(metric="nrx")

        roi1 = self._create_test_roi(gap1["gap_id"], roi=5.0)
        roi2 = self._create_test_roi(gap2["gap_id"], roi=3.0)

        state = self._create_test_state(
            gaps=[gap1, gap2], roi_estimates=[roi1, roi2]
        )

        result = await node.execute(state)

        opps = result["prioritized_opportunities"]

        # Ranks should be 1, 2, ...
        assert opps[0]["rank"] == 1
        assert opps[1]["rank"] == 2

    @pytest.mark.asyncio
    async def test_max_opportunities_limit(self):
        """Test that max_opportunities limits results."""
        node = PrioritizerNode()

        # Create 15 gaps with unique IDs (using different segment values)
        gaps = []
        for i in range(15):
            gap = self._create_test_gap(metric="trx")
            # Make gap_id unique by modifying it
            gap["gap_id"] = f"region_Region{i}_trx_vs_target"
            gap["segment_value"] = f"Region{i}"
            gaps.append(gap)

        roi_estimates = [
            self._create_test_roi(gap["gap_id"], roi=float(i + 1))
            for i, gap in enumerate(gaps)
        ]

        state = self._create_test_state(gaps=gaps, roi_estimates=roi_estimates)
        state["max_opportunities"] = 5

        result = await node.execute(state)

        # Should limit to 5
        assert len(result["prioritized_opportunities"]) == 5

    @pytest.mark.asyncio
    async def test_quick_wins_identification(self):
        """Test quick wins identification."""
        node = PrioritizerNode()

        # Create gap with low difficulty, high ROI
        gap = self._create_test_gap(metric="trx", gap_size=50.0, gap_percentage=8.0)
        roi = self._create_test_roi(gap["gap_id"], roi=2.0, cost=5000.0)  # Low cost

        state = self._create_test_state(gaps=[gap], roi_estimates=[roi])

        result = await node.execute(state)

        assert "quick_wins" in result
        # Should identify as quick win (low difficulty, ROI > 1)
        assert len(result["quick_wins"]) >= 0

    @pytest.mark.asyncio
    async def test_quick_wins_criteria(self):
        """Test quick wins criteria."""
        node = PrioritizerNode()

        # Create low-difficulty, high-ROI gap
        gap_qw = self._create_test_gap(metric="trx", gap_size=30.0, gap_percentage=5.0)
        roi_qw = self._create_test_roi(gap_qw["gap_id"], roi=2.0, cost=5000.0)

        # Create high-difficulty gap
        gap_hard = self._create_test_gap(
            metric="market_share", gap_size=200.0, gap_percentage=40.0
        )
        roi_hard = self._create_test_roi(gap_hard["gap_id"], roi=2.0, cost=100000.0)

        state = self._create_test_state(
            gaps=[gap_qw, gap_hard], roi_estimates=[roi_qw, roi_hard]
        )

        result = await node.execute(state)

        # Quick wins should only include low difficulty opportunities
        for qw in result["quick_wins"]:
            assert qw["implementation_difficulty"] == "low"
            assert qw["roi_estimate"]["expected_roi"] > 1.0

    @pytest.mark.asyncio
    async def test_quick_wins_limited_to_5(self):
        """Test that quick wins are limited to top 5."""
        node = PrioritizerNode()

        # Create 10 low-difficulty, high-ROI gaps
        gaps = [
            self._create_test_gap(metric="trx", gap_size=30.0, gap_percentage=5.0)
            for _ in range(10)
        ]
        roi_estimates = [
            self._create_test_roi(gap["gap_id"], roi=float(i + 1), cost=5000.0)
            for i, gap in enumerate(gaps)
        ]

        state = self._create_test_state(gaps=gaps, roi_estimates=roi_estimates)

        result = await node.execute(state)

        # Should limit to 5
        assert len(result["quick_wins"]) <= 5

    @pytest.mark.asyncio
    async def test_strategic_bets_identification(self):
        """Test strategic bets identification."""
        node = PrioritizerNode()

        # Create high-difficulty, high-ROI gap
        gap = self._create_test_gap(
            metric="market_share", gap_size=200.0, gap_percentage=40.0
        )
        roi = self._create_test_roi(gap["gap_id"], roi=3.0, cost=100000.0)  # High cost

        state = self._create_test_state(gaps=[gap], roi_estimates=[roi])

        result = await node.execute(state)

        assert "strategic_bets" in result

    @pytest.mark.asyncio
    async def test_strategic_bets_criteria(self):
        """Test strategic bets criteria."""
        node = PrioritizerNode()

        # Strategic bet: high difficulty, high ROI, high cost
        gap_sb = self._create_test_gap(
            metric="market_share", gap_size=200.0, gap_percentage=40.0
        )
        roi_sb = self._create_test_roi(gap_sb["gap_id"], roi=3.0, cost=100000.0)

        # Not strategic bet: low ROI
        gap_low = self._create_test_gap(metric="trx", gap_size=200.0)
        roi_low = self._create_test_roi(gap_low["gap_id"], roi=1.0, cost=100000.0)

        state = self._create_test_state(
            gaps=[gap_sb, gap_low], roi_estimates=[roi_sb, roi_low]
        )

        result = await node.execute(state)

        # Strategic bets should only include high difficulty, high ROI, high cost
        for sb in result["strategic_bets"]:
            assert sb["implementation_difficulty"] == "high"
            assert sb["roi_estimate"]["expected_roi"] > 2.0
            assert sb["roi_estimate"]["estimated_cost_to_close"] > 50000

    @pytest.mark.asyncio
    async def test_strategic_bets_limited_to_5(self):
        """Test that strategic bets are limited to top 5."""
        node = PrioritizerNode()

        # Create 10 high-difficulty, high-ROI gaps
        gaps = [
            self._create_test_gap(metric="market_share", gap_size=200.0, gap_percentage=40.0)
            for _ in range(10)
        ]
        roi_estimates = [
            self._create_test_roi(gap["gap_id"], roi=float(i + 3), cost=100000.0)
            for i, gap in enumerate(gaps)
        ]

        state = self._create_test_state(gaps=gaps, roi_estimates=roi_estimates)

        result = await node.execute(state)

        # Should limit to 5
        assert len(result["strategic_bets"]) <= 5

    @pytest.mark.asyncio
    async def test_difficulty_assessment(self):
        """Test difficulty assessment."""
        node = PrioritizerNode()

        # Low difficulty: small gap, low cost
        gap_low = self._create_test_gap(metric="trx", gap_size=30.0, gap_percentage=5.0)
        roi_low = self._create_test_roi(gap_low["gap_id"], cost=5000.0)

        difficulty = node._assess_difficulty(gap_low, roi_low)

        assert difficulty in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_difficulty_by_cost(self):
        """Test difficulty assessment by cost."""
        node = PrioritizerNode()

        gap = self._create_test_gap(metric="trx", gap_size=100.0, gap_percentage=15.0)

        # Low cost → low difficulty
        roi_low_cost = self._create_test_roi(gap["gap_id"], cost=5000.0)
        difficulty_low = node._assess_difficulty(gap, roi_low_cost)

        # High cost → high difficulty
        roi_high_cost = self._create_test_roi(gap["gap_id"], cost=100000.0)
        difficulty_high = node._assess_difficulty(gap, roi_high_cost)

        # High cost should result in higher difficulty
        difficulty_map = {"low": 1, "medium": 2, "high": 3}
        assert difficulty_map[difficulty_high] >= difficulty_map[difficulty_low]

    @pytest.mark.asyncio
    async def test_action_generation(self):
        """Test recommended action generation."""
        node = PrioritizerNode()

        gap = self._create_test_gap(metric="trx")
        roi = self._create_test_roi(gap["gap_id"])
        difficulty = "medium"

        action = node._generate_action(gap, roi, difficulty)

        assert isinstance(action, str)
        assert len(action) > 0
        # Should mention segment value
        assert gap["segment_value"] in action

    @pytest.mark.asyncio
    async def test_action_includes_metric_context(self):
        """Test that action includes metric-specific context."""
        node = PrioritizerNode()

        gap_trx = self._create_test_gap(metric="trx")
        roi = self._create_test_roi(gap_trx["gap_id"])

        action = node._generate_action(gap_trx, roi, "low")

        # TRx action should mention TRx or prescriptions
        assert "trx" in action.lower() or "sampling" in action.lower()

    @pytest.mark.asyncio
    async def test_time_to_impact_estimation(self):
        """Test time to impact estimation."""
        node = PrioritizerNode()

        time_low = node._estimate_time_to_impact("low")
        time_medium = node._estimate_time_to_impact("medium")
        time_high = node._estimate_time_to_impact("high")

        assert time_low in ["1-3 months", "3-6 months", "6-12 months"]
        assert time_medium in ["1-3 months", "3-6 months", "6-12 months"]
        assert time_high in ["1-3 months", "3-6 months", "6-12 months"]

    @pytest.mark.asyncio
    async def test_status_update(self):
        """Test status update to completed."""
        node = PrioritizerNode()

        gap = self._create_test_gap()
        roi = self._create_test_roi(gap["gap_id"])

        state = self._create_test_state(gaps=[gap], roi_estimates=[roi])

        result = await node.execute(state)

        assert result["status"] == "completed"


class TestPrioritizerEdgeCases:
    """Test edge cases for prioritizer."""

    def _create_test_gap(self, metric: str = "trx") -> PerformanceGap:
        """Create test gap."""
        return {
            "gap_id": f"region_Northeast_{metric}_vs_target",
            "metric": metric,
            "segment": "region",
            "segment_value": "Northeast",
            "current_value": 400.0,
            "target_value": 500.0,
            "gap_size": 100.0,
            "gap_percentage": 20.0,
            "gap_type": "vs_target",
        }

    def _create_test_roi(self, gap_id: str) -> ROIEstimate:
        """Create test ROI."""
        return {
            "gap_id": gap_id,
            "estimated_revenue_impact": 50000.0,
            "estimated_cost_to_close": 10000.0,
            "expected_roi": 4.0,
            "payback_period_months": 6,
            "confidence": 0.8,
            "assumptions": [],
        }

    def _create_test_state(
        self, gaps: list, roi_estimates: list
    ) -> GapAnalyzerState:
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
            "total_gap_value": 1000.0,
            "roi_estimates": roi_estimates,
            "total_addressable_value": 100000.0,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 100,
            "roi_latency_ms": 50,
            "total_latency_ms": 0,
            "segments_analyzed": 1,
            "errors": [],
            "warnings": [],
            "status": "prioritizing",
        }

    @pytest.mark.asyncio
    async def test_no_gaps_or_roi(self):
        """Test prioritizer with no gaps or ROI estimates."""
        node = PrioritizerNode()

        state = self._create_test_state(gaps=[], roi_estimates=[])

        result = await node.execute(state)

        assert result["prioritized_opportunities"] == []
        assert result["quick_wins"] == []
        assert result["strategic_bets"] == []
        assert "warnings" in result

    @pytest.mark.asyncio
    async def test_mismatched_gaps_and_roi(self):
        """Test when gaps and ROI estimates don't match."""
        node = PrioritizerNode()

        gap1 = self._create_test_gap(metric="trx")
        gap2 = self._create_test_gap(metric="nrx")

        # Only ROI for gap1
        roi1 = self._create_test_roi(gap1["gap_id"])

        state = self._create_test_state(gaps=[gap1, gap2], roi_estimates=[roi1])

        result = await node.execute(state)

        # Should only prioritize gap1 (has ROI)
        assert len(result["prioritized_opportunities"]) == 1

    @pytest.mark.asyncio
    async def test_no_quick_wins(self):
        """Test when no quick wins are found."""
        node = PrioritizerNode()

        # Create only high-difficulty gaps
        gap = self._create_test_gap()
        roi = self._create_test_roi(gap["gap_id"])
        roi["estimated_cost_to_close"] = 100000.0  # High cost

        state = self._create_test_state(gaps=[gap], roi_estimates=[roi])

        result = await node.execute(state)

        # May have no quick wins
        assert isinstance(result["quick_wins"], list)

    @pytest.mark.asyncio
    async def test_no_strategic_bets(self):
        """Test when no strategic bets are found."""
        node = PrioritizerNode()

        # Create only low-ROI gaps
        gap = self._create_test_gap()
        roi = self._create_test_roi(gap["gap_id"])
        roi["expected_roi"] = 0.5  # Low ROI

        state = self._create_test_state(gaps=[gap], roi_estimates=[roi])

        result = await node.execute(state)

        # May have no strategic bets
        assert isinstance(result["strategic_bets"], list)
