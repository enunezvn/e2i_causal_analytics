"""Tests for Gap Detector Node."""

import pandas as pd
import pytest

from src.agents.gap_analyzer.nodes.gap_detector import GapDetectorNode
from src.agents.gap_analyzer.state import GapAnalyzerState


class TestGapDetectorNode:
    """Test GapDetectorNode."""

    def _create_test_state(
        self, gap_type: str = "vs_potential", min_gap_threshold: float = 5.0
    ) -> GapAnalyzerState:
        """Create test state."""
        return {
            "query": "identify trx gaps",
            "metrics": ["trx", "nrx"],
            "segments": ["region", "specialty"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": gap_type,  # type: ignore
            "min_gap_threshold": min_gap_threshold,
            "max_opportunities": 10,
            "gaps_detected": None,
            "gaps_by_segment": None,
            "total_gap_value": None,
            "roi_estimates": None,
            "total_addressable_value": None,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 0,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 0,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

    @pytest.mark.asyncio
    async def test_detect_gaps_vs_potential(self):
        """Test gap detection with vs_potential comparison."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state(gap_type="vs_potential")

        result = await node.execute(state)

        assert "gaps_detected" in result
        assert isinstance(result["gaps_detected"], list)
        assert len(result["gaps_detected"]) > 0

        # Verify gap structure
        gap = result["gaps_detected"][0]
        assert "gap_id" in gap
        assert "metric" in gap
        assert "segment" in gap
        assert "segment_value" in gap
        assert "current_value" in gap
        assert "target_value" in gap
        assert "gap_size" in gap
        assert "gap_percentage" in gap
        assert "gap_type" in gap
        assert gap["gap_type"] == "vs_potential"

    @pytest.mark.asyncio
    async def test_detect_gaps_vs_target(self):
        """Test gap detection with vs_target comparison."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state(gap_type="vs_target")

        result = await node.execute(state)

        assert result["gaps_detected"] is not None
        assert any(g["gap_type"] == "vs_target" for g in result["gaps_detected"])

    @pytest.mark.asyncio
    async def test_detect_gaps_vs_benchmark(self):
        """Test gap detection with vs_benchmark comparison."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state(gap_type="vs_benchmark")

        result = await node.execute(state)

        assert result["gaps_detected"] is not None
        assert any(g["gap_type"] == "vs_benchmark" for g in result["gaps_detected"])

    @pytest.mark.asyncio
    async def test_detect_gaps_temporal(self):
        """Test gap detection with temporal comparison."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state(gap_type="temporal")

        result = await node.execute(state)

        assert result["gaps_detected"] is not None
        assert any(g["gap_type"] == "temporal" for g in result["gaps_detected"])

    @pytest.mark.asyncio
    async def test_detect_gaps_all_types(self):
        """Test gap detection with all comparison types."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state(gap_type="all")

        result = await node.execute(state)

        gap_types = {g["gap_type"] for g in result["gaps_detected"]}
        # Should have multiple gap types
        assert len(gap_types) > 1

    @pytest.mark.asyncio
    async def test_gap_filtering_by_threshold(self):
        """Test that gaps below threshold are filtered."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state(min_gap_threshold=50.0)  # Very high threshold

        result = await node.execute(state)

        # High threshold should reduce gaps
        assert len(result["gaps_detected"]) < 10

    @pytest.mark.asyncio
    async def test_gaps_by_segment_structure(self):
        """Test gaps_by_segment grouping."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()

        result = await node.execute(state)

        assert "gaps_by_segment" in result
        assert isinstance(result["gaps_by_segment"], dict)

        # Should have entries for each segment
        for segment in state["segments"]:
            assert segment in result["gaps_by_segment"]
            assert isinstance(result["gaps_by_segment"][segment], list)

    @pytest.mark.asyncio
    async def test_total_gap_value_calculation(self):
        """Test total gap value aggregation."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()

        result = await node.execute(state)

        assert "total_gap_value" in result
        assert result["total_gap_value"] >= 0

        # Total should equal sum of individual gaps
        manual_total = sum(g["gap_size"] for g in result["gaps_detected"])
        assert abs(result["total_gap_value"] - manual_total) < 0.01

    @pytest.mark.asyncio
    async def test_segments_analyzed_count(self):
        """Test segments_analyzed count."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()

        result = await node.execute(state)

        assert "segments_analyzed" in result
        assert result["segments_analyzed"] == len(state["segments"])

    @pytest.mark.asyncio
    async def test_detection_latency_measurement(self):
        """Test that detection latency is measured."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()

        result = await node.execute(state)

        assert "detection_latency_ms" in result
        assert result["detection_latency_ms"] > 0
        assert result["detection_latency_ms"] < 10000  # Should be < 10s

    @pytest.mark.asyncio
    async def test_status_update(self):
        """Test that status is updated to calculating."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()

        result = await node.execute(state)

        assert result["status"] == "calculating"

    @pytest.mark.asyncio
    async def test_multiple_metrics(self):
        """Test gap detection across multiple metrics."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()
        state["metrics"] = ["trx", "nrx", "market_share"]

        result = await node.execute(state)

        # Should have gaps for all metrics
        detected_metrics = {g["metric"] for g in result["gaps_detected"]}
        assert len(detected_metrics) > 1

    @pytest.mark.asyncio
    async def test_multiple_segments(self):
        """Test gap detection across multiple segments."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()
        state["segments"] = ["region", "specialty", "hcp_tier"]

        result = await node.execute(state)

        # Should have gaps for all segments
        detected_segments = {g["segment"] for g in result["gaps_detected"]}
        assert len(detected_segments) > 1

    @pytest.mark.asyncio
    async def test_gap_id_uniqueness(self):
        """Test that gap IDs are unique."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()

        result = await node.execute(state)

        gap_ids = [g["gap_id"] for g in result["gaps_detected"]]
        assert len(gap_ids) == len(set(gap_ids))  # All unique

    @pytest.mark.asyncio
    async def test_gap_id_format(self):
        """Test gap ID format."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()

        result = await node.execute(state)

        gap = result["gaps_detected"][0]
        gap_id = gap["gap_id"]

        # Format: {segment}_{segment_value}_{metric}_{gap_type}
        assert gap["segment"] in gap_id
        assert gap["segment_value"] in gap_id
        assert gap["metric"] in gap_id
        assert gap["gap_type"] in gap_id

    @pytest.mark.asyncio
    async def test_gap_percentage_calculation(self):
        """Test gap percentage calculation."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()

        result = await node.execute(state)

        for gap in result["gaps_detected"]:
            current = gap["current_value"]
            target = gap["target_value"]
            gap_size = gap["gap_size"]
            gap_pct = gap["gap_percentage"]

            # Verify formula: gap_pct = (target - current) / target * 100
            expected_gap_size = target - current
            expected_gap_pct = (expected_gap_size / target * 100) if target != 0 else 0.0

            assert abs(gap_size - expected_gap_size) < 0.01
            assert abs(gap_pct - expected_gap_pct) < 0.01

    @pytest.mark.asyncio
    async def test_gaps_sorted_by_percentage(self):
        """Test that gaps are sorted by gap_percentage descending."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()

        result = await node.execute(state)

        gaps = result["gaps_detected"]
        if len(gaps) > 1:
            # Check descending order by absolute gap_percentage
            for i in range(len(gaps) - 1):
                assert abs(gaps[i]["gap_percentage"]) >= abs(gaps[i + 1]["gap_percentage"])


class TestGapDetectorEdgeCases:
    """Test edge cases for gap detector."""

    def _create_test_state(
        self, gap_type: str = "vs_potential", min_gap_threshold: float = 5.0
    ) -> GapAnalyzerState:
        """Create test state."""
        return {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": gap_type,  # type: ignore
            "min_gap_threshold": min_gap_threshold,
            "max_opportunities": 10,
            "gaps_detected": None,
            "gaps_by_segment": None,
            "total_gap_value": None,
            "roi_estimates": None,
            "total_addressable_value": None,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 0,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 0,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

    @pytest.mark.asyncio
    async def test_very_high_threshold(self):
        """Test with threshold that filters out all gaps."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state(min_gap_threshold=1000.0)

        result = await node.execute(state)

        # May have no gaps above threshold
        assert isinstance(result["gaps_detected"], list)

    @pytest.mark.asyncio
    async def test_zero_threshold(self):
        """Test with zero threshold (all gaps included)."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state(min_gap_threshold=0.0)

        result = await node.execute(state)

        # Should have maximum gaps
        assert len(result["gaps_detected"]) > 0

    @pytest.mark.asyncio
    async def test_single_metric_single_segment(self):
        """Test with single metric and single segment."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()
        state["metrics"] = ["trx"]
        state["segments"] = ["region"]

        result = await node.execute(state)

        assert result["gaps_detected"] is not None
        assert result["segments_analyzed"] == 1

    @pytest.mark.asyncio
    async def test_with_filters(self):
        """Test gap detection with additional filters."""
        node = GapDetectorNode(use_mock=True)
        state = self._create_test_state()
        state["filters"] = {"specialty": "Oncology"}

        result = await node.execute(state)

        # Should still work with filters
        assert result["gaps_detected"] is not None


class TestMockDataConnectors:
    """Test mock data connector behavior."""

    @pytest.mark.asyncio
    async def test_mock_connector_returns_dataframe(self):
        """Test that mock connector returns valid DataFrame."""
        from src.agents.gap_analyzer.nodes.gap_detector import MockDataConnector

        connector = MockDataConnector()

        df = await connector.fetch_performance_data(
            brand="kisqali",
            metrics=["trx", "nrx"],
            segments=["region"],
            time_period="current_quarter",
            filters=None,
        )

        assert df is not None
        assert not df.empty
        assert "region" in df.columns
        assert "trx" in df.columns

    @pytest.mark.asyncio
    async def test_mock_benchmark_returns_dataframe(self):
        """Test that mock benchmark store returns valid DataFrame."""
        from src.agents.gap_analyzer.nodes.gap_detector import MockBenchmarkStore

        store = MockBenchmarkStore()

        df = await store.get_targets(brand="kisqali", metrics=["trx"], segments=["region"])

        assert df is not None
        assert not df.empty
        assert "region" in df.columns
        assert "trx" in df.columns


class TestGapDetectorExcludesOverPerformance:
    """Regression: only a genuine SHORTFALL (current BELOW the reference) is a
    capturable opportunity.

    Previously the segment-gap filter used ``abs(gap_percentage)``, so a segment
    that EXCEEDED its target (negative gap = over-performance) leaked through and
    — because the ROI calculator takes ``abs(gap_size)`` — was ranked as a top
    "opportunity" with an absurd ROI (live: a region 54% OVER its TRx target
    surfaced as a $63M / 848x-ROI opportunity, inflating Total Addressable Value
    to ~$850M). These tests pin the corrected contract: over-performance is
    excluded, real shortfalls are kept.
    """

    @pytest.mark.asyncio
    async def test_over_performance_excluded_under_performance_kept(self):
        node = GapDetectorNode(use_mock=True)
        # Region A is 100% OVER target (current 200 vs target 100) -> over-perf.
        # Region B is a 20% shortfall (current 80 vs target 100) -> opportunity.
        current = pd.DataFrame({"region": ["A", "B"], "trx": [200.0, 80.0]})
        targets = pd.DataFrame({"region": ["A", "B"], "trx": [100.0, 100.0]})

        _, gaps = await node._detect_segment_gaps(
            current_data=current,
            comparison_data={"vs_target": targets},
            segment="region",
            metrics=["trx"],
            gap_type="vs_target",
            min_gap_threshold=5.0,
        )

        seg_vals = {g["segment_value"] for g in gaps}
        assert "B" in seg_vals, "genuine shortfall must be detected"
        assert "A" not in seg_vals, "over-performance must NOT be a gap"
        # Every retained gap is a positive shortfall above threshold.
        assert gaps and all(g["gap_percentage"] >= 5.0 for g in gaps)
        assert all(g["gap_percentage"] > 0 for g in gaps)

    @pytest.mark.asyncio
    async def test_all_over_performance_yields_no_gaps(self):
        node = GapDetectorNode(use_mock=True)
        # Both regions exceed target -> nothing to "close".
        current = pd.DataFrame({"region": ["A", "B"], "trx": [300.0, 250.0]})
        targets = pd.DataFrame({"region": ["A", "B"], "trx": [100.0, 100.0]})

        _, gaps = await node._detect_segment_gaps(
            current_data=current,
            comparison_data={"vs_target": targets},
            segment="region",
            metrics=["trx"],
            gap_type="vs_target",
            min_gap_threshold=5.0,
        )

        assert gaps == []

    @pytest.mark.asyncio
    async def test_temporal_decline_kept_growth_excluded(self):
        # vs_target semantics generalize: gap_size = reference - current, so a
        # temporal DECLINE (prior > current) is a positive gap to recover, while
        # GROWTH (prior < current) is over-performance and is excluded.
        node = GapDetectorNode(use_mock=True)
        current = pd.DataFrame({"region": ["Declined", "Grew"], "trx": [70.0, 130.0]})
        prior = pd.DataFrame({"region": ["Declined", "Grew"], "trx": [100.0, 100.0]})

        _, gaps = await node._detect_segment_gaps(
            current_data=current,
            comparison_data={"temporal": prior},
            segment="region",
            metrics=["trx"],
            gap_type="temporal",
            min_gap_threshold=5.0,
        )

        seg_vals = {g["segment_value"] for g in gaps}
        assert "Declined" in seg_vals
        assert "Grew" not in seg_vals


class TestMarketShareSegmentTrxContext:
    """market_share gaps must carry same-segment TRx volume as ``segment_trx``.

    The ROI calculator converts share-point gaps to TRx-equivalents via
    relative share growth x segment TRx (market size constant). The detector
    is the only place that still holds the pivoted frame where the segment's
    trx and market_share sit on the same row, so it attaches the context.
    """

    def _frames(self, with_trx=True):
        cols = {"region": ["northeast"], "market_share": [0.49]}
        target_cols = {"region": ["northeast"], "market_share": [0.5367]}
        if with_trx:
            cols["trx"] = [217347.31]
            target_cols["trx"] = [233805.51]
        return pd.DataFrame(cols), pd.DataFrame(target_cols)

    def test_share_gap_carries_segment_trx(self):
        node = GapDetectorNode(use_mock=True)
        current, target = self._frames()
        gap = node._calculate_gap(
            current, target, "region", "northeast", "market_share", "temporal"
        )
        assert gap is not None
        assert gap["segment_trx"] == pytest.approx(217347.31)

    def test_trx_gap_does_not_carry_context_field(self):
        node = GapDetectorNode(use_mock=True)
        current, target = self._frames()
        gap = node._calculate_gap(current, target, "region", "northeast", "trx", "temporal")
        assert gap is not None
        assert "segment_trx" not in gap

    def test_share_gap_without_trx_column_omits_context(self):
        node = GapDetectorNode(use_mock=True)
        current, target = self._frames(with_trx=False)
        gap = node._calculate_gap(
            current, target, "region", "northeast", "market_share", "temporal"
        )
        assert gap is not None
        assert "segment_trx" not in gap
