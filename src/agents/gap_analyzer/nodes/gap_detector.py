"""Gap Detector Node for Gap Analyzer Agent.

This node detects performance gaps across metrics and segments using parallel execution.
Supports multiple gap types: vs_target, vs_benchmark, vs_potential, temporal.

Architecture:
- Parallel segment analysis using asyncio.gather
- Mock data connectors for initial implementation
- Gap filtering by min_gap_threshold
- Aggregation and metadata collection
"""

import asyncio
import time
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from ..state import GapAnalyzerState, PerformanceGap


class GapDetectorNode:
    """Detect performance gaps across metrics and segments.

    Optimized for parallel execution across segments.
    """

    def __init__(self):
        """Initialize gap detector with mock data connectors."""
        self.data_connector = MockDataConnector()
        self.benchmark_store = MockBenchmarkStore()

    async def execute(self, state: GapAnalyzerState) -> Dict[str, Any]:
        """Execute gap detection workflow.

        Args:
            state: Current gap analyzer state with query, metrics, segments

        Returns:
            Updated state with gaps_detected, gaps_by_segment, detection_latency_ms
        """
        start_time = time.time()

        try:
            # Fetch current performance data
            current_data = await self._fetch_performance_data(
                brand=state["brand"],
                metrics=state["metrics"],
                segments=state["segments"],
                time_period=state["time_period"],
                filters=state.get("filters"),
            )

            # Get comparison data based on gap type
            comparison_data = await self._get_comparison_data(
                gap_type=state["gap_type"],
                brand=state["brand"],
                metrics=state["metrics"],
                segments=state["segments"],
                time_period=state["time_period"],
            )

            # Detect gaps in parallel across segments
            segment_tasks = []
            for segment in state["segments"]:
                segment_tasks.append(
                    self._detect_segment_gaps(
                        current_data=current_data,
                        comparison_data=comparison_data,
                        segment=segment,
                        metrics=state["metrics"],
                        gap_type=state["gap_type"],
                        min_gap_threshold=state["min_gap_threshold"],
                    )
                )

            segment_results = await asyncio.gather(*segment_tasks)

            # Flatten and aggregate gaps
            all_gaps = []
            gaps_by_segment: Dict[str, List[PerformanceGap]] = {}

            for segment, gaps in segment_results:
                gaps_by_segment[segment] = gaps
                all_gaps.extend(gaps)

            # Calculate total gap value
            total_gap_value = sum(gap["gap_size"] for gap in all_gaps)

            # Sort gaps by gap_percentage (descending)
            all_gaps.sort(key=lambda g: abs(g["gap_percentage"]), reverse=True)

            detection_latency_ms = int((time.time() - start_time) * 1000)

            return {
                "gaps_detected": all_gaps,
                "gaps_by_segment": gaps_by_segment,
                "total_gap_value": total_gap_value,
                "segments_analyzed": len(state["segments"]),
                "detection_latency_ms": detection_latency_ms,
                "status": "calculating",
            }

        except Exception as e:
            detection_latency_ms = int((time.time() - start_time) * 1000)
            return {
                "errors": [
                    {
                        "node": "gap_detector",
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                ],
                "detection_latency_ms": detection_latency_ms,
                "status": "failed",
            }

    async def _fetch_performance_data(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
        time_period: str,
        filters: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Fetch current performance data from data connectors.

        Args:
            brand: Brand identifier
            metrics: List of KPIs to analyze
            segments: Segmentation dimensions
            time_period: Analysis period
            filters: Additional filters

        Returns:
            DataFrame with current performance data
        """
        return await self.data_connector.fetch_performance_data(
            brand=brand,
            metrics=metrics,
            segments=segments,
            time_period=time_period,
            filters=filters,
        )

    async def _get_comparison_data(
        self,
        gap_type: Literal["vs_target", "vs_benchmark", "vs_potential", "temporal", "all"],
        brand: str,
        metrics: List[str],
        segments: List[str],
        time_period: str,
    ) -> Dict[str, pd.DataFrame]:
        """Get comparison data based on gap type.

        Args:
            gap_type: Type of gap analysis
            brand: Brand identifier
            metrics: List of KPIs
            segments: Segmentation dimensions
            time_period: Analysis period

        Returns:
            Dictionary mapping gap type to comparison DataFrames
        """
        comparison_data = {}

        gap_types = (
            ["vs_target", "vs_benchmark", "vs_potential", "temporal"]
            if gap_type == "all"
            else [gap_type]
        )

        for gtype in gap_types:
            if gtype == "vs_target":
                comparison_data[gtype] = await self.benchmark_store.get_targets(
                    brand=brand, metrics=metrics, segments=segments
                )
            elif gtype == "vs_benchmark":
                comparison_data[gtype] = await self.benchmark_store.get_peer_benchmarks(
                    brand=brand, metrics=metrics, segments=segments
                )
            elif gtype == "vs_potential":
                comparison_data[gtype] = await self.benchmark_store.get_top_decile(
                    brand=brand, metrics=metrics, segments=segments
                )
            elif gtype == "temporal":
                comparison_data[gtype] = await self.data_connector.fetch_prior_period(
                    brand=brand,
                    metrics=metrics,
                    segments=segments,
                    time_period=time_period,
                )

        return comparison_data

    async def _detect_segment_gaps(
        self,
        current_data: pd.DataFrame,
        comparison_data: Dict[str, pd.DataFrame],
        segment: str,
        metrics: List[str],
        gap_type: Literal["vs_target", "vs_benchmark", "vs_potential", "temporal", "all"],
        min_gap_threshold: float,
    ) -> tuple[str, List[PerformanceGap]]:
        """Detect gaps for a single segment (parallel-safe).

        Args:
            current_data: Current performance DataFrame
            comparison_data: Comparison DataFrames by gap type
            segment: Segmentation dimension
            metrics: List of KPIs
            gap_type: Type of gap analysis
            min_gap_threshold: Minimum gap % to report

        Returns:
            Tuple of (segment, list of gaps detected)
        """
        gaps: List[PerformanceGap] = []

        # Get unique segment values
        segment_values = current_data[segment].unique()

        gap_types = (
            ["vs_target", "vs_benchmark", "vs_potential", "temporal"]
            if gap_type == "all"
            else [gap_type]
        )

        for segment_value in segment_values:
            for metric in metrics:
                for gtype in gap_types:
                    gap = self._calculate_gap(
                        current_data=current_data,
                        comparison_data=comparison_data.get(gtype),
                        segment=segment,
                        segment_value=segment_value,
                        metric=metric,
                        gap_type=gtype,
                    )

                    if gap and abs(gap["gap_percentage"]) >= min_gap_threshold:
                        gaps.append(gap)

        return (segment, gaps)

    def _calculate_gap(
        self,
        current_data: pd.DataFrame,
        comparison_data: Optional[pd.DataFrame],
        segment: str,
        segment_value: str,
        metric: str,
        gap_type: str,
    ) -> Optional[PerformanceGap]:
        """Calculate a single performance gap.

        Args:
            current_data: Current performance DataFrame
            comparison_data: Comparison DataFrame
            segment: Segmentation dimension
            segment_value: Specific segment value
            metric: KPI name
            gap_type: Type of gap

        Returns:
            PerformanceGap or None if data missing
        """
        if comparison_data is None:
            return None

        # Get current value
        current_row = current_data[(current_data[segment] == segment_value)]
        if current_row.empty or metric not in current_row.columns:
            return None

        current_value = float(current_row[metric].iloc[0])

        # Get target value
        target_row = comparison_data[(comparison_data[segment] == segment_value)]
        if target_row.empty or metric not in target_row.columns:
            return None

        target_value = float(target_row[metric].iloc[0])

        # Calculate gap
        gap_size = target_value - current_value
        gap_percentage = (gap_size / target_value * 100) if target_value != 0 else 0.0

        gap_id = f"{segment}_{segment_value}_{metric}_{gap_type}"

        gap: PerformanceGap = {
            "gap_id": gap_id,
            "metric": metric,
            "segment": segment,
            "segment_value": segment_value,
            "current_value": current_value,
            "target_value": target_value,
            "gap_size": gap_size,
            "gap_percentage": gap_percentage,
            "gap_type": gap_type,  # type: ignore
        }

        return gap


class MockDataConnector:
    """Mock data connector for initial implementation.

    Replace with actual SupabaseDataConnector in integration phase.
    """

    async def fetch_performance_data(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
        time_period: str,
        filters: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Fetch mock current performance data.

        Returns realistic pharmaceutical commercial data.
        """
        await asyncio.sleep(0.05)  # Simulate I/O

        # Generate mock data
        np.random.seed(42)
        data = []

        segment_values = {
            "region": ["Northeast", "Southeast", "Midwest", "West"],
            "specialty": ["Oncology", "Cardiology", "Rheumatology", "Neurology"],
            "hcp_tier": ["Tier 1", "Tier 2", "Tier 3"],
        }

        for segment in segments:
            if segment not in segment_values:
                continue

            for value in segment_values[segment]:
                row = {segment: value}
                for metric in metrics:
                    # Realistic values based on metric type
                    if metric == "trx":
                        row[metric] = np.random.randint(100, 500)
                    elif metric == "nrx":
                        row[metric] = np.random.randint(20, 100)
                    elif metric == "market_share":
                        row[metric] = np.random.uniform(5.0, 25.0)
                    elif metric == "conversion_rate":
                        row[metric] = np.random.uniform(0.1, 0.5)
                    elif metric == "hcp_engagement_score":
                        row[metric] = np.random.uniform(50.0, 90.0)
                    else:
                        row[metric] = np.random.uniform(10.0, 100.0)
                data.append(row)

        return pd.DataFrame(data)

    async def fetch_prior_period(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
        time_period: str,
    ) -> pd.DataFrame:
        """Fetch mock prior period data."""
        await asyncio.sleep(0.05)

        # Similar to current, but with slight variations
        current = await self.fetch_performance_data(
            brand=brand,
            metrics=metrics,
            segments=segments,
            time_period=time_period,
            filters=None,
        )

        # Adjust values to simulate temporal change
        prior = current.copy()
        for metric in metrics:
            if metric in prior.columns:
                prior[metric] = prior[metric] * np.random.uniform(0.9, 1.1)

        return prior


class MockBenchmarkStore:
    """Mock benchmark store for initial implementation.

    Replace with actual benchmark data fetching in integration phase.
    """

    async def get_targets(
        self, brand: str, metrics: List[str], segments: List[str]
    ) -> pd.DataFrame:
        """Get mock predefined targets."""
        await asyncio.sleep(0.03)

        np.random.seed(43)
        data = []

        segment_values = {
            "region": ["Northeast", "Southeast", "Midwest", "West"],
            "specialty": ["Oncology", "Cardiology", "Rheumatology", "Neurology"],
            "hcp_tier": ["Tier 1", "Tier 2", "Tier 3"],
        }

        for segment in segments:
            if segment not in segment_values:
                continue

            for value in segment_values[segment]:
                row = {segment: value}
                for metric in metrics:
                    # Targets are typically 10-30% higher than current
                    if metric == "trx":
                        row[metric] = np.random.randint(150, 600)
                    elif metric == "nrx":
                        row[metric] = np.random.randint(30, 120)
                    elif metric == "market_share":
                        row[metric] = np.random.uniform(8.0, 30.0)
                    elif metric == "conversion_rate":
                        row[metric] = np.random.uniform(0.15, 0.6)
                    elif metric == "hcp_engagement_score":
                        row[metric] = np.random.uniform(65.0, 95.0)
                    else:
                        row[metric] = np.random.uniform(15.0, 120.0)
                data.append(row)

        return pd.DataFrame(data)

    async def get_peer_benchmarks(
        self, brand: str, metrics: List[str], segments: List[str]
    ) -> pd.DataFrame:
        """Get mock peer benchmark data."""
        await asyncio.sleep(0.03)

        # Peer benchmarks are similar to targets but with different distribution
        targets = await self.get_targets(brand, metrics, segments)
        benchmarks = targets.copy()

        for metric in metrics:
            if metric in benchmarks.columns:
                benchmarks[metric] = benchmarks[metric] * np.random.uniform(0.95, 1.05)

        return benchmarks

    async def get_top_decile(
        self, brand: str, metrics: List[str], segments: List[str]
    ) -> pd.DataFrame:
        """Get mock top decile performance data."""
        await asyncio.sleep(0.03)

        # Top decile represents best-in-class performance
        targets = await self.get_targets(brand, metrics, segments)
        top_decile = targets.copy()

        for metric in metrics:
            if metric in top_decile.columns:
                # Top decile is 20-40% higher than targets
                top_decile[metric] = top_decile[metric] * np.random.uniform(1.2, 1.4)

        return top_decile
