"""
Benchmark Store for Gap Analyzer.

Production benchmark store that retrieves targets, peer benchmarks,
and top decile performance from the business_metrics table.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from src.memory.services.factories import ServiceConnectionError

logger = logging.getLogger(__name__)


class BenchmarkStore:
    """
    Production benchmark store using business_metrics table.

    Replaces MockBenchmarkStore for production use.
    Provides targets, peer benchmarks, and top decile metrics.
    """

    def __init__(self, supabase_client=None, include_synthetic: bool = False):
        """
        Initialize store with optional Supabase client.

        Args:
            supabase_client: Optional Supabase client. When absent it is resolved
                lazily and fail-closed via ``get_async_supabase_client()`` (#845
                family — a client-less repo silently no-ops; we resolve a real one).
            include_synthetic: When True, repository reads opt in to synthetic rows
                (the validation layer; #851). Default False keeps real-mode isolation.
        """
        self._repository = None
        self._client = supabase_client
        self._client_resolved = supabase_client is not None
        self.include_synthetic = include_synthetic

    async def _ensure_repository(self):
        """Resolve a real async Supabase client (fail-closed) and build the repo.

        Mirrors ``SupabaseDataConnector._ensure_repository`` — the factory builds this
        store without a client, and a client-less repository silently returns no
        benchmarks. ``get_async_supabase_client()`` RAISES when Supabase is
        unconfigured (fail-closed), surfaced as an error rather than fabricated zeros.
        """
        from src.repositories.business_metric import BusinessMetricRepository

        if self._repository is not None:
            return self._repository

        if not self._client_resolved:
            from src.memory.services.factories import get_async_supabase_client

            self._client = await get_async_supabase_client()
            self._client_resolved = True

        self._repository = BusinessMetricRepository(self._client)
        return self._repository

    @property
    def repository(self):
        """Return a BusinessMetricRepository bound to an ALREADY-RESOLVED client.

        FAIL-CLOSED: never construct ``Repository(None)`` here — that would cache a
        silent-no-op repo and re-open the #845 fail-OPEN hole. Async callers must use
        ``await self._ensure_repository()`` (resolves the client fail-closed).
        """
        if self._repository is not None:
            return self._repository
        if self._client is None:
            raise ServiceConnectionError(
                "Supabase",
                "BenchmarkStore.repository accessed before a client was resolved; use "
                "the async path so the client is resolved fail-closed",
            )
        from src.repositories.business_metric import BusinessMetricRepository

        self._repository = BusinessMetricRepository(self._client)
        return self._repository

    async def _resolve_segment(self, brand: str, segments: List[str]) -> tuple:
        """Resolve which requested segment column actually exists in the data.

        ``business_metrics`` carries ``region`` as its queryable segment dimension.
        For each requested segment we probe its distinct values; the first segment
        with real values wins. Segments the table does not carry (``specialty``/
        ``hcp_tier``) yield no values → returns ``(None, [])`` so the caller emits an
        empty frame and ``_calculate_gap`` produces NO gaps for them (no fabrication),
        rather than crashing or inventing data (#851 MED follow-up).

        Returns:
            (segment_name, distinct_values) or (None, []) if none are supported.
        """
        repository = await self._ensure_repository()
        for segment in segments:
            values = await repository.get_distinct_values(
                segment, brand=brand, include_synthetic=self.include_synthetic
            )
            if values:
                return segment, values
        return None, []

    async def _fetch_segment_metric_frame(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
        value_field: str = "value",
    ) -> pd.DataFrame:
        """Fetch a per-segment wide frame of mean metric values from business_metrics.

        Returns a DataFrame with the resolved segment column (e.g. ``region``) plus one
        column per requested metric (the mean of ``value_field`` for that metric in
        that segment value). This is the SAME per-segment wide shape
        ``GapDetectorNode._calculate_gap`` consumes for ``current_data`` and that
        ``MockBenchmarkStore`` returns — the production store previously returned a flat
        row / metric-stat long frame that ``_calculate_gap`` could not align (the
        gap_analyzer production path was therefore non-functional; #851). Segment values
        are discovered from the data (lowercase enum), not hardcoded (#851 block 3).

        Args:
            value_field: "value" for performance/peer frames, "target" for target frames.
        """
        repository = await self._ensure_repository()

        segment, seg_values = await self._resolve_segment(brand, segments)
        if not segment:
            logger.warning(
                f"No supported segment among {segments} for brand={brand} "
                f"(business_metrics carries 'region')"
            )
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for seg_value in seg_values:
            records = await repository.get_by_region(
                region=seg_value,
                brand=brand,
                limit=5000,
                include_synthetic=self.include_synthetic,
            )
            # Collect per-metric values for this segment value, then average.
            metric_values: Dict[str, List[float]] = {m: [] for m in metrics}
            for record in records:
                metric_name = (
                    record.get("metric_name")
                    if isinstance(record, dict)
                    else getattr(record, "metric_name", None)
                )
                if metric_name in metric_values:
                    val = (
                        record.get(value_field)
                        if isinstance(record, dict)
                        else getattr(record, value_field, None)
                    )
                    if val is not None:
                        metric_values[metric_name].append(float(val))

            row: Dict[str, Any] = {segment: seg_value}
            has_any = False
            for metric in metrics:
                vals = metric_values[metric]
                if vals:
                    row[metric] = sum(vals) / len(vals)
                    has_any = True
            if has_any:
                rows.append(row)

        return pd.DataFrame(rows)

    async def get_targets(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
    ) -> pd.DataFrame:
        """
        Get per-segment target values from the business_metrics.target column.

        Args:
            brand: Brand name
            metrics: List of KPI names
            segments: List of segment dimensions

        Returns:
            Per-segment wide DataFrame (segment column + one column per metric),
            aligned with the connector's ``fetch_performance_data`` output so
            ``_calculate_gap`` can compare current vs target per segment value.
        """
        # Targets come from the per-row `target` column (value_field='target').
        # Operational failures propagate (NOT swallowed into an empty frame) — the
        # gap_detector node records them as errors and the run fails closed. An empty
        # frame here means genuinely no data (unsupported segment / no rows), which
        # _fetch_segment_metric_frame returns explicitly.
        return await self._fetch_segment_metric_frame(
            brand, metrics, segments, value_field="target"
        )

    async def get_peer_benchmarks(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
    ) -> pd.DataFrame:
        """
        Get peer benchmark data as a per-region comparison frame.

        Each region is compared against the cross-region peer benchmark (P75 of the
        per-region means) for each metric — a "best peers" bar broadcast to every
        region. Returned in the per-region wide shape ``_calculate_gap`` consumes.

        Args:
            brand: Brand name
            metrics: List of KPI names
            segments: List of segment dimensions

        Returns:
            Per-segment wide DataFrame (segment column + one column per metric).
        """
        # Operational failures propagate (see get_targets). Empty == no real data.
        seg_frame = await self._fetch_segment_metric_frame(brand, metrics, segments)
        if seg_frame.empty:
            logger.warning(f"No peer benchmark data found for brand={brand}")
            return pd.DataFrame()

        # Peer benchmark = P75 across the per-segment-value means (top quartile).
        return self._broadcast_cross_segment_stat(seg_frame, metrics, quantile=0.75)

    async def get_top_decile(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
    ) -> pd.DataFrame:
        """
        Calculate top decile (P90) performance across segment values as a comparison.

        Top decile represents best-in-class performance, broadcast to every segment
        value in the per-segment wide shape ``_calculate_gap`` consumes.

        Args:
            brand: Brand name
            metrics: List of KPI names
            segments: List of segment dimensions

        Returns:
            Per-segment wide DataFrame (segment column + one column per metric).
        """
        # Operational failures propagate (see get_targets). Empty == no real data.
        seg_frame = await self._fetch_segment_metric_frame(brand, metrics, segments)
        if seg_frame.empty:
            return pd.DataFrame()

        # Top decile = P90 across the per-segment-value means (best-in-class).
        return self._broadcast_cross_segment_stat(seg_frame, metrics, quantile=0.90)

    @staticmethod
    def _broadcast_cross_segment_stat(
        seg_frame: pd.DataFrame,
        metrics: List[str],
        quantile: float,
    ) -> pd.DataFrame:
        """Broadcast a cross-segment quantile per metric back onto every segment value.

        Takes a per-segment wide frame (segment column + metric columns), computes the
        cross-segment quantile for each metric, and returns a frame with the SAME
        segment rows where every segment value carries that single peer/best-in-class
        bar. This keeps the per-segment shape ``_calculate_gap`` requires while encoding
        a cross-segment comparison standard. The segment column is whatever the frame
        carries (e.g. ``region``) — discovered, not hardcoded.
        """
        if seg_frame.empty:
            return pd.DataFrame()
        # The segment column is the single non-metric column in the frame.
        seg_cols = [c for c in seg_frame.columns if c not in metrics]
        result = seg_frame[seg_cols].copy()
        for metric in metrics:
            if metric in seg_frame.columns:
                bar = float(seg_frame[metric].quantile(quantile))
                result[metric] = bar
        return result

    async def get_benchmark_summary(
        self,
        brand: str,
    ) -> Dict[str, Any]:
        """
        Get a summary of benchmark data availability.

        Args:
            brand: Brand name

        Returns:
            Dict with benchmark summary info
        """
        try:
            repository = await self._ensure_repository()
            snapshot = await repository.get_latest_snapshot(
                brand, include_synthetic=self.include_synthetic
            )
            achievement = await repository.get_achievement_summary(
                brand, include_synthetic=self.include_synthetic
            )
            roi = await repository.get_roi_summary(brand, include_synthetic=self.include_synthetic)

            return {
                "brand": brand,
                "total_metrics": len(snapshot),
                "metrics_with_targets": sum(
                    1 for m in snapshot.values() if m.get("target") is not None
                ),
                "avg_achievement": achievement.get("avg_achievement", 0),
                "metrics_at_target": achievement.get("metrics_at_target", 0),
                "metrics_below_target": achievement.get("metrics_below_target", 0),
                "avg_roi": roi.get("avg_roi", 0),
            }

        except ServiceConnectionError:
            raise  # FAIL-CLOSED — see get_targets.
        except Exception as e:
            logger.error(f"Failed to get benchmark summary: {e}")
            return {"brand": brand, "error": str(e)}
