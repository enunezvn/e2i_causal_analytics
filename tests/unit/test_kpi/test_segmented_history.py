"""Tests for src.kpi.segmented_history (axis-segmented monthly KPI series).

Row fixtures mirror the REAL kpi_query output shape verified against the live
DB (migration 110 dry-run 2026-07-18): rows of month_start/bucket/value plus
the global prescription date range (data_min/data_max) on every row.
"""

from unittest.mock import patch

from src.kpi.segmented_history import (
    AXIS_SUFFIXES,
    SEGMENTED_KPI_QUERY_FAMILIES,
    bucket_label,
    canonical_buckets,
    shape_segmented_series,
)
from src.kpi.synthetic_mode import monthly_axis_query_id


def _row(month, bucket, value, data_min="2026-01-01", data_max="2026-06-30"):
    return {
        "month_start": month,
        "bucket": bucket,
        "value": value,
        "data_min": data_min,
        "data_max": data_max,
    }


class TestShapeSegmentedSeries:
    def test_buckets_in_canonical_order_with_zero_fill(self):
        # medium has no 2026-02 row -> genuine zero; high absent entirely ->
        # zero-filled flat series (still emitted).
        rows = [
            _row("2026-01-01", "medium_severity", 7),
            _row("2026-01-01", "low_severity", 3),
            _row("2026-02-01", "low_severity", 4),
        ]
        series, data_through = shape_segmented_series(rows, axis="segment")
        assert data_through == "2026-06-30"
        assert [s["key"] for s in series] == [
            "low_severity",
            "medium_severity",
            "high_severity",
        ]
        low, medium, high = series
        # data_min is the 1st and data_max a month-end -> no edge trimming;
        # span = Jan..Jun 2026 inclusive.
        assert [p["metric_date"] for p in low["points"]][:2] == ["2026-01-01", "2026-02-01"]
        assert low["count"] == 6
        assert [p["value"] for p in low["points"][:2]] == [3.0, 4.0]
        assert [p["value"] for p in medium["points"][:2]] == [7.0, 0.0]
        assert all(p["value"] == 0.0 for p in high["points"])

    def test_partial_edge_months_trimmed(self):
        # Mirrors the live substrate: data starts 2023-07-22 (partial July)
        # and the frontier is mid-month -> both edge months dropped, exactly
        # like history_backfill._complete_months does for the headline series.
        rows = [
            _row("2026-01-01", "low_severity", 1, data_min="2026-01-15", data_max="2026-03-10"),
            _row("2026-02-01", "low_severity", 5, data_min="2026-01-15", data_max="2026-03-10"),
            _row("2026-03-01", "low_severity", 9, data_min="2026-01-15", data_max="2026-03-10"),
        ]
        series, _ = shape_segmented_series(rows, axis="segment")
        low = series[0]
        assert [p["metric_date"] for p in low["points"]] == ["2026-02-01"]
        assert [p["value"] for p in low["points"]] == [5.0]

    def test_value_filter_restricts_to_one_bucket(self):
        rows = [
            _row("2026-01-01", "low_severity", 3),
            _row("2026-01-01", "high_severity", 8),
        ]
        series, _ = shape_segmented_series(rows, axis="segment", value="high_severity")
        assert [s["key"] for s in series] == ["high_severity"]
        assert series[0]["points"][0]["value"] == 8.0

    def test_start_end_dates_filter_months(self):
        rows = [
            _row("2026-01-01", "0", 1),
            _row("2026-02-01", "0", 2),
            _row("2026-03-01", "0", 3),
        ]
        series, _ = shape_segmented_series(
            rows, axis="therapy_line", start_date="2026-02-01", end_date="2026-02-28"
        )
        line0 = series[0]
        assert [p["metric_date"] for p in line0["points"]] == ["2026-02-01"]

    def test_therapy_line_buckets_numeric_order(self):
        rows = [
            _row("2026-01-01", "3", 1),
            _row("2026-01-01", "0", 2),
            _row("2026-01-01", "2", 3),
        ]
        series, _ = shape_segmented_series(rows, axis="therapy_line")
        assert [s["key"] for s in series] == ["0", "1", "2", "3"]

    def test_unexpected_bucket_appended_not_dropped(self):
        rows = [
            _row("2026-01-01", "low_severity", 3),
            _row("2026-01-01", "mystery_tier", 2),
        ]
        series, _ = shape_segmented_series(rows, axis="segment")
        assert [s["key"] for s in series][-1] == "mystery_tier"

    def test_empty_rows_return_empty_series(self):
        series, data_through = shape_segmented_series([], axis="segment")
        assert series == []
        assert data_through is None

    def test_malformed_rows_skipped(self):
        rows = [
            {"month_start": None, "bucket": "low_severity", "value": 1},
            {"month_start": "2026-01-01", "bucket": None, "value": 1},
            _row("2026-01-01", "low_severity", 3),
        ]
        series, _ = shape_segmented_series(rows, axis="segment", value="low_severity")
        assert series[0]["points"][0]["value"] == 3.0


class TestLabelsAndIds:
    def test_bucket_labels(self):
        assert bucket_label("segment", "high_severity") == "High severity"
        assert bucket_label("therapy_line", "0") == "0 prior lines"
        assert bucket_label("therapy_line", "1") == "1 prior line"
        assert bucket_label("therapy_line", "2") == "2 prior lines"

    def test_canonical_buckets(self):
        assert canonical_buckets("segment") == [
            "low_severity",
            "medium_severity",
            "high_severity",
        ]
        assert canonical_buckets("therapy_line") == ["0", "1", "2", "3"]

    def test_supported_kpis_match_migration_105_family(self):
        assert set(SEGMENTED_KPI_QUERY_FAMILIES) == {"WS3-BI-005", "WS3-BI-006", "WS3-BI-007"}
        assert set(AXIS_SUFFIXES) == {"segment", "therapy_line"}

    def test_monthly_axis_query_id_plain_and_synthetic(self):
        with patch("src.kpi.synthetic_mode.kpi_include_synthetic", return_value=False):
            assert (
                monthly_axis_query_id("business_impact_trx", axis="segment")
                == "business_impact_trx_monthly_by_segment"
            )
        with patch("src.kpi.synthetic_mode.kpi_include_synthetic", return_value=True):
            assert (
                monthly_axis_query_id("business_impact_nbrx", axis="line")
                == "business_impact_nbrx_monthly_by_line_include_synthetic"
            )
