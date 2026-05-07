"""Unit tests for the mandatory temporal aggregation API."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.temporal_aggregation import temporal_aggregation


@pytest.fixture
def basic_events_and_anchors():
    """Build a small event/anchor pair where windowing matters.

    Patient 1 has events at 2024-01-01 (220d before anchor), 2024-06-01
    (60d before anchor), 2024-08-15 (15d before anchor), 2024-09-15 (15d
    AFTER anchor — must be excluded).
    Patient 2 has only one event, before the window.
    """
    events = pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 1, 2],
            "event_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-06-01",
                    "2024-08-15",
                    "2024-09-15",
                    "2024-01-01",
                ]
            ),
            "value": [10, 20, 30, 999, 5],
        }
    )
    anchors = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "anchor_date": pd.to_datetime(["2024-08-30", "2024-08-30", "2024-08-30"]),
        }
    )
    return events, anchors


def test_excludes_post_anchor_events(basic_events_and_anchors):
    events, anchors = basic_events_and_anchors
    result = temporal_aggregation(
        events,
        anchors,
        anchor_col="anchor_date",
        event_date_col="event_date",
        group_col="patient_id",
        window_days=180,
        agg={"value": "sum"},
    )
    p1_sum = result.loc[result.patient_id == 1, "value_sum"].iloc[0]
    # Post-anchor event (999) must be excluded. In-window: 20 + 30 = 50.
    # 2024-01-01 is 242 days before 2024-08-30 — outside the 180d window.
    assert p1_sum == 50, f"Expected sum=50 (in-window only); got {p1_sum}"


def test_missing_group_filled_with_neutral(basic_events_and_anchors):
    events, anchors = basic_events_and_anchors
    result = temporal_aggregation(
        events,
        anchors,
        anchor_col="anchor_date",
        event_date_col="event_date",
        group_col="patient_id",
        window_days=180,
        agg={"value": "sum"},
    )
    # Patient 3 has no events → must appear with sum=0.
    p3_rows = result[result.patient_id == 3]
    assert len(p3_rows) == 1
    assert p3_rows["value_sum"].iloc[0] == 0


def test_outside_window_excluded(basic_events_and_anchors):
    events, anchors = basic_events_and_anchors
    result = temporal_aggregation(
        events,
        anchors,
        anchor_col="anchor_date",
        event_date_col="event_date",
        group_col="patient_id",
        window_days=180,
        agg={"value": "sum"},
    )
    # Patient 2's only event (2024-01-01) is 242 days before anchor —
    # outside 180d window. Must yield 0.
    p2_sum = result.loc[result.patient_id == 2, "value_sum"].iloc[0]
    assert p2_sum == 0


def test_multiple_aggregations(basic_events_and_anchors):
    events, anchors = basic_events_and_anchors
    result = temporal_aggregation(
        events,
        anchors,
        anchor_col="anchor_date",
        event_date_col="event_date",
        group_col="patient_id",
        window_days=180,
        agg={"value": "sum", "event_date": "max"},
    )
    p1 = result.loc[result.patient_id == 1].iloc[0]
    assert p1["value_sum"] == 50
    assert p1["event_date_max"] == pd.Timestamp("2024-08-15")


def test_window_days_must_be_positive(basic_events_and_anchors):
    events, anchors = basic_events_and_anchors
    with pytest.raises(ValueError, match="window_days must be >= 1"):
        temporal_aggregation(
            events,
            anchors,
            anchor_col="anchor_date",
            event_date_col="event_date",
            group_col="patient_id",
            window_days=0,
            agg={"value": "sum"},
        )


def test_window_days_required_keyword(basic_events_and_anchors):
    """window_days is keyword-only — positional call must fail."""
    events, anchors = basic_events_and_anchors
    with pytest.raises(TypeError):
        temporal_aggregation(  # type: ignore[misc]
            events,
            anchors,
            "anchor_date",
            "event_date",
            "patient_id",
            180,
            {"value": "sum"},
        )


def test_missing_required_column_raises():
    events = pd.DataFrame({"patient_id": [1], "event_date": pd.to_datetime(["2024-01-01"])})
    anchors = pd.DataFrame({"patient_id": [1], "anchor_date": pd.to_datetime(["2024-08-30"])})
    with pytest.raises(ValueError, match="agg references column not in events"):
        temporal_aggregation(
            events,
            anchors,
            anchor_col="anchor_date",
            event_date_col="event_date",
            group_col="patient_id",
            window_days=180,
            agg={"missing_column": "sum"},
        )


def test_non_datetime_columns_raise():
    events = pd.DataFrame({"patient_id": [1], "event_date": ["2024-01-01"], "value": [1]})
    anchors = pd.DataFrame({"patient_id": [1], "anchor_date": pd.to_datetime(["2024-08-30"])})
    with pytest.raises(TypeError, match="must be datetime-like"):
        temporal_aggregation(
            events,
            anchors,
            anchor_col="anchor_date",
            event_date_col="event_date",
            group_col="patient_id",
            window_days=180,
            agg={"value": "sum"},
        )


def test_inclusive_at_anchor_exclusive_at_lower_bound():
    """An event exactly at anchor IS included; event exactly at (anchor - window)
    is on the boundary and excluded (since delta_days == window_days, not <).
    """
    anchor = pd.Timestamp("2024-08-30")
    events = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "event_date": [
                anchor,  # delta=0, included
                anchor - pd.Timedelta(days=180),  # delta=180, excluded (boundary)
            ],
            "value": [100, 200],
        }
    )
    anchors = pd.DataFrame({"patient_id": [1], "anchor_date": [anchor]})
    result = temporal_aggregation(
        events,
        anchors,
        anchor_col="anchor_date",
        event_date_col="event_date",
        group_col="patient_id",
        window_days=180,
        agg={"value": "sum"},
    )
    # Only the at-anchor event (100) should count; the boundary event (200)
    # is excluded by the half-open semantics.
    assert result["value_sum"].iloc[0] == 100
