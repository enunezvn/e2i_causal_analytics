"""Shard 09 Task 5b (WS1-DQ-007): stamp recent, bounded data_lag_hours onto the
synthetic patient_journeys frame so v_kpi_data_lag computes non-NULL over now()-30d.
The column ALREADY exists on patient_journeys (faithful-DB verified, integer)."""

import pandas as pd

from src.ml.synthetic.generators.data_lag import (
    stamp_data_lag_hours,
    stamp_sequence_number,
)


def test_data_lag_hours_recent_and_bounded():
    df = pd.DataFrame({"patient_id": [f"p{i}" for i in range(50)], "is_synthetic": [True] * 50})
    out = stamp_data_lag_hours(df, seed=11)
    assert "data_lag_hours" in out.columns
    assert out["data_lag_hours"].notna().all()  # WS1-DQ-007 non-NULL
    assert out["data_lag_hours"].between(1, 168).all()  # 1h..7d plausible ingest lag
    assert out["data_lag_hours"].mean() < 72  # bulk fresh -> v_kpi_data_lag healthy


def test_data_lag_hours_does_not_mutate_input():
    df = pd.DataFrame({"patient_id": ["p0"], "is_synthetic": [True]})
    _ = stamp_data_lag_hours(df, seed=1)
    assert "data_lag_hours" not in df.columns  # operates on a copy


def test_sequence_number_first_prescription_is_one():
    """WS3-BI-006 NRx counts prescriptions with sequence_number=1. The earliest
    prescription per (patient_id, brand) must get sequence_number=1."""
    df = pd.DataFrame(
        {
            "patient_id": ["p0", "p0", "p0", "p1"],
            "brand": ["Kisqali", "Kisqali", "Kisqali", "Kisqali"],
            "event_type": ["prescription", "prescription", "diagnosis", "prescription"],
            "event_date": ["2026-06-05", "2026-06-01", "2026-05-01", "2026-06-03"],
        }
    )
    out = stamp_sequence_number(df)
    assert "sequence_number" in out.columns
    rx = out[out["event_type"] == "prescription"]
    # p0's earliest prescription (2026-06-01) is sequence 1; the later one is 2
    p0 = rx[rx["patient_id"] == "p0"].sort_values("event_date")
    assert list(p0["sequence_number"]) == [1, 2]
    # p1's single prescription is sequence 1
    assert int(rx[rx["patient_id"] == "p1"]["sequence_number"].iloc[0]) == 1
    # non-prescription rows are not numbered (NaN/None)
    assert out[out["event_type"] == "diagnosis"]["sequence_number"].isna().all()
