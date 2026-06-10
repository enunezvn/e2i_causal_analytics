"""Shard 09 Task 5b (WS1-DQ-007): stamp recent, bounded data_lag_hours onto the
synthetic patient_journeys frame so v_kpi_data_lag computes non-NULL over now()-30d.
The column ALREADY exists on patient_journeys (faithful-DB verified, integer)."""

import pandas as pd

from src.ml.synthetic.generators.data_lag import stamp_data_lag_hours


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
