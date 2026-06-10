"""Post-hoc column stampers for Shard 09 KPI substrate.

stamp_data_lag_hours (WS1-DQ-007): recent, bounded data_lag_hours on the synthetic
patient_journeys frame so v_kpi_data_lag computes non-NULL over now()-30d.

stamp_sequence_number (WS3-BI-006 NRx): per-(patient,brand) chronological index of
prescription events so the NRx KPI (sequence_number=1) counts new prescriptions.

Both columns ALREADY exist on the faithful DB (integer, nullable) -- no DDL needed;
the loader carries them via TABLE_COLUMNS (Task 1).
"""

import numpy as np
import pandas as pd


def stamp_data_lag_hours(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Return a copy of df with a recent, bounded integer data_lag_hours column.

    Right-skewed toward fresh: gamma(shape=2, scale=18) (mean ~36h), clipped to
    [1, 168] (1h..7d plausible ingest lag) so the mean stays well under the 72h
    health threshold while the tail still reaches a week.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    lag = np.clip(rng.gamma(shape=2.0, scale=18.0, size=len(out)).round().astype(int), 1, 168)
    out["data_lag_hours"] = lag
    return out


def stamp_sequence_number(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a per-(patient_id, brand) chronological
    sequence_number on prescription events (1 = the first/new prescription).

    NRx (WS3-BI-006) counts treatment_events WHERE event_type='prescription' AND
    sequence_number=1. The synthetic treatment generator does not emit this column,
    so every synthetic prescription would be missed without this stamp. Only
    prescription rows are numbered; non-prescription rows keep a NULL sequence.
    """
    out = df.copy()
    if "event_type" not in out.columns or out.empty:
        out["sequence_number"] = pd.NA
        return out
    out["sequence_number"] = pd.NA
    is_rx = out["event_type"] == "prescription"
    rx = out.loc[is_rx].copy()
    if not rx.empty:
        group_cols = [c for c in ("patient_id", "brand") if c in rx.columns]
        # Stable chronological rank within each (patient, brand): 1, 2, 3, ...
        rx = rx.sort_values([*group_cols, "event_date"], kind="mergesort")
        rx["__seq"] = rx.groupby(group_cols, dropna=False).cumcount() + 1
        out.loc[rx.index, "sequence_number"] = rx["__seq"].astype(int)
    return out
