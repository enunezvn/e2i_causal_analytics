"""WS1-DQ-007 data-lag stamper (Shard 09).

Adds a recent, bounded data_lag_hours to a synthetic patient_journeys frame so
v_kpi_data_lag computes non-NULL over now()-30d. The column ALREADY exists on
patient_journeys (faithful-DB verified: integer, nullable) -- no DDL needed. The
loader carries it via TABLE_COLUMNS["patient_journeys"] (Task 1).
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
