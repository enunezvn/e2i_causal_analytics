from pathlib import Path

import pandas as pd

from scripts.load_synthetic_data import (
    SMALL_SIZES,
    generate_datasets,
    write_cohort_frames,
    write_parquet_snapshots,
)
from src.ml.synthetic.config import DGPType

PATIENT_CELLS = [
    f"{c}__{b}"
    for c in ("initiation", "discontinuation", "persistence")
    for b in ("Remibrutinib", "Kisqali", "Fabhalta")
]


def test_writes_9_patient_cohort_frames(tmp_path):
    datasets = generate_datasets(SMALL_SIZES, DGPType.HETEROGENEOUS, seed=42)
    out = write_parquet_snapshots(datasets, tmp_path)
    write_cohort_frames(out)
    cf = Path(out) / "cohort_frames"
    pj = pd.read_parquet(Path(out) / "patient_journeys.parquet")
    for cell in PATIENT_CELLS:
        cohort, brand = cell.split("__")
        p = cf / f"{cell}.parquet"
        assert p.exists(), f"missing cohort frame {p}"
        frame = pd.read_parquet(p)
        for col in ("treatment_arm", "outcome", "is_synthetic", "treatment_effect_estimate"):
            assert col in frame.columns, f"{cell} frame missing {col}"
        sub = pj[pj["brand"] == brand]
        if cohort in ("discontinuation", "persistence"):
            sub = sub[sub["treatment_initiated"] == 1]
        assert len(frame) == len(sub), (
            f"{cell} frame has {len(frame)} rows for {len(sub)} eligible patients "
            "(ml_predictions merge fan-out duplicates causal units)"
        )
