"""Every synthetic dataset must be stamped is_synthetic=True before load."""

import pytest

from scripts.load_synthetic_data import SMALL_SIZES, generate_datasets
from src.ml.synthetic.config import DGPType


# MEASURED 2026-08-15: this test generates every synthetic dataset and costs
# 20.2s of call time bare, but 38.6s under `--cov=src` — which is how CI runs
# it, against a `--timeout=30` cap. It therefore sits just the wrong side of
# the per-test timeout on any run where the runner is not fast.
#
# Crossing that line does NOT produce a 30s failure. Under xdist the timeout
# method is `thread`, which cannot preempt this call, so the worker wedges and
# the JOB burns its full 30-minute cap instead — observed twice on #1643,
# both times with this as the last test standing at 99% and orphaned pytest
# processes reaped at the end. A latent landmine for any branch that adds
# enough tests to slow the lane.
#
# The bound is set from the measured cost with headroom, so the test fails
# fast if it ever genuinely regresses rather than costing 30 minutes of CI.
@pytest.mark.timeout(180)
def test_all_datasets_stamped_is_synthetic():
    datasets = generate_datasets(SMALL_SIZES, DGPType.CONFOUNDED, seed=42)
    assert datasets, "generate_datasets returned nothing"
    for table, df in datasets.items():
        assert "is_synthetic" in df.columns, f"{table} not stamped"
        assert df["is_synthetic"].notna().all(), f"{table} has NULL is_synthetic"
        assert bool(df["is_synthetic"].all()) is True, f"{table} has False/0 rows"
