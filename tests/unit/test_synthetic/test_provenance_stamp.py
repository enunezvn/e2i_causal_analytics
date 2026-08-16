"""Every synthetic dataset must be stamped is_synthetic=True before load."""

import pytest

from scripts.load_synthetic_data import SMALL_SIZES, generate_datasets
from src.ml.synthetic.config import DGPType


# MEASURED 2026-08-15. This test generates every synthetic dataset: 20.2s of
# call time bare, 803MB RSS. Two things about it were diagnosed WRONG before
# this landed, so the measurements are recorded rather than the theories.
#
# It is not slow because of coverage alone (38.6s single-test under `--cov`),
# and it is not fixed by raising the per-test timeout — a 180s bound was tried
# and the job still burned its full 30-minute cap. The cliff is xdist AND
# coverage TOGETHER: reproduced locally, `tests/unit/test_synthetic/` under
# `-n 2 --dist=loadscope --cov=src` was still running past 3m23s, while the
# same directory under the heavy lane's flags (identical xdist, no coverage)
# completes all 475 tests in 153s.
#
# So this directory now runs in the heavy lane, which does not trace coverage.
# The bound below is kept as a real ceiling with headroom over the measured
# cost, so a genuine regression fails fast instead of wedging a worker: under
# xdist the timeout method is `thread`, which cannot preempt this call, and a
# wedged worker costs the whole job rather than one test.
@pytest.mark.timeout(180)
def test_all_datasets_stamped_is_synthetic():
    datasets = generate_datasets(SMALL_SIZES, DGPType.CONFOUNDED, seed=42)
    assert datasets, "generate_datasets returned nothing"
    for table, df in datasets.items():
        assert "is_synthetic" in df.columns, f"{table} not stamped"
        assert df["is_synthetic"].notna().all(), f"{table} has NULL is_synthetic"
        assert bool(df["is_synthetic"].all()) is True, f"{table} has False/0 rows"
