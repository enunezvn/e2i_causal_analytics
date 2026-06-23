"""Task 03.5 — ATE/CATE recovery probe (gate 3 data layer).

The cheapest-disproof made permanent: generate one (cohort,brand) frame, recover
TRUE_ATE with LinearDML, recover per-segment CATE ordering with CausalForestDML,
and prove the propensity is estimable with both arms populated. In-process, no DB.
"""

import pytest

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.dgp.recovery_probe import recover_ate_and_cate
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

pytestmark = pytest.mark.heavy_ml  # groups econml import on one worker (pyproject:214)


# All 3 brands — Fabhalta INCLUDED. Its segment CATE ordering was historically
# the fragile case (high>med>low held only 2/3 before the T11 latent-CATE boost);
# leaving it out of the strict-ordering gate let that fragility regress silently.
# Measured (seed=21, n=3000): Fabhalta high 0.227 > med 0.128 > low 0.036, |ATE
# err| 0.025 — now locked.
@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_estimators_recover_true_ate_and_cate_ordering(brand):
    cfg = GeneratorConfig(seed=21, n_records=3000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    out = recover_ate_and_cate(df)

    # propensity estimable, both arms populated (overlap)
    assert out["propensity_auc"] > 0.5
    assert out["n_treated"] >= 30 and out["n_control"] >= 100

    # ATE recovery: LinearDML within tolerance of the realized TRUE_ATE.
    # Risk-difference scale tolerance is wider than the latent CATE (binary
    # outcome attenuates), so we assert SIGN + ORDERING, and ATE within 0.15.
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out

    # per-segment CATE ordering matches the DGP map high>medium>low
    cate = out["cate_by_segment_estimate"]
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"]
