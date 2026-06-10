"""Task 03.3 — prevalence-banded binary outcome carrying per-segment CATE."""

import numpy as np

from src.ml.synthetic.config import Brand
from src.ml.synthetic.dgp.treatment_arm import (
    SEGMENT_HIGH,
    SEGMENT_LOW,
    assign_segment,
    assign_treatment_arm,
    binary_outcome_with_cate,
    brand_scaled_cate,
)


def _frame(brand, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    sev = np.clip(rng.normal(5.0, 2.0, n), 0, 10)
    acad = (rng.random(n) < 0.30).astype(int)
    X = {"disease_severity": sev, "academic_hcp": acad}
    arm, _ = assign_treatment_arm(X, rng)
    seg = assign_segment(sev)
    cate_map = brand_scaled_cate(brand)
    y, tau_i = binary_outcome_with_cate(arm, X, seg, cate_map, rng)
    return arm, seg, y, tau_i, cate_map


def test_prevalence_in_band():
    _, _, y, _, _ = _frame(Brand.REMIBRUTINIB)
    assert 0.20 <= y.mean() <= 0.50, f"prevalence {y.mean():.3f} outside [0.20,0.50]"


def test_risk_difference_orders_by_segment():
    arm, seg, y, _, _ = _frame(Brand.KISQALI)

    def rd(mask):
        t, c = y[mask & (arm == 1)], y[mask & (arm == 0)]
        return t.mean() - c.mean()

    rd_high = rd(seg == SEGMENT_HIGH)
    rd_low = rd(seg == SEGMENT_LOW)
    assert rd_high > rd_low > -0.05, f"high RD {rd_high:.3f} !> low RD {rd_low:.3f}"


def test_tau_i_is_recoverable_rd_scale_segment_cate():
    """tau_i is the per-segment counterfactual RISK-DIFFERENCE CATE (recoverable,
    de-confounded), NOT the latent-scale cate_map. It takes exactly 3 distinct
    values, ordered high>medium>low>0, and on the RD (probability) scale each is
    strictly smaller than the corresponding latent CATE (binarization attenuates).
    """
    from src.ml.synthetic.dgp.treatment_arm import SEGMENT_MEDIUM, rd_map_from_tau

    _, seg, _, tau_i, latent_map = _frame(Brand.FABHALTA)
    # exactly 3 distinct per-segment values
    rd = rd_map_from_tau(seg, tau_i)
    assert len(set(np.round(list(rd.values()), 6))) == 3
    # ordered high>medium>low>0 (de-confounded RD ordering preserved)
    assert rd[SEGMENT_HIGH] > rd[SEGMENT_MEDIUM] > rd[SEGMENT_LOW] > 0
    # each RD-scale value is smaller than the latent CATE it derives from
    for s in (SEGMENT_HIGH, SEGMENT_MEDIUM, SEGMENT_LOW):
        assert 0 < rd[s] < latent_map[s]
