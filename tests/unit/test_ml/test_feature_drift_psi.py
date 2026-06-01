"""#577 WS1-MP-009: unit tests for the parametric-Gaussian PSI helper that grounds the
feature_drift seed (migration 053) + the coherent generator mirror.

`_compute_feature_psi(ref_mean, ref_std, cur_mean, cur_std)` is the SINGLE source of the
PSI formula shared by the migration literals, the data_generator mirror, and the live e2e's
in-test recompute. It is parametric (mean/std -> Gaussian-CDF bin masses) — distinct from the
existing sample-based `calculate_psi(expected, actual)` (np.percentile/np.histogram over raw
samples), which is NOT reused here because the reference is stored as parametric mean/std in
ml_preprocessing_metadata.feature_distributions, not as samples.

These assert the statistic is HONEST (the #574 lesson):
- NO-DRIFT DISPROOF: current==reference => PSI is EXACTLY 0.0 (responds only to real drift).
- MONOTONE: a larger mean shift strictly increases PSI (it is not a baked constant).
- PARITY: matches scipy.stats.norm.cdf PSI to <1e-9 (the formula is textbook, not hand-fudged).
- REPRODUCIBILITY: recomputes each migration-053 seeded row's stored test_statistic to <5e-7
  from that row's own stored baseline/current mean/std (the core anti-fabrication proof).
"""

import math

import numpy as np
import pytest

from src.ml.data_generator import _compute_feature_psi

# The 5 seeded rows from migration 053 (stored rounded baseline/current stats + the
# test_statistic literal). Each row must recompute its own PSI from its own stored stats.
_SEEDED_ROWS = [
    # feature,            ref_mean, ref_std, cur_mean, cur_std, stored_psi
    ("age", 0.6178, 0.2537, 0.6990, 0.2791, 0.107134),
    ("risk_score", 0.6739, 0.1466, 0.7149, 0.1363, 0.091486),
    ("days_since_dx", 0.6017, 0.1677, 0.6520, 0.1778, 0.088612),
    ("prior_rx_count", 0.3795, 0.2731, 0.4614, 0.3059, 0.101624),
    ("comorbidity_count", 0.4261, 0.1788, 0.4690, 0.1609, 0.082802),
]


@pytest.mark.parametrize("name,rm,rs,_cm,_cs,_psi", _SEEDED_ROWS)
def test_no_drift_is_exactly_zero(name, rm, rs, _cm, _cs, _psi):
    """current == reference => PSI = 0.0 exactly: (q-p)*ln(q/p) is 0 in every bin when q==p."""
    assert _compute_feature_psi(rm, rs, rm, rs) == 0.0


def test_psi_is_non_negative_and_monotone_in_mean_shift():
    """PSI never negative; a strictly larger mean shift strictly increases PSI — proving it
    RESPONDS to drift magnitude rather than returning a constant."""
    rm, rs = 0.5, 0.2
    prev = -1.0
    for k in (0.0, 0.25, 0.5, 1.0, 2.0):
        psi = _compute_feature_psi(rm, rs, rm + k * rs, rs)
        assert psi >= 0.0
        assert psi > prev, f"PSI not increasing at shift={k}: {psi} <= {prev}"
        prev = psi


@pytest.mark.parametrize("name,rm,rs,cm,cs,stored", _SEEDED_ROWS)
def test_reproduces_seeded_row_test_statistic(name, rm, rs, cm, cs, stored):
    """ANTI-FABRICATION: each migration-053 row's stored test_statistic recomputes from its
    own stored (rounded) baseline/current mean/std to <5e-7 — the row is self-auditable."""
    assert abs(_compute_feature_psi(rm, rs, cm, cs) - stored) < 5e-7


def test_seeded_average_is_low_moderate_band():
    """The corpus AVG PSI lands just-below the 0.10 target (GOOD under lower-is-better),
    spanning the insignificant/low-moderate boundary — a realistic, NOT-suspicious value."""
    avg = float(
        np.mean([_compute_feature_psi(rm, rs, cm, cs) for _, rm, rs, cm, cs, _ in _SEEDED_ROWS])
    )
    assert 0.08 < avg < 0.10
    assert abs(avg - 0.094332) < 5e-6


@pytest.mark.parametrize("name,rm,rs,cm,cs,_psi", _SEEDED_ROWS)
def test_matches_scipy_norm_cdf(name, rm, rs, cm, cs, _psi):
    """PARITY: the stdlib-erf helper equals a scipy.stats.norm.cdf reference implementation
    (no scipy dependency in the helper, but it must agree with the vetted one)."""
    norm = pytest.importorskip("scipy.stats").norm

    def _psi_scipy(rmm, rss, cmm, css, bins=10, w=3, eps=1e-4):
        edges = np.linspace(rmm - w * rss, rmm + w * rss, bins + 1)

        def masses(mean, std):
            cdf = norm.cdf(edges, loc=mean, scale=std)
            p = np.diff(cdf)
            p[0] += cdf[0]
            p[-1] += 1.0 - cdf[-1]
            return p

        p = np.clip(masses(rmm, rss), eps, None)
        q = np.clip(masses(cmm, css), eps, None)
        return float(np.sum((q - p) * np.log(q / p)))

    assert abs(_compute_feature_psi(rm, rs, cm, cs) - _psi_scipy(rm, rs, cm, cs)) < 1e-9


def test_bin_masses_sum_to_one_and_eps_floor_inert_here():
    """Defensive guard documented as a contract: bins sum to 1.0 (tails folded), and the 1e-4
    epsilon floor never engages for these realistic shifts (min bin ~0.008 >> 1e-4). We assert
    the floor is inert by checking the helper agrees with an un-floored direct computation."""
    rm, rs, cm, cs = 0.6178, 0.2537, 0.6990, 0.2791
    edges = np.linspace(rm - 3 * rs, rm + 3 * rs, 11)

    def masses(mean, std):
        z = (edges - mean) / std
        cdf = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z])
        p = np.diff(cdf)
        p[0] += cdf[0]
        p[-1] += 1.0 - cdf[-1]
        return p

    p, q = masses(rm, rs), masses(cm, cs)
    assert abs(p.sum() - 1.0) < 1e-12 and abs(q.sum() - 1.0) < 1e-12
    assert p.min() > 1e-4 and q.min() > 1e-4  # floor inert
    unfloored = float(np.sum((q - p) * np.log(q / p)))
    assert abs(_compute_feature_psi(rm, rs, cm, cs) - unfloored) < 1e-12
