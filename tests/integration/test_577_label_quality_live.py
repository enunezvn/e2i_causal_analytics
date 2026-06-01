"""#577 WS1-DQ-008 (label_quality / IAA) faithful e2e: the metric computes the RIGHT
value against the LIVE DB over the coherently-reseeded ml_annotations — not just "runs".

These assert MEANING (the #574 lesson):
- κ is a real corpus-level generalized Fleiss agreement in a SUBSTANTIAL range (~0.7565
  live) after the latent-truth reseed — NOT the ~0 the pre-rework independent-noise labels
  gave (κ=0.0174). The pre-rework data would FAIL the substantial-band assertion.
- The calculator's value EQUALS an independent in-test recomputation from the raw per-group
  registry counts — proving it COMPUTES the realized statistic, not a constant.
- The per-group label distributions CONCENTRATE on a latent-truth category (high max-share);
  independent noise would average ~1/3.
- Parity: on the live fixed-n=4 subset the generalized κ equals statsmodels.fleiss_kappa.

CAPABILITY-GATED: skips unless SUPABASE_* is set AND the data_quality_label_quality query_id
exists (migration 052 applied).
"""

import os

import numpy as np
import pytest

from src.kpi.calculators.data_quality import DataQualityCalculator

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]

_QUERY_ID = "data_quality_label_quality"


@pytest.fixture
def calc():
    c = DataQualityCalculator()
    if c.db_client is None:
        pytest.skip("no Supabase client")
    try:
        c.db_client.rpc("kpi_query", {"query_id": _QUERY_ID, "params": []}).execute()
    except Exception as e:
        pytest.skip(f"#577 label_quality query unavailable (migration 052 not applied?): {e}")
    return c


def _live_matrix(calc):
    """The live per-group counts matrix [n_positive, n_negative, n_uncertain] + rater totals."""
    rows = calc.db_client.rpc("kpi_query", {"query_id": _QUERY_ID, "params": []}).execute().data
    matrix = np.array(
        [[r["n_positive"], r["n_negative"], r["n_uncertain"]] for r in rows], dtype=float
    )
    return matrix, rows


def test_label_quality_is_substantial_after_coherent_reseed(calc):
    """The corpus generalized Fleiss κ is a real float in the SUBSTANTIAL range (~0.7565),
    NOT the ~0 of independent-noise labels. Range assertion (>0.6), never the exact value
    (the realization is one deterministic hashtext draw; sd≈0.07)."""
    val = calc._calc_label_quality({})
    assert val is not None
    assert 0.6 < val <= 1.0, (
        f"κ={val} not substantial; ~0 would mean independent-noise labels (the pre-rework state)"
    )


def test_label_quality_equals_independent_recomputation(calc):
    """The calculator's value EQUALS an in-test recomputation of the generalized Fleiss κ from
    the raw registry counts — proving it COMPUTES the realized statistic, not a constant."""
    matrix, _ = _live_matrix(calc)
    n_i = matrix.sum(axis=1)
    keep = n_i >= 2
    matrix, n_i = matrix[keep], n_i[keep]
    p_obs = (np.square(matrix).sum(axis=1) - n_i) / (n_i * (n_i - 1))
    p_j = matrix.sum(axis=0) / n_i.sum()
    p_e = float((p_j**2).sum())
    expected = (float(p_obs.mean()) - p_e) / (1.0 - p_e)
    assert abs(calc._calc_label_quality({}) - expected) < 1e-9


def test_label_quality_groups_concentrate_on_latent_truth(calc):
    """Coherence proof: each group's labels concentrate on one (latent-truth) category — the
    mean max-share is high (>0.7); independent noise would average ~1/3 (the pre-rework state)."""
    matrix, _ = _live_matrix(calc)
    n_i = matrix.sum(axis=1)
    max_share = (matrix.max(axis=1) / n_i).mean()
    assert max_share > 0.7, (
        f"groups not concentrated on a latent truth (mean max-share {max_share})"
    )


def test_label_quality_matches_statsmodels_on_live_fixed_n(calc):
    """Parity: on the live fixed-n=4 subset (where classic Fleiss applies) the calculator's
    generalized κ equals the vetted statsmodels.fleiss_kappa."""
    sm = pytest.importorskip("statsmodels.stats.inter_rater")
    matrix, _ = _live_matrix(calc)
    n_i = matrix.sum(axis=1)
    sub = matrix[n_i == 4]
    if sub.shape[0] < 2:
        pytest.skip("fewer than 2 four-rater groups live")
    assert (
        abs(
            DataQualityCalculator._generalized_fleiss_kappa(sub)
            - sm.fleiss_kappa(sub, method="fleiss")
        )
        < 1e-9
    )
