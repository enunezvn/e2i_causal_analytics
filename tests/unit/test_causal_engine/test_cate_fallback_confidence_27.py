"""RED-FIRST test for #27: CATE estimator fallback CI honors the confidence level.

The CATE estimator's primary CI path is econml's
``cf.effect_interval(X_segment, alpha=alpha)`` -- already derived from the
requested significance level. The ``except`` fallback (cate_estimator.py:652-653)
hardcoded ``1.96 * sigma`` REGARDLESS of alpha, so when a caller requested a 90%
CI (alpha=0.10) and the primary path raised, the fallback silently produced a
95% half-width -- a mislabeled interval.

This test exercises the REAL ``_calculate_cate_by_segment`` fallback. The forest
object's ``.effect_interval`` is made to raise (so we land in the fallback), and
``.effect`` returns a deterministic real numpy array -- the CATE std is a genuine
numpy computation, not a fabricated estimator output. We assert the fallback
half-width tracks the requested alpha (1.645*sigma at alpha=0.10, 1.96*sigma at
alpha=0.05), proving the hardcoded 1.96 is gone.
"""

import numpy as np
import pandas as pd
import pytest


class _FailingIntervalForest:
    """A forest whose point effects are real but whose interval path raises.

    Forces the ``except`` fallback in ``_calculate_cate_by_segment`` while keeping
    the CATE point estimates a genuine, deterministic numpy array (no fabrication
    -- the std the fallback uses is computed from these real values).
    """

    def __init__(self, effects: np.ndarray):
        self._effects = effects

    def effect(self, X):  # noqa: N803 - matches econml signature
        # Return the first len(X) deterministic effects for this segment.
        return self._effects[: len(X)]

    def effect_interval(self, X, alpha):  # noqa: N803, ARG002
        raise RuntimeError("native interval unavailable -> exercise fallback")


@pytest.mark.asyncio
async def test_cate_fallback_half_width_tracks_alpha():
    from src.agents.heterogeneous_optimizer.nodes.cate_estimator import CATEEstimatorNode

    # _calculate_cate_by_segment is a pure helper that never touches the data
    # connector; pass a sentinel so the constructor does not attempt to resolve
    # a real Supabase connector from the environment (offline unit test).
    node = CATEEstimatorNode(data_connector=object())

    # 20 rows, one binary segment var + one effect modifier. >= 10 per segment so
    # the segment is not skipped.
    n = 20
    df = pd.DataFrame(
        {
            "segment": ["A"] * n,
            "mod": list(range(n)),
        }
    )

    # Deterministic, dispersed effects so np.std(cate) > 0 and the z-scaling is
    # observable. Mean 0.30, with spread.
    effects = np.array([0.10, 0.50] * (n // 2), dtype=float)
    cf = _FailingIntervalForest(effects)
    cate_mean = float(np.mean(effects))
    sigma = float(np.std(effects))
    assert sigma > 0  # guard: the test is only meaningful with real dispersion

    # alpha=0.05 (95% CI) -> half-width ~ 1.96 * sigma (legacy, unchanged)
    res_95 = await node._calculate_cate_by_segment(df, cf, ["segment"], ["mod"], 0.05)
    seg95 = res_95["segment"][0]
    hw_95 = seg95["cate_ci_upper"] - seg95["cate_estimate"]
    assert pytest.approx(seg95["cate_estimate"], abs=1e-9) == cate_mean
    assert pytest.approx(hw_95, rel=1e-3) == 1.96 * sigma

    # alpha=0.10 (90% CI) -> half-width ~ 1.645 * sigma, NOT 1.96 * sigma
    res_90 = await node._calculate_cate_by_segment(df, cf, ["segment"], ["mod"], 0.10)
    seg90 = res_90["segment"][0]
    hw_90 = seg90["cate_ci_upper"] - seg90["cate_estimate"]
    assert pytest.approx(hw_90, rel=1e-3) == 1.645 * sigma
    assert hw_90 < hw_95  # 90% interval is strictly narrower than 95%
    # The legacy bug would have produced 1.96*sigma here regardless of alpha:
    assert abs(hw_90 - 1.96 * sigma) > 1e-3, "fallback must NOT hardcode 1.96 at alpha=0.10"
