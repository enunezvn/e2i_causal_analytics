"""P6 (MED) — uplift _calculate_ate ATE/ATT/ATC are subpopulation-specific.

The values are honest MODEL-PREDICTED uplift means (documented as such), not
identification-validated estimands. This test pins that ATT and ATC are genuinely
computed over the treated / control subpopulations — they differ when those
subpopulations differ (they are not a forced duplicate).
"""

from __future__ import annotations

import numpy as np

from src.causal_engine.uplift.base import BaseUpliftModel, UpliftConfig, UpliftModelType


class _MiniUplift(BaseUpliftModel):
    @property
    def model_type(self) -> UpliftModelType:
        return UpliftModelType.UPLIFT_RANDOM_FOREST

    def _create_model(self):
        return None


def test_att_atc_differ_on_asymmetric_subpopulations():
    model = _MiniUplift(UpliftConfig())
    n = 1000
    rng = np.random.RandomState(0)
    treatment = rng.binomial(1, 0.5, n).astype(int)
    # Asymmetric: treated units carry systematically higher predicted uplift.
    scores = np.where(treatment == 1, rng.normal(0.5, 0.1, n), rng.normal(0.1, 0.1, n)).astype(
        float
    )
    y = rng.binomial(1, 0.3, n).astype(float)

    ate, att, atc, ate_disp = model._calculate_ate(scores, treatment, y)

    assert att is not None and atc is not None
    assert abs(att - atc) > 0.2, f"ATT and ATC must differ on asymmetric subpops: {att} vs {atc}"
    # ATE (overall mean) lies between the two subpopulation means.
    assert min(att, atc) - 1e-9 <= ate <= max(att, atc) + 1e-9
    # ate_disp is a dispersion (std of predicted scores), non-negative.
    assert ate_disp is not None and ate_disp >= 0.0
