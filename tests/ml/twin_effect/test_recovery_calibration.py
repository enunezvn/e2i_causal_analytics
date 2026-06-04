"""Phase-2 (heavy): the estimator recovers known synthetic ATEs within 20%.

Memory-heavy (full-size uplift forests across DGPs) -> @pytest.mark.slow, run in
the isolated/sharded slow-tests lane, NOT the light backend lane.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect.estimator import TwinEffectEstimator
from src.digital_twin.effect.provider import SyntheticEffectDataProvider

pytestmark = pytest.mark.slow

# Calibration output (Task 9 recovery sweep). These params recover all four
# synthetic effect sizes within the 20% relative-error bound; mirrored into
# config/digital_twin_config.yaml `effect_engine:`.
_N = 6000
_N_ESTIMATORS = 300
_MAX_DEPTH = 6
_MIN_TRAINING_SAMPLES = 1000


def _population(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "decile": rng.integers(1, 11, size=n).astype(float),
            "engagement_score": rng.normal(0, 1, size=n),
            "adoption_propensity": rng.normal(0, 1, size=n),
            "tenure_years": rng.normal(0, 1, size=n),
        }
    )


@pytest.mark.parametrize("true_ate", [0.05, 0.10, 0.20])
def test_recovers_known_ate_within_20_percent(true_ate: float) -> None:
    frame = SyntheticEffectDataProvider(n=_N, true_ate=true_ate, seed=42).get_training_frame(
        "email_campaign", brand="Remibrutinib", twin_type="hcp"
    )
    pop = _population(n=_N, seed=99)
    est = TwinEffectEstimator(
        n_estimators=_N_ESTIMATORS,
        max_depth=_MAX_DEPTH,
        min_training_samples=_MIN_TRAINING_SAMPLES,
    ).estimate(frame, pop)
    rel_err = abs(est.ate - true_ate) / true_ate
    assert rel_err < 0.20, f"ATE {est.ate:.4f} vs truth {true_ate}: rel_err {rel_err:.2%}"
    del frame, pop, est
    gc.collect()


def test_near_zero_effect_skip_path() -> None:
    """A ~null effect must NOT produce a confidently-positive CI (SKIP must be reachable)."""
    frame = SyntheticEffectDataProvider(n=_N, true_ate=0.0, seed=7).get_training_frame(
        "email_campaign", brand="Remibrutinib", twin_type="hcp"
    )
    pop = _population(n=_N, seed=8)
    est = TwinEffectEstimator(
        n_estimators=_N_ESTIMATORS,
        max_depth=_MAX_DEPTH,
        min_training_samples=_MIN_TRAINING_SAMPLES,
    ).estimate(frame, pop)
    assert est.ate_ci_lower <= 0.05  # CI lower does not clear a 5% min-effect bar
    del frame, pop, est
    gc.collect()
