import numpy as np
import pandas as pd

from src.digital_twin.effect.estimate import PROVENANCE_SYNTHETIC, EffectEstimate
from src.digital_twin.effect.estimator import TwinEffectEstimator
from src.digital_twin.effect.provider import SyntheticEffectDataProvider


def _twin_population(n=300, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "decile": rng.integers(1, 11, size=n).astype(float),
            "engagement_score": rng.normal(0, 1, size=n),
            "adoption_propensity": rng.normal(0, 1, size=n),
            "tenure_years": rng.normal(0, 1, size=n),
        }
    )


def test_estimator_returns_effect_estimate_over_twin_population():
    frame = SyntheticEffectDataProvider(n=800, true_ate=0.2, seed=42).get_training_frame(
        "email_campaign", brand="Remibrutinib", twin_type="hcp"
    )
    population = _twin_population(n=300)
    est = TwinEffectEstimator(n_estimators=40, max_depth=4, min_training_samples=200)

    result = est.estimate(frame, population)

    assert isinstance(result, EffectEstimate)
    assert result.estimator_type == "uplift_random_forest"
    assert result.data_provenance == PROVENANCE_SYNTHETIC
    assert result.per_twin_uplift.ravel().shape[0] == 300
    assert result.ate_ci_lower <= result.ate <= result.ate_ci_upper
    assert result.n_train == 800
    assert result.ate > 0
