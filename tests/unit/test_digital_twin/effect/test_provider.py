import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect.errors import EffectDataUnavailable
from src.digital_twin.effect.provider import SyntheticEffectDataProvider, TrainingFrame


def test_synthetic_provider_returns_labeled_frame_with_known_ate():
    provider = SyntheticEffectDataProvider(n=2000, true_ate=0.15, seed=42)
    frame = provider.get_training_frame("email_campaign", brand="Remibrutinib", twin_type="hcp")

    assert isinstance(frame, TrainingFrame)
    assert frame.treatment_var == "treatment"
    assert frame.outcome_var == "outcome"
    assert frame.ground_truth_ate == pytest.approx(0.15)
    assert set(np.unique(frame.df["treatment"])) <= {0, 1}
    assert set(np.unique(frame.df["outcome"])) <= {0, 1}
    assert len(frame.df) == 2000
    assert frame.confounders
    treated = frame.df[frame.df["treatment"] == 1]["outcome"].mean()
    control = frame.df[frame.df["treatment"] == 0]["outcome"].mean()
    assert abs((treated - control) - 0.15) < 0.05


def test_synthetic_provider_unknown_intervention_fails_closed():
    provider = SyntheticEffectDataProvider(n=500, true_ate=0.1, seed=1)
    with pytest.raises(EffectDataUnavailable):
        provider.get_training_frame("not_a_real_intervention", brand="X", twin_type="hcp")


def test_reference_covariate_frame_matches_columns_and_recovers_known_ate():
    rng = np.random.default_rng(0)
    # A "real twin" covariate frame with arbitrary numeric columns + ranges.
    twins = pd.DataFrame(
        {
            "decile": rng.integers(1, 11, size=500).astype(float),
            "years_experience": rng.normal(15, 5, size=500),
            "digital_engagement_score": rng.uniform(0, 1, size=500),
            "peer_influence_score": rng.uniform(0, 1, size=500),
        }
    )
    provider = SyntheticEffectDataProvider(n=2000, true_ate=0.2, seed=42)
    frame = provider.get_training_frame(
        "email_campaign", brand="Remibrutinib", twin_type="hcp", reference_covariates=twins
    )
    # Confounders are exactly the reference's numeric columns.
    assert frame.confounders == [
        "decile",
        "years_experience",
        "digital_engagement_score",
        "peer_influence_score",
    ]
    assert frame.treatment_var == "treatment"
    assert frame.outcome_var == "outcome"
    assert frame.ground_truth_ate == pytest.approx(0.2)
    # Training covariates are within the reference ranges (resampled, not standardized away).
    assert frame.df["decile"].min() >= 1 and frame.df["decile"].max() <= 10
    # Known marginal effect is recovered by the raw treated-vs-control gap.
    treated = frame.df[frame.df["treatment"] == 1]["outcome"].mean()
    control = frame.df[frame.df["treatment"] == 0]["outcome"].mean()
    assert abs((treated - control) - 0.2) < 0.06


def test_reference_covariates_with_no_numeric_columns_fails_closed():
    provider = SyntheticEffectDataProvider(true_ate=0.1, seed=1)
    non_numeric = pd.DataFrame({"specialty": ["cardio", "onco"], "region": ["NE", "SW"]})
    with pytest.raises(EffectDataUnavailable):
        provider.get_training_frame(
            "email_campaign", brand="X", twin_type="hcp", reference_covariates=non_numeric
        )
