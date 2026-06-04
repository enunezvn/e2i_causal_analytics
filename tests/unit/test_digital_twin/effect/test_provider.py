import numpy as np
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
