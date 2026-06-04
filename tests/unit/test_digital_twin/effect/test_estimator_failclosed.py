import numpy as np
import pandas as pd
import pytest

from src.causal_engine.errors import EstimationError
from src.digital_twin.effect.errors import EffectDataUnavailable
from src.digital_twin.effect.estimator import TwinEffectEstimator
from src.digital_twin.effect.provider import SyntheticEffectDataProvider, TrainingFrame

_CONFOUNDERS = ["decile", "engagement_score", "adoption_propensity", "tenure_years"]


def _pop(n=50):
    return pd.DataFrame({c: np.zeros(n) for c in _CONFOUNDERS})


def test_empty_frame_fails_closed():
    frame = TrainingFrame(
        df=pd.DataFrame(),
        treatment_var="treatment",
        outcome_var="outcome",
        confounders=["decile"],
        ground_truth_ate=None,
    )
    with pytest.raises(EffectDataUnavailable):
        TwinEffectEstimator().estimate(frame, _pop())


def test_insufficient_rows_fails_closed():
    frame = SyntheticEffectDataProvider(n=100, true_ate=0.1, seed=1).get_training_frame(
        "email_campaign", brand="X", twin_type="hcp"
    )
    est = TwinEffectEstimator(min_training_samples=1000)
    with pytest.raises(EstimationError):
        est.estimate(frame, _pop())


def test_population_missing_confounder_fails_closed():
    frame = SyntheticEffectDataProvider(n=1200, true_ate=0.1, seed=1).get_training_frame(
        "email_campaign", brand="X", twin_type="hcp"
    )
    bad_pop = pd.DataFrame({"decile": np.zeros(10)})  # missing the other confounders
    with pytest.raises(EffectDataUnavailable):
        TwinEffectEstimator(min_training_samples=200).estimate(frame, bad_pop)


def test_empty_twin_population_fails_closed():
    frame = SyntheticEffectDataProvider(n=1200, true_ate=0.1, seed=1).get_training_frame(
        "email_campaign", brand="X", twin_type="hcp"
    )
    empty_pop = pd.DataFrame({c: [] for c in _CONFOUNDERS})  # columns present, 0 rows
    with pytest.raises(EffectDataUnavailable):
        TwinEffectEstimator(min_training_samples=200).estimate(frame, empty_pop)
