"""Effect-fit data providers.

v1 ships a transparent synthetic DGP with a KNOWN ground-truth ATE so the
estimator can be validated ("recover known effects", design doc Section 9.3).
The interface is RWD-ready: a future CohortEffectDataProvider returns a frame
with ground_truth_ate=None over the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.digital_twin.effect.errors import EffectDataUnavailable

SUPPORTED_INTERVENTIONS = {
    "email_campaign",
    "call_frequency_increase",
    "speaker_program_invitation",
    "sample_distribution",
    "peer_influence_activation",
    "digital_engagement",
}

_CONFOUNDERS = ["decile", "engagement_score", "adoption_propensity", "tenure_years"]


@dataclass
class TrainingFrame:
    """A labeled frame for uplift fitting."""

    df: pd.DataFrame
    treatment_var: str
    outcome_var: str
    confounders: list[str]
    effect_modifiers: list[str] = field(default_factory=list)
    ground_truth_ate: float | None = None


@runtime_checkable
class EffectDataProvider(Protocol):
    def get_training_frame(
        self, intervention_type: str, brand: str, twin_type: str
    ) -> TrainingFrame: ...


class SyntheticEffectDataProvider:
    """Transparent known-effect DGP for synthetic-first validation.

    DGP: 4 standardized covariates; treatment randomized 50/50 (balanced);
    outcome ~ Bernoulli(clip(p0(X) + true_ate * treatment, 0.01, 0.99)).
    The marginal treated-minus-control conversion gap == true_ate (no clipping
    in the configured operating range), so ground_truth_ate == true_ate.
    """

    def __init__(self, n: int = 2000, true_ate: float = 0.15, seed: int = 42) -> None:
        self.n = n
        self.true_ate = true_ate
        self.seed = seed

    def get_training_frame(
        self, intervention_type: str, brand: str, twin_type: str
    ) -> TrainingFrame:
        if intervention_type not in SUPPORTED_INTERVENTIONS:
            raise EffectDataUnavailable(
                f"SyntheticEffectDataProvider: unsupported intervention '{intervention_type}'."
            )
        # brand/twin_type are unused: the synthetic DGP is intervention-parameterized only.
        rng = np.random.default_rng(self.seed)
        n = self.n
        decile = rng.integers(1, 11, size=n).astype(float)
        engagement = rng.normal(0.0, 1.0, size=n)
        adoption = rng.normal(0.0, 1.0, size=n)
        tenure = rng.normal(0.0, 1.0, size=n)
        treatment = rng.integers(0, 2, size=n)

        logit_like = 0.02 * (decile - 5) + 0.05 * engagement + 0.05 * adoption
        p0 = np.clip(0.35 + logit_like, 0.15, 0.55)
        p = np.clip(p0 + self.true_ate * treatment, 0.01, 0.99)
        outcome = (rng.random(n) < p).astype(int)

        df = pd.DataFrame(
            {
                "decile": decile,
                "engagement_score": engagement,
                "adoption_propensity": adoption,
                "tenure_years": tenure,
                "treatment": treatment,
                "outcome": outcome,
            }
        )
        return TrainingFrame(
            df=df,
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=list(_CONFOUNDERS),
            effect_modifiers=["decile", "engagement_score"],
            ground_truth_ate=self.true_ate,
        )
