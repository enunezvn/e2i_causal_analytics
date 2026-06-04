"""Uplift-based causal effect estimator for the digital twin.

Fits causal_engine.uplift.UpliftRandomForest on a labeled TrainingFrame, then
predicts per-twin uplift over the (covariate-only) twin population. The training
frame and the scoring population are DISTINCT (per design): the model learns the
treatment-effect function from labeled data and applies it to the twins.

Fail-closed (CLAUDE.md anti-mocking): no heuristic fallback. Bad/insufficient
data raises; the caller surfaces a failed simulation rather than a fake ATE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal_engine.errors import EstimationError
from src.causal_engine.uplift import UpliftConfig, UpliftRandomForest
from src.digital_twin.effect.errors import EffectDataUnavailable
from src.digital_twin.effect.estimate import PROVENANCE_SYNTHETIC, EffectEstimate
from src.digital_twin.effect.provider import TrainingFrame

DEFAULT_MIN_TRAINING_SAMPLES = 1000


def _to_1d(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr


class TwinEffectEstimator:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_training_samples: int = DEFAULT_MIN_TRAINING_SAMPLES,
        provenance: str = PROVENANCE_SYNTHETIC,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_training_samples = min_training_samples
        self.provenance = provenance

    def estimate(self, frame: TrainingFrame, twin_population: pd.DataFrame) -> EffectEstimate:
        df = frame.df
        if df is None or len(df) == 0:
            raise EffectDataUnavailable("TwinEffectEstimator: empty training frame.")
        if len(df) < self.min_training_samples:
            raise EstimationError(
                f"TwinEffectEstimator: {len(df)} training rows < "
                f"min_training_samples={self.min_training_samples}."
            )
        missing = [c for c in frame.confounders if c not in twin_population.columns]
        if missing:
            raise EffectDataUnavailable(
                f"TwinEffectEstimator: twin population missing confounders {missing}."
            )
        if len(twin_population) == 0:
            raise EffectDataUnavailable("TwinEffectEstimator: empty twin population.")

        x_train = df[frame.confounders]
        treatment = df[frame.treatment_var].to_numpy()
        y = df[frame.outcome_var].to_numpy().astype(float)
        x_twin = twin_population[frame.confounders]

        config = UpliftConfig(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=max(10, len(df) // 50),
            control_name="0",  # binary 0/1 treatment; control label is "0"
            random_state=42,
        )
        model = UpliftRandomForest(config)
        result = model.estimate(x_train, treatment, y)
        if not result.success:
            raise EstimationError(f"TwinEffectEstimator: uplift fit failed: {result.error_message}")

        twin_scores = _to_1d(model.predict(x_twin))
        population_ate = float(np.mean(twin_scores))

        # CI: take the uplift model's TRAINING-frame inferential CI half-width
        # (driven by the labeled-data sample size n_train, NOT the twin count) and
        # recentre it on the population ATE. This keeps the interval bounded by
        # training evidence, so it does not collapse as more twins are simulated.
        # (A bootstrap CI is the v1.1 high-fidelity mode on the async offload path.)
        if result.ate_ci_lower is not None and result.ate_ci_upper is not None:
            half_width = float(result.ate_ci_upper - result.ate_ci_lower) / 2.0
            ci_lower, ci_upper = population_ate - half_width, population_ate + half_width
        elif result.ate_std is not None:
            n_train = int(result.metadata.get("n_samples_train", len(df)))
            margin = 1.96 * float(result.ate_std) / np.sqrt(max(1, n_train))
            ci_lower, ci_upper = population_ate - margin, population_ate + margin
        else:
            ci_lower = ci_upper = population_ate

        return EffectEstimate(
            ate=population_ate,
            ate_ci_lower=float(ci_lower),
            ate_ci_upper=float(ci_upper),
            att=result.att,
            atc=result.atc,
            per_twin_uplift=twin_scores,
            auuc=None,
            qini=None,
            feature_importances=result.feature_importances,
            n_train=len(df),
            estimator_type="uplift_random_forest",
            data_provenance=self.provenance,
        )
