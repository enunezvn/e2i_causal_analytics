"""``dml_learner``: econml's general ``DML`` with a flexible (featurized) final stage.

Split out of ``estimator_selector.py`` (module-size ratchet, #1991); registered in
its ``ESTIMATOR_WRAPPERS``. Design measurements:
``docs/demos/results/2026-09-19_dml_learner/disproof.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.causal_engine.energy_score.estimator_selector import (
    BaseEstimatorWrapper,
    EstimatorConfig,
    EstimatorResult,
    EstimatorType,
    _honest_ate_ci,
    logger,
)


class DMLLearnerWrapper(BaseEstimatorWrapper):
    """Wrapper for EconML ``DML`` -- the general DML framework with a flexible final stage.

    ``LinearDML`` is the special case whose final stage is linear in the raw X.
    Here X is featurized (degree-2 polynomial: squares + pairwise interactions)
    before the final stage, so the CATE surface is non-linear in X while the
    final stage stays a ``StatsModelsLinearRegression`` -- which is what keeps
    econml's analytic ``ate_inference`` available (#1188 honest interval). All
    models come from ``nuisance_config`` so the refutation rebuild
    (``backdoor.econml.dml.DML``) fits the SAME estimator; see that module for
    the measured reasons behind the GradientBoosting nuisances.
    """

    def __init__(self, config: EstimatorConfig):
        self.config = config

    @property
    def estimator_type(self) -> EstimatorType:
        return EstimatorType.DML_LEARNER

    def fit(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> EstimatorResult:
        import time

        start = time.perf_counter()

        try:
            from econml.dml import DML

            from src.causal_engine.nuisance_config import (
                DML_LEARNER_RANDOM_STATE,
                dml_learner_init_params,
            )

            model = DML(
                **dml_learner_init_params(discrete_treatment=True),
                discrete_treatment=True,
                random_state=DML_LEARNER_RANDOM_STATE,
            )
            X = covariates.values
            model.fit(outcome, treatment, X=X, W=X)

            cate = model.effect(X)
            ate = float(np.mean(cate))

            # Population ATE SAMPLING interval (honest; #1188).
            inference = _honest_ate_ci(model, X)
            if inference is not None:
                ate_ci_lower, ate_ci_upper, ate_std = inference
            else:
                ate_ci_lower = ate_ci_upper = ate_std = None  # type: ignore[assignment]

            # Propensity scores
            from sklearn.linear_model import LogisticRegressionCV

            ps_model = LogisticRegressionCV(cv=3, max_iter=500)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]

            elapsed = (time.perf_counter() - start) * 1000

            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=True,
                ate=ate,
                cate=cate,
                ate_std=ate_std,
                ate_ci_lower=ate_ci_lower,
                ate_ci_upper=ate_ci_upper,
                propensity_scores=propensity_scores,
                estimation_time_ms=elapsed,
                raw_estimate=model,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"DML learner failed: {e}")
            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )
