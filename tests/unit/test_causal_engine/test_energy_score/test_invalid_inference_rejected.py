"""Lane A (codex r2 HIGH): a wrapper whose statsmodels final stage reports
"Co-variance matrix is underdetermined. Inference will be invalid!" must NOT serve
that CI as a valid interval.

``DMLLearnerWrapper`` already refuses (raises -> ``success=False``); on the real
Optum frame (design rank 61 of 77, 2026-09-22) ``LinearDMLWrapper`` served the
warned-about CI and the agent reported p=8.4e-5 with an empty ``warnings`` list.
``LinearDMLWrapper`` and ``DRLearnerWrapper`` share that statsmodels final stage,
so both now mirror the dml_learner refusal. The estimation node then fails
closed on the missing CI (or the tournament skips the estimator) instead of
reporting an interval econml itself calls invalid.

Fast: n=300 rows, one duplicated column, two fits per wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.energy_score.estimator_selector import (
    DRLearnerWrapper,
    EstimatorConfig,
    EstimatorType,
    LinearDMLWrapper,
)


@pytest.fixture(scope="module")
def designs():
    rng = np.random.default_rng(9)
    n = 300
    X = rng.normal(size=(n, 3))
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-X[:, 0]))).astype(int)
    y = 0.5 * t + X[:, 0] + rng.normal(size=n)
    full_rank = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    rank_deficient = full_rank.copy()
    rank_deficient["x0_dup"] = rank_deficient["x0"]  # exact duplicate
    return t, y, full_rank, rank_deficient


@pytest.mark.parametrize(
    ("wrapper_cls", "estimator_type"),
    [(LinearDMLWrapper, EstimatorType.LINEAR_DML), (DRLearnerWrapper, EstimatorType.DRLEARNER)],
)
def test_underdetermined_final_stage_is_refused_not_served(designs, wrapper_cls, estimator_type):
    t, y, full_rank, rank_deficient = designs

    ok = wrapper_cls(EstimatorConfig(estimator_type)).fit(t, y, full_rank)
    assert ok.success, ok.error_message
    assert ok.ate_ci_lower is not None and ok.ate_ci_upper is not None

    bad = wrapper_cls(EstimatorConfig(estimator_type)).fit(t, y, rank_deficient)
    assert not bad.success
    assert bad.ate_ci_lower is None and bad.ate_ci_upper is None
    assert "invalid" in (bad.error_message or "").lower()
