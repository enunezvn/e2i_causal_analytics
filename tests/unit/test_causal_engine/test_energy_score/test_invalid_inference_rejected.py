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


# --- Only a SERVED fit is refused; a tournament ranking fit is not (codex r3 prep) ------
#
# Measured 2026-09-22 on the real frame: after the load-time prune the FULL frame
# is full rank, yet the 5,000-row stratified tournament subsample for two of the
# three outcomes was rank 61 of 62 (dummies with 15-56 supporting rows can be
# exactly dependent inside a subsample without any column being constant).
# Refusing there knocked LinearDML/DRLearner out of the tournament and the run
# fell to CausalForest (575-664 s, refutation budget exhausted). The subsample
# fit only RANKS estimators -- its CI is never served (#1392: the winner is refit
# on the full frame and only that fit is reported) -- so the refusal applies to
# served fits only; an unserved underdetermined fit keeps its point estimate and
# carries NO interval.


@pytest.mark.parametrize(
    ("wrapper_cls", "estimator_type"),
    [(LinearDMLWrapper, EstimatorType.LINEAR_DML), (DRLearnerWrapper, EstimatorType.DRLEARNER)],
)
def test_unserved_underdetermined_fit_keeps_the_point_estimate_without_a_ci(
    designs, wrapper_cls, estimator_type
):
    t, y, _full_rank, rank_deficient = designs
    r = wrapper_cls(EstimatorConfig(estimator_type)).fit(t, y, rank_deficient, served_fit=False)
    assert r.success, r.error_message
    assert r.ate is not None and np.isfinite(r.ate)
    assert r.ate_ci_lower is None and r.ate_ci_upper is None and r.ate_std is None


def test_selector_marks_only_the_full_frame_refit_as_served(monkeypatch, designs):
    """Two-estimator chain: a single-estimator chain never subsamples (#1392), so
    its lone tournament fit IS the served fit."""
    from src.causal_engine.energy_score.estimator_selector import (
        EstimatorSelector,
        EstimatorSelectorConfig,
        OLSWrapper,
    )

    t, y, full_rank, _ = designs
    calls: list[tuple[str, int, object]] = []
    for cls in (LinearDMLWrapper, OLSWrapper):
        real_fit = cls.fit

        def spy(self, treatment, outcome, covariates, _real=real_fit, _name=cls.__name__, **kw):
            calls.append((_name, len(treatment), kw.get("served_fit")))
            return _real(self, treatment, outcome, covariates, **kw)

        monkeypatch.setattr(cls, "fit", spy)

    chain = [EstimatorConfig(EstimatorType.LINEAR_DML), EstimatorConfig(EstimatorType.OLS)]
    sel = EstimatorSelector(
        EstimatorSelectorConfig(estimators=chain, selection_max_rows=100)
    ).select(t, y, full_rank)
    assert sel.selected.success and sel.selection_subsampled
    tournament = [c for c in calls if c[1] == 100]
    refit = [c for c in calls if c[1] == len(t)]
    assert len(tournament) == 2 and all(c[2] is False for c in tournament), calls
    assert len(refit) == 1 and refit[0][2] is True, calls  # only the winner, full frame, served

    calls.clear()
    EstimatorSelector(EstimatorSelectorConfig(estimators=chain, selection_max_rows=10_000)).select(
        t, y, full_rank
    )
    # no subsample: every tournament fit IS a served fit
    assert len(calls) == 2 and all(c[1] == len(t) and c[2] is True for c in calls), calls
