"""Lane T2: the cohort estimate reports the causal forest's DOUBLY-ROBUST ATE and its standard
error, not ``CausalForestDML.ate_interval``.

Why (docs/demos/results/2026-09-23_ab_reload_interval/cert.md section 3, and the live premise
probe docs/demos/results/2026-09-23_t2_premise_probe/): ``ate_interval`` averages the forest's
per-row CATE intervals, which econml documents as an upper bound, and it measured 1.57-3.94x
wider than the DR interval on the live joined cohort. On the planted channels the DR interval
excluded 0 in 12/12 fits and the shipped one in 1/12. ``cf.ate_ +- z * cf.ate_stderr_``
calibrated at 1.06-1.52x the empirical SD in the seed Monte-Carlo.

Every test fits a REAL ``CausalForestDML`` on a synthetic cohort with a PLANTED effect on a 0/1
``adopted`` outcome, built the way the live DGP builds it: a logit shift ``beta * (tbin - 0.5)``
on the channel's median contrast, confounded by ``market_share``. Nothing is mocked; the two
defect tests subclass the real forest and plant one defect after a real fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from econml.dml import CausalForestDML as _RealForest  # bound before any monkeypatch
from scipy.stats import norm

from src.digital_twin.effect import cohort_causal_estimator as cce
from src.digital_twin.effect.cohort_causal_estimator import estimate_cohort_effect
from src.digital_twin.effect.errors import EffectCause, EffectDataUnavailable

REGIONS = ("northeast", "south", "midwest", "west")
N = 3400  # the live joined cohort is 3,354-3,404 HCPs per brand
PLANTED = "patient_support_enrollment"  # planted at beta 0.4: RD ~0.06, below the focus channels
NULL = "rep_training_score"  # planted with no effect
SEED = 42


def _planted_cohort(n: int = N, seed: int = 11, beta: float = 0.4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.uniform(0.0, 1.0, n)
    triggers = rng.poisson(80, n).astype(float)
    region = rng.choice(REGIONS, n)
    # The channel is confounded by market_share, which also moves the outcome.
    psp = 1.0 / (1.0 + np.exp(-(1.5 * (market - 0.5) + rng.normal(0.0, 1.0, n))))
    rep = rng.uniform(0.0, 10.0, n)
    tbin = (psp > np.median(psp)).astype(float)
    logit = -0.6 + 1.2 * (market - 0.5) + beta * (tbin - 0.5) + rng.normal(0.0, 0.8, n)
    adopted = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "hcp_id": [f"h{i:05d}" for i in range(n)],
            "region": region,
            "market_share": market,
            "triggers_total_count": triggers,
            PLANTED: psp,
            NULL: rep,
            "adopted": adopted,
        }
    )


def _mirror_fit(cohort: pd.DataFrame, channel: str):
    """The forest ``estimate_cohort_effect`` fits, built the same way, for reading the
    library's own attributes off it (``ate_interval``, ``_oob_preds``)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    work = cce._usable_rows(
        cohort,
        channel,
        outcome_col="adopted",
        region_col="region",
        confounders=cce.DEFAULT_CONFOUNDERS,
    )
    t = (work["t_raw"] > float(work["t_raw"].median())).astype(int).to_numpy()
    x = cce._effect_modifier_matrix(work)
    w = np.column_stack(
        [
            np.log1p(np.clip(work[c].to_numpy(dtype=float), 0.0, None))
            if c in cce._LOG_CONFOUNDERS
            else work[c].to_numpy(dtype=float)
            for c in cce.DEFAULT_CONFOUNDERS
        ]
    )
    cf = _RealForest(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=SEED),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=SEED),
        discrete_treatment=True,
        n_estimators=200,
        subforest_size=4,
        min_samples_leaf=10,
        random_state=SEED,
    )
    cf.fit(work["y"].to_numpy(dtype=float), t, X=x, W=w)
    return cf, x, work


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return _planted_cohort()


@pytest.fixture(scope="module")
def planted(cohort):
    return estimate_cohort_effect(cohort, PLANTED)


@pytest.fixture(scope="module")
def mirror(cohort):
    return _mirror_fit(cohort, PLANTED)


def _z(alpha: float = 0.05) -> float:
    return float(norm.ppf(1.0 - alpha / 2.0))


def test_the_headline_is_the_forests_dr_ate_and_z_times_its_stderr(planted, mirror):
    cf, _x, _work = mirror
    ate = float(np.ravel(cf.ate_)[0])
    se = float(np.ravel(cf.ate_stderr_)[0])
    assert planted.ate == pytest.approx(ate, abs=1e-12)
    assert planted.ate_stderr == pytest.approx(se, abs=1e-12)
    assert planted.ate_ci_lower == pytest.approx(ate - _z() * se, abs=1e-12)
    assert planted.ate_ci_upper == pytest.approx(ate + _z() * se, abs=1e-12)
    assert planted.interval_method == "dr_ate_stderr"


def test_alpha_sets_z_two_sided(cohort):
    wide = estimate_cohort_effect(cohort, PLANTED, alpha=0.01)
    assert (wide.ate_ci_upper - wide.ate) == pytest.approx(_z(0.01) * wide.ate_stderr, abs=1e-12)


def test_econml_still_aliases_oob_preds_to_the_dr_pseudo_outcomes(mirror):
    """Review finding 3 pin. ``_oob_preds`` holds the DR pseudo-outcomes only because econml
    0.16.0 mutates it in place (``drpreds = oob_preds; drpreds += ...``). If a later econml
    copies first, ``_oob_preds`` reverts to raw OOB predictions and a subset mean over it is a
    plausible-but-wrong ATE with the DR correction silently dropped. Equality to the PUBLIC
    ``ate_`` is the only thing that catches that."""
    cf, _x, _work = mirror
    oob = cf.rlearner_model_final_._oob_preds
    assert np.nanmean(oob) == pytest.approx(float(np.ravel(cf.ate_)[0]), rel=1e-9, abs=1e-12)
    point, stderr = cf.rlearner_model_final_._ate_and_stderr(oob, np.ones(len(oob), dtype=bool))
    assert float(np.ravel(point)[0]) == pytest.approx(float(np.ravel(cf.ate_)[0]), abs=1e-12)
    assert float(np.ravel(stderr)[0]) == pytest.approx(
        float(np.ravel(cf.ate_stderr_)[0]), abs=1e-12
    )


def test_the_dr_interval_is_narrower_than_the_shipped_ate_interval(planted, mirror):
    cf, x, _work = mirror
    lo, hi = cf.ate_interval(x, alpha=0.05)
    shipped = float(np.ravel(hi)[0]) - float(np.ravel(lo)[0])
    assert planted.ci_width() < shipped
    # The live probe measured 1.57-3.94x; the planted frame lands in the same band.
    assert 1.5 < shipped / planted.ci_width() < 6.0


def test_a_planted_channel_turns_significant_where_the_shipped_interval_covered_zero(
    planted, mirror
):
    """The consumer that flips: ``is_significant`` reads the CI's sign (simulation_models)."""
    cf, x, _work = mirror
    lo, hi = cf.ate_interval(x, alpha=0.05)
    assert float(np.ravel(lo)[0]) < 0.0 < float(np.ravel(hi)[0])  # shipped: not significant
    assert planted.ate_ci_lower > 0.0  # DR: significant
    # and the point is the planted risk difference, not an artefact of the narrowing
    assert 0.03 < planted.ate < 0.10  # realised RD ~0.155 * beta (per_hcp_cohort_columns)


def test_the_null_channel_interval_covers_zero(cohort):
    null = estimate_cohort_effect(cohort, NULL)
    assert null.ate_ci_lower < 0.0 < null.ate_ci_upper


def test_a_target_region_gets_the_dr_mean_and_stderr_over_its_rows(cohort, mirror):
    cf, _x, work = mirror
    targeted = estimate_cohort_effect(cohort, PLANTED, target_regions=["northeast", "west"])
    mask = work["region"].isin(["northeast", "west"]).to_numpy()
    point, stderr = cf.rlearner_model_final_._ate_and_stderr(
        cf.rlearner_model_final_._oob_preds, mask
    )
    p, s = float(np.ravel(point)[0]), float(np.ravel(stderr)[0])
    assert targeted.target_n == int(mask.sum())
    assert targeted.target_ate == pytest.approx(p, abs=1e-12)
    assert targeted.target_ci_lower == pytest.approx(p - _z() * s, abs=1e-12)
    assert targeted.target_ci_upper == pytest.approx(p + _z() * s, abs=1e-12)
    # A subset of the rows has a wider interval than the whole cohort.
    assert (targeted.target_ci_upper - targeted.target_ci_lower) > targeted.ci_width()


class _ForestWithoutOobPreds(_RealForest):
    """A real CausalForestDML whose DR pseudo-outcomes are gone after a real fit (e.g. a
    ``drate=False`` construction, or an econml that stops keeping them)."""

    def fit(self, *args, **kw):
        super().fit(*args, **kw)
        del self.rlearner_model_final_._oob_preds
        return self


class _ForestWithRawOobPreds(_RealForest):
    """A real forest after a real fit, with ``_oob_preds`` no longer the DR pseudo-outcomes:
    what an upstream ``.copy()`` before the in-place DR correction would leave behind."""

    def fit(self, *args, **kw):
        super().fit(*args, **kw)
        final = self.rlearner_model_final_
        final._oob_preds = final._oob_preds.copy() + 0.05
        return self


@pytest.mark.parametrize("targets", [[], ["northeast"]])
@pytest.mark.parametrize("forest", [_ForestWithoutOobPreds, _ForestWithRawOobPreds])
def test_an_estimate_without_trustworthy_dr_pseudo_outcomes_is_refused(
    monkeypatch, cohort, forest, targets
):
    """The region effects and any targeted interval are read off the DR pseudo-outcomes; a
    forest whose array is gone or no longer reproduces ``ate_`` gets no estimate at all."""
    monkeypatch.setattr("econml.dml.CausalForestDML", forest)
    with pytest.raises(EffectDataUnavailable) as caught:
        estimate_cohort_effect(cohort, PLANTED, target_regions=targets)
    assert caught.value.cause is EffectCause.ESTIMATION_FAILED
    assert caught.value.details["is_target_inference"] is False


def test_region_effects_are_dr_means_so_a_single_target_region_is_its_own_headline(
    cohort, planted, mirror
):
    """#2023: a single targeted region's declared effect IS the headline. Both are the DR
    mean over that region's rows, as is the cohort ATE over every row."""
    cf, _x, work = mirror
    dr = cf.rlearner_model_final_._oob_preds
    region = work["region"].to_numpy()
    for r in REGIONS:
        assert planted.cate_by_region[r] == pytest.approx(np.nanmean(dr[region == r]), abs=1e-12)
    weighted = sum(planted.cate_by_region[r] * planted.n_by_region[r] for r in REGIONS)
    assert planted.ate == pytest.approx(weighted / planted.n, abs=1e-12)
    west = estimate_cohort_effect(cohort, PLANTED, target_regions=["west"])
    assert west.target_ate == pytest.approx(planted.cate_by_region["west"], abs=1e-12)


class _ForestWithNanStderr(_RealForest):
    def fit(self, *args, **kw):
        super().fit(*args, **kw)
        self.rlearner_model_final_.ate_stderr_ = np.full((1,), np.nan)
        return self


def test_a_non_finite_dr_stderr_is_an_estimation_failure_not_an_interval(monkeypatch, cohort):
    monkeypatch.setattr("econml.dml.CausalForestDML", _ForestWithNanStderr)
    with pytest.raises(EffectDataUnavailable) as caught:
        estimate_cohort_effect(cohort, PLANTED)
    assert caught.value.cause is EffectCause.ESTIMATION_FAILED
    assert caught.value.details["is_target_inference"] is False
