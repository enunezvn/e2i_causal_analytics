"""Direct causal estimation of an intervention effect on a brand's cohort.

Replaces the prior region-only g-formula + synthetic injected-effect handoff
(``CohortEffectDataProvider`` -> ``SyntheticEffectDataProvider(true_ate=...)``) with a
DML estimate computed DIRECTLY on the connected cohort over a defensible PRE-TREATMENT
adjustment set. This is the Direction-2 estimator (design doc 2026-06-19):

- magnitude, uncertainty AND per-region heterogeneity all come from the data;
- nothing is laundered through a synthetic frame, so the CI reflects REAL sampling noise;
- it is substrate-agnostic: identical code recovers the planted ``TRUE_CATE_BY_REGION`` on
  synthetic-gold today and runs unchanged on RWD tomorrow (the adjustment set is the
  present subset of the configured pre-treatment confounders, never hardcoded magnitudes).

Method mirrors the gold-standard recovery probe in
``scripts/backfill_segment_engagement.py`` (CausalForestDML, region as the heterogeneity
axis X, the confounders as controls W, treatment binarized at the median), so this
estimator IS the agent-faithful recovery of the documented DGP.

Fail-closed (CLAUDE.md anti-mocking): degenerate/insufficient data raises
``EffectDataUnavailable`` — the caller surfaces an honest no-effect result, never a
fabricated ATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.digital_twin.effect.errors import EffectDataUnavailable
from src.digital_twin.effect.estimate import PROVENANCE_COHORT, EffectEstimate

# Pre-treatment confounders to adjust for when present in the connected cohort.
# These are CONFOUNDERS (drivers of both the engagement treatment and the conversion
# outcome) in the gold-standard DGP, NOT outcomes/mediators (nrx/trx/conversion are
# excluded to avoid collider/over-control bias). On RWD the present subset is used;
# absent columns are skipped (logged by the caller), never invented.
DEFAULT_CONFOUNDERS: tuple[str, ...] = ("market_share", "total_rx_count")
# total_rx_count is heavy-tailed -> adjust on log1p scale (matches the DGP probe).
_LOG_CONFOUNDERS = frozenset({"total_rx_count"})

_MIN_ROWS = 200  # DML needs a stable nuisance fit; the loader gates cohorts at >= 500.
_OUTCOME_COL = "conversion_rate"
_REGION_COL = "region"


@dataclass
class CohortCausalEffect:
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    cate_by_region: dict[str, float]
    n: int
    treatment_col: str
    outcome_col: str
    adjustment_set: list[str] = field(default_factory=list)
    estimator_type: str = "causal_forest_dml"

    def ci_width(self) -> float:
        return float(self.ate_ci_upper - self.ate_ci_lower)


def estimate_cohort_effect(
    cohort: pd.DataFrame,
    treatment_col: str,
    *,
    outcome_col: str = _OUTCOME_COL,
    region_col: str = _REGION_COL,
    confounders: Sequence[str] = DEFAULT_CONFOUNDERS,
    alpha: float = 0.05,
    seed: int = 42,
) -> CohortCausalEffect:
    """Estimate the ATE + per-region CATE of ``treatment_col`` on ``outcome_col``.

    Treatment is binarized at its median (the pre-registered contrast: high vs low
    intensity, mirroring the DGP). Region is the heterogeneity axis X; the present subset
    of ``confounders`` is the control set W. Returns honest DML inference intervals.
    """
    if treatment_col not in cohort.columns:
        raise EffectDataUnavailable(f"cohort missing treatment column '{treatment_col}'.")
    if outcome_col not in cohort.columns or region_col not in cohort.columns:
        raise EffectDataUnavailable(
            f"cohort missing required column(s): need '{outcome_col}' and '{region_col}'."
        )

    # Require every REQUESTED confounder to be present — refuse to silently drop a known
    # confounder and emit an under-adjusted (confounded) estimate that LOOKS adjusted.
    # (An explicit empty `confounders` is allowed: it is the deliberate naive/unadjusted
    # contrast used for de-confounding validation.)
    missing = [c for c in confounders if c not in cohort.columns]
    if missing:
        raise EffectDataUnavailable(
            f"cohort missing required confounder column(s) {missing}; refusing to "
            "produce an under-adjusted estimate."
        )
    present_confounders = list(confounders)

    # Coerce + drop rows null in any model input (fail-honest, no NaN-as-0 fabrication).
    work = pd.DataFrame(
        {
            "t_raw": pd.to_numeric(cohort[treatment_col], errors="coerce"),
            "y": pd.to_numeric(cohort[outcome_col], errors="coerce"),
            "region": cohort[region_col].astype(str),
        }
    )
    for c in present_confounders:
        work[c] = pd.to_numeric(cohort[c], errors="coerce")
    work = work.dropna().reset_index(drop=True)

    if len(work) < _MIN_ROWS:
        raise EffectDataUnavailable(
            f"cohort has {len(work)} usable rows (< {_MIN_ROWS}) for '{treatment_col}'."
        )

    # Pre-registered contrast: treated = above the cohort median intensity.
    t_thr = float(work["t_raw"].median())
    t = (work["t_raw"] > t_thr).astype(int).to_numpy()
    if len(np.unique(t)) < 2:
        raise EffectDataUnavailable(
            f"treatment '{treatment_col}' has no median contrast (all rows on one side); "
            "cannot identify an effect."
        )

    y = work["y"].to_numpy(dtype=float)

    # X = region (integer-coded heterogeneity axis); W = pre-treatment confounder controls.
    cats = sorted(work["region"].unique())
    if len(cats) < 1:
        raise EffectDataUnavailable("cohort has no region values.")
    code = {c: i for i, c in enumerate(cats)}
    x = work["region"].map(code).to_numpy(dtype=float).reshape(-1, 1)

    w = None
    if present_confounders:
        cols = []
        for c in present_confounders:
            v = work[c].to_numpy(dtype=float)
            cols.append(np.log1p(np.clip(v, 0.0, None)) if c in _LOG_CONFOUNDERS else v)
        w = np.column_stack(cols)

    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cf = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=seed),
            model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=seed),
            discrete_treatment=True,
            n_estimators=200,
            subforest_size=4,
            min_samples_leaf=10,
            random_state=seed,
        )
        cf.fit(y, t, X=x, W=w)
        eff = np.asarray(cf.effect(x), dtype=float).ravel()
        lo, hi = cf.ate_interval(x, alpha=alpha)
    except EffectDataUnavailable:
        raise
    except Exception as e:  # econml/sklearn failure -> honest no-data, never a fake ATE
        raise EffectDataUnavailable(
            f"cohort causal estimation failed for '{treatment_col}': {e}"
        ) from e

    region_arr = work["region"].to_numpy(dtype=str)
    cate_by_region = {
        c: float(np.mean(eff[region_arr == c])) for c in cats if (region_arr == c).any()
    }

    adjustment_set = [region_col] + present_confounders
    return CohortCausalEffect(
        ate=float(np.mean(eff)),
        ate_ci_lower=float(lo),
        ate_ci_upper=float(hi),
        cate_by_region=cate_by_region,
        n=int(len(work)),
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        adjustment_set=adjustment_set,
    )


class CohortCausalEstimator:
    """Engine-seam adapter: turns a raw-cohort ``TrainingFrame`` into an
    :class:`EffectEstimate` via :func:`estimate_cohort_effect`.

    Drop-in replacement for ``TwinEffectEstimator`` on the cohort path. Unlike the
    uplift estimator (which fit on a synthetic injected-effect frame and recentred a
    training-evidence CI), this produces a REAL DML estimate on the cohort with an honest
    inference interval, and reports per-twin uplift as each twin's REGION CATE (genuine
    heterogeneity from the data, not a synthetic-forest artifact).
    """

    def __init__(self, *, alpha: float = 0.05, seed: int = 42) -> None:
        self.alpha = alpha
        self.seed = seed

    def estimate(self, frame, twin_population: pd.DataFrame) -> EffectEstimate:
        eff = estimate_cohort_effect(
            frame.df,
            frame.treatment_var,
            outcome_col=frame.outcome_var,
            confounders=tuple(frame.confounders),
            alpha=self.alpha,
            seed=self.seed,
        )

        # Per-twin uplift = the twin's region CATE (honest, data-driven heterogeneity);
        # twins in a region absent from the cohort fall back to the population ATE.
        if twin_population is not None and "region" in getattr(twin_population, "columns", []):
            regions = twin_population["region"].astype(str)
            per_twin = np.array([eff.cate_by_region.get(r, eff.ate) for r in regions], dtype=float)
        else:
            n = len(twin_population) if twin_population is not None else 0
            per_twin = np.full(max(n, 1), eff.ate, dtype=float)

        return EffectEstimate(
            ate=eff.ate,
            ate_ci_lower=eff.ate_ci_lower,
            ate_ci_upper=eff.ate_ci_upper,
            att=None,
            atc=None,
            per_twin_uplift=per_twin,
            auuc=None,
            qini=None,
            feature_importances={f"cate::{r}": v for r, v in eff.cate_by_region.items()},
            n_train=eff.n,
            estimator_type="cohort_causal_forest_dml",
            data_provenance=PROVENANCE_COHORT,
        )
