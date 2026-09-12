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
    # Inference on a region subset, set only when ``target_regions`` is requested (#2015):
    # the forest's average effect over the cohort rows in those regions and its interval.
    target_regions: list[str] = field(default_factory=list)
    target_ate: float | None = None
    target_ci_lower: float | None = None
    target_ci_upper: float | None = None
    target_n: int = 0

    def ci_width(self) -> float:
        return float(self.ate_ci_upper - self.ate_ci_lower)


def _usable_rows(
    cohort: pd.DataFrame,
    treatment_col: str,
    *,
    outcome_col: str,
    region_col: str,
    confounders: Sequence[str],
) -> pd.DataFrame:
    """The rows every estimate on this cohort uses: required columns present (refusing an
    under-adjusted estimate), numeric model inputs coerced, rows null in any of them dropped.
    Columns: ``t_raw``, ``y``, ``region`` and one per confounder."""
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

    # Coerce + drop rows null in any model input (fail-honest, no NaN-as-0 fabrication).
    work = pd.DataFrame(
        {
            "t_raw": pd.to_numeric(cohort[treatment_col], errors="coerce"),
            "y": pd.to_numeric(cohort[outcome_col], errors="coerce"),
            "region": cohort[region_col].astype(str),
        }
    )
    for c in confounders:
        work[c] = pd.to_numeric(cohort[c], errors="coerce")
    work = work.dropna().reset_index(drop=True)

    return work


def control_outcome_sd(
    cohort: pd.DataFrame,
    treatment_col: str,
    *,
    outcome_col: str = _OUTCOME_COL,
    region_col: str = _REGION_COL,
    confounders: Sequence[str] = DEFAULT_CONFOUNDERS,
    regions: Sequence[str] = (),
) -> tuple[float, int]:
    """Outcome SD in the comparison arm of :func:`estimate_cohort_effect`'s contrast (#2015).

    The comparison arm is the usable rows at or below the cohort-median treatment intensity
    (the estimator's split), restricted to ``regions`` when given. Returns ``(sd, n)`` with
    ``sd`` the sample standard deviation (ddof=1). Raises ``EffectDataUnavailable`` with
    fewer than two such rows.
    """
    work = _usable_rows(
        cohort,
        treatment_col,
        outcome_col=outcome_col,
        region_col=region_col,
        confounders=confounders,
    )
    control = work[work["t_raw"] <= float(work["t_raw"].median())]
    targets = [str(r) for r in regions]
    if targets:
        control = control[control["region"].isin(targets)]
    if len(control) < 2:
        scope = f"regions {targets}" if targets else "the cohort"
        raise EffectDataUnavailable(
            f"{scope} has {len(control)} usable comparison-arm rows for '{treatment_col}'; "
            "the outcome spread cannot be measured."
        )
    return float(control["y"].std(ddof=1)), int(len(control))


def estimate_cohort_effect(
    cohort: pd.DataFrame,
    treatment_col: str,
    *,
    outcome_col: str = _OUTCOME_COL,
    region_col: str = _REGION_COL,
    confounders: Sequence[str] = DEFAULT_CONFOUNDERS,
    alpha: float = 0.05,
    seed: int = 42,
    target_regions: Sequence[str] = (),
) -> CohortCausalEffect:
    """Estimate the ATE + per-region CATE of ``treatment_col`` on ``outcome_col``.

    Treatment is binarized at its median (the pre-registered contrast: high vs low
    intensity, mirroring the DGP). Region is the heterogeneity axis X; the present subset
    of ``confounders`` is the control set W. Returns honest DML inference intervals.

    ``target_regions`` (#2015) adds the same forest's average effect over the cohort rows
    in those regions, with ``ate_interval`` over those rows — the interval the cohort-wide
    ATE gets, on the subset. Each target region must be in the cohort with both treated and
    control rows; otherwise ``EffectDataUnavailable`` (a region the cohort does not cover
    would only get an extrapolated or fallback effect).
    """
    work = _usable_rows(
        cohort,
        treatment_col,
        outcome_col=outcome_col,
        region_col=region_col,
        confounders=confounders,
    )
    present_confounders = list(confounders)

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

    targets = list(dict.fromkeys(str(r) for r in target_regions))
    target_ate = target_lo = target_hi = None
    target_n = 0
    if targets:
        for region in targets:
            in_region = region_arr == region
            if not in_region.any() or len(np.unique(t[in_region])) < 2:
                raise EffectDataUnavailable(
                    f"target region {region!r} has no treated-vs-control contrast in the "
                    f"cohort for '{treatment_col}' (cohort regions: {cats}); its effect "
                    "cannot be estimated."
                )
        mask = np.isin(region_arr, targets)
        try:
            t_lo, t_hi = cf.ate_interval(x[mask], alpha=alpha)
        except Exception as e:  # econml failure -> honest no-data, never a fake interval
            raise EffectDataUnavailable(
                f"target-region inference failed for '{treatment_col}': {e}"
            ) from e
        target_ate, target_lo, target_hi = float(np.mean(eff[mask])), float(t_lo), float(t_hi)
        target_n = int(mask.sum())

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
        target_regions=targets,
        target_ate=target_ate,
        target_ci_lower=target_lo,
        target_ci_upper=target_hi,
        target_n=target_n,
    )


class CohortCausalEstimator:
    """Engine-seam adapter: turns a raw-cohort ``TrainingFrame`` into an
    :class:`EffectEstimate` via :func:`estimate_cohort_effect`.

    Drop-in replacement for ``TwinEffectEstimator`` on the cohort path. Unlike the
    uplift estimator (which fit on a synthetic injected-effect frame and recentred a
    training-evidence CI), this produces a REAL DML estimate on the cohort with an honest
    inference interval, and reports per-twin uplift as each twin's REGION CATE (genuine
    heterogeneity from the data, not a synthetic-forest artifact).

    ``target_regions`` (#2023) SCOPES the headline estimate: the returned ``ate`` and its
    interval are the forest's effect on the cohort rows in those regions — the same numbers
    the chat ``counterfactual_simulator`` reports for the same regions, from the same
    single fit — and the cohort-wide estimate rides along in ``cohort_*``. Without it the
    estimate is cohort-wide, as before. A targeted region the cohort cannot contrast raises
    ``EffectDataUnavailable`` rather than quietly answering with the cohort-wide effect.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        seed: int = 42,
        target_regions: Sequence[str] = (),
    ) -> None:
        self.alpha = alpha
        self.seed = seed
        self.target_regions = [str(r) for r in target_regions]

    def estimate(self, frame, twin_population: pd.DataFrame) -> EffectEstimate:
        eff = estimate_cohort_effect(
            frame.df,
            frame.treatment_var,
            outcome_col=frame.outcome_var,
            confounders=tuple(frame.confounders),
            alpha=self.alpha,
            seed=self.seed,
            target_regions=self.target_regions,
        )

        # The headline estimate is the one the request asked for: scoped to the targeted
        # regions when there are any (with the cohort-wide estimate kept alongside), else
        # cohort-wide. Every quantity derived downstream — the SE, the DEPLOY/REFINE/SKIP
        # policy, the experiment size, the persisted row — then describes one population.
        if eff.target_regions:
            assert eff.target_ate is not None  # set whenever target_regions is non-empty
            assert eff.target_ci_lower is not None and eff.target_ci_upper is not None
            ate, ci_lower, ci_upper = eff.target_ate, eff.target_ci_lower, eff.target_ci_upper
            n_train = eff.target_n
            cohort_ate: float | None = eff.ate
            cohort_ci_lower: float | None = eff.ate_ci_lower
            cohort_ci_upper: float | None = eff.ate_ci_upper
        else:
            ate, ci_lower, ci_upper = eff.ate, eff.ate_ci_lower, eff.ate_ci_upper
            n_train = eff.n
            cohort_ate = cohort_ci_lower = cohort_ci_upper = None

        # Per-twin uplift = the twin's region CATE (honest, data-driven heterogeneity);
        # twins in a region absent from the cohort fall back to the headline ATE.
        if twin_population is not None and "region" in getattr(twin_population, "columns", []):
            regions = twin_population["region"].astype(str)
            per_twin = np.array([eff.cate_by_region.get(r, ate) for r in regions], dtype=float)
        else:
            n = len(twin_population) if twin_population is not None else 0
            per_twin = np.full(max(n, 1), ate, dtype=float)

        return EffectEstimate(
            ate=ate,
            ate_ci_lower=ci_lower,
            ate_ci_upper=ci_upper,
            att=None,
            atc=None,
            per_twin_uplift=per_twin,
            auuc=None,
            qini=None,
            feature_importances={f"cate::{r}": v for r, v in eff.cate_by_region.items()},
            n_train=n_train,
            estimator_type="cohort_causal_forest_dml",
            data_provenance=PROVENANCE_COHORT,
            target_regions=list(eff.target_regions),
            cohort_ate=cohort_ate,
            cohort_ci_lower=cohort_ci_lower,
            cohort_ci_upper=cohort_ci_upper,
        )
