"""Direct causal estimation of an intervention effect on a brand's cohort.

Replaces the prior region-only g-formula + synthetic injected-effect handoff
(``CohortEffectDataProvider`` -> ``SyntheticEffectDataProvider(true_ate=...)``) with a
DML estimate computed DIRECTLY on the connected cohort over a defensible PRE-TREATMENT
adjustment set. This is the Direction-2 estimator (design doc 2026-06-19):

- magnitude, uncertainty, per-region and supported per-specialty heterogeneity all come
  from the cohort data;
- nothing is laundered through a synthetic frame, so the CI reflects REAL sampling noise;
- it is substrate-agnostic: identical code recovers the planted ``TRUE_CATE_BY_REGION`` on
  synthetic-gold today and runs unchanged on RWD tomorrow (the adjustment set is the
  present subset of the configured pre-treatment confounders, never hardcoded magnitudes).

Method extends the gold-standard recovery probe in ``scripts/backfill_segment_engagement.py``
(CausalForestDML, treatment binarized at the median) with nominal region and specialty
modifiers. The current synthetic-gold DGP plants effects by region only, so its specialty
output is a null-interaction/observed-composition check, not planted specialty truth.

Fail-closed (CLAUDE.md anti-mocking): degenerate/insufficient data raises
``EffectDataUnavailable`` — the caller surfaces an honest no-effect result, never a
fabricated ATE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence, cast

import numpy as np
import pandas as pd

from src.digital_twin.effect.errors import EffectCause, EffectDataUnavailable
from src.digital_twin.effect.estimate import (
    PROVENANCE_COHORT,
    AxisProvenance,
    EffectEstimate,
)

logger = logging.getLogger(__name__)

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
_SPECIALTY_COL = "specialty"
_MISSING_SPECIALTY = "__missing_specialty__"

# Publication support for specialty effects.  The 100-row floor matches the repository's
# existing minimum evaluation-support stability boundary; the 20-per-arm floor is twice
# the forest's min_samples_leaf=10.  These are reporting gates, not row filters: all cohort
# rows remain in the pooled fit, while an under-supported label is omitted from the API.
MIN_SPECIALTY_ROWS = 100
MIN_SPECIALTY_ARM_ROWS = 20


@dataclass
class CohortCausalEffect:
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    cate_by_region: dict[str, float]
    n: int
    # Usable cohort rows behind each region's CATE (#2054) — the evidence base for that
    # per-region effect, so a caller reporting it can report what it rests on. Same keys
    # as ``cate_by_region``.
    n_by_region: dict[str, int]
    cate_by_specialty: dict[str, float]
    n_by_specialty: dict[str, int]
    specialty_suppressed_groups: dict[str, str]
    specialty_source_available: bool
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
    specialty_col: str = _SPECIALTY_COL,
) -> pd.DataFrame:
    """The rows every estimate on this cohort uses: required columns present (refusing an
    under-adjusted estimate), numeric model inputs coerced, rows null in any of them dropped.
    Columns: ``t_raw``, ``y``, ``region`` and one per confounder."""
    n_rows = int(len(cohort))
    if treatment_col not in cohort.columns:
        raise EffectDataUnavailable(
            f"cohort missing treatment column '{treatment_col}'.",
            cause=EffectCause.REQUIRED_COLUMN_MISSING,
            details={"n_rows": n_rows, "has_treatment_column": False},
        )
    if outcome_col not in cohort.columns or region_col not in cohort.columns:
        raise EffectDataUnavailable(
            f"cohort missing required column(s): need '{outcome_col}' and '{region_col}'.",
            cause=EffectCause.REQUIRED_COLUMN_MISSING,
            details={
                "n_rows": n_rows,
                "has_treatment_column": True,
                "has_outcome_column": outcome_col in cohort.columns,
                "has_region_column": region_col in cohort.columns,
            },
        )

    # Require every REQUESTED confounder to be present — refuse to silently drop a known
    # confounder and emit an under-adjusted (confounded) estimate that LOOKS adjusted.
    # (An explicit empty `confounders` is allowed: it is the deliberate naive/unadjusted
    # contrast used for de-confounding validation.)
    missing = [c for c in confounders if c not in cohort.columns]
    if missing:
        raise EffectDataUnavailable(
            f"cohort missing required confounder column(s) {missing}; refusing to "
            "produce an under-adjusted estimate.",
            cause=EffectCause.REQUIRED_COLUMN_MISSING,
            details={"n_rows": n_rows, "n_missing_confounder_columns": len(missing)},
        )

    # Coerce + drop rows null in any model input (fail-honest, no NaN-as-0 fabrication).
    work = pd.DataFrame(
        {
            "t_raw": pd.to_numeric(cohort[treatment_col], errors="coerce"),
            "y": pd.to_numeric(cohort[outcome_col], errors="coerce"),
            "region": cohort[region_col].astype(str),
        }
    )
    if specialty_col in cohort.columns:
        specialty = cohort[specialty_col].astype("string").str.strip()
        work["specialty"] = specialty.mask(specialty.eq("")).fillna(_MISSING_SPECIALTY)
    else:
        # Backward-compatible cohorts still produce the region estimate.  Specialty stays
        # undeclared rather than fabricated when its source column is absent.
        work["specialty"] = _MISSING_SPECIALTY
    for c in confounders:
        work[c] = pd.to_numeric(cohort[c], errors="coerce")
    work = work.dropna().reset_index(drop=True)

    return work


def _effect_modifier_matrix(work: pd.DataFrame) -> np.ndarray:
    """One-hot both nominal axes, preserving the legacy region-only matrix when needed."""
    if "specialty" not in work or not work["specialty"].ne(_MISSING_SPECIALTY).any():
        categories = sorted(work["region"].unique())
        code = {category: index for index, category in enumerate(categories)}
        return cast(np.ndarray, work["region"].map(code).to_numpy(dtype=float).reshape(-1, 1))
    axes = ["region", "specialty"]
    return cast(
        np.ndarray,
        pd.get_dummies(work[axes], columns=axes, dtype=float).to_numpy(dtype=float),
    )


def _specialty_aggregates(
    specialty: np.ndarray,
    treatment: np.ndarray,
    effects: np.ndarray,
    scope_mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, int], dict[str, str]]:
    """Publish supported observed-region-mix specialty CATE means within ``scope_mask``."""
    cate: dict[str, float] = {}
    counts: dict[str, int] = {}
    suppressed: dict[str, str] = {}
    for label in sorted(set(specialty[scope_mask])):
        if label == _MISSING_SPECIALTY:
            suppressed["<missing>"] = "source_value_missing"
            continue
        mask = scope_mask & (specialty == label)
        n_group = int(mask.sum())
        n_treated = int(treatment[mask].sum())
        n_control = n_group - n_treated
        if n_group < MIN_SPECIALTY_ROWS:
            suppressed[label] = "group_rows_below_minimum"
        elif n_treated < MIN_SPECIALTY_ARM_ROWS:
            suppressed[label] = "treated_rows_below_minimum"
        elif n_control < MIN_SPECIALTY_ARM_ROWS:
            suppressed[label] = "control_rows_below_minimum"
        else:
            cate[label] = float(np.mean(effects[mask]))
            counts[label] = n_group
    return cate, counts, suppressed


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
            "the outcome spread cannot be measured.",
            cause=EffectCause.TOO_FEW_USABLE_ROWS,
        )
    return float(control["y"].std(ddof=1)), int(len(control))


def estimate_cohort_effect(
    cohort: pd.DataFrame,
    treatment_col: str,
    *,
    outcome_col: str = _OUTCOME_COL,
    region_col: str = _REGION_COL,
    specialty_col: str = _SPECIALTY_COL,
    confounders: Sequence[str] = DEFAULT_CONFOUNDERS,
    alpha: float = 0.05,
    seed: int = 42,
    target_regions: Sequence[str] = (),
) -> CohortCausalEffect:
    """Estimate the ATE + per-region CATE of ``treatment_col`` on ``outcome_col``.

    Treatment is binarized at its median (the pre-registered contrast: high vs low
    intensity, mirroring the DGP). Region and, when sourced, specialty are categorical
    heterogeneity axes X; ``confounders`` is the control set W. Returns honest DML
    inference intervals.

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
        specialty_col=specialty_col,
    )
    present_confounders = list(confounders)

    n_usable = int(len(work))
    if len(work) < _MIN_ROWS:
        raise EffectDataUnavailable(
            f"cohort has {len(work)} usable rows (< {_MIN_ROWS}) for '{treatment_col}'.",
            cause=EffectCause.TOO_FEW_USABLE_ROWS,
            details={"n_usable_rows": n_usable, "n_min_usable_rows": _MIN_ROWS},
        )

    # Pre-registered contrast: treated = above the cohort median intensity.
    t_thr = float(work["t_raw"].median())
    t = (work["t_raw"] > t_thr).astype(int).to_numpy()
    if len(np.unique(t)) < 2:
        # One distinct value is a constant channel; more than one is a skew onto the median.
        raise EffectDataUnavailable(
            f"treatment '{treatment_col}' has no median contrast (all rows on one side); "
            "cannot identify an effect.",
            cause=EffectCause.NO_TREATMENT_CONTRAST,
            details={
                "n_usable_rows": n_usable,
                "n_distinct_treatment_values": int(work["t_raw"].nunique()),
            },
        )

    y = work["y"].to_numpy(dtype=float)

    # X = one-hot region + specialty.  Both are nominal labels, so integer coding would
    # inject a false ordering into the forest's split geometry.
    cats = sorted(work["region"].unique())
    if len(cats) < 1:
        raise EffectDataUnavailable(
            "cohort has no region values.",
            cause=EffectCause.TOO_FEW_USABLE_ROWS,
            details={"n_usable_rows": n_usable},
        )
    specialty_source_available = bool(
        specialty_col in cohort.columns and work["specialty"].ne(_MISSING_SPECIALTY).any()
    )
    x = _effect_modifier_matrix(work)

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
        # Library text reaches the chat answer through the simulator; it goes to the log (#2020).
        logger.warning("cohort causal estimation failed for '%s'", treatment_col, exc_info=e)
        raise EffectDataUnavailable(
            f"cohort causal estimation failed for '{treatment_col}': the causal forest could "
            "not estimate an effect on this cohort.",
            cause=EffectCause.ESTIMATION_FAILED,
            details={"n_usable_rows": n_usable, "is_target_inference": False},
        ) from e

    region_arr = work["region"].to_numpy(dtype=str)
    cate_by_region = {
        c: float(np.mean(eff[region_arr == c])) for c in cats if (region_arr == c).any()
    }
    # Evidence base per region: the usable cohort rows the CATE above averages over.
    n_by_region = {c: int((region_arr == c).sum()) for c in cate_by_region}

    specialty_arr = work["specialty"].to_numpy(dtype=str)
    report_mask: np.ndarray = np.ones(len(work), dtype=bool)
    if targets := list(dict.fromkeys(str(r) for r in target_regions)):
        report_mask = np.isin(region_arr, targets)

    cate_by_specialty, n_by_specialty, specialty_suppressed_groups = _specialty_aggregates(
        specialty_arr, t, eff, report_mask
    )

    targets = list(dict.fromkeys(str(r) for r in target_regions))
    target_ate = target_lo = target_hi = None
    target_n = 0
    if targets:
        # Counted over every target before refusing; the message names the first that fails.
        absent = [r for r in targets if not (region_arr == r).any()]
        one_arm = [r for r in targets if r not in absent and len(np.unique(t[region_arr == r])) < 2]
        for region in targets:
            if region in absent or region in one_arm:
                raise EffectDataUnavailable(
                    f"target region {region!r} has no treated-vs-control contrast in the "
                    f"cohort for '{treatment_col}' (cohort regions: {cats}); its effect "
                    "cannot be estimated.",
                    cause=EffectCause.TARGET_REGION_NOT_COVERED,
                    details={
                        "n_target_regions": len(targets),
                        "n_target_regions_absent": len(absent),
                        "n_target_regions_one_arm": len(one_arm),
                        "n_cohort_regions": len(cats),
                    },
                )
        mask = np.isin(region_arr, targets)
        try:
            t_lo, t_hi = cf.ate_interval(x[mask], alpha=alpha)
        except Exception as e:  # econml failure -> honest no-data, never a fake interval
            logger.warning(
                "target-region inference failed for '%s' on %s", treatment_col, targets, exc_info=e
            )
            raise EffectDataUnavailable(
                f"target-region inference failed for '{treatment_col}': the causal forest could "
                "not compute an interval on the targeted rows.",
                cause=EffectCause.TARGET_INFERENCE_FAILED,
                details={
                    "n_usable_rows": n_usable,
                    "is_target_inference": True,
                    "n_target_rows": int(mask.sum()),
                },
            ) from e
        target_ate, target_lo, target_hi = float(np.mean(eff[mask])), float(t_lo), float(t_hi)
        target_n = int(mask.sum())

    adjustment_set = [region_col]
    if specialty_source_available:
        adjustment_set.append(specialty_col)
    adjustment_set.extend(present_confounders)
    return CohortCausalEffect(
        ate=float(np.mean(eff)),
        ate_ci_lower=float(lo),
        ate_ci_upper=float(hi),
        cate_by_region=cate_by_region,
        n_by_region=n_by_region,
        cate_by_specialty=cate_by_specialty,
        n_by_specialty=n_by_specialty,
        specialty_suppressed_groups=specialty_suppressed_groups,
        specialty_source_available=specialty_source_available,
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
        # cohort-wide. What this carries to: the ATE, its interval, the SE derived from it,
        # the DEPLOY/REFINE/SKIP policy and the experiment size.
        # The subgroup heterogeneity the engine reports is now on the same footing: it
        # comes from ``cate_by_axis`` below, which declares region and supported specialty
        # groups and carries the
        # cohort rows behind each region's effect, so by_specialty / by_decile /
        # by_adoption_stage are no longer averaged over the GENERATED TWINS (#2054), and
        # the simulation confidence follows ``n_train`` below, not the twin count (#2104).
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

        # This estimate resolves region and, when sourced and supported, specialty directly
        # over the cohort rows. The other axes remain undeclared: averaging this forest over
        # generated-twin deciles or adoption stages would recreate the twin-mixture artifact
        # fixed by #2054.
        # Scoped to the targeted regions when there are any: those are the rows this estimate
        # was computed on, and a region outside that scope was not estimated here. The
        # evidence is COHORT rows, so these numbers do not move with the twin count.
        declared_regions = (
            [r for r in eff.target_regions if r in eff.cate_by_region]
            if eff.target_regions
            else list(eff.cate_by_region)
        )
        # With nothing to declare the axis key is OMITTED, never mapped to an empty dict: an
        # empty mapping is the signal for "resolved by the per-twin scores", which for this
        # estimator would put the region step function back through the twin grouping and
        # resurrect the cohort-ATE fallback under an uncovered region's label. Absent means
        # unresolved, and the engine then reports {} — the fail-closed answer.
        cate_by_axis: dict[str, dict[str, float]] = {}
        n_by_axis: dict[str, dict[str, int]] = {}
        if declared_regions:
            cate_by_axis["region"] = {r: eff.cate_by_region[r] for r in declared_regions}
            n_by_axis["region"] = {r: eff.n_by_region[r] for r in declared_regions}
        if eff.cate_by_specialty:
            cate_by_axis["specialty"] = dict(eff.cate_by_specialty)
            n_by_axis["specialty"] = dict(eff.n_by_specialty)
        # Targeting that matches no cohort region is fail-closed above rather than wrong, but
        # it is still a bug: ``estimate_cohort_effect`` rejects an uncovered target region, so
        # every target reaching here is a region the forest produced a CATE for.
        assert declared_regions or not eff.cate_by_region, (
            f"target_regions {eff.target_regions} matched none of the cohort's regions "
            f"{sorted(eff.cate_by_region)}."
        )

        # Per-twin scoring uses the same published specialty CATE when supported. An absent
        # or under-supported specialty falls back to region, then the headline ATE. Fallback
        # scores are never published as specialty CATEs above.
        if twin_population is not None and "region" in getattr(twin_population, "columns", []):
            regions = twin_population["region"].astype(str)
            specialties = (
                twin_population["specialty"].astype("string").fillna(_MISSING_SPECIALTY)
                if "specialty" in twin_population.columns
                else pd.Series([_MISSING_SPECIALTY] * len(twin_population))
            )
            per_twin = np.array(
                [
                    eff.cate_by_specialty.get(str(s), eff.cate_by_region.get(r, ate))
                    for r, s in zip(regions, specialties, strict=True)
                ],
                dtype=float,
            )
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
            feature_importances={
                **{f"cate::region::{r}": v for r, v in eff.cate_by_region.items()},
                **{f"cate::specialty::{s}": v for s, v in eff.cate_by_specialty.items()},
            },
            n_train=n_train,
            estimator_type="cohort_causal_forest_dml",
            data_provenance=PROVENANCE_COHORT,
            target_regions=list(eff.target_regions),
            cohort_ate=cohort_ate,
            cohort_ci_lower=cohort_ci_lower,
            cohort_ci_upper=cohort_ci_upper,
            cate_by_axis=cate_by_axis,
            n_by_axis=n_by_axis,
            axis_provenance={
                "specialty": AxisProvenance(
                    basis="cohort_rows",
                    source="hcp_profiles.specialty",
                    min_group_rows=MIN_SPECIALTY_ROWS,
                    min_treated_rows=MIN_SPECIALTY_ARM_ROWS,
                    min_control_rows=MIN_SPECIALTY_ARM_ROWS,
                    fallback="region_then_cohort",
                    support_unit="cohort_rows",
                    estimand="observed_region_mix_mean_cate",
                    suppressed_groups=dict(eff.specialty_suppressed_groups),
                )
            }
            if eff.specialty_source_available
            else {},
        )
