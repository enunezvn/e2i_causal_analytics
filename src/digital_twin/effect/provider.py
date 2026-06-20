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

# Canonical intervention taxonomy — the single source of truth for the
# digital-twin intervention vocabulary (value + human label). The
# /digital-twin/intervention-types endpoint and the frontend dropdown both
# source from this, so FE and backend can never drift. (The prior FE enum —
# hcp_engagement, rep_training, ... — was @deprecated and DISJOINT from this
# set, so every UI simulation 422'd "unsupported intervention".)
#
# v1 effect basis is "synthetic": the SyntheticEffectDataProvider DGP is
# intervention-agnostic (effect == true_ate for every supported type), so the
# type is an allowlist gate, not an effect differentiator. A future
# CohortEffectDataProvider (Phase 2) sources a real per-brand CATE for the
# mappable types and flips their basis to "modeled".
INTERVENTION_CATALOG: tuple[tuple[str, str], ...] = (
    ("email_campaign", "Email Campaign"),
    ("call_frequency_increase", "Increased Call Frequency"),
    ("speaker_program_invitation", "Speaker Program Invitation"),
    ("sample_distribution", "Sample Distribution"),
    ("peer_influence_activation", "Peer Influence Activation"),
    ("digital_engagement", "Digital Engagement"),
)

SUPPORTED_INTERVENTIONS = {value for value, _label in INTERVENTION_CATALOG}

# Interventions whose effect is IDENTIFIED in the gold-standard cohort DGP
# (business_metrics/per_hcp_rollup) and therefore estimable by direct causal
# estimation. Mapped to the cohort treatment column.
#
# Only `digital_engagement` (-> engagement_score) is identified: the DGP plants a
# real region-heterogeneous causal effect of engagement on conversion
# (backfill_segment_engagement.py: conversion = baseline + tau[region]*T_engagement).
# `call_frequency` is an engagement-linked exposure CORRELATE, explicitly NOT in the
# causal path, so `call_frequency_increase` is NOT identified and is surfaced as an
# honest "unavailable" effect basis rather than a confounded number. Interventions with
# no cohort treatment column (email/speaker/samples/peer) are likewise unavailable.
INTERVENTION_TREATMENT_MAP: dict[str, str] = {
    "digital_engagement": "engagement_score",
}
COHORT_ESTIMABLE_INTERVENTIONS = frozenset(INTERVENTION_TREATMENT_MAP)

# Pre-treatment confounder controls for the direct cohort estimate (present subset used).
COHORT_CONFOUNDERS: tuple[str, ...] = ("market_share", "total_rx_count")

_COHORT_OUTCOME = "conversion_rate"
_COHORT_REGION = "region"
# Minimum usable cohort rows for a stable region-standardized estimate.
COHORT_MIN_ROWS = 500

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
        self,
        intervention_type: str,
        brand: str,
        twin_type: str,
        reference_covariates: pd.DataFrame | None = None,
    ) -> TrainingFrame:
        if intervention_type not in SUPPORTED_INTERVENTIONS:
            raise EffectDataUnavailable(
                f"SyntheticEffectDataProvider: unsupported intervention '{intervention_type}'."
            )
        if reference_covariates is not None:
            return self._frame_from_reference(reference_covariates)
        # brand/twin_type unused; intervention_type is only an allowlist gate — the v1 synthetic DGP is NOT intervention-differentiated (effect magnitude = true_ate for every supported type).
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

    def _frame_from_reference(self, reference_covariates: pd.DataFrame) -> TrainingFrame:
        """Generate a synthetic labeled frame whose covariates are RESAMPLED from a
        real twin population (so a synthetic-trained model scores those twins
        in-distribution), with the known ``true_ate`` injected on a binary outcome.
        """
        numeric = reference_covariates.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0 or len(numeric) == 0:
            raise EffectDataUnavailable(
                "SyntheticEffectDataProvider: reference_covariates has no numeric columns/rows."
            )
        rng = np.random.default_rng(self.seed)
        n = self.n
        idx = rng.integers(0, len(numeric), size=n)
        x = numeric.iloc[idx].reset_index(drop=True)
        std = x.std(ddof=0).replace(0, 1.0)
        x_z = (x - x.mean()) / std
        treatment = rng.integers(0, 2, size=n)
        # Mild covariate dependence keeps p0 within an unclipped band so the
        # marginal treated-vs-control gap equals true_ate.
        lin = 0.05 * x_z.mean(axis=1).to_numpy()
        p0 = np.clip(0.35 + lin, 0.15, 0.55)
        p = np.clip(p0 + self.true_ate * treatment, 0.01, 0.99)
        outcome = (rng.random(n) < p).astype(int)
        df = x.copy()
        df["treatment"] = treatment
        df["outcome"] = outcome
        return TrainingFrame(
            df=df,
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=list(numeric.columns),
            effect_modifiers=[],
            ground_truth_ate=self.true_ate,
        )


def region_standardized_ate(
    cohort: pd.DataFrame,
    treatment_col: str,
    *,
    outcome_col: str = _COHORT_OUTCOME,
    region_col: str = _COHORT_REGION,
) -> float:
    """Region-standardized binary treatment effect from a labeled cohort.

    Binarizes the (continuous) treatment + outcome at their medians (high vs
    low), then standardizes the treated-minus-control adoption gap over regions
    (g-formula with region as the adjustment set), so the estimate is NOT
    confounded by region (e.g. the Northeast has both higher engagement AND
    higher conversion). Returns the population-weighted ATE in adoption-rate
    units, clipped to a sane band. Raises ``EffectDataUnavailable`` if the cohort
    is too small or has no region with both treated and control rows.
    """
    if treatment_col not in cohort.columns:
        raise EffectDataUnavailable(
            f"CohortEffectDataProvider: cohort missing treatment column '{treatment_col}'."
        )
    t_raw = pd.to_numeric(cohort[treatment_col], errors="coerce")
    y_raw = pd.to_numeric(cohort[outcome_col], errors="coerce")
    work = pd.DataFrame({"t_raw": t_raw, "y_raw": y_raw, "region": cohort[region_col]}).dropna()
    if len(work) < COHORT_MIN_ROWS:
        raise EffectDataUnavailable(
            f"CohortEffectDataProvider: only {len(work)} usable cohort rows "
            f"(< {COHORT_MIN_ROWS}) for treatment '{treatment_col}'."
        )
    t_thr = float(work["t_raw"].median())
    y_thr = float(work["y_raw"].median())
    work["t"] = (work["t_raw"] > t_thr).astype(int)
    work["y"] = (work["y_raw"] > y_thr).astype(int)

    weighted_sum = 0.0
    weight_total = 0
    for _region, grp in work.groupby("region"):
        treated = grp.loc[grp["t"] == 1, "y"]
        control = grp.loc[grp["t"] == 0, "y"]
        if len(treated) == 0 or len(control) == 0:
            continue  # region offers no within-stratum contrast
        weighted_sum += len(grp) * float(treated.mean() - control.mean())
        weight_total += len(grp)
    if weight_total == 0:
        raise EffectDataUnavailable(
            "CohortEffectDataProvider: no region has both treated and control rows."
        )
    ate = weighted_sum / weight_total
    return float(np.clip(ate, -0.6, 0.6))


class CohortEffectDataProvider:
    """Effect provider that exposes a brand's synthetic-gold cohort as a RAW labeled
    frame for DIRECT causal estimation (Direction 2, design doc 2026-06-19).

    Returns the cohort's (treatment, outcome, region, pre-treatment confounders) frame
    UNCHANGED — paired with :class:`CohortCausalEstimator`, which runs a DML estimate over
    it (region as the heterogeneity axis, ``COHORT_CONFOUNDERS`` as controls).

    This REPLACES the prior v1 behavior, which computed a region-only standardized ATE
    and laundered it through ``SyntheticEffectDataProvider(true_ate=...)`` — a synthetic
    injected-effect frame whose CI and per-twin heterogeneity were synthetic artifacts.
    Now magnitude, CI, and per-region heterogeneity all come from the cohort. The
    substrate is synthetic-gold (NOT real-world) — the UI keeps the SYNTHETIC badge.
    """

    def __init__(self, cohort_df: pd.DataFrame, *, seed: int = 42) -> None:
        self._cohort = cohort_df
        self._seed = seed

    def get_training_frame(
        self,
        intervention_type: str,
        brand: str,
        twin_type: str,
        reference_covariates: pd.DataFrame | None = None,
    ) -> TrainingFrame:
        treatment_col = INTERVENTION_TREATMENT_MAP.get(intervention_type)
        if treatment_col is None:
            raise EffectDataUnavailable(
                f"CohortEffectDataProvider: intervention '{intervention_type}' is not "
                "identified in the cohort DGP (not cohort-estimable)."
            )
        if treatment_col not in self._cohort.columns:
            raise EffectDataUnavailable(
                f"CohortEffectDataProvider: cohort missing treatment column '{treatment_col}'."
            )
        # RAW cohort frame for direct causal estimation — NO synthetic injected-effect
        # handoff. region is the heterogeneity axis (effect_modifier); the present subset
        # of COHORT_CONFOUNDERS is the pre-treatment control set; ground_truth_ate is None
        # because the effect is ESTIMATED from the data, not injected.
        # Pass the FULL required confounder set; the estimator fails closed if the cohort
        # is missing any (refusing an under-adjusted estimate) rather than silently dropping.
        confounders = list(COHORT_CONFOUNDERS)
        return TrainingFrame(
            df=self._cohort,
            treatment_var=treatment_col,
            outcome_var=_COHORT_OUTCOME,
            confounders=confounders,
            effect_modifiers=[_COHORT_REGION],
            ground_truth_ate=None,
        )
