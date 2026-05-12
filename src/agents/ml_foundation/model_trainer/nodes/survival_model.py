"""v5 Gate B2 — Cox proportional hazards + Random Survival Forest.

Mirrors v5 B3 architecture (pure helper + LangGraph node split for
replay-safety per the H3 lesson). Pre-spec at
``docs/specs/v5_b2_survival_modeling_prespec_2026-05-12.md``.

Public surface:
- ``derive_survival_target(df, manifest_source)`` — pure helper that
  returns ``(time, event)`` numpy arrays for the cohort. Time is
  derived from first post-index ``treatment_events.days_from_diagnosis``
  on cohorts that carry that signal (CSU); falls back to an
  administrative censoring horizon for cohorts that do not (Optum).
- ``fit_cox(X, time, event, alpha, seed)`` — returns a fitted
  ``CoxPHSurvivalAnalysis``.
- ``fit_rsf(X, time, event, n_estimators, min_samples_leaf, seed)`` —
  returns a fitted ``RandomSurvivalForest``.
- ``survival_concordance(model, X, time, event)`` — Harrell c-index.
- ``survival_model_node(state)`` — LangGraph node wrapper, gated on
  ``state["enable_survival_modeling"]``; returns a state patch
  containing the survival target arrays. Replay-safe — does NOT mutate
  state in place.

Important constraint: this module does NOT couple to the binary
classifier pipeline. The Cox/RSF fits are independent measurements
called from ``scripts/measure_b2_cindex_contrast.py``; the LangGraph
node here only emits the derived ``(time, event)`` arrays into state
so downstream consumers can train survival models without re-deriving.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# === Survival target derivation =====================================

# Per pre-spec §4: administrative censoring horizon for cohorts
# without usable post-index rx event dates. Optum's 180d is the
# "initiation" cohort's defining follow-up window (column
# ``initiated_biologic_180d``).
_ADMIN_CENSORING_DAYS_DEFAULT = 180

# CSU follow-up cap: 95th percentile of journey_duration_days, but
# capped at 365 to keep the survival surface interpretable. Computed
# from the cohort itself within ``_derive_csu_survival_target``.
_CSU_FOLLOWUP_CAP_DAYS = 365


def _coerce_days_from_diagnosis(series: pd.Series) -> pd.Series:
    """Cast a days_from_diagnosis column to numeric, surfacing parse failures.

    Some upstream loaders preserve event metadata as object/string. The
    survival target depends on numeric ordering of days, so non-numeric
    values must coerce explicitly. Returns NaN for un-coercible rows;
    caller decides how to handle them.
    """
    return pd.to_numeric(series, errors="coerce")


def _derive_csu_survival_target(
    patient_journeys: pd.DataFrame,
    treatment_events: Optional[pd.DataFrame],
    followup_cap_days: int = _CSU_FOLLOWUP_CAP_DAYS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive ``(time, event)`` for CSU from rx events + patient journeys.

    Per pre-spec §4 (CSU): time = first POST-index rx
    ``days_from_diagnosis`` if event == 1; else administrative
    censoring at ``min(journey_duration_days, followup_cap_days)``.

    If ``treatment_events`` is None or empty, falls back to a constant
    administrative censoring (degenerate; documented in pre-spec §2).
    """
    event = patient_journeys["treatment_initiated"].astype(int).to_numpy()
    n = len(patient_journeys)
    time = np.full(n, fill_value=float(followup_cap_days), dtype=float)

    # Censoring time for non-events: min(journey_duration_days, cap).
    if "journey_duration_days" in patient_journeys.columns:
        jdd = pd.to_numeric(
            patient_journeys["journey_duration_days"], errors="coerce"
        ).fillna(followup_cap_days)
        time = np.minimum(jdd.to_numpy(dtype=float), float(followup_cap_days))

    # Event time for positives: first post-index rx days_from_diagnosis.
    if (
        treatment_events is not None
        and not treatment_events.empty
        and "event_type" in treatment_events.columns
        and "days_from_diagnosis" in treatment_events.columns
        and "patient_id" in treatment_events.columns
        and "patient_id" in patient_journeys.columns
    ):
        rx = treatment_events[treatment_events["event_type"] == "prescription"].copy()
        if not rx.empty:
            rx["days_from_diagnosis"] = _coerce_days_from_diagnosis(rx["days_from_diagnosis"])
            # Only post-index rx events count toward time-to-initiation.
            post_index_rx = rx[rx["days_from_diagnosis"] >= 0]
            if not post_index_rx.empty:
                first_rx_time = (
                    post_index_rx.sort_values("days_from_diagnosis")
                    .groupby("patient_id")["days_from_diagnosis"]
                    .first()
                )
                # Map first-rx time onto patient_journeys order.
                pid_to_time = patient_journeys["patient_id"].map(first_rx_time)
                event_mask = event == 1
                # Cap event times at followup_cap_days to bound the
                # survival surface (cf. clinical 1y follow-up convention).
                pid_time_capped = np.minimum(
                    pid_to_time.to_numpy(dtype=float, na_value=np.nan),
                    float(followup_cap_days),
                )
                # For positives WITH a matched rx time, use it;
                # positives without a matched rx fall through to
                # the journey_duration_days time set above (informative
                # censoring, but the binary event flag still fires).
                use_rx_time = event_mask & ~np.isnan(pid_time_capped)
                time[use_rx_time] = pid_time_capped[use_rx_time]

    # Sanity: time must be strictly positive for sksurv to accept.
    # Replace any zero/negative with the minimum positive observed
    # (1 day) — these are degenerate same-day events.
    time = np.where(time <= 0.0, 1.0, time)
    return time, event.astype(bool)


def _derive_admin_censored_target(
    patient_journeys: pd.DataFrame,
    admin_censoring_days: int = _ADMIN_CENSORING_DAYS_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-time admin censoring derivation (Optum fallback).

    Per pre-spec §4 (Optum): all patients censored at 180d. Survival
    framing is degenerate by construction here (collapses to binary
    with proportional-hazards loss).
    """
    event = patient_journeys["treatment_initiated"].astype(int).to_numpy()
    time = np.full(len(patient_journeys), float(admin_censoring_days), dtype=float)
    return time, event.astype(bool)


_DERIVERS = {
    "csu": _derive_csu_survival_target,
    "optum": _derive_admin_censored_target,
}


def derive_survival_target(
    patient_journeys: pd.DataFrame,
    manifest_source: str,
    treatment_events: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to cohort-specific survival target derivation.

    Returns ``(time_days, event_bool)`` with ``time_days.shape == (n,)``
    and ``event_bool.shape == (n,)``. Raises ``ValueError`` for
    unknown ``manifest_source``.
    """
    if manifest_source == "csu":
        return _derive_csu_survival_target(patient_journeys, treatment_events)
    if manifest_source == "optum":
        return _derive_admin_censored_target(patient_journeys)
    raise ValueError(
        f"derive_survival_target: unknown manifest_source={manifest_source!r}. "
        f"Supported: csu, optum."
    )


# === Survival model fitting =========================================


def _make_structured_target(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Wrap (time, event) into the sksurv structured-array convention.

    sksurv expects ``y`` as a structured ndarray with two fields whose
    names must contain "event" (bool) and "time" (float). We use the
    helper ``Surv.from_arrays`` for stability.
    """
    from sksurv.util import Surv

    return Surv.from_arrays(event=np.asarray(event, dtype=bool), time=np.asarray(time, dtype=float))


def fit_cox(
    X: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    alpha: float = 1e-3,
    seed: int = 42,
) -> Any:
    """Fit a Cox proportional-hazards model on (X, time, event).

    ``alpha`` is the ridge penalty (CoxPHSurvivalAnalysis API). Default
    1e-3 regularizes against the collinear CSU feature surface per B3
    finding.

    Returns the fitted model. ``seed`` is unused by CoxPHSurvivalAnalysis
    but threaded for caller-side reproducibility plumbing.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    del seed  # Cox is deterministic given inputs; no RNG.
    y = _make_structured_target(time, event)
    model = CoxPHSurvivalAnalysis(alpha=alpha)
    model.fit(X.values, y)
    return model


def fit_rsf(
    X: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    n_estimators: int = 100,
    min_samples_leaf: int = 15,
    seed: int = 42,
    n_jobs: int = -1,
) -> Any:
    """Fit a Random Survival Forest on (X, time, event)."""
    from sksurv.ensemble import RandomSurvivalForest

    y = _make_structured_target(time, event)
    model = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=seed,
    )
    model.fit(X.values, y)
    return model


def survival_concordance(
    model: Any,
    X: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """Compute Harrell concordance index on (X, time, event).

    Higher risk score = earlier event (sksurv convention). For Cox,
    ``predict`` returns the linear predictor (risk); for RSF, it
    returns the cumulative-hazard summary. Both are concordance-compatible.
    """
    from sksurv.metrics import concordance_index_censored

    risk = model.predict(X.values)
    c_index, _, _, _, _ = concordance_index_censored(
        event_indicator=np.asarray(event, dtype=bool),
        event_time=np.asarray(time, dtype=float),
        estimate=risk,
    )
    return float(c_index)


# === LangGraph node =================================================


async def survival_model_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node that derives the survival target for downstream use.

    Replay-safety contract (mirrored from B3 H3 lesson): this node
    returns a STATE PATCH (Dict[str, Any]) containing the new arrays.
    It does NOT mutate ``state`` in place. Channel reducers consume
    the patch on resume.

    Gated on ``state["enable_survival_modeling"]`` (default False). When
    disabled, returns an empty patch (no-op) — keeps the binary
    pipeline unchanged.

    Requires in state:
    - ``patient_journeys_df`` (pd.DataFrame) — the cohort frame.
    - ``treatment_events_df`` (Optional[pd.DataFrame]) — for CSU.
    - ``manifest_source`` (str, one of {csu, optum}).
    """
    if not state.get("enable_survival_modeling", False):
        return {}

    manifest_source = state.get("manifest_source")
    if manifest_source not in {"csu", "optum"}:
        logger.warning(
            "survival_model_node: unknown manifest_source=%r; skipping",
            manifest_source,
        )
        return {}

    pj = state.get("patient_journeys_df")
    if pj is None or len(pj) == 0:
        logger.warning("survival_model_node: empty patient_journeys_df; skipping")
        return {}

    ev = state.get("treatment_events_df")
    try:
        time, event = derive_survival_target(pj, manifest_source, treatment_events=ev)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "survival_model_node: derive_survival_target failed (manifest_source=%s): %s",
            manifest_source,
            exc,
        )
        return {"survival_target_error": str(exc)}

    patch: Dict[str, Any] = {
        "survival_time_days": time,
        "survival_event": event,
        "survival_manifest_source": manifest_source,
    }
    logger.info(
        "survival_model_node: derived target for %s (n=%d, n_events=%d, "
        "median_time=%.1f days)",
        manifest_source,
        len(time),
        int(np.sum(event)),
        float(np.median(time)),
    )
    return patch
