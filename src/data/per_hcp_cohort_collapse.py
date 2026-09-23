"""Collapse ``business_metrics`` ``per_hcp_rollup`` rows to ONE row per (hcp_id, brand), and
turn that exposure into the adoption DGP's planted channel term.

Why one row per (hcp, brand): the planted rollups carry 1-6 ``metric_date`` rows per pair
(13,797 rows over 10,136 pairs on 2026-09-23). ``hcp_brand_adoption`` has exactly one row per
pair, so the channel effect planted on ``adopted`` must be a function of ONE exposure value per
pair. Per-row values correlate only 0.42-0.61 (sums) / 0.41-0.69 (means, within multi-row pairs)
with the collapsed value, and estimating on the per-row grain attenuates the recovered effects
by 30-60 % (explore_adoption_dgp.md section 4). Counts are SUMMED over a pair's rows, scores and
shares are AVERAGED, labels (region, specialty -- constant per pair on the live table) take the
FIRST value.

Two consumers, one collapse: lane T1's re-plant (``scripts/backfill_hcp_treatment_arm.py``)
builds the shift from it, and lane T2's twin loader (``src/digital_twin/effect/cohort_loader.py``)
will collapse its cohort frame with the SAME rules so the estimator's median contrast is the
contrast the effect was planted on. Keep this module dependency-light (pandas/numpy only, no
sklearn/dowhy/twin imports): it is imported by scripts that must not pay the twin's import cost.

The exposure bit and the shift:

    tbin_k  = 1{collapsed exposure_k > within-brand median over the joined HCPs}
    shift   = sum_k beta_k * (tbin_k - 0.5)              (beta from ADOPTION_CHANNEL_LOGIT_BETA)

``value > median`` is exactly ``estimate_cohort_effect``'s treatment binarisation, so the planted
contrast is the estimated contrast. Centring at 0.5 keeps the joined base rate where it was (up
to the tie mass of the integer count channels, which makes P(tbin=1) < 0.5). HCPs with no rollup
row for a brand are NOT joined: their shift is 0 and their labels are untouched.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

import numpy as np
import pandas as pd

from src.data.per_hcp_cohort_columns import (
    ADOPTION_CHANNEL_LOGIT_BETA,
    INTERVENTION_TREATMENT_MAP,
)

#: The eight planted channel columns, sorted (the twin loader's ``_TREATMENT_COLUMNS`` order).
CHANNEL_COLUMNS: Final[tuple[str, ...]] = tuple(sorted(set(INTERVENTION_TREATMENT_MAP.values())))

#: Count-like columns: a pair's exposure is the SUM over its metric_date rows.
COLLAPSE_SUM: Final[tuple[str, ...]] = (
    "call_frequency",
    "email_campaign_count",
    "speaker_program_count",
    "sample_volume",
    "triggers_total_count",
)
#: Score / share columns: a pair's exposure is the MEAN over its rows.
COLLAPSE_MEAN: Final[tuple[str, ...]] = (
    "engagement_score",
    "peer_influence_score",
    "rep_training_score",
    "patient_support_enrollment",
    "market_share",
)
#: Labels, constant per pair on the live table: FIRST value.
COLLAPSE_FIRST: Final[tuple[str, ...]] = ("region", "specialty")

COLLAPSE_RULES: Final[dict[str, str]] = {
    **dict.fromkeys(COLLAPSE_SUM, "sum"),
    **dict.fromkeys(COLLAPSE_MEAN, "mean"),
    **dict.fromkeys(COLLAPSE_FIRST, "first"),
}

_KEY: Final = ["hcp_id", "brand"]


def tbin_column(channel: str) -> str:
    """Name of the exposure-bit column for ``channel`` in a tbin frame."""
    return f"tbin_{channel}"


def beta_by_column(
    betas: Mapping[str, float] = ADOPTION_CHANNEL_LOGIT_BETA,
) -> dict[str, float]:
    """``{planted column: beta}`` from the intervention-keyed table."""
    return {INTERVENTION_TREATMENT_MAP[k]: float(v) for k, v in betas.items()}


def collapse_per_hcp_brand(rollups: pd.DataFrame) -> pd.DataFrame:
    """One row per (hcp_id, brand): SUM counts, MEAN scores/shares, FIRST labels, plus
    ``max_metric_date`` (when ``metric_date`` is present) and ``n_metric_rows``.

    Columns absent from ``rollups`` are skipped (a frame without ``specialty`` collapses
    without it); the numeric columns are coerced before aggregation.
    """
    if rollups.empty:
        return pd.DataFrame(columns=[*_KEY, "n_metric_rows"])
    df = rollups.copy()
    for col in (*COLLAPSE_SUM, *COLLAPSE_MEAN):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    g = df.groupby(_KEY, sort=True)
    parts: list[pd.Series] = []
    for col in COLLAPSE_SUM:
        if col in df.columns:
            parts.append(g[col].sum(min_count=1).rename(col))
    for col in COLLAPSE_MEAN:
        if col in df.columns:
            parts.append(g[col].mean().rename(col))
    for col in COLLAPSE_FIRST:
        if col in df.columns:
            parts.append(g[col].first().rename(col))
    if "metric_date" in df.columns:
        parts.append(
            pd.to_datetime(df["metric_date"])
            .groupby([df[k] for k in _KEY])
            .max()
            .rename("max_metric_date")
        )
    parts.append(g.size().rename("n_metric_rows"))
    return pd.concat(parts, axis=1).reset_index()


def channel_tbin(
    collapsed: pd.DataFrame, channels: Iterable[str] = CHANNEL_COLUMNS
) -> pd.DataFrame:
    """Exposure bits: ``tbin_<channel> = 1{value > within-brand median}`` over the rows of
    ``collapsed`` (one per joined (hcp, brand)); NaN where the value is NaN.

    Returns ``hcp_id, brand, tbin_<channel>...`` (float 0/1). The median is per brand: each
    brand's joined HCPs are their own population, as they are for the estimator.
    """
    channels = list(channels)
    out = collapsed[_KEY].copy()
    for col in channels:
        vals = pd.to_numeric(collapsed[col], errors="coerce")
        med = vals.groupby(collapsed["brand"]).transform("median")
        bit = (vals > med).astype(float)
        bit[vals.isna()] = np.nan
        out[tbin_column(col)] = bit.to_numpy()
    return out


def adoption_channel_shift(
    tbin: pd.DataFrame, betas: Mapping[str, float] | None = None
) -> pd.Series:
    """``sum_k beta_k * (tbin_k - 0.5)`` per row of a :func:`channel_tbin` frame; a NaN bit
    contributes 0. Index is ``tbin``'s index."""
    beta = beta_by_column() if betas is None else dict(betas)
    shift = pd.Series(0.0, index=tbin.index)
    for col, b in beta.items():
        bit = tbin[tbin_column(col)].astype(float)
        shift = shift + b * (bit - 0.5).fillna(0.0)
    return shift.rename("channel_shift")


def align_channel_shift(hcp_ids: Iterable[str], shift_by_hcp: pd.Series) -> np.ndarray:
    """The DGP's per-HCP shift vector for ``hcp_ids`` (the generator's order): the joined
    HCPs' shift, 0.0 for every HCP absent from ``shift_by_hcp`` (indexed by hcp_id)."""
    ids = list(hcp_ids)
    aligned = shift_by_hcp.reindex(ids).to_numpy(dtype=float)
    return np.where(np.isnan(aligned), 0.0, aligned)


def dgp_true_channel_rd(
    adoption_logit: np.ndarray, tbin: pd.DataFrame, betas: Mapping[str, float] | None = None
) -> dict[str, float]:
    """The DGP-true realised risk difference per channel: with every other term of the
    realised logit held at its drawn value, ``mean(sigmoid(logit_k=1) - sigmoid(logit_k=0))``
    over the rows of ``tbin`` (aligned to ``adoption_logit``). The null channel is exactly 0."""
    beta = beta_by_column() if betas is None else dict(betas)
    logit = np.asarray(adoption_logit, dtype=float)
    out: dict[str, float] = {}
    for col, b in beta.items():
        bit = tbin[tbin_column(col)].to_numpy(dtype=float)
        ok = ~np.isnan(bit)
        base = logit[ok] - b * (bit[ok] - 0.5)  # remove channel k's contribution
        out[col] = (
            float(np.mean(_sigmoid(base + 0.5 * b) - _sigmoid(base - 0.5 * b)))
            if ok.any()
            else float("nan")
        )
    return out


def stratified_channel_rd(adopted: np.ndarray, tbin: pd.DataFrame) -> dict[str, float]:
    """The observed ``mean(adopted | tbin_k=1) - mean(adopted | tbin_k=0)`` per channel."""
    y = np.asarray(adopted, dtype=float)
    out: dict[str, float] = {}
    for col in CHANNEL_COLUMNS:
        bit = tbin[tbin_column(col)].to_numpy(dtype=float)
        hi, lo = y[bit == 1.0], y[bit == 0.0]
        out[col] = float(hi.mean() - lo.mean()) if len(hi) and len(lo) else float("nan")
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-z)))
