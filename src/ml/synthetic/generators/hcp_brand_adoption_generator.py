"""Per-brand, temporal HCP adoption cohort generator (migration 076 load contract).

Produces one row per (hcp_id, brand) with columns:
    hcp_id, brand, consideration_date, adopted, adoption_category,
    data_split, is_synthetic

CRITICAL correctness: adoption labels are derived from the STORED
hcp_df["peer_influence_score"] (which equals log1p(influence_network_size)),
NOT a fresh random draw.  This is what makes `adopted` predictable from the
model's join features (peer_influence_score, influence_network_size, etc.
live in hcp_profiles) and is why the validated prototype achieves AUC 0.77–0.82.

The shared leakage-safe DGP is _compute_adoption() from hcp_adoption_artifact.py:
    centrality_z = (pis - pis.mean()) / (pis.std() or 1.0)
    _compute_adoption(rng, centrality_z, brand)  → adopted 0/1

Leakage safety: days_to_first / first_adoption_dt / adopter_rank are NEVER emitted.
consideration_date is drawn INDEPENDENTLY of adopted (row attribute, not feature).
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.synthetic.generators.hcp_adoption_artifact import (
    ADOPTER_VALUE,
    _compute_adoption,
)

_NON_ADOPTER_VALUE = "NON_ADOPTER"

_DEFAULT_SPLIT_PROPORTIONS: Dict[str, float] = {
    "train": 0.60,
    "validation": 0.20,
    "test": 0.15,
    "holdout": 0.05,
}
_SPLIT_ORDER = ("train", "validation", "test", "holdout")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_hcp_brand_adoption_frame(
    hcp_df: pd.DataFrame,
    *,
    seed: int,
    end_date: date,
    brands: Tuple[str, ...] = ("Remibrutinib", "Fabhalta", "Kisqali"),
    n_months: int = 37,
    split_proportions: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Generate one row per (hcp_id, brand) for the hcp_brand_adoption table.

    Parameters
    ----------
    hcp_df:
        DataFrame with at minimum columns ``hcp_id`` (str) and
        ``peer_influence_score`` (float, = log1p(network_size)).
    seed:
        Master RNG seed.  Deterministic: same seed + same hcp_df → same frame.
    end_date:
        Anchor date.  consideration_date values span the trailing ``n_months``
        months ending here (first-of-month buckets).
    brands:
        Tuple of brand strings.  Must be in _BRAND_ADOPT_SCALE keys.
    n_months:
        Number of trailing monthly buckets for consideration_date.
    split_proportions:
        {split_name: fraction}.  Defaults to 60/20/15/5.  Stratified by
        ``adopted`` so both classes appear in every split.

    Returns
    -------
    pd.DataFrame with columns exactly:
        hcp_id, brand, consideration_date, adopted, adoption_category,
        data_split, is_synthetic
    """
    ratios = dict(split_proportions or _DEFAULT_SPLIT_PROPORTIONS)
    master_rng = np.random.default_rng(seed)

    # Pre-compute the monthly bucket list (consideration_date pool)
    month_buckets = _trailing_month_buckets(end_date, n_months)

    # Standardise peer_influence_score once (same vector for all brands)
    pis = hcp_df["peer_influence_score"].to_numpy(dtype=float)
    pis_std = pis.std()
    centrality_z = (pis - pis.mean()) / (pis_std if pis_std > 0 else 1.0)

    hcp_ids: List[str] = hcp_df["hcp_id"].tolist()
    n_hcps = len(hcp_ids)

    brand_frames: List[pd.DataFrame] = []
    for brand_idx, brand in enumerate(brands):
        # Derive a per-brand sub-RNG deterministically from master_rng index
        # so brand order doesn't pollute each other's streams.
        brand_seed = int(master_rng.integers(0, 2**32))
        brand_rng = np.random.default_rng(brand_seed)

        # Adoption labels from the SHARED DGP using the STORED centrality
        dgp = _compute_adoption(brand_rng, centrality_z, brand)
        adopted = dgp["adopted"]  # ndarray of 0/1, length n_hcps

        # consideration_date: drawn INDEPENDENTLY of adopted
        # Uniform over month_buckets (one draw per HCP for this brand)
        month_indices = brand_rng.integers(0, len(month_buckets), size=n_hcps)
        consideration_dates = [month_buckets[i] for i in month_indices]

        # Stratified data_split by adopted label
        data_splits = _stratified_splits(adopted, ratios)

        brand_frames.append(
            pd.DataFrame(
                {
                    "hcp_id": hcp_ids,
                    "brand": brand,
                    "consideration_date": consideration_dates,
                    "adopted": adopted.astype(int),
                    "adoption_category": np.where(adopted == 1, ADOPTER_VALUE, _NON_ADOPTER_VALUE),
                    "data_split": data_splits,
                    "is_synthetic": True,
                }
            )
        )

    result = pd.concat(brand_frames, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trailing_month_buckets(end_date: date, n_months: int) -> List[date]:
    """Return list of first-of-month dates for the trailing n_months ending at end_date.

    Month k=0 is the first-of-month of end_date itself; k=n_months-1 is the
    earliest month.  Returned in chronological order (earliest first).
    """
    # first-of-month of end_date

    anchor_month = date(end_date.year, end_date.month, 1)
    buckets: List[date] = []
    for k in range(n_months):
        # go back k months from anchor_month
        total_months = anchor_month.year * 12 + (anchor_month.month - 1) - k
        y, m = divmod(total_months, 12)
        buckets.append(date(y, m + 1, 1))
    # Return chronologically (earliest first)
    buckets.reverse()
    return buckets


def _stratified_splits(
    adopted: np.ndarray,
    ratios: Dict[str, float],
) -> List[str]:
    """Assign data_split stratified by adopted (0/1).

    Each class is split independently by the designed proportions, then the
    two assignment arrays are interleaved back to the original row order.
    This guarantees both classes appear in every split that has enough members.
    """
    n = len(adopted)
    result = np.empty(n, dtype=object)

    for label_val in (0, 1):
        idx = np.where(adopted == label_val)[0]
        if len(idx) == 0:
            continue
        splits_for_label = _chronological_split(len(idx), ratios)
        for i, split_name in zip(idx, splits_for_label, strict=True):
            result[i] = split_name

    return result.tolist()


def _chronological_split(n: int, ratios: Dict[str, float]) -> List[str]:
    """Assign split labels for a group of n rows using designed row quotas.

    The last split ("holdout") absorbs any rounding remainder.
    """
    targets: List[Tuple[str, int]] = []
    cum_ratio = 0.0
    for split in _SPLIT_ORDER[:-1]:
        cum_ratio += ratios.get(split, 0.0)
        targets.append((split, int(round(cum_ratio * n))))
    targets.append((_SPLIT_ORDER[-1], n))

    out: List[str] = []
    tier = 0
    for _ in range(n):
        # Advance tier if current quota is exhausted
        while tier < len(targets) - 1 and len(out) >= targets[tier][1]:
            tier += 1
        out.append(targets[tier][0])
    return out
