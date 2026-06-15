"""Tests for the per-brand temporal HCP adoption generator (T2).

Tests are red-first: the generator module does NOT exist yet when these tests are
written. Run with:
    cd <worktree> && PYTHONPATH=$PWD python -m pytest tests/unit/test_synthetic/test_hcp_brand_adoption_generator.py -n0 -q

Design contract being tested:
  generate_hcp_brand_adoption_frame(hcp_df, *, seed, end_date, brands, n_months,
                                     split_proportions) -> pd.DataFrame

  Columns exactly: hcp_id, brand, consideration_date, adopted, adoption_category,
                   data_split, is_synthetic
  Leakage-safe: days_to_first / first_adoption_dt / adopter_rank MUST NOT appear.
  Labels derived from stored peer_influence_score (=> predictable from stored features).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.ml.synthetic.generators.hcp_brand_adoption_generator import (
    generate_hcp_brand_adoption_frame,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONTRACT_COLS = {
    "hcp_id",
    "brand",
    "consideration_date",
    "adopted",
    "adoption_category",
    "data_split",
    "is_synthetic",
}
_BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
_LEAKY_COLS = {"days_to_first", "first_adoption_dt", "adopter_rank"}
_VALID_SPLITS = {"train", "validation", "test", "holdout"}


def _make_hcp_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """In-memory HCP fixture — no DB.

    peer_influence_score is log1p(network_size), spread realistically.
    """
    rng = np.random.default_rng(seed)
    network_size = rng.lognormal(mean=3.0, sigma=1.1, size=n).round().astype(int)
    peer_influence_score = np.log1p(network_size)
    return pd.DataFrame(
        {
            "hcp_id": [f"h{i:05d}" for i in range(n)],
            "peer_influence_score": peer_influence_score,
            "influence_network_size": network_size,
        }
    )


@pytest.fixture(scope="module")
def hcp_df():
    return _make_hcp_df(n=300, seed=42)


@pytest.fixture(scope="module")
def frame(hcp_df):
    return generate_hcp_brand_adoption_frame(
        hcp_df,
        seed=7,
        end_date=date(2025, 12, 31),
        brands=_BRANDS,
        n_months=37,
    )


# ---------------------------------------------------------------------------
# Test 1: shape — rows == n_hcp * 3; column set == 7 contract columns exactly
# ---------------------------------------------------------------------------


def test_shape(hcp_df, frame):
    assert len(frame) == len(hcp_df) * 3, f"Expected {len(hcp_df) * 3} rows, got {len(frame)}"
    assert set(frame.columns) == _CONTRACT_COLS, f"Column mismatch: got {set(frame.columns)}"


# ---------------------------------------------------------------------------
# Test 2: uniqueness — (hcp_id, brand) unique
# ---------------------------------------------------------------------------


def test_uniqueness(frame):
    dups = frame.duplicated(subset=["hcp_id", "brand"])
    assert not dups.any(), f"Duplicate (hcp_id, brand) pairs: {dups.sum()}"


# ---------------------------------------------------------------------------
# Test 3: per-brand adoption rate within (0.10, 0.60) AND not all identical
# ---------------------------------------------------------------------------


def test_per_brand_adoption_rate(frame):
    rates = {}
    for brand in _BRANDS:
        sub = frame[frame["brand"] == brand]
        rate = sub["adopted"].mean()
        rates[brand] = rate
        assert 0.10 <= rate <= 0.60, f"Brand {brand}: adoption rate {rate:.3f} outside (0.10, 0.60)"
    # Brand scale effect: not all rates identical (Fabhalta scale=1.2 vs Kisqali 0.8)
    assert len({round(r, 3) for r in rates.values()}) > 1, (
        f"All brand adoption rates identical — brand-scale effect missing: {rates}"
    )


# ---------------------------------------------------------------------------
# Test 4: predictability — adopters have higher mean peer_influence_score
#         (point-biserial > 0) per brand
# ---------------------------------------------------------------------------


def test_predictability_per_brand(hcp_df, frame):
    merged = frame.merge(hcp_df[["hcp_id", "peer_influence_score"]], on="hcp_id", how="left")
    for brand in _BRANDS:
        sub = merged[merged["brand"] == brand]
        adopter_mean = sub.loc[sub["adopted"] == 1, "peer_influence_score"].mean()
        non_adopter_mean = sub.loc[sub["adopted"] == 0, "peer_influence_score"].mean()
        gap = adopter_mean - non_adopter_mean
        assert gap > 0, (
            f"Brand {brand}: adopter mean pis ({adopter_mean:.3f}) <= "
            f"non-adopter mean pis ({non_adopter_mean:.3f}), gap={gap:.3f}. "
            "Label is not tracking peer_influence_score — centrality was not reused."
        )


# ---------------------------------------------------------------------------
# Test 5: temporal — >= 2 distinct months; both adopted classes appear across range
# ---------------------------------------------------------------------------


def test_temporal_distribution(frame):
    months = frame["consideration_date"].unique()
    assert len(months) >= 2, f"Only {len(months)} distinct consideration months"
    # Both adopted=0 and adopted=1 must span across multiple months
    for adopted_val in (0, 1):
        sub = frame[frame["adopted"] == adopted_val]
        n_months = sub["consideration_date"].nunique()
        assert n_months >= 2, (
            f"adopted={adopted_val} restricted to {n_months} consideration month(s); "
            "month must be independent of label"
        )


# ---------------------------------------------------------------------------
# Test 6: data_split — train/validation/holdout non-empty; both classes in train + holdout
# ---------------------------------------------------------------------------


def test_data_split(frame):
    for split in ("train", "validation", "holdout"):
        sub = frame[frame["data_split"] == split]
        assert len(sub) > 0, f"Split '{split}' is empty"

    for split in ("train", "holdout"):
        sub = frame[frame["data_split"] == split]
        classes = sub["adopted"].unique()
        assert 0 in classes and 1 in classes, (
            f"Split '{split}' missing a class: adopted values = {sorted(classes)}"
        )

    # All split values must be valid
    assert frame["data_split"].isin(_VALID_SPLITS).all(), (
        f"Invalid split values: {frame['data_split'].unique()}"
    )


# ---------------------------------------------------------------------------
# Test 7: leakage — forbidden columns must not appear
# ---------------------------------------------------------------------------


def test_no_leaky_columns(frame):
    present = _LEAKY_COLS & set(frame.columns)
    assert not present, f"Leaky columns present: {present}"


# ---------------------------------------------------------------------------
# Test 8: determinism — same seed + same hcp_df => identical frame
# ---------------------------------------------------------------------------


def test_determinism(hcp_df):
    kw = {"seed": 99, "end_date": date(2025, 6, 30), "brands": _BRANDS, "n_months": 24}
    df1 = generate_hcp_brand_adoption_frame(hcp_df, **kw)
    df2 = generate_hcp_brand_adoption_frame(hcp_df, **kw)
    assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Test 9: adoption_category values are "ADOPTER" / "NON_ADOPTER" and consistent
# ---------------------------------------------------------------------------


def test_adoption_category_values(frame):
    valid = {"ADOPTER", "NON_ADOPTER"}
    assert frame["adoption_category"].isin(valid).all()
    # Must be consistent with adopted column
    adopter_rows = frame["adopted"] == 1
    assert (frame.loc[adopter_rows, "adoption_category"] == "ADOPTER").all()
    assert (frame.loc[~adopter_rows, "adoption_category"] == "NON_ADOPTER").all()


# ---------------------------------------------------------------------------
# Test 10: is_synthetic is True for all rows
# ---------------------------------------------------------------------------


def test_is_synthetic(frame):
    assert frame["is_synthetic"].all(), "Not all rows have is_synthetic=True"
