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


def test_default_split_proportions_match_44_quota(frame):
    """Red-first #44 pin (2026-07-21): the default split quota is 60/20/10/10
    (test 0.15→0.10, holdout 0.05→0.10 — goldstd holdout enlargement), lockstep
    with BaseGenerator._assign_splits and split_enforcer's expected_ratios. The
    stratified assignment is exact to integer rounding per class, so shares sit
    within ~2pp of the design on any non-trivial frame."""
    shares = frame["data_split"].value_counts(normalize=True)
    assert shares.get("train", 0.0) == pytest.approx(0.60, abs=0.02)
    assert shares.get("validation", 0.0) == pytest.approx(0.20, abs=0.02)
    assert shares.get("test", 0.0) == pytest.approx(0.10, abs=0.02)
    assert shares.get("holdout", 0.0) == pytest.approx(0.10, abs=0.02)


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


# ---------------------------------------------------------------------------
# Test 11 (#1551): brand-specialty adoption affinity — clinical ordering
# ---------------------------------------------------------------------------
# The champion LR trains on hcp_profiles.specialty JOINed to these labels; when
# the DGP carries no specialty term the model's specialty coefficients fit pure
# noise (measured 2026-08-11: Kisqali rheumatology 0.454 > dermatology 0.421 >
# oncology 0.410 served propensity — clinically backwards for a CDK4/6
# breast-cancer brand). The DGP must therefore encode per-brand specialty
# affinity so the fitted ordering is clinically sensible.


def _make_hcp_df_with_specialty(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """HCP fixture carrying a specialty column with a realistic mixed-brand mix
    (mirrors HCPGenerator's equal-brand cohort: ~1/3 oncology, hem-heavy and
    derm/allergy-heavy thirds)."""
    rng = np.random.default_rng(seed)
    df = _make_hcp_df(n=n, seed=seed)
    df["specialty"] = rng.choice(
        [
            "oncology",
            "hematology",
            "dermatology",
            "allergy_immunology",
            "internal_medicine",
            "rheumatology",
            "neurology",
        ],
        size=n,
        p=[0.33, 0.20, 0.17, 0.12, 0.10, 0.05, 0.03],
    )
    return df


@pytest.fixture(scope="module")
def specialty_frame():
    hcp = _make_hcp_df_with_specialty()
    frame = generate_hcp_brand_adoption_frame(
        hcp,
        seed=427,
        end_date=date(2026, 6, 1),
        brands=_BRANDS,
        n_months=37,
    )
    return frame.merge(hcp[["hcp_id", "specialty"]], on="hcp_id", how="left")


def _specialty_means(frame: pd.DataFrame, brand: str) -> "pd.Series[float]":
    sub = frame[frame["brand"] == brand]
    return sub.groupby("specialty")["adopted"].mean()


def test_kisqali_oncology_dominates_specialty_adoption(specialty_frame):
    """#1551 core invariant: Kisqali (CDK4/6, HR+/HER2- breast cancer) adoption
    must be clearly oncology-led; rheumatology/dermatology must not outrank it."""
    means = _specialty_means(specialty_frame, "Kisqali")
    onc = means["oncology"]
    assert onc > means["rheumatology"] + 0.10, (
        f"Kisqali oncology {onc:.3f} must clearly exceed rheumatology {means['rheumatology']:.3f}"
    )
    assert onc > means["dermatology"] + 0.10, (
        f"Kisqali oncology {onc:.3f} must clearly exceed dermatology {means['dermatology']:.3f}"
    )
    # Oncology strictly first among ALL specialties (adjacency never outranks it).
    assert onc == means.max(), (
        f"Kisqali top specialty must be oncology; got\n{means.sort_values(ascending=False)}"
    )


def test_fabhalta_hematology_leads_specialty_adoption(specialty_frame):
    """Fabhalta (iptacopan, complement inhibitor; PNH/IgAN) is hematology-led."""
    means = _specialty_means(specialty_frame, "Fabhalta")
    hem = means["hematology"]
    assert hem == means.max(), (
        f"Fabhalta top specialty must be hematology; got\n{means.sort_values(ascending=False)}"
    )
    for other in ("oncology", "dermatology", "rheumatology"):
        assert hem > means[other] + 0.10, (
            f"Fabhalta hematology {hem:.3f} must clearly exceed {other} {means[other]:.3f}"
        )


def test_remibrutinib_derm_allergy_lead_specialty_adoption(specialty_frame):
    """Remibrutinib (BTK inhibitor, CSU) is dermatology/allergy-immunology-led."""
    means = _specialty_means(specialty_frame, "Remibrutinib")
    top_two = set(means.sort_values(ascending=False).index[:2])
    assert top_two == {"dermatology", "allergy_immunology"}, (
        f"Remibrutinib top-2 specialties must be dermatology + allergy_immunology; "
        f"got\n{means.sort_values(ascending=False)}"
    )
    for lead in ("dermatology", "allergy_immunology"):
        assert means[lead] > means["oncology"] + 0.10, (
            f"Remibrutinib {lead} {means[lead]:.3f} must clearly exceed oncology "
            f"{means['oncology']:.3f}"
        )


def test_specialty_affinity_keeps_adoption_rate_band(specialty_frame):
    """Affinity shifts are population-balanced: per-brand marginal adoption stays
    inside the DGP's designed (0.10, 0.60) band (same band test 3 asserts on the
    no-specialty frame)."""
    for brand in _BRANDS:
        rate = specialty_frame.loc[specialty_frame["brand"] == brand, "adopted"].mean()
        assert 0.10 <= rate <= 0.60, f"{brand}: marginal adoption {rate:.3f} out of band"


def test_specialty_affinity_consumes_no_rng_draws():
    """RNG stream discipline (#1524/#1542): the affinity term is a deterministic
    logit shift, NOT an extra draw — every seeded draw (consideration_date months,
    treatment arms) must be bit-identical with and without the specialty column."""
    hcp = _make_hcp_df_with_specialty(n=500, seed=7)
    kw = {"seed": 427, "end_date": date(2026, 6, 1), "brands": _BRANDS, "n_months": 37}
    with_spec = generate_hcp_brand_adoption_frame(hcp, **kw)
    without_spec = generate_hcp_brand_adoption_frame(hcp.drop(columns=["specialty"]), **kw)
    # Identity + date columns byte-identical => no draw-count change anywhere.
    for col in ("hcp_id", "brand", "consideration_date"):
        assert with_spec[col].tolist() == without_spec[col].tolist(), (
            f"column {col!r} diverged — the specialty path consumed RNG draws"
        )
