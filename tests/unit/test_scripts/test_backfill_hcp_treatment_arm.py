"""Tests for scripts/backfill_hcp_treatment_arm.py's pure ``derive()`` (#1551).

Hermetic — no live DB. ``derive()`` is exercised as a pure function on injected
centrality frames.

Codex iter-1 on PR #1555 (HIGH, verified): the backfill's contract is
"re-derive ``adopted`` from the COMMITTED DGP", and after #1551 the committed
DGP includes the per-(brand, specialty) adoption affinity. A specialty-blind
``derive()`` would — under ``--execute`` — UPDATE all 15k live
``hcp_brand_adoption`` labels back to the specialty-free distribution,
silently clobbering the served-propensity fix. These tests pin:

  * Kisqali oncology-led adoption means out of ``derive()`` (mirrors the
    generator invariant in test_hcp_brand_adoption_generator.py)
  * the specialty-free frame still derives (default-shift/legacy path)
  * treatment_arm is byte-identical with/without the specialty column — the
    affinity consumes NO RNG draws (#1524/#1542 stream discipline), and
    faithful arm reproduction is this script's entire purpose

Run:
    cd <worktree> && PYTHONPATH=$PWD python -m pytest \
        tests/unit/test_scripts/test_backfill_hcp_treatment_arm.py -n0 -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.backfill_hcp_treatment_arm import BRANDS, derive

_SPECIALTIES = (
    "oncology",
    "hematology",
    "dermatology",
    "allergy_immunology",
    "internal_medicine",
    "rheumatology",
    "neurology",
)
# Mirrors the canonical mixed-brand cohort's specialty mix (~1/3 oncology, etc.).
_SPECIALTY_P = (0.33, 0.20, 0.17, 0.12, 0.10, 0.05, 0.03)


def _centrality_frame(n: int = 3000, seed: int = 42, with_specialty: bool = True) -> pd.DataFrame:
    """Injected stand-in for fetch_centrality()'s output (hcp_profiles read)."""
    rng = np.random.default_rng(seed)
    network_size = rng.lognormal(mean=3.0, sigma=1.1, size=n)
    df = pd.DataFrame(
        {
            "hcp_id": [f"scv_hcp_{i:06d}" for i in range(n)],
            # DB stores peer_influence_score rounded; 2dp mirrors the live read.
            "peer_influence_score": np.round(np.log1p(network_size), 2),
        }
    )
    if with_specialty:
        df["specialty"] = rng.choice(_SPECIALTIES, size=n, p=_SPECIALTY_P)
    return df


@pytest.fixture(scope="module")
def derived_with_specialty() -> pd.DataFrame:
    frame = _centrality_frame()
    out = derive(frame, seed=427)
    return out.merge(frame[["hcp_id", "specialty"]], on="hcp_id", how="left")


def test_derive_kisqali_oncology_led(derived_with_specialty):
    """#1551: derive() must apply the committed DGP's specialty affinity — a
    specialty-blind re-derivation would clobber the fixed labels on --execute."""
    sub = derived_with_specialty[derived_with_specialty["brand"] == "Kisqali"]
    means = sub.groupby("specialty")["adopted"].mean()
    assert means.idxmax() == "oncology", (
        f"Kisqali top specialty must be oncology; got\n{means.sort_values(ascending=False)}"
    )
    assert means["oncology"] > means["rheumatology"] + 0.10
    assert means["oncology"] > means["dermatology"] + 0.10


def test_derive_specialty_free_frame_still_works():
    """A frame without a specialty column derives via the legacy default-shift
    path (zero affinity) — shape/columns/bands intact."""
    out = derive(_centrality_frame(n=800, with_specialty=False), seed=427)
    assert len(out) == 800 * len(BRANDS)
    assert set(out.columns) == {
        "hcp_id",
        "brand",
        "treatment_arm",
        "adopted",
        "adoption_category",
        "cate_estimate",
    }
    for brand in BRANDS:
        rate = out.loc[out["brand"] == brand, "adopted"].mean()
        assert 0.10 <= rate <= 0.60, f"{brand}: adoption rate {rate:.3f} out of band"


def test_derive_treatment_arm_unaffected_by_specialty():
    """RNG stream discipline: the affinity is a deterministic logit shift with
    ZERO extra draws, so the arm this script exists to faithfully reproduce must
    be byte-identical with and without the specialty column."""
    with_spec = _centrality_frame(n=1000, seed=7, with_specialty=True)
    out_spec = derive(with_spec, seed=427)
    out_nospec = derive(with_spec.drop(columns=["specialty"]), seed=427)
    assert out_spec["treatment_arm"].tolist() == out_nospec["treatment_arm"].tolist(), (
        "treatment_arm diverged — the specialty path consumed RNG draws"
    )
    assert out_spec["hcp_id"].tolist() == out_nospec["hcp_id"].tolist()
    assert out_spec["brand"].tolist() == out_nospec["brand"].tolist()


def test_derive_null_specialty_takes_default_shift():
    """NULL/missing specialty values (possible in a live hcp_profiles read) take
    the brand's default affinity shift via _specialty_affinity's .get(s, default)
    — derive() must not crash and must keep the marginal in band."""
    frame = _centrality_frame(n=600, with_specialty=True)
    frame.loc[frame.index[:100], "specialty"] = None
    out = derive(frame, seed=427)
    assert len(out) == 600 * len(BRANDS)
    for brand in BRANDS:
        rate = out.loc[out["brand"] == brand, "adopted"].mean()
        assert 0.10 <= rate <= 0.60, f"{brand}: adoption rate {rate:.3f} out of band"
