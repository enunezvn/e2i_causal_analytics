"""``_compute_adoption``'s optional ``channel_shift`` (lane T1, owner decision 2026-09-23:
"approve DGP extension, we need to recover statistical, not structural effects").

The per-HCP logit shift is the #1551 ``_specialty_affinity`` pattern: a deterministic term
added to the adoption logit that consumes NO rng draws. These tests pin

  (i)   the no-shift path is identical to main's DGP: main's ``_compute_adoption`` (c2d0ef303)
        is FROZEN verbatim below as ``_reference_compute_adoption_main`` and every returned
        array must be exactly equal (``np.array_equal``) for the same seeded inputs -- the
        four rng draws and their order are untouched. (A first version pinned hard-coded
        sha256 digests; CI computed different digests for the same inputs on the same numpy
        2.3.5 -- the float bytes are platform-bound -- so the pin is the reference
        implementation, evaluated on whatever platform runs the test.)
  (ii)  the shift is deterministic: same inputs + a shift vector keep treatment_arm and
        hcp_segment identical, move the logit by exactly the shift, and flip ``adopted`` only
        where the SAME uniform crossed the moved sigmoid (the uniform is re-derived from the
        seeded stream, so this also pins the draw order);
  (iii) the planted magnitudes are realised: on n=40,000 with independent Bernoulli(0.5)
        exposure bits the DGP-true risk difference per channel (noise held at its drawn
        value) is within 0.01 of ``ADOPTION_CHANNEL_PLANTED_RD``, the observed stratified
        difference within 0.02, the null channel within the same bands of 0, and the
        treatment_arm ATE moves by less than 0.02. n is 40,000 rather than the brief's
        20,000 because the stratified difference's SE is ~0.007 at 20,000 and one of 24
        brand x channel cells sat 3.6 SE out (0.025) on the first seed tried; at 40,000 the
        measured maximum over four seeds is 0.015.

Run:
    cd <worktree> && python -m pytest tests/unit/test_synthetic/test_hcp_adoption_artifact_channel_shift.py -n 0 -q
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from src.data.per_hcp_cohort_collapse import (
    dgp_true_channel_rd,
    stratified_channel_rd,
    tbin_column,
)
from src.data.per_hcp_cohort_columns import (
    ADOPTION_CHANNEL_LOGIT_BETA,
    ADOPTION_CHANNEL_PLANTED_RD,
    ADOPTION_NULL_CHANNEL,
    INTERVENTION_TREATMENT_MAP,
)
from src.ml.synthetic.generators.hcp_adoption_artifact import (
    _ADOPT_CENTRALITY_SLOPE,
    _ADOPT_INTERCEPT,
    _ADOPT_TREATMENT_LOGIT,
    _BRAND_ADOPT_SCALE,
    _compute_adoption,
    _sigmoid,
    _specialty_affinity,
)

_SPECIALTIES = (
    "oncology",
    "hematology",
    "dermatology",
    "allergy_immunology",
    "internal_medicine",
    "rheumatology",
    "neurology",
)
_SPECIALTY_P = (0.33, 0.20, 0.17, 0.12, 0.10, 0.05, 0.03)
_BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
_DGP_SEED = 427


def _fixed_input(n: int = 2000, seed: int = 11) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    cz = np.log1p(rng.lognormal(3.0, 1.1, n))
    cz = (cz - cz.mean()) / cz.std()
    spec = list(rng.choice(_SPECIALTIES, size=n, p=_SPECIALTY_P))
    return cz, spec


# main's _compute_adoption (origin/main @ c2d0ef303, sha256 of the module
# 7980bdaac778a17c946f8c61abc4d1e899b54aeb13240a57419cca70a403c31d), FROZEN verbatim. Its
# helpers (_sigmoid, _specialty_affinity) and constants are imported from the module: the
# lane did not change them (git diff origin/main..HEAD touches _compute_adoption only).
def _reference_compute_adoption_main(
    rng: np.random.Generator,
    centrality_z: np.ndarray,
    brand: str,
    specialty: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Shared adoption DGP on a standardized centrality vector. Returns
    hcp_segment, treatment_arm, adopted (0/1), and the per-HCP probability-scale
    cate_estimate. Used by BOTH the standalone optum_hcp frame and the hcp_generator
    (hcp_profiles grain) so the two grains share one DGP.

    ``specialty`` (optional, #1551): per-HCP specialty strings aligned to
    ``centrality_z``. When supplied, a deterministic per-(brand, specialty)
    logit shift makes the specialty ordering clinically sensible (see
    ``_BRAND_SPECIALTY_AFFINITY``). When None, behaviour is bit-identical to the
    pre-#1551 DGP — the shift consumes no RNG draws either way."""
    n = len(centrality_z)
    hcp_segment = np.where(
        centrality_z > 0.5,
        "high_influence",
        np.where(centrality_z > -0.5, "medium_influence", "low_influence"),
    )
    # treatment_arm ~ rep/trigger engagement intensity, CONFOUNDED by centrality
    # (central HCPs get more rep attention) -> propensity is estimable.
    p_treat = _sigmoid(0.8 * centrality_z + rng.normal(0, 0.5, n))
    treatment_arm = (rng.random(n) < p_treat).astype(int)

    scale = _BRAND_ADOPT_SCALE.get(brand, 1.0)
    seg_treat = np.array([_ADOPT_TREATMENT_LOGIT[s] for s in hcp_segment], dtype=float)
    # #1551: deterministic specialty-affinity shift (zeros when specialty=None);
    # a pure table lookup — consumes NO rng draws, so the seeded stream is
    # bit-identical with or without a specialty vector.
    affinity = _specialty_affinity(brand, specialty, n)
    logit = (
        _ADOPT_INTERCEPT
        + _ADOPT_CENTRALITY_SLOPE * centrality_z
        + affinity
        + scale * seg_treat * treatment_arm
        + rng.normal(0.0, 0.6, n)
    )
    adopted = (rng.random(n) < _sigmoid(logit)).astype(int)

    # per-HCP CATE on the PROBABILITY scale (P(adopt) at T=1 vs T=0, centrality
    # and specialty fixed).
    base_logit = _ADOPT_INTERCEPT + _ADOPT_CENTRALITY_SLOPE * centrality_z + affinity
    cate_estimate = _sigmoid(base_logit + scale * seg_treat) - _sigmoid(base_logit)
    return {
        "hcp_segment": hcp_segment,
        "treatment_arm": treatment_arm,
        "adopted": adopted,
        "cate_estimate": cate_estimate,
    }


def _channel_columns() -> list[str]:
    return sorted(set(INTERVENTION_TREATMENT_MAP.values()))


def _beta_by_column() -> dict[str, float]:
    return {INTERVENTION_TREATMENT_MAP[k]: v for k, v in ADOPTION_CHANNEL_LOGIT_BETA.items()}


# --------------------------------------------------------------------------- (i)
_OUTPUT_KEYS = ("hcp_segment", "treatment_arm", "adopted", "cate_estimate")


@pytest.mark.parametrize("brand", _BRANDS)
@pytest.mark.parametrize("with_specialty", [True, False])
def test_no_shift_outputs_are_identical_to_the_frozen_main_reference(brand, with_specialty):
    cz, spec = _fixed_input()
    specialty = spec if with_specialty else None
    ref = _reference_compute_adoption_main(
        np.random.default_rng(_DGP_SEED), cz, brand, specialty=specialty
    )
    out = _compute_adoption(
        np.random.default_rng(_DGP_SEED), cz, brand, specialty=specialty, channel_shift=None
    )
    for key in _OUTPUT_KEYS:
        assert np.array_equal(np.asarray(out[key]), np.asarray(ref[key])), key
    assert set(ref) == set(
        _OUTPUT_KEYS
    )  # the reference is main's contract; the new keys are additive
    assert set(out) == {*_OUTPUT_KEYS, "adoption_logit", "channel_shift"}


def test_zero_shift_vector_is_the_no_shift_path():
    cz, spec = _fixed_input()
    none = _compute_adoption(
        np.random.default_rng(_DGP_SEED), cz, "Fabhalta", specialty=spec, channel_shift=None
    )
    zeros = _compute_adoption(
        np.random.default_rng(_DGP_SEED),
        cz,
        "Fabhalta",
        specialty=spec,
        channel_shift=np.zeros(len(cz)),
    )
    for key in (*_OUTPUT_KEYS, "adoption_logit", "channel_shift"):
        assert np.array_equal(np.asarray(none[key]), np.asarray(zeros[key])), key


# --------------------------------------------------------------------------- (ii)
def _uniform_draws(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Replay the DGP's four draws: normal(0,.5) [arm noise], random [arm], normal(0,.6)
    [adoption noise], random [adoption]. Returns the two uniforms."""
    rng = np.random.default_rng(seed)
    rng.normal(0, 0.5, n)
    u_arm = rng.random(n)
    rng.normal(0.0, 0.6, n)
    u_adopt = rng.random(n)
    return u_arm, u_adopt


def test_shift_moves_the_logit_deterministically_and_flips_only_crossed_uniforms():
    cz, spec = _fixed_input(n=3000, seed=5)
    n = len(cz)
    shift = np.random.default_rng(99).normal(0.0, 0.7, n)
    base = _compute_adoption(np.random.default_rng(_DGP_SEED), cz, "Kisqali", specialty=spec)
    moved = _compute_adoption(
        np.random.default_rng(_DGP_SEED), cz, "Kisqali", specialty=spec, channel_shift=shift
    )
    assert moved["treatment_arm"].tolist() == base["treatment_arm"].tolist()
    assert moved["hcp_segment"].tolist() == base["hcp_segment"].tolist()
    # The shift is the ONLY difference on the logit, and it is exposed for writers.
    np.testing.assert_allclose(moved["adoption_logit"] - base["adoption_logit"], shift, atol=1e-12)
    np.testing.assert_array_equal(moved["channel_shift"], shift)
    np.testing.assert_array_equal(base["channel_shift"], np.zeros(n))
    # Same uniform (4th draw) decides adoption in both runs -> flips are exactly the crossings.
    _, u_adopt = _uniform_draws(_DGP_SEED, n)
    np.testing.assert_array_equal(
        base["adopted"], (u_adopt < _sigmoid(base["adoption_logit"])).astype(int)
    )
    np.testing.assert_array_equal(
        moved["adopted"], (u_adopt < _sigmoid(moved["adoption_logit"])).astype(int)
    )
    flipped = moved["adopted"] != base["adopted"]
    assert 0 < flipped.sum() < n
    # Positive shift can only turn 0 -> 1, negative only 1 -> 0.
    assert (moved["adopted"][flipped & (shift > 0)] == 1).all()
    assert (moved["adopted"][flipped & (shift < 0)] == 0).all()
    # cate_estimate's base now includes the shift.
    assert not np.allclose(moved["cate_estimate"], base["cate_estimate"])


def test_shift_length_mismatch_is_refused():
    cz, spec = _fixed_input(n=100)
    with pytest.raises(ValueError, match="channel_shift"):
        _compute_adoption(
            np.random.default_rng(1), cz, "Kisqali", specialty=spec, channel_shift=np.zeros(99)
        )


# --------------------------------------------------------------------------- (iii)
@pytest.mark.parametrize("brand", _BRANDS)
def test_planted_risk_differences_are_realised_on_independent_exposure_bits(brand):
    n = 40_000
    cz, spec = _fixed_input(n=n, seed=23)
    channels = _channel_columns()
    beta = _beta_by_column()
    rd_by_col = {INTERVENTION_TREATMENT_MAP[k]: v for k, v in ADOPTION_CHANNEL_PLANTED_RD.items()}
    bits = (np.random.default_rng(77).random((n, len(channels))) < 0.5).astype(float)
    shift = ((bits - 0.5) * np.array([beta[c] for c in channels])).sum(axis=1)
    tbin = pd.DataFrame({tbin_column(c): bits[:, j] for j, c in enumerate(channels)})

    base = _compute_adoption(np.random.default_rng(_DGP_SEED), cz, brand, specialty=spec)
    out = _compute_adoption(
        np.random.default_rng(_DGP_SEED), cz, brand, specialty=spec, channel_shift=shift
    )
    true_rd = dgp_true_channel_rd(out["adoption_logit"], tbin)
    strat_rd = stratified_channel_rd(out["adopted"], tbin)
    null_col = INTERVENTION_TREATMENT_MAP[ADOPTION_NULL_CHANNEL]
    assert true_rd[null_col] == 0.0
    for col in channels:
        target = rd_by_col[col]
        assert abs(true_rd[col] - target) <= 0.01, (
            f"{brand} {col}: DGP-true RD {true_rd[col]:.3f} vs planted {target:.3f}"
        )
        assert abs(strat_rd[col] - target) <= 0.02, (
            f"{brand} {col}: stratified RD {strat_rd[col]:.3f} vs planted {target:.3f}"
        )
    # The treatment_arm effect survives the channel term (sigmoid non-linearity only).
    assert abs(float(out["cate_estimate"].mean()) - float(base["cate_estimate"].mean())) <= 0.02
    assert out["treatment_arm"].tolist() == base["treatment_arm"].tolist()
