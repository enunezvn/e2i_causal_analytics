"""``_compute_adoption``'s optional ``channel_shift`` (lane T1, owner decision 2026-09-23:
"approve DGP extension, we need to recover statistical, not structural effects").

The per-HCP logit shift is the #1551 ``_specialty_affinity`` pattern: a deterministic term
added to the adoption logit that consumes NO rng draws. These tests pin

  (i)   the no-shift path is bit-identical to main's DGP (sha256 of all four outputs on a
        fixed input, baseline computed from main @ c2d0ef303 BEFORE the edit, saved as
        adgpc_digest_main.txt) -- the four rng draws and their order are untouched;
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

import hashlib

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
from src.ml.synthetic.generators.hcp_adoption_artifact import _compute_adoption, _sigmoid

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

# Baseline digests of main's _compute_adoption (c2d0ef303) on _fixed_input(n=2000, seed=11),
# rng = default_rng(427), over (hcp_segment, treatment_arm, adopted, cate_estimate).
# Regenerate with the recipe in _digest(); a change here means the no-shift DGP moved.
_MAIN_DIGEST = {
    ("Remibrutinib", True): "f19a5ab57ac58b2eee2bd2a1c4a86b0cc227c75f0a006c1d954d10cc504dbc0a",
    ("Remibrutinib", False): "e1208440d98be1dbbe9fabb522e812f41f84dbd387a1a1884732640c9343ae3e",
    ("Fabhalta", True): "78c6c256191daaabf95e409bbb08f5547ce4888999af44aada4b586ada3d35a4",
    ("Fabhalta", False): "9ca2508e71151fb54c029a6c3b61792176638a4438f4749d7d908eb7289980cb",
    ("Kisqali", True): "eb140e0a89f6f5b9e2569b4a17a2731f3ae33289cb9cba12ad92113765441999",
    ("Kisqali", False): "3e9be2b141ddd3781150b0e6f5bc9a310c10fd49a6977b07b0940dfb7f353ccd",
}


def _fixed_input(n: int = 2000, seed: int = 11) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    cz = np.log1p(rng.lognormal(3.0, 1.1, n))
    cz = (cz - cz.mean()) / cz.std()
    spec = list(rng.choice(_SPECIALTIES, size=n, p=_SPECIALTY_P))
    return cz, spec


def _digest(out: dict) -> str:
    h = hashlib.sha256()
    for k in ("hcp_segment", "treatment_arm", "adopted", "cate_estimate"):
        a = np.asarray(out[k])
        h.update(k.encode())
        if a.dtype.kind in ("U", "O"):
            h.update("|".join(map(str, a)).encode())
        else:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _channel_columns() -> list[str]:
    return sorted(set(INTERVENTION_TREATMENT_MAP.values()))


def _beta_by_column() -> dict[str, float]:
    return {INTERVENTION_TREATMENT_MAP[k]: v for k, v in ADOPTION_CHANNEL_LOGIT_BETA.items()}


# --------------------------------------------------------------------------- (i)
@pytest.mark.parametrize("brand", _BRANDS)
@pytest.mark.parametrize("with_specialty", [True, False])
def test_no_shift_outputs_are_bit_identical_to_main(brand, with_specialty):
    cz, spec = _fixed_input()
    out = _compute_adoption(
        np.random.default_rng(_DGP_SEED),
        cz,
        brand,
        specialty=spec if with_specialty else None,
        channel_shift=None,
    )
    assert _digest(out) == _MAIN_DIGEST[(brand, with_specialty)]


def test_zero_shift_vector_is_the_no_shift_path():
    cz, spec = _fixed_input()
    out = _compute_adoption(
        np.random.default_rng(_DGP_SEED),
        cz,
        "Fabhalta",
        specialty=spec,
        channel_shift=np.zeros(len(cz)),
    )
    assert _digest(out) == _MAIN_DIGEST[("Fabhalta", True)]


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
