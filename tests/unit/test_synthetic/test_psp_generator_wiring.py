"""Phase 2 generator wiring: psp columns populate, and EVERY pre-existing column
(including copay's) stays byte-identical to PRE-PSP main — the psp arm draws from an
INDEPENDENT 0x9527 RNG substream, so it cannot shift the main stream or disturb copay.

Cross-version fixture discipline (Phase 1 lesson): a same-code A/B comparison cannot
detect a shifted main RNG stream, so the baseline must be RECORDED from the pre-change
commit, not self-compared. Digests below were captured from main @ 173d8b95 (copay
present, psp NOT wired) at n=2000, HETEROGENEOUS.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

# Every column NOT downstream of the psp-touched latents — must be byte-identical to
# pre-psp main. copay_support + copay_support_propensity are INCLUDED on purpose: psp
# must not perturb the copay arm (both arms draw from distinct, non-overlapping
# substreams). Captured from main @ 173d8b95.
_PRE_PSP_INVARIANT = {
    "Remibrutinib:21": {
        "patient_journey_id": "9d37ed3c10cda091",
        "disease_severity": "dd481632062d2e1e",
        "academic_hcp": "6b838325de84ef60",
        "engagement_score": "eb514850e0d1892c",
        "treatment_arm": "bddd18e722dc784c",
        "propensity_score": "ff7ca1d21394a4ee",
        # treatment_initiated DELIBERATELY dropped from psp's invariant set as of
        # COMM-ARMS Phase 3: rep_detailing_high + sample_dropped now fold into the
        # initiation latent, so treatment_initiated is no longer byte-identical to the
        # pre-psp stream (psp still does not touch it — Phase 3 does). Its invariance is
        # now owned by test_rep_sample_generator_wiring.py's DID-change guard.
        "segment_assignment": "97f865ca7980d7c2",
        "insurance_type": "5e5a51c7f9782291",
        "comorbidity_burden": "104b132fe693dc95",
        "prior_therapy_lines": "b14ef7b8c273c3f5",
        "age_at_diagnosis": "5f0f191c6166a2d2",
        "geographic_region": "2e57809069fb0ac7",
        "data_split": "bf0f985689e143c3",  # re-pinned for #44 split policy 60/20/10/10 (only data_split moved; all other columns held)
        "copay_support": "8ae5b0b33943ad52",
        "copay_support_propensity": "a41d21b251de602e",
        "insurance_access_score": "d003f6a805fc9536",
    },
    "Kisqali:7": {
        "patient_journey_id": "9d37ed3c10cda091",
        "disease_severity": "ffcff60fe074e817",
        "academic_hcp": "9249af9926476c8b",
        "engagement_score": "2bf1f8974caa4d99",
        "treatment_arm": "b85024b8e0902234",
        "propensity_score": "83270b5be5848dd3",
        # treatment_initiated dropped as of COMM-ARMS Phase 3 (see Remibrutinib note above).
        "segment_assignment": "f9e258f8b0cd171a",
        "insurance_type": "b2825bbfbf6b7b0e",
        "comorbidity_burden": "ae031168e27b686f",
        "prior_therapy_lines": "12bcebfc603e8979",
        "age_at_diagnosis": "34ca4305e39a8ce1",
        "geographic_region": "cfea1e7674329b01",
        "data_split": "9a071042dba9492c",  # re-pinned for #44 split policy 60/20/10/10 (only data_split moved; all other columns held)
        "copay_support": "97711255df1feee9",
        "copay_support_propensity": "4c2db6c443a8d193",
        "insurance_access_score": "d7b79fe8ec9ef0f2",
    },
}

# The columns psp DOES touch (both latents). PRE-PSP digests (Remibrutinib:21); the
# "DID change" guard asserts the current generator DIFFERS from these — otherwise psp
# never reached the outcomes and the invariance test above passes for a hollow reason.
_PRE_PSP_OUTCOMES_REMI21 = {
    "adherent_180d": "d86d16d9bbc4d71e",
    "low_gap_180d": "4df1bacc563283e3",
    "adherence_rate": "c26d8cacc644b672",
    "gap_days": "822fe33fc132c3fd",
    "discontinued_180d": "f838842de87a1793",
    "persistent_180d": "e12d3b1a897a2ccd",
}


def _digest(s: pd.Series) -> str:
    v = s.to_numpy()
    if v.dtype == object or str(v.dtype).startswith("datetime"):
        b = "|".join(map(str, v.tolist())).encode()
    else:
        b = np.ascontiguousarray(np.asarray(v, dtype=float)).tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def _frame(seed=21, n=2000, brand=Brand.REMIBRUTINIB):
    cfg = GeneratorConfig(seed=seed, n_records=n, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    return PatientGenerator(cfg).generate()


@pytest.mark.unit
def test_psp_columns_are_populated_not_null():
    df = _frame()
    for col in ("psp_enrolled", "psp_enrolled_propensity"):
        assert df[col].notna().all(), f"{col} still NULL after Phase 2 wiring"
    assert set(df["psp_enrolled"].unique()) <= {0, 1}
    assert 0.20 < df["psp_enrolled"].mean() < 0.60


@pytest.mark.unit
def test_psp_propensity_has_overlap():
    df = _frame()
    p = df["psp_enrolled_propensity"]
    assert p.min() >= 0.01 and p.max() <= 0.99


@pytest.mark.unit
@pytest.mark.parametrize(
    "key,seed,brand",
    [
        ("Remibrutinib:21", 21, Brand.REMIBRUTINIB),
        ("Kisqali:7", 7, Brand.KISQALI),
    ],
)
def test_unrelated_columns_are_byte_identical_to_the_pre_psp_stream(key, seed, brand):
    """psp draws from an INDEPENDENT 0x9527 substream and does not change main-stream
    CONSUMPTION (it shifts baseline VALUES, not the number of rng draws), so every
    column that is NOT an outcome of the adherence/discontinuation latents — copay's
    two columns included — must be byte-identical to pre-psp main."""
    df = _frame(seed=seed, brand=brand)
    for col, expected in _PRE_PSP_INVARIANT[key].items():
        assert _digest(df[col]) == expected, (
            f"{col} changed vs pre-psp baseline {expected} -> {_digest(df[col])}. "
            "The psp arm shifted the main RNG stream (or disturbed copay); it must draw "
            "ONLY from the 0x9527 SeedSequence substream."
        )


@pytest.mark.unit
def test_outcome_columns_DID_change_so_the_fixture_is_not_vacuous():
    """Guard the guard: psp is genuinely in BOTH latents, so all six outcome columns
    MUST differ from pre-psp. If any matches, psp is not actually reaching that outcome
    and the invariance test above passes for a hollow reason."""
    df = _frame(seed=21, brand=Brand.REMIBRUTINIB)
    unchanged = [c for c, d in _PRE_PSP_OUTCOMES_REMI21.items() if _digest(df[c]) == d]
    assert not unchanged, f"{unchanged} identical to pre-psp -- psp never entered that latent"


@pytest.mark.unit
def test_ground_truth_records_the_psp_arm():
    df = _frame()
    truth = df.attrs["true_ate_by_arm"]
    assert "psp_enrolled" in truth
    # psp targets adherent_180d + persistent_180d (NOT low_gap).
    assert set(truth["psp_enrolled"]) == {"adherent_180d", "persistent_180d"}
    for outcome in ("adherent_180d", "persistent_180d"):
        entry = truth["psp_enrolled"][outcome]
        assert entry["ate"] > 0
        cate = entry["cate_by_segment"]
        assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"]
