"""Phase 1 generator wiring: copay columns populate, and EVERY pre-existing
column stays byte-identical (the copay arm draws from an INDEPENDENT RNG
substream, so it cannot shift the main stream)."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

# Captured from main @ 0a3c17dd BEFORE any copay work (n=2000, HETEROGENEOUS).
# Cross-version fixture: a same-code A/B comparison cannot detect a shifted
# main RNG stream, so it must be a recorded baseline, not a self-comparison.
_PRE_COPAY_DIGESTS = {
    "Remibrutinib:21": {
        "patient_journey_id": "9d37ed3c10cda091",
        "disease_severity": "dd481632062d2e1e",
        "academic_hcp": "6b838325de84ef60",
        "treatment_arm": "bddd18e722dc784c",
        "propensity_score": "ff7ca1d21394a4ee",
        # treatment_initiated DELIBERATELY dropped from copay's invariant set as of
        # COMM-ARMS Phase 3: rep_detailing_high + sample_dropped now fold into the
        # initiation latent, so treatment_initiated is no longer byte-identical to the
        # pre-copay stream (copay still does not touch it — Phase 3 does). Its invariance
        # is now owned by test_rep_sample_generator_wiring.py's DID-change guard.
        "segment_assignment": "97f865ca7980d7c2",
        "insurance_type": "5e5a51c7f9782291",
        "comorbidity_burden": "104b132fe693dc95",
        "prior_therapy_lines": "b14ef7b8c273c3f5",
        "age_at_diagnosis": "5f0f191c6166a2d2",
        "geographic_region": "2e57809069fb0ac7",
        "data_split": "57c10764069cebb3",
    },
    "Kisqali:7": {
        "patient_journey_id": "9d37ed3c10cda091",
        "disease_severity": "ffcff60fe074e817",
        "academic_hcp": "9249af9926476c8b",
        "treatment_arm": "b85024b8e0902234",
        "propensity_score": "83270b5be5848dd3",
        # treatment_initiated dropped as of COMM-ARMS Phase 3 (see Remibrutinib note above).
        "segment_assignment": "f9e258f8b0cd171a",
        "insurance_type": "b2825bbfbf6b7b0e",
        "comorbidity_burden": "ae031168e27b686f",
        "prior_therapy_lines": "12bcebfc603e8979",
        "age_at_diagnosis": "34ca4305e39a8ce1",
        "geographic_region": "cfea1e7674329b01",
        "data_split": "c1570c894741bb45",
    },
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
def test_copay_columns_are_populated_not_null():
    df = _frame()
    for col in ("copay_support", "copay_support_propensity", "insurance_access_score"):
        assert df[col].notna().all(), f"{col} still NULL after Phase 1 wiring"
    assert set(df["copay_support"].unique()) <= {0, 1}
    assert 0.20 < df["copay_support"].mean() < 0.60


@pytest.mark.unit
def test_propensity_has_overlap():
    df = _frame()
    p = df["copay_support_propensity"]
    assert p.min() >= 0.01 and p.max() <= 0.99


@pytest.mark.unit
def test_insurance_access_score_matches_insurance_type():
    """The persisted numeric proxy must agree with the categorical it derives from."""
    df = _frame()
    expected = {"commercial": 0.45, "medicare": 0.10, "medicaid": -0.35, "uninsured": -0.55}
    for ins, score in expected.items():
        rows = df[df["insurance_type"] == ins]
        if len(rows):
            assert np.allclose(rows["insurance_access_score"], score)


@pytest.mark.unit
@pytest.mark.parametrize(
    "key,seed,brand",
    [
        ("Remibrutinib:21", 21, Brand.REMIBRUTINIB),
        ("Kisqali:7", 7, Brand.KISQALI),
    ],
)
def test_unrelated_columns_are_byte_identical_to_the_pre_copay_stream(key, seed, brand):
    """copay draws from an INDEPENDENT 0xC0FA substream, so every column that is
    NOT downstream of the adherence latent must be byte-identical to the
    PRE-COPAY generator.

    Task 10 (DONE): discontinued_180d/persistent_180d were MIGRATED out of this
    invariant set — copay is now in the discontinuation logit, so both legitimately
    move. Their pre-copay digests live in
    test_persistence_outcomes_DID_change_so_the_fixture_is_not_vacuous, which
    asserts they DIFFER. Every column remaining here is still expected byte-identical.
    """
    df = _frame(seed=seed, brand=brand)
    for col, expected in _PRE_COPAY_DIGESTS[key].items():
        assert _digest(df[col]) == expected, (
            f"{col} changed vs pre-copay baseline {expected} -> {_digest(df[col])}. "
            "The copay arm shifted the main RNG stream; it must draw ONLY from "
            "the 0xC0FA SeedSequence substream."
        )


@pytest.mark.unit
def test_adherence_outcomes_DID_change_so_the_fixture_is_not_vacuous():
    """Guard the guard: if copay is genuinely in the adherence latent, the
    adherence columns MUST differ from pre-copay. If they match, copay is not
    actually reaching the outcome and the test above passes for a hollow reason."""
    df = _frame(seed=21, brand=Brand.REMIBRUTINIB)
    pre = {
        "adherent_180d": "ac326339e984f099",
        "low_gap_180d": "f3601a95cbc9e3bb",
        "adherence_rate": "64bd37c3215702b7",
        "gap_days": "046cecbe7952f969",
    }
    changed = [c for c, d in pre.items() if _digest(df[c]) != d]
    assert changed, "adherence columns identical to pre-copay -- copay never entered the latent"


@pytest.mark.unit
def test_persistence_outcomes_DID_change_so_the_fixture_is_not_vacuous():
    """Task 10 counterpart of the adherence guard above. These two digests are the
    VERBATIM pre-copay values migrated out of _PRE_COPAY_DIGESTS (not re-recorded),
    so the evidence survives the migration: if copay really entered the
    discontinuation logit, both columns MUST differ from pre-copay. If they match,
    copay never reached the persistence outcome."""
    df = _frame(seed=21, brand=Brand.REMIBRUTINIB)
    pre = {
        "discontinued_180d": "9962009b37a04971",
        "persistent_180d": "79939d5b995bca4f",
    }
    unchanged = [c for c, d in pre.items() if _digest(df[c]) == d]
    assert not unchanged, (
        f"{unchanged} identical to pre-copay -- copay never entered the discontinuation logit"
    )


@pytest.mark.unit
def test_ground_truth_records_the_copay_arm():
    df = _frame()
    truth = df.attrs["true_ate_by_arm"]
    assert "copay_support" in truth
    for outcome in ("adherent_180d", "low_gap_180d"):
        entry = truth["copay_support"][outcome]
        assert entry["ate"] > 0
        cate = entry["cate_by_segment"]
        assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"]
