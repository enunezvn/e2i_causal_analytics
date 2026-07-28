"""COMM-ARMS Phase 4 generator wiring: the trigger_accepted arm column populates, and
EVERY column NOT downstream of the INITIATION latent stays byte-identical to PRE-PHASE-4
main — the arm draws from an INDEPENDENT 0x7ACC RNG substream, so it cannot shift the
main stream or disturb copay/psp/rep/sample.

Cross-version fixture discipline (Phase 1/2/3 lesson): a same-code A/B comparison cannot
detect a shifted main RNG stream, so the baseline must be RECORDED from the pre-change
commit, not self-compared. Digests below were captured from main @ dab30f12 (Phase 3
rep/sample present, trigger_accepted NOT wired) at n=2000, HETEROGENEOUS.

Like rep/sample — and unlike copay/psp — trigger_accepted folds into the INITIATION
latent itself, so the three initiation-downstream columns (treatment_initiated,
treatment_effect_estimate, days_to_treatment) DO change; the "DID change" guard below
asserts that, and everything else — including the Phase 3 arms and the adherence/
persistence/discontinuation outcomes — must be byte-identical.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

# Columns NOT downstream of the trigger_accepted-touched INITIATION latent — must be
# byte-identical to pre-Phase-4 main. The Phase 3 rep/sample arm columns are INCLUDED on
# purpose (trigger_accepted must not perturb them; all arms draw from distinct,
# non-overlapping substreams). Captured from main @ dab30f12.
_PRE_P4_INVARIANT = {
    "Remibrutinib:21": {
        "patient_journey_id": "9d37ed3c10cda091",
        "disease_severity": "dd481632062d2e1e",
        "academic_hcp": "6b838325de84ef60",
        "engagement_score": "eb514850e0d1892c",
        "treatment_arm": "bddd18e722dc784c",
        "propensity_score": "ff7ca1d21394a4ee",
        "segment_assignment": "97f865ca7980d7c2",
        "insurance_type": "5e5a51c7f9782291",
        "comorbidity_burden": "104b132fe693dc95",
        "prior_therapy_lines": "b14ef7b8c273c3f5",
        "age_at_diagnosis": "5f0f191c6166a2d2",
        "geographic_region": "2e57809069fb0ac7",
        "data_split": "bf0f985689e143c3",  # re-pinned for #44 split policy 60/20/10/10 (only data_split moved; all other columns held)
        "copay_support": "8ae5b0b33943ad52",
        "copay_support_propensity": "a41d21b251de602e",
        "psp_enrolled": "b0ff8a972ebfe483",
        "psp_enrolled_propensity": "992d91d87cfbb142",
        "insurance_access_score": "d003f6a805fc9536",
        "adherent_180d": "28960d02b20ca108",
        "low_gap_180d": "5a81b0ef25c78b11",
        "adherence_rate": "0e4d61f03807522b",
        "gap_days": "58e13c9a2cf92950",
        # #1321: re-pinned — Remibrutinib's uncontrolled-CSU axis rewrites ONLY these two
        # persistence columns (verified: every other column above held byte-identical).
        # 2026-07-28: re-pinned again — axis INVERTED positive (main -0.55 / exp_mult 2.08);
        # still ONLY these two columns move (22/24 held byte-identical).
        "discontinued_180d": "369cbe7c78b8208b",
        "persistent_180d": "6275ec3f5ed8f6ab",
        "rep_detailing_high": "ac52bccf36c445cb",
        "sample_dropped": "aa06fe312b6613c9",
        "rep_detailing_high_propensity": "2cf5c5252ae003f6",
        "sample_dropped_propensity": "edeaefb2246228b6",
    },
    "Kisqali:7": {
        "patient_journey_id": "9d37ed3c10cda091",
        "disease_severity": "ffcff60fe074e817",
        "academic_hcp": "9249af9926476c8b",
        "engagement_score": "2bf1f8974caa4d99",
        "treatment_arm": "b85024b8e0902234",
        "propensity_score": "83270b5be5848dd3",
        "segment_assignment": "f9e258f8b0cd171a",
        "insurance_type": "b2825bbfbf6b7b0e",
        "comorbidity_burden": "ae031168e27b686f",
        "prior_therapy_lines": "12bcebfc603e8979",
        "age_at_diagnosis": "34ca4305e39a8ce1",
        "geographic_region": "cfea1e7674329b01",
        "data_split": "9a071042dba9492c",  # re-pinned for #44 split policy 60/20/10/10 (only data_split moved; all other columns held)
        "copay_support": "97711255df1feee9",
        "copay_support_propensity": "4c2db6c443a8d193",
        "psp_enrolled": "675c23408d7dd6f0",
        "psp_enrolled_propensity": "3cd428448b4b335c",
        "insurance_access_score": "d7b79fe8ec9ef0f2",
        "adherent_180d": "1dbcc954a3852e2f",
        "low_gap_180d": "5e7778a88de98f13",
        "adherence_rate": "69fd0db604a514b9",
        "gap_days": "a845e7fc6c66c4d5",
        # #1321: re-pinned — Kisqali's advanced-line axis rewrites ONLY these two
        # persistence columns (verified: every other column above held byte-identical).
        "discontinued_180d": "257409c5d397bf2b",
        "persistent_180d": "7b8edb4b009a2aff",
        "rep_detailing_high": "6798d1782ca514e4",
        "sample_dropped": "cab9c018c32ba54f",
        "rep_detailing_high_propensity": "93c1e3e38517d4a0",
        "sample_dropped_propensity": "7029ea88592bf2aa",
    },
}

# The three columns trigger_accepted DOES change (initiation latent + its day). PRE-P4
# digests (Remibrutinib:21); the "DID change" guard asserts the current generator
# DIFFERS — else trigger_accepted never reached the initiation outcome and the
# invariance test above is hollow.
_PRE_P4_INIT_DOWNSTREAM_REMI21 = {
    "treatment_initiated": "b5eddf56e1879641",
    "treatment_effect_estimate": "93565c6f7dbfa961",
    "days_to_treatment": "fcd500e3b27c8195",
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
def test_trigger_accepted_columns_are_populated_not_null():
    df = _frame()
    for col in ("trigger_accepted", "trigger_accepted_propensity"):
        assert df[col].notna().all(), f"{col} still NULL after Phase 4 wiring"
    assert set(df["trigger_accepted"].unique()) <= {0, 1}
    # design share ~0.55 (PROVISIONAL band until the disproof harness pins the
    # measured intercept; tighten alongside the ArmSpec constants).
    assert 0.40 < df["trigger_accepted"].mean() < 0.70


@pytest.mark.unit
def test_trigger_accepted_propensity_has_overlap():
    df = _frame()
    p = df["trigger_accepted_propensity"]
    assert p.min() >= 0.01 and p.max() <= 0.99


@pytest.mark.unit
@pytest.mark.parametrize(
    "key,seed,brand",
    [
        ("Remibrutinib:21", 21, Brand.REMIBRUTINIB),
        ("Kisqali:7", 7, Brand.KISQALI),
    ],
)
def test_non_initiation_columns_are_byte_identical_to_pre_p4_stream(key, seed, brand):
    """trigger_accepted draws from the INDEPENDENT 0x7ACC substream and consumes the
    SAME number of main-stream draws as pre-P4 (the initiation folder makes exactly one
    noise draw), so every column NOT downstream of the initiation latent — the four
    existing commercial arms and the adherence/persistence/discontinuation outcomes
    included — must be byte-identical to pre-P4 main."""
    df = _frame(seed=seed, brand=brand)
    for col, expected in _PRE_P4_INVARIANT[key].items():
        assert _digest(df[col]) == expected, (
            f"{col} changed vs pre-P4 baseline {expected} -> {_digest(df[col])}. "
            "The trigger_accepted arm shifted the main RNG stream (or disturbed an "
            "existing arm or a non-initiation outcome); it must draw ONLY from its "
            "0x7ACC SeedSequence substream and fold ONLY into the initiation latent."
        )


@pytest.mark.unit
def test_initiation_downstream_columns_DID_change_so_the_fixture_is_not_vacuous():
    """Guard the guard: trigger_accepted genuinely folds into the initiation latent, so
    all three initiation-downstream columns MUST differ from pre-P4. If any matches,
    trigger_accepted never entered the treatment_initiated outcome and the invariance
    test above is hollow."""
    df = _frame(seed=21, brand=Brand.REMIBRUTINIB)
    unchanged = [c for c, d in _PRE_P4_INIT_DOWNSTREAM_REMI21.items() if _digest(df[c]) == d]
    assert not unchanged, (
        f"{unchanged} identical to pre-P4 -- trigger_accepted never entered initiation"
    )


@pytest.mark.unit
def test_ground_truth_records_the_trigger_accepted_arm():
    df = _frame()
    truth = df.attrs["true_ate_by_arm"]
    assert "trigger_accepted" in truth, "trigger_accepted missing from true_ate_by_arm"
    # trigger_accepted targets ONLY treatment_initiated.
    assert set(truth["trigger_accepted"]) == {"treatment_initiated"}
    entry = truth["trigger_accepted"]["treatment_initiated"]
    assert entry["ate"] > 0
    cate = entry["cate_by_segment"]
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"]
