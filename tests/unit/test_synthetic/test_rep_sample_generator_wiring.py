"""COMM-ARMS Phase 3 generator wiring: rep_detailing_high + sample_dropped columns
populate, and EVERY column NOT downstream of the INITIATION latent stays byte-identical
to PRE-PHASE-3 main — the two arms draw from INDEPENDENT 0x8EE9 / 0x5A3D RNG substreams,
so they cannot shift the main stream or disturb copay/psp.

Cross-version fixture discipline (Phase 1/2 lesson): a same-code A/B comparison cannot
detect a shifted main RNG stream, so the baseline must be RECORDED from the pre-change
commit, not self-compared. Digests below were captured from main @ 32cae938 (Phase 2
psp present, rep/sample NOT wired) at n=2000, HETEROGENEOUS.

Unlike copay/psp — which fold into the LATER adherence/persistence latents and so leave
treatment_initiated byte-identical — rep/sample fold into the INITIATION latent itself.
So the three initiation-downstream columns (treatment_initiated, treatment_effect_estimate,
days_to_treatment) DO change; the "DID change" guard below asserts that, and everything
else — including the adherence/persistence/discontinuation outcomes, which are computed
from treatment_arm and NOT from treatment_initiated — must be byte-identical.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

# Columns NOT downstream of the rep/sample-touched INITIATION latent — must be byte-
# identical to pre-Phase-3 main. copay/psp columns INCLUDED on purpose (rep/sample must
# not perturb them; all arms draw from distinct, non-overlapping substreams). The
# adherence/persistence/discontinuation OUTCOMES are included too: they are built from
# treatment_arm, not treatment_initiated, so the initiation fold must not move them.
# Captured from main @ 32cae938.
_PRE_P3_INVARIANT = {
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
        "discontinued_180d": "1d439cd23d1282ea",
        "persistent_180d": "f1d4fb577f0a422e",
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
    },
}

# The three columns rep/sample DO change (initiation latent + its day). PRE-P3 digests
# (Remibrutinib:21); the "DID change" guard asserts the current generator DIFFERS — else
# rep/sample never reached the initiation outcome and the invariance test is hollow.
_PRE_P3_INIT_DOWNSTREAM_REMI21 = {
    "treatment_initiated": "38c219eefa1b88f5",
    "treatment_effect_estimate": "e8f59d2c994f4682",
    "days_to_treatment": "e031d0c4eee88717",
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
def test_rep_sample_columns_are_populated_not_null():
    df = _frame()
    for col in (
        "rep_detailing_high",
        "sample_dropped",
        "rep_detailing_high_propensity",
        "sample_dropped_propensity",
    ):
        assert df[col].notna().all(), f"{col} still NULL after Phase 3 wiring"
    for col in ("rep_detailing_high", "sample_dropped"):
        assert set(df[col].unique()) <= {0, 1}
    # measured shares ~0.49 (rep) / ~0.37 (sample)
    assert 0.35 < df["rep_detailing_high"].mean() < 0.60
    assert 0.25 < df["sample_dropped"].mean() < 0.50


@pytest.mark.unit
def test_rep_sample_propensities_have_overlap():
    df = _frame()
    for col in ("rep_detailing_high_propensity", "sample_dropped_propensity"):
        p = df[col]
        assert p.min() >= 0.01 and p.max() <= 0.99


@pytest.mark.unit
@pytest.mark.parametrize(
    "key,seed,brand",
    [
        ("Remibrutinib:21", 21, Brand.REMIBRUTINIB),
        ("Kisqali:7", 7, Brand.KISQALI),
    ],
)
def test_non_initiation_columns_are_byte_identical_to_pre_p3_stream(key, seed, brand):
    """rep/sample draw from INDEPENDENT 0x8EE9 / 0x5A3D substreams and consume the SAME
    number of main-stream draws as pre-P3 (the initiation folder makes exactly one noise
    draw, like the old binary_outcome_with_cate call), so every column that is NOT
    downstream of the initiation latent — copay/psp columns and the adherence/persistence/
    discontinuation outcomes included — must be byte-identical to pre-P3 main."""
    df = _frame(seed=seed, brand=brand)
    for col, expected in _PRE_P3_INVARIANT[key].items():
        assert _digest(df[col]) == expected, (
            f"{col} changed vs pre-P3 baseline {expected} -> {_digest(df[col])}. "
            "The rep/sample arms shifted the main RNG stream (or disturbed copay/psp or an "
            "adherence/persistence outcome); they must draw ONLY from their 0x8EE9 / 0x5A3D "
            "SeedSequence substreams and fold ONLY into the initiation latent."
        )


@pytest.mark.unit
def test_initiation_downstream_columns_DID_change_so_the_fixture_is_not_vacuous():
    """Guard the guard: rep/sample genuinely fold into the initiation latent, so all three
    initiation-downstream columns MUST differ from pre-P3. If any matches, rep/sample never
    entered the treatment_initiated outcome and the invariance test above is hollow."""
    df = _frame(seed=21, brand=Brand.REMIBRUTINIB)
    unchanged = [c for c, d in _PRE_P3_INIT_DOWNSTREAM_REMI21.items() if _digest(df[c]) == d]
    assert not unchanged, f"{unchanged} identical to pre-P3 -- rep/sample never entered initiation"


@pytest.mark.unit
def test_ground_truth_records_the_rep_and_sample_arms():
    df = _frame()
    truth = df.attrs["true_ate_by_arm"]
    for arm in ("rep_detailing_high", "sample_dropped"):
        assert arm in truth, f"{arm} missing from true_ate_by_arm"
        # rep/sample target ONLY treatment_initiated.
        assert set(truth[arm]) == {"treatment_initiated"}
        entry = truth[arm]["treatment_initiated"]
        assert entry["ate"] > 0
        cate = entry["cate_by_segment"]
        assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"]
