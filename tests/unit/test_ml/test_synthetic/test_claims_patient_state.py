"""P1.1 — latent state + demographics + inpatient generator (Shard 10).

Encodes the SOURCE-VERIFIED converter mechanism facts as assertions:

* Fact #1 (``convert_optum_rwd._derive_index_date`` :1834) — cohort-A index is
  claim-dated: the 2nd of ≥2 distinct L50.x inpatient ``diag1..5`` admit dates.
  Vendor ``indexdt`` is IGNORED, so the DGP encodes timing in claim dates.
* Fact #2 (``_check_enrollment_window`` :1993, production regime 360/180 :117)
  — strict gate requires ``eligeff <= index - 360d`` AND ``eligend >= index +
  180d``. The fragmentation knob deliberately makes ~half the panel violate it.

Hermetic: zero DB rows by construction (pure numpy/pandas).
"""

import numpy as np

from src.ml.synthetic.claims.config import ClaimsDGPConfig
from src.ml.synthetic.claims.patient_state import emit_inpatient, generate_patients


def _l50_prefixes() -> tuple[str, ...]:
    # Matches CSU_DX_PREFIXES in convert_optum_rwd.py:260.
    return ("L501", "L508", "L509")


def test_enrollment_gate_brackets_index_production_regime():
    cfg = ClaimsDGPConfig(n_patients=80, seed=7)  # production pre=360/post=180
    pats = generate_patients(np.random.default_rng(7), cfg)
    # The NON-fragmented patients must satisfy the strict enrollment window so
    # the converter does not drop the whole cohort at :2017.
    ok = pats[pats["continuous_enrollment"] == 1]
    assert len(ok) > 0
    assert (ok["eligeff"] <= ok["claim_index"] - np.timedelta64(cfg.pre_days, "D")).all()
    assert (ok["eligend"] >= ok["claim_index"] + np.timedelta64(cfg.post_days, "D")).all()
    # Latent-state block present (DGP item 1).
    assert {"severity", "response_propensity", "adherence_propensity"} <= set(pats.columns)


def test_fragmentation_drops_roughly_half_not_all():
    cfg = ClaimsDGPConfig(n_patients=400, seed=11, panel_fragmentation_rate=0.50)
    pats = generate_patients(np.random.default_rng(11), cfg)
    frac_ok = (pats["continuous_enrollment"] == 1).mean()
    # Must keep a real cohort (not 0%) and drop a real fraction (not 100%).
    assert 0.3 < frac_ok < 0.7


def test_demographics_diagcode_passes_csu_gate_no_exclusion_collision():
    cfg = ClaimsDGPConfig(n_patients=50, seed=3)
    pats = generate_patients(np.random.default_rng(3), cfg)
    # Fact: demographics gate :1663 keeps L50.x; exclusions :261 must NOT fire.
    assert pats["diagcode"].str.upper().str.startswith(_l50_prefixes()).all()
    # No diagcode may start with an exclusion range (O*, C*, B20, D8*).
    bad = pats["diagcode"].str.upper().str.startswith(("O", "C", "B20", "D8"))
    assert not bad.any()


def test_inpatient_carries_two_distinct_L50x_dates_for_cohortA_index():
    cfg = ClaimsDGPConfig(n_patients=50, seed=7)
    rng = np.random.default_rng(7)
    pats = generate_patients(rng, cfg)
    ip = emit_inpatient(rng, pats, cfg)
    diag_cols = [f"diag{i}" for i in range(1, 6)]
    # Every patient needs >=2 distinct L50.x admit dates so _derive_index_date
    # (:1856) returns ip_dates[1] (the 2nd).
    for pid in pats["patid"].head(10):
        a = ip[ip["patid"] == int(pid)]
        l50 = a[
            a[diag_cols].apply(
                lambda r: r.astype(str).str.upper().str.startswith(_l50_prefixes()).any(),
                axis=1,
            )
        ]
        assert l50["admit_date"].nunique() >= 2


def test_inpatient_comorbidities_scale_with_severity():
    cfg = ClaimsDGPConfig(n_patients=300, seed=9)
    rng = np.random.default_rng(9)
    pats = generate_patients(rng, cfg)
    ip = emit_inpatient(rng, pats, cfg)
    # Comorbidity admit count (non-L50.x rows) should rise with severity so the
    # converter's cci_*/elx_* + has_<comorbidity> features carry pre-index signal.
    diag_cols = [f"diag{i}" for i in range(1, 6)]
    is_l50 = ip[diag_cols].apply(
        lambda r: r.astype(str).str.upper().str.startswith(_l50_prefixes()).any(), axis=1
    )
    com = ip[~is_l50].groupby("patid").size().rename("n_com")
    merged = pats[["patid", "severity"]].merge(com, left_on="patid", right_index=True, how="left")
    merged["n_com"] = merged["n_com"].fillna(0)
    hi = merged.nlargest(50, "severity")["n_com"].mean()
    lo = merged.nsmallest(50, "severity")["n_com"].mean()
    assert hi > lo


def test_inpatient_schema_only_converter_read_columns():
    cfg = ClaimsDGPConfig(n_patients=20, seed=1)
    rng = np.random.default_rng(1)
    pats = generate_patients(rng, cfg)
    ip = emit_inpatient(rng, pats, cfg)
    # inpatientdata read sites: patid, admit_date, disch_date, diag1..5, tos_cd.
    expected = {"patid", "admit_date", "disch_date", "tos_cd"} | {f"diag{i}" for i in range(1, 6)}
    assert expected <= set(ip.columns)
    # No phantom columns the converter never reads.
    assert {"proc1", "proc2", "proc3", "clmid"} & set(ip.columns) == set()
