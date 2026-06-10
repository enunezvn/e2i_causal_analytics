"""P1.2 — medication / procedure / lab / provider emit (Shard 10).

Encodes SOURCE-VERIFIED converter read sites + the label-recovery DGP:

* ``_csu_biologic_mask`` (:1969) fires on XOLAIR/DUPIXENT brands,
  omalizumab/dupilumab generics, J2357/J0517 HCPCS, 50242/00024/0024 NDC
  prefixes. Biologic rows MUST use a recognised value or cohorts B/C are empty.
* ``treatment_initiated`` == ``initiated_biologic_180d`` (:3196,:3193) is ANY
  CSU biologic fill in (index, index+180]. ``response_propensity`` (latent z)
  gates that fill, while pre-index ``severity``/comorbidity admits (same z) are
  the recoverable features.
* disc/persistence labels read the POST-index fill gap structure; the gap is
  driven by ``adherence_propensity`` (same z, opposite sign), so pre-index
  features statistically predict the post-index target (Fact #3).
* Lab provenance honesty: claims labs carry LOINC results, NOT serum-IgE PRO
  scores. The DGP suppresses an IgE-total-as-PRO feature.
"""

import numpy as np
import pandas as pd

from src.ml.synthetic.claims.claims_events import (
    emit_lab,
    emit_medication,
    emit_procedure,
    emit_provider,
)
from src.ml.synthetic.claims.config import ClaimsDGPConfig
from src.ml.synthetic.claims.patient_state import generate_patients


def _setup(n=200, seed=4):
    cfg = ClaimsDGPConfig(n_patients=n, seed=seed)
    rng = np.random.default_rng(seed)
    pats = generate_patients(rng, cfg)
    return rng, pats, cfg


def test_biologic_fill_uses_recognized_csu_brand_and_ndc():
    rng, pats, cfg = _setup()
    med = emit_medication(rng, pats, cfg)
    bio = med[med["Brand_Name"].astype(str).str.upper().str.contains("XOLAIR|DUPIXENT")]
    assert len(bio) > 0, "no biologic fills emitted -> cohorts B/C would be empty"
    # NDC codes must hit a recognised prefix so _csu_biologic_mask fires.
    assert bio["code"].astype(str).str.startswith(("50242", "00024", "0024")).all()
    assert (bio["Generic_Name"].astype(str).str.lower().isin(("omalizumab", "dupilumab"))).all()
    # Verified medication read sites present.
    assert {
        "patid",
        "medication_date",
        "npi",
        "code",
        "days_sup",
        "strength",
        "Brand_Name",
        "Generic_Name",
    } <= set(med.columns)


def test_prior_therapy_is_pre_index_and_non_biologic():
    rng, pats, cfg = _setup()
    med = emit_medication(rng, pats, cfg)
    prior = med[~med["Brand_Name"].astype(str).str.upper().str.contains("XOLAIR|DUPIXENT")]
    assert len(prior) > 0
    # Prior-therapy generics are NON_TARGET_DRUG_CLASSES the converter scores
    # into pre-index *_fill_count / *_days_supply_total features.
    assert (
        prior["Generic_Name"]
        .astype(str)
        .str.lower()
        .isin(("cetirizine", "loratadine", "hydroxyzine", "montelukast"))
        .all()
    )
    # Every prior fill is strictly before its patient's claim index.
    m = med.merge(pats[["patid", "claim_index"]], on="patid")
    prior_m = m[~m["Brand_Name"].astype(str).str.upper().str.contains("XOLAIR|DUPIXENT")]
    assert (prior_m["medication_date"] < prior_m["claim_index"]).all()


def test_biologic_fills_are_post_index_for_responders():
    rng, pats, cfg = _setup(n=400)
    med = emit_medication(rng, pats, cfg)
    m = med.merge(pats[["patid", "claim_index"]], on="patid")
    bio = m[m["Brand_Name"].astype(str).str.upper().str.contains("XOLAIR|DUPIXENT")]
    # First biologic fill must be at/after index (post-index initiation).
    first = bio.groupby("patid")["medication_date"].min()
    cidx = pats.set_index("patid")["claim_index"]
    aligned = first.index.intersection(cidx.index)
    assert (first.loc[aligned].values >= cidx.loc[aligned].values).all()


def test_higher_response_propensity_initiates_more_often():
    rng, pats, cfg = _setup(n=600)
    med = emit_medication(rng, pats, cfg)
    bio_pids = set(
        med[med["Brand_Name"].astype(str).str.upper().str.contains("XOLAIR|DUPIXENT")]["patid"]
    )
    pats = pats.copy()
    pats["initiated"] = pats["patid"].isin(bio_pids).astype(int)
    hi = pats.nlargest(150, "response_propensity")["initiated"].mean()
    lo = pats.nsmallest(150, "response_propensity")["initiated"].mean()
    assert hi > lo, "response_propensity does not drive initiation -> no recoverable signal"


def test_lab_suppresses_rwd_missing_features_and_has_read_columns():
    rng, pats, cfg = _setup()
    lab = emit_lab(rng, pats, cfg)
    # Provenance honesty: no serum-IgE PRO score smuggled in as a LOINC.
    assert "IGE_TOTAL" not in set(lab.get("loinc_cd", pd.Series(dtype=object)).astype(str))
    # Verified lab read sites.
    assert {"patid", "fst_dt", "loinc_cd", "rslt_nbr", "abnl_cd", "tst_desc"} <= set(lab.columns)


def test_procedure_has_npi_for_hcp_graph():
    rng, pats, cfg = _setup()
    proc = emit_procedure(rng, pats, cfg)
    # proc.npi feeds the med.npi ∪ proc.npi shared-patient graph (Fact #4).
    assert {"patid", "proc_date", "proc_code", "npi"} <= set(proc.columns)
    assert proc["npi"].notna().all()


def test_provider_emits_npi_to_taxonomy_only():
    rng, pats, cfg = _setup()
    med = emit_medication(rng, pats, cfg)
    proc = emit_procedure(rng, pats, cfg)
    prov = emit_provider(rng, med, proc, cfg)
    # The sole provider read sites are npi + taxonomy1 (:1550-1553).
    assert set(prov.columns) == {"npi", "taxonomy1"}
    # Every npi appearing in med/proc must resolve to a taxonomy.
    used = set(med["npi"].astype(str)) | set(proc["npi"].astype(str))
    assert used <= set(prov["npi"].astype(str))
