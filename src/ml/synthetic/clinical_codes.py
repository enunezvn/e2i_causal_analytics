"""Canonical real-world clinical codes for the synthetic RWD generator (#577).

Single source of truth for the brand→indication coding used by
``src/ml/data_generator.py`` and the #577 seed migrations. Every code here is a
REAL ontology code resolved/validated during the #577 investigation (RxNav,
NLM LOINC, ICD-10-CM, EAACI urticaria guideline) — never an invented placeholder.
The whole local DB is synthetic RWD; the honest pattern is to emit the missing
clinical concepts (CSU antihistamine baseline therapy, PNH flow-cytometry testing)
using these real codes, then let the KPI calculators compute over them.

Sources:
- ICD-10-CM: config/domain_vocabulary.yaml:2401-2421, config/cohort_vocabulary.yaml.
- Antihistamine RxCUIs/ATC: RxNav /rxcui.json + /rxclass (cetirizine ATC R06AE07 also
  confirmed via ChEMBL CHEMBL1000).
- PNH flow-cytometry LOINC: NLM Clinical Tables LOINC API (the commonly-assumed
  56659-3 is NOT a real LOINC — 0 hits — so the real panel/FLAER/CD55-CD59 codes are
  used instead).
- CSU "uncontrolled" threshold UAS7>=7: EAACI/GA2LEN/EuroGuiDerm/APAAACI guideline,
  Zuberbier 2022 Allergy 77(3):734-766, PMID 34536239 (UAS7 range 0-42).
"""

from __future__ import annotations

# --- Brand -> primary diagnosis (ICD-10-CM) ---------------------------------------
# Remibrutinib = chronic spontaneous urticaria (CSU); Fabhalta = paroxysmal nocturnal
# hemoglobinuria (PNH); Kisqali = HR+ breast cancer.
BRAND_DIAGNOSIS: dict[str, dict[str, object]] = {
    "Remibrutinib": {
        "icd10": ["L50.1", "L50.8", "L50.9"],  # L50.1 idiopathic (CSU) primary
        "desc": "Chronic spontaneous urticaria",
    },
    "Fabhalta": {
        "icd10": ["D59.5"],  # PNH [Marchiafava-Micheli]
        "desc": "Paroxysmal nocturnal hemoglobinuria",
    },
    "Kisqali": {
        "icd10": ["C50.1", "C50.2", "C50.9"],  # malignant neoplasm of breast
        "desc": "Malignant neoplasm of breast",
    },
}

# --- Brand -> mechanism drug_class (fixes the brand-blind random.choice corruption) ---
BRAND_DRUG_CLASS: dict[str, str] = {
    "Remibrutinib": "BTK Inhibitor",
    "Fabhalta": "Complement Inhibitor",
    "Kisqali": "CDK4/6 Inhibitor",
}

# --- H1-antihistamine baseline therapy (CSU first-line) -----------------------------
# ATC R06A "Antihistamines for systemic use" is the drug_class anchor; RxCUIs identify
# the specific agents.
ANTIHISTAMINE_ATC_CLASS = "R06A"
ANTIHISTAMINES: list[dict[str, str]] = [
    {"name": "cetirizine", "rxcui": "20610"},
    {"name": "fexofenadine", "rxcui": "87636"},
    {"name": "loratadine", "rxcui": "28889"},
    {"name": "desloratadine", "rxcui": "275635"},
]

# --- CSU disease activity (UAS7) ----------------------------------------------------
UAS7_ASSAY = "UAS7"
UAS7_MAX = 42
# "Uncontrolled" per the governing urticaria guideline (PMID 34536239).
UAS7_UNCONTROLLED_THRESHOLD = 7
# Target synthetic prevalence of uncontrolled-on-antihistamine CSU patients. A
# product/demo parameter (consistent with real CSU literature where a large share
# remain uncontrolled on antihistamines, the rationale for a BTK inhibitor); the KPI
# COMPUTES the realized fraction from the generated rows, it is not hardcoded.
UAS7_UNCONTROLLED_PREVALENCE = 0.45

# --- PNH flow-cytometry diagnostic test (LOINC) -------------------------------------
# Real PNH-flow LOINC; 56659-3 (commonly assumed) is NOT a real LOINC and is excluded.
PNH_FLOW_LOINC: list[str] = [
    "55164-8",  # Paroxysmal nocturnal panel - Blood (order-level)
    "35468-8",  # FLAER cells [Presence] in Blood (FLAER, gold standard)
    "90735-2",  # PNH GPI-Linked WBC and RBC Ag Panel - Blood
    "44007-3",  # CD55 & CD59 RBC interpretation
]
PNH_FLOW_EVENT_SUBTYPE = "pnh_flow_cytometry"
ANTIHISTAMINE_EVENT_SUBTYPE = "baseline_antihistamine"
# Target synthetic share of PNH-eligible (D59.5) patients who received a PNH test.
PNH_TESTED_PREVALENCE = 0.65
