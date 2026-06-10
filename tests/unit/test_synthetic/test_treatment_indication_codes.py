"""Shard 04 Task 4 — treatment_events must carry indication-correct codes per brand
(dx / NDC / drug_class) and the loader must register the real DB code columns.

REASON-BEFORE-RULES correction to the shard plan: treatment_events has NO scalar
`primary_diagnosis_code` column (verified against the faithful docker DB — it carries
`icd_codes text[]`). So we register `icd_codes` (the real dx column) for the loader,
expose the scalar `primary_diagnosis_code` on the generator frame for Shard 05/06
joins + these tests, and intentionally do NOT register the scalar (registering it
would make the loader send it -> 42703 undefined_column on insert).
"""

import pandas as pd

from src.ml.synthetic.config import Brand
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.treatment_generator import TreatmentGenerator
from src.ml.synthetic.loaders.batch_loader import TABLE_COLUMNS


def _patients(brand_value: str, n: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"pt_{i:06d}" for i in range(n)],
            "patient_journey_id": [f"patient_{i:06d}" for i in range(n)],
            "brand": [brand_value] * n,
            "journey_start_date": ["2024-01-15"] * n,
            "hcp_id": ["hcp_00001"] * n,
            "treatment_initiated": [1] * n,
        }
    )


def test_treatment_events_carry_brand_correct_codes():
    cfg = GeneratorConfig(seed=3, n_records=80)
    df = TreatmentGenerator(cfg, patient_df=_patients(Brand.KISQALI.value)).generate()
    assert (df["drug_class"] == "CDK4/6 Inhibitor").all()
    assert (df["drug_name"] == "ribociclib").all()
    assert df["drug_ndc"].str.startswith("00078-0903").all()
    # scalar primary dx (frame-only) must come from the C50.x breast set
    assert df["primary_diagnosis_code"].str.startswith("C50").all()
    # icd_codes is the real text[] DB column; first element mirrors primary dx
    assert df["icd_codes"].map(lambda a: a[0].startswith("C50")).all()


def test_remi_and_fabhalta_get_distinct_indications():
    cfg = GeneratorConfig(seed=3, n_records=60)
    remi = TreatmentGenerator(cfg, patient_df=_patients(Brand.REMIBRUTINIB.value)).generate()
    fab = TreatmentGenerator(cfg, patient_df=_patients(Brand.FABHALTA.value)).generate()
    assert (remi["drug_class"] == "BTK Inhibitor").all()
    assert remi["primary_diagnosis_code"].str.startswith("L50").all()
    assert (fab["drug_class"] == "Complement Inhibitor").all()
    assert (fab["primary_diagnosis_code"] == "D59.5").all()
    assert fab["icd_codes"].map(lambda a: a == ["D59.5"]).all()


def test_loader_registers_the_real_db_code_columns():
    cols = TABLE_COLUMNS["treatment_events"]
    for c in ("drug_ndc", "drug_name", "drug_class", "event_subtype", "icd_codes"):
        assert c in cols, f"{c} not registered -> loader will drop it"
    # primary_diagnosis_code is a patient_journeys scalar, NOT a treatment_events
    # column. Exposed on the generator frame for joins but deliberately unregistered
    # here so the loader does not send it (treatment_events has no such column).
    assert "primary_diagnosis_code" not in cols
