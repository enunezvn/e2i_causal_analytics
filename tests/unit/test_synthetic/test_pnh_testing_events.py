"""#1116 — BR-003 (Fabhalta % PNH Tested) numerator must exist in the ACTIVE generator.

Root cause of the issue: the `pnh_flow_cytometry` emission lived only in the LEGACY
``src/ml/data_generator.py`` (never invoked by the reseed entrypoint
``scripts/load_synthetic_data.py``) and in one-shot migration 046 block C, whose
seeded rows were destroyed by subsequent full regenerates. The ACTIVE
``TreatmentGenerator`` derived ``event_subtype`` solely from the brand drug_class and
never emitted PNH flow-cytometry lab events (nor ``loinc_codes``/``lab_values``,
which the BR-003 registry SQL requires) — so the live substrate carried a fully
populated 8,412-patient D59.5 denominator with a structurally-zero numerator,
rendered as a plausible-real 0.0% CRITICAL.

These tests pin the durable fix: the active generator emits one deterministic
``pnh_flow_cytometry`` lab_test per ~PNH_TESTED_PREVALENCE (0.65) of D59.5-eligible
Fabhalta patients — INDEPENDENT of treatment initiation, because BR-003's
denominator is every D59.5 journey — and the loader registers the columns the
KPI reads.
"""

import pandas as pd
import pytest

from src.ml.synthetic.clinical_codes import (
    PNH_FLOW_EVENT_SUBTYPE,
    PNH_FLOW_LOINC,
    PNH_TESTED_PREVALENCE,
)
from src.ml.synthetic.config import Brand
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.treatment_generator import TreatmentGenerator
from src.ml.synthetic.loaders.batch_loader import OPTIONAL_COLUMNS, TABLE_COLUMNS


def _patients(
    brand_value: str,
    n: int = 400,
    dx: str = "D59.5",
    initiated: int = 1,
    start: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"pt_{i:06d}" for i in range(start, start + n)],
            "patient_journey_id": [f"patient_{i:06d}" for i in range(start, start + n)],
            "brand": [brand_value] * n,
            "journey_start_date": ["2024-01-15"] * n,
            "hcp_id": ["hcp_00001"] * n,
            "treatment_initiated": [initiated] * n,
            "primary_diagnosis_code": [dx] * n,
        }
    )


def _pnh_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "event_subtype" not in df.columns or df.empty:
        return pd.DataFrame()
    return df[df["event_subtype"] == PNH_FLOW_EVENT_SUBTYPE]


def test_fabhalta_d595_cohort_gets_pnh_flow_cytometry_at_prevalence():
    """~PNH_TESTED_PREVALENCE of the D59.5 cohort get exactly one PNH lab event."""
    patients = _patients(Brand.FABHALTA.value, n=400)
    df = TreatmentGenerator(GeneratorConfig(seed=3, n_records=200), patient_df=patients).generate()
    pnh = _pnh_rows(df)
    assert len(pnh) > 0, "no pnh_flow_cytometry events emitted (BR-003 numerator missing)"
    # One event per tested patient (patient-based KPI; mirrors migration 046 DISTINCT ON)
    assert pnh["patient_id"].nunique() == len(pnh)
    rate = pnh["patient_id"].nunique() / patients["patient_id"].nunique()
    assert abs(rate - PNH_TESTED_PREVALENCE) < 0.07, (
        f"tested rate {rate:.3f} far from intended {PNH_TESTED_PREVALENCE}"
    )


def test_pnh_rows_carry_the_columns_the_kpi_sql_reads():
    """BR-003 SQL: event_subtype='pnh_flow_cytometry' AND loinc_codes && <real PNH LOINCs>."""
    patients = _patients(Brand.FABHALTA.value, n=200)
    df = TreatmentGenerator(GeneratorConfig(seed=3, n_records=100), patient_df=patients).generate()
    pnh = _pnh_rows(df)
    assert len(pnh) > 0
    assert (pnh["treatment_type"] == "lab_test").all()
    assert pnh["loinc_codes"].map(lambda a: len(a) == 1 and a[0] in PNH_FLOW_LOINC).all()
    assert (
        pnh["lab_values"]
        .map(lambda v: v["assay"] == "PNH_clone" and 0.0 <= v["value"] <= 95.0 and v["unit"] == "%")
        .all()
    )
    # Diagnostic lab rows must NOT read as brand-drug prescriptions (the migration-086
    # TRx-contamination lesson): no drug coding, non-prescription event type.
    assert pnh["drug_class"].isna().all()
    assert pnh["drug_ndc"].isna().all()
    # dx context: the real text[] column mirrors the D59.5 eligibility anchor
    assert pnh["icd_codes"].map(lambda a: a == ["D59.5"]).all()


def test_pnh_membership_is_deterministic_and_reseed_idempotent():
    """Same patient -> same tested flag, same PK, same payload across runs
    (uuid4-PK reseed non-idempotency incident guard: PK upsert must overwrite,
    never accumulate)."""
    patients = _patients(Brand.FABHALTA.value, n=300)
    a = _pnh_rows(
        TreatmentGenerator(GeneratorConfig(seed=3, n_records=100), patient_df=patients).generate()
    )
    b = _pnh_rows(
        TreatmentGenerator(GeneratorConfig(seed=99, n_records=100), patient_df=patients).generate()
    )
    # Membership + IDs + payload are a stable property of patient_id, seed-independent
    assert sorted(a["treatment_event_id"]) == sorted(b["treatment_event_id"])
    a_sorted = a.sort_values("treatment_event_id").reset_index(drop=True)
    b_sorted = b.sort_values("treatment_event_id").reset_index(drop=True)
    pd.testing.assert_series_equal(a_sorted["loinc_codes"], b_sorted["loinc_codes"])
    pd.testing.assert_series_equal(a_sorted["lab_values"], b_sorted["lab_values"])
    # Deterministic per-patient PK (VARCHAR(30) on the DB column)
    assert a["treatment_event_id"].str.startswith("pnh_").all()
    assert a["treatment_event_id"].str.len().max() <= 30


def test_pnh_emission_covers_non_initiated_patients_too():
    """BR-003's denominator is EVERY D59.5 journey; the diagnostic test precedes and is
    independent of Fabhalta initiation. A 0-initiated cohort must still be tested."""
    patients = _patients(Brand.FABHALTA.value, n=300, initiated=0)
    df = TreatmentGenerator(GeneratorConfig(seed=3, n_records=100), patient_df=patients).generate()
    pnh = _pnh_rows(df)
    assert len(pnh) > 0, "PNH testing must not be gated on treatment_initiated"
    rate = pnh["patient_id"].nunique() / patients["patient_id"].nunique()
    assert abs(rate - PNH_TESTED_PREVALENCE) < 0.08


def test_pnh_emission_requires_d595_eligibility():
    """Eligibility is DERIVED from the real D59.5 dx (BR-003 contract) — other brands
    and non-D59.5 Fabhalta journeys are never 'tested'."""
    kisqali = _patients(Brand.KISQALI.value, n=100, dx="C50.9")
    fab_wrong_dx = _patients(Brand.FABHALTA.value, n=100, dx="C50.9", start=100)
    for pdf in (kisqali, fab_wrong_dx):
        df = TreatmentGenerator(GeneratorConfig(seed=3, n_records=100), patient_df=pdf).generate()
        assert len(_pnh_rows(df)) == 0


def test_pnh_emission_skipped_without_diagnosis_column():
    """Without primary_diagnosis_code the eligible cohort is underivable -> emit
    nothing (never guess eligibility from brand alone)."""
    patients = _patients(Brand.FABHALTA.value, n=100).drop(columns=["primary_diagnosis_code"])
    df = TreatmentGenerator(GeneratorConfig(seed=3, n_records=100), patient_df=patients).generate()
    assert len(_pnh_rows(df)) == 0


def test_loader_registers_the_kpi_columns():
    """batch_loader gates unlisted columns (they silently vanish at load): the BR-003
    SQL reads loinc_codes (&& overlap) and the payload carries lab_values."""
    cols = TABLE_COLUMNS["treatment_events"]
    for c in ("loinc_codes", "lab_values"):
        assert c in cols, f"{c} not registered -> loader drops it and BR-003 stays zero"
        # Emitted only on lab-test rows; other producers of treatment_events frames
        # (e.g. injected conversion prescriptions) may omit them entirely.
        assert c in OPTIONAL_COLUMNS


@pytest.mark.parametrize("prefix", ["", "v1_"])
def test_pnh_ids_inherit_the_patient_namespace(prefix: str):
    """id_prefix namespacing (disjoint validation datasets) must carry through the
    derived PK so upserts cannot clobber the baseline."""
    patients = _patients(Brand.FABHALTA.value, n=50)
    patients["patient_id"] = prefix + patients["patient_id"]
    cfg = GeneratorConfig(seed=3, n_records=50, id_prefix=prefix)
    pnh = _pnh_rows(TreatmentGenerator(cfg, patient_df=patients).generate())
    assert len(pnh) > 0
    assert pnh["treatment_event_id"].str.startswith(f"pnh_{prefix}").all()
    assert pnh["treatment_event_id"].str.len().max() <= 30
