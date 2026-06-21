"""patient_generator emits disc/persist columns with band-valid prevalence,
consuming the Shard-03 canonical treatment_arm + segment (no second arm source)."""

from __future__ import annotations

import numpy as np

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.patient_generator import PatientGenerator


def _gen(n=6000, seed=3, brand=Brand.REMIBRUTINIB):
    cfg = GeneratorConfig(seed=seed, n_records=n, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    return PatientGenerator(cfg).generate()


def test_new_driver_columns_present_and_varied():
    df = _gen()
    for col in ("comorbidity_burden", "prior_therapy_lines"):
        assert col in df.columns, f"{col} missing from generated frame"
        assert df[col].notna().all(), f"{col} has nulls"
        assert df[col].nunique() > 1, f"{col} has no per-patient variance"


def test_drivers_independent_of_treatment_arm():
    df = _gen()
    # Prognostic-only contract: |corr(driver, treatment_arm)| must be ~0.
    for col in ("comorbidity_burden", "prior_therapy_lines", "age_at_diagnosis"):
        corr = np.corrcoef(df[col].to_numpy(float), df["treatment_arm"].to_numpy(float))[0, 1]
        assert abs(corr) < 0.05, f"{col} must be independent of treatment_arm; corr={corr}"


def test_persistence_carries_driver_signal():
    df = _gen(n=12000)
    # Commercial insurance should persist more than medicaid (signal wired through).
    p = df["persistent_180d"]
    ins = df["insurance_type"]
    assert p[ins == "commercial"].mean() > p[ins == "medicaid"].mean()


def test_generator_emits_cohort_outcome_columns():
    cfg = GeneratorConfig(
        seed=42, n_records=3000, brand=Brand.KISQALI, dgp_type=DGPType.HETEROGENEOUS
    )
    df = PatientGenerator(cfg).generate()
    for col in ("discontinued_180d", "persistent_180d"):
        assert col in df.columns
        assert df[col].isin([0, 1]).all()
        assert 0.05 <= df[col].mean() <= 0.60
    # complement holds row-for-row
    assert (df["discontinued_180d"] + df["persistent_180d"] == 1).all()


def test_disc_persist_present_across_brands():
    for brand in (Brand.REMIBRUTINIB, Brand.FABHALTA):
        cfg = GeneratorConfig(seed=9, n_records=2000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
        df = PatientGenerator(cfg).generate()
        assert 0.05 <= df["discontinued_180d"].mean() <= 0.60
