"""patient_generator emits disc/persist columns with band-valid prevalence,
consuming the Shard-03 canonical treatment_arm + segment (no second arm source)."""
from __future__ import annotations

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.patient_generator import PatientGenerator


def test_generator_emits_cohort_outcome_columns():
    cfg = GeneratorConfig(seed=42, n_records=3000, brand=Brand.KISQALI,
                          dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    for col in ("discontinued_180d", "persistent_180d"):
        assert col in df.columns
        assert df[col].isin([0, 1]).all()
        assert 0.05 <= df[col].mean() <= 0.60
    # complement holds row-for-row
    assert (df["discontinued_180d"] + df["persistent_180d"] == 1).all()


def test_disc_persist_present_across_brands():
    for brand in (Brand.REMIBRUTINIB, Brand.FABHALTA):
        cfg = GeneratorConfig(seed=9, n_records=2000, brand=brand,
                              dgp_type=DGPType.HETEROGENEOUS)
        df = PatientGenerator(cfg).generate()
        assert 0.05 <= df["discontinued_180d"].mean() <= 0.60
