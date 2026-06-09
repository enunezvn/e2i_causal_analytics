"""Shard 05 Task 3 (regression lock) — triggers.brand_id must be an EXACT brand_type
enum label. kisqali_oncologist_reach (migration 044) filters `t.brand_id = 'Kisqali'`
with no ::text cast, so a wrong literal silently reads 0."""
import pandas as pd

from src.ml.synthetic.config import Brand
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.trigger_generator import TriggerGenerator

_VALID = {b.value for b in Brand}  # {"Remibrutinib","Kisqali","Fabhalta"}


def test_brand_id_is_exact_enum_label():
    pf = pd.DataFrame(
        {
            "patient_id": [f"pt{i:05d}" for i in range(100)],
            "hcp_id": [f"hcp{i % 10:03d}" for i in range(100)],
            "journey_start_date": ["2026-05-20"] * 100,
            "treatment_initiated": [0] * 100,
            "engagement_score": [5.0] * 100,
            "brand": ["Kisqali"] * 100,
        }
    )
    gen = TriggerGenerator(GeneratorConfig(seed=3, n_records=200), patient_df=pf, hcp_df=None)
    triggers = gen.generate()
    assert set(triggers["brand_id"].unique()).issubset(_VALID)
    assert (triggers["brand_id"] == "Kisqali").all()


def test_brand_id_matches_brand_column_standalone():
    gen = TriggerGenerator(GeneratorConfig(seed=3, n_records=300, brand=Brand.FABHALTA))
    triggers = gen.generate()  # standalone path
    assert (triggers["brand_id"] == triggers["brand"]).all()
    assert set(triggers["brand_id"].unique()).issubset(_VALID)
