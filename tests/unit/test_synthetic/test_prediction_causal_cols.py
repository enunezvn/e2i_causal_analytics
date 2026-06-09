"""ml_predictions must carry the 3 causal columns and they must be loader-registered."""
import pandas as pd
from src.ml.synthetic.generators import (
    GeneratorConfig, HCPGenerator, PatientGenerator, PredictionGenerator,
)
from src.ml.synthetic.loaders import TABLE_COLUMNS

CAUSAL_COLS = ["treatment_effect_estimate", "heterogeneous_effect", "segment_assignment"]


def test_causal_cols_registered_in_loader():
    for c in CAUSAL_COLS:
        assert c in TABLE_COLUMNS["ml_predictions"], f"{c} not loader-registered -> stripped"


def test_generator_emits_causal_cols():
    cfg = GeneratorConfig(n_records=200, seed=42)
    hcp = HCPGenerator(cfg).generate()
    pts = PatientGenerator(GeneratorConfig(n_records=100, seed=42), hcp_df=hcp).generate()
    df = PredictionGenerator(GeneratorConfig(n_records=200, seed=42), patient_df=pts).generate()
    for c in CAUSAL_COLS:
        assert c in df.columns, f"PredictionGenerator dropped {c}"
