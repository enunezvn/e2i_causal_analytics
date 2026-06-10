"""Task 03.4 — wire treatment arm + per-unit tau into patient_journeys + ml_predictions."""

import numpy as np

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator
from src.ml.synthetic.generators.prediction_generator import PredictionGenerator


def test_patient_frame_carries_arm_segment_tau():
    cfg = GeneratorConfig(
        seed=11, n_records=1000, brand=Brand.KISQALI, dgp_type=DGPType.HETEROGENEOUS
    )
    df = PatientGenerator(cfg).generate()
    for col in (
        "treatment_arm",
        "segment_assignment",
        "treatment_effect_estimate",
        "propensity_score",
    ):
        assert col in df.columns, f"missing {col}"
    assert set(np.unique(df["treatment_arm"])).issubset({0, 1})
    assert df["treatment_arm"].sum() >= 30 and (len(df) - df["treatment_arm"].sum()) >= 100
    # prevalence band (INDEX) for the wired DGP
    assert 0.20 <= df["treatment_initiated"].mean() <= 0.50
    # per-unit tau is the RECOVERABLE RD-scale segment CATE: exactly 3 distinct
    # values, ordered high>medium>low>0, and Kisqali-scaled (distinct per brand).
    # (NOT the latent {0.70,0.42,0.21} — binarization attenuates to the RD scale.)
    taus = sorted(df["treatment_effect_estimate"].dropna().unique())
    assert len(taus) == 3, taus
    assert taus[0] > 0 and taus[0] < taus[1] < taus[2]
    # de-confounded RD scale is well below the latent Kisqali high CATE 0.70
    assert taus[2] < 0.70
    # tie tau back to segment ordering: high segment carries the largest tau
    high = df[df["segment_assignment"] == "high_severity"]["treatment_effect_estimate"].iloc[0]
    low = df[df["segment_assignment"] == "low_severity"]["treatment_effect_estimate"].iloc[0]
    assert high == taus[2] and low == taus[0]


def test_predictions_inherit_causal_cols():
    cfg = GeneratorConfig(
        seed=11, n_records=300, brand=Brand.KISQALI, dgp_type=DGPType.HETEROGENEOUS
    )
    pdf = PatientGenerator(cfg).generate()
    preds = PredictionGenerator(GeneratorConfig(seed=11, n_records=300), patient_df=pdf).generate()
    for col in ("treatment_effect_estimate", "heterogeneous_effect", "segment_assignment"):
        assert col in preds.columns
    assert preds["heterogeneous_effect"].notna().all()
