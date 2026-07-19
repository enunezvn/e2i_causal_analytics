"""Phase 0: PatientGenerator emits adherence columns + true_ate_by_arm."""

import numpy as np
import pytest

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator


@pytest.mark.unit
def test_generator_emits_adherence_columns_and_true_ate_by_arm():
    cfg = GeneratorConfig(
        seed=21, n_records=2000, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS
    )
    df = PatientGenerator(cfg).generate()

    for col in ("adherent_180d", "low_gap_180d", "adherence_rate", "gap_days"):
        assert col in df.columns, f"{col} missing from generated frame"
    assert df["adherent_180d"].notna().all()
    assert set(np.unique(df["adherent_180d"])) <= {0, 1}

    # Phase 1 wired copay: these are now POPULATED, not placeholders (Task 4).
    for col in ("copay_support", "insurance_access_score"):
        assert col in df.columns
        assert df[col].notna().all(), f"{col} should be populated once copay is wired"
    assert set(np.unique(df["copay_support"])) <= {0, 1}

    # Phase 2+ columns remain NULL placeholders (so the loader carries them)
    for col in ("psp_enrolled", "rep_detailing_high", "sample_dropped"):
        assert col in df.columns
        assert df[col].isna().all()

    # per-arm ground truth for the adherence outcomes
    tba = df.attrs["true_ate_by_arm"]
    assert "treatment_arm" in tba
    assert "adherent_180d" in tba["treatment_arm"]
    assert tba["treatment_arm"]["adherent_180d"]["ate"] > 0
    assert set(tba["treatment_arm"]["adherent_180d"]["cate_by_segment"]) == {
        "high_severity",
        "medium_severity",
        "low_severity",
    }
