"""Phase 0: adherence outcomes — recoverable binary + consistent raw proxies."""
import numpy as np
import pytest

from src.ml.synthetic.dgp.adherence_outcomes import generate_adherence_outcomes


@pytest.mark.unit
def test_adherence_outcomes_recoverable_and_proxy_consistent():
    rng = np.random.default_rng(21)
    n = 5000
    severity = rng.uniform(0, 10, n)
    academic = (rng.random(n) < 0.3).astype(int)
    segment = np.where(severity > 7, "high_severity",
                       np.where(severity > 4, "medium_severity", "low_severity"))
    arm = (rng.random(n) < 0.5).astype(int)
    cate_map = {"high_severity": 0.9, "medium_severity": 0.5, "low_severity": 0.2}

    out = generate_adherence_outcomes(
        treatment_arm=arm,
        disease_severity=severity,
        academic_hcp=academic,
        segment=segment,
        cate_map=cate_map,
        rng=rng,
    )

    assert set(np.unique(out["adherent_180d"])) <= {0, 1}
    assert set(np.unique(out["low_gap_180d"])) <= {0, 1}
    assert 0.20 <= out["adherent_180d"].mean() <= 0.50

    assert out["adherence_rate"].min() >= 0.0 and out["adherence_rate"].max() <= 1.0
    assert out["gap_days"].min() >= 0.0

    agree = np.mean((out["adherence_rate"] >= 0.8) == (out["adherent_180d"] == 1))
    assert agree >= 0.80

    rd = out["adherent_rd_by_segment"]
    assert rd["high_severity"] > rd["medium_severity"] > rd["low_severity"] > 0
