"""Phase 0: adherence outcomes — recoverable binaries + EXACTLY consistent proxies."""
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

    # Both curated binaries are in-band and 0/1.
    assert set(np.unique(out["adherent_180d"])) <= {0, 1}
    assert set(np.unique(out["low_gap_180d"])) <= {0, 1}
    assert 0.20 <= out["adherent_180d"].mean() <= 0.50
    assert 0.20 <= out["low_gap_180d"].mean() <= 0.50

    assert out["adherence_rate"].min() >= 0.0 and out["adherence_rate"].max() <= 1.0
    assert out["gap_days"].min() >= 0.0

    # HIGH-3: the STORED proxy must NEVER contradict the STORED binary — exact, by
    # construction (snapped), over the SAME generated frame. Not approximate.
    assert np.all((out["adherence_rate"] >= 0.8) == (out["adherent_180d"] == 1))
    assert np.all((out["gap_days"] <= 30.0) == (out["low_gap_180d"] == 1))

    # Recoverable per-segment RD ground truth for BOTH curated outcomes.
    rd = out["adherent_rd_by_segment"]
    assert rd["high_severity"] > rd["medium_severity"] > rd["low_severity"] > 0
    rd_lg = out["low_gap_rd_by_segment"]
    assert rd_lg["high_severity"] > rd_lg["medium_severity"] > rd_lg["low_severity"] > 0
