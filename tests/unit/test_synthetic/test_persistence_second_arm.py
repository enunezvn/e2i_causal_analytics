"""copay_support enters the DISCONTINUATION logit (negative = better persistence).
Unlike the adherence outcomes this is a logit->Bernoulli draw, so the planted RD is
an expit difference, not a binarize_score threshold."""

import numpy as np
import pytest

from src.ml.synthetic.generators.cohort_outcomes import generate_discontinuation_outcomes


def _inputs(n=6000, seed=21):
    rng = np.random.default_rng(seed)
    severity = np.clip(rng.normal(5.0, 2.0, n), 0, 10)
    return {
        "treatment_arm": (rng.random(n) < 0.4).astype(int),
        "disease_severity": severity,
        "academic_hcp": (rng.random(n) < 0.3).astype(int),
        "geographic_region": rng.choice(["midwest", "northeast", "south", "west"], n),
        "insurance_type": rng.choice(["commercial", "medicare", "medicaid"], n),
        "age_at_diagnosis": rng.normal(55.0, 12.0, n),
        "comorbidity_burden": rng.integers(0, 5, n),
        "prior_therapy_lines": rng.integers(0, 4, n),
        "segment": np.where(
            severity > 7, "high_severity", np.where(severity > 4, "medium_severity", "low_severity")
        ),
        "brand_cate_scale": 1.0,
    }


@pytest.mark.unit
def test_copay_absent_is_identical_to_shipped():
    """No second arm -> byte-identical to the shipped persistence DGP."""
    a = generate_discontinuation_outcomes(rng=np.random.default_rng(5), **_inputs())
    b = generate_discontinuation_outcomes(
        rng=np.random.default_rng(5), copay_support=None, **_inputs()
    )
    np.testing.assert_array_equal(a["discontinued_180d"], b["discontinued_180d"])
    np.testing.assert_array_equal(a["persistent_180d"], b["persistent_180d"])


@pytest.mark.unit
def test_copay_improves_persistence_and_is_ordered():
    ins = _inputs()
    n = len(ins["disease_severity"])
    copay = (np.random.default_rng(9).random(n) < 0.35).astype(int)
    out = generate_discontinuation_outcomes(
        rng=np.random.default_rng(5), copay_support=copay, **ins
    )
    rd = out["copay_persistent_rd_by_segment"]
    # Sign: copay must IMPROVE persistence in every segment.
    assert all(v > 0 for v in rd.values()), rd
    assert rd["high_severity"] > rd["medium_severity"] > rd["low_severity"], rd
    assert 0.05 <= float(np.mean(list(rd.values()))) <= 0.15, rd


@pytest.mark.unit
def test_persistence_is_still_the_strict_complement():
    """No row may be simultaneously discontinued and persistent, with copay wired."""
    ins = _inputs()
    n = len(ins["disease_severity"])
    copay = (np.random.default_rng(9).random(n) < 0.35).astype(int)
    out = generate_discontinuation_outcomes(
        rng=np.random.default_rng(5), copay_support=copay, **ins
    )
    np.testing.assert_array_equal(out["persistent_180d"], 1 - out["discontinued_180d"])
