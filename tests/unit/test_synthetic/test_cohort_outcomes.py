"""Unit tests for the disc/persist cohort DGP (hermetic, 0 DB rows).

Discipline: exercise the REAL DGP math on real-shaped arrays. No DB, no mocks of
the logic. Prevalence + recoverability are asserted directly on the generated
labels — the same properties gate 10 will check on the faithful DB.

T9 (2026-06-21): the equation now carries 4 prognostic drivers (insurance,
comorbidity burden, age, prior-therapy lines) drawn INDEPENDENTLY of treatment so
they raise predictive signal without disturbing the recoverable treatment effect.
"""

from __future__ import annotations

import numpy as np

from src.ml.synthetic.generators.cohort_outcomes import (
    PERSISTENCE_RETENTION_BENEFIT_PER_SEVERITY,
    generate_discontinuation_outcomes,
)


def _inputs(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    treatment_arm = rng.integers(0, 2, n)  # Shard 03 per-unit arm
    disease_severity = np.clip(rng.normal(5.0, 2.0, n), 0, 10)
    academic_hcp = (rng.random(n) < 0.30).astype(int)
    segment = np.where(
        disease_severity > 7,
        "high_severity",
        np.where(disease_severity > 4, "medium_severity", "low_severity"),
    )
    geographic_region = rng.choice(["midwest", "northeast", "south", "west"], n)
    # NEW prognostic drivers — drawn INDEPENDENTLY of treatment_arm.
    insurance_type = rng.choice(["commercial", "medicare", "medicaid"], n, p=[0.6, 0.3, 0.1])
    age_at_diagnosis = rng.integers(18, 85, n)
    comorbidity_burden = rng.poisson(1.3, n).clip(0, 5)
    prior_therapy_lines = rng.integers(0, 4, n)
    return {
        "rng": rng,
        "treatment_arm": treatment_arm,
        "disease_severity": disease_severity,
        "academic_hcp": academic_hcp,
        "segment": segment,
        "geographic_region": geographic_region,
        "insurance_type": insurance_type,
        "age_at_diagnosis": age_at_diagnosis,
        "comorbidity_burden": comorbidity_burden,
        "prior_therapy_lines": prior_therapy_lines,
        "brand_cate_scale": 1.0,
    }


def test_discontinuation_prevalence_in_band():
    out = generate_discontinuation_outcomes(**_inputs())
    prev = out["discontinued_180d"].mean()
    assert 0.05 <= prev <= 0.60, f"disc prevalence {prev} out of [0.05,0.60]"
    assert np.array_equal(out["persistent_180d"], 1 - out["discontinued_180d"])
    assert 0.05 <= out["persistent_180d"].mean() <= 0.60


def test_treatment_reduces_discontinuation_recoverable():
    out = generate_discontinuation_outcomes(**_inputs())
    disc = out["discontinued_180d"]
    t = _inputs()["treatment_arm"]  # same seed => identical arm
    diff = disc[t == 1].mean() - disc[t == 0].mean()
    assert diff < -0.05, f"treatment must lower discontinuation; got diff={diff}"


def test_retention_benefit_is_non_negative():
    out = generate_discontinuation_outcomes(**_inputs())
    assert (out["retention_benefit"] >= 0).all()
    assert PERSISTENCE_RETENTION_BENEFIT_PER_SEVERITY > 0


def test_brand_scale_changes_structure():
    a_in = _inputs(seed=11)
    a_in["brand_cate_scale"] = 0.6
    b_in = _inputs(seed=11)
    b_in["brand_cate_scale"] = 1.4
    a = generate_discontinuation_outcomes(**a_in)
    b = generate_discontinuation_outcomes(**b_in)
    assert a["discontinued_180d"].mean() != b["discontinued_180d"].mean()


def test_region_drives_discontinuation():
    inp = _inputs(n=8000)
    out = generate_discontinuation_outcomes(**inp)
    disc = out["discontinued_180d"]
    geo = inp["geographic_region"]
    # west has the highest positive region pull (+0.9), midwest the most negative (-0.9)
    assert disc[geo == "west"].mean() > disc[geo == "midwest"].mean()


def test_commercial_insurance_improves_persistence():
    inp = _inputs(n=9000)
    out = generate_discontinuation_outcomes(**inp)
    disc, ins = out["discontinued_180d"], inp["insurance_type"]
    # commercial = best access => lowest discontinuation; medicaid the highest.
    assert disc[ins == "commercial"].mean() < disc[ins == "medicaid"].mean()


def test_comorbidity_burden_increases_discontinuation():
    inp = _inputs(n=9000)
    out = generate_discontinuation_outcomes(**inp)
    disc, com = out["discontinued_180d"], inp["comorbidity_burden"]
    assert disc[com >= 3].mean() > disc[com == 0].mean()


def test_prior_therapy_increases_discontinuation():
    inp = _inputs(n=9000)
    out = generate_discontinuation_outcomes(**inp)
    disc, pr = out["discontinued_180d"], inp["prior_therapy_lines"]
    assert disc[pr >= 2].mean() > disc[pr == 0].mean()


def test_drivers_do_not_disturb_treatment_effect():
    # Drivers are prognostic-only: treatment must still lower discontinuation.
    inp = _inputs(n=9000)
    out = generate_discontinuation_outcomes(**inp)
    disc, t = out["discontinued_180d"], inp["treatment_arm"]
    diff = disc[t == 1].mean() - disc[t == 0].mean()
    assert diff < -0.05, f"treatment must still lower discontinuation; got {diff}"
