"""Unit tests for the disc/persist cohort DGP (hermetic, 0 DB rows).

Discipline: exercise the REAL DGP math on real-shaped arrays. No DB, no mocks of
the logic. Prevalence + recoverability are asserted directly on the generated
labels — the same properties gate 10 will check on the faithful DB.
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
    return rng, treatment_arm, disease_severity, academic_hcp, segment, geographic_region


def test_discontinuation_prevalence_in_band():
    rng, t, sev, acad, seg, geo = _inputs()
    out = generate_discontinuation_outcomes(
        rng=rng,
        treatment_arm=t,
        disease_severity=sev,
        academic_hcp=acad,
        segment=seg,
        geographic_region=geo,
        brand_cate_scale=1.0,
    )
    prev = out["discontinued_180d"].mean()
    assert 0.05 <= prev <= 0.60, f"disc prevalence {prev} out of [0.05,0.60]"
    assert np.array_equal(out["persistent_180d"], 1 - out["discontinued_180d"])
    assert 0.05 <= out["persistent_180d"].mean() <= 0.60


def test_treatment_reduces_discontinuation_recoverable():
    rng, t, sev, acad, seg, geo = _inputs()
    out = generate_discontinuation_outcomes(
        rng=rng,
        treatment_arm=t,
        disease_severity=sev,
        academic_hcp=acad,
        segment=seg,
        geographic_region=geo,
        brand_cate_scale=1.0,
    )
    disc = out["discontinued_180d"]
    diff = disc[t == 1].mean() - disc[t == 0].mean()
    assert diff < -0.05, f"treatment must lower discontinuation; got diff={diff}"


def test_retention_benefit_is_non_negative():
    rng, t, sev, acad, seg, geo = _inputs()
    out = generate_discontinuation_outcomes(
        rng=rng,
        treatment_arm=t,
        disease_severity=sev,
        academic_hcp=acad,
        segment=seg,
        geographic_region=geo,
        brand_cate_scale=1.0,
    )
    assert (out["retention_benefit"] >= 0).all()
    assert PERSISTENCE_RETENTION_BENEFIT_PER_SEVERITY > 0


def test_brand_scale_changes_structure():
    rng1, t, sev, acad, seg, geo = _inputs(seed=11)
    rng2, *_ = _inputs(seed=11)
    a = generate_discontinuation_outcomes(
        rng=rng1,
        treatment_arm=t,
        disease_severity=sev,
        academic_hcp=acad,
        segment=seg,
        geographic_region=geo,
        brand_cate_scale=0.6,
    )
    b = generate_discontinuation_outcomes(
        rng=rng2,
        treatment_arm=t,
        disease_severity=sev,
        academic_hcp=acad,
        segment=seg,
        geographic_region=geo,
        brand_cate_scale=1.4,
    )
    assert a["discontinued_180d"].mean() != b["discontinued_180d"].mean()


def test_region_drives_discontinuation():
    rng, t, sev, acad, seg, geo = _inputs(n=8000)
    out = generate_discontinuation_outcomes(
        rng=rng,
        treatment_arm=t,
        disease_severity=sev,
        academic_hcp=acad,
        segment=seg,
        geographic_region=geo,
        brand_cate_scale=1.0,
    )
    disc = out["discontinued_180d"]
    # west has the highest positive region pull (+0.9), midwest the most negative (-0.9)
    assert disc[geo == "west"].mean() > disc[geo == "midwest"].mean()
