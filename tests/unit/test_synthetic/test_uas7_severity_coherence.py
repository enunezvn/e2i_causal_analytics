"""Remibrutinib UAS7 follows the latent severity that cuts the severity tier (2026-09-16).

Before: UAS7 ~ uniform 16..42 independent of disease_severity, so the live mean UAS7
was 28.9 / 28.9 / 29.2 across high / medium / low tiers and "high severity" said
nothing about uncontrolled urticaria. See src/ml/synthetic/dgp/clinical_severity.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.config import Brand
from src.ml.synthetic.dgp import clinical_severity as cs
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.patient_generator import PatientGenerator

_N = 6000


@pytest.fixture(scope="module")
def remi() -> pd.DataFrame:
    return PatientGenerator(
        GeneratorConfig(n_records=_N, seed=42, brand=Brand.REMIBRUTINIB)
    ).generate()


def test_mean_uas7_rises_with_the_severity_tier(remi):
    means = remi.groupby("segment_assignment")["urticaria_severity_uas7"].mean()
    assert means["high_severity"] > means["medium_severity"] + 2.0
    assert means["medium_severity"] > means["low_severity"] + 2.0


def test_correlation_is_the_designed_rho(remi):
    corr = np.corrcoef(remi["urticaria_severity_uas7"].astype(float), remi["disease_severity"])[
        0, 1
    ]
    assert corr > 0.25, corr  # a literal floor: the constant alone cannot witness itself
    assert abs(corr - cs.UAS7_SEVERITY_RHO) < 0.1, corr


def test_marginal_and_range_are_unchanged(remi):
    """The uncontrolled-CSU axis (UAS7 >= 28) keeps its ~15/27 prevalence, and the
    cohort inclusion floor (UAS7 >= 16) still holds for every row."""
    u = remi["urticaria_severity_uas7"]
    assert u.between(16, 42).all()
    assert abs((u >= 28).mean() - 15 / 27) < 0.03


def test_every_tier_keeps_both_axis_arms(remi):
    """Positivity for the UAS7 >= 28 persistence axis inside each tier (the
    rho=0.8 failure: the high tier had 98.5% uncontrolled, no controls left)."""
    p = remi.groupby("segment_assignment")["urticaria_severity_uas7"].apply(
        lambda s: (s >= 28).mean()
    )
    assert p.between(0.05, 0.95).all(), p.to_dict()


def test_rho_zero_is_the_identity_on_the_draw():
    draws = np.arange(16, 43)
    for sev in (0.0, 5.0, 10.0):
        assert np.array_equal(cs.uas7_from_severity(draws, np.full(27, sev), rho=0.0), draws)


def test_mapping_is_monotone_in_severity_and_draw():
    draws = np.arange(16, 43)
    lo = cs.uas7_from_severity(draws, np.full(27, 2.0))
    hi = cs.uas7_from_severity(draws, np.full(27, 8.0))
    assert (hi >= lo).all() and (hi > lo).any()
    assert (np.diff(cs.uas7_from_severity(draws, np.full(27, 5.0))) >= 0).all()


@pytest.mark.parametrize("bad", [15, 43])
def test_draw_outside_the_range_is_rejected(bad):
    with pytest.raises(ValueError):
        cs.uas7_from_severity(np.array([bad]), np.array([5.0]))


def test_only_uas7_and_its_axis_labels_move(remi, monkeypatch):
    """The copula consumes no RNG: versus rho=0 (the pre-2026-09-16 generator, byte
    for byte), the ONLY columns that change are UAS7 and the two persistence labels
    its uncontrolled-CSU axis drives."""
    monkeypatch.setattr(cs, "UAS7_SEVERITY_RHO", 0.0)
    before = PatientGenerator(
        GeneratorConfig(n_records=_N, seed=42, brand=Brand.REMIBRUTINIB)
    ).generate()
    changed = {c for c in before.columns if not before[c].equals(remi[c])}
    assert changed == {"urticaria_severity_uas7", "persistent_180d", "discontinued_180d"}


def test_other_brands_carry_no_uas7():
    df = PatientGenerator(GeneratorConfig(n_records=600, seed=7, brand=Brand.KISQALI)).generate()
    assert df["urticaria_severity_uas7"].isna().all()
