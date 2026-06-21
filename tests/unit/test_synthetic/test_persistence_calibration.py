"""Measure-don't-assume gate for the enriched persistence DGP (T9).

The enriched discontinuation/persistence equation must achieve a realistic
~0.78-0.82 holdout AUC per brand with prevalence in the designed [0.05, 0.60]
band. These asserted numbers LOCK the cohort_outcomes coefficients (the same way
the 2026-06-14 feature experiment locked KEEP_COLUMNS): if a future edit weakens or
inflates the signal, this gate fails. Measured live (2026-06-21): Remi 0.804 /
Fabhalta 0.796 / Kisqali 0.805, prevalence ~0.49.

Hermetic: generates in-memory frames, fits a LogisticRegression on the 7 KEEP_COLUMNS
(encoded like FeatureBuilder). No DB, no mocks.
"""

from __future__ import annotations

import pytest

from src.ml.synthetic.config import Brand
from src.ml.synthetic.dgp.recovery_probe import (
    measure_persistence_signal,
    recover_ate_and_cate,
)
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.patient_generator import PatientGenerator


def _frame(brand: Brand, n: int = 12000, seed: int = 42):
    return PatientGenerator(
        GeneratorConfig(n_records=n, seed=seed, brand=brand)
    ).generate()


@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_persistence_auc_in_target_band(brand):
    m = measure_persistence_signal(_frame(brand))
    assert 0.05 <= m["prevalence"] <= 0.60, f"{brand.value}: prevalence {m['prevalence']} out of band"
    assert 0.77 <= m["holdout_auc"] <= 0.83, (
        f"{brand.value}: AUC {m['holdout_auc']:.4f} out of realistic [0.77, 0.83] band"
    )


def test_brands_vary():
    aucs = [measure_persistence_signal(_frame(b))["holdout_auc"] for b in Brand]
    assert max(aucs) - min(aucs) > 0.005, f"brands should differ in AUC; got {aucs}"


def test_ate_cate_recovery_unchanged_by_drivers():
    """Invariant gate: the new prognostic drivers are independent of treatment_arm, so
    the recoverable treatment effect (ATE) + segment heterogeneity (CATE ordering) are
    preserved end-to-end. Measured (seed=42, n=6000): true_ate 0.171, recovered 0.197,
    CATE high 0.266 >= med 0.233 >= low 0.093."""
    df = _frame(Brand.REMIBRUTINIB, n=6000)
    rec = recover_ate_and_cate(df)
    true_ate = float(df.attrs["true_ate"])
    assert abs(rec["linear_dml_ate"] - true_ate) < 0.10, (
        f"ATE drifted: recovered {rec['linear_dml_ate']:.4f} vs true {true_ate:.4f}"
    )
    cate = rec["cate_by_segment_estimate"]
    assert cate["high_severity"] >= cate["medium_severity"] >= cate["low_severity"], cate
