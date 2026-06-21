"""FAITHFUL calibration gate for the enriched persistence DGP (T9).

This is the measure-don't-assume gate, and it is FAITHFUL: it routes the generated
data through the EXACT real model pipeline — FeatureBuilder(make_patient_spec(...))
to encode the 7 leakage-safe covariates, then train_cohort_model (the calibrated
LogisticRegression the *_goldstd_lr_v1 models use), then the holdout-split AUC, the
same headline _run_one_cohort records. No hand-rolled encoder, no proxy.

The enriched equation must achieve a realistic ~0.78-0.82 holdout AUC per brand with
prevalence in the designed [0.05, 0.60] band. These asserted numbers LOCK the
cohort_outcomes coefficients (the same way the 2026-06-14 feature experiment locked
KEEP_COLUMNS). Measured live (2026-06-21, n=20000): Remi 0.809 / Fabhalta 0.814 /
Kisqali 0.805, 19 encoded features (7 covariates), disc prevalence ~0.50.

Hermetic: generates in-memory frames; no DB, no mocks. n=20000 keeps the holdout
(~1000 rows) large enough that the AUC reflects the DGP's true signal, not
small-sample noise (a 600-row holdout swings ±0.03).
"""

from __future__ import annotations

import pytest
from sklearn.metrics import roc_auc_score

from src.ml.synthetic.config import Brand
from src.ml.synthetic.dgp.recovery_probe import recover_ate_and_cate
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.patient_generator import PatientGenerator
from src.mlops.gold_standard_eval.cohort_deployer import train_cohort_model
from src.mlops.gold_standard_eval.cohort_spec import make_patient_spec
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder


def _faithful(brand: Brand, n: int = 20000, seed: int = 42) -> dict:
    """Train the REAL gold-standard model (FeatureBuilder 7-cov + calibrated LR) on
    train+validation and report the holdout-split AUC — exactly _run_one_cohort's
    headline path."""
    df = PatientGenerator(GeneratorConfig(n_records=n, seed=seed, brand=brand)).generate()
    spec = make_patient_spec("persistence", brand.value)
    fb = FeatureBuilder(spec)
    tr = df[df["data_split"].isin(("train", "validation"))]
    te = df[df["data_split"] == "holdout"]
    x_train, y_train = fb.build_from_frame(tr)
    model = train_cohort_model(spec, x_train, y_train)
    x_te = fb.transform(te)
    y_te = te[spec.label_column].astype(int).to_numpy()
    pos = list(model.classes_).index(1) if 1 in model.classes_ else 0
    auc = float(roc_auc_score(y_te, model.predict_proba(x_te.to_numpy(dtype=float))[:, pos]))
    return {
        "auc": auc,
        "disc_prev": float(df["discontinued_180d"].mean()),
        "n_features": len(fb.feature_columns),
        "n_holdout": int(len(te)),
    }


@pytest.fixture(scope="module")
def faithful() -> dict:
    return {b: _faithful(b) for b in Brand}


def test_persistence_auc_in_target_band(faithful):
    for b, m in faithful.items():
        assert 0.05 <= m["disc_prev"] <= 0.60, f"{b.value}: prevalence {m['disc_prev']} out of band"
        # 7 covariates → ~19 encoded features (was 9 for the 3-covariate model).
        assert m["n_features"] >= 15, (
            f"{b.value}: only {m['n_features']} encoded features (expected the 7-covariate set)"
        )
        assert 0.78 <= m["auc"] <= 0.83, (
            f"{b.value}: faithful holdout AUC {m['auc']:.4f} out of realistic [0.78, 0.83]"
        )


def test_brands_vary(faithful):
    aucs = [m["auc"] for m in faithful.values()]
    assert max(aucs) - min(aucs) > 0.003, f"brands should differ in AUC; got {aucs}"


def test_ate_cate_recovery_unchanged_by_drivers():
    """Invariant gate: the new prognostic drivers are independent of treatment_arm, so
    the recoverable treatment effect (ATE) + segment heterogeneity (CATE ordering) are
    preserved end-to-end. Measured (seed=42, n=6000): true_ate 0.171, recovered 0.197,
    CATE high 0.266 >= med 0.233 >= low 0.093."""
    df = PatientGenerator(
        GeneratorConfig(n_records=6000, seed=42, brand=Brand.REMIBRUTINIB)
    ).generate()
    rec = recover_ate_and_cate(df)
    true_ate = float(df.attrs["true_ate"])
    assert abs(rec["linear_dml_ate"] - true_ate) < 0.10, (
        f"ATE drifted: recovered {rec['linear_dml_ate']:.4f} vs true {true_ate:.4f}"
    )
    cate = rec["cate_by_segment_estimate"]
    assert cate["high_severity"] >= cate["medium_severity"] >= cate["low_severity"], cate
