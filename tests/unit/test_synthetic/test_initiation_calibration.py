"""FAITHFUL calibration gate for the enriched INITIATION DGP (T11).

Mirror of test_persistence_calibration.py (T9), for the third cohort. It routes the
generated data through the EXACT real model pipeline — FeatureBuilder(make_patient_spec
("initiation", ...)) to encode the 7 leakage-safe covariates, then train_cohort_model
(the calibrated LogisticRegression the *_goldstd_lr_v1 models use), then the
holdout-split AUC, the same headline _run_one_cohort records. No hand-rolled encoder.

WHY this gate exists: treatment_initiated's outcome eqn (binary_outcome_with_cate)
previously used ONLY disease_severity + academic_hcp + arm·τ + N(0,0.6) — geographic
region (one of its 3 "base" covariates) was NOT in the eqn, so the goldstd initiation
model's ~0.67 AUC was the Bayes ceiling of that thin eqn (the 2026-06-14 "more features
HURT" experiment measured THAT ceiling). T11 adds 4 prognostic drivers (insurance
access, age, comorbidity, prior-therapy) to the latent baseline via
initiation_prognostic_offset, drawn ⊥ treatment_arm so ATE/CATE recovery is preserved.

The enriched equation must achieve a realistic ~0.78-0.86 holdout AUC per brand with
prevalence ~0.35 (the prevalence-banded construction pins it). These asserted numbers
LOCK _INIT_DRIVER_SCALE (the way T9 locked cohort_outcomes). Measured (2026-06-22,
n=20000, scale=0.75): Remi 0.804 / Fabhalta 0.797 / Kisqali 0.798, init_prev 0.35,
19 encoded features (7 covariates). COMM-ARMS Phase 3 (2026-07-20) adds rep_detailing_high
+ sample_dropped to the initiation spec (9 covariates, 23 encoded features) — pre-index
causal drivers of initiation the model may legitimately observe — lifting the AUC to
Remi 0.846 / Fabhalta 0.829 / Kisqali 0.846 (ceiling re-based to 0.86).

Hermetic: generates in-memory frames; no DB, no mocks. n=20000 keeps the holdout
(~3000 rows) large enough that the AUC reflects the DGP's true signal, not noise.
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
    """Train the REAL gold-standard initiation model (FeatureBuilder 7-cov + calibrated
    LR) on train+validation and report the holdout-split AUC — exactly _run_one_cohort's
    headline path for the initiation cohort."""
    df = PatientGenerator(GeneratorConfig(n_records=n, seed=seed, brand=brand)).generate()
    spec = make_patient_spec("initiation", brand.value)
    assert spec.label_column == "treatment_initiated"
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
        "init_prev": float(df["treatment_initiated"].mean()),
        "n_features": len(fb.feature_columns),
        "n_holdout": int(len(te)),
    }


@pytest.fixture(scope="module")
def faithful() -> dict:
    return {b: _faithful(b) for b in Brand}


# Per-brand initiation AUC ceiling — UNIFORM 0.86 after COMM-ARMS Phase 3.
#
# History: Remibrutinib carried a higher 0.85 ceiling because CLIN-SEG-P3's biologic-
# experience differential sharpened its severity->initiation gradient (0.804 -> ~0.839).
# COMM-ARMS Phase 3 (2026-07-20) adds rep_detailing_high + sample_dropped to the initiation
# SPEC (_BASE9_INITIATION): both arms fold into the treatment_initiated latent AND are
# pre-index (assigned from academic_hcp + engagement_score, BEFORE the initiation decision),
# so letting the model observe them is real observable-driver signal — the exact analogue of
# copay/psp on persistence, NOT leakage. Measured (seed=42, n=20000, 23 encoded features):
# Remibrutinib 0.846 / Fabhalta 0.829 / Kisqali 0.846 (Kisqali/Fabhalta gain the most —
# they lack the biologic lift so rep/sample add the most marginal AUC). Re-based to a
# uniform 0.86 (~0.014-0.031 headroom) which still fails on gross leakage (>0.9).
_INIT_AUC_CEILING = {"Remibrutinib": 0.86, "Fabhalta": 0.86, "Kisqali": 0.86}


def test_initiation_auc_in_target_band(faithful):
    for b, m in faithful.items():
        # prevalence-banded construction pins initiation prevalence at ~0.35
        assert 0.25 <= m["init_prev"] <= 0.45, f"{b.value}: init_prev {m['init_prev']} out of band"
        # 9 covariates (_BASE9_INITIATION: _BASE7 + rep_detailing_high + sample_dropped)
        # → ~23 encoded features (was 19 for the 7-covariate set; 9 for the legacy base-3).
        assert m["n_features"] >= 15, (
            f"{b.value}: only {m['n_features']} encoded features (expected the 9-covariate set)"
        )
        ceiling = _INIT_AUC_CEILING.get(b.value, 0.83)
        assert 0.78 <= m["auc"] <= ceiling, (
            f"{b.value}: faithful holdout AUC {m['auc']:.4f} out of realistic [0.78, {ceiling}]"
        )


def test_brands_vary(faithful):
    aucs = [m["auc"] for m in faithful.values()]
    assert max(aucs) - min(aucs) > 0.002, f"brands should differ in AUC; got {aucs}"


def test_ate_cate_recovery_unchanged_by_initiation_drivers():
    """Invariant gate: the 4 prognostic drivers added to the INITIATION outcome eqn are
    independent of treatment_arm, so the recoverable treatment effect (ATE) + segment
    heterogeneity (CATE ordering) on treatment_initiated are preserved end-to-end. This
    is the causal-correctness guard — initiation's Y IS the tuned CATE-recovery outcome,
    so enrichment must NOT move it. Measured (seed=42, n=6000): |ATE_err| < 0.10, CATE
    high >= med >= low."""
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
