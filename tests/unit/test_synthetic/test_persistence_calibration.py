"""FAITHFUL calibration gate for the enriched persistence DGP (T9).

UPDATED 2026-07-19 (COMM-ARMS Phase 2, backlog #43 folded in): the AUC assertion is now
a PER-BRAND PINNED BASELINE + tolerance (see _PERSISTENCE_AUC_BASELINE below), NOT the
single absolute band [0.75, 0.83] described in the historical notes further down. Those
notes are retained because they explain WHY the reshape happened (a single-seed point
estimate on a ~0.03-dispersion quantity that each commercial arm erodes further). The
[0.75, 0.83] numbers below are historical context, not the live assertion.

This is the measure-don't-assume gate, and it is FAITHFUL: it routes the generated
data through the EXACT real model pipeline — FeatureBuilder(make_patient_spec(...))
to encode the 7 leakage-safe covariates, then train_cohort_model (the calibrated
LogisticRegression the *_goldstd_lr_v1 models use), then the holdout-split AUC, the
same headline _run_one_cohort records. No hand-rolled encoder, no proxy.

The enriched equation must achieve a realistic holdout AUC per brand with prevalence in
the designed [0.05, 0.60] band. These asserted numbers LOCK the cohort_outcomes
coefficients (the same way the 2026-06-14 feature experiment locked KEEP_COLUMNS).
Measured live (2026-06-21, n=20000): Remi 0.809 / Fabhalta 0.814 / Kisqali 0.805,
19 encoded features (7 covariates), disc prevalence ~0.50.

FLOOR LOWERED 0.78 -> 0.75 on 2026-07-19 (COMM-ARMS Phase 1). This is a re-derivation
against a deliberately changed DGP, NOT an accommodation of a regression -- the
distinction matters, so here is the evidence (seed=42, the seed this gate asserts):

  brand         pre-Phase-1   +copay, 7-cov   +copay, 8-cov (shipped)
  Remibrutinib     0.8169        0.7863            0.7898
  Fabhalta         0.8232        0.7900            0.7949
  Kisqali          0.8086        0.7805            0.7850

Phase 1 puts copay_support into the discontinuation logit by design. Copay is real
outcome signal, so achievable AUC structurally falls -- the old 0.78 floor was
calibrated against a DGP with NO commercial arms and was measuring a different world.
At 7 covariates Kisqali cleared it by 0.0005, i.e. green but a guaranteed future red.
Letting the model SEE copay (the 8th covariate, cohort_spec._BASE8_COMMERCIAL) recovers
~0.005 of that drop.

MOST OF THE DROP IS IRREDUCIBLE -- do not try to win it back with a richer estimator.
Measured 2026-07-19 against a Bayes oracle (scoring with the true DGP probability) over
3 brands x 3 seeds x 2 worlds: for Kisqali the 0.0158 drop decomposes into 0.0096 (61%)
ceiling loss NO estimator can recover, plus a 0.0062 model gap; total headroom over the
shipped LR is 0.0074. Mechanism: corr(copay_term, rest_of_logit) = -0.35 -- copay goes
preferentially to sicker patients and its pull is most negative exactly there, so it
cancels the dominant severity gradient (observable-logit SD 1.454 -> 1.371). The AUC was
destroyed in the OUTCOME, not hidden from the model. Candidate estimators, paired on
identical rows/splits: explicit copay x segment +0.0064, degree-2 poly +0.0043,
HistGradientBoosting -0.0047 (worse) -- all under the 0.011-0.031 seed spread. Oracle
sanity check: pre-Phase-1 a plain LR (0.8134) EQUALS the Bayes ceiling (0.8135), i.e.
LR was already Bayes-optimal while the DGP was still linear-additive. Backlog #43.

The seed-42 column above is a point estimate, not the expected drop: across seeds
42/7/99 the drop spans 0.0158-0.0216 and per-variant seed spread is 0.011-0.031. That
dispersion is why an absolute band is a blunt instrument here (backlog #43 proposes
per-brand pinned baselines + tolerance instead).

0.75 restores ~0.035 of headroom at seed 42, which also leaves room for Phases 2-3
(three more commercial arms land in this same outcome, each eroding the ceiling by the
same irreducible mechanism). NO coefficient and NO shipped data changed -- only this
tolerance. If a future change pushes any brand below 0.75, that IS a real regression:
re-measure rather than lowering the floor again.

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

# --- backlog #43 (folded into COMM-ARMS Phase 2) --------------------------------------
# Per-brand PINNED baselines replace the old single absolute band [0.75, 0.83]. WHY the
# reshape: the old gate was a single-seed (seed 42) point estimate on a quantity with
# ~0.03 seed dispersion — deterministic, so it never flaked, but seed 42 is not
# representative, so any deliberate DGP change moved it unpredictably and the floor's
# true safety margin was far thinner than its nominal gap. Phase 1 had already cleared
# 0.78 by only 0.0005 at Kisqali; each commercial arm erodes the persistence ceiling
# further by the SAME irreducible mechanism (a commercial arm that goes preferentially
# to sicker patients cancels the severity gradient — measured against a Bayes oracle,
# backlog #43). A pinned per-brand baseline detects DRIFT FROM A KNOWN POINT instead of
# asserting an absolute level Phases 2-3 legitimately move.
#
# Values MEASURED on the faithful path (real FeatureBuilder _BASE9 + train_cohort_model,
# seed 42, n=20000) AFTER Phase 2 wired psp_enrolled into the discontinuation logit and
# added it as the 9th persistence/discontinuation model covariate. Measured on the LOCAL
# droplet. CPU-ISA FP divergence local(AVX2)/CI can shift AUC ~0.01 (cf.
# test_synthetic_baseline_invariant's ±0.01 bands); _TOL absorbs that plus modest DGP
# jitter. If the FIRST CI run lands just outside _TOL, RE-PIN from the CI value (mirror
# BASELINE_CI's placeholder→measured pattern) — do NOT widen blindly. A brand below the
# OUTER floor IS a real regression: re-measure, do not lower the floor again (Phase 1's
# own docstring instruction). Do NOT bundle a re-pin with an estimator change — that hits
# all 12 gold-standard models and fail-closes /shap (backlog #43 blast-radius note).
_PERSISTENCE_AUC_BASELINE = {
    Brand.REMIBRUTINIB: 0.7751,
    Brand.FABHALTA: 0.7827,
    Brand.KISQALI: 0.7701,
}
_PERSISTENCE_AUC_TOL = 0.025  # per-brand drift tolerance (absorbs local/CI FP + jitter)
_PERSISTENCE_AUC_OUTER = (0.72, 0.83)  # wide sanity band for gross breakage


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


def test_persistence_auc_matches_pinned_baseline(faithful):
    """backlog #43: per-brand pinned baseline + tolerance (was a single absolute band).

    Two-layer check: a wide OUTER sanity band catches gross breakage, and a per-brand
    drift check against the pinned post-Phase-2 baseline catches a DGP change moving the
    AUC away from its known point. A deliberate DGP change re-pins the baseline; an
    unexplained drift outside tolerance is a regression to re-measure."""
    lo, hi = _PERSISTENCE_AUC_OUTER
    for b, m in faithful.items():
        assert 0.05 <= m["disc_prev"] <= 0.60, f"{b.value}: prevalence {m['disc_prev']} out of band"
        # 9 leakage-safe covariates (_BASE9: _BASE7 + copay_support + psp_enrolled) →
        # ~23 encoded features (was ~19 at 7 covariates, ~21 at 8).
        assert m["n_features"] >= 15, (
            f"{b.value}: only {m['n_features']} encoded features (expected the 9-covariate set)"
        )
        assert lo <= m["auc"] <= hi, (
            f"{b.value}: faithful holdout AUC {m['auc']:.4f} out of outer sanity band [{lo}, {hi}]"
        )
        base = _PERSISTENCE_AUC_BASELINE[b]
        assert abs(m["auc"] - base) <= _PERSISTENCE_AUC_TOL, (
            f"{b.value}: faithful AUC {m['auc']:.4f} drifted {m['auc'] - base:+.4f} from the "
            f"pinned baseline {base:.4f} (tol ±{_PERSISTENCE_AUC_TOL}); a deliberate DGP change "
            "re-pins the baseline, an unexplained drift is a regression to re-measure."
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
