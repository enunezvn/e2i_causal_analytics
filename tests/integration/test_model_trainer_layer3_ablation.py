"""Phase 3.4 model_trainer ablation integration test (plan line 245).

Pins the LOAD-BEARING acceptance criterion: the model-eval ablation hook
catches a leak class that Phase 3.3 (data-prep ablation) cannot see.

Without this dual-mode pin, Phase 3.4 would be a duplicative check of
Phase 3.3 and should be closed NULL per the plan's revised sequencing
(.claude/plans/adaptive_temporal_validity_redesign.md line 401).

LEAK CLASS: Categorical per-category leak via OneHotEncoder.

  Construction (deterministic, seeded):
    * target ~ Bernoulli(0.30) — base prevalence matches RWD-realistic.
    * age, eligibility_duration ~ N(0, 1) — independent noise features
      (numeric, will be evaluated by Phase 3.3 numeric ablation).
    * region ~ Categorical(11 categories) with a rare "leak_region"
      assigned to ~6% of rows. For patients in "leak_region", target=1
      with probability ~0.95; for all other regions, target probability
      follows the base 0.30 rate.

  Why Phase 3.3 MISSES this leak:
    1. Layer 3 numeric ablation: ``_select_features`` at
       ``adaptive_validity_check.py:2530`` SKIPS non-numeric columns.
       The ``region`` column is object/categorical dtype and is never
       evaluated by the permutation OR the ablation pass.
    2. Layer 1 manifest: the synthetic regime has no manifest entry for
       ``region`` (``feature_manifest_source=None``) so the manifest
       contract check is inert.
    3. ``check_categorical_class_separation`` (in legacy
       ``leakage_detector.py``) uses Cramér's V on the WHOLE column with
       threshold 0.5/0.7. With 11 categories and only one (~6%) being
       leaky, the whole-column Cramér's V stays below 0.5 (the other 10
       categories dilute the signal). The pin asserts this with a direct
       Cramér's V calculation in the test body.

  Why model-eval ablation CATCHES this leak:
    OneHotEncoder splits ``region`` into 11 binary indicators (one per
    category). ``region_leak_region`` has strong target signal at the
    single-feature level. The model_trainer Phase 3.4 ablation pass
    runs on the ENCODED matrix — dropping ``region_leak_region`` collapses
    the joint model's AUC substantially, producing |delta_AUC| > 0.10
    floor AND high z-score → severity == "high".

Acceptance pins:
  1. Phase 3.3 mode (data-prep ablation, flag ON, categorical NOT
     numeric): ``region`` is NOT in adaptive_flagged_features. Phase 3.3
     ablation result for ``region`` does not exist (skipped from numeric
     selection).
  2. Cramér's V of ``region`` vs target < 0.5 (whole-column categorical
     check would not flag).
  3. Phase 3.4 mode (model_eval ablation, flag ON): a feature named
     ``region_leak_region`` (or whatever OneHotEncoder produces for that
     category) is flagged with ablation_severity=="high".
  4. Schema uniformity: when the Phase 3.4 flag is OFF, the
     ``model_eval_ablation`` key is absent from metrics_result.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest


def _build_categorical_per_category_leak_cohort(
    n: int = 2000, seed: int = 11
) -> tuple[pd.DataFrame, str]:
    """Construct the deterministic categorical per-category leak.

    Returns ``(df, target_name)``. ``df.columns`` = ``[age,
    eligibility_duration, region, y]``. Cramér's V of ``region`` vs ``y``
    stays below the 0.5 ``check_categorical_class_separation`` threshold
    because only one of 11 categories (~8%) is leaky and the rest dilute
    the whole-column signal.

    Cohort sized so that the rare leaky category produces enough rows
    (~160) for the joint-model retrain inside ``compute_feature_ablation``
    to detect a measurable |delta_AUC| > floor=0.10 when the
    ``region_leak_region`` OHE indicator is dropped. Smaller cohorts
    (n=800) under-power the ablation null and leave the leak undetected
    even when the indicator is structurally present.
    """
    rng = np.random.default_rng(seed)

    # Numeric noise features — independent of target. Phase 3.3 numeric
    # ablation evaluates these and produces severity=info.
    age = rng.normal(50, 15, n)
    eligibility_duration = rng.normal(180, 60, n)

    # Categorical feature with 11 regions. "leak_region" carries ~12%
    # of rows and is target-leaky; the other 10 are noise.
    # Calibration constraints (codex MED-2 fix tightened these):
    #   * Whole-cohort Cramér's V must stay < 0.5 (legacy
    #     check_categorical_class_separation HIGH threshold) so the
    #     legacy categorical check ALSO misses the leak — proves the
    #     gap is not artificially constructed.
    #   * Train-split (60% of cohort) single-feature label-shuffle
    #     delta_AUC for the leak indicator must EXCEED the issue #194
    #     floor of 0.10 so the joint check escalates severity to high.
    # At n=2000, leak_p=0.12: train_delta_AUC ≈ 0.157, cohort_V ≈ 0.469
    # — both invariants satisfied with margin. Lower leak_p (0.06-0.10)
    # produces train_delta_AUC right at the floor with too little margin
    # for the test to be robust across seed/split variance.
    non_leak_regions = [f"region_{i}" for i in range(10)]
    leak_p = 0.12
    is_leak = rng.binomial(1, leak_p, n).astype(bool)
    region = np.where(
        is_leak,
        "leak_region",
        rng.choice(non_leak_regions, size=n),
    )

    # Target generation:
    #   * For non-leak rows: target ~ Bernoulli(0.30).
    #   * For leak_region rows: target = 1 (deterministic — extreme
    #     per-category leak. A near-deterministic mapping makes the
    #     ablation signal unmistakable: dropping ``region_leak_region``
    #     forces the joint model to predict ~0.30 for rows that
    #     deterministically had target=1, producing a large |delta_AUC|.
    # An idealized leak (conditional p=1.0) is the cleanest construction;
    # in practice the leak class admits noisier signals but the test pin
    # focuses on the structural mechanism: per-category indicator drives
    # joint AUC.
    base_p = np.full(n, 0.30)
    target = rng.binomial(1, base_p).astype(int)
    target[is_leak] = 1

    df = pd.DataFrame(
        {
            "age": age.astype(float),
            "eligibility_duration": eligibility_duration.astype(float),
            "region": region.astype(object),
            "y": target,
        }
    )
    return df, "y"


def _cramers_v(series: pd.Series, target: pd.Series) -> float:
    """Compute Cramér's V on a 2-way contingency table.

    Mirrors the formula in
    ``check_categorical_class_separation`` (leakage_detector.py:1109).
    """
    from scipy.stats import chi2_contingency

    contingency = pd.crosstab(series, target)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0
    chi2, _p, _dof, _ = chi2_contingency(contingency)
    n = len(series)
    k = min(contingency.shape) - 1
    if k == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def _tree_model_factory():
    """sklearn DecisionTreeClassifier factory (matches Phase 3.3 pattern).

    Used both for Phase 3.3 ablation (over numeric features) and for the
    model-eval Phase 3.4 ablation (over encoded matrix). For the per-
    category leak we plant, a linear classifier would also work — but the
    factory choice is held identical to Phase 3.3 to keep the comparison
    apples-to-apples and to keep the test resilient to future changes
    that introduce interaction structure between the leaky indicator and
    numeric noise.
    """
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(max_depth=5, random_state=42)


# Phase 3.3 helper — invoke ``adaptive_validity_check`` to prove the
# categorical column is not caught at the data-prep stage.
def _run_phase33(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    state = {
        "experiment_id": "test-phase-3-4-perCategory",
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": target,
            "required_features": [c for c in df.columns if c != target],
            "excluded_features": [],
            "feature_manifest_source": None,
        },
        "leakage_findings": [],
        "leaked_features": [],
        "adaptive_n_permutations": 50,
        "adaptive_seed": 7,
        "adaptive_layer3_ablation_enabled": True,
        "adaptive_ablation_n_permutations": 30,
        "adaptive_ablation_model_factory": _tree_model_factory,
    }
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(adaptive_validity_check(state))
    finally:
        loop.close()


def _build_evaluator_state(
    df: pd.DataFrame,
    target: str,
    *,
    ablation_enabled: bool,
    model_factory=None,
) -> tuple[Dict[str, Any], Any]:
    """Build a minimal model_trainer state for the evaluator + trained model.

    Returns (state, trained_model). The trained_model is the model under
    test for the model_eval_ablation pass; the state carries the
    preprocessor + encoded test split so the evaluator's `_wrap_with_feature_names`
    can recover OneHotEncoder output names.
    """
    from sklearn.tree import DecisionTreeClassifier

    from src.agents.ml_foundation.model_trainer.nodes.preprocessor import (
        ModelTrainerPreprocessor,
    )

    # Deterministic split (reuse the same shuffle for stability).
    rng = np.random.default_rng(13)
    idx = rng.permutation(len(df))
    n_train = int(0.6 * len(df))
    n_val = int(0.2 * len(df))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    X_train = df.iloc[train_idx].drop(columns=[target]).reset_index(drop=True)
    y_train = df.iloc[train_idx][target].to_numpy()
    X_val = df.iloc[val_idx].drop(columns=[target]).reset_index(drop=True)
    y_val = df.iloc[val_idx][target].to_numpy()
    X_test = df.iloc[test_idx].drop(columns=[target]).reset_index(drop=True)
    y_test = df.iloc[test_idx][target].to_numpy()

    preprocessor = ModelTrainerPreprocessor(
        numeric_features=["age", "eligibility_duration"],
        categorical_features=["region"],
    )
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)

    # Train a tree on the encoded training matrix. The model sees the
    # OHE-expanded ``region_leak_region`` indicator and learns to depend on
    # it; the ablation pass will measure |delta_AUC| when that indicator
    # is dropped from the post-encoding design matrix.
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train_enc, y_train)

    state: Dict[str, Any] = {
        "trained_model": clf,
        "problem_type": "binary_classification",
        "success_criteria": {},
        "preprocessor": preprocessor,
        "X_train_preprocessed": X_train_enc,
        "X_validation_preprocessed": X_val_enc,
        "X_test_preprocessed": X_test_enc,
        "train_data": {"X": X_train, "y": y_train},
        "validation_data": {"X": X_val, "y": y_val},
        "test_data": {"X": X_test, "y": y_test},
        "model_trainer_layer3_ablation_enabled": ablation_enabled,
        "model_trainer_ablation_n_permutations": 30,
        "model_trainer_ablation_model_factory": model_factory or _tree_model_factory,
        "model_trainer_ablation_seed": 42,
    }
    return state, clf


@pytest.mark.integration
def test_phase33_misses_per_category_categorical_leak() -> None:
    """ACCEPTANCE PIN 1 + 2: Phase 3.3 cannot catch per-category categorical leak.

    Pin 1: Phase 3.3 ``adaptive_validity_check`` does not flag the
    ``region`` column even with the ablation flag ON. The pre-condition
    that motivates Phase 3.4 — the leak class is invisible at data-prep
    time.

    Pin 2: Cramér's V of ``region`` vs target stays below the legacy
    ``check_categorical_class_separation`` HIGH threshold (0.5). This
    proves the legacy categorical check would also miss the leak; Phase
    3.3 is not the only data-prep defense being bypassed.
    """
    df, target = _build_categorical_per_category_leak_cohort()

    # Cramér's V — the whole-column categorical check threshold.
    v = _cramers_v(df["region"], df[target])
    assert v < 0.50, (
        f"Test premise broken: whole-column Cramér's V of region vs {target} = "
        f"{v:.4f} ≥ 0.50; the legacy check_categorical_class_separation would "
        f"catch this. Re-tune leak_p or non-leak base rate so the rare leaky "
        f"category does not dominate the whole-column statistic."
    )

    # Phase 3.3 — adaptive_validity_check with ablation ON.
    result = _run_phase33(df, target)
    flagged_p33 = result.get("adaptive_flagged_features", [])

    assert "region" not in flagged_p33, (
        f"Phase 3.3 unexpectedly flagged region (Phase 3.4 milestone premise "
        f"broken). flagged={flagged_p33}"
    )
    # Stronger pin: no Phase 3.3 verdict whatsoever for ``region``. The
    # _select_features helper drops non-numeric columns BEFORE Layer 3 runs,
    # so the column is not even evaluated.
    region_verdicts = [
        v for v in (result.get("adaptive_verdicts") or []) if v.get("feature") == "region"
    ]
    assert region_verdicts == [], (
        f"Phase 3.3 produced a verdict for the non-numeric ``region`` column; "
        f"_select_features should have excluded it. Got: {region_verdicts}"
    )


@pytest.mark.integration
def test_phase34_model_eval_ablation_catches_per_category_categorical_leak() -> None:
    """ACCEPTANCE PIN 3: model-eval Phase 3.4 ablation flags the OHE-expanded leak.

    The load-bearing assertion. Phase 3.3 (above) doesn't see the leak;
    Phase 3.4 must catch it. ``OneHotEncoder`` exposes the rare leaky
    category as a per-category indicator on which a tree-based model
    trains a tight split — dropping that indicator collapses the joint
    AUC, producing severity="high" via the strong-effect escape OR the
    joint-check ladder.
    """
    df, target = _build_categorical_per_category_leak_cohort()
    state, _model = _build_evaluator_state(df, target, ablation_enabled=True)

    from src.agents.ml_foundation.model_trainer.nodes.evaluator import evaluate_model

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(evaluate_model(state))
    finally:
        loop.close()

    assert "error" not in result, f"evaluator returned error: {result}"

    ablation_payload = result.get("model_eval_ablation")
    assert ablation_payload is not None, (
        "model_eval_ablation key missing from evaluator output. Phase 3.4 "
        "wiring is not consuming the ``model_trainer_layer3_ablation_enabled`` "
        "flag correctly."
    )
    assert ablation_payload.get("ran") is True, (
        f"Phase 3.4 ablation pass did not run. skipped_reason="
        f"{ablation_payload.get('skipped_reason')}"
    )

    flagged = ablation_payload.get("flagged_features") or []
    # The OneHotEncoder under verbose_feature_names_out=False produces
    # names like "region_leak_region". The exact prefix depends on the
    # ColumnTransformer's column naming; we look for any flagged name
    # that mentions ``leak_region`` (the category we planted).
    leak_indicators = [f for f in flagged if "leak_region" in str(f)]
    assert leak_indicators, (
        f"No flagged feature includes 'leak_region'. The per-category leak "
        f"via OneHotEncoder was missed. flagged={flagged}, per_feature_top5="
        f"{ablation_payload.get('per_feature', [])[:5]}"
    )

    # The leak should be caught via the LABEL-SHUFFLE PERMUTATION sub-
    # pass. The column-shuffle ablation null collapses on rare per-
    # category binary indicators (null_mean ≈ actual_delta because
    # shuffling preserves the marginal binary distribution). The label-
    # shuffle pass IS sensitive: shuffling labels destroys the target-
    # conditional structure that makes the leak indicator predictive.
    per_feature = ablation_payload.get("per_feature") or []
    leak_rows = [r for r in per_feature if "leak_region" in str(r.get("feature", ""))]
    assert leak_rows, f"no per_feature row contains 'leak_region'; got {per_feature[:5]}"
    leak_row = leak_rows[0]
    # Combined severity must be ``high`` (the MAX-rule sees high perm
    # severity and floors to that band).
    assert leak_row.get("severity") == "high", (
        f"Leaky OHE indicator should be combined severity=='high'. Got "
        f"row={leak_row}. If permutation_severity is 'high' but combined "
        f"is not, the MAX-rule is broken; if both are below 'high', the "
        f"leak signal is too weak — re-tune leak_p or conditional p."
    )
    # Decided_by must record the permutation sub-pass — this is the
    # LOAD-BEARING pin that Phase 3.4's label-shuffle adds genuinely-
    # new detection capability beyond what Phase 3.3 ablation provides
    # (Phase 3.3 ablation's column-shuffle null collapses; ablation
    # alone here would miss the leak too).
    # Phase 3.3 convention (adaptive_validity_check.py:1315): the perm
    # path tags ``decided_by="adversarial"`` (NOT "adversarial_permutation").
    # Phase 3.4 mirrors this exactly so audit consumers see byte-identical
    # tag strings + layer mapping across both pipeline stages
    # (_DECIDED_BY_TO_LAYER at adaptive_validity_check.py:1155 maps to
    # layer "3"). Codex MED-3 fix.
    assert leak_row.get("decided_by") == "adversarial", (
        f"Leak should be caught by the LABEL-SHUFFLE permutation sub-pass "
        f"(decided_by='adversarial', matching Phase 3.3's convention), "
        f"proving the perm pass adds genuine detection capability beyond "
        f"Phase 3.3 (which cannot run on categoricals). Got "
        f"decided_by={leak_row.get('decided_by')}. Full row: {leak_row}"
    )

    # Promotion: validation_metrics carries the compact summary.
    validation_metrics = result.get("validation_metrics") or {}
    assert validation_metrics.get("model_eval_ablation_ran") is True
    promoted_flagged = validation_metrics.get("model_eval_ablation_flagged_features") or []
    assert any("leak_region" in str(f) for f in promoted_flagged), (
        f"validation_metrics did not surface flagged leak indicator. Got: {promoted_flagged}"
    )


@pytest.mark.integration
def test_phase34_flag_off_is_inert() -> None:
    """SCHEMA pin: with the Phase 3.4 flag OFF, the ablation key is absent.

    The default-OFF contract — exactly matches §4 T2.2 / T2.3 advisory
    pattern. A regression that defaulted ON would silently add a 10-30 s
    cost to every model_trainer run and would slow CI for the 99% of
    runs that have no leak to find.
    """
    df, target = _build_categorical_per_category_leak_cohort()
    state, _model = _build_evaluator_state(df, target, ablation_enabled=False)

    from src.agents.ml_foundation.model_trainer.nodes.evaluator import evaluate_model

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(evaluate_model(state))
    finally:
        loop.close()

    assert "error" not in result, f"evaluator returned error: {result}"
    assert "model_eval_ablation" not in result, (
        f"Phase 3.4 should be inert when flag is OFF. Got "
        f"model_eval_ablation={result['model_eval_ablation']}"
    )
    validation_metrics = result.get("validation_metrics") or {}
    assert "model_eval_ablation_ran" not in validation_metrics, (
        "validation_metrics should not carry model_eval_ablation_* keys when "
        f"the flag is OFF. Got: {validation_metrics}"
    )


@pytest.mark.integration
def test_phase34_permutation_joint_check_clamps_large_n_weak_predictor() -> None:
    """ISSUE #194 PIN: weak predictor at large n with |delta_AUC| <= 0.10
    must stay severity=info despite z >> 5σ.

    Tests the codex MED-2 fix: the simple z-band perm classifier had a
    5σ FPR blowup at n≥10k because the label-shuffle null variance
    shrinks per CLT, making any tiny signal LOOK statistically
    significant. Phase 3.3 mitigates this with the issue #194 joint
    (z, |delta_AUC|) check in ``hblp_classify``; Phase 3.4 must do the
    same on the model-eval axis.

    Construction: a benign weak predictor at n=10000 with
    actual_auc ≈ 0.53 (folded), null_mean ≈ 0.50, null_std ≈ 0.005 →
    z ≈ 6.0σ but |delta_AUC| = 0.03 < 0.10 floor. Pre-fix would have
    classified severity=high (FALSE POSITIVE); post-fix returns
    severity=info via the joint clamp.

    The unit-level test directly invokes ``_classify_permutation_severity``
    rather than the full evaluator — a full-pipeline integration test
    at n=10k is too slow for routine CI (compute_adversarial_score with
    200 perms × 10000 rows × 50 features × tree retrain ≈ 5-10 minutes).
    The integration coverage already proves the joint check is wired
    through (the existing dual-mode test exercises the full pipeline
    at n=2000); this test pins the LARGE-N CLAMP behavior directly.
    """
    from src.agents.ml_foundation.model_trainer.nodes.model_eval_ablation import (
        _classify_permutation_severity,
        MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    )

    # PRE-ISSUE-#194 path (no delta_AUC → z-only ladder): z=6.0 → high.
    assert (
        _classify_permutation_severity(6.0)
        == "high"
    ), "Backward-compat: z-only path with no delta_auc should still ladder."

    # ISSUE #194 fix: z=6.0 but |delta_AUC|=0.03 < floor=0.10 → info.
    sev = _classify_permutation_severity(
        6.0,
        delta_auc=0.03,
        delta_auc_floor=MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    )
    assert sev == "info", (
        f"Issue #194 joint check failed to clamp large-n weak predictor "
        f"(z=6.0σ, |delta_AUC|=0.03 < 0.10 floor). Got severity={sev!r}. "
        f"The clamp prevents the n≥10k FPR blowup that Phase 3.3 mitigates "
        f"via hblp_classify."
    )

    # ISSUE #194 fix: z=6.0 AND |delta_AUC|=0.15 > 0.10 floor → high
    # (joint check confirms strong signal — both z and delta agree).
    sev2 = _classify_permutation_severity(
        6.0,
        delta_auc=0.15,
        delta_auc_floor=MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    )
    assert sev2 == "high", (
        f"Real leak (z=6.0σ AND |delta_AUC|=0.15 > 0.10) should escalate "
        f"to high. Got severity={sev2!r}."
    )

    # Negative delta with z above band — also clamped to info (Phase 3.3
    # joint check is on |delta|; here delta=-0.03 has |delta|=0.03 < floor).
    sev3 = _classify_permutation_severity(
        6.0,
        delta_auc=-0.03,
        delta_auc_floor=MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    )
    assert sev3 == "info", (
        f"|delta_AUC|=0.03 < floor=0.10 should clamp regardless of delta sign. "
        f"Got severity={sev3!r}."
    )

    # +inf z with strong effect → high (Phase 3.3 mirror escape).
    sev4 = _classify_permutation_severity(
        float("inf"),
        delta_auc=0.15,
        delta_auc_floor=MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    )
    assert sev4 == "high", (
        f"+inf z with strong effect (|delta_AUC|=0.15 > 0.10) should escalate "
        f"to high (degenerate-null escape). Got severity={sev4!r}."
    )

    # +inf z without strong effect → info (weak deterministic signal).
    sev5 = _classify_permutation_severity(
        float("inf"),
        delta_auc=0.03,
        delta_auc_floor=MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    )
    assert sev5 == "info", (
        f"+inf z without strong effect (|delta_AUC|=0.03 < 0.10) should "
        f"clamp to info. Got severity={sev5!r}."
    )

    # Moderate band still works: z=4.0, delta=0.15 → moderate.
    sev6 = _classify_permutation_severity(
        4.0,
        delta_auc=0.15,
        delta_auc_floor=MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    )
    assert sev6 == "moderate", (
        f"Moderate-band z + strong delta should escalate to moderate. "
        f"Got severity={sev6!r}."
    )

    # Moderate band gets clamped by joint check: z=4.0, delta=0.05 → info.
    sev7 = _classify_permutation_severity(
        4.0,
        delta_auc=0.05,
        delta_auc_floor=MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    )
    assert sev7 == "info", (
        f"Moderate-band z with weak delta should clamp to info. "
        f"Got severity={sev7!r}."
    )


@pytest.mark.integration
def test_phase34_severity_classifier_mirrors_phase33() -> None:
    """COMPOSITION pin: severity rule is byte-identical to Phase 3.3.

    Both hooks must reason on a unified severity scale; if the model-eval
    classifier drifted from the data-prep classifier the audit would
    become inconsistent. This test feeds identical rows into both
    classifiers and asserts they return the same severity.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _classify_ablation_severity,
    )
    from src.agents.ml_foundation.model_trainer.nodes.model_eval_ablation import (
        classify_model_eval_ablation_severity,
    )

    cases = [
        # Strong-effect escape (positive delta) — high.
        {"delta_auc": 0.40, "z_score": 1.0},
        # Joint-check ladder, z > 5.0 + delta > 0.10 — high.
        {"delta_auc": 0.15, "z_score": 7.0},
        # Joint-check ladder, 3.0 < z <= 5.0 + delta > 0.10 — moderate.
        {"delta_auc": 0.15, "z_score": 4.0},
        # Below floor — info.
        {"delta_auc": 0.05, "z_score": 10.0},
        # Negative delta (nuisance) — info even with large z.
        {"delta_auc": -0.40, "z_score": 10.0},
        # NaN delta — info (degradation contract).
        {"delta_auc": float("nan"), "z_score": 7.0},
        # NaN z — info.
        {"delta_auc": 0.15, "z_score": float("nan")},
        # +inf z (degenerate null) + delta > floor — high (mirror escape).
        {"delta_auc": 0.15, "z_score": float("inf")},
    ]
    for case in cases:
        p33 = _classify_ablation_severity(case)
        p34 = classify_model_eval_ablation_severity(case)
        assert p33 == p34, (
            f"Severity classifier drift between Phase 3.3 and Phase 3.4. "
            f"case={case}, Phase 3.3={p33}, Phase 3.4={p34}. "
            f"The two classifiers must stay byte-identical."
        )


def _build_pure_noise_cohort(n: int = 2000, seed: int = 31) -> tuple[pd.DataFrame, str]:
    """Construct a deterministic cohort with NO leakage.

    Returns ``(df, target_name)``. All features are independent of the
    target. ``region`` is an 11-category column where every category is
    pure noise. Used by the false-positive pin to verify that ablation-
    enabled mode does not flag spurious OHE indicators when no leak exists.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 15, n)
    eligibility_duration = rng.normal(180, 60, n)
    regions = [f"region_{i}" for i in range(11)]
    region = rng.choice(regions, size=n)
    target = rng.binomial(1, 0.30, n).astype(int)
    df = pd.DataFrame(
        {
            "age": age.astype(float),
            "eligibility_duration": eligibility_duration.astype(float),
            "region": region.astype(object),
            "y": target,
        }
    )
    return df, "y"


@pytest.mark.integration
def test_phase34_pure_noise_does_not_flag_any_ohe_indicator() -> None:
    """FALSE-POSITIVE pin: with NO leak planted, no encoded feature flags.

    Mirrors Phase 3.3's ``test_ablation_enabled_does_not_false_flag_noise_features``
    contract on the model-eval axis. A too-loose perm threshold OR a
    bug that treats every OHE indicator as suspicious would regress
    this pin — important because in production the typical state is
    "no leak", so false positives have a high reviewer-fatigue cost.

    The pin is structural: severity for ALL encoded features must stay
    "info" when the planted cohort is pure noise.
    """
    df, target = _build_pure_noise_cohort()
    state, _model = _build_evaluator_state(df, target, ablation_enabled=True)

    from src.agents.ml_foundation.model_trainer.nodes.evaluator import evaluate_model

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(evaluate_model(state))
    finally:
        loop.close()

    assert "error" not in result, f"evaluator returned error: {result}"
    ablation_payload = result.get("model_eval_ablation")
    assert ablation_payload is not None
    assert ablation_payload.get("ran") is True
    flagged = ablation_payload.get("flagged_features") or []
    assert flagged == [], (
        f"Phase 3.4 false-flagged {len(flagged)} feature(s) on a pure-noise "
        f"cohort. flagged={flagged}. Re-tune the threshold or fix the perm/"
        f"ablation pass — false positives are the dominant cost in production."
    )


@pytest.mark.integration
def test_phase34_max_rule_tie_break_mirrors_phase33() -> None:
    """TIE-BREAK pin: when perm and ablation severity TIE, the
    permutation sub-pass gets credit.

    Mirrors Phase 3.3's ``_combine_ablation_with_permutation`` at
    ``adaptive_validity_check.py:2320`` (``if ablation_rank <= perm_rank:
    return perm_input``) — ties go to perm because the permutation pathway
    is the canonical Layer-3 entry point in Phase 3.3 (ablation is an
    ESCALATION applied on top, never a bypass).

    If the model-eval tie-break flipped to "ablation wins ties" the audit
    convention would silently invert: the `decided_by` tag in Phase 3.4
    would attribute ties to ablation while Phase 3.3 attributes them to
    permutation.
    """
    from src.agents.ml_foundation.model_trainer.nodes.model_eval_ablation import (
        _max_rule_severity,
    )

    # All non-info ties — combined severity must equal both perm and
    # ablation (they agree on severity, so combined = either).
    assert _max_rule_severity("info", "info") == "info"
    assert _max_rule_severity("moderate", "moderate") == "moderate"
    assert _max_rule_severity("high", "high") == "high"

    # Perm strictly wins.
    assert _max_rule_severity("moderate", "info") == "moderate"
    assert _max_rule_severity("high", "moderate") == "high"
    assert _max_rule_severity("high", "info") == "high"

    # Ablation strictly wins.
    assert _max_rule_severity("info", "moderate") == "moderate"
    assert _max_rule_severity("moderate", "high") == "high"
    assert _max_rule_severity("info", "high") == "high"
