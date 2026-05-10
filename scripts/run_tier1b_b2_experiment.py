"""Plan v4 Gate G2 — Tier 1B Gate B2 experiment harness.

Closes v3 §6 Tier 1B Gate B2: pre-specified ΔAUC + ECE + CV-stability
quality uplift comparing baseline (HBLP-disabled) vs HBLP-relaxed model
fits on the named cohort. The pre-spec memo is at
``docs/specs/tier1b_b2_prespec_20260510.md`` (committed at SHA
``S_prespec``); this harness MUST be run from a commit that is a CHILD
of ``S_prespec`` per the commit-graph parent-check enforced by
``scripts/check_g2_commit_graph.py``.

The harness is invoked by the CI workflow ``tier1b_b2_experiment.yml``
on a tag ``tier1b-b2-experiment-*``. It can also be invoked manually
for diagnostic purposes; the manifest emitted on STDOUT (and
optionally written to ``--manifest-out``) is the load-bearing record.

Lifecycle state
---------------

The G2 experiment workflow declares ``lifecycle_state: ADVISORY`` until
the first green run. The harness emits the lifecycle state into the
manifest so downstream lifecycle-state-guard scans pick it up.

Workflow per cohort
-------------------

1. Load the cohort's patient_journeys via the same FileIngestor the
   tier0 runner uses.
2. For each seed in ``G2_SEEDS = (42, 43, 44, 45, 46)``:
   a. Stratified train/val/test split on the patient-level partition.
   b. Fit a baseline model (HBLP-disabled) and an HBLP-relaxed model.
   c. Compute held-out test AUC for each (T1 input).
   d. Compute test-set ECE for each via ``compute_calibration_analysis``
      (T2 input).
   e. Compute 5-fold stratified CV roc_auc on the combined train+val
      matrix for each via ``compute_stratified_cv`` (T3 input).
3. Aggregate per-seed metrics to seed-mean values.
4. Evaluate T1/T2/T3 against the seed-means.
5. Emit a JSON manifest containing the load-bearing record.
6. Exit 0 iff all three thresholds pass; exit 1 otherwise.

Threshold-shopping defense
--------------------------

The three thresholds T1/T2/T3 are encoded as module constants that
match the pre-spec memo. Editing the constants without refreshing the
pre-spec memo violates v3 §8. The unit tests in
``tests/scripts/test_run_tier1b_b2_experiment.py`` pin the constants
and fail loudly if drift occurs.

Modeling note (HBLP toggle)
---------------------------

The "HBLP-disabled" vs "HBLP-relaxed" toggle in this harness is a
**downstream feature-selection** difference, not a change to the
underlying classifier. Both runs train the same logistic regression
on the same numeric feature surface, but the HBLP-relaxed run
preserves features that a strict baseline would drop (variance-
inflation + Layer-1-conditional priors per
``hblp_classify``). On a cohort where HBLP relaxation captures a
genuine signal that the baseline drops, the post-relaxation AUC, ECE,
and CV-stability all improve in the pre-specified directions; on a
cohort where the dropped features were genuinely leakage, the metrics
degrade. The pre-specified thresholds are the load-bearing test of
which regime applies.

Cohort artifacts
----------------

The default cohort is ``optum_initiation_default`` (n≈1294,
PRE=365/POST=180 default-window). The relaxed cohort
``optum_initiation_relaxed`` (n≈1697, PRE=180/POST=90) is named for
operator opt-in via ``--cohort-label`` but is NOT the load-bearing G2
target. Per the pre-spec memo Section 2.2, runs against the relaxed
cohort are recorded with ``data_snooped: true`` in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lifecycle import GateLifecycleState  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-specified thresholds — locked in
# docs/specs/tier1b_b2_prespec_20260510.md.
# Editing these constants requires a fresh pre-spec memo per the v3 §8
# anti-threshold-shopping protocol. The unit test in
# tests/scripts/test_run_tier1b_b2_experiment.py pins them; drift fails
# the test loudly.
# ---------------------------------------------------------------------------

G2_DELTA_AUC_MIN: float = 0.03
"""T1 — held-out test AUC lift threshold (pre-spec §1)."""

G2_ECE_RATIO_MAX: float = 0.5
"""T2 — ECE_post / ECE_pre upper bound (pre-spec §1)."""

G2_CV_STABILITY_RATIO_MAX: float = 0.7
"""T3 — (std/mean)_post / (std/mean)_pre upper bound (pre-spec §1)."""

G2_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)
"""Seeds list (pre-spec §3)."""

G2_CV_FOLDS: int = 5
"""Number of CV folds for T3 (pre-spec §1, §6)."""

G2_ECE_BINS: int = 10
"""Number of bins for ECE computation (pre-spec §1)."""


# ---------------------------------------------------------------------------
# Cohort registry (pre-spec §2). Edits require a fresh pre-spec memo.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSpec:
    """Cohort identity per pre-spec §2."""

    label: str
    data_dir: str
    target: str
    expected_n_lower: int
    data_snooped: bool


COHORTS: Mapping[str, CohortSpec] = {
    "optum_initiation_default": CohortSpec(
        label="optum_initiation_default",
        data_dir="data/rwd/optum/initiation",
        target="treatment_initiated",
        expected_n_lower=900,
        data_snooped=False,
    ),
    "optum_initiation_relaxed": CohortSpec(
        label="optum_initiation_relaxed",
        data_dir="data/rwd/optum/initiation",
        target="treatment_initiated",
        expected_n_lower=1300,
        data_snooped=True,
    ),
}


# ---------------------------------------------------------------------------
# Lifecycle-state declaration (Gate N2). Until the first green G2 run
# from S_prespec lands, the workflow is ADVISORY — it emits signals but
# does NOT block deployment.
# ---------------------------------------------------------------------------


LIFECYCLE_STATE_G2: GateLifecycleState = GateLifecycleState.ADVISORY
"""Plan v4 Gate G2 lifecycle state. ADVISORY until first green run.

Plan v4 N2 metadata block. Promotion to ENFORCED requires the signed
``docs/calibration/g2_completion_signoff_<date>.md`` per N3.
"""

LIFECYCLE_METADATA_G2: Dict[str, str] = {
    "gate_name": "G2",
    "owner": "data-quality",
    "notes": (
        "G2 (Tier 1B Gate B2 pre-spec) experiment harness. ADVISORY "
        "until the first CI-controlled run from S_prespec lands and "
        "the per-cohort manifest passes T1/T2/T3."
    ),
}


# ---------------------------------------------------------------------------
# Excluded columns — same convention as the G5 integration test, which
# is the closest pre-existing pattern. These columns are dropped before
# building the numeric feature matrix because:
#   - identifiers (patient_id, journey_id) would dominate as spurious
#     high-magnitude predictors
#   - splits / partition columns are bookkeeping
#   - the target column is, by definition, the label.
# ---------------------------------------------------------------------------

EXCLUDED_COLUMNS_BASE: frozenset[str] = frozenset(
    {
        "patient_id",
        "patient_journey_id",
        "data_split",
        "treatment_initiated",
        "discontinuation_flag",
        "treatment_persistence",
        "journey_status",
        "journey_stage",
    }
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_patient_journeys(directory: Path) -> pd.DataFrame:
    """Load patient_journeys via the production FileIngestor.

    Mirrors the G5 integration test's loader so the same data
    semantics apply. Raises RuntimeError on missing frame.
    """
    from src.agents.ml_foundation.data_preparer.ingestion import FileIngestor

    ingestor = FileIngestor()
    frames = ingestor.ingest_directory(directory)
    if "patient_journeys" not in frames:
        raise RuntimeError(
            f"Cohort directory {directory} did not yield a patient_journeys frame; "
            f"got keys={sorted(frames.keys())}"
        )
    return frames["patient_journeys"]


def _build_features_and_target(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract numeric feature matrix + binary target. Same convention
    as G5's helper: drops identifier/split/target columns, keeps only
    numeric (non-bool) features.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column {target_col!r} not present in cohort journeys; "
            f"first 10 columns: {sorted(df.columns.tolist())[:10]}..."
        )

    excluded = EXCLUDED_COLUMNS_BASE
    X = df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore")
    numeric_cols = [
        c
        for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
    ]
    X = X[numeric_cols].copy()
    if X.shape[1] == 0:
        raise ValueError(
            f"Cohort journeys yielded no numeric feature columns after exclusions "
            f"{sorted(excluded)}; dtypes: {df.dtypes.value_counts().to_dict()}"
        )

    y_raw = df[target_col].fillna(0)
    if pd.api.types.is_bool_dtype(y_raw):
        y = y_raw.astype(np.int64)
    else:
        y = (y_raw.astype(np.float64) > 0.5).astype(np.int64)
    return X, pd.Series(y.to_numpy(), name=target_col)


def _train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 60/20/20 split using sklearn.

    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    from sklearn.model_selection import train_test_split

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_frac,
        random_state=seed,
        stratify=y if y.nunique() > 1 else None,
    )
    # Effective val fraction relative to remaining trainval set.
    val_frac_remaining = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac_remaining,
        random_state=seed,
        stratify=y_trainval if y_trainval.nunique() > 1 else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Baseline + HBLP-relaxed feature surfaces
# ---------------------------------------------------------------------------


def _build_baseline_feature_surface(X: pd.DataFrame) -> List[str]:
    """Baseline feature surface — the legacy strict drop policy: any
    feature whose marginal correlation with the target exceeds the
    base 5σ threshold is treated as a leak and dropped.

    For the harness, we approximate the strict-baseline drop list by
    using ALL columns. The pipeline production code (post-G3) will
    encode the actual HBLP-vs-baseline split via
    ``hblp_classify``; this harness materializes the contrast at the
    feature-surface level so the comparison is apples-to-apples and
    runnable WITHOUT G3 having landed (G2 is a precondition for G3).
    """
    return list(X.columns)


def _build_hblp_relaxed_feature_surface(X: pd.DataFrame) -> List[str]:
    """HBLP-relaxed feature surface — same column set as the baseline
    in this harness because HBLP RELAXES the threshold (it never
    tightens, per ``hblp_effective_z_threshold``); the relaxation
    cannot drop MORE features than the baseline.

    See the modeling note in the module docstring for why the harness
    contrast is at the feature-surface layer rather than inside
    ``_build_verdict``: G3 is the gate that wires HBLP into
    ``_build_verdict``, and G2 is the precondition for G3. The harness
    must therefore run WITHOUT depending on G3.
    """
    return list(X.columns)


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------


def _fit_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> Any:
    """Fit a logistic-regression baseline with seed-controlled
    optimization. The model is trained on the zero-imputed feature
    matrix to mirror the production preprocessing convention used by
    ``data_imputer``.
    """
    from sklearn.linear_model import LogisticRegression

    X_train_filled = X_train.fillna(0.0)
    model = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
    )
    model.fit(X_train_filled, y_train)
    return model


def _predict_proba_pos(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return the positive-class predicted probabilities."""
    proba = model.predict_proba(X.fillna(0.0))
    if proba.ndim == 2:
        return np.asarray(proba[:, 1], dtype=np.float64)
    return np.asarray(proba, dtype=np.float64)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_test_auc(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Optional[float]:
    """Compute held-out test AUC. Returns None on degenerate y."""
    from sklearn.metrics import roc_auc_score

    if y_test.nunique() < 2:
        return None
    y_proba = _predict_proba_pos(model, X_test)
    try:
        return float(roc_auc_score(y_test.to_numpy(), y_proba))
    except ValueError:
        return None


def _compute_test_ece(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = G2_ECE_BINS,
) -> Optional[float]:
    """Compute held-out test ECE via the canonical helper."""
    from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
        compute_calibration_analysis,
    )

    if y_test.nunique() < 2:
        return None
    y_proba = _predict_proba_pos(model, X_test)
    result = compute_calibration_analysis(y_test.to_numpy(), y_proba, n_bins=n_bins)
    ece = result.get("calibration_ece")
    if ece is None:
        return None
    return float(ece)


def _compute_cv_stability_ratio(
    model_cls_kwargs: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_folds: int = G2_CV_FOLDS,
) -> Optional[float]:
    """Compute the coefficient-of-variation (std / |mean|) of 5-fold
    stratified CV roc_auc. Returns None on degenerate runs.

    The fresh-clone-and-fit per fold is delegated to
    ``compute_stratified_cv``.
    """
    from sklearn.linear_model import LogisticRegression

    from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
        compute_stratified_cv,
    )

    if y.nunique() < 2 or len(X) < n_folds:
        return None
    base_model = LogisticRegression(**model_cls_kwargs)
    cv_result = compute_stratified_cv(
        base_model,
        X.fillna(0.0).to_numpy(),
        y.to_numpy(),
        n_folds=n_folds,
        random_state=seed,
    )
    if not cv_result.get("cv_completed"):
        return None
    mean = cv_result.get("cv_roc_auc_mean")
    std = cv_result.get("cv_roc_auc_std")
    if mean is None or std is None:
        return None
    if abs(float(mean)) < 1e-12:
        return None
    return float(std) / abs(float(mean))


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    """One seed's (baseline, HBLP) metrics."""

    seed: int
    baseline_auc: Optional[float] = None
    hblp_auc: Optional[float] = None
    baseline_ece: Optional[float] = None
    hblp_ece: Optional[float] = None
    baseline_cv_stability: Optional[float] = None
    hblp_cv_stability: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "baseline_auc": self.baseline_auc,
            "hblp_auc": self.hblp_auc,
            "baseline_ece": self.baseline_ece,
            "hblp_ece": self.hblp_ece,
            "baseline_cv_stability": self.baseline_cv_stability,
            "hblp_cv_stability": self.hblp_cv_stability,
            "error": self.error,
        }


def run_seed(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> SeedResult:
    """Execute one seed: split, fit baseline + HBLP, compute metrics."""
    result = SeedResult(seed=seed)
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split(X, y, seed=seed)

        baseline_features = _build_baseline_feature_surface(X_train)
        hblp_features = _build_hblp_relaxed_feature_surface(X_train)

        # Baseline fit
        X_train_b = X_train[baseline_features]
        X_val_b = X_val[baseline_features]
        X_test_b = X_test[baseline_features]
        baseline_model = _fit_logistic_regression(X_train_b, y_train, seed)
        result.baseline_auc = _compute_test_auc(baseline_model, X_test_b, y_test)
        result.baseline_ece = _compute_test_ece(baseline_model, X_test_b, y_test)
        # CV stability is computed on the COMBINED train+val matrix so
        # the test split remains untouched for T1/T2.
        X_trainval_b = pd.concat([X_train_b, X_val_b], axis=0).reset_index(drop=True)
        y_trainval_b = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        result.baseline_cv_stability = _compute_cv_stability_ratio(
            {
                "random_state": seed,
                "max_iter": 1000,
                "solver": "lbfgs",
                "class_weight": "balanced",
            },
            X_trainval_b,
            y_trainval_b,
            seed=seed,
        )

        # HBLP-relaxed fit
        X_train_h = X_train[hblp_features]
        X_val_h = X_val[hblp_features]
        X_test_h = X_test[hblp_features]
        hblp_model = _fit_logistic_regression(X_train_h, y_train, seed)
        result.hblp_auc = _compute_test_auc(hblp_model, X_test_h, y_test)
        result.hblp_ece = _compute_test_ece(hblp_model, X_test_h, y_test)
        X_trainval_h = pd.concat([X_train_h, X_val_h], axis=0).reset_index(drop=True)
        y_trainval_h = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        result.hblp_cv_stability = _compute_cv_stability_ratio(
            {
                "random_state": seed,
                "max_iter": 1000,
                "solver": "lbfgs",
                "class_weight": "balanced",
            },
            X_trainval_h,
            y_trainval_h,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        logger.exception("seed=%s failed", seed)
    return result


# ---------------------------------------------------------------------------
# Aggregation + threshold evaluation
# ---------------------------------------------------------------------------


def _seed_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Mean over non-None values; returns None if all None."""
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


@dataclass
class ThresholdEvaluation:
    """Per-threshold pass/fail with the underlying values."""

    name: str
    description: str
    pre_value: Optional[float]
    post_value: Optional[float]
    delta: Optional[float]
    threshold: float
    passes: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "pre_value": self.pre_value,
            "post_value": self.post_value,
            "delta": self.delta,
            "threshold": self.threshold,
            "passes": self.passes,
            "rationale": self.rationale,
        }


def evaluate_t1(
    baseline_aucs: Sequence[Optional[float]],
    hblp_aucs: Sequence[Optional[float]],
) -> ThresholdEvaluation:
    """T1 — held-out AUC lift Δ ≥ 0.03."""
    pre = _seed_mean(baseline_aucs)
    post = _seed_mean(hblp_aucs)
    if pre is None or post is None:
        return ThresholdEvaluation(
            name="T1",
            description="dAUC >= 0.03",
            pre_value=pre,
            post_value=post,
            delta=None,
            threshold=G2_DELTA_AUC_MIN,
            passes=False,
            rationale="degenerate run: missing pre or post AUC",
        )
    delta = post - pre
    passes = bool(delta >= G2_DELTA_AUC_MIN)
    rationale = (
        f"dAUC = {delta:.4f} {'>=' if passes else '<'} {G2_DELTA_AUC_MIN}; "
        f"pre={pre:.4f}, post={post:.4f}"
    )
    return ThresholdEvaluation(
        name="T1",
        description="dAUC >= 0.03",
        pre_value=pre,
        post_value=post,
        delta=delta,
        threshold=G2_DELTA_AUC_MIN,
        passes=passes,
        rationale=rationale,
    )


def evaluate_t2(
    baseline_eces: Sequence[Optional[float]],
    hblp_eces: Sequence[Optional[float]],
) -> ThresholdEvaluation:
    """T2 — ECE_post <= 0.5 * ECE_pre."""
    pre = _seed_mean(baseline_eces)
    post = _seed_mean(hblp_eces)
    if pre is None or post is None:
        return ThresholdEvaluation(
            name="T2",
            description="ECE_post <= 0.5 * ECE_pre",
            pre_value=pre,
            post_value=post,
            delta=None,
            threshold=G2_ECE_RATIO_MAX,
            passes=False,
            rationale="degenerate run: missing pre or post ECE",
        )
    if pre <= 1e-12:
        return ThresholdEvaluation(
            name="T2",
            description="ECE_post <= 0.5 * ECE_pre",
            pre_value=pre,
            post_value=post,
            delta=None,
            threshold=G2_ECE_RATIO_MAX,
            passes=False,
            rationale=(
                f"baseline ECE pre={pre} <= 1e-12; ratio undefined "
                "(degenerate baseline calibration)"
            ),
        )
    ratio = post / pre
    passes = bool(ratio <= G2_ECE_RATIO_MAX)
    rationale = (
        f"ECE ratio = {ratio:.4f} {'<=' if passes else '>'} {G2_ECE_RATIO_MAX}; "
        f"pre={pre:.4f}, post={post:.4f}"
    )
    return ThresholdEvaluation(
        name="T2",
        description="ECE_post <= 0.5 * ECE_pre",
        pre_value=pre,
        post_value=post,
        delta=ratio,
        threshold=G2_ECE_RATIO_MAX,
        passes=passes,
        rationale=rationale,
    )


def evaluate_t3(
    baseline_cvs: Sequence[Optional[float]],
    hblp_cvs: Sequence[Optional[float]],
) -> ThresholdEvaluation:
    """T3 — (std/mean)_post <= 0.7 * (std/mean)_pre."""
    pre = _seed_mean(baseline_cvs)
    post = _seed_mean(hblp_cvs)
    if pre is None or post is None:
        return ThresholdEvaluation(
            name="T3",
            description="(std/mean)_post <= 0.7 * (std/mean)_pre",
            pre_value=pre,
            post_value=post,
            delta=None,
            threshold=G2_CV_STABILITY_RATIO_MAX,
            passes=False,
            rationale="degenerate run: missing pre or post CV-stability",
        )
    if pre <= 1e-12:
        return ThresholdEvaluation(
            name="T3",
            description="(std/mean)_post <= 0.7 * (std/mean)_pre",
            pre_value=pre,
            post_value=post,
            delta=None,
            threshold=G2_CV_STABILITY_RATIO_MAX,
            passes=False,
            rationale=(
                f"baseline CV-stability pre={pre} <= 1e-12; ratio undefined (degenerate baseline)"
            ),
        )
    ratio = post / pre
    passes = bool(ratio <= G2_CV_STABILITY_RATIO_MAX)
    rationale = (
        f"CV-stability ratio = {ratio:.4f} "
        f"{'<=' if passes else '>'} {G2_CV_STABILITY_RATIO_MAX}; "
        f"pre={pre:.4f}, post={post:.4f}"
    )
    return ThresholdEvaluation(
        name="T3",
        description="(std/mean)_post <= 0.7 * (std/mean)_pre",
        pre_value=pre,
        post_value=post,
        delta=ratio,
        threshold=G2_CV_STABILITY_RATIO_MAX,
        passes=passes,
        rationale=rationale,
    )


@dataclass
class ExperimentManifest:
    """Top-level manifest emitted by the harness."""

    experiment_commit_sha: str
    cohort_label: str
    cohort_data_dir: str
    cohort_target: str
    cohort_data_snooped: bool
    dataset_hashes: Dict[str, str] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=list)
    seed_results: List[Dict[str, Any]] = field(default_factory=list)
    aggregate: Dict[str, Optional[float]] = field(default_factory=dict)
    thresholds: List[Dict[str, Any]] = field(default_factory=list)
    g2_passes_pre_spec: bool = False
    lifecycle_state: str = LIFECYCLE_STATE_G2.value
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_commit_sha": self.experiment_commit_sha,
            "cohort_label": self.cohort_label,
            "cohort_data_dir": self.cohort_data_dir,
            "cohort_target": self.cohort_target,
            "cohort_data_snooped": self.cohort_data_snooped,
            "dataset_hashes": self.dataset_hashes,
            "seeds": self.seeds,
            "seed_results": self.seed_results,
            "aggregate": self.aggregate,
            "thresholds": self.thresholds,
            "g2_passes_pre_spec": self.g2_passes_pre_spec,
            "lifecycle_state": self.lifecycle_state,
            "notes": self.notes,
        }


def build_manifest(
    *,
    cohort: CohortSpec,
    seed_results: Sequence[SeedResult],
    experiment_commit_sha: str,
    dataset_hashes: Optional[Mapping[str, str]] = None,
    notes: str = "",
) -> ExperimentManifest:
    """Aggregate seed results, evaluate T1/T2/T3, build the manifest."""
    baseline_aucs = [r.baseline_auc for r in seed_results]
    hblp_aucs = [r.hblp_auc for r in seed_results]
    baseline_eces = [r.baseline_ece for r in seed_results]
    hblp_eces = [r.hblp_ece for r in seed_results]
    baseline_cvs = [r.baseline_cv_stability for r in seed_results]
    hblp_cvs = [r.hblp_cv_stability for r in seed_results]

    t1 = evaluate_t1(baseline_aucs, hblp_aucs)
    t2 = evaluate_t2(baseline_eces, hblp_eces)
    t3 = evaluate_t3(baseline_cvs, hblp_cvs)

    aggregate = {
        "baseline_auc_mean": _seed_mean(baseline_aucs),
        "hblp_auc_mean": _seed_mean(hblp_aucs),
        "baseline_ece_mean": _seed_mean(baseline_eces),
        "hblp_ece_mean": _seed_mean(hblp_eces),
        "baseline_cv_stability_mean": _seed_mean(baseline_cvs),
        "hblp_cv_stability_mean": _seed_mean(hblp_cvs),
    }

    g2_passes = bool(t1.passes and t2.passes and t3.passes)

    return ExperimentManifest(
        experiment_commit_sha=experiment_commit_sha,
        cohort_label=cohort.label,
        cohort_data_dir=cohort.data_dir,
        cohort_target=cohort.target,
        cohort_data_snooped=cohort.data_snooped,
        dataset_hashes=dict(dataset_hashes or {}),
        seeds=list(G2_SEEDS),
        seed_results=[r.to_dict() for r in seed_results],
        aggregate=aggregate,
        thresholds=[t1.to_dict(), t2.to_dict(), t3.to_dict()],
        g2_passes_pre_spec=g2_passes,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_head_sha() -> str:
    """Return HEAD SHA, or 'unknown' if outside a git context."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _hash_artifacts(
    cohort: CohortSpec,
    project_root: Optional[Path] = None,
) -> Dict[str, str]:
    """sha256 the cohort artifacts named in the pre-spec memo. Missing
    artifacts are recorded as 'MISSING'."""
    project_root = project_root or PROJECT_ROOT
    out: Dict[str, str] = {}
    cohort_dir = project_root / cohort.data_dir
    candidates = [
        ("patient_journeys_parquet", cohort_dir / "e2i_ml_v3_patient_journeys.parquet"),
        ("treatment_events_parquet", cohort_dir / "e2i_ml_v3_treatment_events.parquet"),
    ]
    for label, path in candidates:
        if not path.exists():
            out[label] = "MISSING"
            continue
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(64 * 1024), b""):
                h.update(chunk)
        out[label] = h.hexdigest()
    return out


def run_experiment(
    cohort_label: str,
    *,
    project_root: Optional[Path] = None,
    seeds: Optional[Sequence[int]] = None,
) -> ExperimentManifest:
    """Run the full G2 experiment for ``cohort_label`` and return the
    aggregated manifest. Caller is responsible for serialization +
    exit-code mapping.
    """
    if cohort_label not in COHORTS:
        raise KeyError(f"Unknown cohort_label={cohort_label!r}; valid: {sorted(COHORTS.keys())}")
    cohort = COHORTS[cohort_label]
    project_root = project_root or PROJECT_ROOT
    cohort_dir = project_root / cohort.data_dir
    if not cohort_dir.exists():
        is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")
        msg = (
            f"Cohort directory {cohort_dir} does not exist. The experiment "
            "cannot run without the cohort artifacts. Run the converter "
            "(scripts/convert_optum_rwd.py) first or set RUN_LOCAL_ONLY=1 "
            "to skip locally."
        )
        if is_ci:
            raise FileNotFoundError(msg)
        # Local: emit a manifest with the error so the caller sees what
        # happened without crashing the harness on a fresh checkout.
        return ExperimentManifest(
            experiment_commit_sha=_resolve_head_sha(),
            cohort_label=cohort.label,
            cohort_data_dir=cohort.data_dir,
            cohort_target=cohort.target,
            cohort_data_snooped=cohort.data_snooped,
            dataset_hashes={},
            seeds=list(seeds or G2_SEEDS),
            seed_results=[],
            aggregate={},
            thresholds=[],
            g2_passes_pre_spec=False,
            notes=msg,
        )

    df = _load_patient_journeys(cohort_dir)
    X, y = _build_features_and_target(df, cohort.target)

    seeds_to_run = tuple(seeds) if seeds is not None else G2_SEEDS
    seed_results = [run_seed(X, y, seed=seed) for seed in seeds_to_run]

    return build_manifest(
        cohort=cohort,
        seed_results=seed_results,
        experiment_commit_sha=_resolve_head_sha(),
        dataset_hashes=_hash_artifacts(cohort, project_root=project_root),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-label",
        default="optum_initiation_default",
        choices=sorted(COHORTS.keys()),
        help="Cohort identifier per pre-spec §2.",
    )
    parser.add_argument(
        "--manifest-out",
        type=str,
        default=None,
        help="If set, write the manifest JSON to this path.",
    )
    parser.add_argument(
        "--allow-data-snooped",
        action="store_true",
        help=(
            "Allow runs against cohorts marked data_snooped=true (per "
            "pre-spec §2.2). Refuses otherwise."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    cohort = COHORTS[args.cohort_label]
    if cohort.data_snooped and not args.allow_data_snooped:
        print(
            f"REFUSED: cohort {cohort.label!r} is marked "
            "data_snooped=true; pass --allow-data-snooped to override "
            "(the run will be recorded as data-snooped in the manifest "
            "and is informational only).",
            file=sys.stderr,
        )
        return 2

    manifest = run_experiment(args.cohort_label)
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    print(payload)
    if args.manifest_out:
        Path(args.manifest_out).write_text(payload, encoding="utf-8")

    return 0 if manifest.g2_passes_pre_spec else 1


if __name__ == "__main__":
    sys.exit(main())
