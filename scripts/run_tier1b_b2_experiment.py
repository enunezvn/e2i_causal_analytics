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

Modeling note (HBLP toggle — REAL CONTRAST)
-------------------------------------------

The "HBLP-disabled" (baseline) vs "HBLP-relaxed" (post) toggle in this
harness is a **feature-retention** difference materialized BEFORE the
classifier sees the matrix:

  1. For each numeric feature, the harness computes a marginal
     z-score against the binary target (the same statistic the
     production Layer 3 detector uses, simplified to a single-pass
     Welch's-t-style z over the standardized correlation).
  2. The **baseline arm** applies the legacy strict policy from
     ``adaptive_validity_check.py``: any feature with ``z > HIGH_Z``
     (= 5.0σ) is treated as a leak and DROPPED. ``HIGH_Z`` is the
     production constant, NOT a harness invention.
  3. The **HBLP arm** applies the production ``hblp_classify`` from
     the same module: variance-inflation + Layer-1-conditional priors
     produce a higher effective threshold for low-N or
     ``layer_1_declared_safe=True`` features. Only features whose
     post-HBLP severity is ``"high"`` are dropped.

Both arms then fit the same LogisticRegression on the resulting
feature subsets. Feature retention diverges → metric arrays diverge →
ΔAUC / ECE-ratio / CV-stability-ratio are all real per-arm contrasts,
not 0-by-construction.

The harness contract: when a feature has high marginal z (would be
dropped by baseline) but ``layer_1_declared_safe=True`` and small
``n_train_pos`` (variance-inflated), HBLP retains it. This is the
production behavior that G2 measures.

``layer_1_declared_safe`` per feature is sourced from the
``MANIFEST_SOURCES`` registry when a manifest source is provided; a
test-injectable ``layer_1_declared_safe_lookup`` argument lets the
unit suite construct a known-divergent example without depending on
manifest content.

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
    """Cohort identity per pre-spec §2.

    HIGH-7 fix: ``expected_n_exact`` pins the EXACT observed patient
    count for each cohort. The harness asserts the loaded cohort's
    n equals this value; mismatch is a hard refusal regardless of
    which label was passed. Both cohorts point at the same directory
    (different converter regimes produce different parquets there),
    so a label-only refusal can be bypassed by writing the relaxed
    n=1697 parquet under the default label.

    HIGH-1 fix (iter-3): ``manifest_source`` declares the data source
    string used to look up Layer-1 ``declared_safe`` contracts in the
    ``MANIFEST_SOURCES`` registry. Each cohort's manifest is
    declarative — Optum cohorts use ``"optum"``, CSU cohorts use
    ``"csu"``. The harness wires this into ``run_seed`` so the
    HBLP-relaxed arm sees a real Layer-1 lookup (not the all-False
    default), which is the load-bearing precondition for the HIGH-1
    baseline-vs-HBLP retention contrast.
    """

    label: str
    data_dir: str
    target: str
    expected_n_exact: int
    data_snooped: bool
    manifest_source: Optional[str] = None


COHORTS: Mapping[str, CohortSpec] = {
    # HIGH-7: n=1294 is the load-bearing default-window cohort
    # (PRE=365/POST=180). Mismatch on this exact value indicates the
    # converter ran with non-default parameters or the cohort was
    # rebuilt — either way, the data is no longer the cohort the
    # pre-spec memo locked.
    # HIGH-1 (iter-3): manifest_source="optum" wires the Layer-1
    # declared-safe lookup against the Optum manifest registry.
    "optum_initiation_default": CohortSpec(
        label="optum_initiation_default",
        data_dir="data/rwd/optum/initiation",
        target="treatment_initiated",
        expected_n_exact=1294,
        data_snooped=False,
        manifest_source="optum",
    ),
    # HIGH-7: n=1697 is the relaxed-window cohort (PRE=180/POST=90),
    # marked data_snooped=true. Available behind --allow-data-snooped
    # for diagnostic comparison only.
    "optum_initiation_relaxed": CohortSpec(
        label="optum_initiation_relaxed",
        data_dir="data/rwd/optum/initiation",
        target="treatment_initiated",
        expected_n_exact=1697,
        data_snooped=True,
        manifest_source="optum",
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
# Baseline + HBLP-relaxed feature surfaces — REAL CONTRAST
#
# The contrast lives at feature retention: both arms compute the same
# per-feature marginal z-score against the target; the baseline applies
# the legacy strict ``z > HIGH_Z`` policy from adaptive_validity_check;
# the HBLP arm applies ``hblp_classify`` with ``n_train_pos`` and
# ``layer_1_declared_safe`` per feature.
# ---------------------------------------------------------------------------


def _compute_marginal_z_scores(
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """Compute a per-feature marginal z-score against the binary target.

    Implementation: Welch's-t-style two-sample standardized difference
    between feature means in y=1 vs y=0, expressed as a z-statistic.
    This is a simplified, deterministic surrogate for the production
    Layer 3 permutation-baseline z-score (which is computationally
    heavier and depends on a permutation null). The harness uses the
    Welch z because:
      (a) it agrees with the production permutation z in regime — both
          rank-correlate with the leakage signal;
      (b) it is deterministic per (X, y) so the contrast is reproducible
          across CI runs;
      (c) it matches the threshold semantics the legacy strict policy
          uses (``z > HIGH_Z`` is the drop boundary).

    Constant features and degenerate y produce z=0.0 (kept by both
    arms — neither classifier learns anything from a constant).
    """
    z_scores: Dict[str, float] = {}
    y_arr = y.to_numpy(dtype=np.float64)
    pos_mask = y_arr > 0.5
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos < 2 or n_neg < 2:
        # Degenerate target — every z is 0.0 so neither arm drops
        # anything based on the marginal-z policy.
        return dict.fromkeys(X.columns, 0.0)

    for col in X.columns:
        x = X[col].to_numpy(dtype=np.float64)
        # Drop NaN rows symmetrically across feature + target.
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            z_scores[col] = 0.0
            continue
        xv = x[finite_mask]
        yv = y_arr[finite_mask]
        pmask = yv > 0.5
        nmask = ~pmask
        if pmask.sum() < 2 or nmask.sum() < 2:
            z_scores[col] = 0.0
            continue
        x_pos = xv[pmask]
        x_neg = xv[nmask]
        var_pos = float(np.var(x_pos, ddof=1)) if x_pos.size > 1 else 0.0
        var_neg = float(np.var(x_neg, ddof=1)) if x_neg.size > 1 else 0.0
        # Welch SE on the difference of means.
        se = float(np.sqrt(var_pos / max(x_pos.size, 1) + var_neg / max(x_neg.size, 1)))
        if se <= 1e-12 or not np.isfinite(se):
            z_scores[col] = 0.0
            continue
        diff = float(np.mean(x_pos) - np.mean(x_neg))
        z = abs(diff) / se
        z_scores[col] = float(z) if np.isfinite(z) else 0.0
    return z_scores


def _legacy_strict_drop(z_scores: Mapping[str, float]) -> List[str]:
    """Baseline arm — legacy strict policy from
    ``adaptive_validity_check._adversarial_input``: ``z > HIGH_Z``
    drops the feature.

    ``HIGH_Z`` is imported from the production module so the harness
    drift-detects against the same constant the pipeline uses.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        HIGH_Z,
    )

    return [feat for feat, z in z_scores.items() if z > HIGH_Z]


def _hblp_drop(
    z_scores: Mapping[str, float],
    *,
    n_train_pos: int,
    layer_1_declared_safe_lookup: Mapping[str, bool],
) -> List[str]:
    """HBLP arm — production ``hblp_classify`` policy from
    ``adaptive_validity_check``. A feature is dropped iff its post-HBLP
    severity is ``"high"`` (i.e., its z exceeds the variance-inflated +
    Layer-1-conditional threshold).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        hblp_classify,
    )

    dropped: List[str] = []
    for feat, z in z_scores.items():
        declared_safe = bool(layer_1_declared_safe_lookup.get(feat, False))
        verdict = hblp_classify(
            z_score=float(z),
            n_positives=int(n_train_pos),
            layer_1_declared_safe=declared_safe,
        )
        if verdict["severity"] == "high":
            dropped.append(feat)
    return dropped


def _resolve_layer_1_declared_safe_lookup(
    feature_names: Sequence[str],
    *,
    manifest_source: Optional[str] = None,
    explicit_lookup: Optional[Mapping[str, bool]] = None,
) -> Dict[str, bool]:
    """Resolve a per-feature ``layer_1_declared_safe`` mapping.

    Resolution order:
      1. If ``explicit_lookup`` is provided (test injection), use it
         verbatim — features missing default to False.
      2. If ``manifest_source`` is provided, query
         ``MANIFEST_SOURCES`` for each feature; ``declared_safe=True``
         iff a contract exists with a pre-anchor ``knowable_at``
         reference. Manifest unavailable → all features default False.
      3. Otherwise, all features default False (the most conservative
         baseline-equivalent assumption).

    The harness intentionally treats "no manifest information" as
    ``declared_safe=False`` — this is the safe default that makes the
    HBLP arm's behavior match the baseline arm in the absence of
    Layer-1 information, ensuring the contrast only diverges where
    Layer-1 actually has something to say.
    """
    if explicit_lookup is not None:
        return {feat: bool(explicit_lookup.get(feat, False)) for feat in feature_names}

    out = dict.fromkeys(feature_names, False)
    if manifest_source is None:
        return out

    try:
        from src.data.manifests import MANIFEST_SOURCES
    except ImportError:
        return out

    if manifest_source not in MANIFEST_SOURCES:
        return out

    pre_anchor_refs = {"index_date", "lookback_start_date", "eligeff"}
    for feat in feature_names:
        try:
            contract = MANIFEST_SOURCES[manifest_source](feat)
        except Exception:  # noqa: BLE001 — defensive against manifest bugs
            contract = None
        if contract is None:
            continue
        knowable_ref = getattr(getattr(contract, "knowable_at", None), "reference", None)
        out[feat] = knowable_ref in pre_anchor_refs
    return out


def _layer_1_post_anchor_feature_drop(
    X: pd.DataFrame,
    *,
    manifest_source: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop features declared ``knowable_at=post_index`` (or any
    non-pre-anchor reference) per the cohort manifest.

    Returns ``(X_filtered, dropped_feature_names)``.

    When ``manifest_source`` is None, returns X unchanged with an
    empty dropped list (back-compat for callers that have no cohort
    manifest; the per-seed Layer 3 scorer is the fallback).

    When ``manifest_source`` IS provided, the filter FAILS CLOSED:
      - An import failure of ``MANIFEST_SOURCES`` raises RuntimeError.
      - An unregistered ``manifest_source`` raises RuntimeError.
      - Manifest lookup exceptions per feature are also raised (not
        silently swallowed), so a manifest bug causes a hard failure
        rather than silently keeping a post-anchor leak feature.
    This fail-closed contract prevents a typo in ``manifest_source``
    or a manifest-registry regression from silently reintroducing the
    AUC=1.0 failure mode that Layer 3 cannot detect (zero-variance
    blind spot on perfect binary proxies).

    A feature is dropped iff a manifest contract is found AND the
    contract's ``knowable_at`` is NOT pre-or-at-index
    (``is_pre_or_at_index()`` returns False).

    Features with NO manifest contract are KEPT (unknown ≠ forbidden).
    This is the conservative behaviour: only manifest-declared
    post-anchor features are dropped; unknown derivations are left for
    the per-seed Layer 3 scorer.

    Case handling: column names are looked up against the manifest by
    their exact (already lower-snake-case) name.  If a column has NO
    exact-match contract but a case-folded version would match a
    manifest name, a RuntimeError is raised — this catches upstream
    casing drift before it silently bypasses the filter.
    """
    if manifest_source is None:
        return X, []

    try:
        from src.data.manifests import MANIFEST_SOURCES
    except ImportError as exc:
        raise RuntimeError(
            f"Layer 1 pre-filter: failed to import MANIFEST_SOURCES "
            f"(manifest_source={manifest_source!r}). "
            "This import must succeed for the filter to be fail-closed."
        ) from exc

    if manifest_source not in MANIFEST_SOURCES:
        raise RuntimeError(
            f"Layer 1 pre-filter: manifest_source={manifest_source!r} is not "
            f"registered in MANIFEST_SOURCES (known: {sorted(MANIFEST_SOURCES.keys())}). "
            "Fix the cohort spec or register the manifest."
        )

    lookup_fn = MANIFEST_SOURCES[manifest_source]

    # Build a lower-case → original-name map for case-drift detection.
    lower_to_col: Dict[str, str] = {}
    for col in X.columns:
        lc = col.lower()
        if lc in lower_to_col and lower_to_col[lc] != col:
            raise RuntimeError(
                f"Layer 1 pre-filter: ambiguous case collision in feature matrix: "
                f"{col!r} and {lower_to_col[lc]!r} both fold to {lc!r}. "
                "Resolve before running the harness."
            )
        lower_to_col[lc] = col

    dropped: List[str] = []
    for col in X.columns:
        contract = lookup_fn(col)
        if contract is None:
            # Exact-name miss — check for case-drift.
            lc = col.lower()
            if lc != col:
                # col is not already lower-case; try lower-case lookup.
                contract_lc = lookup_fn(lc)
                if contract_lc is not None:
                    raise RuntimeError(
                        f"Layer 1 pre-filter: column {col!r} has no exact manifest "
                        f"contract but its lower-case form {lc!r} does. This indicates "
                        "upstream casing drift that would silently bypass the filter. "
                        "Fix the converter to emit lower-snake-case column names."
                    )
            # No manifest entry (exact or case-folded) — keep.
            continue
        knowable = getattr(contract, "knowable_at", None)
        if knowable is None:
            continue
        if not knowable.is_pre_or_at_index():
            dropped.append(col)

    if not dropped:
        return X, []

    return X.drop(columns=dropped), dropped


def _build_baseline_feature_surface(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[List[str], Dict[str, float], List[str]]:
    """Baseline arm's retained feature list + the z-score map +
    the explicit drop list (kept for manifest provenance).

    Returns ``(retained_features, z_scores, dropped_features)``.
    """
    z_scores = _compute_marginal_z_scores(X_train, y_train)
    dropped = _legacy_strict_drop(z_scores)
    retained = [c for c in X_train.columns if c not in set(dropped)]
    return retained, z_scores, dropped


def _build_hblp_relaxed_feature_surface(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    layer_1_declared_safe_lookup: Mapping[str, bool],
    z_scores: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], Dict[str, float], List[str]]:
    """HBLP arm's retained feature list + the z-score map +
    the explicit drop list.

    Returns ``(retained_features, z_scores, dropped_features)``.

    If ``z_scores`` is provided, it is reused (so both arms see the
    same statistic and the contrast is purely at the policy layer).
    """
    if z_scores is None:
        z_scores = _compute_marginal_z_scores(X_train, y_train)
    n_train_pos = int((y_train > 0.5).sum())
    dropped = _hblp_drop(
        z_scores,
        n_train_pos=n_train_pos,
        layer_1_declared_safe_lookup=layer_1_declared_safe_lookup,
    )
    retained = [c for c in X_train.columns if c not in set(dropped)]
    return retained, dict(z_scores), dropped


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
    """One seed's (baseline, HBLP) metrics + retention provenance."""

    seed: int
    baseline_auc: Optional[float] = None
    hblp_auc: Optional[float] = None
    baseline_ece: Optional[float] = None
    hblp_ece: Optional[float] = None
    baseline_cv_stability: Optional[float] = None
    hblp_cv_stability: Optional[float] = None
    error: Optional[str] = None
    # Retention provenance — surface so the manifest can audit which
    # features each arm dropped (proves the contrast is non-trivial).
    n_train_pos: Optional[int] = None
    baseline_n_features_retained: Optional[int] = None
    hblp_n_features_retained: Optional[int] = None
    baseline_features_dropped: List[str] = field(default_factory=list)
    hblp_features_dropped: List[str] = field(default_factory=list)
    features_diverged: List[str] = field(default_factory=list)

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
            "n_train_pos": self.n_train_pos,
            "baseline_n_features_retained": self.baseline_n_features_retained,
            "hblp_n_features_retained": self.hblp_n_features_retained,
            "baseline_features_dropped": list(self.baseline_features_dropped),
            "hblp_features_dropped": list(self.hblp_features_dropped),
            "features_diverged": list(self.features_diverged),
        }

    def has_complete_metrics(self) -> bool:
        """True iff all six metrics are finite + error is None."""
        if self.error is not None:
            return False
        for v in (
            self.baseline_auc,
            self.hblp_auc,
            self.baseline_ece,
            self.hblp_ece,
            self.baseline_cv_stability,
            self.hblp_cv_stability,
        ):
            if v is None or not np.isfinite(float(v)):
                return False
        return True


def run_seed(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    *,
    layer_1_declared_safe_lookup: Optional[Mapping[str, bool]] = None,
    manifest_source: Optional[str] = None,
) -> SeedResult:
    """Execute one seed: split, build baseline + HBLP feature surfaces
    via the legacy strict / ``hblp_classify`` policies, fit, and
    compute metrics.

    Two contrast arms diverge at feature retention. The classifier and
    preprocessing are otherwise identical so the only signal in the
    metric arrays is the policy difference.
    """
    result = SeedResult(seed=seed)
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split(X, y, seed=seed)

        n_train_pos = int((y_train > 0.5).sum())
        result.n_train_pos = n_train_pos
        lookup = _resolve_layer_1_declared_safe_lookup(
            list(X_train.columns),
            manifest_source=manifest_source,
            explicit_lookup=layer_1_declared_safe_lookup,
        )

        baseline_features, z_scores, baseline_dropped = _build_baseline_feature_surface(
            X_train, y_train
        )
        hblp_features, _, hblp_dropped = _build_hblp_relaxed_feature_surface(
            X_train,
            y_train,
            layer_1_declared_safe_lookup=lookup,
            z_scores=z_scores,
        )

        result.baseline_n_features_retained = len(baseline_features)
        result.hblp_n_features_retained = len(hblp_features)
        result.baseline_features_dropped = list(baseline_dropped)
        result.hblp_features_dropped = list(hblp_dropped)
        # "Diverged" = features the baseline drops but HBLP retains.
        # This is the load-bearing set whose presence proves the
        # contrast is non-trivial.
        result.features_diverged = sorted(set(baseline_dropped) - set(hblp_dropped))

        if not baseline_features or not hblp_features:
            # At least one arm dropped every feature — degenerate by
            # construction. Surface as error so the all-seeds-required
            # gate fails the manifest.
            raise RuntimeError(
                f"Degenerate retention: baseline_n={len(baseline_features)}, "
                f"hblp_n={len(hblp_features)}; one arm dropped every feature"
            )

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


# LOW-11 — ECE method metadata for the manifest. Records the helper
# path, bin count, and binning strategy so a future change to
# compute_calibration_analysis is detectable in the audit trail.
ECE_METHOD_METADATA: Dict[str, Any] = {
    "helper_module": "src.agents.ml_foundation.model_trainer.nodes.advanced_validation",
    "helper_function": "compute_calibration_analysis",
    "n_bins": G2_ECE_BINS,
    "binning_strategy": "equal-width",
    "notes": (
        "ECE = sum(|empirical_freq - mean_pred_prob|) over equal-width bins "
        "of predicted probability. Helper version is captured by the experiment "
        "commit SHA in the manifest's experiment_commit_sha field."
    ),
}


def _seed_completeness_diagnostic(
    seed_results: Sequence[SeedResult],
    *,
    expected_seeds: Sequence[int] = G2_SEEDS,
) -> Dict[str, Any]:
    """HIGH-8 fix: build the diagnostic record + pass/fail flag for
    the all-seeds-required gate.

    Returns a dict with:
      * ``all_seeds_complete``: bool — True iff every expected seed
        appears AND has ``has_complete_metrics() == True``.
      * ``expected_seeds``: list[int]
      * ``observed_seeds``: list[int]
      * ``missing_seeds``: list[int]
      * ``incomplete_seeds``: list[int]
      * ``per_seed_diagnostic``: list[dict]
    """
    expected_set = set(expected_seeds)
    observed_seeds = [r.seed for r in seed_results]
    observed_set = set(observed_seeds)
    missing_seeds = sorted(expected_set - observed_set)
    incomplete_seeds: List[int] = []
    per_seed_diagnostic: List[Dict[str, Any]] = []
    for r in seed_results:
        complete = r.has_complete_metrics()
        if not complete:
            incomplete_seeds.append(r.seed)
        per_seed_diagnostic.append(
            {
                "seed": r.seed,
                "complete": complete,
                "error": r.error,
                "has_baseline_auc": r.baseline_auc is not None,
                "has_hblp_auc": r.hblp_auc is not None,
                "has_baseline_ece": r.baseline_ece is not None,
                "has_hblp_ece": r.hblp_ece is not None,
                "has_baseline_cv": r.baseline_cv_stability is not None,
                "has_hblp_cv": r.hblp_cv_stability is not None,
            }
        )
    all_complete = not missing_seeds and not incomplete_seeds and observed_set == expected_set
    return {
        "all_seeds_complete": all_complete,
        "expected_seeds": list(expected_seeds),
        "observed_seeds": observed_seeds,
        "missing_seeds": missing_seeds,
        "incomplete_seeds": sorted(incomplete_seeds),
        "per_seed_diagnostic": per_seed_diagnostic,
    }


@dataclass
class ExperimentManifest:
    """Top-level manifest emitted by the harness."""

    experiment_commit_sha: str
    cohort_label: str
    cohort_data_dir: str
    cohort_target: str
    cohort_data_snooped: bool
    cohort_expected_n_exact: int = 0
    cohort_observed_n: Optional[int] = None
    dataset_hashes: Dict[str, str] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=list)
    seed_results: List[Dict[str, Any]] = field(default_factory=list)
    aggregate: Dict[str, Optional[float]] = field(default_factory=dict)
    thresholds: List[Dict[str, Any]] = field(default_factory=list)
    g2_passes_pre_spec: bool = False
    lifecycle_state: str = LIFECYCLE_STATE_G2.value
    ece_method_metadata: Dict[str, Any] = field(default_factory=lambda: dict(ECE_METHOD_METADATA))
    seed_completeness: Dict[str, Any] = field(default_factory=dict)
    layer_1_dropped_features: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_commit_sha": self.experiment_commit_sha,
            "cohort_label": self.cohort_label,
            "cohort_data_dir": self.cohort_data_dir,
            "cohort_target": self.cohort_target,
            "cohort_data_snooped": self.cohort_data_snooped,
            "cohort_expected_n_exact": self.cohort_expected_n_exact,
            "cohort_observed_n": self.cohort_observed_n,
            "dataset_hashes": self.dataset_hashes,
            "seeds": self.seeds,
            "seed_results": self.seed_results,
            "aggregate": self.aggregate,
            "thresholds": self.thresholds,
            "g2_passes_pre_spec": self.g2_passes_pre_spec,
            "lifecycle_state": self.lifecycle_state,
            "ece_method_metadata": self.ece_method_metadata,
            "seed_completeness": self.seed_completeness,
            "layer_1_dropped_features": list(self.layer_1_dropped_features),
            "notes": self.notes,
        }


def build_manifest(
    *,
    cohort: CohortSpec,
    seed_results: Sequence[SeedResult],
    experiment_commit_sha: str,
    dataset_hashes: Optional[Mapping[str, str]] = None,
    cohort_observed_n: Optional[int] = None,
    layer_1_dropped_features: Optional[List[str]] = None,
    notes: str = "",
) -> ExperimentManifest:
    """Aggregate seed results, evaluate T1/T2/T3, build the manifest.

    HIGH-8 fix: ALL five seeds must have ``error is None`` AND all six
    metrics finite for threshold evaluation to proceed. Otherwise the
    threshold list records a hard failure with a diagnostic and
    ``g2_passes_pre_spec=False``.
    """
    completeness = _seed_completeness_diagnostic(seed_results)

    if not completeness["all_seeds_complete"]:
        diag = (
            "HIGH-8 hard fail: G2 requires ALL "
            f"{len(G2_SEEDS)} seeds × 6 metrics finite before threshold eval. "
            f"missing_seeds={completeness['missing_seeds']}, "
            f"incomplete_seeds={completeness['incomplete_seeds']}"
        )
        # Surface the failure as a non-passing manifest. Threshold list
        # is empty (no evaluation performed). Aggregate captures
        # whatever values are present for diagnostic only.
        return ExperimentManifest(
            experiment_commit_sha=experiment_commit_sha,
            cohort_label=cohort.label,
            cohort_data_dir=cohort.data_dir,
            cohort_target=cohort.target,
            cohort_data_snooped=cohort.data_snooped,
            cohort_expected_n_exact=cohort.expected_n_exact,
            cohort_observed_n=cohort_observed_n,
            dataset_hashes=dict(dataset_hashes or {}),
            seeds=list(G2_SEEDS),
            seed_results=[r.to_dict() for r in seed_results],
            aggregate={
                "baseline_auc_mean": _seed_mean([r.baseline_auc for r in seed_results]),
                "hblp_auc_mean": _seed_mean([r.hblp_auc for r in seed_results]),
                "baseline_ece_mean": _seed_mean([r.baseline_ece for r in seed_results]),
                "hblp_ece_mean": _seed_mean([r.hblp_ece for r in seed_results]),
                "baseline_cv_stability_mean": _seed_mean(
                    [r.baseline_cv_stability for r in seed_results]
                ),
                "hblp_cv_stability_mean": _seed_mean([r.hblp_cv_stability for r in seed_results]),
            },
            thresholds=[],
            g2_passes_pre_spec=False,
            seed_completeness=completeness,
            layer_1_dropped_features=list(layer_1_dropped_features or []),
            notes=(notes + " | " if notes else "") + diag,
        )

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
        cohort_expected_n_exact=cohort.expected_n_exact,
        cohort_observed_n=cohort_observed_n,
        dataset_hashes=dict(dataset_hashes or {}),
        seeds=list(G2_SEEDS),
        seed_results=[r.to_dict() for r in seed_results],
        aggregate=aggregate,
        thresholds=[t1.to_dict(), t2.to_dict(), t3.to_dict()],
        g2_passes_pre_spec=g2_passes,
        seed_completeness=completeness,
        layer_1_dropped_features=list(layer_1_dropped_features or []),
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


def _resolve_load_bearing_flag(load_bearing: Optional[bool]) -> bool:
    """Return whether this run is load-bearing (default: True in CI).

    HIGH-1 (iter-3): a load-bearing run requires real manifest-backed
    Layer-1 declared-safe coverage; an empty lookup is a hard refusal.
    Local diagnostic runs (e.g., when a developer has no cohort on
    disk and is just exercising the harness) can opt out via
    ``--no-fail-on-empty-declared-safe``, in which case ``load_bearing``
    is False.
    """
    if load_bearing is not None:
        return bool(load_bearing)
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def run_experiment(
    cohort_label: str,
    *,
    project_root: Optional[Path] = None,
    seeds: Optional[Sequence[int]] = None,
    layer_1_declared_safe_lookup: Optional[Mapping[str, bool]] = None,
    load_bearing: Optional[bool] = None,
) -> ExperimentManifest:
    """Run the full G2 experiment for ``cohort_label`` and return the
    aggregated manifest. Caller is responsible for serialization +
    exit-code mapping.

    HIGH-1 fix (iter-3): wires ``cohort.manifest_source`` through to
    ``run_seed`` so the HBLP-relaxed arm sees a real Layer-1 lookup
    sourced from ``MANIFEST_SOURCES``. For load-bearing runs (CI=true
    OR explicit ``load_bearing=True``), an empty resolved lookup
    (i.e. no manifest contract declared any feature ``declared_safe``)
    is a HARD REFUSAL — the contrast between baseline and HBLP arms
    would collapse to "same model twice", invalidating G2's metric.

    The optional ``layer_1_declared_safe_lookup`` argument lets tests
    inject a known divergent example without depending on a real
    manifest. When supplied, the manifest-source resolution is
    skipped (the explicit lookup IS the lookup).
    """
    if cohort_label not in COHORTS:
        raise KeyError(f"Unknown cohort_label={cohort_label!r}; valid: {sorted(COHORTS.keys())}")
    cohort = COHORTS[cohort_label]
    project_root = project_root or PROJECT_ROOT
    cohort_dir = project_root / cohort.data_dir
    is_load_bearing = _resolve_load_bearing_flag(load_bearing)
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
            cohort_expected_n_exact=cohort.expected_n_exact,
            cohort_observed_n=None,
            dataset_hashes={},
            seeds=list(seeds or G2_SEEDS),
            seed_results=[],
            aggregate={},
            thresholds=[],
            g2_passes_pre_spec=False,
            notes=msg,
        )

    df = _load_patient_journeys(cohort_dir)

    # HIGH-7 fix — cohort identity enforcement: assert observed n
    # equals the cohort spec's expected_n_exact. The two cohorts
    # (default n=1294, relaxed n=1697) live in the same data_dir; the
    # converter regime determines which one's parquet is on disk. A
    # label-only refusal can be bypassed by writing the relaxed
    # n=1697 parquet under the default label, so we verify the actual
    # patient count loaded from disk.
    observed_n = int(len(df))
    if observed_n != cohort.expected_n_exact:
        raise ValueError(
            f"HIGH-7 cohort identity mismatch: "
            f"cohort_label={cohort.label!r} declares "
            f"expected_n_exact={cohort.expected_n_exact}, but the loaded "
            f"patient_journeys frame has {observed_n} rows. "
            f"Refusing to run with potentially-snooped data — verify the "
            f"converter regime that produced {cohort_dir} matches the "
            f"cohort label."
        )

    X, y = _build_features_and_target(df, cohort.target)

    # Layer 1 manifest pre-filter — applied ONCE per cohort, before the
    # per-seed split loop. Drops every feature whose manifest contract
    # declares ``knowable_at=post_index`` (or any non-pre-anchor reference).
    # This catches target-derived columns (e.g., ``initiated_biologic_180d``
    # in Optum) BEFORE they reach the model fit or the marginal-z scorer,
    # which has a degenerate-variance blind spot on perfect-binary proxies
    # (within-group variance = 0 → Welch SE = 0 → z = 0 → feature kept).
    # The filter is deterministic per manifest — it does not depend on the
    # seed. The dropped list is surfaced in the manifest for audit traceability.
    X, layer_1_dropped = _layer_1_post_anchor_feature_drop(
        X,
        manifest_source=cohort.manifest_source,
    )
    if layer_1_dropped:
        logger.info(
            "Layer 1 pre-filter dropped %d post-anchor features: %s",
            len(layer_1_dropped),
            sorted(layer_1_dropped),
        )

    # HIGH-1 (iter-3) — resolve the Layer-1 declared-safe lookup via
    # the cohort's manifest_source registry entry. For a load-bearing
    # run, fail loudly when the resolved lookup has zero declared-safe
    # features (the HBLP arm would collapse to baseline).
    resolved_lookup = _resolve_layer_1_declared_safe_lookup(
        list(X.columns),
        manifest_source=cohort.manifest_source,
        explicit_lookup=layer_1_declared_safe_lookup,
    )
    n_declared_safe = sum(1 for v in resolved_lookup.values() if v)
    if is_load_bearing and layer_1_declared_safe_lookup is None and n_declared_safe == 0:
        raise RuntimeError(
            f"HIGH-1 hard fail: load-bearing G2 run for "
            f"cohort_label={cohort.label!r} resolved ZERO Layer-1 "
            f"declared-safe features against manifest_source="
            f"{cohort.manifest_source!r}. Without a non-trivial "
            "declared-safe map the HBLP arm cannot diverge from "
            "baseline and the G2 contrast is meaningless. "
            "Resolution: ensure the cohort's manifest_source has "
            "declared-safe contracts (knowable_at in "
            "{index_date, lookback_start_date, eligeff}) for at least "
            "one numeric feature emitted by the converter; OR pass "
            "an explicit layer_1_declared_safe_lookup; OR re-run with "
            "load_bearing=False (diagnostic only)."
        )

    seeds_to_run = tuple(seeds) if seeds is not None else G2_SEEDS
    seed_results = [
        run_seed(X, y, seed=seed, layer_1_declared_safe_lookup=resolved_lookup)
        for seed in seeds_to_run
    ]

    return build_manifest(
        cohort=cohort,
        seed_results=seed_results,
        cohort_observed_n=observed_n,
        experiment_commit_sha=_resolve_head_sha(),
        dataset_hashes=_hash_artifacts(cohort, project_root=project_root),
        layer_1_dropped_features=layer_1_dropped,
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
    parser.add_argument(
        "--no-fail-on-empty-declared-safe",
        action="store_true",
        help=(
            "HIGH-1 (iter-3) escape hatch: bypass the load-bearing "
            "declared-safe non-emptiness check. Diagnostic only — "
            "DO NOT use for the load-bearing CI run; the HBLP arm "
            "collapses to baseline when no manifest contracts apply."
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

    # HIGH-1 (iter-3): default load_bearing to None (auto-detect via
    # CI env) unless the operator explicitly opts out via the flag.
    load_bearing_flag: Optional[bool] = False if args.no_fail_on_empty_declared_safe else None
    manifest = run_experiment(args.cohort_label, load_bearing=load_bearing_flag)
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    print(payload)
    if args.manifest_out:
        Path(args.manifest_out).write_text(payload, encoding="utf-8")

    return 0 if manifest.g2_passes_pre_spec else 1


if __name__ == "__main__":
    sys.exit(main())
