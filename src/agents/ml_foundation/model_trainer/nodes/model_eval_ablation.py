"""Phase 3.4 — Model-trainer evaluator Layer-3 hook (perm + ablation on encoded features).

Plan reference: .claude/plans/adaptive_temporal_validity_redesign.md line 245.

This module wires Layer 3 — BOTH the label-shuffle adversarial score
(``compute_adversarial_score``) AND the joint-model ablation
(``compute_feature_ablation``) — into the model_trainer evaluator alongside
the existing permutation test. It is the COMPLEMENT of the Phase 3.3
data-prep hook
(``src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py``),
NOT a duplicate.

Phase 3.3 evaluates Layer 3 on RAW manifest-level feature columns at data-
prep time, BEFORE the model_trainer preprocessor runs. The acceptance
criterion for Phase 3.4 (codex review, plan line 401) was that the model-
eval hook must surface a leak class that Phase 3.3 cannot see — otherwise
the check is duplicative and the milestone should be closed NULL.

LEAK CLASS THIS HOOK CATCHES (model-eval, not Phase 3.3):

  **Categorical per-category leak through OneHotEncoder.** Phase 3.3's
  Layer-3 statistical pass skips non-numeric columns by design (the
  ``_select_features`` helper at adaptive_validity_check.py:2530 explicitly
  excludes ``not is_numeric_dtype(df[c])``; categorical leak detection is
  delegated to ``check_categorical_class_separation`` in
  ``leakage_detector.py`` which uses Cramér's V on the WHOLE column).

  Cramér's V measures whole-column association. A categorical column
  ``region`` with 11 categories where ONLY one rare category (~12-15%
  of rows) is target-leaky has whole-column Cramér's V below the 0.5
  threshold (the other 10 categories dilute the signal) and is missed
  by ``check_categorical_class_separation``. Phase 3.3 numeric ablation
  skips ``region`` entirely.

  At model-trainer time, ``OneHotEncoder`` (see preprocessor.py:88) splits
  ``region`` into per-category indicators ``region_<category>`` with
  ``verbose_feature_names_out=False``. The single leaky category becomes
  its own binary feature ``region_<rare_category>`` whose single-feature
  AUC is strongly above the label-shuffle permutation null (z ≫ 5σ).
  Per-encoded-column ``compute_adversarial_score`` catches it.

  Why ablation alone is not sufficient (codex pass-1 self-review insight):
  ``compute_feature_ablation`` uses a COLUMN-SHUFFLE null (same null
  construction as the data-prep ablation pass). For a binary indicator,
  shuffling the column preserves the marginal distribution and keeps
  ~``leak_p`` fraction of rows with indicator=1 — just at random
  positions. The joint model trained on the shuffled column has the
  SAME predictive power as the joint model trained without the column,
  so ``null_mean ≈ actual_delta_auc`` and the z-score collapses. This
  is the same column-shuffle null weakness Phase 3.3 addressed with its
  strong-effect escape (|delta_AUC| > 0.30 → high regardless of z), but
  for rare per-category leaks the absolute delta is also bounded above
  by ~``leak_p × p_lift`` ~= 0.05-0.10, below the strong-effect band.

  The label-shuffle adversarial score IS sensitive to the per-category
  leak: shuffling labels destroys the target-conditional structure that
  makes the leak indicator predictive, so ``null_mean ≪ actual_auc`` and
  z is large. This is the Layer-3 PERMUTATION pass (``compute_adversarial_score``)
  mirrored from Phase 3.3 onto encoded features.

  Because the leak lives IN the encoded space (the leaky thing is a
  specific one-hot indicator, not the raw column), measuring on encoded
  feature names is the CORRECT representation here, not a bug. The codex
  warning that the evaluator works on encoded numpy arrays applies to
  raw-column leaks the data-prep audit already catches; for the per-
  category class, encoded names ARE the right level of granularity.

  See the integration test at
  ``tests/integration/test_model_trainer_layer3_ablation.py`` for a
  deterministic synthetic construction and the dual-mode pin (Phase 3.3
  MISSES, model-eval Layer-3 CATCHES via the permutation sub-pass).

ARCHITECTURE (mirrors Phase 3.3's perm + ablation MAX-rule):

  * Per-encoded-feature label-shuffle ``compute_adversarial_score``
    produces a permutation severity via the legacy band
    (HIGH_Z=5σ → high, MODERATE_Z=3σ → moderate, else info).
  * Per-encoded-feature column-shuffle ``compute_feature_ablation``
    produces an ablation severity via the byte-identical two-tier rule
    from Phase 3.3 (strong-effect escape + joint-check ladder).
  * MAX-rule combines the two per-feature severities (any-source-
    suspicious wins), matching Phase 3.3's
    ``_combine_ablation_with_permutation`` semantics.

NON-NEGOTIABLE CONSTRAINTS (mirrors Phase 3.3):

  * Flag default OFF (``model_trainer_layer3_ablation_enabled=False``):
    the joint-model retrain pass is O(n_features) full retrains × O(n_perms)
    shuffle retrains.
  * Advisory mode: emits signals on validation_metrics, does NOT mutate
    ``success_criteria_met`` (matches the §4 T2.2 / T2.3 lifecycle-state
    pattern at evaluator.py:56-57).
  * Severity classification BYTE-IDENTICAL to Phase 3.3.
  * Schema-uniform output: when the pass is SKIPPED (flag OFF or guard
    fired), the result dict is None / ``ran=False`` so downstream
    consumers can distinguish "did not run" from "ran and produced low
    severity".

Runtime: roughly 10-30 s on a 50-encoded-feature pipeline at the
conservative ``n_permutations=30`` ablation + 200 single-feature
permutation defaults, similar to Phase 3.3.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from src.data.adversarial_leakage import compute_adversarial_score, compute_feature_ablation

logger = logging.getLogger(__name__)


# Mirror Phase 3.3 constants from adaptive_validity_check.py so the two
# hooks reason on a unified severity scale. The MAX-rule combination here
# (perm severity vs ablation severity, any-source-suspicious wins) is
# byte-identical to Phase 3.3's ``_combine_ablation_with_permutation``
# semantics; the severity classifiers themselves are byte-identical to
# ``_classify_ablation_severity`` and the permutation-band classifier
# (re-implemented locally to avoid pulling the full adversarial-input
# adapter from Phase 3.3 — that adapter is coupled to KG/LLM Layer-4
# inputs we do not have at model-eval time).
MODEL_EVAL_ABLATION_HIGH_Z: float = 5.0
MODEL_EVAL_ABLATION_MODERATE_Z: float = 3.0
MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT: float = 0.10
MODEL_EVAL_ABLATION_STRONG_EFFECT_DEFAULT: float = 0.30
DEFAULT_MODEL_EVAL_ABLATION_PERMUTATIONS: int = 30
DEFAULT_MODEL_EVAL_PERMUTATION_PERMS: int = 200
DEFAULT_MODEL_EVAL_ABLATION_MAX_FEATURES: int = 100
MIN_MODEL_EVAL_ABLATION_SAMPLES: int = 50

# MAX-rule severity rank (matches Phase 3.3 _SEVERITY_RANK semantics):
# higher rank wins when combining perm severity and ablation severity.
_SEVERITY_RANK: dict[str, int] = {"info": 0, "moderate": 1, "high": 2}


def _classify_permutation_severity(
    z_score: Any,
    *,
    delta_auc: Any = None,
    delta_auc_floor: float = MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    z_threshold: float = MODEL_EVAL_ABLATION_HIGH_Z,
    moderate_z_threshold: float = MODEL_EVAL_ABLATION_MODERATE_Z,
) -> str:
    """Map a label-shuffle (z, |delta_AUC|) pair to a severity band.

    Mirrors Phase 3.3's ``hblp_classify`` joint (z, |delta_AUC|) check
    from issue #194 (at ``adaptive_validity_check.py:454``), without the
    HBLP variance-inflation factor (no Layer-1 declared-safe input at
    model-eval time; the simple base 5σ band is correct here).

    Issue #194 joint check semantics (codex pass-1 MED-2):
      * delta_auc may be None / NaN / non-finite — the joint check is
        skipped in those cases and the classifier falls through to the
        pre-issue-#194 z-only ladder. This matches Phase 3.3's
        ``delta_auc_known`` predicate at adaptive_validity_check.py:529.
      * When delta_auc is known AND |delta_auc| <= floor, severity is
        FORCED to "info" even if z exceeds the band. This prevents the
        n≥10k FPR blowup on legitimate weak predictors whose null
        variance has shrunk per CLT below the absolute-effect floor of
        pharma-actionable leakage.
      * z=+inf with strong-effect (|delta_auc| > floor) → severity=high
        (mirrors Phase 3.3 ``z_is_positive_inf_strong_effect`` escape at
        adaptive_validity_check.py:545-554).

    Ladder (post-joint-check):
      * z > z_threshold (default 5.0) → high
      * moderate_z_threshold < z <= z_threshold → moderate
      * else (incl. NaN, -inf, etc.) → info
    """
    if z_score is None:
        return "info"
    if isinstance(z_score, bool):
        # bool is a subclass of int — handle separately to avoid treating
        # True/False as 1/0 z-scores.
        return "info"
    if not isinstance(z_score, (int, float)):
        return "info"
    z_f = float(z_score)
    if np.isnan(z_f):
        return "info"

    # Issue #194 joint check. delta_auc is OPTIONAL: when omitted /
    # None / non-finite, the joint check is inactive and we fall through
    # to z-only behaviour (preserves backward compatibility with callers
    # that don't have delta_auc available, matching Phase 3.3's
    # ``delta_auc_known`` predicate).
    delta_auc_known = (
        delta_auc is not None
        and isinstance(delta_auc, (int, float))
        and not isinstance(delta_auc, bool)
        and bool(np.isfinite(float(delta_auc)))
    )
    delta_auc_below_floor = delta_auc_known and abs(float(delta_auc)) <= float(delta_auc_floor)

    # +inf-with-strong-effect escape (mirrors Phase 3.3
    # z_is_positive_inf_strong_effect at adaptive_validity_check.py:545).
    if np.isinf(z_f) and z_f > 0:
        if delta_auc_known and abs(float(delta_auc)) > float(delta_auc_floor):
            return "high"
        # +inf without strong effect = info (legitimate degenerate-null
        # weak signal).
        return "info"

    # Issue #194 forced clamp: |delta_AUC| <= floor → info regardless
    # of z.
    if delta_auc_below_floor:
        return "info"

    # Post-joint-check z-only ladder.
    if z_f > float(z_threshold):
        return "high"
    if z_f > float(moderate_z_threshold):
        return "moderate"
    return "info"


def _max_rule_severity(perm_sev: str, ablation_sev: str) -> str:
    """MAX-rule combination: pick the strictly-stronger severity.

    Mirrors Phase 3.3's ``_combine_ablation_with_permutation`` semantics
    (any-source-suspicious wins). The rationale is identical: the two
    sub-tests measure DIFFERENT leak mechanisms (single-feature signal
    via label-shuffle vs marginal contribution to the joint model via
    column-shuffle), so an interaction-only leak that one sub-test misses
    can still be caught by the other.
    """
    p_rank = _SEVERITY_RANK.get(perm_sev, 0)
    a_rank = _SEVERITY_RANK.get(ablation_sev, 0)
    if a_rank > p_rank:
        return ablation_sev
    return perm_sev


def classify_model_eval_ablation_severity(
    ablation_row: dict[str, Any],
    *,
    z_threshold: float = MODEL_EVAL_ABLATION_HIGH_Z,
    moderate_z_threshold: float = MODEL_EVAL_ABLATION_MODERATE_Z,
    delta_auc_floor: float = MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    strong_effect_threshold: float = MODEL_EVAL_ABLATION_STRONG_EFFECT_DEFAULT,
) -> str:
    """Two-tier ablation severity classification — byte-identical to Phase 3.3.

    Mirrors ``_classify_ablation_severity`` at
    ``src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py:2129``.
    The unification matters because audit consumers reason about both
    severities on the same scale; if the model-eval rule drifted from
    the data-prep rule the audit would be inconsistent.

      0. Degradation: NaN delta_auc OR NaN z_score → "info".
      A. Strong-effect escape: delta_auc > 0.30 (positive only, signed) → "high".
      B. Joint-check ladder: delta_auc > 0.10 AND z passes band
           * z > z_threshold → "high"
           * moderate_z_threshold < z <= z_threshold → "moderate"
           * z == +inf (degenerate null) → "high"
      C. Default → "info".

    Returns one of "high", "moderate", "info".
    """
    z = ablation_row.get("z_score")
    delta_auc = ablation_row.get("delta_auc")

    if delta_auc is None:
        return "info"
    if not isinstance(delta_auc, (int, float)) or isinstance(delta_auc, bool):
        return "info"
    delta_f = float(delta_auc)
    if np.isnan(delta_f):
        return "info"

    z_is_nan = (
        z is None
        or not isinstance(z, (int, float))
        or isinstance(z, bool)
        or (isinstance(z, float) and np.isnan(z))
    )
    if z_is_nan:
        return "info"

    # Strong-effect escape (Case A) — signed; negative delta = nuisance.
    if delta_f > float(strong_effect_threshold):
        return "high"

    # Joint-check ladder (Case B) — signed, AND-rule.
    # mypy: z is narrowed by the z_is_nan check above (None → returned
    # early; non-numeric → returned early; bool → returned early; NaN →
    # returned early). The remaining type is int|float (sans bool, sans
    # NaN), but mypy cannot follow the negated-isinstance branch chain.
    z_f = float(z)  # type: ignore[arg-type]
    if delta_f > float(delta_auc_floor):
        if np.isinf(z_f) and z_f > 0:
            return "high"
        if z_f > float(z_threshold):
            return "high"
        if z_f > float(moderate_z_threshold):
            return "moderate"

    return "info"


def _build_dataframe_with_names(
    X: Any,
    feature_names: Optional[Sequence[str]],
) -> Optional[Any]:
    """Materialize X as a DataFrame with feature_names columns.

    ``compute_feature_ablation`` requires a DataFrame because it indexes by
    column name to drop and to shuffle. The evaluator's X is normally a
    numpy array (preprocessor output) — we use the preprocessor's
    ``get_feature_names_out()`` to attach names. When names are unavailable
    or shape mismatched, return None so the caller can skip cleanly.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("model_eval_ablation: pandas unavailable; skipping pass")
        return None

    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        # Already named. Guard against duplicate column names: pandas'
        # ``X.drop(columns=[name])`` drops ALL columns with that name,
        # which silently breaks ``compute_feature_ablation``'s per-
        # feature drop loop (the ablation null becomes a multi-column
        # drop instead of a single-feature drop). One-hot encoding can
        # produce duplicates if a categorical column shared a name with
        # an existing column pre-encoding (e.g., a column literally
        # named "region_region_0"). Skip cleanly with a warning so the
        # evaluator records a skip-reason rather than producing wrong
        # ablation numbers.
        duplicates = X.columns[X.columns.duplicated()].unique().tolist()
        if duplicates:
            logger.warning(
                "model_eval_ablation: X has duplicate column names %s; "
                "skipping (ablation drop semantics would be wrong)",
                duplicates[:5],
            )
            return None
        return X.copy()

    if not isinstance(X, np.ndarray):
        # Some unwrapped type (e.g., lightgbm.Dataset); skip safely.
        logger.warning(
            "model_eval_ablation: X is type %s, not DataFrame/ndarray; skipping",
            type(X).__name__,
        )
        return None
    if X.ndim != 2:
        logger.warning(
            "model_eval_ablation: X has ndim=%d, expected 2; skipping",
            X.ndim,
        )
        return None
    if feature_names is None or len(feature_names) != X.shape[1]:
        # Without names we cannot run the per-feature drop; the only
        # alternative is integer column names which would defeat audit
        # readability. Skip cleanly with a warning.
        logger.warning(
            "model_eval_ablation: feature_names %s does not match X.shape[1]=%d; skipping",
            "None" if feature_names is None else f"len={len(feature_names)}",
            X.shape[1],
        )
        return None
    # Guard against duplicate names in the provided feature_names list
    # (same rationale as the DataFrame-with-duplicate-columns branch
    # above). Should not happen in practice given OneHotEncoder's per-
    # category naming, but the guard is symmetric.
    names_list = list(feature_names)
    if len(set(names_list)) != len(names_list):
        from collections import Counter

        dups = [n for n, c in Counter(names_list).items() if c > 1]
        logger.warning(
            "model_eval_ablation: duplicate feature_names %s; skipping",
            dups[:5],
        )
        return None
    return pd.DataFrame(X, columns=names_list)


def _skipped_result(
    reason: str,
    *,
    z_threshold: float,
    delta_auc_floor: float,
    strong_effect_threshold: float,
    permutation_n: int,
) -> dict[str, Any]:
    """Build a schema-uniform "ran=False" result.

    Centralising the no-run payload here keeps all skip branches
    structurally identical so downstream consumers can iterate the
    same dict keys regardless of why the pass didn't run.
    """
    return {
        "ran": False,
        "skipped_reason": reason,
        "per_feature": [],
        "flagged_features": [],
        "n_permutations": 0,
        "permutation_n_permutations": permutation_n,
        "z_threshold": z_threshold,
        "delta_auc_floor": delta_auc_floor,
        "strong_effect_threshold": strong_effect_threshold,
    }


def _compute_perm_delta_auc(score: dict[str, Any]) -> Optional[float]:
    """Compute single-feature permutation delta_AUC from a score dict.

    Mirrors Phase 3.3 at ``adaptive_validity_check.py:1016-1024``:
    ``delta_auc = actual_auc - null_mean`` when both are finite, else
    None (signals "delta unknown" so the joint check skips). Folded-AUC
    scale, same as ``compute_adversarial_score``'s output.
    """
    auc = score.get("actual_auc")
    null_mean = score.get("null_mean")
    if (
        isinstance(auc, (int, float))
        and not isinstance(auc, bool)
        and isinstance(null_mean, (int, float))
        and not isinstance(null_mean, bool)
        and np.isfinite(float(auc))
        and np.isfinite(float(null_mean))
    ):
        return float(auc) - float(null_mean)
    return None


def _run_permutation_pass(
    df: Any,
    y: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    z_threshold: float,
    moderate_z_threshold: float,
    delta_auc_floor: float,
) -> dict[str, dict[str, Any]]:
    """Run per-encoded-feature label-shuffle ``compute_adversarial_score``.

    Returns a dict mapping encoded feature name → score dict with the
    augmentation keys ``permutation_severity`` (produced by
    ``_classify_permutation_severity`` with issue #194 joint check) and
    ``permutation_delta_auc`` (single-feature label-shuffle lift used to
    apply the joint check). Failures on individual features (degenerate
    columns, all-NaN after mask, etc.) are tolerated — the row stores
    NaN z and severity=info.

    The CONSTANT-COLUMN check skips features where the encoded column
    has zero variance (e.g., a OneHotEncoder indicator for a category
    that does not appear in the training split — they DO occur when the
    encoder was fit on a superset of categories). roc_auc_score raises
    on a constant feature; we degrade gracefully with NaN.
    """
    out: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        col_vals = df[col].to_numpy(dtype=float)
        if np.all(col_vals == col_vals[0]):
            # Constant column — z is undefined; severity = info.
            out[col] = {
                "actual_auc": float("nan"),
                "z_score": float("nan"),
                "null_mean": float("nan"),
                "null_std": float("nan"),
                "p_value": float("nan"),
                "suspicious": False,
                "permutation_severity": "info",
                "permutation_delta_auc": None,
            }
            continue
        try:
            score = compute_adversarial_score(
                col_vals,
                y,
                n_permutations=n_permutations,
                seed=seed,
                z_threshold=z_threshold,
            )
        except Exception as exc:
            logger.warning(
                "model_eval permutation pass failed for encoded feature %s: %s — "
                "degrading to info severity",
                col,
                exc,
            )
            score = {
                "actual_auc": float("nan"),
                "z_score": float("nan"),
                "null_mean": float("nan"),
                "null_std": float("nan"),
                "p_value": float("nan"),
                "suspicious": False,
            }
        # Issue #194 joint check — compute single-feature label-shuffle
        # delta_AUC and pipe through classifier with the floor.
        perm_delta = _compute_perm_delta_auc(score)
        sev = _classify_permutation_severity(
            score.get("z_score"),
            delta_auc=perm_delta,
            delta_auc_floor=delta_auc_floor,
            z_threshold=z_threshold,
            moderate_z_threshold=moderate_z_threshold,
        )
        score["permutation_severity"] = sev
        score["permutation_delta_auc"] = perm_delta
        out[col] = score
    return out


def run_model_eval_ablation(
    X_test: Any,
    y_test: Any,
    *,
    feature_names: Optional[Sequence[str]],
    n_permutations: int = DEFAULT_MODEL_EVAL_ABLATION_PERMUTATIONS,
    permutation_n_permutations: int = DEFAULT_MODEL_EVAL_PERMUTATION_PERMS,
    seed: int = 42,
    z_threshold: float = MODEL_EVAL_ABLATION_HIGH_Z,
    max_features: int = DEFAULT_MODEL_EVAL_ABLATION_MAX_FEATURES,
    strong_effect_threshold: float = MODEL_EVAL_ABLATION_STRONG_EFFECT_DEFAULT,
    delta_auc_floor: float = MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    moderate_z_threshold: float = MODEL_EVAL_ABLATION_MODERATE_Z,
    model_factory: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
    """Run Layer-3 (perm + ablation MAX-rule) on the model_trainer's encoded matrix.

    Args:
        X_test: Encoded feature matrix (numpy array or DataFrame).
            Convention is to pass the TRAIN split (see evaluator.py
            wiring rationale): ``compute_feature_ablation`` internally
            retrains so it needs sufficient data; on small test splits
            the joint AUC null becomes degenerate.
        y_test: 1D binary target array (matched to X_test rows).
        feature_names: Encoded feature names from
            ``preprocessor.get_feature_names_out()``. Required for the
            per-encoded-column passes to produce auditable names.
        n_permutations: Ablation null permutation rounds (default 30,
            mirrors Phase 3.3 ``DEFAULT_ABLATION_PERMUTATIONS``).
        permutation_n_permutations: Label-shuffle null permutation rounds
            (default 200, mirrors evaluator's ``DEFAULT_PERMUTATION_COUNT``).
        seed: RNG seed (pinned for reproducibility).
        z_threshold: HIGH-band z (default 5.0, mirrors Phase 3.3).
        max_features: O(n²) blowup guard — when encoded width exceeds
            this, the pass is SKIPPED with a warning (subsetting features
            would bias the joint-model AUC the survivors are measured
            against, same rationale as Phase 3.3).
        strong_effect_threshold: |delta_AUC| → high escape (default 0.30).
        delta_auc_floor: |delta_AUC| floor for joint-check ladder (default 0.10).
        moderate_z_threshold: MODERATE-band z (default 3.0).
        model_factory: Callable returning a fresh sklearn-compatible
            classifier with predict_proba. None falls through to
            ``compute_feature_ablation``'s default LogisticRegression. Pass
            a tree-based factory to detect interaction-only leaks the
            linear baseline cannot learn.

    Returns:
        Dict with:
          * ``ran``: bool — True if the pass executed, False if skipped.
          * ``skipped_reason``: str | None — populated when ran=False.
          * ``per_feature``: list of per-encoded-column rows with
            ``feature``, ``delta_auc``, ``ablation_z_score``,
            ``ablation_null_mean``, ``ablation_null_std``,
            ``ablation_severity``, ``actual_auc``, ``permutation_z_score``,
            ``permutation_null_mean``, ``permutation_null_std``,
            ``permutation_severity``, ``severity`` (MAX-rule combination),
            ``decided_by`` (``"adversarial_ablation"``, ``"adversarial_permutation"``,
            or ``None`` for info).
          * ``flagged_features``: list of encoded names with combined
            severity in {moderate, high}.
          * ``n_permutations``: ablation null permutations used.
          * ``permutation_n_permutations``: label-shuffle null permutations used.
          * ``z_threshold``, ``delta_auc_floor``,
            ``strong_effect_threshold``: returned for audit.

        Returns None when no work could be done (e.g., X_test=None,
        feature_names=None). Schema-uniform "ran=False, per_feature=[]"
        result is distinct from None — None means "skipped before even
        considering" (e.g., flag OFF), False means "considered but a
        guard fired".
    """
    if X_test is None or y_test is None:
        return None

    df = _build_dataframe_with_names(X_test, feature_names)
    if df is None:
        return _skipped_result(
            "could not materialize DataFrame from X with feature_names",
            z_threshold=z_threshold,
            delta_auc_floor=delta_auc_floor,
            strong_effect_threshold=strong_effect_threshold,
            permutation_n=permutation_n_permutations,
        )

    y_arr = np.asarray(y_test, dtype=int)
    if df.shape[0] < MIN_MODEL_EVAL_ABLATION_SAMPLES:
        return _skipped_result(
            f"too few samples: n_rows={df.shape[0]} < "
            f"MIN_MODEL_EVAL_ABLATION_SAMPLES={MIN_MODEL_EVAL_ABLATION_SAMPLES}",
            z_threshold=z_threshold,
            delta_auc_floor=delta_auc_floor,
            strong_effect_threshold=strong_effect_threshold,
            permutation_n=permutation_n_permutations,
        )
    if len(np.unique(y_arr)) < 2:
        return _skipped_result(
            "target has < 2 classes on the supplied split",
            z_threshold=z_threshold,
            delta_auc_floor=delta_auc_floor,
            strong_effect_threshold=strong_effect_threshold,
            permutation_n=permutation_n_permutations,
        )

    n_encoded = df.shape[1]
    if n_encoded > max_features:
        # Subsetting which features to ablate would bias the joint-model
        # AUC the survivors are measured against (same rationale as
        # Phase 3.3 adaptive_validity_check.py:2737-2743).
        logger.warning(
            "model_eval_ablation: encoded width %d > max_features=%d; "
            "skipping (O(n²) blowup guard fired)",
            n_encoded,
            max_features,
        )
        return _skipped_result(
            f"encoded width {n_encoded} > max_features={max_features}; O(n²) blowup guard fired",
            z_threshold=z_threshold,
            delta_auc_floor=delta_auc_floor,
            strong_effect_threshold=strong_effect_threshold,
            permutation_n=permutation_n_permutations,
        )

    # Drop rows with any NaN to give the joint model a consistent design
    # matrix. ``compute_feature_ablation`` cannot tolerate NaN in the
    # LogisticRegression default factory; tree factories vary. Matches
    # the masking pattern at adaptive_validity_check.py:2412-2417.
    notna_mask = df.notna().all(axis=1)
    if int(notna_mask.sum()) < MIN_MODEL_EVAL_ABLATION_SAMPLES:
        return _skipped_result(
            f"only {int(notna_mask.sum())} non-NaN rows survived; "
            f"need ≥ {MIN_MODEL_EVAL_ABLATION_SAMPLES}",
            z_threshold=z_threshold,
            delta_auc_floor=delta_auc_floor,
            strong_effect_threshold=strong_effect_threshold,
            permutation_n=permutation_n_permutations,
        )
    df_clean = df.loc[notna_mask].copy()
    y_clean = y_arr[notna_mask.to_numpy()]

    # === Pass 1: Per-encoded-feature LABEL-SHUFFLE permutation ===
    # This is the load-bearing pass for the per-category leak class.
    # Catches: per-OHE-indicator leaks whose single-feature label-shuffle
    # z-score is well above the band (the column-shuffle ablation null
    # collapses on these). Mirrors Phase 3.3's permutation pass at
    # adaptive_validity_check.py:2786-2792 but on encoded features.
    perm_results = _run_permutation_pass(
        df_clean,
        y_clean,
        n_permutations=permutation_n_permutations,
        seed=seed,
        z_threshold=z_threshold,
        moderate_z_threshold=moderate_z_threshold,
        delta_auc_floor=delta_auc_floor,
    )

    # === Pass 2: Joint-model COLUMN-SHUFFLE ablation ===
    # This is the load-bearing pass for the joint-model strong-effect
    # leak class (|delta_AUC| > strong_effect_threshold). Mirrors Phase
    # 3.3's ablation pass at adaptive_validity_check.py:2745-2753.
    try:
        ablation_result = compute_feature_ablation(
            df_clean,
            y_clean,
            model_factory=model_factory,
            n_permutations=n_permutations,
            seed=seed,
            z_threshold=z_threshold,
        )
    except Exception as exc:
        logger.warning(
            "model_eval_ablation: ablation full-model fit failed (%s); "
            "degrading to permutation-only Layer 3",
            exc,
        )
        ablation_result = {"per_feature": [], "full_auc": float("nan")}
    ablation_per_feat = {
        row.get("feature"): row
        for row in (ablation_result.get("per_feature", []) or [])
        if isinstance(row, dict) and row.get("feature") is not None
    }

    # === MAX-rule combination ===
    per_feature: list[dict[str, Any]] = []
    flagged: list[str] = []
    for col in df_clean.columns:
        perm_row = perm_results.get(col, {})
        abl_row = ablation_per_feat.get(col, {})
        perm_sev = perm_row.get("permutation_severity", "info")
        abl_sev = classify_model_eval_ablation_severity(
            abl_row,
            z_threshold=z_threshold,
            moderate_z_threshold=moderate_z_threshold,
            delta_auc_floor=delta_auc_floor,
            strong_effect_threshold=strong_effect_threshold,
        )
        combined_sev = _max_rule_severity(perm_sev, abl_sev)
        # ``decided_by`` records which sub-test produced the combined
        # severity. Uses Phase 3.3's audit convention BYTE-IDENTICALLY:
        #   * "adversarial" for perm-only escalation (Phase 3.3 default
        #     tag at adaptive_validity_check.py:1315, :1371, :1422).
        #   * "adversarial_ablation" only when ablation strictly
        #     escalates above perm (Phase 3.3 overwrite at
        #     adaptive_validity_check.py:2930-2931).
        # Both keys map to layer "3" via _DECIDED_BY_TO_LAYER at
        # adaptive_validity_check.py:1153-1160 — audit consumers see the
        # same layer attribution across both pipeline stages.
        # Tie-break: ties go to perm (matches Phase 3.3's
        # _combine_ablation_with_permutation at :2320 where
        # ablation_rank <= perm_rank keeps perm_input).
        if combined_sev == "info":
            decided_by = None
        elif _SEVERITY_RANK.get(abl_sev, 0) > _SEVERITY_RANK.get(perm_sev, 0):
            decided_by = "adversarial_ablation"
        else:
            decided_by = "adversarial"
        out_row = {
            "feature": str(col),
            # Ablation-side audit fields. Field NAMES match Phase 3.3's
            # ``_combine_ablation_with_permutation`` at
            # ``adaptive_validity_check.py:2312-2316`` so audit consumers
            # see byte-identical keys across both pipeline stages.
            "ablation_delta_auc": abl_row.get("delta_auc"),
            "ablation_z_score": abl_row.get("z_score"),
            "ablation_null_mean": abl_row.get("null_mean"),
            "ablation_null_std": abl_row.get("null_std"),
            "ablation_severity": abl_sev,
            # Permutation-side audit fields.
            "actual_auc": perm_row.get("actual_auc"),
            "permutation_z_score": perm_row.get("z_score"),
            "permutation_null_mean": perm_row.get("null_mean"),
            "permutation_null_std": perm_row.get("null_std"),
            "permutation_p_value": perm_row.get("p_value"),
            # Issue #194 joint check: permutation delta_AUC + floor used.
            "permutation_delta_auc": perm_row.get("permutation_delta_auc"),
            "permutation_severity": perm_sev,
            # Combined.
            "severity": combined_sev,
            "decided_by": decided_by,
        }
        per_feature.append(out_row)
        if combined_sev in ("moderate", "high"):
            flagged.append(str(col))

    logger.info(
        "model_eval_ablation: scored %d encoded features "
        "(perm_n=%d, ablation_n_perms=%d, z_threshold=%.2f, "
        "|delta_AUC| floor=%.4f); flagged=%d",
        len(per_feature),
        permutation_n_permutations,
        n_permutations,
        z_threshold,
        delta_auc_floor,
        len(flagged),
    )

    return {
        "ran": True,
        "skipped_reason": None,
        "per_feature": per_feature,
        "flagged_features": flagged,
        "n_permutations": n_permutations,
        "permutation_n_permutations": permutation_n_permutations,
        "z_threshold": z_threshold,
        "delta_auc_floor": delta_auc_floor,
        "strong_effect_threshold": strong_effect_threshold,
        "full_auc": ablation_result.get("full_auc"),
    }
