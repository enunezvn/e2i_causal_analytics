"""Adaptive validity check — Layer 5 pipeline integration node.

Runs the data-derived Layer 3 adversarial discriminator against every
feature in train_df and emits a structured ``LeakageVerdict`` per feature.
Augments (does not replace) the existing ``detect_leakage`` results, so
both the legacy hardcoded checks and the adaptive permutation-baseline
checks contribute to the leakage_remediation routing.

Decision policy (data-derived, no hardcoded AUC thresholds):

    z > 5σ above null  → severity=high,     remediation=drop      (auto-flag)
    3σ < z ≤ 5σ        → severity=moderate, remediation=ambiguous (Layer 4 review)
    z ≤ 3σ             → severity=info,     remediation=keep

Layer 4 (DSPy CausalRoleClassifier) is invoked for ``ambiguous`` verdicts
when an LM is configured; otherwise the verdict is recorded for manual
governance review. This implementation focuses on Layers 1+3 wiring; Layer
4 LM dispatch lands when the API key configuration story is finalized.

Acceptance criterion #4 of ``adaptive_temporal_validity_redesign.md``:
every feature decision produces a structured record with layer, evidence,
confidence, and remediation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.adversarial_leakage import compute_adversarial_score
from src.data.feature_contract import FeatureContract
from src.data.manifests import lookup_feature_contract

logger = logging.getLogger(__name__)


HIGH_Z = 5.0
MODERATE_Z = 3.0
DEFAULT_PERMUTATIONS = 200


def _layer_1_verdict(feature: str, contract: FeatureContract) -> dict[str, Any]:
    """Build a Layer 1 verdict from a forbidden-by-contract feature.

    A feature whose contract declares `knowable_at = post_index` cannot be
    used as a model input regardless of its statistical properties — the
    contract is sufficient evidence. This catches things like
    `journey_duration_days`, `journey_status`, and target columns that may
    have leaked into the feature surface, BEFORE the permutation-test pass.
    """
    return {
        "feature": feature,
        "layer": "1",
        "z_score": None,
        "actual_auc": None,
        "null_mean": None,
        "null_std": None,
        "p_value": None,
        "n_permutations": 0,
        "severity": "high",
        "remediation": "drop",
        "evidence": (
            f"Layer 1 declarative contract: feature.knowable_at="
            f"{contract.knowable_at} (post_index); the manifest declares this "
            f"column is not knowable at prediction time → drop"
        ),
        "contract_source": contract.source,
        "contract_window_days": contract.window_days,
    }


def _build_verdict(
    feature: str,
    score: dict[str, Any],
) -> dict[str, Any]:
    """Translate a Layer-3 score dict into the audit-trail verdict shape."""
    z = score.get("z_score", float("nan"))
    auc = score.get("actual_auc", float("nan"))
    null_mean = score.get("null_mean", float("nan"))

    if isinstance(z, float) and np.isnan(z):
        severity = "info"
        remediation = "keep"
        evidence = (
            f"Adversarial score undefined (degenerate; actual_auc={auc}, null_mean={null_mean})"
        )
    elif z > HIGH_Z:
        severity = "high"
        remediation = "drop"
        evidence = (
            f"Layer 3 adversarial discriminator: z={z:.2f}σ above null "
            f"(actual_auc={auc:.4f}, null_mean={null_mean:.4f}); "
            f"{HIGH_Z}σ governance threshold exceeded → drop"
        )
    elif z > MODERATE_Z:
        severity = "moderate"
        remediation = "ambiguous"
        evidence = (
            f"Layer 3 adversarial discriminator: z={z:.2f}σ "
            f"(between {MODERATE_Z}σ and {HIGH_Z}σ); ambiguous → "
            f"queued for Layer 4 causal-role classification"
        )
    else:
        severity = "info"
        remediation = "keep"
        evidence = (
            f"Layer 3 adversarial discriminator: z={z:.2f}σ "
            f"(below {MODERATE_Z}σ noise floor); legitimate weak signal"
        )

    return {
        "feature": feature,
        "layer": "3",
        "z_score": float(z) if not (isinstance(z, float) and np.isnan(z)) else None,
        "actual_auc": float(auc) if not (isinstance(auc, float) and np.isnan(auc)) else None,
        "null_mean": float(null_mean)
        if not (isinstance(null_mean, float) and np.isnan(null_mean))
        else None,
        "null_std": score.get("null_std"),
        "p_value": score.get("p_value"),
        "n_permutations": score.get("n_permutations"),
        "severity": severity,
        "remediation": remediation,
        "evidence": evidence,
        # Layer 3 verdicts have no manifest contract; emit None so the audit-
        # trail JSON sidecar has a uniform schema with Layer 1 verdicts.
        "contract_source": None,
        "contract_window_days": None,
    }


def _short_circuit_verdict(feature: str, *, evidence: str) -> dict[str, Any]:
    """Build a Layer 3 verdict for a feature that bypassed the scoring path.

    Used for the too-few-rows and scoring-error cases. Schema matches both
    ``_build_verdict`` and ``_layer_1_verdict`` so the audit-trail JSON sidecar
    is uniform across all verdict shapes.
    """
    return {
        "feature": feature,
        "layer": "3",
        "z_score": None,
        "actual_auc": None,
        "null_mean": None,
        "null_std": None,
        "p_value": None,
        "n_permutations": None,
        "severity": "info",
        "remediation": "keep",
        "evidence": evidence,
        "contract_source": None,
        "contract_window_days": None,
    }


def _select_features(df: pd.DataFrame, target: str, excluded: list[str]) -> list[str]:
    """Return the feature columns Layer 3 should evaluate.

    - Excludes the target itself.
    - Excludes columns the scope spec already declared excluded (PII, declared leakage).
    - Excludes non-numeric columns: Layer 3 needs a continuous score for AUC, and
      categorical handling routes through ``check_categorical_class_separation``
      in the legacy detector. Categorical adaptive scoring is a Layer 5 follow-up.
    """
    # Use pandas' is_numeric_dtype, not np.issubdtype: the latter raises
    # `TypeError: Cannot interpret 'Int64Dtype()' as a data type` on pandas
    # extension dtypes (Int64/Float64/boolean). Any DataFrame ingested from
    # Supabase/SQLAlchemy with nullable-int schema would crash the node.
    excluded_set = set(excluded or [])
    excluded_set.add(target)
    cols = []
    for c in df.columns:
        if c in excluded_set:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


async def adaptive_validity_check(state: dict[str, Any]) -> dict[str, Any]:
    """Run Layer 3 adversarial discriminator on every feature; emit verdicts.

    Args:
        state: Current DataPreparerState (dict-like).

    Returns:
        Dict with state updates:
        - ``adaptive_verdicts``: list of verdict dicts (one per evaluated feature).
        - ``adaptive_flagged_features``: features at ``severity=high`` (z > 5σ).
        - ``leaked_features``: union of pre-existing flagged set + new flags.
        - ``leakage_findings``: pre-existing list extended with adaptive verdicts.
    """
    train_df = state.get("train_df")
    scope_spec = state.get("scope_spec") or {}
    target = scope_spec.get("prediction_target")
    excluded = scope_spec.get("excluded_features", []) or []
    # Layer 1 (manifest-driven contracts) is opt-in per cohort. Scenario_a
    # and other synthetic regimes leave this unset; CSU/Optum runners set
    # ``feature_manifest_source`` in scope_spec so only the matching manifest
    # is consulted. Without this guard the manifest matches any column that
    # happens to share a name across cohorts (e.g., scenario_a's constant
    # ``brand="Kisqali"`` would hit the CSU manifest's post-index contract
    # and halt the pipeline).
    manifest_source = scope_spec.get("feature_manifest_source")

    # Graceful no-op cases
    if train_df is None or target is None or target not in getattr(train_df, "columns", []):
        logger.info("adaptive_validity_check: no target/train_df → skipping")
        return {
            "adaptive_verdicts": [],
            "adaptive_flagged_features": [],
        }

    # Layer 1 (manifest-driven) operates on ALL columns regardless of dtype —
    # the contract is metadata, not data. Layer 3 (statistical) requires a
    # numeric AUC, so non-numeric columns can only be caught by Layer 1.
    excluded_set = set(excluded or [])
    excluded_set.add(target)
    all_columns = [c for c in train_df.columns if c not in excluded_set]
    numeric_candidates = _select_features(train_df, target, excluded)

    if not all_columns:
        logger.info("adaptive_validity_check: no candidate columns → skipping")
        return {
            "adaptive_verdicts": [],
            "adaptive_flagged_features": [],
        }

    # Build a per-row target-validity mask. For a binary classification target
    # we accept ONLY {0, 1}; integer sentinels like -1 (unknown outcome) would
    # otherwise pass the `pd.isna` check (integers can't be NaN), reach
    # `roc_auc_score` as a 3-class input, raise ValueError, get caught, and
    # silently produce severity=info verdicts for every numeric feature —
    # turning Layer 3 into a complete blind spot.
    target_arr = train_df[target].to_numpy()
    target_notna = ~pd.isna(target_arr)
    binary_label_mask = pd.Series(
        np.isin(target_arr, [0, 1]) & target_notna,
        index=train_df.index,
    )
    n_invalid = int((~binary_label_mask).sum() - (~target_notna).sum())
    if n_invalid > 0:
        logger.warning(
            "adaptive_validity_check: target %r has %d rows with non-binary "
            "values (sentinels?); these rows are excluded from Layer 3 scoring",
            target,
            n_invalid,
        )
    valid_target_values = target_arr[binary_label_mask.to_numpy()]
    if len(np.unique(valid_target_values)) < 2:
        logger.info("adaptive_validity_check: target has < 2 classes → skipping")
        return {
            "adaptive_verdicts": [],
            "adaptive_flagged_features": [],
        }

    # Use explicit `is not None` checks: `state.get(...) or DEFAULT` silently
    # replaces a legitimate 0 with the default (Python's falsy-zero semantics).
    # `adaptive_seed=0` is a valid seed; the old form returned 7 instead.
    _n_perms = state.get("adaptive_n_permutations")
    n_perms = int(_n_perms) if _n_perms is not None else DEFAULT_PERMUTATIONS
    _seed = state.get("adaptive_seed")
    seed = int(_seed) if _seed is not None else 7

    verdicts: list[dict[str, Any]] = []
    flagged: list[str] = []

    # Layer 1 pass — every column, manifest-driven catch for post-index ones.
    # Skipped entirely when ``feature_manifest_source`` is unset (e.g.,
    # synthetic regimes); see scope_spec read at the top of this function.
    layer_1_caught: set[str] = set()
    for feat in all_columns:
        contract = lookup_feature_contract(feat, data_source=manifest_source)
        if contract is not None and not contract.knowable_at.is_pre_or_at_index():
            verdict = _layer_1_verdict(feat, contract)
            verdicts.append(verdict)
            flagged.append(feat)
            layer_1_caught.add(feat)

    # Layer 3 pass — numeric columns only, skipping anything Layer 1 already caught.
    for feat in numeric_candidates:
        if feat in layer_1_caught:
            continue

        col = train_df[feat]
        mask = col.notna() & binary_label_mask
        if mask.sum() < 30:
            verdicts.append(
                _short_circuit_verdict(
                    feat,
                    evidence=f"Skipped: only {int(mask.sum())} non-null rows (need ≥30)",
                )
            )
            continue

        try:
            score = compute_adversarial_score(
                col[mask].to_numpy(dtype=float),
                train_df.loc[mask, target].to_numpy(dtype=int),
                n_permutations=n_perms,
                seed=seed,
                z_threshold=HIGH_Z,
            )
        except Exception as exc:
            logger.warning("adaptive_validity_check: scoring failed for %s: %s", feat, exc)
            verdicts.append(
                _short_circuit_verdict(
                    feat,
                    evidence=f"Adversarial scoring error: {exc}",
                )
            )
            continue

        verdict = _build_verdict(feat, score)
        verdicts.append(verdict)
        if verdict["severity"] == "high":
            flagged.append(feat)

    # Merge with existing leakage state — augment, don't replace. The
    # graph re-enters this node after leakage_remediation drops columns,
    # so we extend the prior `adaptive_verdicts` and `adaptive_flagged_features`
    # rather than overwriting them; the audit trail spans every invocation.
    prior_leaked = list(state.get("leaked_features") or [])
    prior_findings = list(state.get("leakage_findings") or [])
    prior_severity = state.get("leakage_severity") or "none"
    prior_verdicts = list(state.get("adaptive_verdicts") or [])
    prior_flagged = list(state.get("adaptive_flagged_features") or [])

    merged_leaked = sorted(set(prior_leaked) | set(flagged))
    merged_findings = prior_findings + verdicts

    # Dedup verdicts by feature name — first verdict wins (the one from the
    # initial invocation, before columns were dropped, has the most evidence).
    seen_features = {v["feature"] for v in prior_verdicts}
    extended_verdicts = list(prior_verdicts)
    for v in verdicts:
        if v["feature"] not in seen_features:
            extended_verdicts.append(v)
            seen_features.add(v["feature"])
    extended_flagged = sorted(set(prior_flagged) | set(flagged))

    # Escalate severity if Layer 3 caught something legacy missed. Severity
    # ordering: critical > high > moderate > info > none. Adaptive only escalates
    # — never downgrades — so the legacy detector's verdict is preserved.
    severity_rank = {"critical": 4, "high": 3, "moderate": 2, "info": 1, "none": 0}
    new_severity = prior_severity
    if flagged and severity_rank.get(prior_severity, 0) < severity_rank["high"]:
        new_severity = "high"

    logger.info(
        "adaptive_validity_check: scored=%d flagged=%d (high) prior_severity=%s new_severity=%s",
        len(verdicts),
        len(flagged),
        prior_severity,
        new_severity,
    )

    update: dict[str, Any] = {
        "adaptive_verdicts": extended_verdicts,
        "adaptive_flagged_features": extended_flagged,
        "leaked_features": merged_leaked,
        "leakage_findings": merged_findings,
    }
    if new_severity != prior_severity:
        update["leakage_severity"] = new_severity
        update["leakage_detected"] = True
    return update
