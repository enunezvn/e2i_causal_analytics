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

logger = logging.getLogger(__name__)


HIGH_Z = 5.0
MODERATE_Z = 3.0
DEFAULT_PERMUTATIONS = 200


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
            f"Adversarial score undefined (degenerate; "
            f"actual_auc={auc}, null_mean={null_mean})"
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
    }


def _select_features(df: pd.DataFrame, target: str, excluded: list[str]) -> list[str]:
    """Return the feature columns Layer 3 should evaluate.

    - Excludes the target itself.
    - Excludes columns the scope spec already declared excluded (PII, declared leakage).
    - Excludes non-numeric columns: Layer 3 needs a continuous score for AUC, and
      categorical handling routes through ``check_categorical_class_separation``
      in the legacy detector. Categorical adaptive scoring is a Layer 5 follow-up.
    """
    excluded_set = set(excluded or [])
    excluded_set.add(target)
    cols = []
    for c in df.columns:
        if c in excluded_set:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
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

    # Graceful no-op cases
    if train_df is None or target is None or target not in getattr(train_df, "columns", []):
        logger.info("adaptive_validity_check: no target/train_df → skipping")
        return {
            "adaptive_verdicts": [],
            "adaptive_flagged_features": [],
        }

    candidates = _select_features(train_df, target, excluded)
    if not candidates:
        logger.info("adaptive_validity_check: no numeric features to score → skipping")
        return {
            "adaptive_verdicts": [],
            "adaptive_flagged_features": [],
        }

    target_arr = train_df[target].to_numpy()
    if len(np.unique(target_arr[~pd.isna(target_arr)])) < 2:
        logger.info("adaptive_validity_check: target has < 2 classes → skipping")
        return {
            "adaptive_verdicts": [],
            "adaptive_flagged_features": [],
        }

    n_perms = int(state.get("adaptive_n_permutations") or DEFAULT_PERMUTATIONS)
    seed = int(state.get("adaptive_seed") or 7)

    verdicts: list[dict[str, Any]] = []
    flagged: list[str] = []

    for feat in candidates:
        col = train_df[feat]
        mask = col.notna() & pd.notna(train_df[target])
        if mask.sum() < 30:
            verdicts.append(
                {
                    "feature": feat,
                    "layer": "3",
                    "severity": "info",
                    "remediation": "keep",
                    "evidence": f"Skipped: only {int(mask.sum())} non-null rows (need ≥30)",
                    "z_score": None,
                    "actual_auc": None,
                }
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
                {
                    "feature": feat,
                    "layer": "3",
                    "severity": "info",
                    "remediation": "keep",
                    "evidence": f"Adversarial scoring error: {exc}",
                    "z_score": None,
                    "actual_auc": None,
                }
            )
            continue

        verdict = _build_verdict(feat, score)
        verdicts.append(verdict)
        if verdict["severity"] == "high":
            flagged.append(feat)

    # Merge with existing leakage state — augment, don't replace
    prior_leaked = list(state.get("leaked_features") or [])
    prior_findings = list(state.get("leakage_findings") or [])
    prior_severity = state.get("leakage_severity") or "none"

    merged_leaked = sorted(set(prior_leaked) | set(flagged))
    merged_findings = prior_findings + verdicts

    # Escalate severity if Layer 3 caught something legacy missed. Severity
    # ordering: critical > high > moderate > info > none. Adaptive only escalates
    # — never downgrades — so the legacy detector's verdict is preserved.
    severity_rank = {"critical": 4, "high": 3, "moderate": 2, "info": 1, "none": 0}
    new_severity = prior_severity
    if flagged and severity_rank.get(prior_severity, 0) < severity_rank["high"]:
        new_severity = "high"

    logger.info(
        "adaptive_validity_check: scored=%d flagged=%d (high) "
        "prior_severity=%s new_severity=%s",
        len(verdicts),
        len(flagged),
        prior_severity,
        new_severity,
    )

    update: dict[str, Any] = {
        "adaptive_verdicts": verdicts,
        "adaptive_flagged_features": flagged,
        "leaked_features": merged_leaked,
        "leakage_findings": merged_findings,
    }
    if new_severity != prior_severity:
        update["leakage_severity"] = new_severity
        update["leakage_detected"] = True
    return update
