"""Synthetic-augmentation node — opt-in consumption of the Phase 3 preview cohort.

Phase 3 (PR #475) *produces* a synthetic preview cohort and writes it to disk
without ever mixing it into training. This node is the *consumption* half: when
an operator has reviewed a preview and opts in by setting
``PipelineConfig.augmentation_data_path`` (threaded into trainer state), this
node concatenates the synthetic rows into the **training split only** — so they
flow through the same preprocessing / resampling / training as the real data.

It runs after ``enforce_splits`` (so the real 60/20/10/10 ratio validation is
done on the real data) and before ``fit_preprocessing``.

Safety (pharma anti-mocking discipline — see CLAUDE.md):
- **Opt-in:** no-op unless ``augmentation_data_path`` is set.
- **Strict schema validation:** the synthetic feature columns must match the
  real training columns exactly. On ANY mismatch the node REFUSES to augment
  (records a skip reason + logs loudly) rather than silently mixing
  incompatible rows — silently-fake training data is forbidden.
- **Validation/test/holdout are NEVER touched** — only ``train_data``.
- **Advisory:** any failure leaves the real training data intact and never
  raises; training proceeds on real data alone.
- **Audited:** original vs synthetic row counts, the source path, and the
  synthetic cohort's audit fingerprint are surfaced for the audit chain.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _skip(reason: str) -> Dict[str, Any]:
    """Audit-only patch that leaves ``train_data`` untouched (no augmentation)."""
    logger.warning("Synthetic augmentation skipped (training proceeds on real data): %s", reason)
    return {"augmentation_applied": False, "augmentation_skip_reason": reason}


async def augment_training_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """Opt-in: concatenate a reviewed synthetic cohort into the TRAINING split.

    No-op unless ``augmentation_data_path`` is set. Returns a state patch with
    the augmented ``train_data`` (+ updated counts) and an audit trail, or an
    audit-only patch (``train_data`` untouched) when augmentation is
    skipped/refused. Never raises.
    """
    if state.get("error"):
        return {}

    path = state.get("augmentation_data_path")
    if not path:
        return {}  # opt-in: augmentation not requested

    train_data = state.get("train_data") or {}
    X_real = train_data.get("X")
    y_real = train_data.get("y")
    if X_real is None or y_real is None:
        return _skip("no real training data present to augment")

    try:
        npz_path = Path(path)
        if not npz_path.exists():
            return _skip(f"augmentation_data_path does not exist: {path}")

        # allow_pickle=False: the cohort is numeric arrays only; refuse to
        # deserialize arbitrary pickled objects from a path.
        with np.load(npz_path, allow_pickle=False) as arrays:
            if "X_train" not in arrays or "y_train" not in arrays:
                return _skip(f"cohort {npz_path.name} missing X_train/y_train arrays")
            X_syn = np.asarray(arrays["X_train"])
            y_syn = np.asarray(arrays["y_train"]).ravel()

        if X_syn.ndim != 2 or X_syn.shape[0] == 0:
            return _skip("synthetic X_train is empty or not 2-D")
        if X_syn.shape[0] != y_syn.shape[0]:
            return _skip(f"synthetic X/y row mismatch: {X_syn.shape[0]} vs {y_syn.shape[0]}")

        # Feature names live in the sibling Phase-3 metadata file (the .npz
        # itself stores only the bare value arrays).
        feature_names: Optional[List[str]] = None
        audit_fingerprint: Optional[str] = None
        meta_path = npz_path.parent / "preview_metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                feature_names = list(meta.get("feature_names") or []) or None
                audit_fingerprint = meta.get("audit_fingerprint")
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read %s (proceeding without it): %s", meta_path, exc)

        # --- STRICT schema validation against the REAL training columns. ---
        if isinstance(X_real, pd.DataFrame):
            real_cols = list(X_real.columns)
            if X_syn.shape[1] != len(real_cols):
                return _skip(
                    f"feature-count mismatch: real training has {len(real_cols)} columns, "
                    f"synthetic cohort has {X_syn.shape[1]}"
                )
            if feature_names is not None and feature_names != real_cols:
                return _skip(
                    "feature name/order mismatch between synthetic cohort and real training "
                    "columns — refusing to mix mismatched schemas"
                )
            if feature_names is None:
                logger.warning(
                    "No preview_metadata.json beside %s; synthetic columns aligned to the "
                    "real training columns by POSITION (names unverified).",
                    npz_path.name,
                )
            X_syn_aligned = pd.DataFrame(X_syn, columns=real_cols)
            X_aug: Any = pd.concat([X_real, X_syn_aligned], axis=0, ignore_index=True)
        else:
            X_real_arr = np.asarray(X_real)
            real_width = X_real_arr.shape[1] if X_real_arr.ndim == 2 else None
            if real_width is None or X_syn.shape[1] != real_width:
                return _skip(
                    f"feature-count mismatch: real training width {real_width}, "
                    f"synthetic width {X_syn.shape[1]}"
                )
            X_aug = np.vstack([X_real_arr, X_syn])

        # Concatenate targets, preserving the real ``y`` container type.
        if isinstance(y_real, pd.Series):
            y_aug: Any = pd.concat([y_real, pd.Series(y_syn, name=y_real.name)], ignore_index=True)
        else:
            y_aug = np.concatenate([np.asarray(y_real).ravel(), y_syn])

        n_original = int(np.asarray(y_real).shape[0])
        n_synthetic = int(y_syn.shape[0])
        new_total = n_original + n_synthetic

        augmented_train = dict(train_data)
        augmented_train["X"] = X_aug
        augmented_train["y"] = y_aug
        augmented_train["row_count"] = new_total

        logger.info(
            "Synthetic augmentation applied: +%d synthetic rows onto %d real "
            "(train_data now %d) from %s [validation/test/holdout untouched]",
            n_synthetic,
            n_original,
            new_total,
            npz_path.name,
        )
        return {
            "train_data": augmented_train,
            "train_samples": new_total,
            "augmentation_applied": True,
            "augmentation_n_original": n_original,
            "augmentation_n_synthetic": n_synthetic,
            "augmentation_source": str(npz_path),
            "augmentation_fingerprint": audit_fingerprint,
            "augmentation_skip_reason": None,
        }
    except Exception as exc:  # advisory — augmentation must never fail training
        logger.warning(
            "Synthetic augmentation failed (advisory; training proceeds on real data): %s",
            exc,
            exc_info=True,
        )
        return _skip(f"augmentation error: {exc}")
