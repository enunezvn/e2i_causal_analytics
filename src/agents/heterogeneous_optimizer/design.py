"""Shared design-matrix and treatment invariants for the HTE estimation nodes.

The CATE, hierarchical and uplift nodes each build their own design matrix from
``state["effect_modifiers"]`` and their own treatment vector from
``state["treatment_var"]``. Two invariants must hold identically in all three, so
they live here rather than in three private copies:

* **X never contains the question slots.** With the treatment inside X the
  median-split treatment is a deterministic function of an X column: the DML
  propensity model is perfect (AUC 1.0), the treatment residual is zero, and
  CausalForestDML divides by nothing. Live 2026-09-03 (``seg_05f29d1b3295``,
  Remibrutinib ``urticaria_severity_uas7 -> persistent_180d``): ATE -0.514 on a
  0/1 outcome, per-segment CATE -1.6..+0.4, against a planted +0.150; dropping
  the column from X alone recovered +0.140. Provenance columns are dropped for
  the same reason they always were (Shard 07 C2).

* **A continuous treatment is binarized by ONE rule.** The CATE and
  hierarchical nodes split at the median; the uplift node did not and handed
  CausalML the raw score (27 "treatment groups", control ``"16.0"``), so the
  cross-library validator compared two different estimands and reported 9%
  agreement. Every node now calls :func:`binarize_treatment`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

__all__ = ["binarize_treatment", "sanitize_effect_modifiers"]


def binarize_treatment(values: Any) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Median-split a continuous treatment into {0, 1}: 1 when STRICTLY above the median.

    This is the rule ``cate_estimator`` has always used (consistent with the
    causal_impact agent). A treatment with at most two distinct values is returned
    unchanged (as an ndarray) with ``info=None``; otherwise ``info`` carries the
    threshold and the group sizes for the caller's log line.
    """
    arr = np.asarray(values)
    uniques = np.unique(arr)
    if len(uniques) <= 2:
        return arr, None
    median_val = float(np.median(arr))
    binary = (arr > median_val).astype(int)
    treated = int(binary.sum())
    return binary, {
        "median_threshold": median_val,
        "original_unique_values": int(len(uniques)),
        "treated_count": treated,
        "control_count": int(len(binary) - treated),
    }


def sanitize_effect_modifiers(state: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """The effect modifiers a node may put in X, and the ones it must not.

    Drops the question slots (``treatment_var`` / ``outcome_var``) and the
    provenance columns, de-duplicates, preserves order. Returns ``(kept, dropped)``
    so the caller can log what was removed.
    """
    from src.repositories.provenance import PROVENANCE_DROP_COLS

    slots = {state.get("treatment_var"), state.get("outcome_var")}
    kept: List[str] = []
    dropped: List[str] = []
    for col in dict.fromkeys(state.get("effect_modifiers") or []):
        if col in slots or col in PROVENANCE_DROP_COLS:
            dropped.append(col)
        else:
            kept.append(col)
    return kept, dropped
