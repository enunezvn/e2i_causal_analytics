"""Cross-fold metric aggregator for repeated_k10 evaluation mode.

Phase 1 W3-lite Day-5 implementation per shard 21 §D. Consumed by
``_run_repeated_splits`` (shard 21 §B). For each scalar metric across k folds
emits ``mean ± std`` + 2.5/97.5 percentile CI + BCa CI + stability flag.

Shape contract (shard 21 §D — relaxed):
- Input: ``fold_metrics: List[Dict[str, Any]]`` — one dict per fold. Each dict
  MAY include bookkeeping fields (``fold_idx``, ``fold_random_state``,
  ``fold_status``, ``mlflow_run_id``, ``exception_repr``) and arbitrary
  scalar metric fields. Folds with ``fold_status == "failed"`` are skipped
  (cycle-15 I-3 partial-fold contract).
- Output: ``Dict[metric_name, AggregateStat]`` — one entry per scalar metric
  present in at least one ok-fold. Metrics absent from some folds are
  aggregated only over folds where they are present (n_folds reflects this).

The strict "all dicts share same keys" enforcement from shard 21 §D Public API
is RELAXED here — Phase 1 production is robust to per-fold metric availability
gaps (e.g., a fold with all-one-class bootstrap resample skipping AUC). Shard
21 §J item 1 (full nested CV / Phase 2) can re-tighten the contract when the
metric population becomes load-bearing for gate promotion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.utils.bootstrap_utils import bca_confidence_interval

logger = logging.getLogger(__name__)

# Bookkeeping field names that MUST NOT be aggregated as metrics.
_BOOKKEEPING_FIELDS = frozenset(
    {
        "fold_idx",
        "fold_random_state",
        "fold_status",
        "mlflow_run_id",
        "exception_repr",
        "fold_seed",
    }
)


@dataclass(frozen=True)
class AggregateStat:
    """Cross-fold aggregate of a single scalar metric (shard 21 §D)."""

    mean: float
    std: float
    n_folds: int
    percentile_ci_lo: float
    percentile_ci_hi: float
    bca_ci_lo: Optional[float]
    bca_ci_hi: Optional[float]
    bca_unstable_warning: bool
    raw_values: Tuple[float, ...]


def flatten_fold_record(fold_record: Dict[str, Any]) -> Dict[str, float]:
    """Extract scalar metrics from a per-fold record into a flat dict.

    Handles the shape used by ``_run_repeated_splits``:
    - Top-level scalars: ``auc_roc``, ``brier_score``, etc.
    - Nested scalars under ``test_metrics`` / ``validation_metrics`` /
      ``train_metrics`` are emitted with prefixed keys (``test_<name>``, etc.).
    - Bookkeeping fields (``fold_idx``, ``fold_random_state``,
      ``fold_status``, ``mlflow_run_id``, ``exception_repr``) are skipped.
    - Non-numeric fields (strings, dicts, None) are skipped.
    """
    flat: Dict[str, float] = {}
    for key, value in fold_record.items():
        if key in _BOOKKEEPING_FIELDS:
            continue
        if key in ("test_metrics", "validation_metrics", "train_metrics") and isinstance(
            value, dict
        ):
            prefix = key.replace("_metrics", "")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, bool):
                    continue
                if isinstance(sub_value, (int, float)) and np.isfinite(sub_value):
                    flat[f"{prefix}_{sub_key}"] = float(sub_value)
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and np.isfinite(value):
            flat[key] = float(value)
    return flat


def aggregate_fold_metrics(
    fold_metrics: List[Dict[str, Any]],
    *,
    bca_n_resamples: int = 1000,
    bca_confidence_level: float = 0.95,
    bca_rng_seed: int = 42,
    metrics: Optional[Iterable[str]] = None,
) -> Dict[str, AggregateStat]:
    """Aggregate per-fold scalar metrics across k folds.

    Skips folds whose ``fold_status == "failed"`` (cycle-15 I-3 partial-fold
    contract). Auto-flattens nested ``*_metrics`` dicts when present (see
    :func:`flatten_fold_record`). When ``metrics`` is None, the metric set is
    auto-discovered from the union of flattened keys across ok folds; pass an
    explicit iterable to restrict.
    """
    if not fold_metrics:
        return {}

    ok_folds_flat: List[Dict[str, float]] = [
        flatten_fold_record(fm)
        for fm in fold_metrics
        if fm.get("fold_status", "ok") != "failed"
    ]
    if not ok_folds_flat:
        logger.warning(
            "aggregate_fold_metrics: all %d folds have fold_status='failed'; "
            "returning empty aggregate dict",
            len(fold_metrics),
        )
        return {}

    if metrics is None:
        all_keys: set = set()
        for flat in ok_folds_flat:
            all_keys.update(flat.keys())
        metric_names = sorted(all_keys)
    else:
        metric_names = sorted(set(metrics))

    out: Dict[str, AggregateStat] = {}
    for name in metric_names:
        values = [fm[name] for fm in ok_folds_flat if name in fm]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        finite = np.isfinite(arr)
        arr = arr[finite]
        if arr.size == 0:
            continue
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
        if arr.size >= 2:
            ci_lo = float(np.percentile(arr, 2.5))
            ci_hi = float(np.percentile(arr, 97.5))
        else:
            ci_lo = ci_hi = mean
        bca = bca_confidence_interval(
            arr,
            confidence_level=bca_confidence_level,
            n_resamples=bca_n_resamples,
            rng_seed=bca_rng_seed,
        )
        out[name] = AggregateStat(
            mean=mean,
            std=std,
            n_folds=int(arr.size),
            percentile_ci_lo=ci_lo,
            percentile_ci_hi=ci_hi,
            bca_ci_lo=bca.ci_lo,
            bca_ci_hi=bca.ci_hi,
            bca_unstable_warning=bca.unstable_warning,
            raw_values=tuple(float(v) for v in arr.tolist()),
        )
    return out


__all__ = ["AggregateStat", "aggregate_fold_metrics", "flatten_fold_record"]
