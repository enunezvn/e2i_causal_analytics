"""Sampling-frame audit node for the data_preparer agent.

This node performs a drift comparison between the training distribution and
an optional ``deployment_reference`` declared on ``scope_spec``. It surfaces
sampling-frame mismatches before training and now also acts as a *blocking
gate* (Phase-1 Task 1.3): when the worst per-column drift exceeds
``sampling_frame_max_drift`` (read from ``scope_spec`` with a module-level
default of ``0.3``), the node appends a structured entry to
``state["blocking_issues"]`` so the QC gate fails downstream.

The reference is OPTIONAL — when no ``deployment_reference`` is provided in
``scope_spec`` (the default for synthetic data), the node emits an advisory
``"no_reference_provided"`` entry to ``state["sampling_frame_audit_report"]``
and passes through without blocking.

When present, ``deployment_reference`` must follow this shape:

.. code-block:: python

    {
        "distributions": {
            "<column_name>": {
                # Numeric columns
                "mean": float,
                "std": float,
                "quantiles": {"q25": float, "q50": float, "q75": float},
                # Categorical columns (mutually exclusive with numeric stats)
                "categorical_freq": {"<value>": float, ...},  # frequencies in [0, 1]
            },
            ...
        },
        "n_reference_samples": int,  # optional — used in the report metadata
    }

Drift methodology (one consistent approach, documented up-front):

* **Numeric** — ``standardized_mean_diff``: an average-of-variances
  standardized mean difference (a Cohen's d variant suitable when only
  summary statistics are available, also called Glass's Δ' / equal-n
  Cohen's d): ``|mean_train - mean_ref| / sqrt((s_train² + s_ref²) / 2)``.
  A column is flagged when this exceeds ``numeric_drift_threshold``
  (default ``0.5``). When both stds collapse to 0 the metric is undefined;
  the audit reports a non-finite SMD as ``metric_value=None`` with
  ``status="extreme_drift"`` and ``drift_flagged=True``.
* **Categorical** — Jensen–Shannon divergence between the two frequency
  vectors. A column is flagged when this exceeds
  ``categorical_drift_threshold`` (default ``0.2``). The JS divergence is
  computed in nats (natural log) and is bounded by ``ln(2) ≈ 0.693``.

Blocking gate (Phase-1 Task 1.3):
  The audit computes a single ``max_drift_score`` = max of all per-column
  ``metric_value`` entries (treating non-finite SMD as +inf). When this
  exceeds ``sampling_frame_max_drift`` (default ``0.3``, overridable via
  ``scope_spec["sampling_frame_max_drift"]``), the node appends a stable
  ``"sampling_frame_drift: ..."`` string to ``state["blocking_issues"]``
  and mirrors structured detail (kind, severity, divergence, threshold)
  into ``sampling_frame_audit_report["blocking_detail"]``. Because
  ``run_quality_checks`` overwrites ``blocking_issues`` with a fresh list,
  ``finalize_output`` re-promotes the drift entry from ``blocking_detail``
  so the gate decision is durable across the pipeline.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


# Default drift thresholds (overridable via scope_spec["sampling_frame_audit"])
DEFAULT_NUMERIC_DRIFT_THRESHOLD = 0.5  # Cohen's d
DEFAULT_CATEGORICAL_DRIFT_THRESHOLD = 0.2  # JS divergence (nats)

# Blocking gate threshold (Phase-1 Task 1.3). Compared against ``max_drift_score``,
# which is the worst per-column ``metric_value`` (with non-finite SMD treated as
# +inf). Overridable via ``scope_spec["sampling_frame_max_drift"]``.
DEFAULT_SAMPLING_FRAME_MAX_DRIFT = 0.3

# Stable identifier embedded in the blocking_issues string so callers can grep for
# the gate trip without parsing the full message.
SAMPLING_FRAME_DRIFT_BLOCKING_KIND = "sampling_frame_drift"


async def audit_sampling_frame(state: DataPreparerState) -> Dict[str, Any]:
    """Compare training distribution to ``scope_spec.deployment_reference``.

    The audit writes the full report to ``sampling_frame_audit_report`` and
    additionally promotes excessive drift to a blocking gate: when the worst
    per-column drift exceeds ``sampling_frame_max_drift`` (default ``0.3``),
    a single descriptive entry is appended to ``state["blocking_issues"]``
    and the structured detail is mirrored into
    ``sampling_frame_audit_report["blocking_detail"]``.

    Args:
        state: Current data_preparer agent state. Must contain ``train_df``;
            reads optional ``scope_spec["deployment_reference"]``,
            ``scope_spec["sampling_frame_audit"]`` (per-metric threshold
            overrides), and ``scope_spec["sampling_frame_max_drift"]``
            (blocking-gate threshold override).

    Returns:
        Partial state update with ``sampling_frame_audit_report`` (always),
        and ``blocking_issues`` (only when ``max_drift_score`` exceeds the
        blocking threshold). The report is JSON-serialisable (no numpy
        types, no DataFrames).
    """
    experiment_id = state.get("experiment_id", "unknown")
    logger.info("Running sampling-frame audit for experiment %s", experiment_id)

    scope_spec = state.get("scope_spec") or {}
    deployment_reference = scope_spec.get("deployment_reference")
    audit_config = scope_spec.get("sampling_frame_audit", {})

    numeric_threshold = float(
        audit_config.get("numeric_drift_threshold", DEFAULT_NUMERIC_DRIFT_THRESHOLD)
    )
    categorical_threshold = float(
        audit_config.get("categorical_drift_threshold", DEFAULT_CATEGORICAL_DRIFT_THRESHOLD)
    )
    blocking_threshold = float(
        scope_spec.get("sampling_frame_max_drift", DEFAULT_SAMPLING_FRAME_MAX_DRIFT)
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    # No reference → advisory pass-through
    if not deployment_reference:
        report: Dict[str, Any] = {
            "status": "no_reference_provided",
            "message": (
                "scope_spec['deployment_reference'] not provided; "
                "skipping sampling-frame audit. This is expected for "
                "synthetic-data runs."
            ),
            "drift_detected": False,
            "columns_checked": 0,
            "columns_with_drift": [],
            "per_column": {},
            "thresholds": {
                "numeric_drift_threshold": numeric_threshold,
                "categorical_drift_threshold": categorical_threshold,
            },
            "audited_at": timestamp,
        }
        logger.info(
            "Sampling-frame audit skipped (no deployment_reference) for %s",
            experiment_id,
        )
        return {"sampling_frame_audit_report": report}

    train_df = state.get("train_df")
    if not isinstance(train_df, pd.DataFrame):
        # train_df missing/invalid — record an error-status advisory but do
        # NOT block. The downstream schema/quality nodes are the source of
        # truth for hard failures here. ``isinstance`` is preferred over
        # duck-typed ``hasattr("columns")`` because the audit's downstream
        # logic depends on full DataFrame semantics (``.value_counts``,
        # ``.quantile``, etc.), not just the existence of a ``columns``
        # attribute.
        report = {
            "status": "error",
            "message": "train_df missing or not a DataFrame; audit skipped.",
            "drift_detected": False,
            "columns_checked": 0,
            "columns_with_drift": [],
            "per_column": {},
            "thresholds": {
                "numeric_drift_threshold": numeric_threshold,
                "categorical_drift_threshold": categorical_threshold,
            },
            "audited_at": timestamp,
        }
        logger.warning(
            "Sampling-frame audit skipped: train_df missing/invalid for %s",
            experiment_id,
        )
        return {"sampling_frame_audit_report": report}

    distributions = deployment_reference.get("distributions") or {}
    if not isinstance(distributions, Mapping):
        distributions = {}

    per_column: Dict[str, Dict[str, Any]] = {}
    columns_with_drift: List[str] = []
    columns_checked = 0

    for col, ref_stats in distributions.items():
        if col not in train_df.columns:
            per_column[col] = {
                "status": "skipped_missing_column",
                "message": (
                    f"Column '{col}' present in deployment_reference but missing from train_df."
                ),
                "drift_flagged": False,
            }
            continue

        if not isinstance(ref_stats, Mapping):
            per_column[col] = {
                "status": "skipped_invalid_reference",
                "message": (
                    f"Reference stats for '{col}' must be a mapping; "
                    f"got {type(ref_stats).__name__}."
                ),
                "drift_flagged": False,
            }
            continue

        is_categorical_ref = "categorical_freq" in ref_stats
        is_numeric_ref = "mean" in ref_stats or "std" in ref_stats or "quantiles" in ref_stats

        if is_categorical_ref:
            entry = _audit_categorical(
                train_df[col],
                ref_stats,
                categorical_threshold,
            )
        elif is_numeric_ref:
            entry = _audit_numeric(
                train_df[col],
                ref_stats,
                numeric_threshold,
            )
        else:
            per_column[col] = {
                "status": "skipped_invalid_reference",
                "message": (
                    f"Reference stats for '{col}' lack both numeric "
                    "(mean/std) and categorical (categorical_freq) keys."
                ),
                "drift_flagged": False,
            }
            continue

        per_column[col] = entry
        columns_checked += 1
        if entry.get("drift_flagged"):
            columns_with_drift.append(col)

    drift_detected = len(columns_with_drift) > 0
    status = "drift_detected" if drift_detected else "no_drift"

    # Aggregate worst-column drift into a single score for the blocking gate.
    # ``extreme_drift`` (non-finite SMD on a constant) is treated as +inf, then
    # serialised as ``None`` for RFC 8259 ``allow_nan=False`` compatibility.
    max_drift_score = _max_drift_score(per_column)
    if max_drift_score is None or not math.isfinite(max_drift_score):
        max_drift_for_payload: Optional[float] = None
    else:
        max_drift_for_payload = float(max_drift_score)

    report: Dict[str, Any] = {
        "status": status,
        "drift_detected": drift_detected,
        "columns_checked": columns_checked,
        "columns_with_drift": columns_with_drift,
        "per_column": per_column,
        "max_drift_score": max_drift_for_payload,
        "thresholds": {
            "numeric_drift_threshold": numeric_threshold,
            "categorical_drift_threshold": categorical_threshold,
            "sampling_frame_max_drift": blocking_threshold,
        },
        "n_reference_samples": _coerce_int(deployment_reference.get("n_reference_samples")),
        "n_train_samples": int(len(train_df)),
        "audited_at": timestamp,
    }

    # Blocking-gate decision (Phase-1 Task 1.3). ``max_drift_score is None``
    # means no columns were checked → cannot evaluate drift → do NOT block.
    blocking_triggered = max_drift_score is not None and (
        not math.isfinite(max_drift_score) or max_drift_score > blocking_threshold
    )

    if blocking_triggered:
        worst_col = _worst_drift_column(per_column)
        score_str = "inf" if max_drift_for_payload is None else f"{max_drift_for_payload:.4f}"
        message = (
            f"Sampling-frame drift exceeds blocking threshold: "
            f"max_drift_score={score_str} > {blocking_threshold:.4f} "
            f"(worst column: {worst_col!r}, columns_with_drift={columns_with_drift})"
        )
        report["blocking_detail"] = {
            "kind": SAMPLING_FRAME_DRIFT_BLOCKING_KIND,
            "severity": "high",
            "divergence": max_drift_for_payload,
            "threshold": float(blocking_threshold),
            "worst_column": worst_col,
            "columns_with_drift": list(columns_with_drift),
            "message": message,
        }
        logger.warning(
            "Sampling-frame audit BLOCKING: max_drift=%s > threshold=%.4f "
            "across %d columns (%s) — appending to blocking_issues",
            score_str,
            blocking_threshold,
            len(columns_with_drift),
            ", ".join(columns_with_drift),
        )
        # ``blocking_issues`` is typed ``List[str]`` (state.py); follow the
        # schema_validator/quality_checker pattern: copy + append a stable,
        # grep-able prefix string. Structured detail is in ``blocking_detail``.
        new_blocking = list(state.get("blocking_issues") or [])
        new_blocking.append(f"{SAMPLING_FRAME_DRIFT_BLOCKING_KIND}: {message}")
        return {
            "sampling_frame_audit_report": report,
            "blocking_issues": new_blocking,
        }

    if drift_detected:
        logger.warning(
            "Sampling-frame audit detected drift in %d/%d columns: %s "
            "(below blocking threshold %.4f — pipeline NOT blocked)",
            len(columns_with_drift),
            columns_checked,
            ", ".join(columns_with_drift),
            blocking_threshold,
        )
    else:
        logger.info(
            "Sampling-frame audit: no drift detected across %d columns",
            columns_checked,
        )

    return {"sampling_frame_audit_report": report}


def _audit_numeric(
    series: pd.Series,
    ref_stats: Mapping[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """Compute numeric drift via average-of-variances standardized mean diff.

    The denominator is ``sqrt((s_train² + s_ref²) / 2)`` (a Cohen's d variant
    suitable when only summary statistics are available — also known as
    Glass's Δ' / equal-n Cohen's d). When both stds collapse to 0 the metric
    is undefined; the result reports ``metric_value=None`` with
    ``status="extreme_drift"`` and ``drift_flagged=True`` so the JSON
    payload stays RFC 8259-compliant (no ``Infinity``/``NaN`` literals).
    """
    train_values = pd.to_numeric(series, errors="coerce").dropna()
    n_train = int(train_values.shape[0])

    if n_train == 0:
        return {
            "status": "skipped_empty_train",
            "message": ("No non-null numeric values available in train_df for this column."),
            "drift_flagged": False,
        }

    train_mean = float(train_values.mean())
    train_std = float(train_values.std(ddof=1)) if n_train > 1 else 0.0

    ref_mean = _coerce_float(ref_stats.get("mean"))
    ref_std = _coerce_float(ref_stats.get("std"))

    if ref_mean is None:
        return {
            "status": "skipped_invalid_reference",
            "message": "Reference 'mean' missing or not numeric.",
            "drift_flagged": False,
        }

    # Combined std (average-of-variances form). When both train and reference
    # variances collapse to 0 (both constants) the metric is undefined: we
    # surface that as a non-finite SMD below.
    train_var = train_std**2
    ref_var = ref_std**2 if ref_std is not None else 0.0
    combined_std = math.sqrt((train_var + ref_var) / 2.0) if (train_var + ref_var) > 0 else 0.0

    if combined_std == 0.0:
        # Both are constants — drift is binary (means equal or not).
        smd = 0.0 if math.isclose(train_mean, ref_mean) else float("inf")
    else:
        smd = abs(train_mean - ref_mean) / combined_std

    # Explicit if/else (no operator-precedence puzzle): a non-finite SMD is
    # always treated as drift; otherwise compare against the threshold.
    if not math.isfinite(smd):
        drift_flagged = True
    else:
        drift_flagged = smd > threshold

    # ``quantile_diffs`` is a SUPPLEMENTARY debug payload — it's emitted
    # so consumers (Grafana, Opik) can show the q25/q50/q75 train-vs-ref
    # delta for visual sanity-checking, but the drift decision itself is
    # taken from ``standardized_mean_diff`` above. We deliberately do NOT
    # cross-reference quantile drift in the ``drift_flagged`` boolean —
    # adding a second drift signal would force a multiple-comparisons
    # correction we don't want at this layer.
    quantile_diffs: Dict[str, float] = {}
    ref_quantiles = ref_stats.get("quantiles")
    if isinstance(ref_quantiles, Mapping):
        # Compute equivalent train quantiles where the keys map to standard
        # quantile values: q25 → 0.25, q50 → 0.50, q75 → 0.75. Unknown keys
        # are ignored.
        quantile_map = {"q25": 0.25, "q50": 0.50, "q75": 0.75}
        for key, q in quantile_map.items():
            ref_q = _coerce_float(ref_quantiles.get(key))
            if ref_q is None:
                continue
            train_q = float(train_values.quantile(q))
            quantile_diffs[key] = float(train_q - ref_q)

    # RFC 8259 strict-JSON safety: when SMD is non-finite (infinite or NaN),
    # set metric_value=None and surface the condition via "extreme_drift".
    # ``json.dumps(report, allow_nan=False)`` then succeeds.
    if math.isfinite(smd):
        metric_value: Optional[float] = float(smd)
        entry_status = "checked"
    else:
        metric_value = None
        entry_status = "extreme_drift"

    return {
        "status": entry_status,
        "type": "numeric",
        "metric": "standardized_mean_diff",
        "metric_value": metric_value,
        "threshold": float(threshold),
        "drift_flagged": bool(drift_flagged),
        "train_mean": train_mean,
        "train_std": train_std,
        "reference_mean": float(ref_mean),
        "reference_std": (float(ref_std) if ref_std is not None else None),
        "n_train_samples": n_train,
        "quantile_diffs": quantile_diffs,
    }


def _audit_categorical(
    series: pd.Series,
    ref_stats: Mapping[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """Compute categorical drift via Jensen–Shannon divergence."""
    train_values = series.dropna().astype(str)
    n_train = int(train_values.shape[0])

    if n_train == 0:
        return {
            "status": "skipped_empty_train",
            "message": ("No non-null categorical values available in train_df for this column."),
            "drift_flagged": False,
        }

    raw_ref_freq = ref_stats.get("categorical_freq")
    if not isinstance(raw_ref_freq, Mapping) or not raw_ref_freq:
        return {
            "status": "skipped_invalid_reference",
            "message": ("Reference 'categorical_freq' missing, empty, or not a mapping."),
            "drift_flagged": False,
        }

    # Coerce reference frequencies to a normalised distribution over a
    # union of categories with the train distribution.
    ref_freq: Dict[str, float] = {}
    for key, value in raw_ref_freq.items():
        coerced = _coerce_float(value)
        if coerced is None or coerced < 0.0:
            continue
        ref_freq[str(key)] = coerced

    if not ref_freq:
        return {
            "status": "skipped_invalid_reference",
            "message": ("Reference 'categorical_freq' has no non-negative numeric entries."),
            "drift_flagged": False,
        }

    train_counts = train_values.value_counts()
    train_freq = (train_counts / train_counts.sum()).to_dict()

    js = _jensen_shannon_divergence(train_freq, ref_freq)
    drift_flagged = js > threshold

    return {
        "status": "checked",
        "type": "categorical",
        "metric": "jensen_shannon_divergence",
        "metric_value": float(js),
        "threshold": float(threshold),
        "drift_flagged": bool(drift_flagged),
        "train_freq": {str(k): float(v) for k, v in train_freq.items()},
        "reference_freq": {str(k): float(v) for k, v in ref_freq.items()},
        "n_train_samples": n_train,
    }


def _jensen_shannon_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Symmetric JS divergence (in nats) between two discrete distributions.

    Both inputs are normalised to sum to 1 over the union of their support.
    The result is bounded by ``ln(2) ≈ 0.693``.
    """
    keys = sorted(set(p.keys()) | set(q.keys()))
    if not keys:
        return 0.0

    p_sum = sum(max(0.0, float(v)) for v in p.values()) or 1.0
    q_sum = sum(max(0.0, float(v)) for v in q.values()) or 1.0

    p_vec = np.array(
        [max(0.0, float(p.get(k, 0.0))) / p_sum for k in keys],
        dtype=float,
    )
    q_vec = np.array(
        [max(0.0, float(q.get(k, 0.0))) / q_sum for k in keys],
        dtype=float,
    )

    m_vec = 0.5 * (p_vec + q_vec)

    return 0.5 * _kl_divergence(p_vec, m_vec) + 0.5 * _kl_divergence(q_vec, m_vec)


def _kl_divergence(p_vec: np.ndarray, q_vec: np.ndarray) -> float:
    """KL(p || q) in nats. Skips terms where ``p`` is 0 (0 log 0 := 0)."""
    mask = p_vec > 0
    if not mask.any():
        return 0.0
    # Floor q at 1e-12 only inside the masked positions: this avoids log(0)
    # without biasing terms where p_i > 0 and q_i > 0 normally. The 1e-12
    # epsilon is far below the typical normalised-probability resolution
    # for our use case (counts in the hundreds), so its contribution to
    # the sum is dwarfed by genuine signal whenever the divergence is
    # non-zero.
    safe_q = np.where(q_vec > 0, q_vec, 1e-12)
    return float(np.sum(p_vec[mask] * np.log(p_vec[mask] / safe_q[mask])))


def _max_drift_score(per_column: Mapping[str, Mapping[str, Any]]) -> Optional[float]:
    """Return the worst per-column drift score, or ``None`` if no columns checked.

    A ``status="extreme_drift"`` entry (non-finite SMD on a constant column) is
    treated as ``+inf`` so the blocking gate always trips on it. Skipped
    entries (no metric_value) are ignored.
    """
    worst: Optional[float] = None
    for entry in per_column.values():
        if not isinstance(entry, Mapping):
            continue
        if entry.get("status") == "extreme_drift":
            return float("inf")
        value = entry.get("metric_value")
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value_f):
            return float("inf")
        if worst is None or value_f > worst:
            worst = value_f
    return worst


def _worst_drift_column(per_column: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    """Return the column name with the highest per-column drift score.

    Tiebreaker: first column reaching the max wins (deterministic given the
    mapping iteration order, which in modern Python is insertion-ordered).
    """
    worst_name: Optional[str] = None
    worst_value: Optional[float] = None
    for col, entry in per_column.items():
        if not isinstance(entry, Mapping):
            continue
        if entry.get("status") == "extreme_drift":
            return col
        value = entry.get("metric_value")
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if worst_value is None or value_f > worst_value:
            worst_value = value_f
            worst_name = col
    return worst_name


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort float coercion that returns None on failure or non-finite."""
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced


def _coerce_int(value: Any) -> Optional[int]:
    """Best-effort int coercion that returns None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
