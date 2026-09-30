"""Holdout harness + non-inferiority acceptance rule for candidate activation (#2318).

Before a retrained ``candidate`` replaces the bundle the sidecar serves, both models
are scored on ONE snapshot of the goldstd holdout rows (``data_split`` in test ∪
holdout), each through its OWN persisted preprocessor (no encoder is re-fit: the
exact-artifact principle applied to evaluation), and the candidate must pass every
check below (OD-2, owner-accepted 2026-09-30):

  * AUC non-inferiority — DeLong paired test for correlated ROC curves (DeLong,
    DeLong & Clarke-Pearson 1988; midrank algorithm of Sun & Xu 2014): the one-sided
    ``1 - alpha`` lower bound of ``AUC_cand - AUC_served`` must exceed ``-auc_margin``;
  * Brier non-inferiority — the paired-bootstrap ``1 - alpha`` upper bound of
    ``Brier_cand - Brier_served`` must be below ``brier_margin``;
  * the candidate's Cox calibration slope must lie in ``slope_band``;
  * at least ``min_class_n`` rows of each class, else the harness refuses outright.

What this is, stated honestly: a pre-registered, DETERMINISTIC ACCEPTANCE RULE with a
DeLong-based bound — not a confirmatory alpha = 0.05 trial. The harness rows are
reused across weekly candidates, the candidate's trainer already evaluated on the
``test`` half of them, and the served comparator has seen them. The constants live in
:class:`GateConfig` and in Lane 4's SQL predicate ``activation_gate_passes``; they
change only by a reviewed PR, never per activation.

For ``hcp_adoption_*`` names (served at ``production``) the #1354 calibration
pathology gate also runs, because Lane 4's SQL predicate requires it for production.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import average_precision_score

from src.mlops.gold_standard_eval.scorer import calibration_slope

GATE_KIND = "deterministic_acceptance_rule"

# Lockstep with run_persistence_eval / run_initiation_eval ``_OOS_EVAL_SPLITS``: the
# rows the v1.0 registry ``auc`` is measured on (F22).
OOS_EVAL_SPLITS: tuple[str, ...] = ("test", "holdout")

# The bundle shape a #2318 retrain logs (Lane 2); anything else must be the legacy
# FeatureBuilder bundle {"model", "preprocessor", "feature_columns"}.
SKLEARN_CT_BUNDLE_FORMAT = "sklearn_ct_v1"

# #1354 calibration-pathology slope band for hcp_adoption (moved here from
# scripts/promote_hcp_adoption_champions.py, which now imports it).
HCP_PATHOLOGY_SLOPE_RANGE = (0.5, 2.0)
_HCP_PREFIX = "hcp_adoption_"

# Bootstrap resamples are materialised this many index cells at a time, so memory
# stays flat for large holdouts (2000 x 1766 is ~7 chunks).
_BOOTSTRAP_CHUNK_CELLS = 500_000


@dataclass(frozen=True)
class GateConfig:
    """Frozen acceptance-rule constants (OD-2). Lane 4's SQL mirrors them; a unit
    test there parses the migration and compares. Not overridable from the CLI."""

    auc_margin: float = 0.010
    alpha: float = 0.05
    brier_margin: float = 0.005
    slope_band: tuple[float, float] = (0.8, 1.25)
    bootstrap_b: int = 2000
    seed: int = 0
    min_class_n: int = 100
    # A bootstrap with fewer usable (two-class) resamples than this fails the rule.
    min_usable_bootstrap_frac: float = 0.90


# ---------------------------------------------------------------------------
# Shared scoring helpers (moved from scripts/promote_hcp_adoption_champions.py)
# ---------------------------------------------------------------------------


def positive_class_scores(model: Any, x: Any) -> "np.ndarray":
    """Positive-class probabilities from ``predict_proba``, honoring ``classes_``."""
    proba = np.asarray(model.predict_proba(x), dtype=float)
    if proba.ndim != 2 or proba.shape[1] == 1:
        # Degenerate single-class model: the constant class value is the score
        # (mirrors backfill_goldstd_holdout_metrics._window_scores).
        classes = getattr(model, "classes_", [0])
        return np.full(proba.shape[0], float(classes[0]))
    classes = list(model.classes_)
    pos = classes.index(1) if 1 in classes else 0
    return proba[:, pos]


def calibration_intercept(
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    *,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> float | None:
    """Calibration-in-the-large: intercept-only logistic MLE with offset logit(p).

    The standard companion to the Cox calibration slope (Van Calster framework):
    fit ``logit(P(y=1)) = a + logit(y_score)`` with the slope FIXED at 1; ``a``
    near 0 means the score level matches the outcome rate, ``a > 0`` means the
    model under-predicts. Solved by Newton iteration on the 1-D score equation
    (deterministic; no solver dependency). Returns ``None`` for single-class
    labels or any numerical failure — never a fabricated value.
    """
    y = np.asarray(y_true, dtype=float)
    if np.unique(y).size < 2:
        return None
    eps = 1e-6
    p = np.clip(np.asarray(y_score, dtype=float), eps, 1.0 - eps)
    offset = np.log(p / (1.0 - p))
    a = 0.0
    converged = False
    try:
        for _ in range(max_iter):
            mu = 1.0 / (1.0 + np.exp(-(a + offset)))
            hess = float(np.sum(mu * (1.0 - mu)))
            if not np.isfinite(hess) or hess <= 0.0:
                return None
            step = float(np.sum(y - mu)) / hess
            a += step
            if abs(step) < tol:
                converged = True
                break
        # A non-converged final iterate is NOT reported: better to omit the
        # intercept than to print a number the solver did not actually reach.
        return float(a) if (converged and np.isfinite(a)) else None
    except (FloatingPointError, OverflowError, ValueError):
        return None


def pathology_gate(metrics: dict, prevalence: float) -> tuple[bool, list[str]]:
    """Apply the #1354 calibration pathology gate; returns (ok, hold_reasons)."""
    reasons: list[str] = []
    slope = metrics.get("calibration_slope")
    if slope is None:
        reasons.append("calibration_slope unfittable on the held-out window")
    elif not (HCP_PATHOLOGY_SLOPE_RANGE[0] <= float(slope) <= HCP_PATHOLOGY_SLOPE_RANGE[1]):
        reasons.append(
            f"calibration_slope {float(slope):.4f} outside "
            f"[{HCP_PATHOLOGY_SLOPE_RANGE[0]}, {HCP_PATHOLOGY_SLOPE_RANGE[1]}] "
            "(logits mis-scaled >2x)"
        )
    brier = metrics.get("brier_score")
    baseline = float(prevalence) * (1.0 - float(prevalence))
    if brier is None:
        reasons.append("brier_score missing")
    elif float(brier) >= baseline:
        reasons.append(
            f"brier_score {float(brier):.4f} >= prevalence baseline {baseline:.4f} "
            "(no skill over a constant base-rate forecast)"
        )
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------


def _binary_labels(y: Any) -> "np.ndarray":
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {arr.shape}")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return arr.astype(int)


def delong_paired(y: Any, preds: Any) -> tuple["np.ndarray", "np.ndarray"]:
    """AUCs of ``k`` models scored on the same rows and their DeLong covariance.

    ``preds`` has shape ``(k, n)``. Midranks (``rankdata(method="average")``) give
    ties half credit, so the AUC equals the Mann-Whitney statistic and
    ``sklearn.metrics.roc_auc_score``. The covariance is
    ``S10 / m + S01 / n`` over the structural components of the m positives and
    n negatives (sample covariance, ddof=1), as in pROC's ``var``/``cov``
    with ``method="delong"``.
    """
    labels = _binary_labels(y)
    scores = np.atleast_2d(np.asarray(preds, dtype=float))
    if scores.ndim != 2 or scores.shape[1] != labels.shape[0]:
        raise ValueError(
            f"preds must have shape (k, {labels.shape[0]}), got {np.asarray(preds).shape}"
        )
    pos = scores[:, labels == 1]
    neg = scores[:, labels == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        raise ValueError(f"delong_paired needs at least 2 rows of both classes (got {m}/{n})")
    tx = rankdata(pos, axis=1, method="average")
    ty = rankdata(neg, axis=1, method="average")
    tz = rankdata(np.hstack([pos, neg]), axis=1, method="average")
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v_pos = (tz[:, :m] - tx) / n  # per positive: share of negatives it outranks
    v_neg = 1.0 - (tz[:, m:] - ty) / m  # per negative: share of positives above it
    cov = np.atleast_2d(np.cov(v_pos)) / m + np.atleast_2d(np.cov(v_neg)) / n
    return aucs, cov


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def _weighted_aucs(counts: "np.ndarray", labels: "np.ndarray", scores: "np.ndarray") -> Any:
    """Mann-Whitney AUC per row of ``counts`` (resample multiplicities), ties = 1/2."""
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    c = counts[:, order]
    is_pos = labels[order] == 1
    pos_w = np.add.reduceat(np.where(is_pos, c, 0.0), starts, axis=1)
    neg_w = np.add.reduceat(np.where(is_pos, 0.0, c), starts, axis=1)
    below = np.cumsum(neg_w, axis=1) - neg_w
    num = (pos_w * (below + 0.5 * neg_w)).sum(axis=1)
    return num, pos_w.sum(axis=1), neg_w.sum(axis=1)


def paired_bootstrap(
    y: Any, served: "np.ndarray", candidate: "np.ndarray", cfg: GateConfig
) -> dict[str, Any]:
    """Paired row bootstrap of the Brier and AUC differences (candidate - served).

    Resample ``b`` is ``default_rng(cfg.seed).integers(0, n, (B, n))[b]`` (drawn in
    chunks; numpy's stream is identical). A resample that draws a single class has
    no AUC; it is skipped for BOTH differences and counted.
    """
    labels = _binary_labels(y)
    s = np.asarray(served, dtype=float)
    c = np.asarray(candidate, dtype=float)
    n = labels.shape[0]
    d_brier = (c - labels) ** 2 - (s - labels) ** 2
    rng = np.random.default_rng(cfg.seed)
    chunk = max(1, _BOOTSTRAP_CHUNK_CELLS // max(n, 1))
    auc_deltas: list["np.ndarray"] = []
    brier_deltas: list["np.ndarray"] = []
    skipped = 0
    done = 0
    while done < cfg.bootstrap_b:
        b = min(chunk, cfg.bootstrap_b - done)
        idx = rng.integers(0, n, size=(b, n))
        flat = (np.arange(b)[:, None] * n + idx).ravel()
        counts = np.bincount(flat, minlength=b * n).reshape(b, n).astype(float)
        num_c, npos, nneg = _weighted_aucs(counts, labels, c)
        num_s, _, _ = _weighted_aucs(counts, labels, s)
        ok = (npos > 0) & (nneg > 0)
        skipped += int((~ok).sum())
        denom = npos[ok] * nneg[ok]
        auc_deltas.append(num_c[ok] / denom - num_s[ok] / denom)
        brier_deltas.append((counts[ok] @ d_brier) / n)
        done += b
    auc_arr = np.concatenate(auc_deltas)
    return {
        "auc_deltas": auc_arr,
        "brier_deltas": np.concatenate(brier_deltas),
        "usable": int(auc_arr.shape[0]),
        "skipped_single_class": skipped,
    }


# ---------------------------------------------------------------------------
# The acceptance rule
# ---------------------------------------------------------------------------


def _probabilities(name: str, scores: Any, n: int) -> "np.ndarray":
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(
            f"{name} scores must be 1-D over the same rows as the labels "
            f"(expected {n}, got shape {arr.shape})"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} scores contain non-finite values; refusing to score")
    if (arr < 0.0).any() or (arr > 1.0).any():
        raise ValueError(f"{name} scores fall outside [0, 1]; they are not probabilities")
    return arr


def _is_hcp(model_name: str | None) -> bool:
    return bool(model_name) and str(model_name).startswith(_HCP_PREFIX)


def evaluate_gate(
    y: Any,
    served: Any,
    candidate: Any,
    cfg: GateConfig | None = None,
    *,
    served_bundle_sha256: str | None = None,
    candidate_bundle_sha256: str | None = None,
    snapshot: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Apply the acceptance rule; return a JSON-serialisable report.

    ``served`` and ``candidate`` are positive-class probabilities on the SAME rows as
    ``y``. Raises ``ValueError`` (nothing is decided) on a thin or one-class holdout,
    misaligned inputs, or non-finite / out-of-range scores. Every other outcome is a
    report whose ``passed`` is true iff ``failed_checks`` is empty.
    """
    cfg = cfg or GateConfig()
    labels = _binary_labels(y)
    n = int(labels.shape[0])
    s = _probabilities("served", served, n)
    c = _probabilities("candidate", candidate, n)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos < cfg.min_class_n or n_neg < cfg.min_class_n:
        raise ValueError(
            f"holdout needs at least {cfg.min_class_n} rows of each class "
            f"(got {n_pos} positives, {n_neg} negatives); refusing to gate"
        )
    failed: list[str] = []

    # AUC non-inferiority (DeLong).
    aucs, cov = delong_paired(labels, np.vstack([s, c]))
    auc_delta = float(aucs[1] - aucs[0])
    var_delta = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    se_delta = math.sqrt(max(var_delta, 0.0))
    auc_lower_bound = auc_delta - float(norm.ppf(1.0 - cfg.alpha)) * se_delta
    if not auc_lower_bound > -cfg.auc_margin:
        failed.append("auc_noninferiority")

    # Brier non-inferiority (paired bootstrap); the AUC bootstrap is a cross-check only.
    brier_served = float(np.mean((s - labels) ** 2))
    brier_candidate = float(np.mean((c - labels) ** 2))
    boot = paired_bootstrap(labels, s, c, cfg)
    usable = int(boot["usable"])
    if usable < cfg.min_usable_bootstrap_frac * cfg.bootstrap_b:
        failed.append("bootstrap_unusable")
    brier_delta_upper: float | None = None
    bootstrap_auc_delta_p05: float | None = None
    if usable > 0:
        brier_delta_upper = float(np.quantile(boot["brier_deltas"], 1.0 - cfg.alpha))
        bootstrap_auc_delta_p05 = float(np.quantile(boot["auc_deltas"], cfg.alpha))
    if brier_delta_upper is None or not brier_delta_upper < cfg.brier_margin:
        failed.append("brier_noninferiority")

    # Candidate calibration.
    slope = calibration_slope(labels, c)
    slope = None if slope is None or not math.isfinite(slope) else float(slope)
    if slope is None:
        failed.append("calibration_slope_unfittable")
    elif not (cfg.slope_band[0] <= slope <= cfg.slope_band[1]):
        failed.append("calibration_slope")
    intercept = calibration_intercept(labels, c)
    prevalence = n_pos / n

    report: dict[str, Any] = {
        "kind": GATE_KIND,
        "n": n,
        "n_pos": n_pos,
        "prevalence": prevalence,
        "auc_served": float(aucs[0]),
        "auc_candidate": float(aucs[1]),
        "auc_delta": auc_delta,
        "se_delta": se_delta,
        "auc_lower_bound": auc_lower_bound,
        "delong_corr": (
            float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]))
            if cov[0, 0] > 0 and cov[1, 1] > 0
            else None
        ),
        "brier_served": brier_served,
        "brier_candidate": brier_candidate,
        "brier_delta_upper": brier_delta_upper,
        "bootstrap_auc_delta_p05": bootstrap_auc_delta_p05,
        "bootstrap_usable": usable,
        "bootstrap_skipped_single_class": int(boot["skipped_single_class"]),
        "pr_auc_served": float(average_precision_score(labels, s)),
        "pr_auc_candidate": float(average_precision_score(labels, c)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "served_bundle_sha256": served_bundle_sha256,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "snapshot": snapshot,
        "model_name": model_name,
    }

    if _is_hcp(model_name):
        ok, reasons = pathology_gate(
            {"calibration_slope": slope, "brier_score": brier_candidate}, prevalence
        )
        report["hcp_pathology_passed"] = ok
        report["hcp_pathology_slope_ok"] = slope is not None and (
            HCP_PATHOLOGY_SLOPE_RANGE[0] <= slope <= HCP_PATHOLOGY_SLOPE_RANGE[1]
        )
        report["hcp_pathology_brier_ok"] = brier_candidate < prevalence * (1.0 - prevalence)
        report["hcp_pathology_reasons"] = reasons
        if not ok:
            failed.append("hcp_pathology")

    report["failed_checks"] = failed
    report["passed"] = not failed
    report["config"] = json.loads(json.dumps(asdict(cfg)))
    # The report is stored as jsonb (Lane 4): NaN / inf must never reach it.
    json.dumps(report, allow_nan=False)
    return report


# ---------------------------------------------------------------------------
# Scoring a persisted bundle (no re-fit)
# ---------------------------------------------------------------------------


def _is_feature_builder_bundle(bundle: dict[str, Any]) -> bool:
    pre = bundle.get("preprocessor")
    return (
        bundle.get("bundle_format") is None
        and bool(bundle.get("feature_columns"))
        and hasattr(pre, "keep_columns")
        and hasattr(pre, "transform")
    )


def bundle_keep_columns(bundle: dict[str, Any]) -> list[str]:
    """The raw covariates a bundle consumes (the serving contract)."""
    if bundle.get("bundle_format") == SKLEARN_CT_BUNDLE_FORMAT:
        return list(bundle["keep_columns"])
    if _is_feature_builder_bundle(bundle):
        return list(bundle["preprocessor"].keep_columns)
    raise ValueError(f"unrecognised serving bundle format {bundle.get('bundle_format')!r}")


def require_same_raw_contract(served: dict[str, Any], candidate: dict[str, Any]) -> None:
    """Refuse a candidate whose raw covariates differ from the served model's: the API
    and Feast supply the served contract (F7), so a different set cannot be served."""
    s_cols = set(bundle_keep_columns(served))
    c_cols = set(bundle_keep_columns(candidate))
    if s_cols != c_cols:
        raise ValueError(
            "raw contract differs: served keep_columns "
            f"{sorted(s_cols)} vs candidate {sorted(c_cols)}"
        )


def score_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> "np.ndarray":
    """Positive-class scores of ``bundle`` on ``frame``, through the bundle's OWN fitted
    preprocessor, exactly as the sidecar feeds it (``np.asarray`` of the encoding).
    Never calls ``fit``."""
    if not isinstance(bundle, dict) or "model" not in bundle or "preprocessor" not in bundle:
        raise ValueError("not a serving bundle: needs 'model' and 'preprocessor'")
    pre = bundle["preprocessor"]
    feature_columns = list(bundle.get("feature_columns") or [])
    if bundle.get("bundle_format") == SKLEARN_CT_BUNDLE_FORMAT:
        keep = list(bundle["keep_columns"])
        missing = [col for col in keep if col not in frame.columns]
        if missing:
            raise ValueError(f"holdout frame is missing covariate(s) {missing}")
        x = np.asarray(pre.transform(frame[keep]), dtype=float)
        if x.shape[1] != len(feature_columns):
            raise ValueError(
                f"encoder produced {x.shape[1]} columns; the bundle declares "
                f"{len(feature_columns)} feature_columns"
            )
    elif _is_feature_builder_bundle(bundle):
        encoded = pre.transform(frame)
        if list(encoded.columns) != feature_columns:
            # The sidecar feeds np.asarray(transform(...)) positionally; an order that
            # differs from feature_columns would score something other than what serves.
            raise ValueError("encoder output columns differ from the bundle's feature_columns")
        x = encoded.to_numpy(dtype=float)
    else:
        raise ValueError(f"unrecognised serving bundle format {bundle.get('bundle_format')!r}")
    scores = positive_class_scores(bundle["model"], x)
    return _probabilities("bundle", scores, len(frame))


# ---------------------------------------------------------------------------
# One holdout snapshot
# ---------------------------------------------------------------------------


def _hash_column(col: pd.Series) -> pd.Series:
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col.astype(object)
    if pd.api.types.is_datetime64_any_dtype(col.dtype):
        utc = col.dt.tz_localize("UTC") if col.dt.tz is None else col.dt.tz_convert("UTC")
        return utc.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").astype(object)
    return col


def rows_sha256(frame: pd.DataFrame, key: str, label: str, keep_columns: Sequence[str]) -> str:
    """sha256 of the rows the gate scores: key, ``data_split``, label and the raw
    covariates, columns in sorted order, rows sorted by ``key``. Categoricals hash as
    their values and datetimes as ISO-8601 UTC, so a re-load of the same rows hashes
    the same regardless of dtype or row order."""
    cols = sorted({key, "data_split", label, *keep_columns})
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"cannot hash the snapshot: missing column(s) {missing}")
    sub = frame[cols].sort_values(key, kind="mergesort").reset_index(drop=True)
    sub = sub.apply(_hash_column)
    text = sub.to_csv(index=False, float_format="%.17g", na_rep="<NA>")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def load_holdout_snapshot(
    db: Any, spec: Any, *, splits: Sequence[str] = OOS_EVAL_SPLITS
) -> tuple[pd.DataFrame, "np.ndarray", dict[str, Any]]:
    """Load the holdout rows once and describe them.

    Reads ``patient_journeys`` / ``hcp_brand_adoption`` through
    ``FeatureBuilder.load_frame`` (read-only; synthetic rows, as the eval does).
    Returns ``(frame, y, snapshot)`` with ``frame`` sorted by the grain's key
    (``patient_id`` or ``hcp_id``, asserted unique) and ``y`` aligned to it.
    """
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder

    fb = FeatureBuilder(spec)
    raw = await fb.load_frame(db, splits=list(splits))
    if raw.empty or "data_split" not in raw.columns:
        raise ValueError(
            f"holdout snapshot has no rows for cohort={spec.name!r} brand={spec.brand!r} "
            f"splits={list(splits)}; refusing to gate"
        )
    frame = raw.loc[raw["data_split"].isin(list(splits))]
    if frame.empty:
        raise ValueError(f"holdout snapshot has no rows in splits={list(splits)}")
    key = "hcp_id" if spec.grain == "hcp" else "patient_id"
    if key not in frame.columns:
        raise ValueError(f"holdout frame has no {key!r} column")
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f"holdout rows are not unique by {key!r}; refusing to gate")
    label = spec.label_column
    if label not in frame.columns or frame[label].isna().any():
        raise ValueError(f"holdout label {label!r} is missing or null on some rows")
    frame = frame.sort_values(key, kind="mergesort").reset_index(drop=True)
    y = _binary_labels(frame[label].astype(int).to_numpy())
    snapshot = {
        "splits": list(splits),
        "n": int(len(frame)),
        "n_pos": int(y.sum()),
        "rows_sha256": rows_sha256(frame, key, label, fb.keep_columns),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "key": key,
        "label_column": label,
        "keep_columns": list(fb.keep_columns),
        "cohort": spec.name,
        "brand": spec.brand,
    }
    return frame, y, snapshot
