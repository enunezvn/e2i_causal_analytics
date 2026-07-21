"""Per-brand gold-standard model-performance aggregation.

The 12 gold-standard models are registered as ``{cohort}_{brand}_goldstd_lr_v1``
(stage='staging', is_synthetic=False) with holdout scalar metrics
(accuracy/precision/recall/f1/auc_roc) in ``ml_performance_metrics``
(source='holdout'). This module averages those metrics per brand for the Home
"Model Accuracy" tile and the WS1-MP-001/003 KPI grid cards.

Pure helpers (``select_goldstd_models``, ``average_holdout``) take plain row
dicts and are unit-tested without a DB. ``summarize_sync`` / ``summarize_async``
wrap the same logic around the sync (KPI calculator) and async (monitoring
service) Supabase clients — the PostgREST query builder is identical; only
``execute()`` differs (awaited on the async client).
"""

from __future__ import annotations

from typing import Any, Optional

GOLDSTD_SUFFIX = "_goldstd_lr_v1"
# Holdout scalar metrics emitted by the gold-standard eval scorer
# (src/mlops/gold_standard_eval/scorer.py:score). accuracy/precision/recall/f1/
# auc_roc are the originals; pr_auc/brier_score/calibration_slope were added so
# WS1-MP-002/005/006 compute per-brand too. Recall@k (WS1-MP-004) stays
# MLflow-only; the Fairness Gap KPI (WS1-MP-008) was removed (it needed a
# designated protected attribute the eval does not carry).
GOLDSTD_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc_roc",
    "pr_auc",
    "brier_score",
    "calibration_slope",
)


def select_goldstd_models(registry_rows: Any, brand: Optional[str]) -> list[dict]:
    """Filter ml_model_registry rows to the gold-standard models for a brand.

    ``registry_rows``: iterable of dicts with at least ``id`` and ``model_name``.
    ``brand``: brand name (case-insensitive) or ``None``/``"all"`` for all 12.
    Rows without the ``_goldstd_lr_v1`` suffix (e.g. the synthetic sweep) are
    always excluded.
    """
    want = (brand or "").strip().lower()
    out: list[dict] = []
    for r in registry_rows or []:
        name = (r.get("model_name") or "").lower()
        if not name.endswith(GOLDSTD_SUFFIX):
            continue
        if want and want != "all" and f"_{want}_goldstd" not in name:
            continue
        out.append(r)
    return out


def average_holdout(models: list[dict], metric_rows: Any) -> Optional[dict[str, Any]]:
    """Aggregate holdout ``metric_value`` per metric over the given models.

    Every metric EXCEPT ``calibration_slope`` is the signed mean over models
    that have a (non-null) ``source='holdout'`` row, or ``None`` if none do
    (never fabricated). Returns ``None`` when there are no models.

    ``calibration_slope`` aggregates as ``1 + mean(|slope_i - 1|)`` — a
    deviation-from-ideal fold mapped back into slope-band units so the
    WS1-MP-006 band in ``config/kpi_definitions.yaml`` (ideal 1.0, tolerances
    0.05/0.15) applies UNCHANGED. Two design flaws of the previous signed mean
    (codex-confirmed, 2026-07-21 frontend review) motivate this:

    * **Signed cancellation**: slope is a both-directions-bad metric (<1
      over-confident, >1 under-confident). Slopes 0.70 and 1.30 signed-mean to
      1.00 and read GOOD; under the fold they read 1.30 = CRITICAL. Opposite
      miscalibrations can no longer cancel into a green headline.
    * **Mirror-pair correlation**: the persistence and discontinuation models
      score the SAME patients with mirrored labels
      (``persistent_180d == 1 - discontinued_180d``), so 2 of a brand's 4
      slots are one correlated holdout draw, not independent evidence. The
      fold cannot remove that correlation — the per-model
      ``calibration_slope_detail`` below exposes each slope (with holdout n
      and bootstrap CI) so the DIRECTION and pairing of deviations stay
      visible instead of vanishing into a single signed number.

    When any ``calibration_slope`` rows exist the summary also carries
    ``calibration_slope_detail``::

        {
          "aggregation": "one_plus_mean_abs_deviation",
          "models": [  # sorted by model_name for a stable payload
            {"model_name": str | None, "slope": float,
             "n": int | None, "ci_lower": float | None, "ci_upper": float | None},
            ...
          ],
        }

    ``n``/``ci_lower``/``ci_upper`` come from the row's ``sample_size`` /
    ``ci_lower`` / ``ci_upper`` columns and degrade to ``None`` on pre-B2 rows
    where the eval has not yet written them (never fabricated).
    """
    model_ids = {m["id"] for m in models}
    if not model_ids:
        return None
    names = {m["id"]: m.get("model_name") for m in models}
    values: dict[str, list[float]] = {}
    slope_detail: list[dict[str, Any]] = []
    for row in metric_rows or []:
        if row.get("model_id") not in model_ids:
            continue
        if (row.get("source") or "") != "holdout":
            continue
        name = row.get("metric_name")
        val = row.get("metric_value")
        if name not in GOLDSTD_METRICS or val is None:
            continue
        values.setdefault(name, []).append(float(val))
        if name == "calibration_slope":
            ci_lo = row.get("ci_lower")
            ci_hi = row.get("ci_upper")
            n = row.get("sample_size")
            slope_detail.append(
                {
                    "model_name": names.get(row.get("model_id")),
                    "slope": float(val),
                    "n": int(n) if n is not None else None,
                    "ci_lower": float(ci_lo) if ci_lo is not None else None,
                    "ci_upper": float(ci_hi) if ci_hi is not None else None,
                }
            )
    summary: dict[str, Any] = {"n_models": len(model_ids)}
    for m in GOLDSTD_METRICS:
        vals = values.get(m)
        if not vals:
            summary[m] = None
        elif m == "calibration_slope":
            summary[m] = 1.0 + sum(abs(v - 1.0) for v in vals) / len(vals)
        else:
            summary[m] = sum(vals) / len(vals)
    if slope_detail:
        slope_detail.sort(key=lambda d: d.get("model_name") or "")
        summary["calibration_slope_detail"] = {
            "aggregation": "one_plus_mean_abs_deviation",
            "models": slope_detail,
        }
    return summary


def _registry_query(client: Any) -> Any:
    return (
        client.table("ml_model_registry")
        .select("id,model_name")
        .eq("stage", "staging")
        .eq("is_synthetic", False)
    )


def _metrics_query(client: Any, model_ids: list[str]) -> Any:
    # sample_size + ci_lower/ci_upper feed the calibration_slope per-model
    # detail (holdout n + bootstrap CI); NULL on rows the eval has not yet
    # rewritten — average_holdout degrades those to None.
    return (
        client.table("ml_performance_metrics")
        .select("model_id,metric_name,metric_value,source,sample_size,ci_lower,ci_upper")
        .in_("model_id", list(model_ids))
        .eq("source", "holdout")
    )


def summarize_sync(client: Any, brand: Optional[str]) -> Optional[dict[str, Any]]:
    """Per-brand gold-standard holdout summary via a SYNC Supabase client."""
    reg = _registry_query(client).execute()
    models = select_goldstd_models(getattr(reg, "data", None), brand)
    if not models:
        return None
    mets = _metrics_query(client, [m["id"] for m in models]).execute()
    return average_holdout(models, getattr(mets, "data", None))


async def summarize_async(client: Any, brand: Optional[str]) -> Optional[dict[str, Any]]:
    """Per-brand gold-standard holdout summary via an ASYNC Supabase client."""
    reg = await _registry_query(client).execute()
    models = select_goldstd_models(getattr(reg, "data", None), brand)
    if not models:
        return None
    mets = await _metrics_query(client, [m["id"] for m in models]).execute()
    return average_holdout(models, getattr(mets, "data", None))
