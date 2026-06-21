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
# Holdout scalar metrics that exist for the gold-standard models (verified
# 2026-06-20). PR-AUC / Brier / calibration / fairness / recall@k are NOT
# present, so they are deliberately absent here.
GOLDSTD_METRICS = ("accuracy", "precision", "recall", "f1", "auc_roc")


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
    """Average holdout ``metric_value`` per metric over the given models.

    Returns ``{n_models, accuracy, precision, recall, f1, auc_roc}`` where each
    metric is the mean over models that have a (non-null) ``source='holdout'``
    row, or ``None`` if none do (never fabricated). Returns ``None`` when there
    are no models.
    """
    model_ids = {m["id"] for m in models}
    if not model_ids:
        return None
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in metric_rows or []:
        if row.get("model_id") not in model_ids:
            continue
        if (row.get("source") or "") != "holdout":
            continue
        name = row.get("metric_name")
        val = row.get("metric_value")
        if name not in GOLDSTD_METRICS or val is None:
            continue
        sums[name] = sums.get(name, 0.0) + float(val)
        counts[name] = counts.get(name, 0) + 1
    summary: dict[str, Any] = {"n_models": len(model_ids)}
    for m in GOLDSTD_METRICS:
        summary[m] = (sums[m] / counts[m]) if counts.get(m) else None
    return summary


def _registry_query(client: Any) -> Any:
    return (
        client.table("ml_model_registry")
        .select("id,model_name")
        .eq("stage", "staging")
        .eq("is_synthetic", False)
    )


def _metrics_query(client: Any, model_ids: list[str]) -> Any:
    return (
        client.table("ml_performance_metrics")
        .select("model_id,metric_name,metric_value,source")
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
