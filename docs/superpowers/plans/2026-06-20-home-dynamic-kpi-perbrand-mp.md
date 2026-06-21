# Home Dashboard — Dynamic KPI Visibility + Per-Brand Model Performance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Home dashboard show only KPI cards that actually compute (brand-reactive), surface calculable brand-specific KPIs, and replace the invariant "Model Accuracy" tile + the WS1-MP-001/003 grid cards with per-brand averages sourced from the gold-standard eval holdout metrics.

**Architecture:** A pure aggregation module (`src/kpi/goldstd_model_perf.py`) averages the 12 gold-standard models' holdout metrics per brand. A new async `PerformanceTracker.get_brand_goldstd_summary` + `GET /monitoring/performance/brand-summary` endpoint feeds the tile; the sync `ModelPerformanceCalculator` reuses the same pure logic for the WS1-MP-001/003 grid KPIs. The frontend filters the KPI grid/tabs/counts to computed-only cards and points the tile at the new endpoint. No DB migration (reuses the service-layer direct-read pattern that already serves confusion/ROC).

**Tech Stack:** Python 3.12 / FastAPI / Supabase (async + sync `postgrest` clients) / pytest; React + TypeScript / TanStack Query / Zod / Vitest + Testing Library.

**Spec:** `docs/superpowers/specs/2026-06-20-home-dynamic-kpi-perbrand-mp-design.md`

**Verified data (deployed Supabase, 2026-06-20):** 12 gold-standard models `{cohort}_{brand}_goldstd_lr_v1` (`stage='staging'`, `is_synthetic=false`); `ml_performance_metrics` `source='holdout'` rows with `metric_name` ∈ {accuracy, precision, recall, f1, auc_roc}, 12 each. Per-brand avg accuracy: Fabhalta 0.710 · Kisqali 0.700 · Remibrutinib 0.703 · All 0.704.

---

## File Structure

- **Create** `src/kpi/goldstd_model_perf.py` — pure helpers (`select_goldstd_models`, `average_holdout`) + thin sync/async readers (`summarize_sync`, `summarize_async`).
- **Modify** `src/services/performance_tracking.py` — add async `PerformanceTracker.get_brand_goldstd_summary`.
- **Modify** `src/api/routes/monitoring.py` — add `BrandPerformanceSummaryResponse` + `GET /performance/brand-summary`.
- **Modify** `src/kpi/calculators/model_performance.py` — `_calc_roc_auc` / `_calc_f1_score` prefer the per-brand gold-standard average; add `_goldstd_metric` helper.
- **Modify** `frontend/src/lib/api-schemas.ts` — `BrandModelSummaryWireSchema`.
- **Modify** `frontend/src/api/monitoring.ts` + `frontend/src/hooks/api/use-monitoring.ts` — `getBrandModelSummary` client + `useBrandModelSummary` hook.
- **Modify** `frontend/src/pages/Home.tsx` — `QuickStatTile` `sublabel`; tile → `useBrandModelSummary`; dynamic visibility (`computedKPIs` drives grid/tabs/counts); unhide `brand_specific`; "N of M" note; loading/empty states.
- **Modify** `frontend/src/pages/Home.test.tsx` — update the brand_specific test; add visibility + tile tests.
- **Tests** `tests/unit/test_kpi/test_goldstd_model_perf.py`, `tests/unit/test_services/test_brand_goldstd_summary.py`, `tests/unit/test_api/test_brand_performance_summary.py`, `tests/unit/test_kpi/test_model_performance.py` (extend).

---

## Task 1: Pure per-brand gold-standard aggregation module

**Files:**
- Create: `src/kpi/goldstd_model_perf.py`
- Test: `tests/unit/test_kpi/test_goldstd_model_perf.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_kpi/test_goldstd_model_perf.py
from src.kpi.goldstd_model_perf import (
    GOLDSTD_METRICS,
    average_holdout,
    select_goldstd_models,
)

REG = [
    {"id": "1", "model_name": "initiation_kisqali_goldstd_lr_v1"},
    {"id": "2", "model_name": "persistence_kisqali_goldstd_lr_v1"},
    {"id": "3", "model_name": "initiation_fabhalta_goldstd_lr_v1"},
    {"id": "9", "model_name": "synth_kisqali_exp_0001_model_1"},  # sweep, excluded
]


def test_select_filters_by_brand_case_insensitive_and_suffix():
    out = select_goldstd_models(REG, "Kisqali")
    assert {r["id"] for r in out} == {"1", "2"}  # excludes fabhalta + synth sweep


def test_select_all_returns_every_goldstd_model():
    for brand in (None, "", "all", "ALL"):
        out = select_goldstd_models(REG, brand)
        assert {r["id"] for r in out} == {"1", "2", "3"}  # all 3 goldstd, no synth


def test_average_holdout_means_only_present_values():
    models = [{"id": "1"}, {"id": "2"}]
    rows = [
        {"model_id": "1", "metric_name": "accuracy", "metric_value": 0.6, "source": "holdout"},
        {"model_id": "2", "metric_name": "accuracy", "metric_value": 0.8, "source": "holdout"},
        {"model_id": "1", "metric_name": "f1", "metric_value": 0.4, "source": "holdout"},
        {"model_id": "2", "metric_name": "auc_roc", "metric_value": 0.9, "source": "backtest_wf"},  # wrong source, ignored
    ]
    summary = average_holdout(models, rows)
    assert summary["n_models"] == 2
    assert summary["accuracy"] == 0.7          # (0.6+0.8)/2
    assert summary["f1"] == 0.4                # single value
    assert summary["auc_roc"] is None          # only a backtest_wf row -> not counted


def test_average_holdout_none_when_no_models():
    assert average_holdout([], []) is None


def test_goldstd_metrics_constant_is_the_verified_set():
    assert set(GOLDSTD_METRICS) == {"accuracy", "precision", "recall", "f1", "auc_roc"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_kpi/test_goldstd_model_perf.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.kpi.goldstd_model_perf'`.

- [ ] **Step 3: Write the implementation**

```python
# src/kpi/goldstd_model_perf.py
"""Per-brand gold-standard model-performance aggregation.

The 12 gold-standard models are registered as ``{cohort}_{brand}_goldstd_lr_v1``
(stage='staging', is_synthetic=False) with holdout scalar metrics
(accuracy/precision/recall/f1/auc_roc) in ``ml_performance_metrics``
(source='holdout'). This module averages those metrics per brand for the Home
"Model Accuracy" tile and the WS1-MP-001/003 KPI grid cards.

Pure helpers (`select_goldstd_models`, `average_holdout`) take plain row dicts and
are unit-tested without a DB. `summarize_sync` / `summarize_async` wrap the same
logic around the sync (KPI calculator) and async (monitoring service) Supabase
clients — the PostgREST query builder is identical; only ``execute()`` differs
(awaited on the async client).
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/unit/test_kpi/test_goldstd_model_perf.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/kpi/goldstd_model_perf.py tests/unit/test_kpi/test_goldstd_model_perf.py
git commit -m "feat(kpi): per-brand gold-standard model-perf aggregation (pure + sync/async readers)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Service method + brand-summary endpoint

**Files:**
- Modify: `src/services/performance_tracking.py` (add method after `get_roc_curve`, ~line 449)
- Modify: `src/api/routes/monitoring.py` (response model near `RocCurveResponse` ~line 1336; route before the first `/performance/{model_id}` GET, ~line 1138)
- Test: `tests/unit/test_services/test_brand_goldstd_summary.py`, `tests/unit/test_api/test_brand_performance_summary.py`

- [ ] **Step 1: Write the failing service test**

```python
# tests/unit/test_services/test_brand_goldstd_summary.py
import pytest
from src.services.performance_tracking import PerformanceTracker


class _Resp:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    """Minimal async PostgREST builder stub: records table, returns canned data."""
    def __init__(self, store, table):
        self._store, self._table = store, table

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def in_(self, *_a, **_k):
        return self

    async def execute(self):
        return _Resp(self._store.get(self._table, []))


class _FakeClient:
    def __init__(self, store):
        self._store = store

    def table(self, name):
        return _FakeQuery(self._store, name)


@pytest.mark.asyncio
async def test_get_brand_goldstd_summary_averages_holdout(monkeypatch):
    store = {
        "ml_model_registry": [
            {"id": "1", "model_name": "initiation_kisqali_goldstd_lr_v1"},
            {"id": "2", "model_name": "persistence_kisqali_goldstd_lr_v1"},
        ],
        "ml_performance_metrics": [
            {"model_id": "1", "metric_name": "accuracy", "metric_value": 0.69, "source": "holdout"},
            {"model_id": "2", "metric_name": "accuracy", "metric_value": 0.71, "source": "holdout"},
        ],
    }

    async def _fake_client():
        return _FakeClient(store)

    monkeypatch.setattr(
        "src.repositories.drift_monitoring.get_drift_monitoring_client", _fake_client
    )
    summary = await PerformanceTracker().get_brand_goldstd_summary("Kisqali")
    assert summary["brand"] == "Kisqali"
    assert summary["n_models"] == 2
    assert summary["accuracy"] == pytest.approx(0.70)
    assert summary["is_synthetic_cohort"] is True


@pytest.mark.asyncio
async def test_get_brand_goldstd_summary_none_when_no_models(monkeypatch):
    async def _fake_client():
        return _FakeClient({"ml_model_registry": [], "ml_performance_metrics": []})

    monkeypatch.setattr(
        "src.repositories.drift_monitoring.get_drift_monitoring_client", _fake_client
    )
    assert await PerformanceTracker().get_brand_goldstd_summary("Kisqali") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_services/test_brand_goldstd_summary.py -q`
Expected: FAIL — `AttributeError: 'PerformanceTracker' object has no attribute 'get_brand_goldstd_summary'`.

- [ ] **Step 3: Add the service method**

Insert into `class PerformanceTracker` in `src/services/performance_tracking.py`, immediately after `get_roc_curve` (before the closing of the class, ~line 449):

```python
    async def get_brand_goldstd_summary(
        self,
        brand: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Per-brand average of the gold-standard models' holdout metrics.

        Averages accuracy/precision/recall/f1/auc_roc across the brand's
        ``*_goldstd_lr_v1`` staging models (``brand=None``/``"all"`` -> all 12).
        Returns ``None`` when no gold-standard models are found (honest empty —
        the endpoint renders ``available=false``, never a fabricated 0).
        """
        from src.kpi.goldstd_model_perf import summarize_async
        from src.repositories.drift_monitoring import get_drift_monitoring_client

        client = await get_drift_monitoring_client()
        summary = await summarize_async(client, brand)
        if summary is None:
            return None
        # The gold-standard models are is_synthetic=False, but the eval cohort is
        # synthetic demo data — carried for honest disclosure by consumers.
        return {"brand": brand or "all", **summary, "is_synthetic_cohort": True}
```

- [ ] **Step 4: Run service test to verify it passes**

Run: `python3 -m pytest tests/unit/test_services/test_brand_goldstd_summary.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Write the failing endpoint test**

```python
# tests/unit/test_api/test_brand_performance_summary.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import monitoring


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(monitoring.router, prefix="/api")
    return TestClient(app)


def test_brand_summary_available(client, monkeypatch):
    class _Tracker:
        async def get_brand_goldstd_summary(self, brand):
            return {
                "brand": brand or "all", "n_models": 4, "accuracy": 0.70,
                "precision": 0.66, "recall": 0.55, "f1": 0.60, "auc_roc": 0.75,
                "is_synthetic_cohort": True,
            }

    monkeypatch.setattr(
        "src.services.performance_tracking.get_performance_tracker", lambda: _Tracker()
    )
    r = client.get("/api/monitoring/performance/brand-summary", params={"brand": "Kisqali"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["n_models"] == 4
    assert body["accuracy"] == 0.70
    assert body["brand"] == "Kisqali"


def test_brand_summary_honest_empty(client, monkeypatch):
    class _Tracker:
        async def get_brand_goldstd_summary(self, brand):
            return None

    monkeypatch.setattr(
        "src.services.performance_tracking.get_performance_tracker", lambda: _Tracker()
    )
    r = client.get("/api/monitoring/performance/brand-summary")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["accuracy"] is None
    assert body["brand"] == "all"
```

- [ ] **Step 6: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_api/test_brand_performance_summary.py -q`
Expected: FAIL — 404 (route not defined) / model import error.

- [ ] **Step 7: Add the response model + route in `monitoring.py`**

Add the response model immediately after `RocCurveResponse` (~line 1336):

```python
class BrandPerformanceSummaryResponse(BaseModel):
    """Per-brand average of the gold-standard models' holdout metrics.

    ``available=false`` is an HONEST empty state (no gold-standard models found),
    not an error — never a fabricated 0. ``is_synthetic_cohort`` flags that the
    eval cohort is synthetic demo data.
    """

    model_config = ConfigDict(protected_namespaces=())

    brand: str
    available: bool
    n_models: int = 0
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc_roc: Optional[float] = None
    is_synthetic_cohort: bool = False
```

Add the route immediately BEFORE the first `@router.get("/performance/{model_id}...` (i.e. before `get_performance_trend`, ~line 1138) so the static `brand-summary` segment is never shadowed by a dynamic `{model_id}`:

```python
@router.get(
    "/performance/brand-summary",
    response_model=BrandPerformanceSummaryResponse,
    summary="Per-brand gold-standard model-performance averages",
    operation_id="get_brand_performance_summary",
)
async def get_brand_performance_summary(
    brand: Optional[str] = Query(
        default=None,
        description="Brand filter (case-insensitive); omitted/'all' = all gold-standard models",
    ),
) -> BrandPerformanceSummaryResponse:
    """Average accuracy/precision/recall/f1/auc_roc over the brand's gold-standard
    models, or ``available=false`` when none are found (honest empty state)."""
    from src.services.performance_tracking import get_performance_tracker

    try:
        tracker = get_performance_tracker()
        summary = await tracker.get_brand_goldstd_summary(brand)
        if summary is None:
            return BrandPerformanceSummaryResponse(brand=(brand or "all"), available=False)
        return BrandPerformanceSummaryResponse(available=True, **summary)
    except Exception as e:
        raise _log_and_500("Failed to load brand performance summary", e)
```

- [ ] **Step 8: Run both test files**

Run: `python3 -m pytest tests/unit/test_services/test_brand_goldstd_summary.py tests/unit/test_api/test_brand_performance_summary.py -q`
Expected: PASS (4 tests).

- [ ] **Step 9: Commit**

```bash
git add src/services/performance_tracking.py src/api/routes/monitoring.py tests/unit/test_services/test_brand_goldstd_summary.py tests/unit/test_api/test_brand_performance_summary.py
git commit -m "feat(monitoring): GET /performance/brand-summary (per-brand gold-standard averages)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Per-brand WS1-MP-001 (ROC-AUC) + WS1-MP-003 (F1) in the calculator

**Files:**
- Modify: `src/kpi/calculators/model_performance.py` (`_calc_roc_auc` ~155, `_calc_f1_score` ~186; add `_goldstd_metric` helper)
- Test: `tests/unit/test_kpi/test_model_performance.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_kpi/test_model_performance.py  (add)
from src.kpi.calculators.model_performance import ModelPerformanceCalculator
from src.kpi.models import KPIMetadata, Workstream


class _Resp:
    def __init__(self, data):
        self.data = data


class _SyncQuery:
    def __init__(self, store, table):
        self._store, self._table = store, table

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def in_(self, *_a, **_k):
        return self

    def execute(self):
        return _Resp(self._store.get(self._table, []))


class _SyncClient:
    def __init__(self, store):
        self._store = store

    def table(self, name):
        return _SyncQuery(self._store, name)


def _kpi(kpi_id):
    return KPIMetadata(id=kpi_id, name=kpi_id, workstream=Workstream.WS1_MODEL_PERFORMANCE)


def test_roc_auc_uses_per_brand_goldstd_average():
    store = {
        "ml_model_registry": [
            {"id": "1", "model_name": "initiation_kisqali_goldstd_lr_v1"},
            {"id": "2", "model_name": "persistence_kisqali_goldstd_lr_v1"},
        ],
        "ml_performance_metrics": [
            {"model_id": "1", "metric_name": "auc_roc", "metric_value": 0.68, "source": "holdout"},
            {"model_id": "2", "metric_name": "auc_roc", "metric_value": 0.76, "source": "holdout"},
        ],
    }
    calc = ModelPerformanceCalculator(db_client=_SyncClient(store))
    result = calc.calculate(_kpi("WS1-MP-001"), {"brand": "Kisqali"})
    assert result.value == 0.72  # (0.68+0.76)/2
    assert result.error is None


def test_f1_uses_per_brand_goldstd_average():
    store = {
        "ml_model_registry": [{"id": "1", "model_name": "persistence_fabhalta_goldstd_lr_v1"}],
        "ml_performance_metrics": [
            {"model_id": "1", "metric_name": "f1", "metric_value": 0.69, "source": "holdout"},
        ],
    }
    calc = ModelPerformanceCalculator(db_client=_SyncClient(store))
    result = calc.calculate(_kpi("WS1-MP-003"), {"brand": "Fabhalta"})
    assert result.value == 0.69


def test_roc_auc_falls_back_when_no_goldstd(monkeypatch):
    # Empty registry -> gold-standard returns None -> falls back to the existing
    # corpus SQL leg (which we stub to also be empty) -> MLflow fail-closed.
    calc = ModelPerformanceCalculator(db_client=_SyncClient({}))
    monkeypatch.setattr(calc, "_execute_query", lambda *_a, **_k: ([], None))
    monkeypatch.setattr(calc, "_get_metric_from_mlflow", lambda *_a, **_k: (None, "model_not_found:default_model"))
    result = calc.calculate(_kpi("WS1-MP-001"), {"brand": "Kisqali"})
    assert result.value is None
    assert "model_not_found" in (result.error or "")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_kpi/test_model_performance.py -q -k "goldstd or per_brand"`
Expected: FAIL — `_calc_roc_auc` reads the corpus SQL leg / MLflow, not the gold-standard average; `WS1-MP-001` returns a non-0.72 value (or the fake client's `.table` is never called).

- [ ] **Step 3: Add `_goldstd_metric` and rewire the two `_calc_*` methods**

Add the helper to `ModelPerformanceCalculator` (e.g. just above `_calc_roc_auc`):

```python
    def _goldstd_metric(
        self, context: dict[str, Any], metric_name: str
    ) -> float | None:
        """Per-brand average of the gold-standard models' holdout ``metric_name``.

        Best-effort PRIMARY source for the dashboard: ``context['brand']`` scopes
        to that brand's ``*_goldstd_lr_v1`` staging models (absent/All -> all 12).
        Returns ``None`` (caller falls back to the existing corpus/MLflow legs)
        when no gold-standard data is available or a read fails — never raises,
        never fabricates.
        """
        from src.kpi.goldstd_model_perf import summarize_sync

        try:
            summary = summarize_sync(self.db_client, context.get("brand"))
        except Exception:
            return None
        if not summary:
            return None
        val = summary.get(metric_name)
        return float(val) if val is not None else None
```

Replace `_calc_roc_auc` (keep its docstring's fallback note; prepend the gold-standard leg):

```python
    def _calc_roc_auc(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-001: ROC-AUC.

        PRIMARY: per-brand average of the gold-standard models' holdout ``auc_roc``
        (brand-reactive; fixes the corpus-wide invariant value). FALLBACKS
        (unchanged): the ``ml_predictions`` corpus SQL leg, then MLflow — never a
        fabricated default.
        """
        gs = self._goldstd_metric(context, "auc_roc")
        if gs is not None:
            return gs, None
        result, db_error = self._execute_query("model_performance_roc_auc", [])
        if db_error is None and result:
            roc_auc = result[0].get("roc_auc")
            if roc_auc is not None:
                return float(roc_auc), None
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "roc_auc")
```

Replace `_calc_f1_score`:

```python
    def _calc_f1_score(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-003: F1 Score.

        PRIMARY: per-brand average of the gold-standard models' holdout ``f1``.
        FALLBACK: MLflow (fail-closed; no fabricated default).
        """
        gs = self._goldstd_metric(context, "f1")
        if gs is not None:
            return gs, None
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "f1_score")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/unit/test_kpi/test_model_performance.py -q`
Expected: PASS (new tests + existing ones still green).

- [ ] **Step 5: Commit**

```bash
git add src/kpi/calculators/model_performance.py tests/unit/test_kpi/test_model_performance.py
git commit -m "feat(kpi): WS1-MP-001/003 per-brand from gold-standard holdout (fallbacks preserved)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Frontend client + schema + hook for brand summary

**Files:**
- Modify: `frontend/src/lib/api-schemas.ts` (add schema; mirror `KPIResultWireSchema` style)
- Modify: `frontend/src/api/monitoring.ts` (add `getBrandModelSummary`; mirror the existing alerts client + `get(...)` usage from `frontend/src/api/kpi.ts`)
- Modify: `frontend/src/hooks/api/use-monitoring.ts` (add `useBrandModelSummary`; mirror `useAlerts`)

> Read `frontend/src/api/kpi.ts` (`getKPIValue`) and `frontend/src/hooks/api/use-monitoring.ts` (`useAlerts`) first for the exact `get(...)` wrapper + `queryKey` conventions in those files.

- [ ] **Step 1: Add the wire schema** (in `frontend/src/lib/api-schemas.ts`)

```typescript
export const BrandModelSummaryWireSchema = z.object({
  brand: z.string(),
  available: z.boolean(),
  n_models: z.number().int().nonnegative(),
  accuracy: z.number().nullable().optional(),
  precision: z.number().nullable().optional(),
  recall: z.number().nullable().optional(),
  f1: z.number().nullable().optional(),
  auc_roc: z.number().nullable().optional(),
  is_synthetic_cohort: z.boolean(),
});
export type BrandModelSummary = z.infer<typeof BrandModelSummaryWireSchema>;
```

- [ ] **Step 2: Add the API client** (in `frontend/src/api/monitoring.ts`, mirroring `getKPIValue`'s `get(...)` call)

```typescript
import { BrandModelSummaryWireSchema, type BrandModelSummary } from '@/lib/api-schemas';

/** Per-brand average of the gold-standard models' holdout metrics. Pass no brand
 *  (or 'All') for the all-12-model average. */
export async function getBrandModelSummary(brand?: string): Promise<BrandModelSummary> {
  return get<BrandModelSummary>(
    '/monitoring/performance/brand-summary',
    brand && brand !== 'All' ? { brand } : undefined,
    { schema: BrandModelSummaryWireSchema }
  );
}
```

- [ ] **Step 3: Add the hook** (in `frontend/src/hooks/api/use-monitoring.ts`)

```typescript
import { getBrandModelSummary } from '@/api/monitoring';
import type { BrandModelSummary } from '@/lib/api-schemas';

export function useBrandModelSummary(
  brand?: string,
  options?: Omit<UseQueryOptions<BrandModelSummary, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: ['monitoring', 'brand-model-summary', brand ?? 'all'] as const,
    queryFn: () => getBrandModelSummary(brand),
    staleTime: 5 * 60 * 1000,
    ...options,
  });
}
```

- [ ] **Step 4: Type-check**

Run: `cd frontend && npx tsc -b 2>&1 | head -30`
Expected: no new errors referencing the changed files.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api-schemas.ts frontend/src/api/monitoring.ts frontend/src/hooks/api/use-monitoring.ts
git commit -m "feat(fe): useBrandModelSummary hook + client + schema for /performance/brand-summary" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: "Model Accuracy" tile → per-brand average accuracy

**Files:**
- Modify: `frontend/src/pages/Home.tsx` (`QuickStatTile` props ~318; tile JSX ~938; `useKPIValue` removal ~453; `isSyntheticKpis` ~600)
- Modify: `frontend/src/pages/Home.test.tsx`

- [ ] **Step 1: Write the failing test** (in `Home.test.tsx`)

Add `useBrandModelSummary` to the `use-monitoring` mock and a test:

```typescript
// extend the existing vi.mock('@/hooks/api/use-monitoring', ...) to include:
//   useBrandModelSummary: vi.fn(),
// and in resetHomeHookDefaults():
//   (useBrandModelSummary as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined, isLoading: false });

it('shows per-brand model accuracy as an average of N models', () => {
  (useBrandModelSummary as ReturnType<typeof vi.fn>).mockReturnValue({
    data: {
      brand: 'Kisqali', available: true, n_models: 4, accuracy: 0.7,
      precision: 0.66, recall: 0.55, f1: 0.6, auc_roc: 0.75, is_synthetic_cohort: true,
    },
    isLoading: false,
  });
  renderWithAllProviders(<Home />);
  expect(screen.getByText('Model Accuracy')).toBeInTheDocument();
  expect(screen.getByText('70.0%')).toBeInTheDocument();
  expect(screen.getByText('avg of 4 models')).toBeInTheDocument();
});

it('shows an honest dash when no model summary is available', () => {
  (useBrandModelSummary as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { brand: 'all', available: false, n_models: 0, is_synthetic_cohort: false },
    isLoading: false,
  });
  renderWithAllProviders(<Home />);
  const tile = screen.getByText('Model Accuracy').closest('div');
  expect(tile).toBeTruthy();
  expect(screen.queryByText(/avg of/)).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/pages/Home.test.tsx --no-file-parallelism --pool=forks -t "model accuracy"`
Expected: FAIL — `useBrandModelSummary is not defined` / no "avg of 4 models" text.

- [ ] **Step 3: Add `sublabel` to `QuickStatTile`**

Add to `QuickStatTileProps` (~line 332):

```typescript
  /** Small muted line under the value (e.g. "avg of 4 models"). */
  sublabel?: string;
```

Destructure `sublabel` and render it under the value `<div>` (after the loading/error/display block, ~line 374):

```typescript
            {sublabel && !loading && !error && (
              <div className="text-[11px] text-muted-foreground">{sublabel}</div>
            )}
```

- [ ] **Step 4: Repoint the tile + imports**

Add the import (with the other `use-monitoring` import, ~line 26):

```typescript
import { useAlerts, useBrandModelSummary } from '@/hooks/api/use-monitoring';
```

Replace the `useKPIValue('WS1-MP-001', ...)` block (~lines 453-461) with:

```typescript
  // QUICK_STATS: Model Accuracy = per-brand average of the gold-standard models'
  // holdout accuracy (labeled as an average); per-model detail lives on the
  // model-performance page. Replaces the old corpus-wide ROC-AUC (WS1-MP-001),
  // which was identical for every brand.
  const { data: modelSummary, isLoading: modelSummaryLoading } = useBrandModelSummary(
    selectedBrand !== 'All' ? selectedBrand : undefined
  );
```

Replace the tile JSX (~lines 938-948):

```typescript
                <QuickStatTile
                  label="Model Accuracy"
                  icon={<Brain className="h-4 w-4 text-rose-500" />}
                  loading={modelSummaryLoading}
                  display={
                    modelSummary?.available && modelSummary.accuracy != null
                      ? `${(modelSummary.accuracy * 100).toFixed(1)}%`
                      : '—'
                  }
                  sublabel={
                    modelSummary?.available && modelSummary.n_models
                      ? `avg of ${modelSummary.n_models} models`
                      : undefined
                  }
                />
```

Update `isSyntheticKpis` (~line 600) to drop the removed `rocAucResult` term (banner stays driven by `kpiSummary` + the batch grid):

```typescript
  const isSyntheticKpis =
    kpiSummary?.data_source === 'synthetic' ||
    (batchData?.results?.some((r) => r.data_source === 'synthetic') ?? false);
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd frontend && npx vitest run src/pages/Home.test.tsx --no-file-parallelism --pool=forks`
Expected: PASS (the two new tests; the prior `WS1-MP-001` quick-stats test must be updated to use `useBrandModelSummary` instead of `useKPIValue` for accuracy — update it in this step).

- [ ] **Step 6: Type-check + commit**

```bash
cd frontend && npx tsc -b 2>&1 | head -20
cd .. && git add frontend/src/pages/Home.tsx frontend/src/pages/Home.test.tsx
git commit -m "feat(home): Model Accuracy tile = per-brand average accuracy (avg of N models)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Dynamic KPI visibility (hide non-computed; brand-reactive; unhide brand_specific)

**Files:**
- Modify: `frontend/src/pages/Home.tsx` (`HIDDEN_HOME_WORKSTREAMS` ~174; add `computedKPIs`; `kpiCategories` ~611, `filteredKPIs` ~632, `summaryStats` ~638; grid render ~1038; "N of M" note)
- Modify: `frontend/src/pages/Home.test.tsx`

- [ ] **Step 1: Write the failing tests** (in `Home.test.tsx`, `KPI tabs & navigation (live mode)` describe)

```typescript
it('hides non-computed KPI cards (null value) from grid, tabs, and counts', () => {
  mockLiveKpis(
    [
      { id: 'WS1-MP-001', name: 'ROC-AUC', workstream: 'ws1_model_performance' },
      { id: 'WS1-MP-005', name: 'Brier Score', workstream: 'ws1_model_performance' },
      { id: 'WS2-TR-005', name: 'False Alert Rate', workstream: 'ws2_triggers' },
    ],
    { 'WS1-MP-001': 0.75 } // only this one computes; others null -> hidden
  );
  renderWithAllProviders(<Home />);
  expect(screen.getByText('ROC-AUC')).toBeInTheDocument();
  expect(screen.queryByText('Brier Score')).not.toBeInTheDocument();      // hidden
  expect(screen.queryByText('False Alert Rate')).not.toBeInTheDocument(); // hidden
  expect(screen.queryByText(/Not yet computed/)).not.toBeInTheDocument();
  const tablist = screen.getByRole('tablist');
  expect(within(tablist).queryByText('Triggers')).not.toBeInTheDocument(); // empty ws -> no tab
  expect(screen.getByText(/Showing 1 of 3 defined KPIs/)).toBeInTheDocument();
});

it('surfaces calculable brand-specific KPIs and hides the null ones', () => {
  mockLiveKpis(
    [
      { id: 'BR-002', name: 'CSU Severity Index', workstream: 'brand_specific' },
      { id: 'BR-001', name: 'AH Uncontrolled %', workstream: 'brand_specific' },
    ],
    { 'BR-002': 1.0 } // BR-002 computes; BR-001 null
  );
  renderWithAllProviders(<Home />);
  const tablist = screen.getByRole('tablist');
  expect(within(tablist).getByText('Brand')).toBeInTheDocument();      // now surfaced
  expect(screen.getByText('CSU Severity Index')).toBeInTheDocument();
  expect(screen.queryByText('AH Uncontrolled %')).not.toBeInTheDocument(); // null -> hidden
});
```

Also UPDATE the existing `does NOT render the Brand (brand_specific) tab or its KPIs` test: its premise is now reversed (brand_specific is surfaced). Replace it with the new `surfaces calculable brand-specific KPIs` test above (delete the old assertion that the Brand tab is absent).

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/pages/Home.test.tsx --no-file-parallelism --pool=forks -t "non-computed|brand-specific"`
Expected: FAIL — null cards still render as "Not yet computed"; Brand tab absent; no "Showing N of M" text.

- [ ] **Step 3: Unhide brand_specific**

Change `HIDDEN_HOME_WORKSTREAMS` (~line 174):

```typescript
// Nothing is hidden wholesale anymore: dynamic visibility (computedKPIs below)
// hides any KPI whose value did not compute, so brand-specific KPIs surface when
// they are calculable (e.g. BR-002…005) and drop out when they are not (BR-001).
const HIDDEN_HOME_WORKSTREAMS = new Set<string>([]);
```

- [ ] **Step 4: Add `computedKPIs` + a settled gate**

Immediately after the `valueByKpiId` useMemo (~line 594), add:

```typescript
  // Batch values settle asynchronously; until they do we cannot know which KPIs
  // compute, so don't flash an empty grid (or a wrong "N of M"). In demo mode
  // (samples) there is no batch, so it is always "settled".
  const batchSettled = !liveKpiMode || !!batchData || batchFailed;

  // Dynamic visibility (rule A): show ONLY KPIs whose batch value actually
  // computed (real value, no error). Null / "Not yet computed" cards are hidden
  // from the grid, the tabs, AND the counts together. Demo mode shows samples.
  const computedKPIs = useMemo(() => {
    if (!liveKpiMode) return effectiveKPIs;
    if (!batchSettled) return [];
    return effectiveKPIs.filter((kpi) => {
      const r = valueByKpiId.get(kpi.id);
      return r != null && r.value != null && !r.error;
    });
  }, [liveKpiMode, batchSettled, effectiveKPIs, valueByKpiId]);
```

- [ ] **Step 5: Drive tabs / filtered / counts from `computedKPIs`**

In `kpiCategories` (~613) replace `effectiveKPIs.map((k) => k.category)` with `computedKPIs.map((k) => k.category)` and update the dep array to `[liveKpiMode, computedKPIs]`.
In `filteredKPIs` (~632-635) replace both `effectiveKPIs` references with `computedKPIs` and update deps to `[computedKPIs, activeCategory]`.
In `summaryStats` (~638-654) replace the three `effectiveKPIs.filter(...)` and `total: effectiveKPIs.length` with `computedKPIs`, deps `[computedKPIs, kpiHealthData, kpiListData]`.

- [ ] **Step 6: Add the "N of M" note + loading/empty states around the grid**

Just above the grid `<div className="grid ...">` (~line 1038), wrap the grid:

```typescript
              {liveKpiMode && batchSettled && (
                <p className="mb-3 text-xs text-muted-foreground">
                  Showing {computedKPIs.length} of {effectiveKPIs.length} defined KPIs
                  {selectedBrand !== 'All' ? ` for ${selectedBrand}` : ''}
                </p>
              )}
              {liveKpiMode && !batchSettled ? (
                <div className="py-8 text-center text-sm text-muted-foreground">Loading KPIs…</div>
              ) : filteredKPIs.length === 0 ? (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  No computed KPIs for this selection.
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                  {filteredKPIs.map((kpi) => {
                    /* ...existing card-map body unchanged... */
                  })}
                </div>
              )}
```

(The existing card-map body stays as-is; in live mode `hasValue` is now always true because `computedKPIs` pre-filters, so the "Not yet computed" branch becomes dead for live mode — harmless, and still correct for any edge.)

- [ ] **Step 7: Run to verify it passes**

Run: `cd frontend && npx vitest run src/pages/Home.test.tsx --no-file-parallelism --pool=forks`
Expected: PASS (all Home tests, including the two new + the rewritten brand_specific test).

- [ ] **Step 8: Type-check + lint + commit**

```bash
cd frontend && npx tsc -b 2>&1 | head -20 && npx eslint src/pages/Home.tsx 2>&1 | head -20
cd .. && git add frontend/src/pages/Home.tsx frontend/src/pages/Home.test.tsx
git commit -m "feat(home): dynamic KPI visibility — hide non-computed cards, brand-reactive grid/tabs/counts, N-of-M note" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification (after all tasks)

- [ ] Backend: `python3 -m pytest tests/unit/test_kpi/test_goldstd_model_perf.py tests/unit/test_kpi/test_model_performance.py tests/unit/test_services/test_brand_goldstd_summary.py tests/unit/test_api/test_brand_performance_summary.py -q`
- [ ] Frontend: `cd frontend && npx vitest run src/pages/Home.test.tsx --no-file-parallelism --pool=forks` and `npx tsc -b`
- [ ] Targeted mypy on changed files only (NOT whole-tree — see CLAUDE.md droplet policy): `mypy src/kpi/goldstd_model_perf.py src/kpi/calculators/model_performance.py src/services/performance_tracking.py src/api/routes/monitoring.py`
- [ ] Dispatch a final code-review subagent over the whole branch diff, then `superpowers:finishing-a-development-branch`.

## Notes / out of scope

- No DB migration: reuses the `PerformanceTracker` service-layer direct-read pattern (same as confusion/ROC). The KPI value cache key already includes `brand` (`src/kpi/cache.py:_make_key`), so per-brand WS1-MP-001/003 cache correctly.
- Out of scope (tracked): #1064 (Group 4 / #577 synthetic-exclusion); synthetic-inclusive twins for other excluded KPIs; per-brand PR-AUC/Brier/Recall@K/Calibration/Fairness (no gold-standard source).
