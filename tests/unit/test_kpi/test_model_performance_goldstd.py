"""Per-brand gold-standard sourcing for WS1-MP-001 (ROC-AUC) and WS1-MP-003 (F1).

These KPIs prefer the per-brand average of the gold-standard models' holdout
metrics (brand-reactive), falling back to the existing corpus SQL / MLflow legs
when no gold-standard data is available.
"""

from src.kpi.calculators.model_performance import ModelPerformanceCalculator
from src.kpi.models import CalculationType, KPIMetadata, KPIThreshold, Workstream


class _Resp:
    def __init__(self, data):
        self.data = data


class _SyncQuery:
    """Real (non-Mock) sync PostgREST builder stub returning canned table data."""

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


def _kpi(kpi_id, name):
    return KPIMetadata(
        id=kpi_id,
        name=name,
        definition=name,
        formula="f",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(target=0.80, warning=0.70, critical=0.60),
    )


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
    result = calc.calculate(_kpi("WS1-MP-001", "ROC-AUC"), {"brand": "Kisqali"})
    assert result.value == 0.72  # (0.68 + 0.76) / 2
    assert result.error is None


def test_f1_uses_per_brand_goldstd_average():
    store = {
        "ml_model_registry": [{"id": "1", "model_name": "persistence_fabhalta_goldstd_lr_v1"}],
        "ml_performance_metrics": [
            {"model_id": "1", "metric_name": "f1", "metric_value": 0.69, "source": "holdout"},
        ],
    }
    calc = ModelPerformanceCalculator(db_client=_SyncClient(store))
    result = calc.calculate(_kpi("WS1-MP-003", "F1"), {"brand": "Fabhalta"})
    assert result.value == 0.69


def test_pr_auc_uses_per_brand_goldstd_average():
    store = {
        "ml_model_registry": [{"id": "1", "model_name": "discontinuation_kisqali_goldstd_lr_v1"}],
        "ml_performance_metrics": [
            {"model_id": "1", "metric_name": "pr_auc", "metric_value": 0.62, "source": "holdout"},
        ],
    }
    calc = ModelPerformanceCalculator(db_client=_SyncClient(store))
    result = calc.calculate(_kpi("WS1-MP-002", "PR-AUC"), {"brand": "Kisqali"})
    assert result.value == 0.62
    assert result.error is None


def test_brier_uses_per_brand_goldstd_average():
    store = {
        "ml_model_registry": [{"id": "1", "model_name": "initiation_remibrutinib_goldstd_lr_v1"}],
        "ml_performance_metrics": [
            {
                "model_id": "1",
                "metric_name": "brier_score",
                "metric_value": 0.18,
                "source": "holdout",
            },
        ],
    }
    calc = ModelPerformanceCalculator(db_client=_SyncClient(store))
    result = calc.calculate(_kpi("WS1-MP-005", "Brier"), {"brand": "Remibrutinib"})
    assert result.value == 0.18


def test_calibration_uses_per_brand_goldstd_average():
    store = {
        "ml_model_registry": [{"id": "1", "model_name": "hcp_adoption_fabhalta_goldstd_lr_v1"}],
        "ml_performance_metrics": [
            {
                "model_id": "1",
                "metric_name": "calibration_slope",
                "metric_value": 0.95,
                "source": "holdout",
            },
        ],
    }
    calc = ModelPerformanceCalculator(db_client=_SyncClient(store))
    result = calc.calculate(_kpi("WS1-MP-006", "Calibration"), {"brand": "Fabhalta"})
    assert result.value == 0.95


def test_roc_auc_falls_back_when_no_goldstd(monkeypatch):
    # Empty registry -> gold-standard returns None -> falls back to the existing
    # corpus SQL leg (stubbed empty) -> MLflow fail-closed (stubbed not-found).
    calc = ModelPerformanceCalculator(db_client=_SyncClient({}))
    monkeypatch.setattr(calc, "_execute_query", lambda *_a, **_k: ([], None))
    monkeypatch.setattr(
        calc, "_get_metric_from_mlflow", lambda *_a, **_k: (None, "model_not_found:default_model")
    )
    result = calc.calculate(_kpi("WS1-MP-001", "ROC-AUC"), {"brand": "Kisqali"})
    assert result.value is None
    assert "model_not_found" in (result.error or "")
