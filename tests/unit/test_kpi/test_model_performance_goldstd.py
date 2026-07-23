"""Per-brand gold-standard sourcing for WS1-MP-001 (ROC-AUC) and WS1-MP-003 (F1).

These KPIs prefer the per-brand average of the gold-standard models' holdout
metrics (brand-reactive), falling back to the existing corpus SQL / MLflow legs
when no gold-standard data is available.
"""

import pytest

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


def _band_kpi() -> KPIMetadata:
    """WS1-MP-006 with the real (unchanged) band threshold from kpi_definitions.yaml."""
    return KPIMetadata(
        id="WS1-MP-006",
        name="Calibration Slope Deviation",
        definition="Calibration Slope Deviation",
        formula="f",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(ideal=1.0, good_tolerance=0.05, warning_tolerance=0.15),
    )


def test_calibration_uses_per_brand_goldstd_aggregate():
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
    result = calc.calculate(_band_kpi(), {"brand": "Fabhalta"})
    # B3: headline = 1 + mean(|slope - 1|) -> a single 0.95 reads 1.05.
    assert result.value == pytest.approx(1.05)
    assert result.status == "good"  # deviation 0.05 <= good_tolerance


def test_calibration_signed_cancellation_is_dead():
    """0.70 & 1.30 must surface as 1.30 (CRITICAL), not signed-mean 1.00 (GOOD)."""
    store = {
        "ml_model_registry": [
            {"id": "1", "model_name": "initiation_kisqali_goldstd_lr_v1"},
            {"id": "2", "model_name": "persistence_kisqali_goldstd_lr_v1"},
        ],
        "ml_performance_metrics": [
            {
                "model_id": "1",
                "metric_name": "calibration_slope",
                "metric_value": 0.70,
                "source": "holdout",
            },
            {
                "model_id": "2",
                "metric_name": "calibration_slope",
                "metric_value": 1.30,
                "source": "holdout",
            },
        ],
    }
    calc = ModelPerformanceCalculator(db_client=_SyncClient(store))
    result = calc.calculate(_band_kpi(), {"brand": "Kisqali"})
    assert result.value == pytest.approx(1.30)
    assert result.status == "critical"


def test_calibration_detail_surfaced_in_result_metadata():
    """B2 payload: per-model slopes + holdout n + CI ride in KPIResult.metadata
    so a wide-CI red is visibly a small-sample red."""
    store = {
        "ml_model_registry": [
            {"id": "1", "model_name": "persistence_remibrutinib_goldstd_lr_v1"},
        ],
        "ml_performance_metrics": [
            {
                "model_id": "1",
                "metric_name": "calibration_slope",
                "metric_value": 1.4455,
                "source": "holdout",
                "sample_size": 415,
                "ci_lower": 1.2192,
                "ci_upper": 1.6704,
            },
        ],
    }
    calc = ModelPerformanceCalculator(db_client=_SyncClient(store))
    result = calc.calculate(_band_kpi(), {"brand": "Remibrutinib"})
    detail = result.metadata["calibration_slope_detail"]
    assert detail["aggregation"] == "one_plus_mean_abs_deviation"
    entry = detail["models"][0]
    assert entry["model_name"] == "persistence_remibrutinib_goldstd_lr_v1"
    assert entry["slope"] == pytest.approx(1.4455)
    assert entry["n"] == 415
    assert entry["ci_lower"] == pytest.approx(1.2192)
    assert entry["ci_upper"] == pytest.approx(1.6704)


def test_calibration_detail_null_ci_rows_degrade_gracefully():
    """Pre-next-eval DB state: slope rows with NULL ci/sample_size still produce
    a detail entry with None fields (no crash, no fabricated interval)."""
    store = {
        "ml_model_registry": [{"id": "1", "model_name": "initiation_kisqali_goldstd_lr_v1"}],
        "ml_performance_metrics": [
            {
                "model_id": "1",
                "metric_name": "calibration_slope",
                "metric_value": 0.97,
                "source": "holdout",
                "sample_size": None,
                "ci_lower": None,
                "ci_upper": None,
            },
        ],
    }
    calc = ModelPerformanceCalculator(db_client=_SyncClient(store))
    result = calc.calculate(_band_kpi(), {"brand": "Kisqali"})
    assert result.value == pytest.approx(1.03)
    entry = result.metadata["calibration_slope_detail"]["models"][0]
    assert entry["n"] is None
    assert entry["ci_lower"] is None
    assert entry["ci_upper"] is None


def test_ws1_mp006_named_for_deviation_semantics():
    """The headline is 1 + mean(|slope - 1|), NOT a literal slope — the KPI's
    display name/definition/formula must say so (codex HIGH: a consumer reading
    'Calibration Slope' would misread the folded value as a slope). The metric
    storage key (`calibration_slope`) is unchanged — this is labeling only."""
    from src.kpi.registry import KPIRegistry

    KPIRegistry.reset()
    try:
        kpi = KPIRegistry().get("WS1-MP-006")
        assert kpi is not None
        assert kpi.name == "Calibration Slope Deviation"
        assert "1 + mean(|slope - 1|)" in kpi.definition
        # The definition must point readers at the per-model TRUE slopes.
        assert "detail" in kpi.definition
        assert "1 + mean(" in kpi.formula
    finally:
        KPIRegistry.reset()


def test_ws1_mp006_band_in_kpi_definitions_unchanged():
    """B3 keeps the headline in slope-band units — the YAML stays in BAND mode.

    The exact good_tolerance value is owned by test_ws1_frontier_thresholds.py
    (0.05 -> 0.10 on 2026-07-23: the folded headline's sampling-noise floor at
    the current holdout sizes is ~1.08 even under perfect calibration); this
    test pins the band STRUCTURE the fold depends on.
    """
    from src.kpi.registry import KPIRegistry

    KPIRegistry.reset()
    try:
        kpi = KPIRegistry().get("WS1-MP-006")
        assert kpi is not None and kpi.threshold is not None
        assert kpi.threshold.ideal == 1.0
        assert kpi.threshold.good_tolerance == 0.10
        assert kpi.threshold.warning_tolerance == 0.15
        assert kpi.threshold.target is None
    finally:
        KPIRegistry.reset()


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
