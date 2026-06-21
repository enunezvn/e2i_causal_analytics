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
        {
            "model_id": "2",
            "metric_name": "auc_roc",
            "metric_value": 0.9,
            "source": "backtest_wf",
        },  # wrong source, ignored
    ]
    summary = average_holdout(models, rows)
    assert summary["n_models"] == 2
    assert summary["accuracy"] == 0.7  # (0.6+0.8)/2
    assert summary["f1"] == 0.4  # single value
    assert summary["auc_roc"] is None  # only a backtest_wf row -> not counted


def test_average_holdout_none_when_no_models():
    assert average_holdout([], []) is None


def test_average_holdout_includes_new_scalar_extras():
    models = [{"id": "1"}, {"id": "2"}]
    rows = [
        {"model_id": "1", "metric_name": "pr_auc", "metric_value": 0.5, "source": "holdout"},
        {"model_id": "2", "metric_name": "pr_auc", "metric_value": 0.7, "source": "holdout"},
        {"model_id": "1", "metric_name": "brier_score", "metric_value": 0.2, "source": "holdout"},
        {"model_id": "1", "metric_name": "calibration_slope", "metric_value": 0.9, "source": "holdout"},
    ]
    summary = average_holdout(models, rows)
    assert summary["pr_auc"] == 0.6  # (0.5 + 0.7) / 2
    assert summary["brier_score"] == 0.2
    assert summary["calibration_slope"] == 0.9


def test_goldstd_metrics_constant_is_the_verified_set():
    assert set(GOLDSTD_METRICS) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "pr_auc",
        "brier_score",
        "calibration_slope",
    }
