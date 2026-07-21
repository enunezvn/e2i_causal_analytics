import pytest

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
        {
            "model_id": "1",
            "metric_name": "calibration_slope",
            "metric_value": 0.9,
            "source": "holdout",
        },
    ]
    summary = average_holdout(models, rows)
    assert summary["pr_auc"] == 0.6  # (0.5 + 0.7) / 2
    assert summary["brier_score"] == 0.2
    # B3: calibration_slope aggregates as 1 + mean(|slope - 1|), so a single
    # 0.9 reads 1.1 — same distance from ideal, direction folded away.
    assert summary["calibration_slope"] == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# B3 — calibration_slope aggregation semantics: 1 + mean(|slope_i - 1|)
# ---------------------------------------------------------------------------


def test_average_holdout_calibration_slope_kills_signed_cancellation():
    """0.70 & 1.30 must aggregate to 1.30 (red), NOT signed-mean 1.00 (green)."""
    models = [
        {"id": "1", "model_name": "a_goldstd_lr_v1"},
        {"id": "2", "model_name": "b_goldstd_lr_v1"},
    ]
    rows = [
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
    ]
    summary = average_holdout(models, rows)
    assert summary["calibration_slope"] == pytest.approx(1.30)


def test_average_holdout_other_metrics_stay_signed_means():
    """The abs-deviation fold applies to calibration_slope ONLY."""
    models = [{"id": "1"}, {"id": "2"}]
    rows = [
        {"model_id": "1", "metric_name": "auc_roc", "metric_value": 0.60, "source": "holdout"},
        {"model_id": "2", "metric_name": "auc_roc", "metric_value": 0.80, "source": "holdout"},
        {"model_id": "1", "metric_name": "brier_score", "metric_value": 0.10, "source": "holdout"},
        {"model_id": "2", "metric_name": "brier_score", "metric_value": 0.30, "source": "holdout"},
    ]
    summary = average_holdout(models, rows)
    assert summary["auc_roc"] == pytest.approx(0.70)
    assert summary["brier_score"] == pytest.approx(0.20)


def test_average_holdout_realistic_remi_panel():
    """Live-shaped fixture: mirror pair 1.4455/1.4452 + initiation pair ~0.97/0.99.

    Headline = 1 + (0.4455 + 0.4452 + 0.03 + 0.01) / 4 = 1.232675 — still red
    under the unchanged WS1-MP-006 band (deviation 0.2327 > 0.15).
    """
    models = [
        {"id": "1", "model_name": "persistence_remibrutinib_goldstd_lr_v1"},
        {"id": "2", "model_name": "discontinuation_remibrutinib_goldstd_lr_v1"},
        {"id": "3", "model_name": "initiation_remibrutinib_goldstd_lr_v1"},
        {"id": "4", "model_name": "hcp_adoption_remibrutinib_goldstd_lr_v1"},
    ]
    rows = [
        {
            "model_id": mid,
            "metric_name": "calibration_slope",
            "metric_value": v,
            "source": "holdout",
        }
        for mid, v in (("1", 1.4455), ("2", 1.4452), ("3", 0.97), ("4", 0.99))
    ]
    summary = average_holdout(models, rows)
    assert summary["calibration_slope"] == pytest.approx(1.232675)


# ---------------------------------------------------------------------------
# B2 — per-model detail payload: slopes + holdout n + bootstrap CI
# ---------------------------------------------------------------------------


def test_average_holdout_slope_detail_carries_n_and_ci():
    models = [
        {"id": "1", "model_name": "persistence_remibrutinib_goldstd_lr_v1"},
        {"id": "2", "model_name": "discontinuation_remibrutinib_goldstd_lr_v1"},
    ]
    rows = [
        {
            "model_id": "1",
            "metric_name": "calibration_slope",
            "metric_value": 1.4455,
            "source": "holdout",
            "sample_size": 415,
            "ci_lower": 1.2192,
            "ci_upper": 1.6704,
        },
        {
            "model_id": "2",
            "metric_name": "calibration_slope",
            "metric_value": 1.4452,
            "source": "holdout",
            "sample_size": 415,
            "ci_lower": 1.2189,
            "ci_upper": 1.6701,
        },
    ]
    summary = average_holdout(models, rows)
    detail = summary["calibration_slope_detail"]
    assert detail["aggregation"] == "one_plus_mean_abs_deviation"
    by_name = {m["model_name"]: m for m in detail["models"]}
    entry = by_name["persistence_remibrutinib_goldstd_lr_v1"]
    assert entry["slope"] == pytest.approx(1.4455)
    assert entry["n"] == 415
    assert entry["ci_lower"] == pytest.approx(1.2192)
    assert entry["ci_upper"] == pytest.approx(1.6704)
    # Deterministic ordering (by model_name) so the payload is stable.
    names = [m["model_name"] for m in detail["models"]]
    assert names == sorted(names)


def test_average_holdout_slope_detail_null_ci_degrades_gracefully():
    """Pre-B2 rows (no ci/sample_size written yet) still produce entries —
    with None fields, never a fabricated interval."""
    models = [{"id": "1", "model_name": "initiation_kisqali_goldstd_lr_v1"}]
    rows = [
        {
            "model_id": "1",
            "metric_name": "calibration_slope",
            "metric_value": 0.97,
            "source": "holdout",
        },
    ]
    summary = average_holdout(models, rows)
    assert summary["calibration_slope"] == pytest.approx(1.03)
    entry = summary["calibration_slope_detail"]["models"][0]
    assert entry["slope"] == pytest.approx(0.97)
    assert entry["n"] is None
    assert entry["ci_lower"] is None
    assert entry["ci_upper"] is None


def test_average_holdout_no_slope_rows_omits_detail():
    models = [{"id": "1"}]
    rows = [
        {"model_id": "1", "metric_name": "auc_roc", "metric_value": 0.7, "source": "holdout"},
    ]
    summary = average_holdout(models, rows)
    assert summary["calibration_slope"] is None
    assert "calibration_slope_detail" not in summary


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
