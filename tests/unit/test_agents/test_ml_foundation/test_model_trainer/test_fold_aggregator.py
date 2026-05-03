"""Tests for :mod:`src.agents.ml_foundation.model_trainer.aggregation.fold_aggregator`.

Phase 1 W3-lite Day-5 (shard 21 §D). Covers:
- AggregateStat shape (mean / std / n_folds / percentile CI / BCa CI / raw_values).
- aggregate_fold_metrics over a list of per-fold dicts (auto-flatten + skip-failed).
- BCa unstable_warning passthrough.
- Cycle-15 I-3 partial-fold contract: fold_status='failed' rows are excluded.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.aggregation import (
    AggregateStat,
    aggregate_fold_metrics,
    flatten_fold_record,
)


def _build_fold_records(
    auc_values: List[float],
    *,
    brier_values: List[float] | None = None,
    fold_status: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Build a list of per-fold records mirroring the orchestrator shape."""
    n = len(auc_values)
    if brier_values is None:
        brier_values = [0.20 - 0.01 * i for i in range(n)]
    if fold_status is None:
        fold_status = ["ok"] * n
    records: List[Dict[str, Any]] = []
    for i, (auc, brier, status) in enumerate(
        zip(auc_values, brier_values, fold_status, strict=True)
    ):
        records.append(
            {
                "fold_idx": i,
                "fold_random_state": 1000 + i,
                "fold_status": status,
                "auc_roc": auc,
                "brier_score": brier,
                "test_metrics": {"accuracy": 0.80 + 0.01 * i, "precision": 0.75},
                "validation_metrics": {"accuracy": 0.78 + 0.01 * i},
                "train_metrics": {"accuracy": 0.85 + 0.005 * i},
            }
        )
    return records


class TestFlattenFoldRecord:
    def test_flattens_top_level_scalars_and_nested_metrics(self) -> None:
        record = {
            "fold_idx": 3,
            "fold_random_state": 1234,
            "auc_roc": 0.85,
            "brier_score": 0.12,
            "test_metrics": {"accuracy": 0.82, "precision": 0.79},
            "validation_metrics": {"accuracy": 0.80},
            "train_metrics": {"accuracy": 0.88},
            "mlflow_run_id": "run-42",
            "fold_status": "ok",
        }
        flat = flatten_fold_record(record)
        assert flat == {
            "auc_roc": 0.85,
            "brier_score": 0.12,
            "test_accuracy": 0.82,
            "test_precision": 0.79,
            "validation_accuracy": 0.80,
            "train_accuracy": 0.88,
        }

    def test_skips_bookkeeping_fields(self) -> None:
        record = {"fold_idx": 0, "fold_random_state": 7, "fold_status": "ok", "auc_roc": 0.9}
        flat = flatten_fold_record(record)
        assert "fold_idx" not in flat
        assert "fold_random_state" not in flat
        assert "fold_status" not in flat
        assert flat["auc_roc"] == 0.9

    def test_skips_non_numeric_and_non_finite(self) -> None:
        record = {
            "auc_roc": float("nan"),
            "brier_score": float("inf"),
            "label": "lightgbm",
            "test_metrics": {"accuracy": 0.8, "name": "lgb"},
        }
        flat = flatten_fold_record(record)
        assert "auc_roc" not in flat
        assert "brier_score" not in flat
        assert "label" not in flat
        assert flat == {"test_accuracy": 0.8}

    def test_skips_bool_values(self) -> None:
        record = {"is_ok": True, "test_metrics": {"flag": False, "score": 0.5}}
        flat = flatten_fold_record(record)
        assert "is_ok" not in flat
        assert "test_flag" not in flat
        assert flat["test_score"] == 0.5


class TestAggregateFoldMetrics:
    def test_returns_empty_dict_for_empty_input(self) -> None:
        assert aggregate_fold_metrics([]) == {}

    def test_aggregate_stat_shape_for_k10(self) -> None:
        records = _build_fold_records([0.80 + 0.01 * i for i in range(10)])
        agg = aggregate_fold_metrics(records, bca_n_resamples=200, bca_rng_seed=42)
        assert "auc_roc" in agg
        stat = agg["auc_roc"]
        assert isinstance(stat, AggregateStat)
        assert stat.n_folds == 10
        assert stat.mean == pytest.approx(0.845, abs=1e-9)
        assert stat.std == pytest.approx(
            np.std([0.80 + 0.01 * i for i in range(10)], ddof=1), abs=1e-9
        )
        assert stat.percentile_ci_lo <= stat.mean <= stat.percentile_ci_hi
        assert len(stat.raw_values) == 10
        assert stat.bca_ci_lo is not None
        assert stat.bca_ci_hi is not None

    def test_aggregate_means_match_per_fold_means(self) -> None:
        # Shard 21 G.3 contract — fixture for the integration-level test
        auc_values = [0.80 + 0.013 * i for i in range(10)]
        records = _build_fold_records(auc_values)
        agg = aggregate_fold_metrics(records, bca_n_resamples=100, bca_rng_seed=42)
        assert agg["auc_roc"].mean == pytest.approx(np.mean(auc_values), abs=1e-9)

    def test_skips_failed_folds_per_cycle15_i3(self) -> None:
        # 4 ok + 6 failed → n_folds=4, BCa lo/hi None (n=4 OK, but values uniform → maybe stable;
        # what matters is that failed folds are NOT included in the array).
        auc_values = [0.80, 0.85, 0.82, 0.78, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        statuses = ["ok"] * 4 + ["failed"] * 6
        records = _build_fold_records(auc_values, fold_status=statuses)
        agg = aggregate_fold_metrics(records, bca_n_resamples=100, bca_rng_seed=42)
        # 4 ok folds → arr = [0.80, 0.85, 0.82, 0.78]; mean should NOT include 0.5s
        assert agg["auc_roc"].n_folds == 4
        assert agg["auc_roc"].mean == pytest.approx(np.mean([0.80, 0.85, 0.82, 0.78]), abs=1e-9)

    def test_returns_empty_when_all_folds_failed(self) -> None:
        records = _build_fold_records([0.5] * 5, fold_status=["failed"] * 5)
        assert aggregate_fold_metrics(records) == {}

    def test_bca_ci_brackets_mean_for_well_conditioned_data(self) -> None:
        rng = np.random.default_rng(7)
        auc_values = list(rng.normal(loc=0.85, scale=0.02, size=10))
        records = _build_fold_records(auc_values)
        agg = aggregate_fold_metrics(records, bca_n_resamples=500, bca_rng_seed=42)
        stat = agg["auc_roc"]
        assert stat.bca_ci_lo is not None and stat.bca_ci_lo <= stat.mean
        assert stat.bca_ci_hi is not None and stat.bca_ci_hi >= stat.mean

    def test_bca_skipped_below_4_folds(self) -> None:
        # Cycle-15 I-3 boundary — k=3 should yield BCa endpoints None (helper enforces min_samples=4)
        records = _build_fold_records([0.80, 0.82, 0.85])
        agg = aggregate_fold_metrics(records, bca_n_resamples=200, bca_rng_seed=42)
        assert agg["auc_roc"].n_folds == 3
        assert agg["auc_roc"].bca_ci_lo is None
        assert agg["auc_roc"].bca_ci_hi is None
        assert agg["auc_roc"].bca_unstable_warning is True

    def test_aggregates_top_level_and_nested_metrics(self) -> None:
        records = _build_fold_records([0.80 + 0.01 * i for i in range(10)])
        agg = aggregate_fold_metrics(records, bca_n_resamples=100, bca_rng_seed=42)
        # Top-level
        assert "auc_roc" in agg
        assert "brier_score" in agg
        # Nested test_/validation_/train_ metrics surface with prefixed keys
        assert "test_accuracy" in agg
        assert "test_precision" in agg
        assert "validation_accuracy" in agg
        assert "train_accuracy" in agg

    def test_explicit_metrics_arg_restricts_output(self) -> None:
        records = _build_fold_records([0.80 + 0.01 * i for i in range(10)])
        agg = aggregate_fold_metrics(
            records,
            metrics=["auc_roc"],
            bca_n_resamples=100,
            bca_rng_seed=42,
        )
        assert set(agg.keys()) == {"auc_roc"}

    def test_handles_single_fold_gracefully(self) -> None:
        records = _build_fold_records([0.85])
        agg = aggregate_fold_metrics(records, bca_n_resamples=50, bca_rng_seed=42)
        stat = agg["auc_roc"]
        assert stat.n_folds == 1
        assert stat.std == 0.0
        assert stat.percentile_ci_lo == stat.mean == stat.percentile_ci_hi
        assert stat.bca_ci_lo is None  # below min_samples=4

    def test_raw_values_preserved_in_order(self) -> None:
        auc_values = [0.80, 0.83, 0.79, 0.85, 0.82, 0.81, 0.84, 0.78, 0.86, 0.80]
        records = _build_fold_records(auc_values)
        agg = aggregate_fold_metrics(records, bca_n_resamples=50, bca_rng_seed=42)
        assert list(agg["auc_roc"].raw_values) == auc_values

    def test_metric_present_in_only_some_folds_aggregates_only_present(self) -> None:
        records = _build_fold_records([0.80] * 10)
        # Inject a metric that only appears in some folds
        records[0]["custom_metric"] = 0.5
        records[1]["custom_metric"] = 0.6
        records[5]["custom_metric"] = 0.55
        agg = aggregate_fold_metrics(records, bca_n_resamples=50, bca_rng_seed=42)
        assert "custom_metric" in agg
        assert agg["custom_metric"].n_folds == 3  # only 3 folds had the metric
