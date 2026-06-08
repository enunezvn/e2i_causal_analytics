"""R5 outcome feed — pure aggregation logic for ExperimentOutcomeRepository.

These tests exercise the REAL aggregation logic (no mocks) over real-shaped
business_metrics ``per_hcp_rollup`` rows: map a primary_metric to its typed
column, collapse multiple metric_date rows per HCP to one scalar (SUM for
counts, MEAN for rates), then split per-unit values by assignment variant into
the (control, treatment) arrays that ResultsAnalysisService._compute_results
consumes. DB I/O is integration-tested separately (gated, real Supabase).
"""

from __future__ import annotations

import numpy as np
import pytest


def _repo():
    from src.repositories.experiment_outcome import ExperimentOutcomeRepository

    # resolve_column/aggregate_to_arrays are @staticmethod — exercise them on the
    # CLASS so no Supabase client is resolved (key-less CI would otherwise raise
    # ServiceConnectionError at construction).
    return ExperimentOutcomeRepository


class TestResolveColumn:
    def test_count_metrics_map_to_sum(self):
        repo = _repo()
        for metric in ("trx", "trx_count", "nrx", "nrx_count", "total_rx", "total_rx_count"):
            column, reducer = repo.resolve_column(metric)
            assert reducer == "sum", f"{metric} should sum"
            assert column in {"trx_count", "nrx_count", "total_rx_count"}

    def test_rate_metrics_map_to_mean(self):
        repo = _repo()
        for metric in ("market_share", "conversion_rate", "engagement_score", "call_frequency"):
            column, reducer = repo.resolve_column(metric)
            assert reducer == "mean", f"{metric} should mean"
            assert column == metric

    def test_unknown_metric_fails_closed(self):
        repo = _repo()
        with pytest.raises(ValueError):
            repo.resolve_column("adoption_propensity")


class TestAggregateToArrays:
    def test_sums_count_values_per_hcp_then_splits_by_variant(self):
        repo = _repo()
        assignments = [
            ("HCP_1", "control"),
            ("HCP_2", "control"),
            ("HCP_3", "treatment"),
        ]
        # Two date rows for HCP_1 -> summed to 3; HCP_2 -> 5; HCP_3 -> 9.
        rows = [
            {"hcp_id": "HCP_1", "trx_count": 1},
            {"hcp_id": "HCP_1", "trx_count": 2},
            {"hcp_id": "HCP_2", "trx_count": 5},
            {"hcp_id": "HCP_3", "trx_count": 4},
            {"hcp_id": "HCP_3", "trx_count": 5},
        ]
        control, treatment = repo.aggregate_to_arrays(
            assignments, rows, column="trx_count", reducer="sum"
        )
        assert sorted(control.tolist()) == [3.0, 5.0]
        assert treatment.tolist() == [9.0]

    def test_means_rate_values_per_hcp(self):
        repo = _repo()
        assignments = [("HCP_1", "control"), ("HCP_2", "treatment")]
        rows = [
            {"hcp_id": "HCP_1", "market_share": 0.2},
            {"hcp_id": "HCP_1", "market_share": 0.4},  # mean 0.3
            {"hcp_id": "HCP_2", "market_share": 0.8},
        ]
        control, treatment = repo.aggregate_to_arrays(
            assignments, rows, column="market_share", reducer="mean"
        )
        assert control.tolist() == pytest.approx([0.3])
        assert treatment.tolist() == pytest.approx([0.8])

    def test_skips_null_outcome_values(self):
        repo = _repo()
        assignments = [("HCP_1", "control"), ("HCP_2", "treatment")]
        # HCP_2 has only NULLs -> excluded entirely (no NaN in the array).
        rows = [
            {"hcp_id": "HCP_1", "trx_count": 7},
            {"hcp_id": "HCP_2", "trx_count": None},
        ]
        control, treatment = repo.aggregate_to_arrays(
            assignments, rows, column="trx_count", reducer="sum"
        )
        assert control.tolist() == [7.0]
        assert treatment.size == 0
        assert not np.isnan(control).any()

    def test_empty_when_no_assignments(self):
        repo = _repo()
        control, treatment = repo.aggregate_to_arrays(
            [], [{"hcp_id": "HCP_1", "trx_count": 1}], column="trx_count", reducer="sum"
        )
        assert control.size == 0 and treatment.size == 0

    def test_unassigned_hcp_metrics_ignored(self):
        repo = _repo()
        assignments = [("HCP_1", "control"), ("HCP_2", "treatment")]
        rows = [
            {"hcp_id": "HCP_1", "trx_count": 1},
            {"hcp_id": "HCP_2", "trx_count": 2},
            {"hcp_id": "HCP_999", "trx_count": 100},  # not assigned -> ignored
        ]
        control, treatment = repo.aggregate_to_arrays(
            assignments, rows, column="trx_count", reducer="sum"
        )
        assert control.tolist() == [1.0]
        assert treatment.tolist() == [2.0]
