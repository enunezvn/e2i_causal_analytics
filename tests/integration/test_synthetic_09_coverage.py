"""Shard 09 Task 8: faithful coverage gate (docker Supabase).

Asserts that after the synthetic substrate is loaded (scripts/load_synthetic_data.py
--small --anchor-to-now), the view-backed KPIs read non-zero over now()-30d, the
empty MLOps/AB tables carry synthetic rows, and the data_lag / causal_paths / NRx /
CFR substrate is present. Gated E2I_DB_INTEGRATION=1; run -n0.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful docker-Supabase test; set E2I_DB_INTEGRATION=1",
)


def _count(sql: str) -> int:
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tAc", sql],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip() or "0")


def test_view_kpis_nonzero_after_load():
    assert (
        _count(
            "SELECT count(distinct user_id) FROM user_sessions "
            "WHERE is_synthetic AND session_start >= now()-interval '30 days'"
        )
        > 0
    )
    assert (
        _count(
            "SELECT count(*) FROM etl_pipeline_metrics "
            "WHERE is_synthetic AND run_end >= now()-interval '30 days'"
        )
        > 0
    )
    assert (
        _count(
            "SELECT count(*) FROM ml_annotations "
            "WHERE is_synthetic AND iaa_group_id IS NOT NULL "
            "AND annotation_timestamp >= now()-interval '30 days'"
        )
        > 0
    )
    assert (
        _count(
            "SELECT count(*) FROM hcp_intent_surveys "
            "WHERE is_synthetic AND intent_to_prescribe_change IS NOT NULL "
            "AND survey_date >= (now()-interval '30 days')::date"
        )
        > 0
    )
    assert (
        _count(
            "SELECT count(*) FROM data_source_tracking "
            "WHERE is_synthetic AND tracking_date >= (now()-interval '30 days')::date"
        )
        > 0
    )


def test_mlops_and_ab_substrate_present():
    assert _count("SELECT count(*) FROM ml_model_registry WHERE is_synthetic") >= 6
    assert _count("SELECT count(*) FROM ml_deployments WHERE is_synthetic AND status='active'") >= 3
    assert (
        _count(
            "SELECT count(*) FROM ab_experiment_results "
            "WHERE is_synthetic AND (treatment_mean-control_mean) > 0"
        )
        > 0
    )
    assert _count("SELECT count(*) FROM ml_experiments WHERE is_synthetic AND status='running'") > 0
    assert (
        _count("SELECT count(*) FROM learning_signals WHERE is_synthetic AND is_training_example")
        >= 10
    )


def test_data_lag_and_causal_paths_and_nrx_and_cfr_produced():
    # WS1-DQ-007 data lag + CM-003/CM-005 causal_paths
    assert (
        _count(
            "SELECT count(*) FROM patient_journeys WHERE is_synthetic AND data_lag_hours IS NOT NULL"
        )
        > 0
    )
    assert (
        _count(
            "SELECT count(*) FROM causal_paths WHERE is_synthetic AND causal_effect_size IS NOT NULL"
        )
        > 0
    )
    assert (
        _count(
            "SELECT count(*) FROM causal_paths "
            "WHERE is_synthetic AND array_length(mediators_identified, 1) >= 1"
        )
        > 0
    )
    # WS3-BI-006 NRx: synthetic prescriptions with sequence_number=1
    assert (
        _count(
            "SELECT count(*) FROM treatment_events "
            "WHERE is_synthetic AND event_type::text='prescription' AND sequence_number = 1"
        )
        > 0
    )
    # WS2-TR-008 CFR: synthetic triggers with a supersession chain
    assert (
        _count(
            "SELECT count(*) FROM triggers WHERE is_synthetic AND previous_trigger_id IS NOT NULL"
        )
        > 0
    )


def test_model_metrics_stamped_on_synthetic_predictions():
    # WS1-MP-002..008 + CM-004: model-quality metrics on the synthetic frame
    for col in (
        "model_auc",
        "model_pr_auc",
        "brier_score",
        "calibration_score",
        "shap_values",
        "fairness_metrics",
        "counterfactual_outcome",
    ):
        assert (
            _count(f"SELECT count(*) FROM ml_predictions WHERE is_synthetic AND {col} IS NOT NULL")
            > 0
        ), f"{col} not stamped on synthetic ml_predictions"


def test_all_45_kpis_return_nonnull():
    """Run the coverage probe and assert ZERO EMPTY / ZERO N/A across the 45 KPIs."""
    out = subprocess.run(
        [sys.executable, "scripts/check_kpi_coverage.py"],
        capture_output=True,
        text=True,
        env={**os.environ, "E2I_DB_INTEGRATION": "1", "PYTHONPATH": os.getcwd()},
    )
    assert out.returncode == 0, f"coverage probe failed:\n{out.stdout}\n{out.stderr}"
    summary = out.stdout.strip().splitlines()[-1]
    assert "MAPPED 45" in summary and "EMPTY 0" in summary and "N/A 0" in summary, summary
