"""Shard 09 Task 6: the load orchestration must emit the new breadth substrate.

NOTE (stale-plan correction): the plan referenced build_synthetic_datasets(seed,
n_records, run_date_iso); the real entrypoint is generate_datasets(sizes, dgp_type,
...) (see test_load_script_conversion_wiring.py). We exercise that, hermetic, no DB.
"""

import importlib

from src.ml.synthetic.config import DGPType

load_mod = importlib.import_module("scripts.load_synthetic_data")

_SMALL = {
    "hcp": 50,
    "patient": 200,
    "treatment": 200,
    "prediction": 50,
    "trigger": 400,
    "business_metrics": 30,
    "feature_values": 50,
}

_NEW_KEYS = (
    "ml_experiments",
    "ml_model_registry",
    "ml_training_runs",
    "ml_deployments",
    "ab_experiment_assignments",
    "ab_experiment_enrollments",
    "ab_experiment_results",
    "ml_observability_spans",
    "learning_signals",
    "user_sessions",
    "hcp_intent_surveys",
    "data_source_tracking",
    "etl_pipeline_metrics",
    "ml_annotations",
    "causal_paths",
)


def test_build_datasets_includes_new_substrate():
    datasets = load_mod.generate_datasets(sizes=_SMALL, dgp_type=DGPType.CONFOUNDED, seed=42)
    for k in _NEW_KEYS:
        assert k in datasets and not datasets[k].empty, f"{k} missing/empty from build"
        assert bool(datasets[k]["is_synthetic"].all()), f"{k} not all is_synthetic"


def test_experiments_running_and_ab_uplift_positive():
    datasets = load_mod.generate_datasets(sizes=_SMALL, dgp_type=DGPType.CONFOUNDED, seed=42)
    assert (datasets["ml_experiments"]["status"] == "running").all()
    res = datasets["ab_experiment_results"]
    # designed +0.15 uplift -> the mean treatment effect across experiments is positive
    assert (res["treatment_mean"] - res["control_mean"]).mean() > 0


def test_patient_journeys_carry_data_lag_hours():
    datasets = load_mod.generate_datasets(sizes=_SMALL, dgp_type=DGPType.CONFOUNDED, seed=42)
    pj = datasets["patient_journeys"]
    assert "data_lag_hours" in pj.columns
    assert pj["data_lag_hours"].notna().all()
