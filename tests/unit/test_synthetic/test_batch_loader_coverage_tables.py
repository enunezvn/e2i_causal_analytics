"""Shard 09 Task 1: the 15 new substrate tables must be registered in the loader.

batch_loader.py:370-372 column-gates every frame against TABLE_COLUMNS; an
unregistered table loads no columns (and `is_synthetic` defaults to false at the
DB), silently mixing synthetic into real reads. LOADING_ORDER must also list the
table or load_all skips it entirely (batch_loader.py:325). FK parents must precede
children or the load throws an FK violation.
"""

from src.ml.synthetic.loaders.batch_loader import LOADING_ORDER, TABLE_COLUMNS

NEW_TABLES = [
    "ml_experiments",
    "ab_experiment_assignments",
    "ab_experiment_enrollments",
    "ab_experiment_results",
    "ml_model_registry",
    "ml_training_runs",
    "ml_deployments",
    "ml_observability_spans",
    "causal_paths",
    "learning_signals",
    "user_sessions",
    "hcp_intent_surveys",
    "data_source_tracking",
    "etl_pipeline_metrics",
    "ml_annotations",
]


def test_new_tables_registered_in_loading_order():
    for t in NEW_TABLES:
        assert t in LOADING_ORDER, f"{t} missing from LOADING_ORDER -> loader skips it"


def test_new_tables_have_column_lists_including_is_synthetic():
    for t in NEW_TABLES:
        assert t in TABLE_COLUMNS, (
            f"{t} missing from TABLE_COLUMNS -> all rows dropped at batch_loader.py:370"
        )
        assert "is_synthetic" in TABLE_COLUMNS[t], (
            f"{t} omits is_synthetic -> provenance tag stripped by gating"
        )


def test_fk_parents_precede_children_in_loading_order():
    idx = {t: i for i, t in enumerate(LOADING_ORDER)}
    # ml_experiments before ab_* and ml_model_registry
    assert idx["ml_experiments"] < idx["ab_experiment_assignments"]
    assert idx["ab_experiment_assignments"] < idx["ab_experiment_enrollments"]  # FK assignment_id
    assert idx["ml_experiments"] < idx["ml_model_registry"]  # FK experiment_id
    assert idx["ml_model_registry"] < idx["ml_training_runs"]  # FK model_registry_id
    assert idx["ml_model_registry"] < idx["ml_deployments"]  # FK model_registry_id
