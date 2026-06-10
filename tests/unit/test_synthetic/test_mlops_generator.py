"""Shard 09 Task 3: MLOps substrate (model_selector / model_trainer /
model_deployer). ml_training_runs / ml_model_registry / ml_deployments are 0 rows
on the faithful DB; generate registry + runs + deployments consistent with the
experiments frame. Enum-exact stage / status; is_synthetic-tagged."""

from src.ml.synthetic.config import Brand
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.experiment_generator import ExperimentGenerator
from src.ml.synthetic.generators.mlops_generator import MLOpsGenerator


def test_mlops_registry_runs_deployments_consistent_and_enum_safe():
    exp = ExperimentGenerator(GeneratorConfig(seed=1, n_records=2, brand=Brand.KISQALI)).generate()
    out = MLOpsGenerator(
        GeneratorConfig(seed=2), experiments_df=exp, models_per_experiment=2
    ).generate()
    reg, runs, dep = (
        out["ml_model_registry"],
        out["ml_training_runs"],
        out["ml_deployments"],
    )
    assert len(reg) == 4 and len(runs) == 4  # 2 exps x 2 models
    assert set(reg["stage"]).issubset(
        {"development", "staging", "shadow", "production", "archived", "deprecated"}
    )
    assert set(dep["status"]).issubset(
        {"pending", "deploying", "active", "draining", "rolled_back", "failed"}
    )
    assert runs["model_registry_id"].isin(reg["id"]).all()  # FK
    assert dep["model_registry_id"].isin(reg["id"]).all()  # FK
    assert reg["experiment_id"].isin(exp["id"]).all()  # FK
    assert reg["auc"].between(0.6, 0.95).all()  # WS1-MP-001 substrate non-degenerate
    assert reg["is_champion"].sum() >= 1  # at least one champion per cohort
    for f in (reg, runs, dep):
        assert f["is_synthetic"].all()


def test_mlops_requires_non_empty_experiments():
    import pandas as pd
    import pytest

    with pytest.raises(ValueError):
        MLOpsGenerator(GeneratorConfig(seed=1), experiments_df=pd.DataFrame())
