"""Shard 09 Task 2: experiment + A/B substrate (experiment_monitor /
experiment_designer / scope_definer). Mirrors the 621 real "running" experiments
and attaches assignments/enrollments/results with a KNOWN, recoverable uplift.
All values enum-exact (22P02 landmine) and is_synthetic-tagged."""

from src.ml.synthetic.config import Brand
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.experiment_generator import (
    ABExperimentGenerator,
    ExperimentGenerator,
)


def test_experiments_running_and_branded_and_tagged():
    g = ExperimentGenerator(GeneratorConfig(seed=7, n_records=30, brand=Brand.KISQALI))
    df = g.generate()
    assert len(df) == 30
    assert (df["status"] == "running").all()  # mirrors the 621 real running exps
    assert (df["brand"] == "Kisqali").all()  # brand_type enum-exact
    assert df["is_synthetic"].all()
    assert df["minimum_auc"].between(0.5, 1.0).all()  # ml_experiments valid_auc CHECK
    assert df["minimum_precision_at_k"].between(0.0, 1.0).all()  # valid_precision CHECK
    assert set(df["region"]).issubset({"northeast", "south", "midwest", "west"})


def test_ab_known_uplift_recoverable_and_enum_safe():
    exp = ExperimentGenerator(GeneratorConfig(seed=7, n_records=3, brand=Brand.KISQALI)).generate()
    ab = ABExperimentGenerator(
        GeneratorConfig(seed=9),
        experiments_df=exp,
        # 600 units/exp (300/arm) -> SE(diff) ~= 0.028, so the empirical uplift
        # recovers the true 0.15 within +/-0.05 (powered, not a lucky seed).
        units_per_experiment=600,
        true_uplift=0.15,
    )
    out = ab.generate()
    asn, enr, res = (
        out["ab_experiment_assignments"],
        out["ab_experiment_enrollments"],
        out["ab_experiment_results"],
    )
    # enum-exact values only
    assert set(asn["variant"]).issubset({"control", "treatment"})
    assert set(asn["unit_type"]).issubset({"hcp", "patient", "territory", "account"})
    assert set(asn["randomization_method"]).issubset(
        {"simple", "stratified", "block", "cluster", "adaptive"}
    )
    assert set(res["analysis_method"]).issubset({"itt", "per_protocol", "as_treated", "cace"})
    assert set(res["analysis_type"]).issubset({"interim", "final", "post_hoc"})
    assert set(enr["enrollment_status"]).issubset(
        {"active", "withdrawn", "excluded", "completed", "lost_to_followup"}
    )
    # known uplift must be recoverable from the results row
    r = res.iloc[0]
    assert abs((r["treatment_mean"] - r["control_mean"]) - 0.15) < 0.05
    assert enr["assignment_id"].isin(asn["id"]).all()  # FK integrity
    assert res["experiment_id"].isin(exp["id"]).all()  # FK integrity
    for f in (asn, enr, res):
        assert f["is_synthetic"].all()


def test_ab_requires_non_empty_experiments():
    import pandas as pd
    import pytest

    with pytest.raises(ValueError):
        ABExperimentGenerator(GeneratorConfig(seed=1), experiments_df=pd.DataFrame())


def test_experiment_ids_deterministic_across_runs():
    """Reseed idempotency: experiment_name is already deterministic
    (synth_<brand>_exp_NNNN) but the id was uuid4 -> every reseed INSERTed 360 fresh-id
    rows -> ml_experiments accumulated (2,160 = 6x the intended 360) and the
    include-synthetic 'Active Campaigns' tile inflated. The id must be a stable function
    of the natural key so the upsert UPDATES in place."""
    a = ExperimentGenerator(GeneratorConfig(seed=7, n_records=30, brand=Brand.KISQALI)).generate()
    b = ExperimentGenerator(GeneratorConfig(seed=7, n_records=30, brand=Brand.KISQALI)).generate()
    assert list(a["id"]) == list(b["id"]), "experiment ids must be stable across runs"
    assert a["id"].is_unique


def test_ab_ids_deterministic_across_runs():
    """The A/B substrate must be idempotent too. ab_experiment_assignments carries
    UNIQUE(experiment_id, unit_id); once experiment_id is deterministic that natural key
    is stable, so a fresh-uuid assignment would collide (23505). Deterministic ids
    (keyed on the same natural key) make the upsert UPDATE in place, and the FK chain
    (enrollment.assignment_id, *.experiment_id) stays coherent across runs."""
    exp = ExperimentGenerator(GeneratorConfig(seed=7, n_records=3, brand=Brand.KISQALI)).generate()

    def gen():
        return ABExperimentGenerator(
            GeneratorConfig(seed=9), experiments_df=exp, units_per_experiment=20, true_uplift=0.15
        ).generate()

    o1, o2 = gen(), gen()
    for key in (
        "ab_experiment_assignments",
        "ab_experiment_enrollments",
        "ab_experiment_results",
    ):
        assert list(o1[key]["id"]) == list(o2[key]["id"]), f"{key} ids not stable across runs"
        assert o1[key]["id"].is_unique, f"{key} ids must be unique within a run"
    # FK chain stays coherent (ids are functions of the natural keys, not random)
    asn, enr, res = (
        o1["ab_experiment_assignments"],
        o1["ab_experiment_enrollments"],
        o1["ab_experiment_results"],
    )
    assert enr["assignment_id"].isin(asn["id"]).all()
    assert asn["experiment_id"].isin(exp["id"]).all()
    assert res["experiment_id"].isin(exp["id"]).all()
    # the (experiment_id, unit_id) natural key is unique within a run (matches the DB UNIQUE)
    assert not asn.duplicated(subset=["experiment_id", "unit_id"]).any()
