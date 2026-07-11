"""Shard 09 Task 2: experiment + A/B substrate (experiment_monitor /
experiment_designer / scope_definer). Mirrors the 621 real "running" experiments
and attaches assignments/enrollments/results with KNOWN, recoverable
PER-CHANNEL uplifts (2026-07-11 meaningful-portfolio redesign). All values
enum-exact (22P02 landmine) and is_synthetic-tagged."""

from datetime import datetime, timedelta, timezone

from src.ml.synthetic.config import Brand
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.experiment_generator import (
    CHANNEL_TRUE_UPLIFT,
    ABExperimentGenerator,
    ExperimentGenerator,
    _exp_id,
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


def test_experiments_are_meaningful_and_explainable():
    """2026-07-11 redesign: every experiment names its intervention, outcome,
    cohort and region; carries a hypothesis description; and is channel-tagged
    (migration 100). No more 360 clones of one template."""
    df = ExperimentGenerator(GeneratorConfig(seed=7, n_records=16, brand=Brand.FABHALTA)).generate()
    assert df["experiment_name"].is_unique
    # All 8 taxonomy channels are cycled through
    assert set(df["intervention_channel"]) == set(CHANNEL_TRUE_UPLIFT)
    # Names lead with the brand and embed the outcome label
    assert df["experiment_name"].str.startswith("Fabhalta: ").all()
    assert df["experiment_name"].str.contains("PNH therapy persistence").all()
    # Descriptions state the hypothesis + in-silico design, per experiment
    assert df["description"].str.contains("In-silico A/B test").all()
    assert df["description"].str.contains("Hypothesis:").all()
    assert df["target_population"].str.len().gt(0).all()
    # Staggered starts (10-90 days back), not one same-instant burst
    created = df["created_at"].map(lambda s: datetime.fromisoformat(s))
    ages = [(datetime.now(timezone.utc) - c).days for c in created]
    assert min(ages) >= 9 and max(ages) <= 91
    assert len(set(ages)) > 1, "starts must be staggered, not a single burst"


def test_generator_columns_are_registered_with_the_loader():
    """Every column the generator emits must be registered in the loader's
    TABLE_COLUMNS whitelist — BatchLoader silently gates out unregistered
    columns at load time (caught live 2026-07-11: the enrollment-plan refresh
    wrote all 360 rows with a NULL plan because the two new columns were
    missing from the whitelist)."""
    from src.ml.synthetic.loaders.batch_loader import TABLE_COLUMNS

    df = ExperimentGenerator(GeneratorConfig(seed=7, n_records=1, brand=Brand.KISQALI)).generate()
    missing = set(df.columns) - set(TABLE_COLUMNS["ml_experiments"])
    assert not missing, f"generator emits columns the loader would silently drop: {missing}"


def test_experiments_carry_a_real_enrollment_plan():
    """Migration 101: every synthetic experiment records a REAL enrollment plan
    (nominal 10 units/day over a 45-120 day window) so the monitor's
    plan-relative health checks and information fraction have honest inputs —
    the fabricated config.target_sample_size=1000 default flagged the entire
    live portfolio "warning" (2026-07-11 incident)."""
    df = ExperimentGenerator(GeneratorConfig(seed=7, n_records=24, brand=Brand.KISQALI)).generate()
    assert df["planned_duration_days"].between(45, 120).all()
    assert (df["target_enrollment"] == df["planned_duration_days"] * 10).all()
    # Varied plans (not one constant), so the portfolio shows honest variety
    assert df["planned_duration_days"].nunique() > 1


def test_experiment_ids_stay_keyed_on_legacy_slug():
    """REGRESSION GUARD: the id must stay uuid5(legacy 'synth_<brand>_exp_NNNN')
    even though the display name is now meaningful — that identity is what lets
    the redesigned portfolio UPDATE the deployed 360 rows (and their FK fan-out)
    in place. Anchored to the live prod row for fabhalta #0000."""
    df = ExperimentGenerator(GeneratorConfig(seed=7, n_records=1, brand=Brand.FABHALTA)).generate()
    assert df["id"].iloc[0] == _exp_id("synth_fabhalta_exp_0000")
    # Verified against the deployed DB row 2026-07-11:
    assert df["id"].iloc[0] == "d2e67172-3b75-5357-b6e2-03c0abf9163a"


def test_channel_taxonomy_mirrors_digital_twin_catalog():
    """_CHANNELS mirrors the user-approved digital-twin INTERVENTION_CATALOG
    (mirrored, not imported, to keep the generator free of the twin stack —
    this test is the drift tripwire)."""
    from src.digital_twin.effect.provider import INTERVENTION_CATALOG
    from src.ml.synthetic.generators.experiment_generator import _CHANNELS

    assert [(c[0], c[1]) for c in _CHANNELS] == list(INTERVENTION_CATALOG)


def test_ab_known_per_channel_uplift_recoverable_and_enum_safe():
    exp = ExperimentGenerator(GeneratorConfig(seed=7, n_records=16, brand=Brand.KISQALI)).generate()
    ab = ABExperimentGenerator(GeneratorConfig(seed=9), experiments_df=exp)
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
    # PER-CHANNEL ground truth must be recoverable: mean observed effect within
    # 3 unpooled SEs of the planted uplift for every channel (2 experiments per
    # channel at n>=120; tolerance scales with the actual arm sizes).
    merged = res.merge(exp[["id", "intervention_channel"]], left_on="experiment_id", right_on="id")
    for channel, group in merged.groupby("intervention_channel"):
        truth = CHANNEL_TRUE_UPLIFT[channel]
        n_min = int(min(group["treatment_n"].min(), group["control_n"].min()))
        # Bernoulli worst-case SE per experiment, shrunk by #experiments
        se = (0.5 / (n_min**0.5)) * 2 / (len(group) ** 0.5)
        observed = float(group["effect_estimate"].mean())
        assert abs(observed - truth) < 3 * se, (
            f"{channel}: observed {observed:.3f} vs truth {truth:.3f} (3se={3 * se:.3f})"
        )
    assert enr["assignment_id"].isin(asn["id"]).all()  # FK integrity
    assert res["experiment_id"].isin(exp["id"]).all()  # FK integrity
    for f in (asn, enr, res):
        assert f["is_synthetic"].all()


def test_ab_statistics_are_honest_and_enrollment_rolls_to_frontier():
    """p-values are real two-proportion z-tests (the null channel must not come
    out significant by construction) and enrollment rolls forward to the
    generation frontier so weekly refreshes keep the substrate fresh."""
    exp = ExperimentGenerator(
        GeneratorConfig(seed=7, n_records=16, brand=Brand.FABHALTA)
    ).generate()
    out = ABExperimentGenerator(GeneratorConfig(seed=9), experiments_df=exp).generate()
    asn, res = out["ab_experiment_assignments"], out["ab_experiment_results"]
    merged = res.merge(
        exp[["id", "intervention_channel", "created_at"]],
        left_on="experiment_id",
        right_on="id",
    )
    # The deliberate null channel (digital_engagement, uplift 0.00): with
    # n>=120/exp the two tests must not BOTH clear p<0.05.
    nulls = merged[merged["intervention_channel"] == "digital_engagement"]
    assert len(nulls) == 2
    assert not nulls["is_significant"].all(), "null channel reported uniformly significant"
    # p-values vary (not the old hardcoded 0.01) and CIs bracket the estimate
    assert res["p_value"].nunique() > 1
    assert (res["effect_ci_lower"] <= res["effect_estimate"]).all()
    assert (res["effect_ci_upper"] >= res["effect_estimate"]).all()
    # Freshness: every experiment's newest assignment lands within 24h of the
    # generation frontier (rolling enrollment, not a frozen batch stamp)...
    now = datetime.now(timezone.utc)
    newest = asn.groupby("experiment_id")["assigned_at"].max().map(datetime.fromisoformat)
    assert ((now - newest) < timedelta(hours=24)).all()
    # ...and no assignment predates its experiment's start.
    joined = asn.merge(exp[["id", "created_at"]], left_on="experiment_id", right_on="id")
    assert (
        joined["assigned_at"].map(datetime.fromisoformat)
        >= joined["created_at"].map(datetime.fromisoformat)
    ).all()


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
