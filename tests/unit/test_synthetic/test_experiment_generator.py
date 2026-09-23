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

# The namespaced HCP universe (hcp_profiles.hcp_id) every A/B panel is sampled
# from — the unit contract (see the parity/namespace tests at the bottom).
_HCP_IDS = [f"scvhcp_{i:05d}" for i in range(5000)]


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
    ab = ABExperimentGenerator(GeneratorConfig(seed=9), experiments_df=exp, hcp_ids=_HCP_IDS)
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
    out = ABExperimentGenerator(
        GeneratorConfig(seed=9), experiments_df=exp, hcp_ids=_HCP_IDS
    ).generate()
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
            GeneratorConfig(seed=9),
            experiments_df=exp,
            units_per_experiment=20,
            true_uplift=0.15,
            hcp_ids=_HCP_IDS,
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


# --------------------------------------------------------------------------
# unit_id namespace contract (2026-09-23, Part of #2207).
#
# Measured on prod 2026-09-23: 185,532 assignments / 360 experiments carried
# unit_id = f"hcp_{u:05d}" where u was a per-experiment loop counter, so
# (a) no unit_id was ever an hcp_profiles.hcp_id (scvhcp_NNNNN) and the
#     ExperimentOutcomeRepository join unit_id == business_metrics.hcp_id
#     matched ZERO rows for every experiment, and
# (b) variant == counter parity, so hcp_00000 was treatment in all 360
#     experiments — HCP identity perfectly confounded with arm.
# The generator must draw each experiment's panel from the supplied
# namespaced HCP universe WITHOUT replacement and draw the arm from the rng,
# independent of HCP identity and of enrollment position.
# --------------------------------------------------------------------------


def _exp_without_created_at(n_records: int, brand=Brand.KISQALI, seed: int = 7):
    """Experiments frame that routes ABExperimentGenerator through its
    ``units_per_experiment`` fallback (no created_at -> no age x rate sizing),
    so a test can pin the exact panel size."""
    return (
        ExperimentGenerator(GeneratorConfig(seed=seed, n_records=n_records, brand=brand))
        .generate()
        .drop(columns=["created_at"])
    )


def test_ab_unit_ids_are_drawn_from_the_namespaced_hcp_universe():
    """(i) every unit_id IS an id from the supplied hcp_ids (an
    hcp_profiles.hcp_id), so the unit_id == business_metrics.hcp_id join in
    src/repositories/experiment_outcome.py can match."""
    exp = ExperimentGenerator(GeneratorConfig(seed=7, n_records=16, brand=Brand.KISQALI)).generate()
    out = ABExperimentGenerator(
        GeneratorConfig(id_prefix="scv", seed=9), experiments_df=exp, hcp_ids=_HCP_IDS
    ).generate()
    asn = out["ab_experiment_assignments"]
    assert asn["unit_id"].isin(_HCP_IDS).all(), "unit_id must be an hcp_profiles.hcp_id"
    assert asn["unit_id"].str.startswith("scvhcp_").all()
    assert not asn["unit_id"].str.match(r"^hcp_\d{5}$").any(), "old counter ids leaked"


def test_ab_no_experiment_reuses_a_unit_and_panels_are_samples():
    """(ii) sampling WITHOUT replacement per experiment: every experiment's
    panel has exactly n distinct HCPs; panels differ across experiments
    (a fixed prefix of the universe would be the old confound in disguise)."""
    exp = ExperimentGenerator(
        GeneratorConfig(seed=7, n_records=16, brand=Brand.FABHALTA)
    ).generate()
    out = ABExperimentGenerator(
        GeneratorConfig(id_prefix="scv", seed=9), experiments_df=exp, hcp_ids=_HCP_IDS
    ).generate()
    asn = out["ab_experiment_assignments"]
    per_exp = asn.groupby("experiment_id")["unit_id"].agg(["size", "nunique"])
    assert (per_exp["size"] == per_exp["nunique"]).all()
    assert not asn.duplicated(subset=["experiment_id", "unit_id"]).any()
    panels = asn.groupby("experiment_id")["unit_id"].apply(frozenset)
    assert panels.nunique() == len(panels), "every experiment must draw its own panel"


def test_ab_arm_is_independent_of_hcp_identity_and_enrollment_position():
    """(iii) parity guard. Old bug: variant = counter parity -> the same unit
    was treatment in EVERY experiment. Now, for HCPs seen in >= 6 experiments
    the treatment share must be strictly inside (0, 1) for >= 95% of them
    (P(all-one-arm | k >= 6 draws) <= 2 * 0.5**6 = 3.1%, and most HCPs here
    are drawn ~10x, so the expected violation rate is well under 1%). Arms
    stay balanced within an experiment (SRM sees 50/50, +-1 unit), and the
    arm is NOT a function of enrollment position either (a 'first half =
    treatment' rule would confound arm with assigned_at)."""
    exp = _exp_without_created_at(16, brand=Brand.KISQALI)
    hcp_ids = [f"scvhcp_{i:05d}" for i in range(300)]
    asn = ABExperimentGenerator(
        GeneratorConfig(id_prefix="scv", seed=9),
        experiments_df=exp,
        units_per_experiment=200,
        hcp_ids=hcp_ids,
    ).generate()["ab_experiment_assignments"]
    asn = asn.assign(is_t=(asn["variant"] == "treatment").astype(int))
    # per-HCP mixing across experiments
    per_hcp = asn.groupby("unit_id")["is_t"].agg(["size", "mean"])
    seen = per_hcp[per_hcp["size"] >= 6]
    assert len(seen) >= 200, f"test setup: only {len(seen)} HCPs drawn >= 6 times"
    mixed = ((seen["mean"] > 0) & (seen["mean"] < 1)).mean()
    assert mixed >= 0.95, f"only {mixed:.1%} of frequently-drawn HCPs appear in both arms"
    # balanced arms within each experiment (odd n -> treatment gets the extra)
    per_exp = asn.groupby("experiment_id")["is_t"].agg(["size", "sum"])
    assert ((per_exp["sum"] - (per_exp["size"] - per_exp["sum"])).abs() <= 1).all()
    assert (per_exp["sum"] >= per_exp["size"] - per_exp["sum"]).all()
    # arm independent of enrollment position: pooled treatment share among the
    # FIRST half of each experiment's panel is ~0.5 (1600 units -> sd ~0.0125)
    asn["pos"] = asn.groupby("experiment_id").cumcount()
    asn["n"] = asn.groupby("experiment_id")["unit_id"].transform("size")
    first_half = asn[asn["pos"] < asn["n"] // 2]
    share = float(first_half["is_t"].mean())
    assert 0.45 <= share <= 0.55, f"arm follows enrollment position: first-half share {share:.3f}"


def test_ab_requires_hcp_universe_and_caps_panels_at_its_size(caplog):
    """(iv) fail LOUD, never fall back to the counter: no hcp_ids -> ValueError
    at generate(). A universe SMALLER than the requested panel (never at
    production scale: FULL_SIZES hcp=5000 >= the 1400 clamp; but --small has
    500 and the hermetic loader tests 50) caps the panel at the universe —
    still without replacement, every HCP drawn exactly once — and warns."""
    import logging

    import pytest

    exp = _exp_without_created_at(3)
    with pytest.raises(ValueError, match="hcp_ids"):
        ABExperimentGenerator(GeneratorConfig(seed=9), experiments_df=exp).generate()
    with pytest.raises(ValueError, match="hcp_ids"):
        ABExperimentGenerator(GeneratorConfig(seed=9), experiments_df=exp, hcp_ids=[]).generate()
    small = [f"scvhcp_{i:05d}" for i in range(10)]
    with caplog.at_level(
        logging.WARNING, logger="src.ml.synthetic.generators.experiment_generator"
    ):
        asn = ABExperimentGenerator(
            GeneratorConfig(seed=9), experiments_df=exp, units_per_experiment=20, hcp_ids=small
        ).generate()["ab_experiment_assignments"]
    per_exp = asn.groupby("experiment_id")["unit_id"].agg(["size", "nunique"])
    assert (per_exp["size"] == 10).all() and (per_exp["nunique"] == 10).all()
    assert "capped at the universe size" in caplog.text


def test_ab_panels_deterministic_for_same_seed_and_hcp_order():
    """(v) reseed idempotency extends to the panel: same seed + same hcp_ids
    ORDER -> identical (id, experiment_id, unit_id, variant). The caller's
    order is part of the seed contract (the generator does NOT sort), which is
    why both loader paths feed hcp_profiles ids in PK order."""
    import pandas as pd

    exp = ExperimentGenerator(GeneratorConfig(seed=7, n_records=3, brand=Brand.KISQALI)).generate()
    cols = ["id", "experiment_id", "unit_id", "variant"]

    def gen(ids):
        return (
            ABExperimentGenerator(
                GeneratorConfig(id_prefix="scv", seed=9), experiments_df=exp, hcp_ids=ids
            )
            .generate()["ab_experiment_assignments"][cols]
            .reset_index(drop=True)
        )

    pd.testing.assert_frame_equal(gen(_HCP_IDS), gen(list(_HCP_IDS)))
    # a different order is a different draw (documented contract, not an accident)
    assert not gen(_HCP_IDS)["unit_id"].equals(gen(list(reversed(_HCP_IDS)))["unit_id"])


# ---------------------------------------------------------------------------
# Per-experiment UNIT OUTCOME feed (option d1, owner decision 2026-09-23, Part of
# #2207). The generator writes each unit's drawn outcome ``y`` — the SAME draw the
# aggregate ab_experiment_results row is computed from — to a fourth frame keyed
# (experiment_id, unit_id, metric_name) and time-indexed by observed_at, so
# ExperimentOutcomeRepository.load_arrays can measure the planted per-channel
# truth instead of joining an independent outcome DGP (structural null).
# ---------------------------------------------------------------------------


def _ab_with_unit_outcomes(seed_exp: int = 7, seed_ab: int = 9, n: int = 16):
    exp = ExperimentGenerator(
        GeneratorConfig(seed=seed_exp, n_records=n, brand=Brand.KISQALI)
    ).generate()
    out = ABExperimentGenerator(
        GeneratorConfig(seed=seed_ab), experiments_df=exp, hcp_ids=_HCP_IDS
    ).generate()
    return exp, out


def test_unit_outcomes_frame_is_one_row_per_assignment_keyed_and_time_indexed():
    """(i) exactly one outcome row per assignment; keys match the assignment
    frame; metric_name is the experiment's prediction_target; observed_at never
    precedes the assignment and never lands after the generation frontier."""
    exp, out = _ab_with_unit_outcomes()
    asn, uo = out["ab_experiment_assignments"], out["ab_experiment_unit_outcomes"]
    assert len(uo) == len(asn)
    assert uo["assignment_id"].is_unique
    assert set(uo["assignment_id"]) == set(asn["id"])
    assert set(uo.columns) >= {
        "id",
        "assignment_id",
        "experiment_id",
        "unit_id",
        "metric_name",
        "outcome_value",
        "observed_at",
        "is_synthetic",
    }
    joined = uo.merge(asn, left_on="assignment_id", right_on="id", suffixes=("", "_asn"))
    assert (joined["unit_id"] == joined["unit_id_asn"]).all()
    assert (joined["experiment_id"] == joined["experiment_id_asn"]).all()
    # metric_name == the experiment's prediction_target (Kisqali -> kisqali_dx_adoption)
    target_by_exp = exp.set_index("id")["prediction_target"]
    assert (uo["metric_name"] == uo["experiment_id"].map(target_by_exp)).all()
    assert set(uo["metric_name"]) == {"kisqali_dx_adoption"}
    # time index: assignment <= observed_at <= now
    now = datetime.now(timezone.utc)
    observed = joined["observed_at"].map(datetime.fromisoformat)
    assigned = joined["assigned_at"].map(datetime.fromisoformat)
    assert (observed >= assigned).all(), "an outcome must never precede its assignment"
    assert (observed <= now).all(), "an outcome must never be observed in the future"
    assert (observed > assigned).any(), "the lag must not be identically zero"
    assert uo["is_synthetic"].all()
    assert uo["id"].is_unique
    # deterministic ids: uuid5 on the natural key (reseed UPDATEs in place)
    assert (
        uo["id"]
        == [
            _exp_id("uo", a, m) for a, m in zip(uo["assignment_id"], uo["metric_name"], strict=True)
        ]
    ).all()
    # the outcome is the drawn Bernoulli y
    assert set(uo["outcome_value"].unique()).issubset({0.0, 1.0})


def test_unit_outcome_feed_reproduces_the_stored_result_exactly():
    """(ii) The cheapest disproof of the whole design, in-process, no DB: for every
    experiment mean(y | treatment) - mean(y | control) from the UNIT frame equals
    the aggregate row's effect_estimate to 1e-9 and the arm counts match."""
    exp, out = _ab_with_unit_outcomes()
    asn, uo, res = (
        out["ab_experiment_assignments"],
        out["ab_experiment_unit_outcomes"],
        out["ab_experiment_results"],
    )
    j = uo.merge(asn[["id", "variant"]], left_on="assignment_id", right_on="id")
    for _, r in res.iterrows():
        rows = j[j["experiment_id"] == r["experiment_id"]]
        c = rows[rows["variant"] == "control"]["outcome_value"]
        t = rows[rows["variant"] == "treatment"]["outcome_value"]
        assert len(c) == r["control_n"] and len(t) == r["treatment_n"]
        assert abs((t.mean() - c.mean()) - r["effect_estimate"]) < 1e-9, r["experiment_id"]
        assert abs(c.mean() - r["control_mean"]) < 1e-9
        assert abs(t.mean() - r["treatment_mean"]) < 1e-9


def test_result_primary_metric_equals_the_experiment_prediction_target():
    """(iii) ab_experiment_results.primary_metric must name the SAME quantity as
    ml_experiments.prediction_target (and the unit outcome's metric_name) — the
    pre-existing literal 'conversion_rate' was a label no synthetic experiment
    carries."""
    exp, out = _ab_with_unit_outcomes()
    res, uo = out["ab_experiment_results"], out["ab_experiment_unit_outcomes"]
    target_by_exp = exp.set_index("id")["prediction_target"]
    assert (res["primary_metric"] == res["experiment_id"].map(target_by_exp)).all()
    assert set(res["primary_metric"]) == set(uo["metric_name"]) == {"kisqali_dx_adoption"}


def test_unit_outcome_columns_are_registered_with_the_loader():
    """(iv) mirror of the ml_experiments pin: BatchLoader silently drops
    unregistered columns, so every column the frame carries must be whitelisted."""
    from src.ml.synthetic.loaders.batch_loader import TABLE_COLUMNS

    _, out = _ab_with_unit_outcomes(n=1)
    uo = out["ab_experiment_unit_outcomes"]
    assert "ab_experiment_unit_outcomes" in TABLE_COLUMNS
    missing = set(uo.columns) - set(TABLE_COLUMNS["ab_experiment_unit_outcomes"])
    assert not missing, f"generator emits columns the loader would silently drop: {missing}"


def test_unit_outcomes_are_deterministic_for_a_seed():
    """Same seed -> same ids, outcomes and observation LAGS. Absolute timestamps
    roll with the generation frontier (assigned_at already does), so the pin is
    on the lag (observed_at - assigned_at), which comes from the seeded stream."""
    _, a = _ab_with_unit_outcomes()
    _, b = _ab_with_unit_outcomes()
    ua, ub = a["ab_experiment_unit_outcomes"], b["ab_experiment_unit_outcomes"]
    assert ua["id"].tolist() == ub["id"].tolist()
    assert ua["outcome_value"].tolist() == ub["outcome_value"].tolist()

    def _lags(out):
        uo = out["ab_experiment_unit_outcomes"].merge(
            out["ab_experiment_assignments"][["id", "assigned_at"]],
            left_on="assignment_id",
            right_on="id",
        )
        return [
            (datetime.fromisoformat(o) - datetime.fromisoformat(s)).total_seconds()
            for o, s in zip(uo["observed_at"], uo["assigned_at"], strict=True)
        ]

    la, lb = _lags(a), _lags(b)
    assert (
        max(abs(x - y) for x, y in zip(la, lb, strict=True)) < 5.0
    )  # sub-second frontier drift only
    assert 3600.0 <= max(la) <= 14 * 86400.0
