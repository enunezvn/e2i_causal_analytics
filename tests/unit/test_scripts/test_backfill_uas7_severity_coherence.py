"""scripts/backfill_uas7_severity_coherence: plan, refusals, and the transactional SQL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import backfill_brand_axis_persistence as bf
from scripts.backfill_uas7_severity_coherence import (
    OLD_GENERATOR_CUTOFF,
    plan,
    report,
    to_sql,
)
from src.ml.synthetic.dgp.clinical_severity import uas7_from_severity

_OLD = "2026-09-01T00:00:00Z"
_NEW = "2026-09-21T03:00:00Z"  # a weekly frontier-append written by the new generator


def _live(n: int = 3000, *, seed: int = 0, applied: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sev = np.clip(rng.normal(5.0, 2.0, n), 0, 10).round(2)
    draw = rng.integers(16, 43, n)
    uas7 = uas7_from_severity(draw, sev) if applied else draw
    tier = np.where(sev > 7, "high_severity", np.where(sev > 4, "medium_severity", "low_severity"))
    # Live labels are drawn independently of the regeneration stream: the live
    # Remibrutinib labels never came from it (34% disagree), which is the whole reason
    # the rollout applies a delta instead of the full re-derivation.
    persistent = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "patient_id": [f"scvpt_{seed}_{i:06d}" for i in range(n)],
            "brand": "Remibrutinib",
            "treatment_arm": rng.integers(0, 2, n),
            "disease_severity": sev,
            "academic_hcp": rng.integers(0, 2, n),
            "geographic_region": rng.choice(["northeast", "south", "midwest", "west"], n),
            "segment_assignment": tier,
            "insurance_type": rng.choice(["commercial", "medicare", "medicaid"], n),
            "age_at_diagnosis": rng.integers(18, 80, n),
            "comorbidity_burden": rng.integers(0, 4, n),
            "prior_therapy_lines": rng.integers(0, 3, n),
            "copay_support": rng.integers(0, 2, n),
            "psp_enrolled": rng.integers(0, 2, n),
            "persistent_180d": persistent,
            "discontinued_180d": 1 - persistent,
            "urticaria_severity_uas7": uas7,
            "created_at": _OLD,
        }
    )


def test_plan_recomputes_with_the_generator_function():
    live = _live()
    rows = plan(live)
    expected = uas7_from_severity(live["urticaria_severity_uas7"], live["disease_severity"])
    assert np.array_equal(rows["uas7_new"].to_numpy(), expected)
    assert np.corrcoef(rows["uas7_new"], rows["disease_severity"])[0, 1] > 0.3
    assert "uncontrolled-CSU axis flips" in report(rows)


def test_rows_at_or_after_the_cutoff_refuse_until_vouched_for():
    """Codex r2: a weekly append BEFORE the deploy writes old-generator rows after the
    fixed cutoff; silently excluding them would leave them incoherent forever."""
    old = _live(2700, seed=1)
    newer = _live(300, seed=2).assign(created_at=_NEW)
    with pytest.raises(SystemExit, match="300 Remibrutinib rows were created at/after"):
        plan(pd.concat([old, newer], ignore_index=True))


def test_vouched_newer_rows_are_excluded_even_in_a_mixed_export():
    """Codex r1 HIGH: a mixed cohort passes any correlation band. Selection is by the
    cutoff, so rows the copula generator wrote are never remapped."""
    old = _live(2700, seed=1)
    new = _live(300, seed=2, applied=True).assign(created_at=_NEW)
    rows = plan(pd.concat([old, new], ignore_index=True), newer_rows_are_new_generator=True)
    assert set(rows["patient_id"]) == set(old["patient_id"])
    assert pd.Timestamp(_NEW) > OLD_GENERATOR_CUTOFF > pd.Timestamp(_OLD)


def test_the_selection_cannot_be_widened_past_the_fixed_cutoff():
    """Codex r3: widening to a container StartedAt remapped 232 of 300 new-generator
    rows (the append runs from the host checkout). There is no widening knob."""
    import inspect

    assert list(inspect.signature(plan).parameters) == ["live", "newer_rows_are_new_generator"]
    assert list(inspect.signature(to_sql).parameters) == ["rows"]
    mixed = pd.concat(
        [_live(2700, seed=1), _live(300, seed=2, applied=True).assign(created_at=_NEW)],
        ignore_index=True,
    )
    rows = plan(mixed, newer_rows_are_new_generator=True)
    assert len(rows) == 2700  # the mapped rows are excluded, never remapped


def test_refuses_a_second_run_on_the_same_rows():
    with pytest.raises(SystemExit, match="already"):
        plan(_live(applied=True))


def test_refuses_undefined_correlation():
    live = _live(200)
    live["disease_severity"] = 5.0
    with pytest.raises(SystemExit, match="undefined"):
        plan(live)


@pytest.mark.parametrize("bad", [np.nan, -0.5, 10.5])
def test_refuses_missing_or_out_of_domain_severity(bad):
    live = _live(200)
    live.loc[7, "disease_severity"] = bad
    with pytest.raises(SystemExit, match="disease_severity"):
        plan(live)


def test_helper_rejects_nan_severity_instead_of_an_integer_sentinel():
    with pytest.raises(ValueError, match="finite"):
        uas7_from_severity(np.array([30]), np.array([np.nan]))


@pytest.mark.parametrize("bad_id", ["scvpt_1'); DROP TABLE x; --", "a b", ""])
def test_refuses_patient_ids_that_are_not_plain_identifiers(bad_id):
    live = _live(200)
    live.loc[3, "patient_id"] = bad_id
    with pytest.raises(SystemExit, match="patient_id"):
        plan(live)


def test_sql_is_one_guarded_compare_and_set_over_the_staged_cohort():
    rows = plan(_live(500))
    sql = to_sql(rows)
    n_changed = int((rows["uas7_old"] != rows["uas7_new"]).sum())
    assert 0 < n_changed < len(rows)
    assert sql.startswith("BEGIN;") and sql.rstrip().endswith("COMMIT;")
    # The whole cohort is staged (the correlation guard reads it), not only changed rows.
    assert sum(ln.startswith("('scvpt_") for ln in sql.splitlines()) == len(rows)
    assert f"IF updated <> {n_changed} THEN" in sql
    # Codex r2: EVERY staged row (changed or not) is locked and verified first.
    assert "FOR UPDATE OF pj" in sql
    assert f"IF matched <> {len(rows)} THEN" in sql
    assert sql.index("IF matched <>") < sql.index("UPDATE patient_journeys pj SET")
    for predicate in (
        "pj.is_synthetic",
        f"pj.created_at < '{OLD_GENERATOR_CUTOFF.isoformat()}'::timestamptz",
        "pj.disease_severity = p.severity",
        "pj.urticaria_severity_uas7 = p.uas7_old",
        "IS DISTINCT FROM p.uas7_new",
    ):
        assert predicate in sql, predicate
    assert "JOIN _uas7_plan p USING (patient_id)" in sql
    assert "IF c IS NULL OR" in sql
    assert " IN (" not in sql  # no interpolated id list


def _regenerated(rows: pd.DataFrame, uas7_col: str) -> pd.Series:
    frame = rows.assign(urticaria_severity_uas7=rows[uas7_col])
    regen = bf.regenerate(frame, bf._AXES["Remibrutinib"])
    return regen.set_index("patient_id").loc[rows["patient_id"], "persistent_180d"]


def test_labels_move_only_where_the_uas7_change_moves_the_regenerated_label():
    """The live labels never came from the regeneration stream, so a full re-derivation
    rewrote 3,132 of 8,863 live labels. Only the change the new UAS7 CAUSES is applied:
    where regenerate(new UAS7) != regenerate(old UAS7) the label becomes the new one,
    everywhere else the live label stays."""
    live = _live(3000)
    rows = plan(live)
    r0 = _regenerated(rows, "uas7_old").to_numpy()
    r1 = _regenerated(rows, "uas7_new").to_numpy()
    moved = r0 != r1
    expected = np.where(moved, r1, rows["persistent_180d"].to_numpy())

    assert moved.any() and not moved.all()
    assert np.array_equal(rows["persist_new"].to_numpy(), expected)
    assert np.array_equal(rows["disc_new"].to_numpy(), 1 - expected)
    assert np.array_equal(rows["persist_old"].to_numpy(), live["persistent_180d"].to_numpy())
    label_changes = int((rows["persist_new"] != rows["persist_old"]).sum())
    full_regen_churn = int((r1 != rows["persistent_180d"].to_numpy()).sum())
    assert 0 < label_changes <= int(moved.sum()) < full_regen_churn
    assert "persistence labels changed" in report(rows)


def test_positive_control_a_rewrite_that_ignores_the_live_labels_is_caught():
    """Teeth: the full re-derivation (the old rollout step) must NOT satisfy the delta."""
    rows = plan(_live(3000))
    full = _regenerated(rows, "uas7_new").to_numpy()
    assert not np.array_equal(rows["persist_new"].to_numpy(), full)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f.loc.__setitem__((4, "persistent_180d"), 2),
        lambda f: f.loc.__setitem__((4, "discontinued_180d"), f.loc[4, "persistent_180d"]),
        lambda f: f.loc.__setitem__((4, "persistent_180d"), np.nan),
    ],
    ids=["non-binary", "not-complementary", "missing"],
)
def test_refuses_persistence_labels_that_are_not_complementary_binary(mutate):
    live = _live(200)
    mutate(live)
    with pytest.raises(SystemExit, match="persistent_180d"):
        plan(live)


def test_refuses_an_export_without_the_persistence_covariates():
    with pytest.raises(SystemExit, match="treatment_arm"):
        plan(_live(200).drop(columns=["treatment_arm"]))


def test_sql_moves_labels_in_the_same_guarded_transaction():
    rows = plan(_live(3000))
    sql = to_sql(rows)
    uas7_changed = rows["uas7_old"] != rows["uas7_new"]
    label_changed = rows["persist_old"] != rows["persist_new"]
    n_any = int((uas7_changed | label_changed).sum())
    n_labels = int(label_changed.sum())
    assert n_labels > 0
    assert f"IF updated <> {n_any} THEN" in sql
    for fragment in (
        # every staged row's old labels are verified before anything is written
        "AND pj.persistent_180d = p.persist_old",
        "AND pj.discontinued_180d = p.disc_old",
        "persistent_180d = p.persist_new",
        "discontinued_180d = p.disc_new",
        # a label can move while UAS7 does not (mean-centred axis pull)
        "OR pj.persistent_180d IS DISTINCT FROM p.persist_new",
        f"IF relabelled <> {n_labels} THEN",
    ):
        assert fragment in sql, fragment
    assert sql.index("AND pj.persistent_180d = p.persist_old") < sql.index(
        "UPDATE patient_journeys pj SET"
    )
    # after the update every staged row carries exactly its planned values
    assert "pj.persistent_180d IS DISTINCT FROM p.persist_new" in sql.split("GET DIAGNOSTICS")[1]


_REGEN_COVARIATES = (
    "treatment_arm",
    "academic_hcp",
    "geographic_region",
    "insurance_type",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
    "segment_assignment",
    "copay_support",
    "psp_enrolled",
)


def test_sql_refuses_an_export_that_is_not_the_whole_eligible_cohort():
    """Codex r1 HIGH: regenerate is one stream over the frame, so a subset export plans
    DIFFERENT labels (33 of 2,000 on the fixture) and every per-row guard still passes.
    The transaction must prove no eligible live row was left out of the plan."""
    full = plan(_live(3000))
    subset = plan(_live(3000).iloc[:2000])
    merged = subset.merge(full, on="patient_id", suffixes=("_sub", "_full"))
    assert (merged["persist_new_sub"] != merged["persist_new_full"]).any()

    sql = to_sql(full)
    guard = sql[: sql.index("UPDATE patient_journeys pj SET")]
    assert "NOT EXISTS (SELECT 1 FROM _uas7_plan p WHERE p.patient_id = pj.patient_id)" in guard
    assert "IF missing <> 0 THEN" in guard
    # the eligibility predicate does NOT filter on UAS7, so a NULL-UAS7 row fails closed
    eligible = guard[guard.index("INTO missing") : guard.index("IF missing <> 0")]
    assert "urticaria_severity_uas7" not in eligible
    assert f"pj.created_at < '{OLD_GENERATOR_CUTOFF.isoformat()}'::timestamptz" in eligible


def test_sql_compare_and_sets_every_regeneration_input():
    """Codex r1 MED: a covariate changed after the export (e.g. treatment_arm) changes
    the planned labels while severity/UAS7/labels still match."""
    live = _live(3000)
    flipped = live.assign(treatment_arm=1 - live["treatment_arm"])
    assert (plan(live)["persist_new"] != plan(flipped)["persist_new"]).any()

    sql = to_sql(plan(live))
    guard = sql[: sql.index("UPDATE patient_journeys pj SET")]
    update = sql[sql.index("UPDATE patient_journeys pj SET") : sql.index("GET DIAGNOSTICS")]
    for col in _REGEN_COVARIATES:
        predicate = f"pj.{col}::text = p.{col}"
        assert predicate in guard, predicate
        assert predicate in update, predicate


@pytest.mark.parametrize("col", ["geographic_region", "insurance_type", "segment_assignment"])
def test_refuses_string_covariates_that_are_not_plain_tokens(col):
    live = _live(200)
    live[col] = live[col].astype(object)
    live.loc[5, col] = "south'); DROP TABLE x; --"
    with pytest.raises(SystemExit, match=col):
        plan(live)


@pytest.mark.parametrize("col", ["treatment_arm", "age_at_diagnosis", "copay_support"])
def test_refuses_missing_integer_covariates(col):
    live = _live(200)
    live[col] = live[col].astype(float)
    live.loc[5, col] = np.nan
    with pytest.raises(SystemExit, match=col):
        plan(live)
