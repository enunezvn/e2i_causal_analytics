"""scripts/backfill_uas7_severity_coherence: plan, refusals, and the transactional SQL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    return pd.DataFrame(
        {
            "patient_id": [f"scvpt_{seed}_{i:06d}" for i in range(n)],
            "disease_severity": sev,
            "urticaria_severity_uas7": uas7,
            "segment_assignment": tier,
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
