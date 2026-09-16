"""scripts/backfill_uas7_severity_coherence: plan, refusal, and the transactional SQL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.backfill_uas7_severity_coherence import plan, report, to_sql
from src.ml.synthetic.dgp.clinical_severity import uas7_from_severity

_CUTOFF = pd.Timestamp("2026-09-16T12:00:00Z")


def _live(n: int = 3000, *, seed: int = 0, applied: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sev = np.clip(rng.normal(5.0, 2.0, n), 0, 10).round(2)
    draw = rng.integers(16, 43, n)
    uas7 = uas7_from_severity(draw, sev) if applied else draw
    tier = np.where(sev > 7, "high_severity", np.where(sev > 4, "medium_severity", "low_severity"))
    return pd.DataFrame(
        {
            "patient_id": [f"scvpt_{i:06d}" for i in range(n)],
            "disease_severity": sev,
            "urticaria_severity_uas7": uas7,
            "segment_assignment": tier,
            "created_at": "2026-09-01T00:00:00Z",
        }
    )


def test_plan_recomputes_with_the_generator_function():
    live = _live()
    rows = plan(live, _CUTOFF)
    expected = uas7_from_severity(live["urticaria_severity_uas7"], live["disease_severity"])
    assert np.array_equal(rows["uas7_new"].to_numpy(), expected)
    assert np.corrcoef(rows["uas7_new"], rows["disease_severity"])[0, 1] > 0.3
    assert "uncontrolled-CSU axis flips" in report(rows)


def test_rows_after_the_cutoff_are_not_touched():
    live = _live(100)
    live.loc[:49, "created_at"] = "2026-09-20T00:00:00Z"
    live.loc[50:, "urticaria_severity_uas7"] = _live(100, seed=1)["urticaria_severity_uas7"][50:]
    rows = plan(live, _CUTOFF)
    assert set(rows["patient_id"]) == set(live["patient_id"][50:])


def test_refuses_rows_that_already_follow_severity():
    """A re-run would map the mapping's own output: fail closed."""
    with pytest.raises(SystemExit, match="REFUSING"):
        plan(_live(applied=True), _CUTOFF)


def test_sql_updates_only_changed_rows_inside_one_guarded_transaction():
    rows = plan(_live(500), _CUTOFF)
    sql = to_sql(rows)
    n_changed = int((rows["uas7_old"] != rows["uas7_new"]).sum())
    assert 0 < n_changed < len(rows)
    assert sql.startswith("BEGIN;") and sql.rstrip().endswith("COMMIT;")
    assert sum(ln.startswith("('scvpt_") for ln in sql.splitlines()) == n_changed
    assert f"IF updated <> {n_changed} THEN" in sql
    assert "IS DISTINCT FROM p.uas7_new" in sql
    assert "pj.urticaria_severity_uas7 = p.uas7_old" in sql  # compare-and-set on the export
    assert "RAISE EXCEPTION 'post-update corr" in sql
