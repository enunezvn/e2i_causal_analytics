"""The twin's planted cohort columns and the per-HCP ETL must never share a column.

``business_metrics`` ``per_hcp_rollup`` rows have two writers. The per-HCP ETL recomputes its
value columns from the base tables on every upsert; ``scripts/backfill_segment_engagement.py``
plants the synthetic-gold DGP the Digital Twin estimates from. While the DGP outcome lived in
``conversion_rate`` both wrote it: the ETL silently replaced the planted outcome on every row it
touched, and re-planting made the ETL's own recompute preview report ``rows_changed`` on rows it
had just written (the #2114 certification asserts that preview is all zeros).

2026-09-21: a full-window per-HCP backfill replaced the rows wholesale and the twin went dark
for every brand. These tests pin the separation structurally, so the conflict cannot come back
by someone adding a column to either writer.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from src.digital_twin.effect import cohort_causal_estimator, cohort_loader, provider
from src.etl import business_metrics_per_hcp_etl as etl

_REPO = Path(__file__).resolve().parents[4]


def _etl_upsert_set_columns() -> set[str]:
    cols = set(re.findall(r"(\w+)\s*=\s*EXCLUDED\.", etl._PER_HCP_ROLLUP_ON_CONFLICT))
    assert "conversion_rate" in cols, "parser lost the SET arm — the guard below would be vacuous"
    return cols


def _plant_script():
    path = _REPO / "scripts" / "backfill_segment_engagement.py"
    spec = importlib.util.spec_from_file_location("backfill_segment_engagement", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def test_cohort_outcome_is_not_a_column_the_etl_recomputes():
    assert provider.COHORT_OUTCOME_COLUMN not in _etl_upsert_set_columns()


def test_no_planted_treatment_channel_is_a_column_the_etl_recomputes():
    planted = set(provider.INTERVENTION_TREATMENT_MAP.values())
    assert planted.isdisjoint(_etl_upsert_set_columns())


def test_the_outcome_has_one_name_across_loader_estimator_and_provider():
    assert cohort_causal_estimator._OUTCOME_COL == provider.COHORT_OUTCOME_COLUMN
    assert provider.COHORT_OUTCOME_COLUMN in cohort_loader._COHORT_COLUMNS.split(",")
    assert provider.COHORT_OUTCOME_COLUMN in cohort_loader._NUMERIC_COLUMNS


def test_the_plant_writes_the_cohort_outcome_and_leaves_the_etl_columns_alone():
    script = _plant_script()
    written = set(script.PLANTED_WRITE_COLUMNS)
    assert provider.COHORT_OUTCOME_COLUMN in written
    assert set(provider.INTERVENTION_TREATMENT_MAP.values()) <= written
    assert written.isdisjoint(_etl_upsert_set_columns())


class _RecordingQuery:
    def __init__(self, log, rows):
        self._log, self._rows, self._preds = log, rows, []

    def select(self, *_a, **_k):
        return self

    def update(self, payload):
        self._preds.append(("update", tuple(sorted(payload))))
        return self

    def eq(self, column, value):
        self._preds.append(("eq", column, value))
        return self

    def order(self, *_a, **_k):
        return self

    def range(self, *_a, **_k):
        return self

    def execute(self):
        self._log.append(self._preds)
        return type("R", (), {"data": self._rows})()


class _RecordingClient:
    def __init__(self, rows=()):
        self.log, self._rows = [], list(rows)

    def table(self, _name):
        return _RecordingQuery(self.log, self._rows)


def test_the_plant_reads_and_writes_synthetic_rows_only():
    """codex r1 MEDIUM: the DGP is synthetic-gold. The fetch filtered on metric_type alone, so the
    day real per-HCP rows land, --execute would plant fabricated treatments and outcomes into them
    and leave is_synthetic = false."""
    import pandas as pd

    script = _plant_script()
    client = _RecordingClient()
    script.fetch_rows(client)
    assert ("eq", "is_synthetic", True) in client.log[0]

    regen = pd.DataFrame([{script.KEY: "m1", **dict.fromkeys(script.PLANTED_WRITE_COLUMNS, 1.0)}])
    writer = _RecordingClient()
    script.update_rows(writer, regen)
    assert ("eq", "is_synthetic", True) in writer.log[0]
    assert ("eq", script.KEY, "m1") in writer.log[0]


def test_the_plant_refuses_a_frame_that_holds_a_row_not_marked_synthetic():
    import pandas as pd
    import pytest

    script = _plant_script()
    live = pd.DataFrame([{"is_synthetic": True}, {"is_synthetic": False}, {"is_synthetic": None}])
    with pytest.raises(SystemExit):
        script.require_synthetic_only(live)
    script.require_synthetic_only(pd.DataFrame([{"is_synthetic": True}]))
