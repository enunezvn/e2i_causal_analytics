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

import pytest

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
    with pytest.raises(SystemExit) as exc:
        script.require_synthetic_only(live)
    assert exc.value.code == 3
    # codex r2/r3 LOW: with pandas' nullable boolean dtype `flags != True` yields <NA> for a
    # missing flag and sum() skips it, so the missing row was not counted.
    nullable = pd.DataFrame({"is_synthetic": pd.array([True, pd.NA], dtype="boolean")})
    with pytest.raises(SystemExit):
        script.require_synthetic_only(nullable)
    script.require_synthetic_only(pd.DataFrame([{"is_synthetic": True}]))


def test_the_etl_preview_guards_exactly_the_columns_the_plant_writes():
    """A channel added to the plant must reach the preview's obsolete-with-cohort-data count,
    or the reconcile can again delete planted data the readout never mentioned."""
    script = _plant_script()
    assert set(etl.COHORT_DATA_COLUMNS) == set(script.PLANTED_WRITE_COLUMNS)
    assert set(etl.COHORT_DATA_COLUMNS).isdisjoint(_etl_upsert_set_columns())


def test_the_plant_derives_every_channel_column_from_the_shared_map():
    """codex r1 (2026-09-22): a second intervention->column map in the plant would let a
    swap in the shared map plant one channel's DGP under another's name while every
    set-based check stayed green. The plant may name interventions; it may not name columns."""
    from src.data.per_hcp_cohort_columns import INTERVENTION_TREATMENT_MAP

    script = _plant_script()
    assert script.ChannelSpec.__dataclass_fields__["column"].init is False
    derived = {spec.intervention: spec.column for spec in script.CHANNEL_SPECS}
    derived["digital_engagement"] = script.LEGACY_ENGAGEMENT_COLUMN
    assert derived == INTERVENTION_TREATMENT_MAP
    with pytest.raises(KeyError):
        script.ChannelSpec(
            intervention="not_an_intervention",
            kind="poisson",
            intercept=0.0,
            beta_market=0.0,
            beta_volume=0.0,
            region_offset={},
            noise_std=0.0,
            tau_by_region={},
        )
