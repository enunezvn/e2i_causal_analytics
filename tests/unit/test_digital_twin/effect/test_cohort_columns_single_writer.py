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
