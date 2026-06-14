"""ACCEPTANCE GATE — Shard 07 (INDEX gate 9, PROVENANCE).

Proves the read-path blast radius default-excludes is_synthetic rows:

(1) test_business_metrics_real_value_unchanged_after_synthetic_insert
    - the REAL reader (BusinessMetricRepository.get_time_series) returns the same
      rows before and after inserting N synthetic business_metrics rows (real-mode
      count UNCHANGED), and INCREASES by N when include_synthetic=True. Faithful
      docker supabase-db; all inserted rows cleaned up.

(2) test_business_metrics_db_level_invariant
    - the DB-level cross-check: the exact `is_synthetic = false` predicate
      apply_provenance_filter emits hides freshly-inserted synthetic rows.

(3) test_every_blast_radius_reader_accepts_include_synthetic
    - the leakage-enumeration backstop: every ENFORCED_READERS callable is
      imported and introspected (inspect.signature) and MUST accept an
      include_synthetic kwarg. NON-VACUOUS — a forgotten/renamed reader fails.

Gated E2I_DB_INTEGRATION=1; run -n0.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful docker-Supabase test; set E2I_DB_INTEGRATION=1",
)

# A valid brand_type enum value present in the live table.
_BRAND = "Kisqali"


def _psql(sql: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tAc", sql],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _insert_bm(metric_name: str, n: int, is_synthetic: bool) -> None:
    flag = "true" if is_synthetic else "false"
    values = ",".join(
        f"('{uuid.uuid4()}', '2025-01-{(i % 27) + 1:02d}', '{metric_name}', "
        f"'{_BRAND}'::brand_type, {1.0 + i}, {flag})"
        for i in range(n)
    )
    _psql(
        "INSERT INTO business_metrics "
        "(metric_id, metric_date, metric_name, brand, value, is_synthetic) "
        f"VALUES {values};"
    )


# ---------------------------------------------------------------------------
# (1) real value UNCHANGED before/after synthetic insert (via the REAL reader)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_business_metrics_real_value_unchanged_after_synthetic_insert():
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.business_metric import BusinessMetricRepository

    metric_name = f"PROV_GATE_{uuid.uuid4().hex[:12]}"
    n_real, n_synth = 3, 5

    client = await get_async_supabase_client()
    repo = BusinessMetricRepository(supabase_client=client)

    try:
        # Seed REAL rows so real-mode count is non-trivial (> 0).
        _insert_bm(metric_name, n_real, is_synthetic=False)

        before = await repo.get_time_series(metric_name, _BRAND)
        assert len(before) == n_real, f"expected {n_real} real rows, got {len(before)}"

        # Insert synthetic rows for the SAME metric/brand.
        _insert_bm(metric_name, n_synth, is_synthetic=True)

        # Real-mode read is UNCHANGED (synthetic invisible).
        after = await repo.get_time_series(metric_name, _BRAND)
        assert len(after) == n_real, (
            f"real-mode read changed after synthetic insert: {len(after)} != {n_real}"
        )

        # Opt-in FLIPS it: count increases by exactly n_synth.
        opted_in = await repo.get_time_series(metric_name, _BRAND, include_synthetic=True)
        assert len(opted_in) == n_real + n_synth, (
            f"include_synthetic=True did not surface synthetic rows: "
            f"{len(opted_in)} != {n_real + n_synth}"
        )
    finally:
        _psql(f"DELETE FROM business_metrics WHERE metric_name = '{metric_name}';")


# ---------------------------------------------------------------------------
# (2) DB-level invariant: the is_synthetic=false predicate hides synthetic rows
# ---------------------------------------------------------------------------
def test_business_metrics_db_level_invariant():
    metric_name = f"PROV_GATE_DB_{uuid.uuid4().hex[:12]}"
    n_synth = 7
    try:
        # Baseline real-mode count for this isolated metric_name is 0.
        base_real = int(
            _psql(
                "SELECT count(*) FROM business_metrics "
                f"WHERE metric_name='{metric_name}' AND is_synthetic = false;"
            )
        )
        assert base_real == 0

        _insert_bm(metric_name, n_synth, is_synthetic=True)

        # Real-mode predicate (what apply_provenance_filter emits) is UNCHANGED.
        after_real = int(
            _psql(
                "SELECT count(*) FROM business_metrics "
                f"WHERE metric_name='{metric_name}' AND is_synthetic = false;"
            )
        )
        assert after_real == base_real, f"real-mode count changed: {after_real} != {base_real}"

        # Without the predicate (opt-in equivalent), the synthetic rows ARE there.
        total = int(
            _psql(f"SELECT count(*) FROM business_metrics WHERE metric_name='{metric_name}';")
        )
        assert total == n_synth, f"expected {n_synth} synthetic rows, got {total}"
    finally:
        _psql(f"DELETE FROM business_metrics WHERE metric_name = '{metric_name}';")


# ---------------------------------------------------------------------------
# (3) leakage-enumeration backstop — every blast-radius reader is enforced
# ---------------------------------------------------------------------------
def _enforced_readers():
    """Import + return (name, callable) for every blast-radius reader.

    Each MUST accept an include_synthetic kwarg (default-exclude with opt-in).
    A forgotten/renamed reader -> ImportError/AttributeError here = test FAILS.
    """
    from src.agents.drift_monitor.connectors.supabase_connector import SupabaseDataConnector
    from src.memory import episodic_memory
    from src.repositories.base import BaseRepository
    from src.repositories.business_metric import BusinessMetricRepository
    from src.repositories.experiment_outcome import ExperimentOutcomeRepository
    from src.repositories.ml_data_loader import MLDataLoader
    from src.repositories.patient_journey import PatientJourneyRepository
    from src.repositories.prediction import PredictionRepository
    from src.repositories.trigger import TriggerRepository
    from src.services import cohort_resolution, kpi_resolution

    return [
        # BaseRepository id-path reader (gates on HAS_PROVENANCE; inherited by
        # every taggable repo).
        ("BaseRepository.get_by_id", BaseRepository.get_by_id),
        ("BusinessMetricRepository.get_time_series", BusinessMetricRepository.get_time_series),
        (
            "BusinessMetricRepository.get_latest_snapshot",
            BusinessMetricRepository.get_latest_snapshot,
        ),
        ("BusinessMetricRepository.get_by_kpi", BusinessMetricRepository.get_by_kpi),
        ("BusinessMetricRepository.get_by_region", BusinessMetricRepository.get_by_region),
        # #931: paged per-region reader for benchmark means — provenance-enforced.
        (
            "BusinessMetricRepository.get_by_region_paged",
            BusinessMetricRepository.get_by_region_paged,
        ),
        ("TriggerRepository.get_recent_triggers", TriggerRepository.get_recent_triggers),
        ("TriggerRepository.get_by_patient", TriggerRepository.get_by_patient),
        # PatientJourneyRepository's explicit-query readers carry the kwarg
        # directly; its dict-path readers (get_by_brand) enforce via the inherited
        # get_many + HAS_PROVENANCE (asserted separately below).
        (
            "PatientJourneyRepository.get_data_freshness",
            PatientJourneyRepository.get_data_freshness,
        ),
        (
            "PatientJourneyRepository.get_freshness_by_source",
            PatientJourneyRepository.get_freshness_by_source,
        ),
        ("ExperimentOutcomeRepository.load_arrays", ExperimentOutcomeRepository.load_arrays),
        ("MLDataLoader.load_table_sample", MLDataLoader.load_table_sample),
        ("MLDataLoader._load_date_range", MLDataLoader._load_date_range),
        ("kpi_resolution._fetch_df", kpi_resolution._fetch_df),
        (
            "cohort_resolution._resolve_via_patient_journeys",
            cohort_resolution._resolve_via_patient_journeys,
        ),
        # PredictionRepository over ml_predictions (Shard 07 HIGH-2): every
        # actionable read default-excludes synthetic with an opt-in kwarg.
        ("PredictionRepository.get_by_model", PredictionRepository.get_by_model),
        ("PredictionRepository.get_top_predictions", PredictionRepository.get_top_predictions),
        ("PredictionRepository.get_model_performance", PredictionRepository.get_model_performance),
        ("PredictionRepository.get_by_patient", PredictionRepository.get_by_patient),
        (
            "PredictionRepository.get_high_confidence_predictions",
            PredictionRepository.get_high_confidence_predictions,
        ),
        (
            "PredictionRepository.get_calibration_summary",
            PredictionRepository.get_calibration_summary,
        ),
        # drift_monitor Supabase connector ml_predictions reads (Shard 07 HIGH-3).
        ("SupabaseDataConnector.query_predictions", SupabaseDataConnector.query_predictions),
        (
            "SupabaseDataConnector.query_labeled_predictions",
            SupabaseDataConnector.query_labeled_predictions,
        ),
        # User-facing episodic_memories ORM reads (Shard 07 HIGH-4).
        (
            "episodic_memory.search_episodic_by_e2i_entity",
            episodic_memory.search_episodic_by_e2i_entity,
        ),
        (
            "episodic_memory.get_enriched_episodic_memory",
            episodic_memory.get_enriched_episodic_memory,
        ),
        ("episodic_memory.get_recent_memories", episodic_memory.get_recent_memories),
        ("episodic_memory.get_memory_by_id", episodic_memory.get_memory_by_id),
        ("episodic_memory.count_memories_by_type", episodic_memory.count_memories_by_type),
    ]


def test_every_blast_radius_reader_accepts_include_synthetic():
    readers = _enforced_readers()
    assert len(readers) >= 27, "enumeration shrank — a reader was dropped"
    missing = []
    for name, fn in readers:
        params = inspect.signature(fn).parameters
        if "include_synthetic" not in params:
            missing.append(name)
    assert not missing, f"readers missing include_synthetic kwarg: {missing}"


def test_dict_path_readers_enforce_via_get_many_and_has_provenance():
    """Dict-filter readers (get_by_kpi/get_by_region/get_by_hcp/get_by_brand)
    enforce via the inherited BaseRepository.get_many predicate, gated on the
    per-repo HAS_PROVENANCE flag. Assert both halves of that contract."""
    from src.repositories.base import BaseRepository
    from src.repositories.business_metric import BusinessMetricRepository
    from src.repositories.patient_journey import PatientJourneyRepository
    from src.repositories.trigger import TriggerRepository

    # get_many is the SSOT dict-path predicate site and must accept the opt-in.
    assert "include_synthetic" in inspect.signature(BaseRepository.get_many).parameters

    # Every taggable repo must opt INTO the get_many predicate via HAS_PROVENANCE.
    for repo_cls in (
        BusinessMetricRepository,
        TriggerRepository,
        PatientJourneyRepository,
    ):
        assert repo_cls.HAS_PROVENANCE is True, (
            f"{repo_cls.__name__}.HAS_PROVENANCE must be True for dict-path enforcement"
        )
    # And the default on the base must stay False so untagged tables are untouched.
    assert BaseRepository.HAS_PROVENANCE is False
