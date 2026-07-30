"""Drift-lock for migration 117 (cohort_profiler ask-bound allowlist, #1356).

The cohort_profiler agent resolves these query_ids at runtime for the
criteria-bound patient profile and the HCP-entity TRx-threshold cohort; if the
migration ever stops registering one, those asks would fail closed in prod for
the wrong reason. Pure file parse, no DB — mirrors
test_mig116_registry_presence.py.
"""

import re
from pathlib import Path

from src.agents.cohort_profiler.agent import (
    _HCP_COHORT_QUERY_ID,
    _PATIENT_CRITERIA_QUERY_ID,
    _PATIENT_CRITERIA_WINDOWED_QUERY_ID,
    _profiler_query_id,
)

MIG = (
    Path(__file__).resolve().parents[3]
    / "database/migrations/117_cohort_profiler_ask_bound_queries.sql"
)

_FAMILIES = (
    _HCP_COHORT_QUERY_ID,
    _PATIENT_CRITERIA_QUERY_ID,
    _PATIENT_CRITERIA_WINDOWED_QUERY_ID,
)


def _expected_ids():
    return [f"{family}{syn}" for family in _FAMILIES for syn in ("", "_include_synthetic")]


def test_all_6_query_ids_registered():
    sql = MIG.read_text()
    expected = _expected_ids()
    assert len(expected) == 6
    for qid in expected:
        assert f"('{qid}'," in sql, f"migration 117 missing query_id {qid}"


def test_plain_variants_are_synthetic_excluding_twins_are_not():
    sql = MIG.read_text()
    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", sql, re.S)
    ids = re.findall(r"\('([a-z0-9_]+)',\s*\$kpi\$", sql)
    assert len(ids) == 6 and len(bodies) == 6
    for qid, body in zip(ids, bodies, strict=True):
        if qid.endswith("_include_synthetic"):
            assert "is_synthetic = false" not in body, qid
        else:
            assert "is_synthetic = false" in body, qid


def test_hcp_cohort_uses_trx_kpi_substrate_and_binds_all_params():
    """The HCP cohort must aggregate the SAME substrate as the platform TRx KPI
    (treatment_events prescription rows) and bind every ask parameter: brand
    ($1), half-open window ($2/$3), exclusive threshold ($4)."""
    sql = MIG.read_text()
    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", sql, re.S)
    hcp_bodies = bodies[:2]
    for body in hcp_bodies:
        assert "event_type::text = 'prescription'" in body
        assert "hcp_id IS NOT NULL" in body
        assert "$2::date" in body and "$3::date" in body
        assert "COUNT(*) > $4::int" in body
        assert "($1::text IS NULL OR" in body
        assert "hcp_profiles" in body  # segment axes come from the profile join
        assert "priority_tier" in body and "specialty" in body


def test_patient_criteria_binds_age_at_diagnosis_and_joins_on_patient_id():
    """Age bounds bind to patient_journeys.age_at_diagnosis (populated on all
    rows — verified READ-ONLY 2026-07-30) and the join is on patient_id, never
    patient_journey_id (the #1208 gotcha: NULL on ~17% of prescriptions).
    The windowed sibling (codex iter-2) binds the explicit [$2,$3) window with
    $4 = exclusive min age (no max age — the RPC's 4-param cap)."""
    sql = MIG.read_text()
    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", sql, re.S)
    ids = re.findall(r"\('([a-z0-9_]+)',\s*\$kpi\$", sql)
    patient = [
        (qid, body)
        for qid, body in zip(ids, bodies, strict=True)
        if qid.startswith(_PATIENT_CRITERIA_QUERY_ID)
    ]
    assert len(patient) == 4
    for qid, body in patient:
        assert "pj.patient_id = te.patient_id" in body, qid
        assert "patient_journey_id" not in body, qid
        assert "sequence_number = 1" in body, qid  # NRx, mirroring mig-105
        if "_windowed" in qid:
            assert "event_date >= $2::date" in body and "event_date < $3::date" in body, qid
            assert "age_at_diagnosis > $4::int" in body, qid
            assert "age_at_diagnosis <" not in body, qid  # max-age cannot bind here
        else:
            assert "age_at_diagnosis > $2::int" in body, qid
            assert "age_at_diagnosis < $3::int" in body, qid


def test_no_ddl_and_no_transaction_wrappers():
    sql = MIG.read_text().lower()
    assert "alter table" not in sql
    assert "add column" not in sql
    assert "begin;" not in sql
    assert "commit;" not in sql


def test_idempotent_upsert_and_schema_reload():
    sql = MIG.read_text()
    assert "ON CONFLICT (query_id) DO UPDATE" in sql
    assert "NOTIFY pgrst" in sql


class TestProfilerQueryId:
    """_profiler_query_id follows the ADDITIVE-variant idiom: absent from
    SYNTHETIC_TWINNED_QUERY_IDS, self-suffixing under the showcase flag."""

    def test_plain(self, monkeypatch):
        monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
        monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
        assert _profiler_query_id(_HCP_COHORT_QUERY_ID) == "cohort_profiler_hcp_trx_cohort"

    def test_synthetic_flag_appends_suffix(self, monkeypatch):
        monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
        assert (
            _profiler_query_id(_PATIENT_CRITERIA_QUERY_ID)
            == "cohort_profiler_patient_criteria_profile_include_synthetic"
        )

    def test_ids_stay_out_of_the_locked_twin_registry(self):
        from src.kpi.synthetic_mode import SYNTHETIC_TWINNED_QUERY_IDS

        for family in _FAMILIES:
            assert family not in SYNTHETIC_TWINNED_QUERY_IDS
