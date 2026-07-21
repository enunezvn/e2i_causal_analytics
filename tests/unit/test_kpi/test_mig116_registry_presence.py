"""Drift-lock for migration 116 (claims-nowcast lag-triangle registry, backlog #45).

The /{kpi_id}/history/nowcast endpoint resolves these query_ids at runtime; if
the migration ever stops registering one the endpoint would 500 in prod. Pure
file parse, no DB — mirrors test_mig111_registry_presence.py.

Migration 116 registers ONLY the live-compute triangle queries. The DDL for the
arrival-plane columns (claim_available_date / adjudication_lag_days) is
migration 115 (PR-A) — 116 must not duplicate it.
"""

import re
from pathlib import Path

from src.kpi.synthetic_mode import nowcast_triangle_query_id

MIG = Path(__file__).resolve().parents[3] / "database/migrations/116_kpi_nowcast_registry.sql"

_FAMILIES = ("business_impact_trx", "business_impact_nrx", "business_impact_nbrx")


def _expected_ids():
    return [
        f"{family}_nowcast_triangle{syn}"
        for family in _FAMILIES
        for syn in ("", "_include_synthetic")
    ]


def test_all_6_query_ids_registered():
    sql = MIG.read_text()
    expected = _expected_ids()
    assert len(expected) == 6
    for qid in expected:
        assert f"('{qid}'," in sql, f"migration 116 missing query_id {qid}"


def test_triangle_reads_only_the_new_arrival_column():
    """The additive-only proof (design item 2): the ONLY registry SQL allowed to
    read claim_available_date is the explicitly-provisional nowcast triangle.
    Every statement here must read it; none may re-apply an as-of mask to
    event_date (the migration-113 falsified pattern)."""
    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", MIG.read_text(), re.S)
    assert len(bodies) == 6
    for body in bodies:
        assert "claim_available_date" in body
        # Month bucketing on event_date is fine; an event_date <= comparison
        # would be the falsified as-of mask.
        assert "event_date <=" not in body
        assert "event_date <" not in body


def test_plain_variants_are_synthetic_excluding_twins_are_not():
    sql = MIG.read_text()
    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", sql, re.S)
    ids = re.findall(r"\('([a-z_]+)',\s*\$kpi\$", sql)
    assert len(ids) == 6 and len(bodies) == 6
    for qid, body in zip(ids, bodies, strict=True):
        if qid.endswith("_include_synthetic"):
            assert "is_synthetic = false" not in body, qid
        else:
            assert "is_synthetic = false" in body, qid


def test_no_ddl_and_no_transaction_wrappers():
    """116 is registry-only: columns are migration 115's DDL (parallel PR-A file);
    migrations must not carry BEGIN/COMMIT (project policy: the runner wraps)."""
    sql = MIG.read_text().lower()
    assert "alter table" not in sql
    assert "add column" not in sql
    assert "begin;" not in sql
    assert "commit;" not in sql


def test_idempotent_upsert_and_schema_reload():
    sql = MIG.read_text()
    assert "ON CONFLICT (query_id) DO UPDATE" in sql
    assert "NOTIFY pgrst" in sql


def test_nbrx_joins_on_patient_id_not_journey_id():
    """The #1208 gotcha: patient-level subqueries MUST join on patient_id
    (patient_journey_id is NULL on ~17% of prescriptions)."""
    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", MIG.read_text(), re.S)
    joined = "\n".join(bodies)
    assert "patient_journey_id" not in joined


class TestNowcastTriangleQueryId:
    def test_plain(self, monkeypatch):
        monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
        monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
        assert (
            nowcast_triangle_query_id("business_impact_trx")
            == "business_impact_trx_nowcast_triangle"
        )

    def test_synthetic_flag_appends_suffix(self, monkeypatch):
        monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
        assert (
            nowcast_triangle_query_id("business_impact_nbrx")
            == "business_impact_nbrx_nowcast_triangle_include_synthetic"
        )
