"""Drift-lock for migration 108 (biologic / IgE-tertile KPI variants).

The business_impact calculator resolves these query_ids at runtime; if the
migration ever stops registering one (rename, typo, dropped row) the KPI call
would 500 in prod. This test asserts the migration FILE registers all 28 ids
and keeps the faithfulness-critical SQL shape (patient_id join, empirical IgE
cutpoints, brand-agnostic predicate) -- a pure file parse, no DB.
"""

from pathlib import Path

MIG = Path(__file__).resolve().parents[3] / "database/migrations/108_kpi_biologic_ige_variants.sql"


def _expected_ids():
    ids = []
    for axis in ("biologic", "ige_tier"):
        for kpi in ("trx", "nrx", "nbrx", "trx_share"):
            for syn in ("", "_include_synthetic"):
                ids.append(f"business_impact_{kpi}_{axis}{syn}")
        # windowed: trx/nrx/nbrx only (share has no windowed variant, mig 105 parity)
        for kpi in ("trx", "nrx", "nbrx"):
            for syn in ("", "_include_synthetic"):
                ids.append(f"business_impact_{kpi}_{axis}_windowed{syn}")
    return ids


def test_all_28_query_ids_registered():
    sql = MIG.read_text()
    expected = _expected_ids()
    assert len(expected) == 28
    for qid in expected:
        assert f"('{qid}'," in sql, f"migration 108 missing query_id {qid}"


def test_predicates_and_cutpoints_are_faithful():
    sql = MIG.read_text()
    # biologic axis maps the smallint to semantic buckets, guarded on NOT NULL.
    assert "biologic_experienced IS NOT NULL" in sql
    assert "WHEN biologic_experienced = 1 THEN 'experienced' ELSE 'naive'" in sql
    # IgE tertile uses the EMPIRICAL p33/p66 cutpoints, guarded on NOT NULL.
    assert "ige_level IS NOT NULL" in sql
    assert "ige_level < 105.21" in sql
    assert "ige_level < 208.50" in sql


def test_joins_on_patient_id_not_journey_id():
    """The #1208 gotcha: NRx events have NULL patient_journey_id ~45% of the time,
    so the axis subquery MUST join on patient_id or buckets under-count. Checks the
    SQL BODIES (between $kpi$ markers) only -- the header comment legitimately
    explains the NULL-pjid rationale, so a whole-file scan would false-positive."""
    import re

    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", MIG.read_text(), re.S)
    assert len(bodies) == 28
    joined = "\n".join(bodies)
    assert "patient_id IN (SELECT patient_id FROM" in joined
    assert "patient_journey_id" not in joined


def test_no_transaction_wrappers():
    """Migrations must not carry BEGIN/COMMIT (project policy: the runner wraps)."""
    sql = MIG.read_text().lower()
    assert "begin;" not in sql
    assert "commit;" not in sql
