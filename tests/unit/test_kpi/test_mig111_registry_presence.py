"""Drift-lock for migration 111 (conversion-rate / TRx-share axis+window variants).

The business_impact calculator resolves these query_ids at runtime; if the
migration ever stops registering one the KPI call would 500 in prod. Pure file
parse, no DB — mirrors test_mig108_registry_presence.py.
"""

from pathlib import Path

MIG = (
    Path(__file__).resolve().parents[3]
    / "database/migrations/111_kpi_conversion_share_axis_window.sql"
)


def _expected_ids():
    ids = []
    for variant in ("brand", "segment", "line", "windowed", "segment_windowed", "line_windowed"):
        for syn in ("", "_include_synthetic"):
            ids.append(f"business_impact_conversion_rate_{variant}{syn}")
    for variant in ("windowed", "segment_windowed", "line_windowed"):
        for syn in ("", "_include_synthetic"):
            ids.append(f"business_impact_trx_share_{variant}{syn}")
    return ids


def test_all_18_query_ids_registered():
    sql = MIG.read_text()
    expected = _expected_ids()
    assert len(expected) == 18
    for qid in expected:
        assert f"('{qid}'," in sql, f"migration 111 missing query_id {qid}"


def test_conversion_brand_scoping_is_same_brand_and_null_tolerant():
    """Brand-scoped conversion counts SAME-brand trigger->Rx pairs; NULL brand
    must reduce to the certified base semantics (validated equal to the base
    statement, 0.6390964..., in the pre-merge dry-run)."""
    import re

    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", MIG.read_text(), re.S)
    assert len(bodies) == 18
    conversion_bodies = [b for b in bodies if "conversion_rate" in b]
    assert len(conversion_bodies) == 12
    for body in conversion_bodies:
        assert "($1::text IS NULL OR t.brand_id = $1)" in body
        assert "($1::text IS NULL OR te.brand::text = $1)" in body
        # The 30-day trigger->Rx horizon is the KPI definition; windows bound
        # WHICH triggers count, never the conversion horizon.
        assert "(t.trigger_timestamp + INTERVAL '30 days')::date" in body


def test_joins_on_patient_id_not_journey_id():
    """The #1208 gotcha: axis subqueries MUST join on patient_id."""
    import re

    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", MIG.read_text(), re.S)
    joined = "\n".join(bodies)
    assert "patient_id IN (SELECT patient_id FROM" in joined
    assert "patient_journey_id" not in joined


def test_no_transaction_wrappers():
    """Migrations must not carry BEGIN/COMMIT (project policy: the runner wraps)."""
    sql = MIG.read_text().lower()
    assert "begin;" not in sql
    assert "commit;" not in sql
