"""Drift-lock for migration 144: the patient-axis TRx-share statements are deleted.

The calculator refuses TRx share on every patient axis (chat
session_1789548670222_fcscf3u, 2026-09-16), so these 12 registry rows are
unreachable. Pure file parse, no DB -- mirrors test_mig111_registry_presence.py.
The delete names every id explicitly: a LIKE pattern would also match the brand-
level `business_impact_trx_share_windowed*` and `_region*` rows that stay live.
"""

import re
from pathlib import Path

MIG = (
    Path(__file__).resolve().parents[3]
    / "database/migrations/144_drop_trx_share_patient_axis_variants.sql"
)


def _expected_ids() -> set:
    ids = set()
    for axis in ("segment", "line"):
        for windowed in ("", "_windowed"):
            for syn in ("", "_include_synthetic"):
                ids.add(f"business_impact_trx_share_{axis}{windowed}{syn}")
    for axis in ("biologic", "ige_tier"):
        for syn in ("", "_include_synthetic"):
            ids.add(f"business_impact_trx_share_{axis}{syn}")
    return ids


def _deleted_ids() -> set:
    sql = MIG.read_text()
    m = re.search(r"DELETE FROM kpi_query_registry\s+WHERE query_id IN \((.*?)\);", sql, re.S)
    assert m, "migration 144 must DELETE FROM kpi_query_registry WHERE query_id IN (...)"
    return set(re.findall(r"'([a-z_]+)'", m.group(1)))


def test_deletes_exactly_the_12_patient_axis_share_ids():
    expected = _expected_ids()
    assert len(expected) == 12
    assert _deleted_ids() == expected


def test_brand_level_share_statements_are_not_deleted():
    deleted = _deleted_ids()
    for keep in (
        "business_impact_trx_share",
        "business_impact_trx_share_include_synthetic",
        "business_impact_trx_share_windowed",
        "business_impact_trx_share_windowed_include_synthetic",
        "business_impact_trx_share_region",
        "business_impact_trx_share_region_include_synthetic",
    ):
        assert keep not in deleted


def test_no_like_pattern_and_no_transaction_wrappers():
    body = "\n".join(
        ln for ln in MIG.read_text().splitlines() if not ln.lstrip().startswith("--")
    ).lower()
    assert " like " not in body
    assert "begin;" not in body
    assert "commit;" not in body


def test_the_calculator_can_no_longer_resolve_any_deleted_id(monkeypatch):
    """Every share context the calculator accepts resolves to a KEPT id."""
    from src.kpi.calculators.business_impact import BusinessImpactCalculator

    deleted = _deleted_ids()
    resolved = []

    class _Client:
        def rpc(self, name, payload):
            resolved.append(payload["query_id"])
            return type("E", (), {"execute": lambda s: type("R", (), {"data": []})()})()

    window = {"start": "S", "end": "E"}
    axes = [
        {},
        {"region": "northeast"},
        {"segment": "high_severity"},
        {"therapy_line": 0},
        {"biologic": "naive"},
        {"ige_tier": "high"},
    ]
    for syn in ("0", "1"):
        monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", syn)
        for axis in axes:
            for w in (None, window):
                ctx = {"brand": "Remibrutinib", **axis}
                if w:
                    ctx["window"] = w
                try:
                    BusinessImpactCalculator(db_client=_Client())._calc_trx_share(ctx)
                except RuntimeError:
                    pass
    assert resolved, "positive control: the brand-level reads must still issue queries"
    assert not deleted & set(resolved)
