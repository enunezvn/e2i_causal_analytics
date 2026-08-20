"""Drift-lock for migration 130 (cohort_profiler volume-tier allowlist, #1736).

The cohort_profiler agent resolves these query_ids at runtime for the HCP
volume-tier segmentation (eval 4.3: "Segment HCPs by prescription volume into
high, medium, and low tiers"); if the migration ever stops registering one,
that ask would fail closed in prod for the wrong reason. Pure file parse, no DB
— mirrors test_mig117_registry_presence.py.
"""

import re
from pathlib import Path

from src.agents.cohort_profiler.agent import (
    _HCP_VOLUME_TIER_QUERY_ID,
    _HCP_VOLUME_TIER_REGION_QUERY_ID,
    _profiler_query_id,
)

MIG = (
    Path(__file__).resolve().parents[3]
    / "database/migrations/130_cohort_profiler_hcp_volume_tiers.sql"
)

_FAMILIES = (
    _HCP_VOLUME_TIER_QUERY_ID,
    _HCP_VOLUME_TIER_REGION_QUERY_ID,
)


def _expected_ids():
    return [f"{family}{syn}" for family in _FAMILIES for syn in ("", "_include_synthetic")]


def test_all_4_query_ids_registered():
    sql = MIG.read_text()
    expected = _expected_ids()
    assert len(expected) == 4
    for qid in expected:
        assert f"('{qid}'," in sql, f"migration 130 missing query_id {qid}"


def test_plain_variants_are_synthetic_excluding_twins_are_not():
    sql = MIG.read_text()
    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", sql, re.S)
    ids = re.findall(r"\('([a-z0-9_]+)',\s*\$kpi\$", sql)
    assert len(ids) == 4 and len(bodies) == 4
    for qid, body in zip(ids, bodies, strict=True):
        if qid.endswith("_include_synthetic"):
            assert "is_synthetic = false" not in body, qid
        else:
            assert "is_synthetic = false" in body, qid


def test_tier_statements_use_trx_substrate_and_scope_relative_terciles():
    """Every tier statement must (a) aggregate the SAME substrate as the
    platform TRx KPI (treatment_events prescription rows), (b) bind brand ($1),
    half-open window ($2/$3) and exclusive threshold ($4), and (c) compute the
    tercile cut points WITHIN the queried scope (percentile_disc over the
    scoped cohort — measured 2026-08-19: the northeast cohort's cuts are 1/5
    while the global cohort's are 2/5, so a global-cut implementation would
    misassign tiers on every scoped ask)."""
    sql = MIG.read_text()
    bodies = re.findall(r"\$kpi\$(.*?)\$kpi\$", sql, re.S)
    assert len(bodies) == 4
    for body in bodies:
        assert "treatment_events" in body
        assert "'prescription'" in body
        assert "$1" in body and "$2::date" in body and "$3::date" in body
        assert "COUNT(*) > $4::int" in body
        assert body.count("percentile_disc") == 2
        assert "volume_tier" in body
    # Region variants additionally bind $5 case-insensitively, INSIDE the
    # scope that feeds the percentile cuts.
    region_bodies = [
        b
        for qid, b in zip(re.findall(r"\('([a-z0-9_]+)',\s*\$kpi\$", sql), bodies, strict=True)
        if "_region" in qid
    ]
    assert len(region_bodies) == 2
    for body in region_bodies:
        assert "LOWER($5)" in body
        scoped_pos = body.index("$5")
        cuts_pos = body.index("percentile_disc")
        assert scoped_pos < cuts_pos, "region filter must precede the tercile cuts"


def test_agent_resolves_synthetic_variant_under_flag(monkeypatch):
    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    assert _profiler_query_id(_HCP_VOLUME_TIER_QUERY_ID) == _HCP_VOLUME_TIER_QUERY_ID
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    assert (
        _profiler_query_id(_HCP_VOLUME_TIER_QUERY_ID)
        == f"{_HCP_VOLUME_TIER_QUERY_ID}_include_synthetic"
    )
