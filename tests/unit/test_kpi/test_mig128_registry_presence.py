"""Drift-lock for migration 128 (conversion-rate brand+region variants, #1575).

`_calc_conversion_rate` resolves ``business_impact_conversion_rate_brand_region``
at runtime once brand+region asks are routed; if the migration ever stops
registering it, a brand+region chat ask would 500 in prod. Pure file parse, no
DB — mirrors test_mig127_registry_presence.py / test_mig111_registry_presence.py.

Also pins the correctness invariants the SQL must keep (each measured from the
family's vetted statements, not invented here):

* SAME-BRAND conversion semantics (mig 111, dry-run-verified): the brand
  predicate must appear on the triggered CTE (``triggers.brand_id``), on the
  converted CTE's triggers (``t.brand_id``) AND on the converting prescription
  (``te.brand::text``) — dropping any leg silently reverts to any-brand
  conversion under a brand caption;
* the brand param stays NULL-tolerant ``($1::text IS NULL OR ...)`` — every
  mig-111 conversion leg is; with $1 NULL the statement reduces to the
  ``_region`` semantics, never to an error or an empty cut;
* region is patient MEMBERSHIP through ``patient_id`` on BOTH CTEs
  (``patient_journeys`` is 1:1 on it), NEVER
  ``treatment_events.patient_journey_id`` — that FK is NULL on ~45% of NRx
  events and silently drops rows (#1208, the mig-077 defect corrected in
  105/111);
* region match is case-insensitive ``LOWER(geographic_region::text) = LOWER($2)``
  (mig 077/078 idiom);
* no ``NOW()`` anywhere and the ``data_through`` wrapper present — new
  registry rows must be born frontier-anchored (mig 089; the anchoring
  generator's replay deliberately stops at 089, so post-089 rows carry the
  anchor themselves and this test IS the sync guarantee);
* synthetic gate parity: the base variant wraps taggable tables in
  ``is_synthetic = false`` subselects, the ``_include_synthetic`` twin does not
  (mig 066 generator invariant).
"""

import re
from pathlib import Path

MIG = (
    Path(__file__).resolve().parents[3]
    / "database/migrations/128_kpi_conversion_rate_brand_region.sql"
)

BASE = "business_impact_conversion_rate_brand_region"


def _rows():
    """[(query_id, sql_body, max_params)] parsed from the INSERT statement."""
    text = MIG.read_text()
    return [
        (qid, body, int(n))
        for qid, body, n in re.findall(
            r"\('([a-z0-9_]+)',\s*\$kpi\$(.*?)\$kpi\$,\s*(\d+),", text, re.S
        )
    ]


def test_both_query_ids_registered_with_two_params():
    by_id = {qid: n for qid, _, n in _rows()}
    expected = {BASE, f"{BASE}_include_synthetic"}
    missing = expected - set(by_id)
    assert not missing, f"migration 128 missing query_ids: {sorted(missing)}"
    for qid in expected:
        assert by_id[qid] == 2, f"{qid}: max_params must be 2 [brand, region]"


def test_same_brand_conversion_semantics_on_all_three_legs():
    """Mig-111 brand shape: trigger brand, converted-trigger brand, AND
    same-brand prescription. All three predicates NULL-tolerant on $1."""
    for qid, body, _ in _rows():
        assert "($1::text IS NULL OR brand_id = $1)" in body, (
            f"{qid}: triggered CTE must filter triggers.brand_id"
        )
        assert "($1::text IS NULL OR t.brand_id = $1)" in body, (
            f"{qid}: converted CTE must filter t.brand_id"
        )
        assert "($1::text IS NULL OR te.brand::text = $1)" in body, (
            f"{qid}: converting prescription must be SAME-brand (te.brand)"
        )


def test_region_predicate_is_case_insensitive_dollar2():
    for qid, body, _ in _rows():
        assert "LOWER(geographic_region::text) = LOWER($2)" in body, (
            f"{qid}: region must bind $2 via LOWER(geographic_region::text)"
        )


def test_region_membership_scopes_both_ctes():
    """Triggered AND converted must both cut to region patients — filtering
    only one CTE would mix scopes into an incoherent ratio."""
    for qid, body, _ in _rows():
        assert body.count("IN (SELECT patient_id FROM region_patients)") == 2, (
            f"{qid}: both triggered and converted CTEs must use region membership"
        )


def test_joins_on_patient_id_never_patient_journey_id():
    """#1208: patient_journey_id is NULL on ~45% of NRx events — a join on it
    silently under-counts. Every patient-region membership must use patient_id."""
    for qid, body, _ in _rows():
        assert "patient_journey_id" not in body, f"{qid}: uses the defective FK join"


def test_frontier_anchored_no_wall_clock():
    """Mig-089 contract for post-089 rows: born anchored (no NOW()), with the
    data_through provenance wrapper. The gen_kpi_frontier_anchoring replay
    stops at 089 by design, so this drift-lock IS the generator-sync check."""
    for qid, body, _ in _rows():
        assert "NOW()" not in body, f"{qid}: NOW() re-introduces the pre-089 decay"
        assert body.startswith("SELECT base.*, ("), qid
        assert ")::date AS data_through FROM (" in body, qid
        assert body.endswith(") base"), qid
        assert "MAX(trigger_timestamp)" in body, (
            f"{qid}: conversion windows anchor on the TRIGGERS frontier"
        )


def test_conversion_horizon_stays_30_days():
    """The trigger->Rx horizon is the KPI's definition (mig 111) — the region
    cut must not touch it."""
    for qid, body, _ in _rows():
        assert "te.event_date <= (t.trigger_timestamp + INTERVAL '30 days')::date" in body, qid


def test_synthetic_gate_parity():
    for qid, body, _ in _rows():
        if qid.endswith("_include_synthetic"):
            assert "is_synthetic" not in body, f"{qid}: twin must NOT exclude synthetic"
        else:
            assert "is_synthetic = false" in body, f"{qid}: base must exclude synthetic"


def test_readonly_check_shape():
    """The registry's CHECK constraint admits statements starting WITH/SELECT."""
    for qid, body, _ in _rows():
        assert body.lstrip().startswith(("SELECT", "WITH")), qid


def test_upsert_is_idempotent_and_reloads_postgrest():
    text = MIG.read_text()
    assert "ON CONFLICT (query_id) DO UPDATE" in text
    assert "NOTIFY pgrst, 'reload schema';" in text
