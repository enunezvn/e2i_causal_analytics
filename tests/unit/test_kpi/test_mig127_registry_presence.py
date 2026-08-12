"""Drift-lock for migration 127 (brand_specific region variants, #1564).

BrandSpecificCalculator resolves these query_ids at runtime; if the migration
ever stops registering one, a region+brand chat ask would 500 in prod. Pure
file parse, no DB — mirrors test_mig111_registry_presence.py.

Also pins the correctness invariants the SQL must keep:

* patient-region joins go through ``patient_id`` (``patient_journeys`` is 1:1
  on it), NEVER ``treatment_events.patient_journey_id`` — that FK is NULL on
  ~45% of NRx events and silently drops rows (#1208, the mig-077 defect
  corrected in mig 105/111);
* region match is case-insensitive ``LOWER(geographic_region::text) = LOWER($n)``
  (mig 077/078 idiom);
* no ``NOW()`` anywhere — the windowed statements (BR-002 fallback, BR-005)
  stay frontier-anchored to the GLOBAL domain MAX exactly like their mig-089
  bases (a per-region frontier would shift maturation cutoffs; the backfill
  precedent keeps cutoffs global);
* synthetic gate parity: base variants wrap taggable tables in
  ``is_synthetic = false`` subselects, ``_include_synthetic`` twins do not;
* BR-003 keeps the #1116 structural-zero guard column ``pnh_events_total``
  TABLE-WIDE (substrate coverage is not a per-region fact).
"""

import re
from pathlib import Path

MIG = (
    Path(__file__).resolve().parents[3]
    / "database/migrations/127_kpi_brand_specific_region_variants.sql"
)

BASES = [
    "brand_specific_remi_ah_uncontrolled",
    "brand_specific_remi_intent_delta_primary",
    "brand_specific_remi_intent_delta_fallback",
    "brand_specific_fabhalta_pnh_tested",
    "brand_specific_kisqali_dx_adoption",
    "brand_specific_kisqali_oncologist_reach",
]


def _rows():
    """[(query_id, sql_body, max_params)] parsed from the INSERT statement."""
    text = MIG.read_text()
    return [
        (qid, body, int(n))
        for qid, body, n in re.findall(
            r"\('([a-z0-9_]+)',\s*\$kpi\$(.*?)\$kpi\$,\s*(\d+),", text, re.S
        )
    ]


def test_all_12_query_ids_registered():
    ids = {qid for qid, _, _ in _rows()}
    expected = {f"{base}_region{syn}" for base in BASES for syn in ("", "_include_synthetic")}
    assert len(expected) == 12
    missing = expected - ids
    assert not missing, f"migration 127 missing query_ids: {sorted(missing)}"


def test_region_predicate_is_case_insensitive_everywhere():
    for qid, body, _ in _rows():
        assert re.search(r"LOWER\((?:\w+\.)?geographic_region::text\) = LOWER\(\$\d\)", body), (
            f"{qid}: region predicate must be LOWER(geographic_region::text) = LOWER($n)"
        )


def test_joins_on_patient_id_never_patient_journey_id():
    """#1208: patient_journey_id is NULL on ~45% of NRx events — a join on it
    silently under-counts. Every patient-region membership must use patient_id."""
    for qid, body, _ in _rows():
        assert "patient_journey_id" not in body, f"{qid}: uses the defective FK join"


def test_no_wall_clock_windows():
    """Frontier anchoring (mig 089) must survive into the region variants."""
    for qid, body, _ in _rows():
        assert "NOW()" not in body, f"{qid}: NOW() re-introduces the pre-089 drift"


def test_max_params():
    """BR-001 keeps its $1 UAS7 threshold, region appended as $2; all other
    variants take region as their only param."""
    by_id = {qid: n for qid, _, n in _rows()}
    for base in BASES:
        expected_n = 2 if base == "brand_specific_remi_ah_uncontrolled" else 1
        for syn in ("", "_include_synthetic"):
            assert by_id[f"{base}_region{syn}"] == expected_n


def test_synthetic_gate_parity():
    for qid, body, _ in _rows():
        if qid.endswith("_include_synthetic"):
            assert "is_synthetic" not in body, f"{qid}: twin must NOT exclude synthetic"
        else:
            assert "is_synthetic = false" in body, f"{qid}: base must exclude synthetic"


def test_br003_keeps_table_wide_structural_zero_guard():
    """#1116: pnh_events_total is the TABLE-WIDE pnh_flow_cytometry count — the
    guard distinguishes 'concept never recorded in the substrate' from a
    genuine regional 0%. Its subselect must NOT be region-filtered."""
    for qid, body, _ in _rows():
        if "fabhalta_pnh_tested" not in qid:
            continue
        assert "pnh_events_total" in body, f"{qid}: guard column missing"
        guard = re.search(
            r"\(SELECT COUNT\(\*\) FROM [^)]*treatment_events[^)]*"
            r"event_subtype = 'pnh_flow_cytometry'[^)]*\)",
            body,
        )
        assert guard is not None, f"{qid}: table-wide guard subselect missing"
        assert "geographic_region" not in guard.group(0), (
            f"{qid}: guard must stay table-wide, not region-filtered"
        )


def test_br005_filters_both_denominator_and_numerator():
    """Oncologist reach in a region = engaged-in-region / oncologists-in-region.
    Filtering only one CTE would mix scopes into an incoherent ratio."""
    for qid, body, _ in _rows():
        if "oncologist_reach" not in qid:
            continue
        assert len(re.findall(r"LOWER\((?:\w+\.)?geographic_region::text\)", body)) >= 2, (
            f"{qid}: both the oncologists and engaged CTEs must be region-filtered"
        )


def test_br002_variants_join_hcp_profiles_for_region():
    """Intent surveys carry no region; the region is the surveyed HCP's
    (hcp_intent_surveys.hcp_id -> hcp_profiles.geographic_region)."""
    for qid, body, _ in _rows():
        if "intent_delta" not in qid:
            continue
        assert "hcp_profiles" in body, f"{qid}: must join hcp_profiles for region"
        assert "hcp_intent_surveys" in body, f"{qid}: must read hcp_intent_surveys"


def test_upsert_is_idempotent():
    assert "ON CONFLICT (query_id) DO UPDATE" in MIG.read_text()
