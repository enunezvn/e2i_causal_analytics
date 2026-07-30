"""Migration 118 content lock — trigger-effectiveness KPI statements (#1360).

The #1360 ruling (2026-07-30, owner-decided): the four trigger-effectiveness
KPIs — trigger precision (WS2-TR-001), acceptance rate (WS2-TR-004), override
rate (WS2-TR-006) and the new trigger funnel conversion (WS2-TR-009) — live in
the CHAT KPI PATH: registered in ``kpi_query_registry`` and served by
``kpi_calculate_tool``. Migration 118 adds the ask-bound statement families
(mirroring the 117 cohort_profiler idiom):

  * ``trigger_effectiveness_<metric>``            3 params: $1 brand, $2 region,
    $3 trigger_type (ALL nullable), frontier-anchored default window.
  * ``trigger_effectiveness_<metric>_windowed``   4 params: $1 brand,
    $2 trigger_type (nullable), $3/$4 half-open [start, end) window.
    Region can NOT ride along (the kpi_query RPC binds at most 4 positional
    params) — the calculator fails closed on region+window instead of
    silently dropping the region.

Every id has an ``_include_synthetic`` twin (the live path on this substrate:
prod sets E2I_KPI_INCLUDE_SYNTHETIC=true and all 37k trigger rows are
synthetic). These text-level locks keep the statement content from silently
reverting: the v2 precision definition (migration 113), the delivered
denominator (090/092), frontier anchoring (089), and the funnel stage columns.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = REPO_ROOT / "database" / "migrations" / "118_trigger_effectiveness_kpis.sql"

METRICS = ["precision", "acceptance_rate", "override_rate", "funnel_conversion"]

BASE_IDS = [f"trigger_effectiveness_{m}" for m in METRICS]
WINDOWED_IDS = [f"{qid}_windowed" for qid in BASE_IDS]
ALL_IDS = [
    variant for qid in BASE_IDS + WINDOWED_IDS for variant in (qid, f"{qid}_include_synthetic")
]

DELIVERED_DENOMINATOR = "delivery_status IN ('delivered', 'viewed')"


def _registration_lines(content: str) -> dict[str, str]:
    """Map each query_id to its VALUES registration line in the migration."""
    lines: dict[str, str] = {}
    for line in content.splitlines():
        for qid in ALL_IDS:
            if f"('{qid}'" in line:
                lines[qid] = line
    return lines


def _kpi_sql(line: str) -> str:
    """Extract the $kpi$-delimited SQL body from a registration line."""
    start = line.index("$kpi$") + len("$kpi$")
    end = line.index("$kpi$", start)
    return line[start:end]


def test_migration_file_exists():
    assert MIGRATION.exists(), f"missing migration: {MIGRATION.name}"


def test_all_sixteen_statements_registered():
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    missing = [qid for qid in ALL_IDS if qid not in lines]
    assert not missing, f"trigger-effectiveness statements not registered: {missing}"


def test_param_arity_matches_the_declared_contract():
    """Base families bind 3 nullable params; windowed families bind 4."""
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        expected = 4 if "_windowed" in qid else 3
        assert f"$kpi$, {expected}," in line, f"{qid}: max_params must be {expected}"


def test_base_statements_are_frontier_anchored_with_data_through():
    """Non-windowed forms keep the 089 contract: window anchored at
    MAX(trigger_timestamp) (never NOW()) + a data_through provenance column."""
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        sql = _kpi_sql(line)
        assert "NOW()" not in sql, f"{qid}: NOW()-anchored window (089 regression)"
        if "_windowed" not in qid:
            assert "MAX(trigger_timestamp)" in sql, f"{qid}: frontier anchor dropped"
            assert "data_through" in sql, f"{qid}: data_through provenance dropped"


def test_windowed_statements_bind_half_open_window_without_data_through():
    """Windowed forms bind the ask's explicit [start, end) on trigger_timestamp
    and drop data_through (the window is explicit — mirroring the WS3 _windowed
    idiom where the provenance column is only for engine-default windows)."""
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        if "_windowed" not in qid:
            continue
        sql = _kpi_sql(line)
        assert "trigger_timestamp >= $3::timestamptz" in sql, f"{qid}: window start"
        assert "trigger_timestamp < $4::timestamptz" in sql, f"{qid}: half-open window end"
        assert "data_through" not in sql, f"{qid}: data_through must not ride on windowed forms"


def test_precision_keeps_the_v2_truth_definition():
    """Migration 113's v2 precision — accepted-and-converted / accepted-and-
    tracked — must not revert to the v1 outcome-tracking-coin form."""
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    for qid in [i for i in ALL_IDS if "precision" in i]:
        sql = _kpi_sql(lines[qid])
        assert "acceptance_status = 'accepted' AND outcome_tracked AND outcome_value > 0" in sql
        assert "acceptance_status = 'accepted' AND outcome_tracked" in sql


def test_precision_base_keeps_the_matured_lagged_window():
    """The non-windowed precision cohort must stay [frontier-60d, frontier-30d]
    so the 30-day conversion window has elapsed (migration 113)."""
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    for qid in [
        "trigger_effectiveness_precision",
        "trigger_effectiveness_precision_include_synthetic",
    ]:
        sql = _kpi_sql(lines[qid])
        assert "INTERVAL '60 days'" in sql, f"{qid}: lagged cohort start dropped"
        assert "INTERVAL '30 days'" in sql, f"{qid}: lagged cohort end dropped"


def test_acceptance_and_override_keep_the_delivered_denominator():
    """Migrations 090/092: denominator = delivered triggers, numerators =
    accepted / overridden respectively."""
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    for qid in [i for i in ALL_IDS if "acceptance_rate" in i]:
        assert DELIVERED_DENOMINATOR in lines[qid], f"{qid}: delivered denominator lost"
        assert "acceptance_status = 'accepted'" in lines[qid], f"{qid}: numerator lost"
        assert "acceptance_status IS NOT NULL" not in lines[qid], f"{qid}: pre-092 denominator"
    for qid in [i for i in ALL_IDS if "override_rate" in i]:
        assert DELIVERED_DENOMINATOR in lines[qid], f"{qid}: delivered denominator lost"
        assert "acceptance_status = 'overridden'" in lines[qid], f"{qid}: numerator lost"


def test_funnel_carries_all_stage_counts_and_the_actioned_headline():
    """The funnel statement must return the stage counts (delivered -> viewed ->
    accepted -> actioned -> outcome) plus the headline rate = actioned/delivered.
    The headline deliberately stops at actioned: extending to outcome would
    conflate outcome-TRACKING coverage with effectiveness (the v1 precision
    trap)."""
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    for qid in [i for i in ALL_IDS if "funnel_conversion" in i]:
        sql = _kpi_sql(lines[qid])
        for col in ("n_delivered", "n_viewed", "n_accepted", "n_actioned", "n_outcome"):
            assert col in sql, f"{qid}: funnel stage column {col} missing"
        assert "funnel_conversion" in sql, f"{qid}: headline rate column missing"
        assert "n_actioned::float / NULLIF" in sql, f"{qid}: headline must be actioned/delivered"


def test_ask_params_are_nullable_filters():
    """$1 brand / $2 region / $3 trigger_type (base) and $1 brand / $2
    trigger_type (windowed) must be optional — NULL means no filter."""
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        sql = _kpi_sql(line)
        assert "$1::text IS NULL OR brand_id::text = $1" in sql, f"{qid}: brand not nullable"
        if "_windowed" in qid:
            assert "$2::text IS NULL OR trigger_type::text = $2" in sql, f"{qid}: trigger_type"
        else:
            assert "$3::text IS NULL OR trigger_type::text = $3" in sql, f"{qid}: trigger_type"
            assert "$2::text IS NULL OR" in sql and "geographic_region" in sql, (
                f"{qid}: region must filter via the patient_journeys join (triggers has no region)"
            )


def test_synthetic_gating_follows_the_additive_twin_idiom():
    """Base ids default-exclude synthetic rows; _include_synthetic twins are the
    unwrapped originals (mig 066/117 idiom)."""
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        sql = _kpi_sql(line)
        if qid.endswith("_include_synthetic"):
            assert "is_synthetic = false" not in sql, f"{qid}: twin must include synthetic"
        else:
            assert "is_synthetic = false" in sql, f"{qid}: base must exclude synthetic"


def test_idempotent_upsert_and_postgrest_reload():
    content = MIGRATION.read_text()
    assert "ON CONFLICT (query_id) DO UPDATE" in content
    assert "NOTIFY pgrst" in content
