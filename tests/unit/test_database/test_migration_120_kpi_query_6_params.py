"""Migration 120 content lock — kpi_query 6-param unroll + regioned+windowed
trigger-effectiveness statements (#1388).

The 4-param positional cap in ``kpi_query()`` (migration 044:93-105) was a design
cap, not a Postgres limit — no registered statement needed more than 4 when it
was written. #1360's trigger-effectiveness ``_windowed`` variants (migration 118)
therefore had to DROP the region axis: brand + region + trigger_type +
window_start + window_end = 5 positional params, one over the cap. Migration 120:

  1. Extends the ``kpi_query()`` positional unroll to 6 params (``ELSIF n = 5`` /
     ``ELSIF n = 6``), keeping the arity check (``n <> expected_n``) and the
     registry-vetted allowlist model UNCHANGED — the RPC still runs only a stored
     statement, params still positionally bound, never client SQL.
  2. Registers the 5-param regioned+windowed trigger-effectiveness variants
     ``trigger_effectiveness_<metric>_windowed_region[_include_synthetic]`` — the
     migration-118 windowed SQL with region re-added ($2, via the
     patient_journeys join — triggers carry no region column, the 078/118 idiom)
     and the params shifted: $1 brand, $2 region, $3 trigger_type, $4/$5 half-open
     window on trigger_timestamp.

These text-level locks keep the unroll extension and the statement content from
silently reverting. The migration is validated by READING it (DB application is
deferred to batch-deploy — the local DB is prod).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = REPO_ROOT / "database" / "migrations" / "120_kpi_query_6_params.sql"

METRICS = ["precision", "acceptance_rate", "override_rate", "funnel_conversion"]

REGIONED_BASE_IDS = [f"trigger_effectiveness_{m}_windowed_region" for m in METRICS]
ALL_REGIONED_IDS = [
    variant for qid in REGIONED_BASE_IDS for variant in (qid, f"{qid}_include_synthetic")
]


def _registration_lines(content: str) -> dict[str, str]:
    """Map each regioned+windowed query_id to its VALUES registration line."""
    lines: dict[str, str] = {}
    for line in content.splitlines():
        for qid in ALL_REGIONED_IDS:
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


# ---------------------------------------------------------------------------
# Part 1: the kpi_query() function unroll is extended 4 -> 6 params
# ---------------------------------------------------------------------------
def test_function_is_redefined():
    content = MIGRATION.read_text()
    assert "CREATE OR REPLACE FUNCTION public.kpi_query(" in content, (
        "migration must redefine kpi_query() to extend the unroll"
    )


def test_unroll_extended_to_six_params():
    """The positional unroll must add the n=5 and n=6 arms binding param_arr[5]
    and param_arr[6]."""
    content = MIGRATION.read_text()
    assert "ELSIF n = 5 THEN" in content, "n=5 unroll arm missing"
    assert "ELSIF n = 6 THEN" in content, "n=6 unroll arm missing"
    assert "param_arr[5]" in content, "5th positional bind missing"
    assert "param_arr[6]" in content, "6th positional bind missing"


def test_over_cap_message_says_six_not_four():
    """The terminal RAISE must reflect the new 6-param cap, not the old 4."""
    content = MIGRATION.read_text()
    assert "at most 6 positional parameters" in content, "over-cap RAISE must say 6"
    assert "at most 4 positional parameters" not in content, (
        "stale 4-param over-cap message must be gone"
    )


def test_security_model_unchanged():
    """The arity check and SECURITY DEFINER allowlist model must be preserved —
    the fix widens the unroll, it does NOT relax the security posture."""
    content = MIGRATION.read_text()
    assert "n <> expected_n" in content, "arity check dropped (security regression)"
    assert "SECURITY DEFINER" in content, "SECURITY DEFINER dropped"
    assert "SET search_path = public, pg_temp" in content, "search_path pin dropped"


# ---------------------------------------------------------------------------
# Part 2: the 8 regioned+windowed trigger-effectiveness statements
# ---------------------------------------------------------------------------
def test_all_eight_regioned_windowed_statements_registered():
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    missing = [qid for qid in ALL_REGIONED_IDS if qid not in lines]
    assert not missing, f"regioned+windowed statements not registered: {missing}"


def test_regioned_windowed_bind_five_params():
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        assert "$kpi$, 5," in line, f"{qid}: max_params must be 5"


def test_param_order_brand_region_triggertype_window():
    """Declared contract: $1 brand, $2 region (patient_journeys join), $3
    trigger_type, $4/$5 half-open window on trigger_timestamp."""
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        sql = _kpi_sql(line)
        assert "$1::text IS NULL OR brand_id::text = $1" in sql, f"{qid}: brand at $1"
        assert "$2::text IS NULL OR" in sql and "geographic_region" in sql, (
            f"{qid}: region must filter via the patient_journeys join at $2"
        )
        assert "LOWER(geographic_region::text) = LOWER($2)" in sql, f"{qid}: region $2 join"
        assert "$3::text IS NULL OR trigger_type::text = $3" in sql, f"{qid}: trigger_type at $3"
        assert "trigger_timestamp >= $4::timestamptz" in sql, f"{qid}: window start at $4"
        assert "trigger_timestamp < $5::timestamptz" in sql, f"{qid}: half-open window end at $5"


def test_regioned_windowed_carry_no_data_through():
    """Windowed forms bind the ask's explicit window, so — like the 118 windowed
    variants — they must NOT carry the frontier data_through provenance column."""
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        sql = _kpi_sql(line)
        assert "data_through" not in sql, f"{qid}: data_through must not ride on windowed forms"
        assert "NOW()" not in sql, f"{qid}: NOW()-anchored window (089 regression)"


def test_synthetic_gating_follows_the_additive_twin_idiom():
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        sql = _kpi_sql(line)
        if qid.endswith("_include_synthetic"):
            assert "is_synthetic = false" not in sql, f"{qid}: twin must include synthetic"
        else:
            assert "is_synthetic = false" in sql, f"{qid}: base must exclude synthetic"


def test_precision_keeps_the_v2_truth_definition():
    """The regioned+windowed precision must keep migration 113's v2 outcome
    (accepted-and-converted / accepted-and-tracked), not revert to v1."""
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    for qid in [i for i in ALL_REGIONED_IDS if "precision" in i]:
        sql = _kpi_sql(lines[qid])
        assert "acceptance_status = 'accepted' AND outcome_tracked AND outcome_value > 0" in sql


def test_acceptance_and_override_keep_the_delivered_denominator():
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    denom = "delivery_status IN ('delivered', 'viewed')"
    for qid in [i for i in ALL_REGIONED_IDS if "acceptance_rate" in i]:
        assert denom in lines[qid], f"{qid}: delivered denominator lost"
        assert "acceptance_status = 'accepted'" in lines[qid], f"{qid}: numerator lost"
    for qid in [i for i in ALL_REGIONED_IDS if "override_rate" in i]:
        assert denom in lines[qid], f"{qid}: delivered denominator lost"
        assert "acceptance_status = 'overridden'" in lines[qid], f"{qid}: numerator lost"


def test_funnel_carries_all_stage_counts_and_the_actioned_headline():
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    for qid in [i for i in ALL_REGIONED_IDS if "funnel_conversion" in i]:
        sql = _kpi_sql(lines[qid])
        for col in ("n_delivered", "n_viewed", "n_accepted", "n_actioned", "n_outcome"):
            assert col in sql, f"{qid}: funnel stage column {col} missing"
        assert "n_actioned::float / NULLIF" in sql, f"{qid}: headline must be actioned/delivered"


def test_idempotent_upsert_and_postgrest_reload():
    content = MIGRATION.read_text()
    assert "ON CONFLICT (query_id) DO UPDATE" in content
    assert "NOTIFY pgrst" in content
