"""Migration 092 content lock — WS2-TR-004 Acceptance Rate denominator (#1124).

The documented formula is ``count(accepted) / count(delivered)``
(config/kpi_definitions.yaml, docs/data/06-KPI-REFERENCE.md), but migrations
044/066/078/089 registered a denominator of ALL non-null acceptance_status
rows — which degenerates to COUNT(*) because the DGP never emits a NULL
status (pending included), and since #1122 also counts 'overridden'.
Migration 092 re-registers the four acceptance_rate variants with the
delivered-based denominator established by migration 090 (#1119):

    COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END)

('viewed' is strictly post-delivery in the trigger delivery lifecycle, so it
counts as delivered.) These text-level assertions lock the migration content
so a later re-registration cannot silently revert the denominator.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = (
    REPO_ROOT / "database" / "migrations" / "092_kpi_tr004_acceptance_delivered_denominator.sql"
)

ACCEPTANCE_QUERY_IDS = [
    "trigger_performance_acceptance_rate",
    "trigger_performance_acceptance_rate_include_synthetic",
    "trigger_performance_acceptance_rate_region",
    "trigger_performance_acceptance_rate_region_include_synthetic",
]

DELIVERED_DENOMINATOR = "delivery_status IN ('delivered', 'viewed')"


def _registration_lines(content: str) -> dict[str, str]:
    """Map each acceptance_rate query_id to its VALUES line in the migration."""
    lines: dict[str, str] = {}
    for line in content.splitlines():
        for qid in ACCEPTANCE_QUERY_IDS:
            if f"('{qid}'" in line:
                lines[qid] = line
    return lines


def test_migration_file_exists():
    assert MIGRATION.exists(), f"missing migration: {MIGRATION.name}"


def test_all_four_acceptance_variants_reregistered():
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    missing = [qid for qid in ACCEPTANCE_QUERY_IDS if qid not in lines]
    assert not missing, f"acceptance_rate variants not re-registered: {missing}"


def test_denominator_is_delivered_based():
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        assert DELIVERED_DENOMINATOR in line, (
            f"{qid}: denominator must count delivered triggers "
            f"({DELIVERED_DENOMINATOR}), matching the documented formula"
        )
        assert "acceptance_status IS NOT NULL" not in line, (
            f"{qid}: the old all-non-null-status denominator (degenerates to "
            "COUNT(*)) must not survive migration 092"
        )


def _kpi_sql(line: str) -> str:
    """Extract the $kpi$-delimited SQL body from a registration line."""
    start = line.index("$kpi$") + len("$kpi$")
    end = line.index("$kpi$", start)
    return line[start:end]


def test_frontier_anchoring_and_data_through_preserved():
    """092 must keep the 089 frontier-anchoring contract: window anchored at
    MAX(trigger_timestamp) (not NOW()) and a data_through provenance column."""
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        sql = _kpi_sql(line)
        assert "data_through" in sql, f"{qid}: data_through provenance dropped"
        assert "MAX(trigger_timestamp)" in sql, f"{qid}: frontier anchor dropped"
        assert "NOW()" not in sql, f"{qid}: NOW()-anchored window reintroduced (089 regression)"


def test_numerator_still_counts_accepted():
    content = MIGRATION.read_text()
    for qid, line in _registration_lines(content).items():
        assert "acceptance_status = 'accepted'" in line, f"{qid}: numerator lost"


def test_region_variants_keep_single_param_arity():
    """Region variants must stay max_params=1 (arity is part of the allowlist
    contract); non-region variants stay 0."""
    content = MIGRATION.read_text()
    lines = _registration_lines(content)
    for qid, line in lines.items():
        expected = 1 if "_region" in qid else 0
        assert f"$kpi$, {expected}," in line, f"{qid}: max_params must stay {expected}"


def test_idempotent_upsert_and_postgrest_reload():
    content = MIGRATION.read_text()
    assert "ON CONFLICT (query_id) DO UPDATE" in content
    assert "NOTIFY pgrst" in content
