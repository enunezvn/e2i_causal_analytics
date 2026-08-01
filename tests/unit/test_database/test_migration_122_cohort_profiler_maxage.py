"""Migration 122 content lock — cohort_profiler windowed max-age bind (#1402).

Migration 117 registered ``cohort_profiler_patient_criteria_profile_windowed``
(+ its ``_include_synthetic`` twin) with only FOUR positional params — brand
($1), the half-open [$2, $3) window, and the exclusive MIN-age bound ($4). The
MAX-age bound was DROPPED because the migration-044 ``kpi_query()`` RPC capped
at 4 positional params (documented in the mig-117 header and the agent's
disclosure path). #1388 (mig-120) raised that cap to 6, so the constraint that
forced the drop is gone.

Migration 122 upserts BOTH windowed rows with the MAX-age bound restored as the
5th positional param (``age_at_diagnosis < $5::int``) and ``max_params`` bumped
4 -> 5. It is additive (``ON CONFLICT DO UPDATE`` + ``NOTIFY pgrst``), touches
only the two windowed ids, and leaves every other mig-117 statement untouched.

These text-level locks keep the restored bind from silently reverting. The
migration is validated by READING it (DB application is deferred to
batch-deploy — the local DB is prod). Mirrors
test_migration_120_kpi_query_6_params.py / test_mig117_registry_presence.py.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = REPO_ROOT / "database" / "migrations" / "122_cohort_profiler_windowed_maxage_bind.sql"

_WINDOWED_BASE = "cohort_profiler_patient_criteria_profile_windowed"
_WINDOWED_IDS = (_WINDOWED_BASE, f"{_WINDOWED_BASE}_include_synthetic")


def _registration_lines(content: str) -> dict[str, str]:
    """Map each windowed query_id to its VALUES registration line."""
    lines: dict[str, str] = {}
    for line in content.splitlines():
        for qid in _WINDOWED_IDS:
            # Anchor on the exact tuple opener so the base id does not also
            # match the _include_synthetic line.
            if f"('{qid}', $kpi$" in line or f"('{qid}',$kpi$" in line:
                lines[qid] = line
    return lines


def _kpi_sql(line: str) -> str:
    start = line.index("$kpi$") + len("$kpi$")
    end = line.index("$kpi$", start)
    return line[start:end]


def _max_params(line: str) -> int:
    m = re.search(r"\$kpi\$,\s*(\d+),", line)
    assert m, f"could not read max_params from: {line[:80]}..."
    return int(m.group(1))


def test_migration_file_exists():
    assert MIGRATION.exists(), f"missing migration: {MIGRATION.name}"


def test_both_windowed_ids_upserted():
    lines = _registration_lines(MIGRATION.read_text())
    for qid in _WINDOWED_IDS:
        assert qid in lines, f"migration 122 missing windowed query_id {qid}"


def test_max_params_bumped_to_five():
    lines = _registration_lines(MIGRATION.read_text())
    for qid, line in lines.items():
        assert _max_params(line) == 5, f"{qid} must declare max_params=5"


def test_maxage_bound_restored_as_fifth_param():
    lines = _registration_lines(MIGRATION.read_text())
    for qid, line in lines.items():
        body = _kpi_sql(line)
        # The restored MAX-age bound at $5 (exclusive, nullable).
        assert "age_at_diagnosis < $5::int" in body, qid
        # ...alongside the still-present window + MIN-age binds.
        assert "event_date >= $2::date" in body and "event_date < $3::date" in body, qid
        assert "age_at_diagnosis > $4::int" in body, qid
        assert "($1::text IS NULL OR" in body, qid
        # NRx substrate + patient_id join preserved (the #1208 gotcha).
        assert "pj.patient_id = te.patient_id" in body, qid
        assert "sequence_number = 1" in body, qid
        assert "patient_journey_id" not in body, qid


def test_synthetic_gating_preserved():
    lines = _registration_lines(MIGRATION.read_text())
    base_body = _kpi_sql(lines[_WINDOWED_BASE])
    twin_body = _kpi_sql(lines[f"{_WINDOWED_BASE}_include_synthetic"])
    assert "is_synthetic = false" in base_body
    assert "is_synthetic = false" not in twin_body


def test_no_ddl_and_no_transaction_wrappers():
    sql = MIGRATION.read_text().lower()
    assert "alter table" not in sql
    assert "add column" not in sql
    assert "begin;" not in sql
    assert "commit;" not in sql


def test_idempotent_upsert_and_schema_reload():
    sql = MIGRATION.read_text()
    assert "ON CONFLICT (query_id) DO UPDATE" in sql
    assert "NOTIFY pgrst" in sql
