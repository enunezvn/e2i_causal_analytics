"""Issue #376 — Phase 4 schema completion.

Tests pin:
  * Migration ``database/memory/025_crystaldigest_schema_completion.sql``
    adds the 15 missing CrystalDigest fields with idempotent
    ``ADD COLUMN IF NOT EXISTS`` syntax.
  * ``ExecutiveInsightResponse`` is extended with all 15 new fields
    + ``to_dashboard_payload()`` method.
  * ``staleness_score`` is DROPPED per Decision 3 = KEEP BINARY —
    must NOT appear in the migration or the Pydantic response.

Source of truth: GitHub issue #376 DoD §A.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION_PATH = REPO_ROOT / "database" / "memory" / "025_crystaldigest_schema_completion.sql"

# Per issue #376 §A — 15 missing CrystalDigest fields. ORDER MATTERS for
# readable migrations; this is the canonical order from the DoD.
EXPECTED_NEW_COLUMNS = [
    ("effect_size", "FLOAT"),
    ("effect_ci_lower", "FLOAT"),
    ("effect_ci_upper", "FLOAT"),
    ("effect_direction", "TEXT"),
    ("cohort_size", "INTEGER"),
    ("confounders_controlled", "TEXT[]"),
    ("sensitivity_checks_passed", "TEXT[]"),
    ("sensitivity_checks_failed", "TEXT[]"),
    ("limitations", "TEXT"),
    ("recommended_next_analysis", "TEXT"),
    ("provenance_chain_id", "TEXT"),
    ("provenance_depth", "INTEGER"),
    ("consolidation_tier", "TEXT"),
    ("replication_count", "INTEGER"),
    ("data_version", "TEXT"),
]


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


def test_migration_025_file_exists():
    """The migration file must exist at the expected path.

    Per memory `[[feat-381-decision3-followups-close-20260519]]`,
    023 = sentinel_cooldown, 024 = sentinel_invalidation_count_pattern,
    so this PR picks 025.
    """
    assert MIGRATION_PATH.exists(), (
        f"Migration not found: {MIGRATION_PATH}. "
        "Per issue #376 DoD (with correction in plan §Recommended sequencing item 3): "
        "use migration number 025 (023 and 024 are already taken)."
    )


def test_migration_025_is_idempotent():
    """All ALTER TABLE statements must use ``ADD COLUMN IF NOT EXISTS`` so
    the migration can be re-run without erroring."""
    if not MIGRATION_PATH.exists():
        pytest.fail("Migration file missing — see test_migration_025_file_exists")

    sql = MIGRATION_PATH.read_text()
    # Strip comments + whitespace for pattern matching
    body = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)

    # Count ADD COLUMN occurrences
    add_columns = re.findall(r"ADD\s+COLUMN\s+(IF\s+NOT\s+EXISTS\s+)?(\w+)", body, re.IGNORECASE)
    assert len(add_columns) == 15, (
        f"Expected 15 ADD COLUMN statements; found {len(add_columns)}: "
        f"{[m[1] for m in add_columns]}"
    )

    # Every ADD COLUMN must use IF NOT EXISTS
    missing_idempotent = [m[1] for m in add_columns if not m[0].strip()]
    assert not missing_idempotent, (
        f"All ADD COLUMN statements must include IF NOT EXISTS for idempotency; "
        f"missing: {missing_idempotent}"
    )


def test_migration_025_adds_all_15_expected_columns():
    """Each of the 15 specced columns must appear in the migration with
    its expected type."""
    if not MIGRATION_PATH.exists():
        pytest.fail("Migration file missing — see test_migration_025_file_exists")

    sql = MIGRATION_PATH.read_text()
    body = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)

    for col_name, col_type in EXPECTED_NEW_COLUMNS:
        # ADD COLUMN IF NOT EXISTS <name> <TYPE>
        pattern = rf"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+{re.escape(col_name)}\s+{re.escape(col_type)}"
        assert re.search(pattern, body, re.IGNORECASE), (
            f"Expected column not found: {col_name} {col_type}. "
            f"Pattern: {pattern}"
        )


def test_migration_025_does_not_add_staleness_score():
    """Per Decision 3 = KEEP BINARY (adopted 2026-05-19), the
    ``staleness_score`` field is DROPPED from Phase 4 scope.

    The 15 columns in this migration must NOT include staleness_score.
    """
    if not MIGRATION_PATH.exists():
        pytest.fail("Migration file missing — see test_migration_025_file_exists")

    sql = MIGRATION_PATH.read_text()
    # Comments are OK to mention it (e.g. to document the omission);
    # only check the executable DDL body.
    body = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    assert "staleness_score" not in body.lower(), (
        "staleness_score column must NOT be added — Decision 3 = KEEP BINARY"
    )


def test_migration_025_effect_direction_check_constraint():
    """``effect_direction`` is enumerated to 'positive'/'negative'/'null'.

    The check constraint is the data-shape contract; absent it, callers
    can write arbitrary strings.
    """
    if not MIGRATION_PATH.exists():
        pytest.fail("Migration file missing")

    sql = MIGRATION_PATH.read_text()
    body = sql.lower()
    # Look for a check constraint mentioning the three allowed values
    assert "positive" in body and "negative" in body, (
        "effect_direction must have a CHECK constraint enumerating "
        "'positive'/'negative'/'null'"
    )


def test_migration_025_targets_executive_insights_table():
    """The migration extends ``executive_insights`` (shipped in 021),
    not a new table."""
    if not MIGRATION_PATH.exists():
        pytest.fail("Migration file missing")

    sql = MIGRATION_PATH.read_text()
    body = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    # Every ALTER TABLE must target executive_insights
    alter_targets = re.findall(r"ALTER\s+TABLE\s+(\w+)", body, re.IGNORECASE)
    assert alter_targets, "No ALTER TABLE statements found"
    assert all(t.lower() == "executive_insights" for t in alter_targets), (
        f"All ALTER TABLE statements must target executive_insights; "
        f"found: {set(alter_targets)}"
    )


# ---------------------------------------------------------------------------
# Pydantic schema tests
# ---------------------------------------------------------------------------


def test_executive_insight_response_includes_all_15_new_fields():
    """ExecutiveInsightResponse must expose all 15 new columns + the
    13 already-shipped fields."""
    from src.api.routes.executive_insights import ExecutiveInsightResponse

    fields = ExecutiveInsightResponse.model_fields

    for col_name, _ in EXPECTED_NEW_COLUMNS:
        assert col_name in fields, (
            f"Expected field {col_name} not found in ExecutiveInsightResponse. "
            f"Found: {sorted(fields.keys())}"
        )


def test_executive_insight_response_effect_fields_are_numeric():
    """Per sub-decision 2a (2026-05-19): effect_size + ci bounds are
    numeric, not categorical strings."""
    from src.api.routes.executive_insights import ExecutiveInsightResponse

    fields = ExecutiveInsightResponse.model_fields

    # The annotation should resolve to float-compatible Optional
    for numeric_col in ("effect_size", "effect_ci_lower", "effect_ci_upper"):
        annotation = fields[numeric_col].annotation
        # Optional[float] resolves to Union[float, None] or float | None
        annotation_str = str(annotation)
        assert "float" in annotation_str.lower(), (
            f"{numeric_col} must be numeric (float). Got: {annotation_str}"
        )


def test_executive_insight_response_has_to_dashboard_payload_method():
    """The response must expose ``to_dashboard_payload()`` (per plan
    §Phase 4 line 133)."""
    from src.api.routes.executive_insights import ExecutiveInsightResponse

    assert hasattr(ExecutiveInsightResponse, "to_dashboard_payload"), (
        "ExecutiveInsightResponse must implement to_dashboard_payload()"
    )


def test_to_dashboard_payload_returns_dict_with_expected_keys():
    """The dashboard payload must surface the 15 new analytical/lineage
    fields so the frontend can render them."""
    from datetime import datetime, timezone

    from src.api.routes.executive_insights import ExecutiveInsightResponse

    instance = ExecutiveInsightResponse(
        insight_id="abc-123",
        title="Test",
        narrative="Test narrative",
        brand="Kisqali",
        crystallized_at=datetime.now(timezone.utc),
        effect_size=0.42,
        effect_ci_lower=0.30,
        effect_ci_upper=0.55,
        effect_direction="positive",
        cohort_size=1200,
        confounders_controlled=["age", "prior_use"],
        sensitivity_checks_passed=["placebo_treatment", "random_common_cause"],
        sensitivity_checks_failed=["data_subset"],
        limitations="Pre-period n=120; sensitivity to outliers HIGH.",
        recommended_next_analysis="Replicate on Q3 cohort with 360d washout.",
        provenance_chain_id="chain-7",
        provenance_depth=4,
        consolidation_tier="semantic",
        replication_count=2,
        data_version="2026-05-19-snapshot",
    )

    payload = instance.to_dashboard_payload()
    assert isinstance(payload, dict)
    # The 15 new fields must round-trip into the dashboard payload
    for col_name, _ in EXPECTED_NEW_COLUMNS:
        assert col_name in payload, f"Dashboard payload missing field {col_name}"
    # Plus the existing fields
    assert payload["insight_id"] == "abc-123"
    assert payload["brand"] == "Kisqali"


def test_to_dashboard_payload_serializes_datetime_to_iso():
    """The frontend ingests JSON; datetimes must be ISO strings."""
    from datetime import datetime, timezone

    from src.api.routes.executive_insights import ExecutiveInsightResponse

    instance = ExecutiveInsightResponse(
        insight_id="abc-123",
        title="t",
        narrative="n",
        brand="Kisqali",
        crystallized_at=datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc),
    )
    payload = instance.to_dashboard_payload()
    assert isinstance(payload["crystallized_at"], str)
    assert "2026-05-19" in payload["crystallized_at"]
