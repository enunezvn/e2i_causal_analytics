"""Unit tests for sentinel YAML config + startup loader (#375).

Plan §Phase 3 Step 3.7 / 3.10 ship the 4 specced sentinels (Optum quarterly
data drop, staleness alert, Pluvicto cohort drift, weekly consolidation)
from a YAML file at startup. Pattern-vocabulary divergence note: the YAML
uses the plan-specified trigger names (data_drop, staleness_threshold,
cohort_drift, schedule) and the loader translates them to the shipped
internal vocabulary (freshness, threshold_breach, drift_score,
new_causal_path) — see ``PLAN_TRIGGER_TO_INTERNAL_PATTERN`` mapping.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# RED first: this module does not yet exist; import error proves the test fails
# for the right reason (feature missing, not typo).
from src.memory.sentinels.config_loader import (
    DEFAULT_CONFIG_PATH,
    PLAN_TRIGGER_TO_INTERNAL_PATTERN,
    SentinelConfigLoadError,
    load_sentinels_from_yaml,
)


def _strip_sql_comments(sql: str) -> str:
    """Strip SQL line comments (``--``) and block comments (``/* */``) from a string.

    Used by the sentinel-pattern lockstep test (#381 codex iter-1 MED) to
    prevent false enum-coverage satisfaction from commented-out ``ALTER TYPE
    ... ADD VALUE`` clauses. A future migration carrying
    ``/* ALTER TYPE sentinel_pattern_type ADD VALUE 'phantom' */`` in a block
    comment, or ``-- ALTER TYPE sentinel_pattern_type ADD VALUE 'phantom'`` in
    a line comment, must NOT register ``phantom`` as a real enum value.

    Order matters: block comments are stripped first via a non-greedy
    multi-line match so that a ``--`` inside ``/* */`` does not pre-trigger
    line-comment stripping. Dollar-quoted strings (``$tag$ ... $tag$``) are
    uncommon in plain DDL and intentionally not handled here.
    """
    # Strip /* ... */ block comments (non-greedy, multi-line).
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    # Strip -- line comments (to end of line).
    sql = re.sub(r"--[^\n]*", "", sql)
    return sql


def test_strip_sql_comments_helper() -> None:
    """Sanity-check for ``_strip_sql_comments`` itself (the helper used by the
    lockstep test below). Verifies that both block- and line-commented
    ``ALTER TYPE`` clauses are stripped before any downstream scan.

    Codex iter-1 MED (#381): the pre-existing parser only stripped ``--``
    comments, so a ``/* ALTER TYPE ... ADD VALUE 'foo' */`` block-comment in a
    future migration would falsely satisfy enum coverage.
    """
    fake_block = "/* ALTER TYPE sentinel_pattern_type ADD VALUE 'phantom' */"
    assert "phantom" not in _strip_sql_comments(fake_block), (
        "Parser should strip block comments"
    )

    fake_line = "-- ALTER TYPE sentinel_pattern_type ADD VALUE 'phantom'"
    assert "phantom" not in _strip_sql_comments(fake_line), (
        "Parser should strip line comments"
    )

    # Multi-line block comment that spans real-looking DDL.
    fake_multiline = (
        "ALTER TYPE sentinel_pattern_type ADD VALUE 'real';\n"
        "/* this is a multi-line\n"
        "   ALTER TYPE sentinel_pattern_type ADD VALUE 'phantom'\n"
        "   comment */"
    )
    stripped = _strip_sql_comments(fake_multiline)
    assert "real" in stripped, "Parser must keep live DDL"
    assert "phantom" not in stripped, "Parser must strip multi-line block comments"

    # A ``--`` inside a ``/* */`` should not survive: block-stripping happens
    # first, so the inner ``--`` and everything after vanish together.
    nested = "/* foo -- bar */ ALTER TYPE x ADD VALUE 'kept'"
    stripped_nested = _strip_sql_comments(nested)
    assert "foo" not in stripped_nested
    assert "bar" not in stripped_nested
    assert "kept" in stripped_nested


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._mode: Optional[str] = None
        self._filters: Dict[str, Any] = {}
        self._insert_payload: Any = None
        self._update_payload: Dict[str, Any] = {}

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        return self

    def insert(self, payload: Any) -> "_FakeQuery":
        self._mode = "insert"
        self._insert_payload = payload
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._update_payload = payload
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def limit(self, n: int) -> "_FakeQuery":
        return self

    def execute(self) -> MagicMock:
        mock = MagicMock()
        if self._mode == "insert":
            payload = self._insert_payload
            rows_to_insert = payload if isinstance(payload, list) else [payload]
            inserted = []
            for r in rows_to_insert:
                row = dict(r)
                row.setdefault("sentinel_id", f"fake-{len(self.store.rows[self.table_name]) + 1}")
                self.store.rows.setdefault(self.table_name, []).append(row)
                inserted.append(row)
            mock.data = inserted
            return mock
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            rows = [r for r in rows if r.get(col) == want]
        if self._mode == "update":
            for r in rows:
                for orig in self.store.rows[self.table_name]:
                    if orig is r:
                        orig.update(self._update_payload)
                        break
        mock.data = rows
        return mock


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {"sentinels": []}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)


@pytest.fixture
def fake_supabase() -> FakeSupabase:
    return FakeSupabase()


@pytest.fixture(autouse=True)
def patch_supabase(fake_supabase):
    with (
        patch("src.memory.sentinels.registry.get_supabase_client", return_value=fake_supabase),
        patch(
            "src.memory.sentinels.config_loader.get_supabase_client",
            return_value=fake_supabase,
        ),
    ):
        yield fake_supabase


@pytest.mark.asyncio
async def test_default_config_path_points_to_sentinels_yaml():
    """Sanity: the default YAML config path is the file we shipped."""
    assert DEFAULT_CONFIG_PATH.name == "sentinels.yaml"
    assert DEFAULT_CONFIG_PATH.exists(), (
        f"sentinels.yaml not found at {DEFAULT_CONFIG_PATH}; #375 scope requires this file"
    )


@pytest.mark.asyncio
async def test_load_sentinels_yaml_registers_four_plan_specced(
    fake_supabase: FakeSupabase,
):
    """Load the shipped default YAML and verify exactly the 4 plan-specced
    sentinels land in the sentinels table."""
    loaded = await load_sentinels_from_yaml(DEFAULT_CONFIG_PATH)
    assert loaded == 4, f"expected 4 plan-specced sentinels, got {loaded}"
    names = sorted(r["name"] for r in fake_supabase.rows["sentinels"])
    # Per plan §3.7 + issue #375 scope.
    assert any("Optum" in n for n in names), names
    assert any("staleness" in n.lower() for n in names), names
    assert any("Pluvicto" in n for n in names), names
    assert any("consolidation" in n.lower() for n in names), names


@pytest.mark.asyncio
async def test_load_sentinels_yaml_idempotent_skips_existing(
    fake_supabase: FakeSupabase, tmp_path: Path
):
    """Re-loading the same YAML twice must NOT create duplicates.

    Identity is determined by ``(name, brand)`` — the YAML-shipped sentinels
    have stable names; re-registering must be a no-op.
    """
    yaml_path = tmp_path / "test_sentinels.yaml"
    yaml_path.write_text(
        """\
sentinels:
  - id: sentinel_test_one
    name: Test sentinel one
    trigger_type: staleness_threshold
    condition:
      max_staleness: 0.6
    action: notify_and_queue_reanalysis
    brands: ["*"]
    active: true
    cooldown_minutes: 360
"""
    )
    first = await load_sentinels_from_yaml(yaml_path)
    second = await load_sentinels_from_yaml(yaml_path)
    assert first == 1
    assert second == 0, "second load must not duplicate"
    assert len(fake_supabase.rows["sentinels"]) == 1


@pytest.mark.asyncio
async def test_load_sentinels_yaml_disabled_sentinels_skipped(
    fake_supabase: FakeSupabase, tmp_path: Path
):
    """An entry with ``active: false`` must not be registered at startup."""
    yaml_path = tmp_path / "disabled.yaml"
    yaml_path.write_text(
        """\
sentinels:
  - id: disabled_one
    name: Disabled sentinel
    trigger_type: staleness_threshold
    condition:
      max_staleness: 0.6
    action: notify_and_queue_reanalysis
    brands: ["*"]
    active: false
    cooldown_minutes: 60
"""
    )
    loaded = await load_sentinels_from_yaml(yaml_path)
    assert loaded == 0
    assert fake_supabase.rows["sentinels"] == []


@pytest.mark.asyncio
async def test_plan_trigger_vocabulary_mapping_covers_four_types():
    """Plan-vocabulary `(data_drop|staleness_threshold|cohort_drift|schedule)`
    must each map to a shipped internal pattern_type the registry understands.
    """
    from src.memory.sentinels.registry import VALID_PATTERN_TYPES

    assert "data_drop" in PLAN_TRIGGER_TO_INTERNAL_PATTERN
    assert "staleness_threshold" in PLAN_TRIGGER_TO_INTERNAL_PATTERN
    assert "cohort_drift" in PLAN_TRIGGER_TO_INTERNAL_PATTERN
    assert "schedule" in PLAN_TRIGGER_TO_INTERNAL_PATTERN
    # Every mapped value must be a real registry-accepted pattern.
    for plan_trigger, internal in PLAN_TRIGGER_TO_INTERNAL_PATTERN.items():
        assert internal in VALID_PATTERN_TYPES, (
            f"plan trigger {plan_trigger!r} maps to {internal!r} which "
            f"is not in registry VALID_PATTERN_TYPES={VALID_PATTERN_TYPES}"
        )


@pytest.mark.asyncio
async def test_load_sentinels_yaml_missing_file_raises():
    with pytest.raises(SentinelConfigLoadError):
        await load_sentinels_from_yaml(Path("/nonexistent/sentinels.yaml"))


@pytest.mark.asyncio
async def test_load_sentinels_yaml_malformed_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_sentinels_block: 1\n")
    with pytest.raises(SentinelConfigLoadError):
        await load_sentinels_from_yaml(bad)


def test_internal_pattern_types_have_db_enum_coverage():
    """Lock the invariant: every value in ``PLAN_TRIGGER_TO_INTERNAL_PATTERN``
    must exist in the ``sentinel_pattern_type`` Postgres enum.

    The enum is declared in ``database/memory/021_insight_lifecycle.sql``
    (4 original values) and extended by subsequent migrations
    (``024_sentinel_invalidation_count_pattern.sql`` adds
    ``invalidation_count`` per issue #381 codex iter-0 HIGH-1).

    Without this lockstep test, a future change that adds a value to
    ``PLAN_TRIGGER_TO_INTERNAL_PATTERN`` without a corresponding DB migration
    would fail at production INSERT time with a Postgres enum-violation —
    the unit-test suite mocks Supabase and would not catch the drift.
    """
    # ``re`` is imported at module level.

    # Resolve repo root via Path.parents — this test lives at
    # ``tests/unit/test_memory/test_sentinel_yaml_loader.py`` so parents[3]
    # is the worktree/repo root.
    repo_root = Path(__file__).resolve().parents[3]
    # Strip SQL comments (block ``/* */`` AND line ``--``) before any scan.
    # Codex iter-1 MED (#381): without block-comment stripping, a future
    # migration carrying ``/* ALTER TYPE sentinel_pattern_type ADD VALUE 'foo' */``
    # in a block comment would falsely satisfy enum coverage.
    migration_021 = _strip_sql_comments(
        (repo_root / "database/memory/021_insight_lifecycle.sql").read_text()
    )
    migration_024_raw = (
        repo_root / "database/memory/024_sentinel_invalidation_count_pattern.sql"
    ).read_text()
    migration_024 = _strip_sql_comments(migration_024_raw)

    # Sanity: migration 024 must exist and carry the right ALTER TYPE
    # (checked against the comment-stripped text, so a commented-out clause
    # would NOT satisfy this assertion).
    assert (
        "ADD VALUE IF NOT EXISTS 'invalidation_count'" in migration_024
    ), "migration 024 missing expected ALTER TYPE for invalidation_count"

    # Parse migration 021's CREATE TYPE block for sentinel_pattern_type. The
    # DDL shape (021:59-64) is:
    #   CREATE TYPE sentinel_pattern_type AS ENUM (
    #       'threshold_breach',  -- ...
    #       'freshness',         -- ...
    #       'drift_score',       -- ...
    #       'new_causal_path'    -- ...
    #   );
    # Extract the block, then collect every single-quoted string within it.
    create_match = re.search(
        r"CREATE TYPE\s+sentinel_pattern_type\s+AS\s+ENUM\s*\((.*?)\);",
        migration_021,
        re.DOTALL,
    )
    assert (
        create_match is not None
    ), "could not locate CREATE TYPE sentinel_pattern_type in migration 021"
    # ``migration_021`` is already comment-stripped above, so the ``--``
    # line-comments documented in 021 (e.g. ``-- threshold_breach: ...``)
    # have already been removed; English-apostrophes in those comments
    # therefore cannot leak into the quoted-value scan.
    enum_block = create_match.group(1)
    declared_values = set(re.findall(r"'([^']+)'", enum_block))
    # Cross-check against the doc-of-record 4 values to catch silent edits to 021.
    assert declared_values == {
        "threshold_breach",
        "freshness",
        "drift_score",
        "new_causal_path",
    }, f"migration 021 sentinel_pattern_type values drifted: {declared_values}"

    # Collect ADD VALUE additions across all later migrations against this enum.
    added_values: set[str] = set()
    for migration_path in sorted((repo_root / "database/memory").glob("*.sql")):
        if migration_path.name == "021_insight_lifecycle.sql":
            continue
        # Strip block + line comments BEFORE scanning so commented-out
        # ALTER TYPE clauses cannot falsely satisfy enum coverage.
        text = _strip_sql_comments(migration_path.read_text())
        for match in re.finditer(
            r"ALTER\s+TYPE\s+sentinel_pattern_type\s+"
            r"ADD\s+VALUE(?:\s+IF\s+NOT\s+EXISTS)?\s+'([^']+)'",
            text,
            re.IGNORECASE,
        ):
            added_values.add(match.group(1))

    # ``invalidation_count`` must be in the added set (migration 024 contract).
    assert (
        "invalidation_count" in added_values
    ), f"invalidation_count not added by any migration; added={added_values}"

    valid_db_enum_values = declared_values | added_values

    used_internal_values = set(PLAN_TRIGGER_TO_INTERNAL_PATTERN.values())

    missing_in_db = used_internal_values - valid_db_enum_values
    assert not missing_in_db, (
        f"Internal pattern types not in DB enum: {sorted(missing_in_db)}. "
        f"Add a migration to extend `sentinel_pattern_type` before shipping "
        f"these values via PLAN_TRIGGER_TO_INTERNAL_PATTERN."
    )
