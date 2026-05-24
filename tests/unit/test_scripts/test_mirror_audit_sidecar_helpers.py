"""Unit tests for ``scripts.mirror_audit_sidecar_to_supabase`` helpers.

Iter-2 codex gate-on-diff regressions (PR #294):

1. ``_parse_since`` MUST return a tz-aware datetime even when the input
   is naive (no Z, no offset). The SidecarReader compares its ``since``
   against tz-aware ``written_at`` values, and a naive cursor raises
   ``TypeError`` at iteration time. Pre-iter-2 code returned naive
   datetimes from ``datetime.fromisoformat`` directly — production
   invocations with ``--since=2025-01-01T00:00:00`` blew up.

2. ``_resolve_cursor`` MUST treat ``--since`` as a FLOOR on top of the
   DB cursor, not a replacement. Pre-iter-2 code set
   ``cursor = since_override`` when ``--since`` was passed, so
   ``--since=1970-01-01`` re-scanned the entire sidecar history —
   re-creating the write-amp risk that motivated removing
   ``--no-cursor`` in iter-1.

These tests don't need a live DB and so always run.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.mirror_audit_sidecar_to_supabase import _parse_since, _resolve_cursor

# ----------------------------------------------------------------------------
# _parse_since (MED area 2 — naive-datetime TypeError)
# ----------------------------------------------------------------------------


class TestParseSince:
    def test_naive_iso_normalizes_to_utc(self) -> None:
        """``--since=2025-01-01T00:00:00`` (no Z, no offset) returns
        tz-aware datetime in UTC. FALSIFIABILITY: revert the
        ``if parsed.tzinfo is None: parsed.replace(tzinfo=utc)`` block
        and the assertion ``tzinfo is not None`` fails."""
        result = _parse_since("2025-01-01T00:00:00")
        assert result.tzinfo is not None, (
            f"naive --since must be normalized to tz-aware UTC; got {result!r}"
        )
        assert result.utcoffset() == datetime.now(timezone.utc).utcoffset(), (
            f"naive --since should be normalized to UTC; got offset {result.utcoffset()!r}"
        )
        # Bit-exact comparison
        assert result == datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_zulu_suffix_parses_as_utc(self) -> None:
        """Trailing ``Z`` shorthand parses to tz-aware UTC."""
        result = _parse_since("2025-01-01T00:00:00Z")
        assert result == datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_naive_equals_zulu(self) -> None:
        """Naive and Zulu forms of the same wall-clock time are equal
        (the load-bearing equivalence assertion from the task spec)."""
        assert _parse_since("2025-01-01T00:00:00") == _parse_since("2025-01-01T00:00:00Z")

    def test_explicit_offset_is_honored(self) -> None:
        """Explicit non-UTC offsets are honored, not coerced to UTC."""
        result = _parse_since("2025-01-01T00:00:00+05:00")
        # 00:00:00+05:00 == 19:00:00 UTC the day before.
        assert result == datetime(2024, 12, 31, 19, 0, 0, tzinfo=timezone.utc)
        # And the offset is preserved on the parsed object.
        offset = result.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 5 * 3600

    def test_returned_value_compares_safely_with_aware_datetime(self) -> None:
        """The actual TypeError reproduction: a naive datetime would
        raise TypeError when compared against a tz-aware one.
        ``_parse_since`` must return aware datetimes so this never
        happens at SidecarReader iteration time."""
        cursor = _parse_since("2025-01-01T00:00:00")  # naive input
        aware = datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc)
        # If _parse_since regressed to naive output, this comparison
        # would raise: "can't compare offset-naive and offset-aware datetimes".
        assert aware > cursor

    def test_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _parse_since("not-a-date")


# ----------------------------------------------------------------------------
# _resolve_cursor (MED area 6 — --since as floor not replacement)
# ----------------------------------------------------------------------------


class TestResolveCursor:
    """The load-bearing iter-2 fix: --since must be a FLOOR on top of
    db_cursor, not a replacement. ``max(db_cursor, since_override)``
    when both are present; the non-None one otherwise."""

    def test_neither_set_returns_none(self) -> None:
        """First run, no --since: reader admits every sidecar."""
        assert _resolve_cursor(db_cursor=None, since_override=None) is None

    def test_only_db_cursor_returns_db_cursor(self) -> None:
        """No --since, populated DB: reader uses db_cursor."""
        db = datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert _resolve_cursor(db_cursor=db, since_override=None) == db

    def test_only_since_override_returns_since(self) -> None:
        """First run with --since: reader uses --since."""
        since = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert _resolve_cursor(db_cursor=None, since_override=since) == since

    def test_since_in_past_with_later_db_cursor_uses_db_cursor(self) -> None:
        """THE BUG iter-2 fixes: ``--since=1970-01-01`` with a populated
        DB MUST NOT replace the DB cursor. The DB cursor wins because
        it's later — old sidecars stay filtered.

        FALSIFIABILITY: revert _resolve_cursor to ``return since_override
        if since_override else db_cursor`` and this assertion trips
        (returns the 1970 floor instead of the 2026 db_cursor)."""
        db_cursor = datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc)
        since = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _resolve_cursor(db_cursor=db_cursor, since_override=since)
        assert result == db_cursor, (
            f"FALSIFIABILITY-ANCHOR: --since={since.isoformat()} with "
            f"later db_cursor={db_cursor.isoformat()} must yield db_cursor "
            f"(max-floor semantics); got {result.isoformat() if result else None}"
        )

    def test_since_in_future_with_earlier_db_cursor_uses_since(self) -> None:
        """The inverse: ``--since`` in the future (e.g. operator wants
        to ignore old sidecars) wins over an earlier db_cursor."""
        db_cursor = datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc)
        since = datetime(2099, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _resolve_cursor(db_cursor=db_cursor, since_override=since)
        assert result == since

    def test_equal_db_cursor_and_since_returns_either(self) -> None:
        """Edge case: identical timestamps. ``max`` is stable; result
        equals both."""
        ts = datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = _resolve_cursor(db_cursor=ts, since_override=ts)
        assert result == ts


# ----------------------------------------------------------------------------
# Issue #240 Stage 1 — dedicated shadow columns written by the upsert.
#
# Migration 042 adds three dedicated typed columns. They are only populated
# if (a) ``_UPSERT_SQL`` lists them and (b) ``_upsert_records`` passes the
# VerdictRecord's shadow values in the matching positions. These tests pin
# both without a live DB: a fake cursor captures the (sql, params) pair.
# ----------------------------------------------------------------------------


class _FakeCursor:
    """Captures every (sql, params) pair passed to ``execute``. Returns a
    1-tuple from ``fetchone`` so the upsert counts the row as inserted."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def execute(self, sql: str, params: tuple) -> None:
        self.calls.append((sql, params))

    def fetchone(self) -> tuple:
        return (True,)  # xmax == 0 → counted as a new insert


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.commits = 0

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1


def _verdict_record_with_shadow(
    *,
    would_promote_severity,
    would_flag_for_review,
    rationale_incomplete_flag,
):
    from datetime import datetime, timezone
    from pathlib import Path

    from src.data.audit_sidecar_reader import VerdictRecord

    return VerdictRecord(
        experiment_id="exp-1",
        written_at=datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc),
        source_path=Path("/dev/null/synthetic"),
        feature="age",
        layer="4",
        severity="moderate",
        remediation="keep_with_caveat",
        evidence="layer-4 llm",
        z_score=4.2,
        p_value=0.0001,
        delta_auc=0.12,
        evaluator_satisfied=False,
        evaluator_rationale_complete=False,
        evaluator_missed_considerations=["temporal_filter"],
        evaluator_notes="thin",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
        raw_verdict={"feature": "age"},
        would_promote_severity=would_promote_severity,
        would_flag_for_review=would_flag_for_review,
        rationale_incomplete_flag=rationale_incomplete_flag,
    )


class TestUpsertShadowColumns:
    def test_upsert_sql_references_three_shadow_columns(self) -> None:
        from scripts.mirror_audit_sidecar_to_supabase import _UPSERT_SQL

        for col in (
            "would_promote_severity",
            "would_flag_for_review",
            "rationale_incomplete_flag",
        ):
            # INSERT column list + DO UPDATE SET + IS DISTINCT FROM WHERE
            # → the column name must appear at least 3 times.
            assert _UPSERT_SQL.count(col) >= 3, (
                f"{col!r} must be in the INSERT list, the DO UPDATE SET, and the "
                f"IS DISTINCT FROM change-detection WHERE clause; found "
                f"{_UPSERT_SQL.count(col)} occurrence(s)"
            )

    def test_upsert_param_count_matches_placeholders_and_carries_shadow_values(
        self,
    ) -> None:
        from scripts.mirror_audit_sidecar_to_supabase import (
            _UPSERT_SQL,
            _upsert_records,
        )

        rec = _verdict_record_with_shadow(
            would_promote_severity="high",
            would_flag_for_review=True,
            rationale_incomplete_flag=True,
        )
        cur = _FakeCursor()
        conn = _FakeConn(cur)

        new, updated, noop = _upsert_records(conn, [rec], dry_run=False)  # type: ignore[arg-type]

        assert (new, updated, noop) == (1, 0, 0)
        assert len(cur.calls) == 1
        sql, params = cur.calls[0]
        # Positional binding contract: one %s per passed param. A mismatch
        # here is the bug a live DB would raise as "not enough/too many
        # arguments"; this catches it without a DB.
        assert _UPSERT_SQL.count("%s") == len(params), (
            f"placeholder/param mismatch: {_UPSERT_SQL.count('%s')} %s vs "
            f"{len(params)} params — the upsert would raise against a real DB"
        )
        # The three shadow values must be carried into the param tuple.
        assert "high" in params
        assert params.count(True) >= 2  # would_flag_for_review + rationale_incomplete_flag

    def test_upsert_passes_none_shadow_values_when_no_rule_fired(self) -> None:
        from scripts.mirror_audit_sidecar_to_supabase import (
            _UPSERT_SQL,
            _upsert_records,
        )

        rec = _verdict_record_with_shadow(
            would_promote_severity=None,
            would_flag_for_review=None,
            rationale_incomplete_flag=None,
        )
        cur = _FakeCursor()
        conn = _FakeConn(cur)

        _upsert_records(conn, [rec], dry_run=False)  # type: ignore[arg-type]

        _sql, params = cur.calls[0]
        assert _UPSERT_SQL.count("%s") == len(params)
        # The last three positional params are the shadow columns; all None
        # when no rule fired (column stays NULL).
        assert params[-3:] == (None, None, None)
