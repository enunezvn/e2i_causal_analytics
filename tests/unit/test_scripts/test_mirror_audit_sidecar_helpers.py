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
