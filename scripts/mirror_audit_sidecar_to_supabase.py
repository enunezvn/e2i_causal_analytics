#!/usr/bin/env python3
"""Mirror adaptive-validity sidecars (JSON) into Supabase for cross-experiment queries.

Issue #238. Plan precedent: see ``database/migrations/040_adaptive_validity_verdicts.sql``.

Sidecars on the ``audit_artifacts`` named volume are CANONICAL. This script
mirrors them into the ``adaptive_validity_verdicts`` table for cross-experiment
queryability. The mirror is upsert-keyed on ``(experiment_id, feature,
written_at)`` so re-running over the same sidecars produces zero net writes
(or no-op UPDATEs).

Consistency story
-----------------
Cursor: the mirror reads ``max(imported_at)`` from the table on startup,
subtracts a small overlap window (default 1 hour), and feeds that into
``SidecarReader.since``. Rationale:

  - Worker is idempotent at the row level (ON CONFLICT DO UPDATE), so any
    overlap is a safe no-op.

  - The overlap window covers the case where the worker crashed AFTER
    reading some sidecars and BEFORE committing them: the next run sees
    those same files (their ``written_at`` is older than max(imported_at)
    by less than the overlap), re-upserts them harmlessly.

  - On first run (table empty), ``max(imported_at)`` returns NULL and the
    cursor defaults to ``NULL`` (process every sidecar in the directory).

Database connection
-------------------
Uses ``DATABASE_URL`` (psycopg-v3 connection string). Matches the env-var
convention of ``scripts/run_migration.py``. The script EXIT-1's if the
env var is missing — no in-memory fallback. Reuses ``psycopg`` (v3) which
the project already pulls in via dependencies (see ``pyproject.toml``).

Schema-tolerance
----------------
Reuses ``SidecarReader.iter_verdict_records()`` so:
  - Parse errors / malformed JSON are skipped with WARN (consistent with
    the reader's existing behavior).
  - Unknown verdict keys are preserved verbatim in the ``verdict`` JSONB
    column (the reader emits ONE WARN per file for unknown keys; this
    script does NOT re-implement that).
  - Pre-2026-05-15 sidecars without evaluator_* keys still mirror; the
    ``evaluator_audit`` column is NULL for those rows.

Logging
-------
Counts logged at INFO on success:
  - ``read``: total VerdictRecord rows yielded by the reader.
  - ``upserted_new``: rows where ON CONFLICT did NOT fire.
  - ``upserted_updated``: rows where ON CONFLICT fired AND verdict /
    evaluator_audit payload changed (IS DISTINCT FROM WHERE passed).
  - ``noop``: rows where ON CONFLICT fired BUT the payload was
    byte-identical (IS DISTINCT FROM WHERE filtered the UPDATE).
    On a steady-state re-run over unchanged sidecars this should be
    the dominant count — that's the write-amplification dampener
    working.

The split between new/updated is computed via the ``xmax`` system column
(see ``_UPSERT_SQL`` below) — Postgres-standard trick. The noop count is
inferred from no row being returned: when the WHERE clause filters out
the UPDATE there's no RETURNING tuple.

CLI
---
    python scripts/mirror_audit_sidecar_to_supabase.py \\
        --artifacts-dir /app/data/audit_artifacts \\
        [--overlap-hours 1] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit_sidecar_reader import (  # noqa: E402
    SidecarReader,
    VerdictRecord,
)

logger = logging.getLogger("mirror_audit_sidecar_to_supabase")


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Default cursor overlap window. The mirror re-processes sidecars whose
# ``written_at`` is at most this much older than ``max(imported_at)``;
# safe because the upsert is idempotent.
_DEFAULT_OVERLAP_HOURS = 1

# Subset of VerdictRecord fields that constitute the "evaluator_audit"
# JSONB column. Pulled out so the disagreement query path doesn't need
# to traverse the full ``verdict`` blob.
_EVALUATOR_FIELDS: tuple[str, ...] = (
    "evaluator_satisfied",
    "evaluator_rationale_complete",
    "evaluator_missed_considerations",
    "evaluator_notes",
    "evaluator_model",
    "evaluator_latency_ms",
    "evaluator_input_tokens",
    "evaluator_output_tokens",
    "evaluator_cost_usd",
)

# Upsert statement. ``RETURNING (xmax = 0) AS inserted`` is the standard
# Postgres trick: xmax is 0 on a freshly inserted row, nonzero on a row
# that an ON CONFLICT UPDATE touched. Lets us count new-vs-existing
# without a second query.
#
# WHERE ... IS DISTINCT FROM ... clause: prevents write-amplification when
# rerunning the mirror on byte-identical sidecars. Without it, every
# conflict would refresh ``imported_at = now()`` and produce a write per
# row per run. ``IS DISTINCT FROM`` is the NULL-safe equality negation
# Postgres ships (NULL IS DISTINCT FROM NULL is FALSE; NULL = NULL is
# NULL, which would silently skip the UPDATE). This is exactly the
# semantic we want: "only fire UPDATE when the payload actually changed".
#
# Note for jsonb: ``IS DISTINCT FROM`` on jsonb is structural equality
# (key-order-insensitive object compare, see Postgres docs on jsonb
# comparison). That is the natural meaning for a payload-changed test.
_UPSERT_SQL = """
INSERT INTO adaptive_validity_verdicts (
    experiment_id, feature, written_at, source_path,
    verdict, evaluator_audit, causal_role_final, causal_role_source,
    imported_at
)
VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, now())
ON CONFLICT (
    COALESCE(experiment_id, '__unknown__'),
    COALESCE(feature, '__unknown__'),
    written_at
) DO UPDATE SET
    verdict = EXCLUDED.verdict,
    evaluator_audit = EXCLUDED.evaluator_audit,
    -- Phase 1 of Issue #237 causal-role propagation: source-of-truth
    -- writes through on every conflict. Together with the IS DISTINCT
    -- FROM filter below, repeated mirror runs over unchanged sidecars
    -- remain no-ops.
    causal_role_final = EXCLUDED.causal_role_final,
    causal_role_source = EXCLUDED.causal_role_source,
    source_path = EXCLUDED.source_path,
    imported_at = now()
WHERE adaptive_validity_verdicts.verdict IS DISTINCT FROM EXCLUDED.verdict
   OR adaptive_validity_verdicts.evaluator_audit IS DISTINCT FROM EXCLUDED.evaluator_audit
   OR adaptive_validity_verdicts.causal_role_final IS DISTINCT FROM EXCLUDED.causal_role_final
   OR adaptive_validity_verdicts.causal_role_source IS DISTINCT FROM EXCLUDED.causal_role_source
RETURNING (xmax = 0) AS inserted;
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _evaluator_audit_payload(record: VerdictRecord) -> Optional[dict[str, Any]]:
    """Pull the evaluator-audit subset off ``record``. Returns ``None`` if
    every evaluator field is None (the evaluator was disabled or
    pre-#241), which surfaces as a NULL ``evaluator_audit`` JSONB.

    Surfacing as NULL rather than ``{}`` lets the query layer write
    ``WHERE evaluator_audit IS NOT NULL`` as the natural "rows with
    evaluator signal" predicate.
    """
    payload: dict[str, Any] = {}
    for field in _EVALUATOR_FIELDS:
        value = getattr(record, field, None)
        if value is not None:
            payload[field] = value
    return payload or None


def _role_attribution_payload(
    record: VerdictRecord,
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(causal_role_final, causal_role_source)`` for the row's mirror columns.

    Phase 1 of Issue #237 causal-role propagation. The sidecar reader's
    ``VerdictRecord.role_attribution`` is ``None`` on pre-1.1 sidecars
    (the producer did not yet emit the ``role_attributions`` list) — in
    that case both columns are NULL.

    For 1.1+ sidecars, the per-feature lookup map built by the reader at
    file-load time hands us a dict of shape ``{feature, causal_role,
    source, evaluator_satisfied, evaluator_model}``. We pull ``causal_role``
    and ``source`` for the mirror; the other fields stay in the sidecar
    JSON as audit context (the database does not need them for
    cross-experiment queries).
    """
    if record.role_attribution is None:
        return (None, None)
    causal_role = record.role_attribution.get("causal_role")
    source = record.role_attribution.get("source")
    return (
        causal_role if isinstance(causal_role, str) else None,
        source if isinstance(source, str) else None,
    )


def _parse_since(value: str) -> datetime:
    """Parse the ``--since`` CLI value into a tz-AWARE datetime.

    Accepts ISO8601 with trailing ``Z`` (Zulu), explicit offset, or
    naive (no offset / no Z). Naive timestamps are normalized to UTC
    so the value is safe to compare against ``SidecarReader``'s
    tz-aware ``written_at`` (audit_sidecar_reader.py:268-269 forces
    written_at to UTC; comparing naive-vs-aware raises TypeError).

    Raises ``ValueError`` on unparseable input — caller is responsible
    for mapping that to a user-facing CLI error.

    Iter-2 codex MED: production callers passing
    ``--since=2025-01-01T00:00:00`` (no Z) would TypeError before this
    function existed. Pinned by
    ``tests/unit/test_scripts/test_mirror_audit_sidecar_helpers.py``.
    """
    raw = value.strip()
    # ``datetime.fromisoformat`` only learned ``Z`` in 3.11; tolerate
    # older releases by rewriting to ``+00:00``.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)  # raises ValueError
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _resolve_cursor(
    *,
    db_cursor: Optional[datetime],
    since_override: Optional[datetime],
) -> Optional[datetime]:
    """Compute the effective ``since`` floor for the SidecarReader.

    ``--since`` is a FLOOR ON TOP OF the DB cursor, never a replacement.
    Iter-1 set ``cursor = since_override`` which let
    ``--since=1970-01-01`` re-scan the entire sidecar history —
    re-creating the write-amp risk the cursor exists to prevent.

    Returns:
      - ``max(db_cursor, since_override)`` when both are set.
      - The non-None one when exactly one is set.
      - ``None`` (first run, no --since) → reader admits every sidecar.

    Iter-2 codex MED: pinned by
    ``tests/unit/test_scripts/test_mirror_audit_sidecar_helpers.py``.
    """
    if since_override is not None and db_cursor is not None:
        return max(db_cursor, since_override)
    if since_override is not None:
        return since_override
    return db_cursor


def _read_cursor(conn: psycopg.Connection, overlap_hours: int) -> Optional[datetime]:
    """Return the "since" cursor: ``max(imported_at) - overlap_hours``.

    None on first run (empty table) → reader processes every sidecar.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT max(imported_at) FROM adaptive_validity_verdicts;")
        row = cur.fetchone()
    if row is None or row[0] is None:
        logger.info("cursor: table empty — processing every sidecar")
        return None
    max_imported: datetime = row[0]
    cursor = max_imported - timedelta(hours=overlap_hours)
    logger.info(
        "cursor: max(imported_at)=%s, applying overlap=%dh → since=%s",
        max_imported.isoformat(),
        overlap_hours,
        cursor.isoformat(),
    )
    # The reader compares against the sidecar's ``written_at``, not the
    # imported_at. Conceptually that's fine because a sidecar can't be
    # imported before it was written, so any sidecar with
    # written_at < cursor is guaranteed to be already in the table.
    return cursor


def _upsert_records(
    conn: psycopg.Connection,
    records: list[VerdictRecord],
    *,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Upsert ``records`` and return (new_count, updated_count, noop_count).

    - ``new_count``: rows where ON CONFLICT did NOT fire (xmax = 0 path).
    - ``updated_count``: rows where ON CONFLICT fired AND the WHERE clause
      passed because payload changed (xmax != 0 with row returned).
    - ``noop_count``: rows where ON CONFLICT fired BUT the WHERE clause
      filtered out the UPDATE (no row returned). These are byte-identical
      re-imports — exactly the case the WHERE clause exists to skip.
    """
    if dry_run:
        logger.info("DRY-RUN: would upsert %d records; skipping DB writes.", len(records))
        return (0, 0, 0)
    new_count = 0
    updated_count = 0
    noop_count = 0
    with conn.cursor() as cur:
        for r in records:
            evaluator_payload = _evaluator_audit_payload(r)
            causal_role_final, causal_role_source = _role_attribution_payload(r)
            cur.execute(
                _UPSERT_SQL,
                (
                    r.experiment_id,
                    r.feature,
                    r.written_at,
                    str(r.source_path),
                    json.dumps(r.raw_verdict, default=str),
                    json.dumps(evaluator_payload, default=str)
                    if evaluator_payload is not None
                    else None,
                    causal_role_final,
                    causal_role_source,
                ),
            )
            row = cur.fetchone()
            if row is None:
                # WHERE clause filtered out the UPDATE: unchanged payload.
                noop_count += 1
            elif row[0]:
                new_count += 1
            else:
                updated_count += 1
    conn.commit()
    return (new_count, updated_count, noop_count)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror adaptive-validity sidecars from $ADAPTIVE_VALIDITY_ARTIFACTS_DIR "
            "into the adaptive_validity_verdicts table for cross-experiment queries."
        )
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing adaptive_verdicts_*.json files. "
            "Defaults to $ADAPTIVE_VALIDITY_ARTIFACTS_DIR."
        ),
    )
    parser.add_argument(
        "--overlap-hours",
        type=int,
        default=_DEFAULT_OVERLAP_HOURS,
        help=(
            f"Re-process sidecars whose written_at is at most this many hours older "
            f"than max(imported_at) (default: {_DEFAULT_OVERLAP_HOURS}). "
            "The upsert is idempotent, so the overlap is a safe no-op."
        ),
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help=(
            "ISO8601 timestamp used as a FLOOR on top of the default "
            "max(imported_at) - overlap_hours cursor. The effective cursor "
            "is max(db_cursor, --since) when both are present, so passing "
            "--since=1970-01-01 does NOT force a full re-scan of years-old "
            "sidecars (the DB cursor wins if it is more recent). Naive "
            "timestamps (no offset or Z) are normalized to UTC. Used by "
            "integration tests that write synthetic sidecars with older "
            "written_at values than the in-DB imported_at would otherwise "
            "admit."
        ),
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Postgres connection string. Defaults to $DATABASE_URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read sidecars and log counts but do NOT write to the DB.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Resolve artifacts directory.
    artifacts_dir = args.artifacts_dir
    if artifacts_dir is None:
        env_dir = os.environ.get("ADAPTIVE_VALIDITY_ARTIFACTS_DIR")
        if not env_dir:
            parser.error("neither --artifacts-dir nor $ADAPTIVE_VALIDITY_ARTIFACTS_DIR is set")
        artifacts_dir = Path(env_dir)

    if not artifacts_dir.exists():
        logger.warning(
            "artifacts-dir %s does not exist; nothing to mirror",
            artifacts_dir,
        )
        return 0

    # Resolve database URL.
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("neither --database-url nor $DATABASE_URL is set")

    # Parse --since floor (test escape-hatch + production knob).
    since_override: Optional[datetime] = None
    if args.since is not None:
        try:
            since_override = _parse_since(args.since)
        except ValueError as exc:
            parser.error(f"--since={args.since!r} is not a valid ISO8601 timestamp: {exc}")

    # Open connection (autocommit=False; we commit explicitly after the
    # full batch upsert so a mid-batch crash rolls back cleanly).
    # Lazy import: keeps ``import scripts.mirror_audit_sidecar_to_supabase``
    # working in unit tests that don't have psycopg-v3 installed (the
    # helpers _parse_since / _resolve_cursor have no DB dependency).
    import psycopg  # noqa: PLC0415

    logger.info("connecting to Postgres ...")
    with psycopg.connect(database_url) as conn:
        # Effective cursor = max(db_cursor, since_override). --since is a
        # FLOOR, not a replacement: passing --since=1970-01-01 must NOT
        # rescan the entire sidecar history (that would re-create the
        # write-amp risk the cursor exists to prevent). See _resolve_cursor.
        db_cursor = _read_cursor(conn, overlap_hours=args.overlap_hours)
        cursor = _resolve_cursor(db_cursor=db_cursor, since_override=since_override)
        if since_override is not None and db_cursor is not None:
            logger.info(
                "cursor: --since=%s and db_cursor=%s; effective floor=%s",
                since_override.isoformat(),
                db_cursor.isoformat(),
                cursor.isoformat() if cursor is not None else "<none>",
            )
        elif since_override is not None:
            logger.info(
                "cursor: db empty; using --since floor=%s",
                since_override.isoformat(),
            )
        reader = SidecarReader(artifacts_dir=artifacts_dir, since=cursor)
        records = list(reader.iter_verdict_records())
        logger.info("read %d verdict records from %s", len(records), artifacts_dir)
        new_count, updated_count, noop_count = _upsert_records(conn, records, dry_run=args.dry_run)
        logger.info(
            "done: read=%d, upserted_new=%d, upserted_updated=%d, noop=%d, dry_run=%s",
            len(records),
            new_count,
            updated_count,
            noop_count,
            args.dry_run,
        )
        # ``conn.commit()`` is called inside ``_upsert_records`` on success;
        # the context-manager closes the connection on exit.
    return 0


if __name__ == "__main__":
    sys.exit(main())
