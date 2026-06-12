#!/usr/bin/env python3
"""One-off convergence of NaN-bearing episodic JSONB string-scalar rows (#891).

Background
----------
Migration 073 (PR #888) repaired the #883 episodic double-encode: JSONB
columns that stored JSON *string scalars* instead of objects were plain-cast
back (``(col #>> '{}')::jsonb``). 137 ``episodic_memories.raw_content`` rows
were DELIBERATELY skipped: they are model_trainer metric payloads whose
pre-#888 ``json.dumps`` emitted bare ``NaN``/``Infinity`` tokens — valid for
Python ``json.loads``, rejected by Postgres ``::jsonb``. An in-SQL regex
rewrite was rejected twice (permission classifier mid-#888 + codex R2:
quote-UNaware — it would also rewrite ``: NaN`` inside legitimate string
values such as ``"threshold: NaN means missing"``, which is pinned by
tests/integration/test_episodic_jsonb_shape_883c.py and must survive
verbatim).

This script is the endorsed Python repair sketched in the migration 073
header: ``json.loads`` each string scalar (quote-aware by construction), map
non-finite floats to ``None`` (the same semantics the writers now apply at
source via ``src.memory.jsonb_sanitize``), and write back proper JSONB
objects. Readers are unaffected (``hydrate_raw_content`` already parses both
shapes); this only converges the at-rest representation so server-side JSONB
operators (``->``, ``@>``, jsonpath) work on every row.

Safety properties
-----------------
- ``--dry-run`` is the DEFAULT: reports what would change, writes nothing.
- ``--execute`` first copies every affected row (all three JSONB columns) to
  a timestamped backup table ``episodic_raw_content_backup_891_<UTC ts>``,
  takes an exclusive flock on ``--lock-file`` (default /tmp/e2i_dbtest.lock,
  the shared DB-mutation lock), and runs in a single transaction.
- Only payloads that parse to a JSON object/array are written back; string
  scalars whose inner text is unparseable or parses to a bare scalar are
  reported and left untouched (never mutated).
- Row-level guard ``AND jsonb_typeof(<col>) = 'string'`` on every UPDATE
  makes the write race-safe; re-running converges 0 rows (idempotent).
- Post-write verification (inside the same transaction, before COMMIT):
  every converged value is now object/array, the total table row count is
  unchanged, and the remaining string-scalar count equals the deliberate
  skips. Any mismatch aborts with ROLLBACK.

Connection
----------
Pass ``--dsn`` or set ``E2I_PG_DSN``. The .env ``SUPABASE_DB_URL`` points at
the supavisor pooler (127.0.0.1:5432) and fails with "Tenant or user not
found"; use the worker container's direct DSN rewritten to host port 5433::

    DSN=$(docker exec e2i-causal-analytics-worker_light-1 sh -c 'echo "$SUPABASE_DB_URL"' \\
          | sed -E 's#@[^/]+/#@127.0.0.1:5433/#')
    .venv/bin/python scripts/maintenance/converge_episodic_nan_rows_891.py --dsn "$DSN"            # dry-run
    .venv/bin/python scripts/maintenance/converge_episodic_nan_rows_891.py --dsn "$DSN" --execute  # repair

Proof: tests/integration/test_episodic_nan_convergence_891.py (red->green,
single BEGIN..ROLLBACK against the live docker DB).
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

# The JSONB columns the #883 double-encode touched. Live (2026-06-12) only
# raw_content has string scalars left (137), but entities/outcome_details are
# checked uniformly so a writer regression on any of them is caught and
# backed up together.
JSONB_COLUMNS = ("raw_content", "entities", "outcome_details")

DEFAULT_LOCK_FILE = "/tmp/e2i_dbtest.lock"
BACKUP_TABLE_PREFIX = "episodic_raw_content_backup_891"


@dataclass
class ConvergeStats:
    """Outcome of one convergence pass (no commit — caller owns the txn)."""

    candidates: int = 0  # string-scalar column values found
    converged: int = 0  # values rewritten to proper jsonb objects/arrays
    converged_ids: List[str] = field(default_factory=list)  # memory_ids touched
    skipped: List[Tuple[str, str, str]] = field(default_factory=list)  # (id, col, reason)


def parse_and_sanitize(txt: str) -> Any:
    """Parse a stored string-scalar payload and strip non-finite floats.

    Quote-aware BY CONSTRUCTION: ``json.loads`` only treats ``NaN``/
    ``Infinity`` as constants in value position; the same characters inside a
    quoted string are just characters. ``parse_constant`` maps the bare
    tokens to None at parse time; the recursive sweep additionally catches
    floats that overflow to inf via plain number syntax (e.g. ``1e999``).
    The strict ``json.dumps(..., allow_nan=False)`` round-trip is a hard
    guarantee no non-finite value survived.
    """
    from src.memory.jsonb_sanitize import sanitize_jsonb_payload

    parsed = json.loads(txt, parse_constant=lambda _tok: None)
    cleaned = sanitize_jsonb_payload(parsed)
    json.dumps(cleaned, allow_nan=False)  # raises if anything non-finite slipped through
    return cleaned


def find_candidates(conn) -> List[Tuple[str, str, str]]:
    """Return (memory_id, column, inner_text) for every string-scalar value."""
    out: List[Tuple[str, str, str]] = []
    with conn.cursor() as cur:
        for col in JSONB_COLUMNS:
            cur.execute(
                f"SELECT memory_id::text, {col} #>> '{{}}' "  # noqa: S608 — col from fixed tuple
                f"FROM episodic_memories WHERE jsonb_typeof({col}) = 'string' "
                f"ORDER BY memory_id"
            )
            out.extend((mid, col, txt) for mid, txt in cur.fetchall())
    return out


def converge(conn, candidates: Optional[List[Tuple[str, str, str]]] = None) -> ConvergeStats:
    """Rewrite parseable string-scalar payloads as proper JSONB. No commit.

    Each UPDATE re-guards on ``jsonb_typeof(col) = 'string'`` so a row
    repaired by a concurrent pass (or already an object) is never rewritten.
    """
    if candidates is None:
        candidates = find_candidates(conn)
    stats = ConvergeStats(candidates=len(candidates))
    touched = set()
    with conn.cursor() as cur:
        for mid, col, txt in candidates:
            try:
                cleaned = parse_and_sanitize(txt)
            except (ValueError, TypeError) as exc:
                stats.skipped.append((mid, col, f"unparseable: {exc}"))
                continue
            if not isinstance(cleaned, (dict, list)):
                stats.skipped.append(
                    (mid, col, f"payload is {type(cleaned).__name__}, not object/array")
                )
                continue
            cur.execute(
                f"UPDATE episodic_memories SET {col} = %s::jsonb "  # noqa: S608
                f"WHERE memory_id = %s AND jsonb_typeof({col}) = 'string'",
                (json.dumps(cleaned, allow_nan=False, ensure_ascii=False), mid),
            )
            if cur.rowcount == 1:
                stats.converged += 1
                touched.add(mid)
            else:  # raced: someone repaired it between SELECT and UPDATE
                stats.skipped.append((mid, col, "no longer a string scalar (raced)"))
    stats.converged_ids = sorted(touched)
    return stats


def _backup_rows(conn, memory_ids: List[str]) -> Tuple[str, int]:
    """Copy affected rows (all JSONB columns, pre-repair) to a timestamped table."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    table = f"{BACKUP_TABLE_PREFIX}_{ts}"
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE {table} AS "  # noqa: S608 — name built from fixed prefix + ts
            f"SELECT memory_id, raw_content, entities, outcome_details, "
            f"now() AS backed_up_at FROM episodic_memories "
            f"WHERE memory_id = ANY(%s::uuid[])",
            (memory_ids,),
        )
        cur.execute(f"SELECT count(*) FROM {table}")  # noqa: S608
        n = cur.fetchone()[0]
    if n != len(memory_ids):
        raise RuntimeError(f"backup incomplete: {n} rows in {table}, expected {len(memory_ids)}")
    return table, n


def _counts(conn) -> Tuple[int, int]:
    """(total rows, string-scalar column values across the JSONB columns)."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM episodic_memories")
        total = cur.fetchone()[0]
        strings = 0
        for col in JSONB_COLUMNS:
            cur.execute(
                f"SELECT count(*) FROM episodic_memories "  # noqa: S608
                f"WHERE jsonb_typeof({col}) = 'string'"
            )
            strings += cur.fetchone()[0]
    return total, strings


def _resolve_dsn(arg_dsn: Optional[str]) -> str:
    dsn = arg_dsn or os.environ.get("E2I_PG_DSN")
    if dsn:
        return dsn
    dsn = os.environ.get("SUPABASE_DB_URL")
    if dsn:
        # Codex iter-1 M1: on the droplet this env var points at the supavisor
        # pooler and connect fails with "Tenant or user not found" — keep the
        # fallback (it IS the direct DSN inside the worker containers) but say
        # so loudly instead of letting the pooler error mystify the operator.
        print(
            "WARNING: no --dsn/E2I_PG_DSN; falling back to SUPABASE_DB_URL. On the "
            "droplet host that is the supavisor pooler and will fail with 'Tenant or "
            "user not found' — use the worker-DSN recipe in the script header.",
            file=sys.stderr,
        )
        return dsn
    sys.exit("No DSN: pass --dsn or set E2I_PG_DSN (see script header for the recipe)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dsn", help="postgres DSN (default: $E2I_PG_DSN, then $SUPABASE_DB_URL)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report only, write nothing (this is the DEFAULT behavior)",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="actually mutate (default is dry-run); backs up affected rows first",
    )
    ap.add_argument(
        "--lock-file",
        default=DEFAULT_LOCK_FILE,
        help=f"flock taken for --execute (default {DEFAULT_LOCK_FILE})",
    )
    args = ap.parse_args(argv)
    if args.dry_run and args.execute:
        ap.error("--dry-run and --execute are mutually exclusive")

    import psycopg2

    conn = psycopg2.connect(_resolve_dsn(args.dsn))
    lock_fh = None
    try:
        conn.autocommit = False
        total_before, strings_before = _counts(conn)
        candidates = find_candidates(conn)
        print(
            f"episodic_memories: {total_before} rows; {strings_before} string-scalar "
            f"JSONB values across {JSONB_COLUMNS}; {len(candidates)} candidates"
        )

        if not args.execute:
            stats = converge(conn, candidates)  # transient: rolled back below
            conn.rollback()
            print(
                f"[DRY-RUN] would converge {stats.converged} values on "
                f"{len(stats.converged_ids)} rows; {len(stats.skipped)} skipped"
            )
            for mid, col, reason in stats.skipped:
                print(f"[DRY-RUN]   skip {mid}.{col}: {reason}")
            if stats.converged_ids:
                sample = ", ".join(stats.converged_ids[:5])
                print(f"[DRY-RUN] sample memory_ids: {sample}")
            print("[DRY-RUN] no changes written; re-run with --execute to repair")
            return 0

        # ---- execute -------------------------------------------------------
        if not candidates:
            print("Nothing to converge (already idempotent-clean); no backup table created.")
            return 0

        import fcntl

        lock_fh = open(args.lock_file, "a+")  # noqa: SIM115 — held until finally
        print(f"Waiting for exclusive flock on {args.lock_file} ...")
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        print("Lock acquired.")

        affected_ids = sorted({mid for mid, _col, _txt in candidates})
        backup_table, backed_up = _backup_rows(conn, affected_ids)
        print(f"Backed up {backed_up} affected rows to {backup_table}")

        stats = converge(conn, candidates)

        # Verify inside the transaction; any mismatch rolls back everything
        # (including the backup table, which is fine — nothing was mutated).
        total_after, strings_after = _counts(conn)
        if total_after != total_before:
            raise RuntimeError(f"row count changed {total_before} -> {total_after}; aborting")
        if strings_after != len(stats.skipped):
            raise RuntimeError(
                f"{strings_after} string scalars remain but only {len(stats.skipped)} "
                f"were deliberately skipped; aborting"
            )
        conn.commit()
        print(
            f"Converged {stats.converged} values on {len(stats.converged_ids)} rows; "
            f"{len(stats.skipped)} skipped; string scalars {strings_before} -> {strings_after}; "
            f"total rows unchanged ({total_after}). Backup: {backup_table} ({backed_up} rows)."
        )
        for mid, col, reason in stats.skipped:
            print(f"  skipped {mid}.{col}: {reason}")
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        if lock_fh is not None:
            lock_fh.close()
        conn.close()


if __name__ == "__main__":
    # Allow `python scripts/maintenance/...py` from the repo root without
    # PYTHONPATH (the sanitize helper lives in src/).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
