"""Offline PHI/PII audit harness for crystal narratives + LLM-prompt audit.

Issue #391 security box 4. Companion to :mod:`src.security.phi_scanner`.

What it does
------------
1. Loads CrystalDigest rows from ``executive_insights`` plus their
   paired ``LLMCrystalNarrativeAudit`` records from Postgres (via
   ``psycopg``).
2. Runs :func:`src.security.phi_scanner.scan_text` on each text field
   that could carry PHI/PII leaks:
   * ``key_finding``   — the LLM-generated headline
   * ``narrative``     — the deterministic + LLM-blended body
   * ``audit_input_prompt`` — the FULL prompt sent to Anthropic Haiku
     (sourced from ``crystal_narrative_audits.input_prompt``, added by
     migration 028).
3. Emits a JSON report to stdout.
4. Returns / exits non-zero if any matches were found.

When to run
-----------
* Pre-deploy: ``python scripts/audit_phi_in_crystal_narratives.py``
* Weekly: from cron, parsing the JSON output to alert ops if matches > 0.
* Investigation: pass ``--brand=<X>`` to scope to one brand.

Determinism
-----------
The underlying scanner is regex-only (NO LLM / ML — see
:mod:`src.security.phi_scanner`). The same DB snapshot always produces
the same report, which is the security contract: audit reports must
be reviewable + diff-able across runs.

DB connectivity
---------------
``--db-url`` (or env ``TEST_POSTGRES_URL`` for tests / ``DATABASE_URL``
for prod) selects the connection. When invoked with ``records=`` in
:func:`main` (test path), no DB is touched — tests pass records
directly to avoid requiring a live Postgres.

Output shape
------------
.. code-block:: json

    {
      "records_scanned": 2,
      "phi_match_count": 1,
      "findings": [
        {
          "insight_id": "i-bad-1",
          "field": "key_finding",
          "matches": [
            {"pattern_name": "ssn", "match": "555-12-3456",
             "start": 12, "end": 23}
          ]
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

# Ensure repo-root is importable when invoked as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.security.phi_scanner import PhiMatch, scan_text  # noqa: E402

# ---------------------------------------------------------------------------
# Fields scanned per record. The order is the order findings appear in the
# JSON report when multiple fields hit on one record.
# ---------------------------------------------------------------------------
_SCANNED_FIELDS = ("key_finding", "narrative", "audit_input_prompt")


def _match_to_dict(m: PhiMatch) -> Dict[str, Any]:
    """Render a :class:`PhiMatch` as JSON-serializable dict."""
    return dataclasses.asdict(m)


def audit_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Run the PHI/PII scanner across each record's text fields.

    Args:
        records: Iterable of dicts; each dict represents one crystal +
            its paired narrator-audit row, with keys ``insight_id`` and
            at least one of the fields in :data:`_SCANNED_FIELDS`.

    Returns:
        Report dict matching the documented shape (see module docstring).
    """
    findings: List[Dict[str, Any]] = []
    records_scanned = 0
    phi_match_count = 0

    for record in records:
        records_scanned += 1
        insight_id = record.get("insight_id", "<unknown>")
        for field in _SCANNED_FIELDS:
            value = record.get(field)
            if not value or not isinstance(value, str):
                continue
            matches = scan_text(value)
            if not matches:
                continue
            phi_match_count += len(matches)
            findings.append(
                {
                    "insight_id": str(insight_id),
                    "field": field,
                    "matches": [_match_to_dict(m) for m in matches],
                }
            )

    return {
        "records_scanned": records_scanned,
        "phi_match_count": phi_match_count,
        "findings": findings,
    }


def _load_records_from_postgres(db_url: str, brand: Optional[str] = None) -> List[Dict[str, Any]]:
    """Pull crystal+audit rows from Postgres via psycopg.

    Joins ``executive_insights`` with the per-crystal narrator audit
    table (``crystal_narrative_audits``, created by migration 028) on
    ``insight_id``. Audit rows may be missing for legacy crystals; LEFT
    JOIN preserves them so the audit harness still scans
    ``key_finding`` + ``narrative``.

    Note: The audit table is created by migration 028. If the table does
    not exist yet (i.e. migration not applied) the function logs +
    falls back to scanning ONLY ``key_finding`` + ``narrative``.
    """
    # Lazy import — psycopg is only needed when running against a real DB.
    import psycopg

    sql = """
        SELECT
            ei.insight_id::TEXT       AS insight_id,
            ei.brand                  AS brand,
            ei.title                  AS key_finding,
            ei.narrative              AS narrative,
            cna.input_prompt          AS audit_input_prompt
        FROM executive_insights ei
        LEFT JOIN crystal_narrative_audits cna
            ON cna.insight_id = ei.insight_id
        WHERE (%(brand)s IS NULL OR ei.brand = %(brand)s)
    """
    fallback_sql = """
        SELECT
            insight_id::TEXT          AS insight_id,
            brand                     AS brand,
            title                     AS key_finding,
            narrative                 AS narrative,
            NULL::TEXT                AS audit_input_prompt
        FROM executive_insights
        WHERE (%(brand)s IS NULL OR brand = %(brand)s)
    """

    with psycopg.connect(db_url, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, {"brand": brand})
            except psycopg.errors.UndefinedTable:
                conn.rollback()
                cur.execute(fallback_sql, {"brand": brand})
            cols = [d.name for d in cur.description] if cur.description else []
            return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def main(
    *,
    db_url: Optional[str] = None,
    brand: Optional[str] = None,
    records: Optional[Iterable[Dict[str, Any]]] = None,
) -> int:
    """CLI entrypoint. Returns process exit code.

    Args:
        db_url: Postgres DSN. Defaults to ``TEST_POSTGRES_URL`` or
            ``DATABASE_URL`` env. Ignored when ``records`` is supplied.
        brand: Optional brand scope filter (e.g. ``kisqali``).
        records: Test-injection hook — if supplied, skips DB fetch and
            audits this iterable directly. Used by the unit tests.

    Returns:
        ``0`` if no PHI matches, ``1`` if matches found, ``2`` if a
        configuration error prevented the audit (e.g. no DB URL when one
        was needed).
    """
    if records is None:
        url = db_url or os.environ.get("TEST_POSTGRES_URL") or os.environ.get("DATABASE_URL")
        if not url:
            print(
                json.dumps(
                    {
                        "error": "no_db_url",
                        "message": ("Pass --db-url or set TEST_POSTGRES_URL / DATABASE_URL"),
                    }
                ),
                file=sys.stdout,
            )
            return 2
        records = _load_records_from_postgres(url, brand=brand)

    report = audit_records(records)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["phi_match_count"] == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit crystal narratives + LLM audit prompts for PHI/PII.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Postgres DSN (default: TEST_POSTGRES_URL or DATABASE_URL env)",
    )
    parser.add_argument(
        "--brand",
        default=None,
        help="Optional brand scope (e.g. kisqali)",
    )
    args = parser.parse_args()
    sys.exit(main(db_url=args.db_url, brand=args.brand))
