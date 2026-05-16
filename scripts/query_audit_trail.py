#!/usr/bin/env python3
"""Cross-experiment query CLI over the ``adaptive_validity_verdicts`` mirror table.

Issue #238. Companion to ``scripts/mirror_audit_sidecar_to_supabase.py``.

Three query modes exposed:

  --by-feature-family
      Group verdicts by the first underscore-delimited feature prefix
      (the de-facto feature "family"; e.g. ``demographics_age`` →
      ``demographics``). Report total verdict count, evaluator-disagreement
      count, and rate per family. Top-K controlled via --limit.

  --by-quarter
      Group verdicts by calendar quarter of ``written_at``. Report total
      verdict count, evaluator-disagreement count, and rate.

  --by-data-source
      Group verdicts by the ``verdict->>'contract_source'`` JSONB field
      (the producer's data-source attribution; see VerdictRecord.raw_verdict).
      Falls back to ``'__missing__'`` for verdicts that omit the key.

A verdict is counted as an "evaluator disagreement" when
``evaluator_audit->>'evaluator_satisfied'`` is the JSON literal ``false``
(string ``'false'``). Records without an evaluator (NULL evaluator_audit)
are counted in the denominator but never in the disagreement numerator
— the metric is "fraction of evaluated verdicts where the evaluator
disagreed", with the unevaluated fraction visible in the denominator.

Output: TSV by default (header row + data rows). Use --format=table for
a pretty-printed aligned form.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Optional

import psycopg

logger = logging.getLogger("query_audit_trail")


# ----------------------------------------------------------------------------
# Queries
# ----------------------------------------------------------------------------

# Disagreement predicate: evaluator_audit->>'evaluator_satisfied' is the
# Postgres JSONB text-cast of the boolean field. The string ``'false'``
# is what jsonb_build_object writes when the value is JSON false.
_DISAGREEMENT_PREDICATE = "evaluator_audit ->> 'evaluator_satisfied' = 'false'"

_BY_FEATURE_FAMILY_SQL = f"""
WITH families AS (
    SELECT
        split_part(feature, '_', 1) AS family,
        ({_DISAGREEMENT_PREDICATE})::int AS is_disagreement
    FROM adaptive_validity_verdicts
)
SELECT
    family,
    count(*)                        AS total,
    sum(is_disagreement)::int       AS disagreements,
    round(sum(is_disagreement)::numeric / count(*), 4) AS rate
FROM families
GROUP BY family
ORDER BY disagreements DESC, total DESC
LIMIT %s;
"""

_BY_QUARTER_SQL = f"""
SELECT
    to_char(date_trunc('quarter', written_at), 'YYYY"-Q"Q') AS quarter,
    count(*)                                                 AS total,
    sum(({_DISAGREEMENT_PREDICATE})::int)::int               AS disagreements,
    round(sum(({_DISAGREEMENT_PREDICATE})::int)::numeric / count(*), 4) AS rate
FROM adaptive_validity_verdicts
GROUP BY 1
ORDER BY 1 DESC;
"""

# data_source comes from the verdict JSONB (the producer stores it as
# ``contract_source`` per the layer-4 verdict shape; see
# audit_sidecar_reader.py:_KNOWN_VERDICT_KEYS).
_BY_DATA_SOURCE_SQL = f"""
SELECT
    COALESCE(verdict ->> 'contract_source', '__missing__') AS data_source,
    count(*)                                               AS total,
    sum(({_DISAGREEMENT_PREDICATE})::int)::int             AS disagreements,
    round(sum(({_DISAGREEMENT_PREDICATE})::int)::numeric / count(*), 4) AS rate
FROM adaptive_validity_verdicts
GROUP BY 1
ORDER BY disagreements DESC, total DESC;
"""


# ----------------------------------------------------------------------------
# Output formatters
# ----------------------------------------------------------------------------


def _format_tsv(rows: list[tuple[Any, ...]], headers: list[str]) -> str:
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append("\t".join("" if v is None else str(v) for v in row))
    return "\n".join(lines)


def _format_table(rows: list[tuple[Any, ...]], headers: list[str]) -> str:
    if not rows:
        return "\t".join(headers) + "\n(no rows)"
    widths = [len(h) for h in headers]
    str_rows = [[("" if v is None else str(v)) for v in row] for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append(sep.join("-" * w for w in widths))
    for row in str_rows:
        lines.append(sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _execute(
    conn: psycopg.Connection, query: str, params: tuple[Any, ...]
) -> tuple[list[tuple[Any, ...]], list[str]]:
    with conn.cursor() as cur:
        cur.execute(query, params)
        headers = [desc.name for desc in (cur.description or ())]
        rows = list(cur.fetchall())
    return rows, headers


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run cross-experiment queries against the adaptive_validity_verdicts "
            "mirror table (issue #238)."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--by-feature-family",
        action="store_true",
        help="Top-K disagreement rate by feature family (first underscore-prefix).",
    )
    group.add_argument(
        "--by-quarter",
        action="store_true",
        help="Disagreement rate by calendar quarter of written_at.",
    )
    group.add_argument(
        "--by-data-source",
        action="store_true",
        help="Disagreement rate by verdict.contract_source.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Top-K cap for --by-feature-family (default 20). Ignored elsewhere.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Postgres connection string. Defaults to $DATABASE_URL.",
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "table"],
        default="tsv",
        help="Output format (default: tsv).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("neither --database-url nor $DATABASE_URL is set")

    with psycopg.connect(database_url) as conn:
        if args.by_feature_family:
            rows, headers = _execute(conn, _BY_FEATURE_FAMILY_SQL, (args.limit,))
        elif args.by_quarter:
            rows, headers = _execute(conn, _BY_QUARTER_SQL, ())
        else:  # args.by_data_source
            rows, headers = _execute(conn, _BY_DATA_SOURCE_SQL, ())

    if args.format == "table":
        print(_format_table(rows, headers))
    else:
        print(_format_tsv(rows, headers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
