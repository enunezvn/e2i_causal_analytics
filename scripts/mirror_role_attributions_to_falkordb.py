#!/usr/bin/env python3
"""Mirror Phase-1 role_attributions from Supabase into FalkorDB Feature nodes.

Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §6.2.

This is the peer of ``scripts/mirror_audit_sidecar_to_supabase.py``:

  - sidecars (JSON) → Supabase ``adaptive_validity_verdicts.causal_role_*``
    columns (Phase 1, migration 041) — that script.
  - Supabase ``adaptive_validity_verdicts`` → FalkorDB
    ``(:Feature)-[:FOR_BRAND]->(:Brand)`` nodes (Phase 6) — THIS script.

The two-hop mirror keeps Supabase as the canonical query surface for
cross-experiment audit (used by Phase 7's
``query_active_role_attributions``) while FalkorDB serves the Phase-6
Layer-2 KG voter: ``ensemble_voter.layer_2_kg_signal(feature)`` queries
the ``(:Feature)`` node to corroborate or contradict the LLM verdict at
``kg_role_enrichment`` time.

Schema (codex-2 §6.1):

    (:Feature {
        name: string,
        experiment_id: string,
        causal_role: string,
        causal_role_source: string,
        evaluator_model: string,
        written_at: datetime
    })-[:FOR_BRAND]->(:Brand {name: string})

The ``FOR_BRAND`` edge type is chosen rather than ``BELONGS_TO`` to
avoid type-name overload with
``src/agents/ml_foundation/model_trainer/memory_hooks.py:367`` which
already uses ``BELONGS_TO`` for ``(:Model)-[:BELONGS_TO]->(:Experiment)``.

CLI
---
    python scripts/mirror_role_attributions_to_falkordb.py \\
        --brand dupixent \\
        [--since 2026-05-01T00:00:00Z] \\
        [--dry-run]

Idempotency: ``MERGE (:Feature {name, experiment_id})`` does not
create duplicates; ``SET`` overwrites the role/source/model/written_at
fields. Re-running over the same Supabase rows produces zero net new
nodes.

Database connection: reuses ``DATABASE_URL`` (psycopg-v3 connection
string), matching the Phase-1 sidecar mirror's convention.

FalkorDB connection: reuses ``FALKORDB_URL`` / ``FALKORDB_HOST`` etc.
from ``src.api.dependencies.falkordb_client``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.kg.ensemble_voter import upsert_feature_role_node  # noqa: E402

logger = logging.getLogger("mirror_role_attributions_to_falkordb")


# SQL: select role-attribution rows that have a non-NULL causal_role.
# The Phase-1 sidecar mirror writes ``(causal_role_final,
# causal_role_source)`` together-or-not-at-all per migration 041's
# co-presence CHECK; this SELECT preserves that invariant.
#
# evaluator_audit is JSONB; the ``->>`` operator extracts text. We pull
# ``evaluator_model`` for the provenance string. NULL is tolerated for
# manifest rows (their evaluator_audit may be NULL); the worker
# substitutes ``"n/a"`` per the Phase-1 RoleAttribution sentinel.
_SELECT_SQL = """
SELECT
    experiment_id,
    feature,
    causal_role_final,
    causal_role_source,
    evaluator_audit->>'evaluator_model' AS evaluator_model
FROM adaptive_validity_verdicts
WHERE causal_role_final IS NOT NULL
  AND causal_role_source IS NOT NULL
  AND (%s::timestamptz IS NULL OR written_at >= %s::timestamptz)
ORDER BY written_at;
"""


def mirror_role_attributions(
    *,
    conn: Any,
    graph: Any,
    brand: str,
    dry_run: bool = False,
    since: Optional[str] = None,
) -> int:
    """Mirror role-attribution rows from Postgres to FalkorDB.

    Args:
        conn: A psycopg-v3 connection. The script's CLI entrypoint
            opens one; tests pass a fake.
        graph: A FalkorDB graph handle (``client.select_graph(...)``).
            Tests pass a fake.
        brand: The brand name to attach to each Feature node via
            ``(:Feature)-[:FOR_BRAND]->(:Brand {name: $brand})``.
            Passed via CLI; the SQL row does not currently carry the
            brand directly (Phase 1 did not add a brand column).
        dry_run: When True, executes the SELECT and counts eligible
            rows but does not call ``upsert_feature_role_node``.
        since: Optional ISO8601 string lower bound on ``written_at``.

    Returns:
        Count of rows that would be (or were) upserted. Rows where any
        of {causal_role_final, causal_role_source, evaluator_model}
        is None or non-string after SQL extraction are skipped.
    """
    written = 0
    with conn.cursor() as cur:
        # Pass ``since`` twice — once for the IS NULL check, once for the
        # comparison. Postgres needs both occurrences as scalar params
        # under psycopg-v3's ``%s`` style.
        cur.execute(_SELECT_SQL, (since, since))
        rows = cur.fetchall()
    for row in rows:
        # psycopg-v3 returns sequences; tests pass plain tuples.
        if len(row) < 5:
            continue
        experiment_id, feature, causal_role, causal_role_source, evaluator_model = row[:5]
        # Co-presence check matches migration 041's CHECK constraint —
        # one NULL is malformed and must be skipped (would otherwise
        # break the upsert query at the FalkorDB layer).
        if not isinstance(experiment_id, str) or not experiment_id:
            continue
        if not isinstance(feature, str) or not feature:
            continue
        if not isinstance(causal_role, str) or not causal_role:
            continue
        if not isinstance(causal_role_source, str) or not causal_role_source:
            continue
        # evaluator_model may be NULL for manifest sources (no model);
        # substitute the documented sentinel for downstream consistency.
        if not isinstance(evaluator_model, str) or not evaluator_model:
            evaluator_model = "n/a" if causal_role_source == "manifest" else "<unknown>"
        if dry_run:
            written += 1
            continue
        try:
            upsert_feature_role_node(
                graph,
                feature=feature,
                experiment_id=experiment_id,
                causal_role=causal_role,
                causal_role_source=causal_role_source,
                evaluator_model=evaluator_model,
                brand=brand,
            )
            written += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "upsert failed for feature=%r exp=%r — skipping. Cause: %s",
                feature,
                experiment_id,
                exc,
            )
    logger.info(
        "mirror_role_attributions: brand=%s dry_run=%s rows_examined=%d written=%d",
        brand,
        dry_run,
        len(rows),
        written,
    )
    return written


def _open_db_connection() -> "psycopg.Connection":
    """Open a psycopg-v3 connection from ``DATABASE_URL``."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL env var not set; cannot mirror role_attributions")
        sys.exit(1)
    import psycopg  # local import to keep test-time surface lean

    return psycopg.connect(database_url)


async def _open_falkordb_graph() -> Any:
    """Open a FalkorDB graph handle via the FastAPI dependency module."""
    from src.api.dependencies.falkordb_client import get_graph

    return await get_graph()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror Phase-1 role_attributions from Supabase to FalkorDB."
    )
    parser.add_argument(
        "--brand",
        required=True,
        help="Brand name for the (:Feature)-[:FOR_BRAND]->(:Brand) edge",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="ISO8601 lower bound on adaptive_validity_verdicts.written_at",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute the SELECT but skip FalkorDB writes",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    conn = _open_db_connection()
    graph = asyncio.run(_open_falkordb_graph()) if not args.dry_run else None
    if graph is None and not args.dry_run:
        logger.error("FalkorDB graph unavailable; aborting mirror")
        return 1
    try:
        written = mirror_role_attributions(
            conn=conn,
            graph=graph,
            brand=args.brand,
            dry_run=args.dry_run,
            since=args.since,
        )
    finally:
        conn.close()
    logger.info("done; written=%d", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
