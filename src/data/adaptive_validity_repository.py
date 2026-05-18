"""Phase 7.1 of the causal-role propagation contract (Issue #237 reframe).

Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §7.1.

Single SQL source-of-truth for active ``RoleAttribution`` rows from the
``adaptive_validity_verdicts`` mirror table (Migration 040 + 041).

The repository is the SECOND consumer of Phase 1's typed
``RoleAttribution`` (after Phase 2's policy node). Tool composer
(Phase 7.2) calls ``query_active_role_attributions(experiment_id)`` to
pre-fill ``confounders`` for tools that accept the parameter when the
caller didn't supply an explicit value AND the C1 trust-gate is open.

**Trust-boundary constraint C1** (from the plan): the producer emits
attributions with ``source ∈ {manifest, llm, kg}``. Manifest and KG
attributions are verification-grade (a maintainer wrote the FeatureContract
or the FalkorDB enrichment node corroborated). LLM attributions are
gated on ``evaluator_audit.satisfied=true``.

**JSONB cast (codex-2 fix)**: postgres ``->>`` returns TEXT, so the
naive query ``WHERE (evaluator_audit->>'satisfied') = true`` would type-
error or silently mis-filter. The explicit ``::boolean`` cast in the
SQL below is the load-bearing fix from the plan §7.1.

**Manifest / KG rows have NO evaluator_audit row** (their satisfied
flag is implicit-True per C1), so they bypass the satisfied filter
via the OR-arm ``causal_role_source IN ('manifest', 'kg')``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from src.data.role_attribution import RoleAttribution

__all__ = ["query_active_role_attributions"]

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# SQL fragments. Pinned by tests in
# ``tests/unit/test_data/test_adaptive_validity_repository.py``:
#   - the ``::boolean`` cast text is checked verbatim.
#   - experiment_id is bound as a parameter, never f-string-injected.
# ----------------------------------------------------------------------------

_BASE_SELECT = """
SELECT
    feature,
    causal_role_final,
    causal_role_source,
    evaluator_audit,
    verdict
FROM adaptive_validity_verdicts
WHERE experiment_id = %s
  AND causal_role_final IS NOT NULL
  AND causal_role_source IS NOT NULL
"""

# C1 gate: manifest and kg sources bypass the satisfied check (their
# trust label IS the gate); LLM sources require an explicit ``satisfied``
# boolean in the evaluator_audit JSONB. The ``::boolean`` cast is
# load-bearing — ``->>`` returns TEXT and ``'true' = true`` is a
# postgres type error.
_SATISFIED_FILTER = """
  AND (
      causal_role_source IN ('manifest', 'kg')
      OR (
          causal_role_source = 'llm'
          AND (evaluator_audit->>'satisfied')::boolean = true
      )
  )
"""

# Newest first — Phase 7.2 will keep the latest attribution per feature.
_ORDER_BY = "\nORDER BY written_at DESC"


def query_active_role_attributions(
    experiment_id: str,
    *,
    only_evaluator_satisfied: bool = True,
    conn: Optional[Any] = None,
    database_url: Optional[str] = None,
) -> list[RoleAttribution]:
    """Return active ``RoleAttribution`` rows for ``experiment_id``.

    Args:
        experiment_id: Sidecar / verdict ``experiment_id`` to filter on.
        only_evaluator_satisfied: When True (default), LLM-sourced rows
            must have ``evaluator_audit.satisfied=true`` to be returned.
            Manifest and KG rows always come back. Pass ``False`` to
            include unsatisfied LLM rows (audit / debug use cases).
        conn: Optional pre-opened psycopg connection (testing seam +
            production caller-owned connection pool). When omitted, a
            short-lived connection is opened from ``database_url`` or
            ``$DATABASE_URL``.
        database_url: Postgres connection string. Falls back to
            ``$DATABASE_URL``. Ignored when ``conn`` is supplied.

    Returns:
        List of ``RoleAttribution`` rows (typed dicts). Empty list when
        no rows match.

    Raises:
        RuntimeError: when neither ``conn`` nor a resolvable database URL
            is available.
    """
    sql = _BASE_SELECT
    if only_evaluator_satisfied:
        sql = sql + _SATISFIED_FILTER
    sql = sql + _ORDER_BY

    params: tuple[Any, ...] = (experiment_id,)

    if conn is not None:
        rows = _fetch(conn, sql, params)
    else:
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "query_active_role_attributions: no conn supplied and "
                "neither database_url nor $DATABASE_URL is set"
            )
        # Lazy import — psycopg is a heavy optional dep; tests stub the
        # ``conn`` parameter so the production import path is exercised
        # only at runtime when an actual DB is configured.
        import psycopg  # type: ignore[import-not-found]

        with psycopg.connect(url) as opened:
            rows = _fetch(opened, sql, params)

    return [attr for attr in (_row_to_attribution(row) for row in rows) if attr is not None]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _fetch(conn: Any, sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    """Execute ``sql`` against ``conn`` and return all rows.

    Mirrors the cursor protocol used in
    ``scripts/query_audit_trail.py``.
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def _row_to_attribution(
    row: tuple[Any, ...],
) -> Optional[RoleAttribution]:
    """Convert a DB row to a typed ``RoleAttribution`` or ``None``.

    Defensive: a row with NULL ``causal_role_final`` / ``causal_role_source``
    should not exist given the copresence CHECK constraint
    (``chk_adaptive_validity_verdicts_role_copresence`` in migration 041
    §4), but the guard preserves the RoleAttribution typed-dict
    invariants under hypothetical schema drift.
    """
    feature, causal_role_final, causal_role_source, evaluator_audit, verdict = row

    if not isinstance(feature, str) or not feature:
        return None
    if not isinstance(causal_role_final, str) or not causal_role_final:
        return None
    if causal_role_source not in ("manifest", "llm", "kg"):
        return None

    # evaluator_satisfied: derived from source + audit per C1.
    # - manifest|kg: True unconditionally.
    # - llm: read from evaluator_audit.satisfied; conservatively False
    #   when missing or unparseable (absence of evidence = evidence of
    #   absence under the C1 conservative-failure rule).
    if causal_role_source in ("manifest", "kg"):
        evaluator_satisfied = True
    else:  # llm
        # Codex audit (PR #367): an LLM row missing ``evaluator_audit`` is
        # malformed/incomplete — skip defensively rather than converting to
        # ``evaluator_satisfied=False``. Letting such rows through with a
        # downgraded flag would still expose them at the consumer boundary
        # (filtered out only by ``should_act``); skipping at the conversion
        # layer is the conservative-failure choice under C1.
        if not isinstance(evaluator_audit, dict):
            return None
        raw = evaluator_audit.get("satisfied")
        evaluator_satisfied = raw is True

    # evaluator_model: pulled from ``verdict.evaluator_model`` for llm
    # sources; sentinel for manifest; KG sentinel for kg.
    evaluator_model: str
    if causal_role_source == "manifest":
        evaluator_model = "n/a"
    elif causal_role_source == "kg":
        evaluator_model = "kg:falkordb"
    else:  # llm
        candidate = verdict.get("evaluator_model") if isinstance(verdict, dict) else None
        evaluator_model = candidate if isinstance(candidate, str) and candidate else "<unknown>"

    return RoleAttribution(
        feature=feature,
        causal_role=causal_role_final,
        source=causal_role_source,  # type: ignore[typeddict-item]
        evaluator_satisfied=evaluator_satisfied,
        evaluator_model=evaluator_model,
    )
