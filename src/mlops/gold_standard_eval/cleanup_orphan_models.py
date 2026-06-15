"""cleanup_orphan_models — safe decommission script for superseded gold-standard models.

Three pooled/single-brand models were superseded by the per-brand gold-standard models
introduced in P3 (gold_standard_model_eval_20260614).  This script retires EXACTLY those
three handles — and ONLY those three — by:

1. Resolving each handle to a registry UUID via ``_resolve_model_id``.
2. Counting (and optionally deleting) their ``ml_performance_metrics`` rows.
3. Setting ``ml_model_registry.stage = 'archived'`` on the registry row (reversible;
   does NOT hard-DELETE the registry row, preserving audit history and FK integrity).

The DEFAULT is a **dry-run** (``execute=False``): nothing is mutated; the script only
reports what WOULD happen.  Pass ``execute=True`` (or ``--execute`` on the CLI) to
actually apply the changes.

Safety invariants
-----------------
- ORPHAN_MODELS is a hardcoded tuple.  No wildcards, no user-supplied handles.
- An assertion guards every mutation path so only ORPHAN_MODELS handles are ever touched.
- Metrics DELETE precedes registry UPDATE to satisfy the FK RESTRICT constraint.
- Absent handles (already gone / never registered) are reported as "absent" and skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The ONLY three handles this script will ever touch.
# ---------------------------------------------------------------------------
ORPHAN_MODELS: tuple[str, ...] = (
    "csu_initiation_goldstd_lr_v1",  # P1 single Remi-only initiation → superseded by initiation_{brand}_goldstd_lr_v1
    "pnh_persistence_goldstd_lr_v1",  # P2 pooled persistence → superseded by persistence_{brand}_goldstd_lr_v1
    "pnh_discontinuation_goldstd_lr_v1",  # P2 pooled discontinuation → superseded by discontinuation_{brand}_goldstd_lr_v1
)


async def _resolve_client(db: Any) -> Any:
    """Return the async Supabase client; fail-closed when unconfigured.

    Mirrors the pattern in ``run_persistence_eval._resolve_client``:
    when ``db`` is supplied (e.g. from tests) it is used directly; otherwise
    the real client is resolved from the environment.
    """
    if db is not None:
        return db
    from src.memory.services.factories import (
        ServiceConnectionError,
        get_async_supabase_client,
    )

    client = await get_async_supabase_client()
    if client is None:
        raise ServiceConnectionError(
            "Supabase",
            "async Supabase client resolved to None for cleanup_orphan_models "
            "(refusing to run a no-op).",
        )
    return client


async def decommission(db: Any = None, *, execute: bool = False) -> dict[str, Any]:
    """Decommission the three superseded gold-standard model handles.

    Parameters
    ----------
    db:
        Optional async Supabase client.  When None the real client is resolved
        from the environment (fail-closed).  Tests inject a mock here.
    execute:
        When False (the default) the function is a pure dry-run — it reads the
        DB to build a plan but makes NO writes.  When True the metrics rows are
        deleted and the registry stage is set to 'archived'.

    Returns
    -------
    dict::

        {
          "executed": bool,
          "results": [
            {
              "handle": str,
              "model_id": str | None,
              "metrics_rows": int,
              "status": "archived" | "would_archive" | "absent",
            },
            ...
          ],
          "summary": {
            "total": int,
            "absent": int,
            "archived": int,
            "would_archive": int,
          },
        }
    """
    from src.repositories.drift_monitoring import _resolve_model_id

    client = await _resolve_client(db)

    if execute:
        logger.info(
            "cleanup_orphan_models: EXECUTE mode — mutations WILL be applied "
            "to ml_performance_metrics and ml_model_registry."
        )
    else:
        logger.info(
            "cleanup_orphan_models: DRY-RUN mode (default) — NO writes will be "
            "made.  Pass execute=True / --execute to apply changes."
        )

    results: list[dict[str, Any]] = []

    for handle in ORPHAN_MODELS:
        # Safety guard: this script must NEVER act on a non-orphan handle.
        assert handle in ORPHAN_MODELS, (
            f"BUG: attempted to act on non-orphan handle {handle!r}; this should be unreachable"
        )

        model_id = await _resolve_model_id(client, handle)

        if model_id is None:
            logger.info("[%s] absent — not found in ml_model_registry; skipping.", handle)
            results.append(
                {
                    "handle": handle,
                    "model_id": None,
                    "metrics_rows": 0,
                    "status": "absent",
                }
            )
            continue

        # Count existing ml_performance_metrics rows for this model.
        count_res = await (
            client.table("ml_performance_metrics")
            .select("id", count="exact")
            .eq("model_id", model_id)
            .execute()
        )
        metrics_rows: int = (
            count_res.count if count_res.count is not None else len(count_res.data or [])
        )

        if execute:
            # Step 1 — DELETE metrics first (FK RESTRICT: metrics → registry).
            await client.table("ml_performance_metrics").delete().eq("model_id", model_id).execute()
            logger.info(
                "[%s] Deleted %d ml_performance_metrics row(s).",
                handle,
                metrics_rows,
            )

            # Step 2 — Archive the registry row (reversible; preserves audit history).
            await (
                client.table("ml_model_registry")
                .update({"stage": "archived"})
                .eq("id", model_id)
                .execute()
            )
            logger.info(
                "[%s] ml_model_registry.stage set to 'archived' (id=%s).",
                handle,
                model_id,
            )
            status = "archived"
        else:
            logger.info(
                "[%s] DRY-RUN: would delete %d metrics row(s) and set stage='archived' "
                "(model_id=%s).",
                handle,
                metrics_rows,
                model_id,
            )
            status = "would_archive"

        results.append(
            {
                "handle": handle,
                "model_id": model_id,
                "metrics_rows": metrics_rows,
                "status": status,
            }
        )

    summary: dict[str, int] = {
        "total": len(results),
        "absent": sum(1 for r in results if r["status"] == "absent"),
        "archived": sum(1 for r in results if r["status"] == "archived"),
        "would_archive": sum(1 for r in results if r["status"] == "would_archive"),
    }

    logger.info("cleanup_orphan_models summary: %s", summary)

    return {
        "executed": execute,
        "results": results,
        "summary": summary,
    }


def main() -> None:
    """CLI entry point — dry-run by default; pass --execute to mutate."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Decommission the 3 superseded pooled/single-brand gold-standard models. "
            "DRY-RUN by default — pass --execute to apply changes."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help=(
            "Apply the decommission (delete metrics + archive registry rows). "
            "Without this flag the script only reports what WOULD happen."
        ),
    )
    args = parser.parse_args()

    if not args.execute:
        logger.info("DRY-RUN mode (default).  Pass --execute to actually mutate the DB.")

    report = asyncio.run(decommission(execute=args.execute))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
