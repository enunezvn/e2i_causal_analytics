"""Which ``ml_model_registry`` row a reader means (#2310, owner decision 2026-09-28, R3).

A retrain registers its row at stage ``candidate`` (migration 159) under the model_name of the
row it retrains and records ``retrain_of_id`` (migration 160). The two record different things:

* **role** (``stage``) says what a row is for NOW. ``candidate`` = a retrain awaiting review.
* **lineage** (``retrain_of_id``) says where a row came from. It is immutable and is never used
  to exclude a row: a retrain that is later promoted becomes canonical by a stage change alone.

The CANONICAL row for a name is one whose stage is not in :data:`NON_CANONICAL_STAGES`.
Readers pick rows in one of three ways, and this module is the shared vocabulary:

* a NAME handle (Monitoring UI, gold-standard eval re-records, the manual retrain trigger by
  name) → :func:`resolve_canonical_model_id` — canonical rows only, newest first;
* an exact identity → the row's uuid, or :func:`resolve_model_id_by_name_version`
  (``(model_name, model_version)`` is unique), which never filters by stage;
* a stage-scoped reader (``stage IN ('production','staging')``) needs nothing: ``candidate``
  is outside that set.

Promoting a candidate to a served stage is a separate, owner-gated step (R6) and is not here.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

TABLE = "ml_model_registry"

#: Stage of an unreviewed retrain (migration 159).
CANDIDATE_STAGE = "candidate"

#: Stages that never make a row the canonical one for its name.
NON_CANONICAL_STAGES: tuple[str, ...] = (CANDIDATE_STAGE, "archived", "deprecated")

#: The MLflow model-version tag the deployer sets on a retrain candidate (Lane A, #2310). The
#: candidate's MLflow ``current_stage`` stays ``None``, so the tag is how MLflow tells it apart.
MLFLOW_ROLE_TAG = "e2i.role"
MLFLOW_CANDIDATE_ROLE = "candidate"


#: The PostgREST ``or`` filter that keeps canonical rows. A NULL stage (the column has a default
#: but no NOT NULL) is kept: it is not a candidate. Needs migration 159 on the database:
#: ``'candidate'`` is a ``model_stage_enum`` literal and Postgres rejects an unknown enum label.
#: Chain it as ``.or_(CANONICAL_STAGE_FILTER)`` or call :func:`canonical_rows`.
CANONICAL_STAGE_FILTER = f"stage.is.null,stage.not.in.({','.join(NON_CANONICAL_STAGES)})"


def canonical_rows(query: Any) -> Any:
    """Restrict a PostgREST ``ml_model_registry`` query to canonical rows."""
    return query.or_(CANONICAL_STAGE_FILTER)


def non_candidate_rows(query: Any) -> Any:
    """Restrict a PostgREST ``ml_model_registry`` query to rows that are not retrain candidates
    (NULL stage kept). For readers already scoped by another role predicate (e.g. champion)."""
    return query.or_(f"stage.is.null,stage.neq.{CANDIDATE_STAGE}")


def is_canonical_stage(stage: Optional[str]) -> bool:
    return (stage or "").lower() not in NON_CANONICAL_STAGES


def is_mlflow_candidate(tags: Any) -> bool:
    """True when an MLflow model version carries the retrain-candidate role tag."""
    if not isinstance(tags, Mapping):
        return False
    return tags.get(MLFLOW_ROLE_TAG) == MLFLOW_CANDIDATE_ROLE


async def resolve_canonical_model_id(client: Any, handle: Optional[str]) -> Optional[str]:
    """The canonical registry id for a name-style handle, or ``None``.

    Looks the handle up as a ``model_version`` label, then as a ``model_name`` (the order the
    monitoring handles have always used), among canonical rows only, newest ``registered_at``
    first with ``id`` as the tie-break, so the answer never depends on physical row order.
    More than one canonical match is logged: it is legitimate (e.g. a development and a staging
    version of one name) but the caller gets the newest, and an operator should know.
    A lookup failure returns ``None`` (callers fall back to their preserved-handle path).
    """
    if not handle or client is None:
        return None
    try:
        for col in ("model_version", "model_name"):
            res = await (
                canonical_rows(client.table(TABLE).select("id,stage,registered_at").eq(col, handle))
                .order("registered_at", desc=True)
                .order("id", desc=True)
                .limit(2)
                .execute()
            )
            rows = res.data or []
            if rows:
                if len(rows) > 1:
                    logger.warning(
                        "ml_model_registry: more than one canonical row for %s=%r; using the "
                        "newest (id=%s, stage=%s) over id=%s",
                        col,
                        handle,
                        rows[0]["id"],
                        rows[0].get("stage"),
                        rows[1]["id"],
                    )
                return str(rows[0]["id"])
    except Exception as e:  # noqa: BLE001 — never block a recording on a lookup failure
        logger.warning("ml_model_registry canonical lookup failed for %r: %s", handle, e)
        return None
    return None


async def resolve_model_id_by_name_version(
    client: Any, model_name: str, model_version: str
) -> Optional[str]:
    """The id of the exact ``(model_name, model_version)`` row, whatever its stage.

    ``UNIQUE(model_name, model_version)`` (``database/ml/mlops_tables.sql``) makes this
    deterministic. For callers that mean one specific row, e.g. the row a re-registration is
    about to replace.
    """
    if client is None or not model_name or not model_version:
        return None
    res = await (
        client.table(TABLE)
        .select("id")
        .eq("model_name", model_name)
        .eq("model_version", model_version)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return str(rows[0]["id"]) if rows else None
