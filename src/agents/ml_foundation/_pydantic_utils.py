"""Shared pydantic v2 utilities for ml_foundation agent schemas.

This module establishes the base class and helpers that the per-agent
``schemas.py`` files (and, after Shards A-C land, the per-agent
``state.py`` modules) consume during the TypedDict → Pydantic v2
migration tracked in
``.claude/plans/typeddict_to_pydantic_migration_plan_20260504.md``.

The plan keeps schemas per-agent (Decision 1b) so each agent owns its
output contracts. This module is the *only* shared piece — it captures
two cross-cutting concerns that EVERY agent needs:

1. ``BaseAgentSchema`` — a common ``BaseModel`` configured for the
   migration's permissiveness contract. The ``extra="allow"`` setting
   is load-bearing: during the multi-shard rollout, an agent migrated
   to pydantic may still receive partial state from a TypedDict-shaped
   upstream. Without ``extra="allow"`` those un-declared keys would
   trigger ``ValidationError`` and break the LangGraph reducer's
   merge-then-reduce semantics. Once all 7 agents are migrated, this
   permissiveness can be tightened in a follow-up PR.

2. ``coerce_uuid`` — the field-validator helper that lets every agent
   accept ``audit_workflow_id`` as either a ``UUID`` instance (when
   constructed in-process) or a ``str`` (when restored from a JSON
   checkpoint). Per Decision 7a in the migration plan, the audit
   chain integrity invariant requires UUID typing at the schema
   boundary while permitting string round-trips through Redis /
   FalkorDB / Postgres checkpoints.

Why ``audit_workflow_id`` is special: the central audit-trail
repository (``src/repositories/audit_chain/``) stitches multi-agent
invocations into a single workflow by joining on this field. A string
ID would silently lose the uniqueness guarantee the type system
provides; a missing field would corrupt the audit chain entirely.
This is why ``audit_workflow_id`` is the only field excluded from
the Decision 8a "Optional[T]=None as default" convention.
"""

from __future__ import annotations

from typing import Any, Callable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


class BaseAgentSchema(BaseModel):
    """Common base for ml_foundation agent state and output schemas.

    Configured for the migration's transition window:
    - ``extra="allow"``: tolerate undeclared keys flowing in from
      upstream agents that have not yet migrated. After all 7 agents
      land in the pydantic world, a follow-up PR can tighten this
      to ``extra="ignore"`` or ``extra="forbid"``.
    - ``arbitrary_types_allowed=True``: needed for fields like
      ``trained_model: Any`` (sklearn / xgboost), ``preprocessor: Any``
      (fitted pipeline), and ``shap_values: np.ndarray`` that appear
      in ``model_trainer`` and ``feature_analyzer`` states. Setting it
      at the base class avoids per-class repetition; subclasses that
      do NOT use arbitrary types pay no runtime cost.
    - ``populate_by_name=True``: lets aliased fields accept either
      the alias or the python attribute name during construction.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )


def coerce_uuid(value: Any) -> UUID:
    """Coerce a string or UUID into a UUID instance.

    Used as ``mode="before"`` field validator on ``audit_workflow_id``
    fields so that pydantic schemas accept both representations:

    - In-process construction: ``audit_workflow_id=UUID(...)``
      (the canonical path; type system catches errors at the call
      site).
    - Checkpoint restore: ``audit_workflow_id="..."``
      (JSON serialization round-trips through ``str(UUID)``;
      checkpoint replay deserializes as ``str`` and the validator
      coerces back to ``UUID``).

    Raises ``ValueError`` for malformed strings (not a valid UUID
    representation) — this is the right behavior because a malformed
    audit_workflow_id would corrupt the audit chain at workflow-level.
    Better to fail loud at schema validation than silently lose
    audit-chain integrity.
    """
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise ValueError(
        f"audit_workflow_id must be UUID or str, got {type(value).__name__}: {value!r}"
    )


def audit_workflow_id_validator() -> Callable[..., Any]:
    """Build a reusable ``field_validator`` for ``audit_workflow_id``.

    Per-agent schemas can attach this directly:

    .. code-block:: python

        from src.agents.ml_foundation._pydantic_utils import (
            BaseAgentSchema,
            audit_workflow_id_validator,
        )

        class ScopeDefinerState(BaseAgentSchema):
            audit_workflow_id: UUID
            ...
            _validate_audit_id = audit_workflow_id_validator()

    The factory pattern keeps the coercion logic in one place — every
    agent picks up bug fixes / clarifications without copy-paste drift.
    """
    return field_validator("audit_workflow_id", mode="before")(lambda cls, v: coerce_uuid(v))
