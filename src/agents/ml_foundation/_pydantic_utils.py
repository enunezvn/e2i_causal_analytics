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

    TypedDict-compat dict-like accessors:
    The ``__getitem__`` / ``__setitem__`` / ``__contains__`` / ``get``
    methods exist so that the 270+ ``state["key"]`` / ``state.get(...)``
    call sites across the ml_foundation node files keep working
    unchanged after their corresponding ``State`` class is migrated
    from ``TypedDict(total=False)`` to a pydantic v2 ``BaseModel``.
    Without these methods, every node's state-access call site would
    need a separate edit, which would explode Shard A/B/C blast radius
    by ~1000+ call sites.

    Semantics: ``state.get("key", default)`` returns ``default`` when
    the field is missing OR the value is ``None``. This matches the
    TypedDict ``total=False`` semantics where unset fields are
    indistinguishable from ``None`` at a call site that uses
    ``.get(key, default)``.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        # validate_assignment=True is load-bearing per codex review I1
        # (2026-05-05): without it, ``state["status"] = "off_spec_value"``
        # bypasses pydantic field validation via ``setattr`` (see
        # ``__setitem__`` below), silently storing values that violate
        # Literal/Annotated/etc. constraints. With it on, every assignment
        # triggers the full validator pipeline — slightly more expensive
        # but catches typos and off-spec writes that would otherwise
        # surface as silent runtime drift. Decision 4b (accept perf cost
        # in exchange for type safety).
        validate_assignment=True,
    )

    def __getitem__(self, key: str) -> Any:
        """Dict-like read access. Raises ``KeyError`` if the key is
        absent (neither a declared field nor in ``model_extra``).

        Note: returns ``None`` when a declared field exists with value
        ``None``; ``KeyError`` only fires for genuinely-unknown keys.
        Existing ``state["key"]`` call sites that previously raised
        ``KeyError`` on missing-from-TypedDict keys will now return
        ``None`` for declared-Optional fields. This is the migration's
        intentional semantic shift — see Decision 8a in the plan.
        """
        if key in type(self).model_fields:
            return getattr(self, key)
        extra = self.model_extra or {}
        if key in extra:
            return extra[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like write access for partial-state updates.

        Routes declared fields to attribute assignment; routes unknown
        keys to ``model_extra`` (preserved by ``extra="allow"``). This
        matches the TypedDict semantic where any string key was a
        valid update target.
        """
        if key in type(self).model_fields:
            setattr(self, key, value)
            return
        if self.model_extra is None:
            object.__setattr__(self, "__pydantic_extra__", {})
        # mypy can't see __pydantic_extra__ on BaseModel; runtime is fine.
        self.__pydantic_extra__[key] = value  # type: ignore[index]

    def __contains__(self, key: object) -> bool:
        """Dict-like ``key in state`` check.

        Returns ``True`` when ``key`` is a declared field (regardless
        of value) OR present in ``model_extra``. Note: ``True`` even
        when the field's value is ``None`` — a declared field is
        always considered "present" in pydantic semantics.

        SEMANTIC ASYMMETRY warning (per codex review I2, 2026-05-05):
        ``key in state`` and ``state.get(key, default)`` disagree on
        declared-but-None fields. Specifically:

        - ``"minimum_auc" in state`` → ``True`` (declared field exists)
        - ``state.get("minimum_auc", 0.5)`` → ``0.5`` (None coalesced)

        This is intentional — see ``get()`` docstring for the rationale.
        Code that wants to distinguish "field set" from "field None"
        should use the contains check, then attribute access:

        .. code-block:: python

            if "minimum_auc" in state and state.minimum_auc is not None:
                ...

        ``state.get("minimum_auc")`` (no default) returns ``None`` for
        BOTH "absent" and "set to None" — same shim trade-off.
        """
        if not isinstance(key, str):
            return False
        if key in type(self).model_fields:
            return True
        extra = self.model_extra or {}
        return key in extra

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like ``get`` with TypedDict-compat semantics.

        Returns ``default`` when:
        - The key is genuinely absent (neither declared nor in
          ``model_extra``), OR
        - The key resolves to a value of ``None``.

        The "None → default" coalescing is the migration's
        compatibility shim. Pydantic-Optional fields default to
        ``None`` (Decision 8a), but the ~232 ``state.get("key", X)``
        call sites in node code expect ``X`` returned when the field
        was never set. Conflating absent + None preserves that
        intent without a per-call-site rewrite.

        For users that need to distinguish "missing field" from
        "field set to None", use ``key in state`` (which discriminates)
        followed by attribute access.
        """
        try:
            value = self[key]
        except KeyError:
            return default
        return default if value is None else value


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
