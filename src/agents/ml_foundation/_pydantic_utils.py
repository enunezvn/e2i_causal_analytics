"""Shared pydantic v2 utilities for ml_foundation agent schemas.

This module establishes the base class and helpers that the per-agent
``schemas.py`` files and ``state.py`` modules consume after the
TypedDict → Pydantic v2 migration tracked in
``.claude/plans/typeddict_to_pydantic_migration_plan_20260504.md``.

ARC STATUS (post-2026-05-05):
- Core 4-shard migration arc CLOSED (PRs #48, #49, #50, #51).
- TypedDict→Pydantic follow-up arc CLOSED across 12 PRs:
  - PR #57 — D5 (np.ndarray serializer for shap_values: WONT-FIX, documented).
  - PR #58 — D1.1 (orchestrator audit_workflow_id threading; production fix).
  - PR #59 — D2.0 (registry_manager auc_roc/roc_auc producer-key fix).
  - PR #60 — D2.1 (hyperparameter_search_space typed schema wiring).
  - PR #61 — D2.2 (qc_report typed schema wiring; remove runner shim).
  - PR #62 — D1.2 (per-agent input_data → State threading).
  - PR #63 — D2.3 (success_criteria typed schema wiring).
  - PR #64 — D2.4 (scope_spec typed schema wiring; 24 missing fields).
  - PR #65 — D1.3 (workflow-level UUID minting + conftest fixture).
  - PR #66 — D2.5 (validation_metrics typed schema wiring + AliasChoices).
  - PR #67 — D3 (extra="allow" → extra="ignore" tightening).
  - PR #68 — D1.4 (this PR; arc-closure documentation).

The base class is in its STEADY-STATE configuration:
- ``extra="ignore"`` (no longer the transition-window ``allow``).
- Per-class overrides where needed (e.g., ``SuccessCriteriaSchema``
  retains ``extra="allow"`` for underscore-prefixed adaptive audit keys).
- ``audit_workflow_id`` retains ``Field(default_factory=uuid4)`` as a
  documented transition mechanism — production paths thread caller-
  provided UUIDs via D1.1 + D1.2; test fixtures may still rely on the
  default. Strict-required tightening deferred indefinitely (~50+ test
  sites would need rewriting; not load-bearing for correctness).

The plan keeps schemas per-agent (Decision 1b) so each agent owns its
output contracts. This module is the *only* shared piece — it captures
two cross-cutting concerns that EVERY agent needs:

1. ``BaseAgentSchema`` — a common ``BaseModel`` configured for the
   STEADY-STATE permissiveness contract (post-D3, 2026-05-05). The
   ``extra="ignore"`` setting silently drops undeclared keys at
   construction time. Production hot-path is unaffected: LangGraph 1.0
   already drops ``model_extra`` at every channel boundary anyway (the
   same mechanism that motivated D4). The dict-shim ``__setitem__``
   continues to populate ``__pydantic_extra__`` directly so
   ``state["foo"] = v`` writes still work.

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

    Configuration:
    - ``extra="ignore"``: D3 (2026-05-05) tightened from the migration's
      transition-window ``extra="allow"`` once all 7 ml_foundation
      agents landed in the pydantic world AND the D2 typed-schema
      wirings closed every cross-agent ``Dict[str, Any]`` surface.
      Tightening to ``ignore`` catches typo'd construction kwargs
      (``ScopeDefinerState(typo_field=...)`` previously stored
      ``typo_field`` in ``model_extra`` silently; under ``ignore`` the
      key is dropped — no residue masking the typo for a debugger).
      Note: LangGraph 1.0 already drops ``model_extra`` at every
      channel boundary (the same mechanism that motivated D4), so
      production hot-path runtime behavior is unaffected by this flip.
      Only test fixtures that explicitly inspect ``model_extra`` see
      a contract change. The dict-shim ``__setitem__`` continues to
      populate ``__pydantic_extra__`` directly so ``state["foo"] = v``
      writes still work for compatibility with the migration's
      transition test fixtures.
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
        extra="ignore",  # D3 (2026-05-05): tightened from "allow"; see class docstring.
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

    def items(self):  # type: ignore[no-untyped-def]
        """Dict-like ``items()`` for iteration patterns like
        ``for k, v in state.items()``.

        Yields (key, value) pairs from BOTH declared fields AND
        ``model_extra`` contents. Skips fields whose value is ``None``
        — matches ``get(key, default)`` coalescing semantics so a
        consumer iterating ``items()`` sees the same set of keys it
        would see via ``get()`` returning a non-default value.

        Added (D2.3 follow-up, 2026-05-05): consumers like
        ``model_trainer/nodes/evaluator.py:2465`` iterate
        ``success_criteria.items()`` after the schema-typed wiring; the
        underlying pydantic BaseModel does not provide ``items()``, so
        we expose it here. Skipping None values is intentional — a
        criterion threshold of ``None`` means "not configured" and the
        evaluator's loop should treat it as absent.
        """
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if value is not None:
                yield field_name, value
        extra = self.model_extra or {}
        for k, v in extra.items():
            if v is not None:
                yield k, v

    def keys(self):  # type: ignore[no-untyped-def]
        """Dict-like ``keys()`` companion to ``items()``.

        Yields keys from declared fields with non-None values + all
        ``model_extra`` keys with non-None values. See ``items()`` for
        the None-skip rationale.
        """
        for k, _ in self.items():
            yield k

    def values(self):  # type: ignore[no-untyped-def]
        """Dict-like ``values()`` companion to ``items()``.

        Yields values for declared non-None fields + ``model_extra``
        non-None values. See ``items()`` for the None-skip rationale.
        """
        for _, v in self.items():
            yield v


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
