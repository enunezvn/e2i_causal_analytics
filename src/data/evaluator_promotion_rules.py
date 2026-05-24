"""Audit-evaluator promotion rules (Stage 1 of Issue #240).

Design reference: ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3
(Stage 1 — Shadow mode) and §4 (rules R1 / R2 / R3).

Stage 1 contract
================

This module is **shadow-mode** in Stage 1. The voter
(``src.data.kg.ensemble_voter.EnsembleVoter``) does NOT call these
functions; only ``_ensemble_to_legacy_dict`` in
``src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check``
does — for the sole purpose of populating three nullable shadow columns
on the ``adaptive_validity_verdicts`` sidecar / table. The functions are
PURE: no LM calls, no I/O, no mutation of inputs.

The byte-identity invariant (acceptance criterion AC1.2 of the design
doc) is enforced by
``tests/integration/test_audit_evaluator_shadow_byte_identity.py``: with
the same input, ``EnsembleVerdict.severity`` and
``EnsembleVerdict.remediation`` flowing into ``_ensemble_to_legacy_dict``
are identical whether the rule functions return their real verdict or
``None``. Any change to this module that violates that invariant — for
example, by mutating any argument or by being called from a code path
that consumes the return value to mutate a decision — is a regression.

Stage 2 (curation surfacing) and Stage 3 (soft-gate severity modulation,
env-var-gated) consume the same rules; promoting a rule to Stage 3 must
not require any signature change in this module.
"""

from __future__ import annotations

from typing import Callable, Optional

from src.data.kg.types import LLMEvaluatorAudit

# ---------------------------------------------------------------------------
# R1 — Moderate→High escalation on dissatisfied evaluator (Stage-3-eligible).
# ---------------------------------------------------------------------------


def evaluate_r1(
    worker_severity: str,
    evaluator_audit: Optional[LLMEvaluatorAudit],
) -> Optional[str]:
    """Trigger: ``worker_severity == "moderate"`` AND
    ``evaluator_audit.satisfied == False`` AND
    ``len(evaluator_audit.missed_considerations) >= 1``.

    Stage 1 action: returns ``"high"`` when the trigger fires (the
    sidecar/column records the proposed severity), else ``None``.

    Stage 3 action (env-var-gated; NOT live at Stage 1): the voter
    substitutes ``severity="high"`` and lets the deterministic
    remediation helper recompute ``remediation``.
    """
    if evaluator_audit is None:
        return None
    if worker_severity != "moderate":
        return None
    if evaluator_audit.satisfied is not False:
        return None
    if len(evaluator_audit.missed_considerations) < 1:
        return None
    return "high"


# ---------------------------------------------------------------------------
# R2 — Flag for review on multiple missed considerations (curation only).
# ---------------------------------------------------------------------------


def evaluate_r2(
    worker_severity: str,  # accepted for registry uniformity; unused
    evaluator_audit: Optional[LLMEvaluatorAudit],
) -> Optional[bool]:
    """Trigger: ``evaluator_audit.satisfied == False`` AND
    ``len(evaluator_audit.missed_considerations) >= 2``.

    Worker severity is intentionally ignored: R2 routes the row to
    curation review regardless of severity. The argument is kept in the
    signature so every rule in :data:`PROMOTION_RULES` shares the same
    ``(worker_severity, evaluator_audit)`` shape and the iteration site
    in ``_ensemble_to_legacy_dict`` does not branch per-rule.

    Stage 1 action: returns ``True`` when the trigger fires, else ``None``.

    Stage 3 action: NONE. R2 is a curation accelerator only and is
    deliberately not promoted to runtime severity modulation per design
    §4 R2.
    """
    del worker_severity  # explicitly unused
    if evaluator_audit is None:
        return None
    if evaluator_audit.satisfied is not False:
        return None
    if len(evaluator_audit.missed_considerations) < 2:
        return None
    return True


# ---------------------------------------------------------------------------
# R3 — Rationale-incomplete soft flag (audit-only forever).
# ---------------------------------------------------------------------------


def evaluate_r3(
    worker_severity: str,  # accepted for registry uniformity; unused
    evaluator_audit: Optional[LLMEvaluatorAudit],
) -> Optional[bool]:
    """Trigger: ``evaluator_audit.rationale_complete == False``.

    Independent of ``satisfied`` and ``missed_considerations``: R3
    captures documentation-quality issues even on otherwise-passing
    worker verdicts.

    Stage 1 action: returns ``True`` when the trigger fires, else ``None``.

    Stage 3 action: NEVER promoted (design §4 R3). This is a
    documentation-quality signal, not a correctness signal.
    """
    del worker_severity  # explicitly unused
    if evaluator_audit is None:
        return None
    if evaluator_audit.rationale_complete is not False:
        return None
    return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Type of a rule callable. Return type is intentionally ``Optional[object]``
# because each rule returns a different concrete shape (R1: str, R2/R3: bool).
# The caller in ``_ensemble_to_legacy_dict`` maps the rule_id to the
# correct sidecar key, so the heterogeneous return is contained at the
# write boundary.
RuleFn = Callable[[str, Optional[LLMEvaluatorAudit]], Optional[object]]

PROMOTION_RULES: tuple[tuple[str, RuleFn], ...] = (
    ("R1", evaluate_r1),
    ("R2", evaluate_r2),
    ("R3", evaluate_r3),
)
"""Stage-1 rule registry. Iterated by ``_ensemble_to_legacy_dict`` so a
single call site populates all three shadow columns. The tuple-of-pairs
shape (instead of a dict) preserves a stable iteration order for the
byte-identity invariant test."""
