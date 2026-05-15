"""Haiku audit evaluator for Layer-4 (Adaptive Temporal-Validity Redesign).

Plan: ``.claude/plans/layer4_evaluator_audit_signal.md``.

This module is **audit-only**. The :class:`CausalRoleEvaluator` reads the
worker's :class:`LLMVerdict` and emits a :class:`LLMEvaluatorAudit` that
the orchestrator stamps onto the verdict's audit fields. It does not
gate, does not override, and does not loop. The voter and the issue-#212
cap path do not read these fields.

The criteria text is the load-bearing contract: it tells the evaluator
what an adequate worker rationale must cover. Changing the criteria
changes the audit shape; coordinate any change with the audit-schema
consumers documented in ``write_adaptive_verdicts_sidecar``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import dspy

from src.data.kg.types import LLMEvaluatorAudit, LLMVerdict

logger = logging.getLogger(__name__)


EVALUATOR_CRITERIA: str = """\
You are an audit reviewer (NOT a verdict gate) for a pharma analytics
causal-role classifier. The worker has classified one feature with a
causal_role, a mechanism rationale, and a recommended_remediation.

Your job is to read the worker's outputs and emit a structured audit on
whether the rationale meets the following criteria. You DO NOT change
the verdict; the orchestrator uses your audit only as a downstream
signal.

Adequate rationale criteria:
  1. Temporal filter: the mechanism either (a) cites the prefix-censoring
     window or feature lookup horizon, or (b) explicitly notes that the
     derivation has no temporal filter and explains why that is
     acceptable.
  2. Pearl arrowheads: the mechanism identifies which Pearl causal arrows
     are claimed (ancestor edge into target, fork from a common cause,
     collider at the target, instrument constraint, etc.).
  3. Remediation mapping: the recommended_remediation matches the role
     per the documented role-to-remediation mapping (ancestor/confounder
     -> keep_with_caveat; collider/descendant -> drop; instrument ->
     keep_with_caveat). The voter uses keep_with_caveat for accept-roles
     when confidence is below promotion thresholds.
  4. No leakage red flags missed: the mechanism does not silently ignore
     a post-anchor data dependency visible in the derivation_pseudocode.

If ANY criterion fails, set satisfied=False and list the failing axes in
missed_considerations.

This is an AUDIT step. Be conservative: if uncertain, set satisfied=False
and note the uncertainty. You are NOT the verdict.
"""


DEFAULT_EVALUATOR_MODEL: str = "anthropic/claude-haiku-4-5-20251001"
ENABLE_ENV_VAR: str = "ADAPTIVE_VALIDITY_EVALUATOR_ENABLED"
MODEL_ENV_VAR: str = "ADAPTIVE_VALIDITY_EVALUATOR_MODEL"


def evaluator_is_enabled() -> bool:
    """Return True iff the operator has explicitly opted in via env.

    Default is OFF. Treat any value other than ``"1"`` / ``"true"`` /
    ``"yes"`` / ``"on"`` (case-insensitive) as disabled, including
    missing.
    """
    raw = os.environ.get(ENABLE_ENV_VAR, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _evaluator_lm_is_configured() -> bool:
    """Mirror of ``causal_role_classifier_loader._lm_is_configured`` for
    the evaluator's Anthropic credential.

    Returns False when ``ANTHROPIC_API_KEY`` is unset or empty.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def resolve_evaluator_model() -> str:
    """Return the configured Haiku model string, or the default."""
    raw = os.environ.get(MODEL_ENV_VAR, "").strip()
    return raw or DEFAULT_EVALUATOR_MODEL


class CausalRoleEvaluatorSignature(dspy.Signature):
    """DSPy signature for the Haiku audit evaluator."""

    feature_name: str = dspy.InputField(desc="Feature being audited.")
    derivation_pseudocode: str = dspy.InputField(
        desc="The worker's input — plain-English or pseudo-code describing "
        "how the feature is derived."
    )
    dataset_context: str = dspy.InputField(desc="Target + cohort + prediction-anchor context.")
    worker_causal_role: str = dspy.InputField(desc="The worker's causal_role classification.")
    worker_mechanism: str = dspy.InputField(desc="The worker's mechanism rationale.")
    worker_recommended_remediation: str = dspy.InputField(
        desc="The worker's recommended_remediation."
    )
    criteria: str = dspy.InputField(
        desc="Audit criteria text. The evaluator must check the worker's "
        "outputs against this contract."
    )

    satisfied: bool = dspy.OutputField(
        desc="True iff ALL criteria pass. Conservative: False if uncertain."
    )
    rationale_complete: bool = dspy.OutputField(
        desc="True iff the worker's mechanism cites the temporal filter "
        "(or lack of one) AND identifies Pearl arrowheads."
    )
    missed_considerations: str = dspy.OutputField(
        desc="Comma-separated short labels (≤5 items) for failing axes. Empty when satisfied=True."
    )
    notes: str = dspy.OutputField(desc="Free-text rationale. Will be truncated to 500 chars.")


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return False


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _parse_missed_considerations(raw: object) -> tuple[str, ...]:
    """Coerce the LM's comma-separated string into a tuple of short labels.

    Returns an empty tuple on missing / non-string input. Caps at 5
    items; each item truncated to 80 chars.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ()
    items = [s.strip()[:80] for s in raw.split(",") if s.strip()]
    return tuple(items[:5])


@dataclass
class CausalRoleEvaluator:
    """Haiku audit evaluator wrapping a DSPy module.

    Use the ``module`` argument to inject a stub DSPy module in unit
    tests. Production callers pass ``None`` and the constructor builds
    ``dspy.ChainOfThought(CausalRoleEvaluatorSignature)``.
    """

    module: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.module is None:
            self.module = dspy.ChainOfThought(CausalRoleEvaluatorSignature)

    def evaluate(
        self,
        *,
        feature_name: str,
        derivation_pseudocode: str,
        dataset_context: str,
        worker_verdict: LLMVerdict,
        evaluator_model: str,
    ) -> LLMEvaluatorAudit:
        """Run the evaluator on one worker verdict and return an audit.

        Failures raised by the underlying DSPy module propagate to the
        caller; ``classify_feature`` is responsible for catching them
        and returning the worker's verdict with ``evaluator_audit=None``.
        """
        prediction = self.module(
            feature_name=feature_name,
            derivation_pseudocode=derivation_pseudocode,
            dataset_context=dataset_context,
            worker_causal_role=worker_verdict.causal_role,
            worker_mechanism=worker_verdict.mechanism,
            worker_recommended_remediation=worker_verdict.recommended_remediation,
            criteria=EVALUATOR_CRITERIA,
        )
        notes_raw = _coerce_str(getattr(prediction, "notes", ""))
        return LLMEvaluatorAudit(
            satisfied=_coerce_bool(getattr(prediction, "satisfied", False)),
            rationale_complete=_coerce_bool(getattr(prediction, "rationale_complete", False)),
            missed_considerations=_parse_missed_considerations(
                getattr(prediction, "missed_considerations", None)
            ),
            notes=notes_raw[:500],
            evaluator_model=evaluator_model,
        )
