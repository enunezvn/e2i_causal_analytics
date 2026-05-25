"""Deterministic, non-LLM leakage × role cross-check (Issue #501 / #240).

Design reference: ``docs/plans/501-leakage-role-crosscheck.md``.

Stage contract
==============

This module is **shadow-mode** and **non-acting**. It fires when the LLM
causal-role classifier assigns a feature a "keep-as-clean-predictor" role
(``ancestor`` / ``confounder`` / ``instrument``) BUT the platform's statistical
``detect_leakage`` independently flags that SAME feature as a leak at severity
``critical`` or ``high``.

The statistical detector reasons on feature values-vs-target — a genuinely
different mechanism than the LLM (which reasons on name/derivation metadata) —
so it is the independent leak signal. This is defense-in-depth: the role
classifier currently shows zero leak-FN on the golden set, so the cross-check
MUST be additive and shadow only. It MUST NOT change ``leakage_severity``,
routing, the voter, or any existing decision.

The functions here are **PURE**: no LM calls, no I/O, no mutation of inputs.

Shadow invariant (the load-bearing safety test)
===============================================

Adding ``would_flag_role_leak_disagreement`` to the sidecar must not change
any other verdict field. Enforced by
``tests/integration/test_leak_crosscheck_shadow_byte_identity.py``.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENIGN_KEEP_ROLES: frozenset[str] = frozenset({"ancestor", "confounder", "instrument"})
"""LLM causal-role values that classify a feature as safe to keep.

When the LLM assigns one of these roles yet the statistical detector flags
the same feature as a critical/high-severity leak, the disagreement is
surfaced as ``would_flag_role_leak_disagreement=True`` in the sidecar.
"""

STAT_FLAG_SEVERITIES: frozenset[str] = frozenset({"critical", "high"})
"""Statistical leakage severities that constitute a meaningful flag.

Matches ``_get_leaked_features`` in ``leakage_detector.py`` which keeps only
critical/high findings. ``moderate`` and ``info`` are intentionally excluded:
they represent weak or ambiguous signals that do not warrant a disagreement flag.
"""


# ---------------------------------------------------------------------------
# Cross-check function
# ---------------------------------------------------------------------------


def evaluate_role_vs_statistical_leak(
    llm_role: Optional[str],
    statistical_leak_severity: Optional[str],
) -> Optional[bool]:
    """Evaluate whether a feature's LLM role contradicts its statistical leak flag.

    Returns ``True`` (flag the disagreement) iff:

    * ``llm_role`` is one of the benign keep-as-clean-predictor roles
      (``BENIGN_KEEP_ROLES``: ``ancestor``, ``confounder``, ``instrument``);
      AND
    * ``statistical_leak_severity`` is in ``STAT_FLAG_SEVERITIES``
      (``critical`` or ``high``).

    Returns ``None`` in all other cases:

    * Either input is ``None`` — no signal to cross-check.
    * The role is NOT in ``BENIGN_KEEP_ROLES`` (e.g. ``descendant``,
      ``mediator``, ``collider``) — these roles already imply the feature
      is problematic; no cross-check disagreement to surface.
    * The statistical severity is below the threshold (``moderate``,
      ``info``) — not a strong enough independent signal.

    This function is PURE: it does not call any LM, perform any I/O,
    or mutate its inputs.

    Args:
        llm_role: The causal role assigned by the LLM classifier
            (``CausalRoleClassifier.causal_role``). ``None`` when the
            classifier was disabled, failed, or did not run for this feature.
        statistical_leak_severity: The maximum severity from
            ``state["leakage_findings"]`` entries for this feature, as
            produced by ``detect_leakage``. ``None`` when the feature
            has no statistical leakage finding.

    Returns:
        ``True`` — LLM says keep-clean but stats say leak (the disagreement).
        ``None`` — either input is absent, role is non-benign, or severity
        is below the threshold. Explicitly ``None`` (not ``False``) so the
        sidecar column is NULL when the cross-check did not fire, consistent
        with the existing shadow-column nullable contract (``would_promote_severity``
        / ``would_flag_for_review`` / ``rationale_incomplete_flag``).
    """
    if llm_role is None or statistical_leak_severity is None:
        return None
    if llm_role not in BENIGN_KEEP_ROLES:
        return None
    if statistical_leak_severity not in STAT_FLAG_SEVERITIES:
        return None
    return True
