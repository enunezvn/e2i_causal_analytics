"""Honest robustness-gate verdict phrases, shared by every surface (#1868).

"Survived all robustness checks" used to be emitted for EVERY proceed gate —
but PROCEED deliberately tolerates a critical test in WARNING, and a
non-critical test can even be FAILED under a proceed gate, so the phrase
over-claimed (observed live: a proceed-gated analysis whose own payload said
tests_passed 2/3). The phrase is now a function of the per-test verdicts:

- all executed tests PASSED           -> "survived all N robustness checks"
- warnings / non-critical failures    -> "passed the robustness gate (X of N
                                          checks passed; ...named caveats)"
- legacy two-state data (no status)   -> counts only, no caveat claims
- no per-test data at all             -> "passed the robustness gate"

This module is deliberately dspy-free: route modules import it EAGERLY, while
``clinical_narrative`` pulls dspy at import time and is therefore only ever
imported function-locally by routes (memory-capped prod container).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

# Contract test keys -> prose labels (matches the FE's display names in spirit;
# the sensitivity test is what the UI calls "Unobserved Common Cause").
_TEST_LABELS = {
    "placebo_treatment": "placebo-treatment",
    "random_common_cause": "random-common-cause",
    "data_subset": "data-subset",
    "bootstrap": "bootstrap-stability",
    "unobserved_common_cause": "unmeasured-confounding sensitivity",
    "sensitivity_e_value": "unmeasured-confounding sensitivity",
}


def test_label(test_name: str) -> str:
    """Prose label for a contract test key (unknown keys degrade readably)."""
    return _TEST_LABELS.get(test_name, test_name.replace("_", "-"))


def _join(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return names[0] if names else ""
    return ", ".join(names[:-1]) + " and " + names[-1]


def _verdict(t: Mapping[str, Any]) -> str:
    """Three-state verdict when present; legacy two-state ``passed`` otherwise.
    A legacy not-passed entry is 'unknown' — it may be a warning OR a
    non-critical failure, and the phrase must claim neither."""
    s = str(t.get("status") or "").lower()
    if s in ("passed", "warning", "failed"):
        return s
    return "passed" if t.get("passed") else "unknown"


def gate_verdict_phrase(
    gate: Optional[str],
    tests: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Optional[str]:
    """The verdict clause for "the estimate <phrase>" sentences.

    Returns None for an unmapped/absent gate so call sites keep their existing
    raw-report ("Robustness gate: pass.") / "robustness unknown" handling.
    """
    g = (gate or "").lower()
    if g == "review":
        return "needs review (mixed robustness)"
    if g == "block":
        return "failed robustness checks"
    if g != "proceed":
        return None

    entries = [t for t in (tests or []) if isinstance(t, Mapping)]
    if not entries:
        return "passed the robustness gate"

    verdicts = [(str(t.get("test_name") or ""), _verdict(t)) for t in entries]
    n = len(verdicts)
    n_passed = sum(1 for _, v in verdicts if v == "passed")
    if n_passed == n:
        return f"survived all {n} robustness checks"

    warn_names = [test_label(name) for name, v in verdicts if v == "warning"]
    fail_names = [test_label(name) for name, v in verdicts if v == "failed"]
    clauses = []
    if warn_names:
        noun = "a warning" if len(warn_names) == 1 else "warnings"
        clauses.append(
            f"the {_join(warn_names)} check{'s' if len(warn_names) > 1 else ''} raised {noun}"
        )
    if fail_names:
        # A FAILED critical test forces BLOCK, so under a proceed gate any
        # failure here is by construction non-critical.
        clauses.append(
            f"the {_join(fail_names)} check{'s' if len(fail_names) > 1 else ''} "
            "failed but does not gate the estimate"
        )
    if clauses:
        return f"passed the robustness gate ({n_passed} of {n} checks passed; {'; '.join(clauses)})"
    return f"passed the robustness gate ({n_passed} of {n} checks fully passed)"
