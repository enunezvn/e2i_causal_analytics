"""Unit tests for :mod:`src.data.leakage_role_crosscheck`.

Issue #501 / #240 — deterministic, non-LLM leakage cross-check.

TDD red-first: these tests MUST fail before ``leakage_role_crosscheck.py``
is created; once the module exists they must ALL pass (pure fn, no LM,
no I/O).

Test inventory
==============
* All 9 specified parametrized cases for ``evaluate_role_vs_statistical_leak``.
* Purity / no-mutation guard (the function may not modify its inputs in any
  way; callers pass string literals so the only risk is implementation bugs
  that inadvertently modify a mutable container, but we guard anyway).
* Constants guard — BENIGN_KEEP_ROLES and STAT_FLAG_SEVERITIES must contain
  the specified members and nothing else.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Parametrized cases specified in the design brief.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "llm_role, stat_severity, expected",
    [
        # Fire cases — benign keep-role + critical/high statistical flag.
        ("confounder", "high", True),
        ("ancestor", "critical", True),
        # Non-benign roles — should NOT fire even at critical/high severity.
        ("descendant", "high", None),
        ("mediator", "high", None),
        ("collider", "critical", None),
        # Benign role but severity below threshold — should NOT fire.
        ("confounder", "moderate", None),
        # Benign role but no statistical finding — should NOT fire.
        ("confounder", None, None),
        # No LLM role — should NOT fire regardless of severity.
        (None, "high", None),
        # Benign role + instrument (third benign role) at high — should fire.
        ("instrument", "high", True),
    ],
)
def test_evaluate_role_vs_statistical_leak(llm_role, stat_severity, expected):
    """Parametrized specification cases from the design brief."""
    from src.data.leakage_role_crosscheck import evaluate_role_vs_statistical_leak

    result = evaluate_role_vs_statistical_leak(llm_role, stat_severity)
    assert result is expected, (
        f"evaluate_role_vs_statistical_leak({llm_role!r}, {stat_severity!r}) "
        f"returned {result!r}, expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# Additional edge cases.
# ---------------------------------------------------------------------------


def test_returns_none_for_both_none():
    """Both inputs None → None (no signal at all)."""
    from src.data.leakage_role_crosscheck import evaluate_role_vs_statistical_leak

    assert evaluate_role_vs_statistical_leak(None, None) is None


def test_returns_none_for_info_severity():
    """Statistical finding at info severity (not critical/high) → None."""
    from src.data.leakage_role_crosscheck import evaluate_role_vs_statistical_leak

    assert evaluate_role_vs_statistical_leak("confounder", "info") is None


def test_instrument_at_critical_fires():
    """Instrument is a BENIGN_KEEP_ROLE — should fire at critical."""
    from src.data.leakage_role_crosscheck import evaluate_role_vs_statistical_leak

    assert evaluate_role_vs_statistical_leak("instrument", "critical") is True


# ---------------------------------------------------------------------------
# Purity / no-mutation guard.
# ---------------------------------------------------------------------------


def test_purity_inputs_not_mutated():
    """The function is pure — inputs (str / None) cannot be mutated, and
    the function must not raise or modify module state."""
    from src.data.leakage_role_crosscheck import evaluate_role_vs_statistical_leak

    role = "confounder"
    sev = "high"
    # Call multiple times; each call must return the same result without
    # any observable side effect.
    r1 = evaluate_role_vs_statistical_leak(role, sev)
    r2 = evaluate_role_vs_statistical_leak(role, sev)
    assert r1 is True
    assert r2 is True
    # Inputs unchanged (trivially true for str/None — guards against
    # any future implementation mistake that passes a mutable container).
    assert role == "confounder"
    assert sev == "high"


# ---------------------------------------------------------------------------
# Constants guard — frozensets contain exactly the specified members.
# ---------------------------------------------------------------------------


def test_benign_keep_roles_constant():
    from src.data.leakage_role_crosscheck import BENIGN_KEEP_ROLES

    assert BENIGN_KEEP_ROLES == frozenset({"ancestor", "confounder", "instrument"})


def test_stat_flag_severities_constant():
    from src.data.leakage_role_crosscheck import STAT_FLAG_SEVERITIES

    assert STAT_FLAG_SEVERITIES == frozenset({"critical", "high"})
