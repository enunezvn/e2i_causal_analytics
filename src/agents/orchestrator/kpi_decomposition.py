"""Shared recognition for KPI decompositions with a real chat owner.

The NRx patient profile is deliberately narrow: CohortProfiler serves disease-
severity segments and therapy-line buckets from the KPI calculator.  It does
not serve TRx, biologic status, or IgE tiers, so those must not be widened into
this route merely because they are also decomposition-shaped.
"""

from __future__ import annotations

import re

NRX_COHORT_DECOMPOSITION_PATTERN = (
    r"(?s)\A\s*(?:please\s+)?"
    r"(?:(?:can|could|would)\s+you\s+(?:please\s+)?)?"
    r"(?:what(?:'?s| is| are| was| were)|show me|give me|tell me about|how many)\s+"
    r"(?:(?:the|an)\s+)?(?:current\s+)?nrx(?:\s+panel)?\b"
    r"(?:\s+(?:numbers?|counts?|breakdown|profile))?"
    r"\s+(?:by|per|across)\s+"
    r"(?:(?:(?:patient|clinical|disease[- ]severity|severity)\s+){0,2}segments?"
    r"|severity\s+tiers?|therapy[-_\s]+lines?|lines?[-_\s]+of[-_\s]+therapy)"
    r"(?:\s+for\s+(?:remibrutinib|fabhalta|kisqali))?"
    r"\s*(?:please|thanks)?[?.!]*\s*\Z"
)
NRX_COHORT_DECOMPOSITION_RE = re.compile(NRX_COHORT_DECOMPOSITION_PATTERN, re.IGNORECASE)
NRX_COHORT_DECOMPOSITION_EVIDENCE = "nrx_cohort_decomposition"


def is_nrx_cohort_decomposition(query: str) -> bool:
    """Whether the whole ask is a CohortProfiler-served NRx breakdown."""
    return bool(NRX_COHORT_DECOMPOSITION_RE.search(query))
