"""Shared fixtures for the ensemble-voter test suite (Layer-2 KG voter).

Plan v4 Phase 1 demoted the Layer-4 LLM to AUDIT-ONLY by default — the voter no
longer sets ``decided_by="llm"`` unless ``ADAPTIVE_LAYER4_LLM_DECIDES=1`` (see
``ensemble_voter._llm_decides_enabled``). The pre-existing tests in this
directory pin the LLM-*decides* mechanism — voter precedence (rule 4), the
confidence/citation scoring, the remediation mapping, and the issue-#240
evaluator soft-gate (which reframes an LLM-decided ``info`` verdict to
``moderate``). All of that is preserved behind the flag for the Phase-3 ramp /
back-compat, so we enable it here for the whole directory. The Phase-1
audit-only-DEFAULT tests opt back out by calling
``monkeypatch.delenv("ADAPTIVE_LAYER4_LLM_DECIDES", raising=False)`` in their own
bodies (same per-test monkeypatch instance, so the delenv runs after this
fixture's setenv).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _enable_llm_decides_for_legacy_voter_tests(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_LAYER4_LLM_DECIDES", "1")
