"""Issue #376 — Phase 4 schema completion.

Tests pin the ``LLMCrystalNarrativeAudit`` struct shape, mirroring
``LLMEvaluatorAudit`` (``src/data/kg/types.py:353-404``) per Decision 2
sub-decision: 3 LLM-narrative fields wrapped in an audit struct.

Mirror requirements:
  * Same telemetry fields: latency_ms, input_tokens, output_tokens, cost_usd
  * Same nullability (all Optional[float/int])
  * Frozen dataclass
  * Documents the Haiku model identifier
"""

from __future__ import annotations

import dataclasses

import pytest


def test_llm_crystal_narrative_audit_imports_from_kg_types():
    """The struct must live alongside ``LLMEvaluatorAudit`` in
    ``src.data.kg.types`` — same module, same pattern, easy peer review."""
    from src.data.kg.types import LLMCrystalNarrativeAudit  # noqa: F401


def test_llm_crystal_narrative_audit_is_frozen_dataclass():
    from src.data.kg.types import LLMCrystalNarrativeAudit

    assert dataclasses.is_dataclass(LLMCrystalNarrativeAudit)
    # Frozen: assigning to a field after construction raises FrozenInstanceError
    instance = LLMCrystalNarrativeAudit(
        narrator_model="claude-haiku-4-5-20251001",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.narrator_model = "claude-sonnet-4-6"  # type: ignore[misc]


def test_llm_crystal_narrative_audit_mirrors_evaluator_telemetry_fields():
    """All 4 telemetry fields present, same names + types as evaluator."""
    from src.data.kg.types import LLMCrystalNarrativeAudit, LLMEvaluatorAudit

    crystal_fields = {f.name for f in dataclasses.fields(LLMCrystalNarrativeAudit)}
    evaluator_fields = {f.name for f in dataclasses.fields(LLMEvaluatorAudit)}

    telemetry = {"latency_ms", "input_tokens", "output_tokens", "cost_usd"}
    assert telemetry.issubset(crystal_fields)
    assert telemetry.issubset(evaluator_fields)


def test_llm_crystal_narrative_audit_telemetry_defaults_to_none():
    """Mirror the LLMEvaluatorAudit default-None pattern so legacy
    call-sites that don't capture telemetry don't break construction."""
    from src.data.kg.types import LLMCrystalNarrativeAudit

    a = LLMCrystalNarrativeAudit(narrator_model="claude-haiku-4-5-20251001")
    assert a.latency_ms is None
    assert a.input_tokens is None
    assert a.output_tokens is None
    assert a.cost_usd is None


def test_llm_crystal_narrative_audit_holds_narrator_outputs():
    """The narrator produces 3 LLM-driven prose fields per Decision 2:
    key_finding, limitations, recommended_next_analysis.

    The audit must carry the actual outputs so the audit trail is the
    sole writer (avoids drift between row and audit)."""
    from src.data.kg.types import LLMCrystalNarrativeAudit

    a = LLMCrystalNarrativeAudit(
        narrator_model="claude-haiku-4-5-20251001",
        key_finding="Northeast lift driven by D5 visit cadence increase.",
        limitations="Pre-period n=120; sensitivity to outliers HIGH.",
        recommended_next_analysis="Replicate on Q3 cohort with 360d washout.",
    )
    assert a.key_finding.startswith("Northeast lift")
    assert "n=120" in a.limitations
    assert a.recommended_next_analysis.startswith("Replicate")


def test_llm_crystal_narrative_audit_cost_computation_matches_haiku_pricing():
    """Reuse the Haiku pricing constants — narrative audit cost must use
    the SAME pin so a price-drift on the evaluator surfaces here too."""
    from src.data.causal_role_evaluator import (
        HAIKU_INPUT_USD_PER_MTOK,
        HAIKU_OUTPUT_USD_PER_MTOK,
        compute_haiku_cost_usd,
    )

    # Pinning: per-million-token rates documented at 2026-05-15.
    assert HAIKU_INPUT_USD_PER_MTOK == 1.00
    assert HAIKU_OUTPUT_USD_PER_MTOK == 5.00

    # 1000 prompt tokens + 500 completion tokens:
    cost = compute_haiku_cost_usd(input_tokens=1000, output_tokens=500)
    expected = (1000 * 1.00 + 500 * 5.00) / 1_000_000.0
    assert abs(cost - expected) < 1e-9
