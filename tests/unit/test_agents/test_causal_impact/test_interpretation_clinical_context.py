"""Task 7: the interpretation node weaves a fail-open clinical/market-context
sentence into the narrative (FDA-label use + limitation of use + competitor count),
and never lets a context failure break the interpretation."""

from __future__ import annotations

import pytest

import src.agents.causal_impact.nodes.interpretation as interp_mod
from src.agents.causal_impact.nodes.interpretation import (
    InterpretationNode,
    _clinical_context_sentence,
)


def _state() -> dict:
    return {
        "outcome_var": "persistent_180d",
        "brand": "Kisqali",
        "estimation_result": {
            "ate": 0.12,
            "ate_ci_lower": 0.05,
            "ate_ci_upper": 0.19,
            "effect_size": "moderate",
            "statistical_significance": True,
            "method": "CausalForestDML",
        },
        "refutation_results": {"tests_passed": 5, "total_tests": 5, "overall_robust": True},
        "sensitivity_analysis": {"e_value": 2.0, "robust_to_confounding": True},
    }


class _FakeSvc:
    def get_context(self, brand, outcome):
        return {
            "approved_indications": {
                "indications": ["HR+/HER2- breast cancer"],
                "limitations_of_use": None,
                "boxed_warning": None,
                "source": "openfda",
            },
            "competitor_landscape": {
                "competitors": ["Ibrance (palbociclib)", "Verzenio (abemaciclib)"],
                "count": 2,
                "source": "curated",
            },
        }


class _BoomSvc:
    def get_context(self, brand, outcome):
        raise RuntimeError("clinical-context API unavailable")


@pytest.mark.unit
def test_clinical_context_sentence_content():
    s = _clinical_context_sentence(
        {
            "approved_indications": {
                "indications": ["HR+ breast cancer"],
                "limitations_of_use": "Not indicated for X",
                "boxed_warning": None,
                "source": "openfda",
            },
            "competitor_landscape": {"competitors": ["a", "b"], "count": 2, "source": "curated"},
        }
    )
    assert s.startswith("Clinical/market context:")
    assert "FDA-approved use includes HR+ breast cancer" in s
    assert "limitation of use" in s
    assert "2 therapeutic competitor" in s


@pytest.mark.unit
def test_clinical_context_sentence_empty_when_nothing_useful():
    assert _clinical_context_sentence({}) == ""
    assert (
        _clinical_context_sentence(
            {"approved_indications": {"indications": []}, "competitor_landscape": {"count": 0}}
        )
        == ""
    )


@pytest.mark.asyncio
async def test_interpretation_weaves_clinical_context(monkeypatch):
    monkeypatch.setattr(interp_mod, "_get_clinical_context_service", lambda: _FakeSvc())
    result = await InterpretationNode()._generate_standard_interpretation(_state(), "standard")
    blob = str(result)
    assert "Clinical/market context" in blob
    assert "therapeutic competitor" in blob


@pytest.mark.asyncio
async def test_interpretation_fail_open_when_context_raises(monkeypatch):
    monkeypatch.setattr(interp_mod, "_get_clinical_context_service", lambda: _BoomSvc())
    # Must NOT raise; a full interpretation is still produced, just without the sentence.
    result = await InterpretationNode()._generate_standard_interpretation(_state(), "standard")
    assert result
    assert "Clinical/market context" not in str(result)


@pytest.mark.asyncio
async def test_interpretation_no_context_without_brand(monkeypatch):
    called = {"n": 0}

    class _Track:
        def get_context(self, brand, outcome):
            called["n"] += 1
            return {}

    monkeypatch.setattr(interp_mod, "_get_clinical_context_service", lambda: _Track())
    state = _state()
    del state["brand"]  # no brand -> skip the fetch entirely
    result = await InterpretationNode()._generate_standard_interpretation(state, "standard")
    assert called["n"] == 0
    assert "Clinical/market context" not in str(result)
