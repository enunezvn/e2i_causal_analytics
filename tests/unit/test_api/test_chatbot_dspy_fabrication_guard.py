"""Phase 1 fabrication-guard tests for the DSPy synthesis path (audit C5).

Faithful: exercises the real synthesize_response_dspy logic. The only doubled
unit is the DSPy LLM call (a true external) via the _get_dspy_synthesizer seam.
"""

import pytest

# Group all DSPy tests on the same xdist worker to avoid import races
pytestmark = pytest.mark.xdist_group(name="dspy_integration")

from unittest.mock import MagicMock, patch

import src.api.routes.chatbot_dspy as cd
from src.api.routes.chatbot_dspy import synthesize_response_dspy


def _fabricating_synthesizer(citations="SRC-INVENTED-1, SRC-INVENTED-2"):
    """A stubbed DSPy synthesizer that fabricates a confident narrative + bogus citations."""
    fake = MagicMock()
    fake.return_value = MagicMock(
        response=(
            "Kisqali TRx rose 12% in the Northeast last quarter, "
            "driven by HCP engagement and formulary wins."
        ),
        confidence_statement="High confidence based on 3 corroborating sources",
        evidence_citations=citations,
        follow_up_suggestions="",
    )
    return fake


class TestFailClosedOnNoEvidence:
    """C5a/C5b: empty evidence must abstain, not synthesize a fluent narrative."""

    @pytest.mark.asyncio
    async def test_empty_evidence_abstains_in_dspy_path(self):
        with (
            patch.object(cd, "CHATBOT_DSPY_SYNTHESIS_ENABLED", True),
            patch.object(cd, "_get_dspy_synthesizer", return_value=_fabricating_synthesizer()),
        ):
            result = await synthesize_response_dspy(
                query="TRx trend for Kisqali in Northeast",
                intent="kpi_query",
                evidence=[],
                brand_context="Kisqali",
                collect_signal=False,
            )
        # Must NOT surface the fabricated narrative.
        assert "rose 12%" not in result.response
        # Must be the honest abstain text + low confidence.
        assert "don't have specific data" in result.response
        assert result.confidence_level == "low"
        # No fabricated citations leak through.
        assert result.evidence_citations == []


class TestFailClosedOnLowEvidence:
    """C5e: a single row at the exact 0.3 boundary must hedge, not assert."""

    @pytest.mark.asyncio
    async def test_single_boundary_row_hedges_not_high_confidence(self):
        evidence = [
            {
                "source_id": "PROC-571cb03a",
                "content": "[PROC] composition_comp_571cb03a: Needs causal then regional analysis",
                "score": 0.0082,
                "relevance_score": 0.3,  # the exact keep==skip boundary value
                "source": "procedural",
            }
        ]
        with (
            patch.object(cd, "CHATBOT_DSPY_SYNTHESIS_ENABLED", True),
            patch.object(
                cd,
                "_get_dspy_synthesizer",
                return_value=_fabricating_synthesizer(citations="PROC-571cb03a"),
            ),
        ):
            result = await synthesize_response_dspy(
                query="TRx trend for Kisqali in Northeast",
                intent="kpi_query",
                evidence=evidence,
                brand_context="Kisqali",
                collect_signal=False,
            )
        # avg_evidence_score==0.3 AND count==1 -> insufficient -> confidence must be capped low.
        assert result.confidence_level == "low"
        # The fabricated "rose 12%" narrative must not be asserted as fact.
        assert "rose 12%" not in result.response


class TestCitationValidation:
    """C5c: citations not present in the supplied source_ids must be dropped."""

    @pytest.mark.asyncio
    async def test_invented_citations_are_dropped(self):
        evidence = [
            {
                "source_id": "REAL-1",
                "content": "Causal analysis: hcp_engagement_level -> patient_conversion_rate, ATE=0.413",
                "score": 0.5,
                "relevance_score": 0.8,
                "source": "episodic",
            },
            {
                "source_id": "REAL-2",
                "content": "Causal analysis: adherence -> persistence, ATE=0.21",
                "score": 0.5,
                "relevance_score": 0.75,
                "source": "episodic",
            },
        ]
        # LLM returns one real id and one invented id.
        synth = _fabricating_synthesizer(citations="REAL-1, SRC-INVENTED-9")
        with (
            patch.object(cd, "CHATBOT_DSPY_SYNTHESIS_ENABLED", True),
            patch.object(cd, "_get_dspy_synthesizer", return_value=synth),
        ):
            result = await synthesize_response_dspy(
                query="Why did adoption increase?",
                intent="causal_analysis",
                evidence=evidence,
                brand_context="Kisqali",
                collect_signal=False,
            )
        assert "REAL-1" in result.evidence_citations
        assert "SRC-INVENTED-9" not in result.evidence_citations


class TestConfidenceNotFromProseAlone:
    """C5d: a single weak row must not be high confidence (routes to hardcoded fallback).

    With EVIDENCE_MIN_COUNT=2, a single ~0.31 row is insufficient for DSPy
    synthesis, so it falls through to synthesize_response_hardcoded. The LLM's
    'High confidence' prose is irrelevant there; the evidence-derived level must
    not be 'high'. (This exercises the fail-closed routing, not the DSPy branch.)
    """

    @pytest.mark.asyncio
    async def test_high_confidence_prose_capped_by_weak_evidence(self):
        evidence = [
            {
                "source_id": "PROC-x",
                "content": "[PROC] composition_comp: generic",
                "score": 0.0082,
                "relevance_score": 0.31,  # > skip but only one item
                "source": "procedural",
            }
        ]
        with (
            patch.object(cd, "CHATBOT_DSPY_SYNTHESIS_ENABLED", True),
            patch.object(
                cd,
                "_get_dspy_synthesizer",
                return_value=_fabricating_synthesizer(citations="PROC-x"),
            ),
        ):
            result = await synthesize_response_dspy(
                query="TRx for Kisqali?",
                intent="kpi_query",
                evidence=evidence,
                brand_context="Kisqali",
                collect_signal=False,
            )
        # LLM said "High confidence"; a single ~0.31 row must NOT yield high.
        assert result.confidence_level != "high"


class TestProseSubstringDoesNotFalselyLowerConfidence:
    """C5d hardening: a benign 'low' substring (e.g. 'follow-up') in the LLM's
    confidence prose must NOT force the evidence-derived level down to 'low'.

    Two corroborating rows at avg relevance 0.6 with two valid citations yield a
    'moderate' evidence-grounded level. The synthesizer's prose contains
    'follow-up' (which holds the substring 'low') but does not express low
    confidence. A naive `if "low" in prose` would corrupt this to 'low'; the
    word-boundary check must keep it 'moderate'.
    """

    @pytest.mark.asyncio
    async def test_followup_substring_does_not_force_low(self):
        evidence = [
            {
                "source_id": "REAL-1",
                "content": "Causal analysis: hcp_engagement_level -> conversion, ATE=0.41",
                "score": 0.6,
                "relevance_score": 0.6,
                "source": "episodic",
            },
            {
                "source_id": "REAL-2",
                "content": "Causal analysis: adherence -> persistence, ATE=0.21",
                "score": 0.6,
                "relevance_score": 0.6,
                "source": "episodic",
            },
        ]
        synth = MagicMock()
        synth.return_value = MagicMock(
            response="Adoption rose because of expanded access.",
            # contains 'follow-up' (substring 'low') but NOT a word-boundary 'low'
            confidence_statement="Based on the data, a useful follow-up would be region detail",
            evidence_citations="REAL-1, REAL-2",
            follow_up_suggestions="",
        )
        with (
            patch.object(cd, "CHATBOT_DSPY_SYNTHESIS_ENABLED", True),
            patch.object(cd, "_get_dspy_synthesizer", return_value=synth),
        ):
            result = await synthesize_response_dspy(
                query="Why did adoption increase?",
                intent="causal_analysis",
                evidence=evidence,
                brand_context="Kisqali",
                collect_signal=False,
            )
        # Evidence-grounded level is 'moderate' (avg 0.6, 2 valid citations); the
        # benign 'follow-up' substring must NOT drop it to 'low'.
        assert result.synthesis_method == "dspy"
        assert result.confidence_level == "moderate"
