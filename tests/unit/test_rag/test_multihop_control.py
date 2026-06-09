"""Offline guards for the cognitive-RAG multi-hop control logic (audit F6).

The audit found enable_multi_hop ignored, hop_count hardcoded to 1, and
ChatbotHopDecider never instantiated. F6 wires a real hop loop. These test the
PURE control predicates with real values (no mock): when to stop early (strong
evidence -> no decider LLM call), when the decider says continue, and evidence
dedup across hops. The real-LM loop is proven by the gated E2E.
"""

from src.api.routes.chatbot_dspy import (
    _dedupe_evidence,
    _evidence_sufficient,
    _should_continue_hop,
)


class TestEvidenceSufficient:
    def test_strong_evidence_is_sufficient_stop_without_decider(self):
        evidence = [{"source_id": str(i)} for i in range(3)]
        assert _evidence_sufficient(evidence, avg_relevance=0.8) is True

    def test_few_rows_is_insufficient(self):
        assert _evidence_sufficient([{"source_id": "a"}], avg_relevance=0.9) is False

    def test_low_relevance_is_insufficient(self):
        evidence = [{"source_id": str(i)} for i in range(5)]
        assert _evidence_sufficient(evidence, avg_relevance=0.3) is False

    def test_empty_is_insufficient(self):
        assert _evidence_sufficient([], avg_relevance=0.0) is False


class TestShouldContinueHop:
    def test_continue_when_more_needed_and_room_left(self):
        assert _should_continue_hop("episodic", confidence=0.8, hop_number=1, max_hops=3) is True

    def test_stop_on_explicit_stop_token(self):
        assert _should_continue_hop("STOP", confidence=0.9, hop_number=1, max_hops=3) is False
        assert _should_continue_hop(" stop ", confidence=0.9, hop_number=1, max_hops=3) is False

    def test_stop_when_decider_confidence_low(self):
        # confidence is "more evidence needed"; low -> sufficient -> stop
        assert _should_continue_hop("semantic", confidence=0.2, hop_number=1, max_hops=3) is False

    def test_stop_at_max_hops(self):
        assert _should_continue_hop("episodic", confidence=0.9, hop_number=3, max_hops=3) is False


class TestDedupeEvidence:
    def test_dedupes_by_source_id_preserving_order(self):
        ev = [
            {"source_id": "a", "content": "x"},
            {"source_id": "b", "content": "y"},
            {"source_id": "a", "content": "x-dup"},
        ]
        out = _dedupe_evidence(ev)
        assert [e["source_id"] for e in out] == ["a", "b"]
        assert out[0]["content"] == "x"  # first occurrence kept

    def test_empty(self):
        assert _dedupe_evidence([]) == []
