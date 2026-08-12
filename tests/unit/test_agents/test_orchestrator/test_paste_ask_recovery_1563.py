"""±filler routing parity for over-cap pastes (issue #1563).

WHAT BROKE (measured live 2026-08-12, raw_cap_probe_chat.jsonl)
    A 68-char KPI ask ("Give me an NRx breakdown by patient clinical segment
    for Remibrutinib") pattern-classifies ``segment_analysis@0.867`` and routes
    to ``heterogeneous_optimizer``. The SAME ask preceded by ~2.5KB of pasted
    meeting-notes context crosses ``_PATTERN_SCAN_MAX_CHARS`` (#1470), the
    pattern layer abstains, and the Haiku fallback — fed the whole paste —
    lands the turn on a lone ``explainer`` that answers with a bare KPI total
    (221 chars), dropping the requested per-segment breakdown.

    The #1470 abstention itself is correct and must stay (it closed a
    quadratic regex-backtracking CPU hazard). The defect is that the ask
    inside the paste is never given to the deterministic pattern layer at all.

WHY TRAILING-PARAGRAPH RECOVERY, NOT A TAIL WINDOW (measured 2026-08-12)
    Scoring the last 2000 chars of the padded probe — or hypothetically the
    whole padded query with the cap lifted — returns ``multi_faceted@0.867``
    (the FILLER itself trips patterns: "payer mix shifts" fires drift_check),
    i.e. any window that includes filler is poisoned by it and gives a third,
    different route. Only scoring the trailing ask paragraph alone reproduces
    the unpadded verdict exactly. Hence the recovery in
    ``_classify_trailing_ask``: last blank-line-separated paragraph, bounded
    at ``_ASK_TAIL_MAX_CHARS``, gated on an ask-shape marker, trusted only at
    the same >=0.8 floor ``execute`` already applies. Everything else abstains
    exactly as #1470 shipped it.

This file pins (a) the ±filler parity at the routing seam, (b) every
fail-closed edge of the recovery (ask-first pastes, oversized tails,
single-paragraph pastes must all still abstain), and (c) the bounded-cost
property with the recovery in place. The #1470 pins in
``test_intent_classifier_backtracking_1470.py`` are untouched and must stay
green alongside this file.
"""

import asyncio
import time

from src.agents.orchestrator.nodes import intent_classifier as ic
from src.agents.orchestrator.nodes.router import RouterNode

# The live probe's exact ask (question 2.2 of the eval corpus) and filler
# block (cap_probe_questions.json, P.base / P.fill). The padded probe is the
# filler block repeated past the scan cap, then a blank line, then the ask —
# the "context first, ask last" paste shape the issue measured.
CONTROL_ASK = "Give me an NRx breakdown by patient clinical segment for Remibrutinib"

FILLER_BLOCK = (
    "Context from today's brand review meeting: the field team discussed "
    "Q3 pull-through, payer mix shifts in the commercial book, and the new "
    "speaker program calendar. Action items included refreshing the target "
    "list, aligning with market access on prior-auth friction, and reviewing "
    "the omnichannel cadence for the top decile. "
)


def _node() -> ic.IntentClassifierNode:
    """Construct without ``__init__`` — ``_pattern_classify`` needs no LLM."""
    return ic.IntentClassifierNode.__new__(ic.IntentClassifierNode)


def _padded_probe() -> str:
    """Rebuild the P.fill probe: ~2.5KB of filler, blank line, the ask."""
    filler = (FILLER_BLOCK * 10)[:2500]
    return f"{filler}\n\nWith that context: {CONTROL_ASK}"


def _classify(query: str) -> ic.IntentClassification:
    """Mirror ``execute``'s call shape: the query arrives lowercased."""
    return _node()._pattern_classify(query.lower())


def _route(intent: ic.IntentClassification) -> list:
    """Legacy routing seam: the dispatch plan RouterNode builds for an intent.

    ``ORCHESTRATOR_CLASSIFIER_MODE`` is unset in tests => shadow mode, so the
    active-pipeline branch never runs — exactly the live configuration the
    probe measured (docker-compose defaults the mode to shadow).
    """
    state = {"query": "unused-by-legacy-routing", "intent": intent}
    result = asyncio.run(RouterNode().execute(state))  # type: ignore[arg-type]
    return result["dispatch_plan"]


class TestPaddedAskRoutingParity1563:
    """The ±filler acceptance: padded and unpadded probe route identically."""

    def test_control_probe_anchor(self):
        """Anchor the control verdict this whole file is measured against.

        segment_analysis@0.867 with secondary [explanation] is what the live
        control probe routed on (heterogeneous_optimizer critical, gap_analyzer
        as its dispatch-time fallback). If a routing change moves this anchor
        deliberately, the parity tests below still hold — they compare padded
        AGAINST control rather than pinning either to a constant.
        """
        verdict = _classify(CONTROL_ASK)
        assert verdict["primary_intent"] == "segment_analysis"
        assert verdict["confidence"] >= 0.8

    def test_padded_probe_classifies_like_unpadded(self):
        control = _classify(CONTROL_ASK)
        padded = _classify(_padded_probe())

        assert padded["primary_intent"] == control["primary_intent"], (
            f"padded probe classified {padded['primary_intent']}@"
            f"{padded['confidence']:.3f} vs control "
            f"{control['primary_intent']}@{control['confidence']:.3f} — the "
            "pasted context changed the routing decision (issue #1563)"
        )
        assert padded["confidence"] == control["confidence"]
        assert padded["secondary_intents"] == control["secondary_intents"]
        assert padded["requires_multi_agent"] == control["requires_multi_agent"]

    def test_padded_probe_dispatches_like_unpadded(self):
        """Same routing seam, one level down: the dispatch plan itself."""
        control_plan = _route(_classify(CONTROL_ASK))
        padded_plan = _route(_classify(_padded_probe()))

        control_view = [(d["agent_name"], d["priority"], d["fallback_agent"]) for d in control_plan]
        padded_view = [(d["agent_name"], d["priority"], d["fallback_agent"]) for d in padded_plan]
        assert padded_view == control_view, (
            f"padded probe dispatches {padded_view} vs control {control_view}"
        )

    def test_recovery_never_scans_the_filler(self):
        """The recovery must score the trailing ask ONLY — never the paste.

        Measured with the cap lifted: scanning the WHOLE padded probe (or its
        last 2000 chars) returns ``multi_faceted@0.867`` because the filler
        itself trips patterns ("payer mix shifts" -> drift_check). Getting
        multi_faceted here would mean the bounded-scan property of #1470 was
        silently replaced by a full scan.
        """
        padded = _classify(_padded_probe())
        assert padded["primary_intent"] != "multi_faceted", (
            "the poisoned whole-query verdict surfaced — the recovery scanned "
            "the filler, not just the trailing ask"
        )


class TestRecoveryFailsClosed1563:
    """Every non-recoverable shape must abstain exactly as #1470 shipped."""

    def _assert_abstained(self, query: str) -> None:
        verdict = _classify(query)
        assert verdict["primary_intent"] == "general"
        assert verdict["confidence"] < 0.8, (
            f"over-cap query scored {verdict['primary_intent']}@"
            f"{verdict['confidence']} — a confident verdict here bypasses the "
            "LLM fallback on partial evidence"
        )

    def test_ask_first_paste_still_abstains(self):
        """Ask first, 2.5KB of notes after: the trailing paragraph is filler.

        The trailing filler exceeds the ask-tail bound, so recovery must not
        engage. Scoring filler at full trust was measured to produce a
        CONFIDENT drift_check@0.867 misroute ("payer mix shifts") where today
        the query safely escalates to the LLM with full context.
        """
        filler = (FILLER_BLOCK * 10)[:2500]
        self._assert_abstained(f"{CONTROL_ASK}\n\n{filler}")

    def test_short_trailing_filler_paragraph_still_abstains(self):
        """A SHORT trailing notes paragraph passes the length bound but must
        fail the ask-shape gate.

        The guard is what stands between this shape and a confident misroute:
        the filler block alone IS a confident pattern verdict when scored
        (asserted below, so this test cannot rot into vacuity if the filler
        or the pattern table changes).
        """
        filler_alone = _classify(FILLER_BLOCK)
        assert filler_alone["confidence"] >= 0.8, (
            "precondition lost: the filler block no longer scores confidently "
            "on its own, so this test no longer exercises the ask-shape gate — "
            "pick a filler that trips the pattern table"
        )

        long_head = (FILLER_BLOCK * 10)[:2200]
        assert len(FILLER_BLOCK) <= ic._ASK_TAIL_MAX_CHARS
        self._assert_abstained(f"{CONTROL_ASK}\n\n{long_head}\n\n{FILLER_BLOCK}")

    def test_single_paragraph_over_cap_still_abstains(self):
        """No blank line => no recoverable trailing ask => #1470 behavior."""
        self._assert_abstained("what is the trx for kisqali " + "filler " * 500)

    def test_oversized_trailing_ask_not_recovered(self):
        """An 'ask' longer than any real routed query is treated as paste."""
        oversized_ask = ("what about the nrx trend for remibrutinib and also " * 20).strip()
        assert len(oversized_ask) > ic._ASK_TAIL_MAX_CHARS
        filler = (FILLER_BLOCK * 10)[:2200]
        self._assert_abstained(f"{filler}\n\n{oversized_ask}")

    def test_weak_trailing_ask_still_abstains(self):
        """Ask-shaped tail the table cannot score confidently => abstain.

        The recovery may only ever return a verdict at the same >=0.8 floor
        ``execute`` applies; a weak tail verdict must not displace the LLM
        fallback (which sees the full query).
        """
        filler = (FILLER_BLOCK * 10)[:2500]
        self._assert_abstained(f"{filler}\n\nplease advise on next steps?")


class TestParagraphScopedEvidence1563:
    """Cross-paragraph veto evidence is context, not the ask's own lexemes.

    #1470's ``test_evidence_past_the_cap_cannot_flip_a_veto`` pins that a
    SINGLE ask whose own trailing directive carries a veto lexeme ("what is
    the trx ... please forecast it") must not be confidently routed on partial
    evidence — that construction has no paragraph structure and still
    abstains. This class pins the deliberate complement: when the veto lexeme
    sits in a SEPARATE pasted-context paragraph, it is not part of the
    trailing ask at all, and suppressing the recovery because pasted notes
    mention "forecast" would reintroduce exactly the filler-poisoning this
    fix removes. The trailing paragraph is the operative ask (the paste
    workload's shape, per the live probe), so it is scored on its own text.
    """

    def test_context_mention_of_forecast_does_not_block_trailing_kpi_ask(self):
        head = "Context: leadership wants the forecast deck ready next month."
        filler = (FILLER_BLOCK * 10)[:2400]
        query = f"{head}\n\n{filler}\n\nWhat is the TRx for Kisqali?"
        assert len(query) > ic._PATTERN_SCAN_MAX_CHARS

        verdict = _classify(query)
        assert verdict["primary_intent"] == "explanation", (
            f"trailing KPI ask classified {verdict['primary_intent']}@"
            f"{verdict['confidence']:.3f} — a forecast MENTION in pasted "
            "context suppressed the KPI value-lookup route for the actual ask"
        )
        assert verdict["confidence"] >= 0.8


class TestBoundedCost1563:
    """#1470's bounded-scan property survives the recovery."""

    def test_paragraphed_adversarial_input_stays_fast_and_recovers(self):
        """Adversarial paragraphed head + tiny ask tail: fast AND recovered.

        The recovery adds one linear paragraph split over the paste plus one
        table scan bounded at ``_ASK_TAIL_MAX_CHARS`` — the quadratic
        whole-query scan #1470 removed must never run.
        """
        head = ("remibrutinib " * 25 + "\n\n") * 40  # ~13.6K chars, many paragraphs
        query = f"{head}what is the trx for kisqali?"
        assert len(query) > ic._PATTERN_SCAN_MAX_CHARS

        node = _node()
        start = time.perf_counter()
        verdict = node._pattern_classify(query.lower())
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"_pattern_classify took {elapsed:.2f}s on {len(query)} chars — "
            "the #1470 quadratic scan is back"
        )
        assert verdict["primary_intent"] == "explanation", (
            f"trailing KPI ask not recovered from a paragraphed paste "
            f"(got {verdict['primary_intent']}@{verdict['confidence']:.3f})"
        )
