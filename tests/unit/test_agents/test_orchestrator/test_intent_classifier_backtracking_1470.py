"""Regex-backtracking guards for ``_pattern_classify`` (issue #1470).

WHAT BROKE
    ``_pattern_classify`` runs the whole ``INTENT_PATTERNS`` table over the raw
    query on EVERY request, ahead of the LLM fallback. Most entries in that
    table join two token groups with an unbounded ``.*``, and one
    (``experiment_monitor[3]``) is an unanchored pair of ``.*`` lookaheads. Both
    shapes are quadratic in the query length, so a single long request burns
    CPU on the classification path. Measured on this box, before the fix:

        _pattern_classify("remibrutinib " * n)   8 000 chars ->  1.43 s
                                                12 000 chars ->  3.15 s
                                                20 000 chars ->  6.90 s
                                                45 000 chars -> 32.75 s

    ``experiment_monitor[3]`` alone is the single largest contributor and — being
    a zero-width lookahead pair tried at every start offset — is quadratic on
    input containing NONE of its tokens: 28.0 s on 45 000 chars of ``"a"``.

THE TWO GUARDS BELOW
    1. A scan cap at the ``_pattern_classify`` boundary, past which the layer
       ABSTAINS (general@0.5, below the router's 0.8 trust floor) rather than
       scoring a prefix. This covers the WHOLE table, including the ~35
       unbounded-``.*`` entries this PR deliberately does not rewrite — each
       rewrite is a routing-semantics change; abstaining is not.
       Scoring a truncated prefix was the first cut and it was WRONG: it hid
       whole-query vetoes and manufactured a confident misroute. See
       ``test_evidence_past_the_cap_cannot_flip_a_veto``.
    2. Line-anchoring ``experiment_monitor[3]``, which is a semantics-PRESERVING
       rewrite (see ``TestInterimPatternLinearization1470``) and removes the
       dominant cost below the cap.

The semantic behaviour of the #1408 interim gate itself is pinned by
``test_intent_classifier_weakclass_tuning.py::TestSingleAgentTieBreakRecovered1408``
(``test_interim_readout_routes_experiment_monitor`` and the
``test_interim_without_experiment_noun_not_monitor`` probes); this file guards
the cost and the truncation boundary, not the routing decision.
"""

import json
import re
import time
from pathlib import Path

import pytest

from src.agents.orchestrator.nodes import intent_classifier as ic

REPO_ROOT = Path(__file__).resolve().parents[4]

# Threshold with two-sided margin measured on this box: the slowest pre-fix
# shape at 12 000 chars is ~6 s (>=3x over) and the slowest post-fix shape is
# ~74 ms (>=13x under). Wide enough that CI load cannot flip it either way.
ADVERSARIAL_CHARS = 12_000
MAX_CLASSIFY_SECONDS = 1.0


def _node() -> ic.IntentClassifierNode:
    """Construct without ``__init__`` — ``_pattern_classify`` needs no LLM."""
    return ic.IntentClassifierNode.__new__(ic.IntentClassifierNode)


def _repeat(token: str, n: int) -> str:
    return (token * (n // len(token) + 1))[:n]


class TestPatternClassifyBacktracking1470:
    """The classification path must stay cheap on adversarially long input."""

    @pytest.mark.parametrize(
        "label,token",
        [
            # Many anchors for the unbounded-gap patterns, no completing token:
            # every anchor occurrence starts a full-length backtracking scan.
            ("brand_anchors", "remibrutinib "),
            ("what_anchors", "what "),
            # No table token at all — still quadratic pre-fix, because the
            # unanchored lookahead pair is retried at every start offset.
            ("token_free_filler", "a"),
        ],
        ids=["brand_anchors", "what_anchors", "token_free_filler"],
    )
    def test_adversarial_long_query_classifies_fast(self, label, token):
        query = _repeat(token, ADVERSARIAL_CHARS)
        node = _node()
        start = time.perf_counter()
        node._pattern_classify(query)
        elapsed = time.perf_counter() - start
        assert elapsed < MAX_CLASSIFY_SECONDS, (
            f"{label}: _pattern_classify took {elapsed:.2f}s on "
            f"{len(query)} chars (limit {MAX_CLASSIFY_SECONDS}s) — the "
            "quadratic backtracking of issue #1470 is back"
        )

    def test_cap_boundary_is_exact(self):
        """At the cap the table still runs; one char over, it abstains.

        Deterministic companion to the timing pin above: it fails if the cap is
        removed even on a machine fast enough to hide the cost, and it pins the
        boundary so a query at the limit is not needlessly escalated.
        """
        cap = ic._PATTERN_SCAN_MAX_CHARS
        trigger = "build a remibrutinib patient cohort for csu "

        at_cap = (trigger + "filler " * cap)[:cap]
        assert len(at_cap) == cap
        assert _node()._pattern_classify(at_cap)["primary_intent"] == "cohort_definition", (
            "a query exactly at the cap must still be classified normally"
        )

        over_cap = at_cap + "x"
        assert _node()._pattern_classify(over_cap)["primary_intent"] == "general", (
            "one character past the cap the pattern layer must abstain"
        )

    def test_scan_cap_clears_every_real_query_bound(self):
        """The cap may never be tightened below real traffic.

        ``/chat`` bounds one message at ``MAX_MESSAGE_CHARS``; the routing gold
        set is the widest corpus of real queries we hold. Abstention must be
        unreachable for both, so the cap costs real traffic nothing.
        """
        from src.api.routes.chat import MAX_MESSAGE_CHARS

        cap = ic._PATTERN_SCAN_MAX_CHARS
        assert cap >= MAX_MESSAGE_CHARS, (
            f"scan cap {cap} is below /chat's own {MAX_MESSAGE_CHARS}-char "
            "message limit — legitimate traffic would be truncated"
        )

        gold = REPO_ROOT / "scripts/benchmarks/routing/data/benchmark_queries_gold.jsonl"
        lengths = [
            len(json.loads(line)["text"]) for line in gold.read_text().splitlines() if line.strip()
        ]
        assert lengths, "gold corpus is empty — the guard would be vacuous"
        assert cap > max(lengths), (
            f"scan cap {cap} does not clear the longest gold query ({max(lengths)} chars)"
        )

    def test_over_cap_query_abstains_below_the_llm_trust_floor(self):
        """Past the cap the pattern layer must ABSTAIN, not classify a prefix.

        ``execute`` trusts a pattern verdict only at ``confidence >= 0.8``;
        below that the query goes to the LLM fallback, which receives it in
        FULL. Abstaining is what makes "the cap only ever fails safe" true.
        """
        cap = ic._PATTERN_SCAN_MAX_CHARS
        query = "build a remibrutinib patient cohort for csu " + ("filler " * cap)
        assert len(query) > cap

        result = _node()._pattern_classify(query)
        assert result["primary_intent"] == "general"
        assert result["confidence"] < 0.8, (
            f"an over-cap query scored {result['confidence']} — at or above the "
            "0.8 trust floor the router skips the LLM fallback and acts on "
            "evidence read from a truncated prefix"
        )

    def test_evidence_past_the_cap_cannot_flip_a_veto(self):
        """A veto beyond the cap must not turn into a CONFIDENT misroute.

        codex iter-1 HIGH, reproduced before the fix. Several patterns are
        ``\\A``-anchored with a whole-query negative lookahead — the KPI
        ``explanation`` entry vetoes itself when any forecast lexeme appears
        ANYWHERE. Scoring a truncated prefix hides that evidence, so the veto
        silently lifts. Measured on the truncating build:

            "what is the trx for kisqali ... please forecast it"
              veto visible (short) -> prediction  @0.867   (correct)
              veto past the cap    -> explanation @0.867   (confident misroute)

        0.867 is above the 0.8 trust floor, so the LLM never saw it — the exact
        confident-misroute class as the #1408 ``\\binterim\\b`` incident.
        """
        cap = ic._PATTERN_SCAN_MAX_CHARS
        head = "what is the trx for kisqali "
        tail = " please forecast it"

        short = head + tail
        assert _node()._pattern_classify(short)["primary_intent"] == "prediction", (
            "control: with the whole query visible the forecast veto holds"
        )

        long_query = head + ("and some more context " * cap) + tail
        assert len(long_query) > cap
        result = _node()._pattern_classify(long_query)
        assert result["confidence"] < 0.8, (
            f"hiding the forecast veto past the cap produced "
            f"{result['primary_intent']}@{result['confidence']} — a confident "
            "route decided on partial evidence"
        )


class TestInterimPatternLinearization1470:
    """``experiment_monitor[3]`` is line-anchored, and that changes nothing.

    Unanchored, the pattern is a pair of zero-width ``.*`` lookaheads: ``re``
    retries it at every one of the N start offsets and each try scans to
    end-of-line, which is the quadratic cost. Because ``.`` excludes newlines,
    the original matches iff SOME SINGLE LINE holds both an ``interim`` and an
    experiment noun — and that is exactly what ``(?m)^`` tests, at line starts
    only. Same language, one pass.
    """

    PATTERN = ic.IntentClassifierNode.INTENT_PATTERNS["experiment_monitor"][3]

    def test_pattern_is_line_anchored(self):
        assert self.PATTERN.startswith("(?m)^"), (
            "the interim gate lost its (?m)^ anchor — issue #1470's quadratic "
            "retry-at-every-offset behaviour returns"
        )

    def test_anchoring_is_semantics_preserving(self):
        """Anchored and unanchored forms agree on every probe, incl. newlines."""
        unanchored = self.PATTERN[len("(?m)^") :]
        probes = [
            "give me an interim readout on the running fabhalta speaker-program test",
            "are there interim analysis triggers in the current experiments?",
            "check enrollment interim",
            "what's our interim cfo's view on q3 pharma spend?",
            "summarize the interim guidance from fda on labeling for fabhalta",
            "who is the interim head of oncology commercial?",
            "in the interim, can you tell me what caused the kisqali trx drop?",
            "interim",
            "the trial finished",
            "",
            # Multi-line: the only place the two forms could diverge.
            "interim numbers\nplease",
            "interim numbers\nfor the trial",
            "the trial is done\ngive me an interim look",
            "run the interim test\nnow",
        ]
        for probe in probes:
            assert bool(re.search(self.PATTERN, probe, re.IGNORECASE)) == bool(
                re.search(unanchored, probe, re.IGNORECASE)
            ), f"anchoring changed the verdict for {probe!r}"

    def test_anchoring_agrees_across_the_routing_corpora(self):
        """Exhaustive equivalence over every stored routing query."""
        unanchored = self.PATTERN[len("(?m)^") :]
        corpora = [
            "scripts/benchmarks/routing/data/benchmark_queries_gold.jsonl",
            "scripts/benchmarks/routing/data/query_pool.jsonl",
            "scripts/benchmarks/routing/data/perturbations.jsonl",
        ]
        checked = 0
        for rel in corpora:
            path = REPO_ROOT / rel
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                query = json.loads(line).get("text", "")
                if not query:
                    continue
                checked += 1
                assert bool(re.search(self.PATTERN, query, re.IGNORECASE)) == bool(
                    re.search(unanchored, query, re.IGNORECASE)
                ), f"anchoring changed the verdict for {query!r}"
        assert checked > 300, f"only {checked} corpus queries checked — guard is too thin"


class TestGenuineQueriesUnaffected1470:
    """Real queries route exactly as they did before either change."""

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("build a remibrutinib patient cohort for csu", "cohort_definition"),
            ("create a fabhalta cohort for pnh patients", "cohort_definition"),
            ("define a cohort of kisqali patients with prior cdk therapy", "cohort_definition"),
            ("filter patients by eligibility criteria", "cohort_definition"),
            (
                "give me an interim readout on the running fabhalta speaker-program test",
                "experiment_monitor",
            ),
            ("what causes conversion drops?", "causal_effect"),
            ("what will be the forecast for next quarter?", "prediction"),
            ("explain how this model works", "explanation"),
        ],
    )
    def test_representative_query_routes_unchanged(self, query, expected):
        assert _node()._pattern_classify(query)["primary_intent"] == expected
