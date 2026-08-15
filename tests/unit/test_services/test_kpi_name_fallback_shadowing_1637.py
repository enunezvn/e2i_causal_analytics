"""Regression tests for #1637: KPI-name shadowing in the recognition fallback.

Reported symptom (eval 2026-08-15, turn 4.6): *"What is the false alert rate and
override rate for triggers?"* returned override rate only, and the answer said
the platform might not track a false-alert metric. It does — ``WS2-TR-005`` has a
calculator, a certified query, and (measured on prod, 2026-08-15) a real value of
**11.26%** over the frontier 30-day window.

The issue was filed as "WS2-TR-005 is missing from the chat alias map". That
diagnosis was wrong, and the measurement that disproved it is worth keeping:
``recognize_kpi("false alert rate")`` ALREADY returned WS2-TR-005 via the
name-token fallback. What actually broke was the phrasing a user reaches for —
``"... for triggers"`` — because the fallback returned the FIRST registry KPI
holding any 4+ char name token found as a SUBSTRING of the query, and
``WS2-TR-001 Trigger Precision`` sorts earlier and owns the token "trigger".

So the defect was never one missing alias. Measured across the registry, TEN of
45 KPIs resolved to some OTHER KPI when asked by their own name:

    WS1-DQ-002 -> WS1-DQ-001    WS2-TR-003 -> WS1-DQ-004   ("lift" inside "up|lift|")
    WS1-DQ-003 -> WS1-DQ-001    WS2-TR-007 -> WS1-DQ-009   ("time" from "Time-to-Release")
    WS1-MP-007 -> WS1-DQ-001    WS3-BI-002 -> WS3-BI-001
    WS2-TR-002 -> WS2-TR-001    BR-002     -> BR-001
    BR-005     -> BR-004        CM-005     -> CM-001

Two independent bugs produced that: **substring** matching (so "lift" matched
inside "uplift") and **registry order** standing in for relevance (so a 1-token
brush beat an exact full-name match). The fix scores candidates by how much of
their name the query actually covers, on word boundaries, and keeps registry
order only as a stable tie-break — so today's winner still wins a genuine tie.

#1360 fixed four members of this class by adding explicit aliases, deliberately
"without touching the fallback". That worked but is per-KPI maintenance; the
class kept re-opening. The invariant test below is the tripwire that makes a new
shadow fail loudly instead of surfacing in an eval months later.
"""

import pytest

from src.kpi.registry import get_registry
from src.services.kpi_resolution import recognize_kpi, recognize_kpi_span

pytestmark = pytest.mark.unit


class TestEveryKpiIsReachableByItsOwnName:
    """The invariant. A KPI that cannot be reached by typing its own name is
    unreachable from chat, no matter how complete its calculator is."""

    def test_no_kpi_resolves_to_a_different_kpi(self):
        """Wrong answers are worse than no answer: a mis-resolution silently
        computes and reports a DIFFERENT metric under the asked-for name."""
        wrong = []
        for kpi in get_registry().get_all():
            got = recognize_kpi(str(kpi.name).lower())
            if got is not None and got.id != kpi.id:
                wrong.append(f"{kpi.id} ({kpi.name!r}) -> {got.id} ({got.name!r})")
        assert not wrong, "KPIs shadowed by another KPI's name token:\n  " + "\n  ".join(wrong)

    def test_every_kpi_resolves_to_itself(self):
        """Stronger form: no KPI is unreachable (None) either. Abbreviation-only
        names (ROC-AUC, PR-AUC, F1 Score) have no token the fallback can see, so
        they carry explicit aliases; this asserts the two tiers together cover
        the whole registry."""
        unreachable = []
        for kpi in get_registry().get_all():
            got = recognize_kpi(str(kpi.name).lower())
            if got is None or got.id != kpi.id:
                unreachable.append(f"{kpi.id} ({kpi.name!r}) -> {got.id if got else None}")
        assert not unreachable, "KPIs unreachable by their own name:\n  " + "\n  ".join(unreachable)


class TestTriggerFamilyNoLongerShadowed:
    """The reported defect and its four siblings, asked the way a user asks."""

    @pytest.mark.parametrize(
        "query,expected_id",
        [
            # The 4.6 defect itself.
            ("what is the false alert rate for triggers", "WS2-TR-005"),
            ("false alert rate for triggers", "WS2-TR-005"),
            # Siblings shadowed by the same "trigger" token.
            ("what is the change-fail rate for triggers", "WS2-TR-008"),
            ("trigger recall", "WS2-TR-002"),
            ("what is the trigger recall this quarter", "WS2-TR-002"),
            # Shadowed by substring matching, not by "trigger".
            ("what is the action rate uplift for triggers", "WS2-TR-003"),
            # Shadowed across workstreams by WS1-DQ-009 "Time-to-Release".
            ("what is the lead time for triggers", "WS2-TR-007"),
        ],
    )
    def test_shadowed_trigger_kpis_resolve_to_themselves(self, query, expected_id):
        kpi = recognize_kpi(query)
        assert kpi is not None, f"{query!r} did not resolve at all"
        assert kpi.id == expected_id, f"{query!r} -> {kpi.id} ({kpi.name}), wanted {expected_id}"


class TestMatchingIsWordBoundaryNotSubstring:
    def test_lift_does_not_match_inside_uplift(self):
        """WS1-DQ-004 'Stacking Lift' owned every query containing 'uplift'."""
        kpi = recognize_kpi("action rate uplift")
        assert kpi is not None and kpi.id == "WS2-TR-003", (
            f"'uplift' matched a substring token: -> {kpi.id if kpi else None}"
        )

    def test_stacking_lift_still_resolves_on_a_real_word_boundary(self):
        """The boundary rule must not cost WS1-DQ-004 its own legitimate asks."""
        for q in ("stacking lift", "what is the stacking lift"):
            kpi = recognize_kpi(q)
            assert kpi is not None and kpi.id == "WS1-DQ-004", f"{q!r} -> {kpi}"


class TestBestMatchBeatsRegistryOrder:
    def test_two_token_match_beats_one_token_match(self):
        """'lead time' matches WS2-TR-007 on BOTH tokens and WS1-DQ-009
        ('Time-to-Release') on one. Coverage decides, not registry position."""
        kpi = recognize_kpi("lead time")
        assert kpi is not None and kpi.id == "WS2-TR-007", f"-> {kpi.id if kpi else None}"

    def test_time_to_release_still_wins_its_own_phrasing(self):
        kpi = recognize_kpi("time-to-release")
        assert kpi is not None and kpi.id == "WS1-DQ-009", f"-> {kpi.id if kpi else None}"


class TestAbbreviationOnlyKpisAreReachable:
    """Names whose every token is under the 4-char floor were invisible to the
    fallback entirely — recognize_kpi returned None and the chat tool answered
    "did not resolve to a defined KPI" for a KPI that is fully implemented."""

    @pytest.mark.parametrize(
        "query,expected_id",
        [
            ("roc-auc", "WS1-MP-001"),
            ("roc auc", "WS1-MP-001"),
            ("what is our roc-auc", "WS1-MP-001"),
            ("pr-auc", "WS1-MP-002"),
            ("pr auc", "WS1-MP-002"),
            ("f1 score", "WS1-MP-003"),
        ],
    )
    def test_abbreviation_kpis_resolve(self, query, expected_id):
        kpi = recognize_kpi(query)
        assert kpi is not None, f"{query!r} did not resolve"
        assert kpi.id == expected_id, f"{query!r} -> {kpi.id}, wanted {expected_id}"


class TestSecondMetricProbeIsInflectionConsistent:
    """``recognize_distinct_metric`` must see a metric mention in either number.

    codex iter-3 raised this as a fail-closed regression for the dispatcher,
    whose causal-ask veto trips on any second metric without directional grammar:
    "what drives conversion rates after patient touches" now detects
    Patient Touch Rate as a second mention and vetoes, where before it did not.

    Measured, that is the removal of an ACCIDENT rather than a new behaviour —
    the SINGULAR spelling of the very same sentence already vetoed. The old
    matcher's rule was "veto on 'patient touch', ignore 'patient touches'", which
    is not a designed carve-out for contextual nouns; it is just the singular-only
    regex showing through. Blast radius on real traffic: of the 48 distinct user
    questions actually asked in the 2026-08-15 eval, ZERO change verdict.

    Weakening the shared matcher for one caller would re-create precisely the
    iter-2 defect (two matchers disagreeing about what names a metric), so the
    rule stays shared and the two spellings are pinned to agree here. Whether the
    #1475 veto is too aggressive for BOTH spellings is a separate question about
    that guard, not about this one.
    """

    @pytest.mark.parametrize(
        "singular,plural",
        [
            ("patient touch", "patient touches"),
            ("override rate", "override rates"),
            ("acceptance rate", "acceptance rates"),
            ("conversion rate", "conversion rates"),
        ],
    )
    def test_both_numbers_are_seen_as_the_same_metric(self, singular, plural):
        from src.services.kpi_resolution import recognize_distinct_metric

        got_singular = recognize_distinct_metric(singular, exclude_id="__none__")
        got_plural = recognize_distinct_metric(plural, exclude_id="__none__")
        assert got_singular is not None, f"{singular!r} is not seen as a metric at all"
        assert got_plural is not None, f"{plural!r} was not seen while {singular!r} was"
        assert got_singular[0].id == got_plural[0].id


class TestSpanContractPreserved:
    """``recognize_kpi_span`` feeds the #1475 governing-head guards, which slice
    the query around the returned span. A span that does not actually cover the
    matched text would corrupt those guards silently."""

    @pytest.mark.parametrize(
        "query",
        [
            "what is the false alert rate for triggers",
            "lead time",
            "action rate uplift",
            "trigger recall",
        ],
    )
    def test_span_covers_real_text_within_the_normalized_query(self, query):
        match = recognize_kpi_span(query)
        assert match is not None, f"{query!r} did not resolve"
        _kpi, normalized, start, end = match
        assert 0 <= start < end <= len(normalized), f"span ({start},{end}) vs {normalized!r}"
        assert normalized[start:end].strip(), "span covers only whitespace"

    def test_span_points_at_the_kpi_mention_not_an_unrelated_word(self):
        match = recognize_kpi_span("what is the false alert rate for triggers")
        assert match is not None
        kpi, normalized, start, end = match
        assert kpi.id == "WS2-TR-005"
        # The span must land inside the "false alert rate" mention, never on the
        # trailing "triggers" that used to drive the match.
        assert "trigger" not in normalized[start:end]
        assert normalized[start:end] in "false alert rate"
