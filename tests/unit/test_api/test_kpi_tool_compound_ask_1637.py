"""#1637 codex iter-1 HIGH: a coordinated multi-metric ask must not be answered as complete.

``kpi_calculate_tool`` computes ONE KPI but takes free-text ``kpi_name``. Given
"false alert rate and override rate for triggers" it resolved to whichever alias
matched first (override rate) and returned it with ``success: True`` — the eval's
turn 4.6 shape, where the answer then reported the missing metric as a tool
limitation rather than a question it never asked.

The guard refuses instead, naming both KPIs, so the caller issues one call per
metric.

WHY IT IS GATED ON A COORDINATOR — this is the part that must not regress. Two
metric mentions alone are NOT evidence of two asks. Measured against the 20
distinct ``kpi_name`` values the model actually passed across the 51-turn
2026-08-15 eval (755 calls), an ungated two-mention guard refuses
**"TRx market share"** — 32 real calls — because it contains both a TRx mention
and a market-share mention while naming exactly one KPI (WS3-BI-008 TRx Share).
Adjacency is a modifier chain; a coordinator between the spans is two asks.

``REAL_EVAL_KPI_NAMES`` below is that full observed corpus, pinned as a
must-not-refuse fixture so a future tightening of this guard fails here.
"""

import pytest

from src.api.routes.chatbot_tools import kpi_calculate_tool

pytestmark = pytest.mark.unit

#: Every distinct kpi_name the model passed across the 51-turn 2026-08-15 eval.
REAL_EVAL_KPI_NAMES = [
    "TRx",
    "NRx",
    "conversion rate",
    "TRx market share",
    "market share",
    "intent to prescribe",
    "treatment effect",
    "trigger precision",
    "acceptance rate",
    "trigger override rate",
    "trigger action prescription lift",
    "trigger funnel conversion",
    "rep visit",
    "TRx share",
    "ROI",
    "conversion_rate",
    "hcp_engagement_score",
    "nrx",
    "trx",
    "market_share",
]


def _is_compound_refusal(result: dict) -> bool:
    """A refusal specifically from the multi-metric guard (not 'did not resolve',
    not a downstream calculation error)."""
    return result.get("success") is False and "names more than one KPI" in str(
        result.get("error", "")
    )


class TestCoordinatedAsksAreRefused:
    @pytest.mark.parametrize(
        "kpi_name",
        [
            "false alert rate and override rate for triggers",
            "what is the false alert rate and override rate for triggers",
            "TRx and NRx",
            "TRx, NRx",
            "conversion rate & ROI",
        ],
    )
    async def test_coordinated_multi_metric_is_refused(self, kpi_name):
        result = await kpi_calculate_tool.ainvoke({"kpi_name": kpi_name})
        assert _is_compound_refusal(result), (
            f"{kpi_name!r} was answered as a single metric: "
            f"success={result.get('success')} kpi={result.get('kpi_id')} "
            f"value={result.get('value')}"
        )

    async def test_refusal_names_both_metrics_and_tells_caller_what_to_do(self):
        """A refusal that does not name the two metrics just moves the dead end —
        the caller needs enough to issue the two calls itself."""
        result = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "false alert rate and override rate for triggers"}
        )
        blob = f"{result.get('error', '')} {result.get('hint', '')}".lower()
        assert "false alert rate" in blob, blob
        assert "override rate" in blob, blob
        assert "once per metric" in blob, blob

    async def test_refusal_does_not_fabricate_a_value(self):
        result = await kpi_calculate_tool.ainvoke({"kpi_name": "TRx and NRx"})
        assert result.get("value") is None
        assert result.get("success") is False


class TestRealEvalArgumentsAreNeverRefused:
    """The regression this guard could plausibly cause, pinned against the real
    observed corpus rather than against invented examples."""

    @pytest.mark.parametrize("kpi_name", REAL_EVAL_KPI_NAMES)
    async def test_observed_kpi_names_are_not_refused_as_compound(self, kpi_name):
        result = await kpi_calculate_tool.ainvoke({"kpi_name": kpi_name})
        assert not _is_compound_refusal(result), (
            f"{kpi_name!r} is a single-metric ask the model really made, "
            f"but the compound guard refused it: {result.get('error')}"
        )

    async def test_trx_market_share_is_a_modifier_chain_not_two_asks(self):
        """The specific 32-call case an ungated guard breaks. It must still
        resolve to WS3-BI-008 and compute."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": "TRx market share"})
        assert not _is_compound_refusal(result), result.get("error")
        assert result.get("kpi_id") == "WS3-BI-008", result

    @pytest.mark.parametrize(
        "kpi_name",
        [
            # A coordinator token with a NON-metric right side is not two asks.
            "conversion rate, by brand",
            "conversion rate for Kisqali, west region",
            "acceptance rate, last quarter",
            # A second metric MENTION that is part of the first metric's own
            # phrase — adjacency, so the gap holds no coordinator.
            "TRx market share, Kisqali",
        ],
    )
    async def test_punctuation_alone_does_not_trigger_the_guard(self, kpi_name):
        """The gate reads the text BETWEEN the two mentions. A comma elsewhere in
        the string — before a brand, a region, a period — is not coordination."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": kpi_name})
        assert not _is_compound_refusal(result), (
            f"{kpi_name!r} is one metric with trailing scope, but was refused: "
            f"{result.get('error')}"
        )


class TestCoordinatorFormsAreCovered:
    """Coordination is not only the word 'and'."""

    @pytest.mark.parametrize(
        "kpi_name",
        ["TRx, NRx", "conversion rate & ROI", "ROI as well as TRx", "NRx plus TRx"],
    )
    async def test_other_coordinators_also_refuse(self, kpi_name):
        result = await kpi_calculate_tool.ainvoke({"kpi_name": kpi_name})
        assert _is_compound_refusal(result), (
            f"{kpi_name!r} coordinates two metrics but was answered as one: "
            f"kpi={result.get('kpi_id')} value={result.get('value')}"
        )

    @pytest.mark.parametrize(
        "kpi_name",
        [
            "TRx or NRx",
            "conversion rate or ROI",
            "TRx alongside NRx",
            "TRx together with NRx",
            "TRx along with NRx",
        ],
    )
    async def test_additive_and_disjunctive_phrasings_refuse(self, kpi_name):
        """codex iter-7. "or" is as much a two-metric ask as "and"."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": kpi_name})
        assert _is_compound_refusal(result), (
            f"{kpi_name!r} names two metrics but was answered as one: "
            f"kpi={result.get('kpi_id')} value={result.get('value')}"
        )

    async def test_bare_with_is_not_a_coordinator(self):
        """The deliberate omission. "with" is a preposition that forms MODIFIER
        relationships — "market share with respect to TRx" names ONE KPI. The
        asymmetry justifies the caution: a missed coordinator degrades to the
        pre-guard behaviour, while a false one refuses an answerable question."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": "market share with respect to TRx"})
        assert not _is_compound_refusal(result), result.get("error")

    @pytest.mark.parametrize(
        "kpi_name",
        ["TRx vs NRx", "TRx vs. NRx", "TRx versus NRx", "TRx compared to NRx"],
    )
    async def test_comparison_phrasing_also_refuses(self, kpi_name):
        """Comparison is the MOST natural way to ask for two metrics at once, so
        omitting it would have left the single-call failure intact for the very
        shape most likely to produce it (codex iter-2)."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": kpi_name})
        assert _is_compound_refusal(result), (
            f"{kpi_name!r} compares two metrics but was answered as one: "
            f"kpi={result.get('kpi_id')} value={result.get('value')}"
        )


class TestEveryMentionIsExamined:
    """#1637 codex iter-4 HIGH — stopping at the FIRST further mention re-created
    the same fail-silent one mention along.

    ``recognize_distinct_metric`` returns a single match in vocabulary order. For
    "TRx market share and ROI" that is the ADJACENT "TRx" (gap " ", no
    coordinator), so the guard passed and the tool computed TRx share while
    silently dropping ROI — with ROI sitting right there behind an "and".
    """

    @pytest.mark.parametrize(
        "kpi_name",
        [
            "TRx market share and ROI",
            "TRx market share, ROI",
            "TRx market share and conversion rate",
        ],
    )
    async def test_coordinated_metric_behind_an_adjacent_mention_is_caught(self, kpi_name):
        result = await kpi_calculate_tool.ainvoke({"kpi_name": kpi_name})
        assert _is_compound_refusal(result), (
            f"{kpi_name!r} has a coordinated metric behind an adjacent modifier "
            f"mention and was answered as one: kpi={result.get('kpi_id')} "
            f"value={result.get('value')}"
        )

    async def test_refusal_names_the_coordinated_metric_not_the_adjacent_one(self):
        """The message must name the metric the caller actually also asked for
        (ROI) — naming only the adjacent modifier mention would send the caller
        after the wrong second call."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": "TRx market share and ROI"})
        blob = f"{result.get('error', '')} {result.get('hint', '')}"
        assert "ROI" in blob or "Return on Investment" in blob, blob

    async def test_scan_cap_exhaustion_fails_closed(self):
        """codex iter-6: hitting the scan cap used to log and then compute. Once
        the code has established that a further coordinated metric may be
        unexamined, returning success is a false-complete — the warning would
        document the wrong answer rather than avoid it. Adversarial input, but
        the contract is the point: never answer an ask you did not fully read."""
        stuffed = "TRx market share for " + "TRx " * 10 + "and ROI"
        result = await kpi_calculate_tool.ainvoke({"kpi_name": stuffed})
        assert result.get("success") is False, result
        assert result.get("value") is None, result

    async def test_a_repeated_mention_does_not_abort_the_scan(self):
        """codex iter-5: breaking on a repeated id abandoned the scan before the
        coordinated metric. "TRx market share for TRx and ROI" mentions TRx twice
        and ROI once — stopping at the repeat answered TRx share as complete and
        dropped ROI. Only lack of PROGRESS may end the scan."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": "TRx market share for TRx and ROI"})
        assert _is_compound_refusal(result), (
            f"a repeated mention aborted the scan before the coordinated metric: {result}"
        )
        blob = f"{result.get('error', '')} {result.get('hint', '')}"
        assert "ROI" in blob or "Return on Investment" in blob, blob

    async def test_three_metrics_are_all_named_in_the_hint(self):
        """Naming only the first two would steer the caller into dropping the
        third — the very failure this guard exists to prevent (codex iter-5)."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": "TRx and NRx and ROI"})
        assert _is_compound_refusal(result), result
        hint = str(result.get("hint", ""))
        assert "TRx" in hint, hint
        assert "NRx" in hint, hint
        assert "Return on Investment" in hint or "ROI" in hint, hint

    async def test_adjacent_mention_alone_still_computes(self):
        """The fix must not turn the 32-call modifier chain into a refusal."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": "TRx market share"})
        assert not _is_compound_refusal(result), result.get("error")
        assert result.get("kpi_id") == "WS3-BI-008", result


class TestAggressiveCoordinatorsStayHarmless:
    """`against` and `/` are the widest tokens in the gate — pinned here because
    they are the ones most likely to over-fire (codex iter-3)."""

    @pytest.mark.parametrize(
        "kpi_name", ["market share against competitors", "conversion rate against target"]
    )
    async def test_coordinator_word_without_a_second_metric_computes(self, kpi_name):
        """The gate is only consulted AFTER a second metric is found, so a
        coordinator word with an ordinary noun after it changes nothing."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": kpi_name})
        assert not _is_compound_refusal(result), result.get("error")

    async def test_slash_between_two_metrics_refuses_rather_than_answering_one(self):
        """ "TRx/NRx ratio" is not a defined KPI. Without the guard it resolves to
        NRx alone and reports that single number AS the ratio — a wrong answer.
        Refusing and naming both lets the caller fetch each and divide."""
        result = await kpi_calculate_tool.ainvoke({"kpi_name": "TRx/NRx ratio"})
        assert _is_compound_refusal(result), result
        assert result.get("value") is None


class TestPluralAsksAreDetectedOnBothSides:
    """#1637 codex iter-2 HIGH — the asymmetry that reopened the fail-silent.

    Primary recognition learned plurals (``_alias_pattern``) while the
    second-metric probe (``recognize_distinct_metric``) still matched singulars
    only, so a fully-plural coordinated ask resolved the first KPI, never saw the
    second, and answered one metric as complete. Both now share
    ``_PLURAL_SUFFIX``.
    """

    @pytest.mark.parametrize(
        "kpi_name",
        [
            "acceptance rates and override rates",
            "override rates and acceptance rates",
            "conversion rates and ROI",
        ],
    )
    async def test_plural_coordinated_asks_are_refused(self, kpi_name):
        result = await kpi_calculate_tool.ainvoke({"kpi_name": kpi_name})
        assert _is_compound_refusal(result), (
            f"{kpi_name!r} names two metrics in plural form but was answered as "
            f"one: kpi={result.get('kpi_id')} value={result.get('value')}"
        )

    async def test_singular_and_plural_forms_agree(self):
        """The singular and plural spellings of one ask must reach the same
        verdict — divergence there IS the bug."""
        singular = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "acceptance rate and override rate"}
        )
        plural = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "acceptance rates and override rates"}
        )
        assert _is_compound_refusal(singular) == _is_compound_refusal(plural)
