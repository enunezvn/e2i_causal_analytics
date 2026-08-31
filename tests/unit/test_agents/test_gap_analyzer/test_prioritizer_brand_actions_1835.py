"""#1835 — recommended_action must be brand-aware.

Measured 2026-08-30 on prod (gap_analyses payloads): Kisqali's #1 and
Remibrutinib's #1 opportunity carried the VERBATIM identical sentence
("Execute comprehensive market access and HCP engagement program in west to
close TRx gap (restore prior performance)"). The Strategic Brief grounds on
this text, so an oncology brand and a CSU brand got the same recommendation.

Brand identity and HCP audience are derived from existing SSOTs — never
invented here:
- ``SUPPORTED_BRANDS`` (cohort_constructor.constants) — brand identity
- ``Brand`` / ``SpecialtyEnum`` (ml.synthetic.config) — DB-enum display names
- ``HCPGenerator.BRAND_SPECIALTY_DIST`` (ml.synthetic.generators.hcp_generator)
  — the targeted specialties per brand (mirrored + pinned, see
  ``TestSsotPins``)
- ``INTERVENTION_CATALOG`` (digital_twin.effect.provider) — channel labels
"""

from itertools import combinations
from typing import Dict, List

import pytest

from src.agents.cohort_constructor.constants import SUPPORTED_BRANDS
from src.agents.gap_analyzer.action_templates import (
    ACTION_CHANNELS,
    ACTION_METRICS,
    BRAND_TARGET_SPECIALTIES,
    GAP_TYPE_SUFFIXES,
    MAX_ACTION_CHARS,
    SPECIALTY_PRACTITIONER,
    brand_action_context,
)
from src.agents.gap_analyzer.nodes.prioritizer import PrioritizerNode
from src.agents.gap_analyzer.state import GapAnalyzerState, PerformanceGap, ROIEstimate
from src.digital_twin.effect.provider import SUPPORTED_INTERVENTIONS
from src.ml.synthetic.config import Brand, RegionEnum, SpecialtyEnum
from src.ml.synthetic.generators.hcp_generator import HCPGenerator

DIFFICULTIES = ("low", "medium", "high")
GAP_TYPES = ("vs_target", "vs_benchmark", "vs_potential", "temporal")
# Every metric the templates know about PLUS one they don't (default template).
METRICS = tuple(ACTION_METRICS) + ("unknown_metric",)
# Longest (segment, segment_value) pair the node can ever see. The DB
# ``region_type`` enum is the only substrate dimension backed by real data
# (gap_detector #851); the detector's OWN mock-fallback path
# (``_fetch_mock_current/target_performance``, gap_detector.py:769/842) also
# fabricates "specialty" (Oncology/Cardiology/Rheumatology/Neurology) and
# "hcp_tier" (Tier 1/2/3) when the caller requests those segments — so both
# the segment NAME and the segment VALUE vary, and they are NOT independent:
# "Rheumatology" only ever appears together with segment="specialty" (never
# with segment="region"). #1835's codex iter-1 audit caught exactly this —
# an earlier version of this test paired the longest VALUE with the
# ("region") default segment, missing the ("specialty") pairing that is
# actually 1 char over budget (161 > 160) because "specialty" is longer than
# "region" and the brand-aware trx/low template names the segment dimension.
LONGEST_SEGMENT_NAME = "specialty"
LONGEST_SEGMENT_VALUE = max([*(r.value for r in RegionEnum), "Rheumatology"], key=len)
assert LONGEST_SEGMENT_VALUE == "Rheumatology"  # the pairing partner of LONGEST_SEGMENT_NAME


def _gap(
    metric: str = "trx",
    segment_value: str = "west",
    gap_type: str = "temporal",
    segment: str = "region",
) -> PerformanceGap:
    return {
        "gap_id": f"{segment}_{segment_value}_{metric}_{gap_type}",
        "metric": metric,
        "segment": segment,
        "segment_value": segment_value,
        "current_value": 400.0,
        "target_value": 500.0,
        "gap_size": 100.0,
        "gap_percentage": 20.0,
        "gap_type": gap_type,  # type: ignore[typeddict-item]
    }


def _roi(gap_id: str, cost: float = 100_000.0) -> ROIEstimate:
    # risk_adjusted_roi == expected_roi because total_risk_adjustment is the
    # risk REDUCTION amount (roi_calculation.py feeds 1 - total_risk_adj into
    # the simulator), so 0.0 means no adjustment was applied.
    return {
        "gap_id": gap_id,
        "estimated_revenue_impact": cost * 4,
        "estimated_cost_to_close": cost,
        "expected_roi": 3.0,
        "risk_adjusted_roi": 3.0,
        "payback_period_months": 6,
        "confidence_interval": None,
        "attribution_level": "full",
        "attribution_rate": 1.0,
        "total_risk_adjustment": 0.0,
        "value_by_driver": None,
        "confidence": 0.8,
        "assumptions": [],
    }


def _state(brand: str, gaps: List[PerformanceGap], rois: List[ROIEstimate]) -> GapAnalyzerState:
    return {  # type: ignore[typeddict-item]
        "query": "test",
        "metrics": ["trx"],
        "segments": ["region"],
        "brand": brand,
        "time_period": "current_quarter",
        "filters": None,
        "gap_type": "temporal",
        "min_gap_threshold": 5.0,
        "max_opportunities": 10,
        "gaps_detected": gaps,
        "gaps_by_segment": None,
        "total_gap_value": 1000.0,
        "roi_estimates": rois,
        "total_addressable_value": 100000.0,
        "prioritized_opportunities": None,
        "quick_wins": None,
        "strategic_bets": None,
        "executive_summary": None,
        "key_insights": None,
        "detection_latency_ms": 100,
        "roi_latency_ms": 50,
        "total_latency_ms": 0,
        "segments_analyzed": 1,
        "errors": [],
        "warnings": [],
        "status": "prioritizing",
    }


def _render(
    brand,
    metric="trx",
    difficulty="high",
    segment_value="west",
    gap_type="temporal",
    segment="region",
):
    gap = _gap(metric=metric, segment_value=segment_value, gap_type=gap_type, segment=segment)
    return PrioritizerNode()._generate_action(gap, _roi(gap["gap_id"]), difficulty, brand)


# The sentence measured identical across brands on prod (issue #1835).
# HISTORICAL pin: #1854 moved every segment mention to the trailing strippable
# position, so the LIVE neutral text is NEUTRAL_TRX_HIGH_TEMPORAL below; this
# constant stays as the incident-repro reference the brand texts must differ from.
PROD_IDENTICAL_SENTENCE = (
    "Execute comprehensive market access and HCP engagement program in west "
    "to close TRx gap (restore prior performance)"
)

# Today's exact neutral fail-open text (trx/high/west/temporal, post-#1854):
# same pre-#1835 neutral voice (no brand, no audience), segment moved trailing.
NEUTRAL_TRX_HIGH_TEMPORAL = (
    "Execute comprehensive market access and HCP engagement program to close "
    "TRx gap in west (restore prior performance)"
)


class TestBrandDistinctness:
    """Same metric / difficulty / segment, different brand -> different text."""

    def test_prod_repro_kisqali_vs_remibrutinib_differ(self):
        kisqali = _render("Kisqali")
        remi = _render("Remibrutinib")
        assert kisqali != remi
        assert kisqali != PROD_IDENTICAL_SENTENCE
        assert remi != PROD_IDENTICAL_SENTENCE

    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_every_template_is_pairwise_distinct_across_brands(self, metric, difficulty):
        rendered = {b.value: _render(b.value, metric, difficulty) for b in Brand}
        for a, b in combinations(rendered, 2):
            assert rendered[a] != rendered[b], f"{metric}/{difficulty}: {a} == {b}"


class TestBrandAndAudienceNamed:
    """Each brand's text names the brand and its SSOT-derived HCP audience."""

    @pytest.mark.parametrize("brand", [b.value for b in Brand])
    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_names_brand_and_audience(self, brand, metric, difficulty):
        text = _render(brand, metric, difficulty)
        ctx = brand_action_context(brand)
        assert ctx is not None
        assert ctx.name in text, text
        # Every practitioner noun of the audience appears (singular attributive
        # "oncologist engagement" or plural "with oncologists").
        for noun in ctx.audience.split("/"):
            assert noun in text, f"{noun!r} missing from {text!r}"
        assert "west" in text

    def test_kisqali_audience_is_oncologist(self):
        assert brand_action_context("kisqali").audience == "oncologist"

    def test_remibrutinib_audience_is_dermatologist_and_allergist(self):
        assert brand_action_context("remibrutinib").audience == "dermatologist/allergist"

    def test_fabhalta_audience_is_hematologist_and_internist(self):
        assert brand_action_context("fabhalta").audience == "hematologist/internist"

    def test_display_name_is_db_enum_casing_regardless_of_request_casing(self):
        for raw in ("kisqali", "Kisqali", "KISQALI", "  kisqali "):
            assert brand_action_context(raw).name == Brand.KISQALI.value


class TestLengthBudget:
    """Every metric x difficulty x gap_type x brand x LONGEST segment value
    renders within the brief's 160-char truncation budget (exhaustive)."""

    def test_budget_constant_matches_executive_brief_truncation(self):
        assert MAX_ACTION_CHARS == 160

    # brand=None exercises the NEUTRAL/unknown-brand fallback path too — #1835
    # codex iter-2 found the exhaustive length test covered only the 3 real
    # brands, leaving the fallback path's budget unverified (its own worst
    # case, NEUTRAL_TEMPLATES["trx"]["low"], is the one template that still
    # interpolates {segment}, as "{segment}-targeted" since #1854 — see
    # test_neutral_path_worst_case_fits below).
    @pytest.mark.parametrize("brand", [b.value for b in Brand] + [None])
    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    @pytest.mark.parametrize("gap_type", GAP_TYPES)
    def test_longest_rendering_fits(self, brand, metric, difficulty, gap_type):
        text = _render(
            brand,
            metric,
            difficulty,
            LONGEST_SEGMENT_VALUE,
            gap_type,
            segment=LONGEST_SEGMENT_NAME,
        )
        assert len(text) <= MAX_ACTION_CHARS, f"{len(text)} chars: {text}"

    def test_neutral_path_worst_case_fits(self):
        """NEUTRAL_TEMPLATES["trx"]["low"] is the one template that still
        interpolates {segment} — inline as "{segment}-targeted" since #1854
        moved the segment value trailing (pre-#1854 it was a trailing
        "({segment})" parenthetical) — exercise its worst real (segment, value)
        pairing explicitly rather than relying on it being swept up by
        test_longest_rendering_fits' LONGEST_SEGMENT_NAME default."""
        text = _render(None, "trx", "low", LONGEST_SEGMENT_VALUE, "temporal", segment="specialty")
        assert len(text) <= MAX_ACTION_CHARS, f"{len(text)} chars: {text}"

    @pytest.mark.parametrize("brand", [b.value for b in Brand])
    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    @pytest.mark.parametrize("gap_type", GAP_TYPES)
    def test_brand_aware_rendering_is_insensitive_to_segment_name(
        self, brand, metric, difficulty, gap_type
    ):
        """The #1835 codex iter-1 fix dropped "({segment})" from
        BRAND_TEMPLATES["trx"]["low"] (the only brand-aware template that
        ever referenced {segment}) to close a 161-char overflow, so — unlike
        NEUTRAL_TEMPLATES — the brand-resolved rendering no longer depends on
        WHICH segment dimension the gap came from, only on segment_value.
        #1835 codex iter-2 found a superseded version of this test asserted
        only "fits the budget" for segment="region", which was already
        implied by (and byte-identical to) the segment="specialty" case
        covered above — a redundant, not-actually-distinct assertion. This
        pins the real invariant (identity, not just length) so a future
        change that re-adds {segment} to only SOME brand templates — the
        exact shape of bug that caused the overflow — breaks this test
        first, before a length budget is even at risk."""
        region_text = _render(
            brand, metric, difficulty, LONGEST_SEGMENT_VALUE, gap_type, segment="region"
        )
        specialty_text = _render(
            brand, metric, difficulty, LONGEST_SEGMENT_VALUE, gap_type, segment="specialty"
        )
        assert region_text == specialty_text


class TestUnknownBrandFallsOpen:
    """Unknown / None brand -> today's neutral template, never a KeyError."""

    @pytest.mark.parametrize("brand", [None, "", "competitor", "other", "acme-brand"])
    def test_neutral_template_is_todays_exact_text(self, brand):
        assert _render(brand) == NEUTRAL_TRX_HIGH_TEMPORAL

    def test_neutral_trx_low_keeps_segment_dimension_inline(self):
        # #1854: the pre-#1835 trailing "({segment})" moved inline as
        # "{segment}-targeted" — a paren group between the segment value and the
        # gap-type qualifier would defeat the brief's suffix strip entirely.
        assert _render(None, "trx", "low", "Northeast", "vs_target") == (
            "Launch a region-targeted sampling campaign to drive TRx growth in Northeast"
        )

    def test_missing_brand_argument_keeps_todays_signature(self):
        gap = _gap()
        assert PrioritizerNode()._generate_action(gap, _roi(gap["gap_id"]), "high") == (
            NEUTRAL_TRX_HIGH_TEMPORAL
        )

    def test_unknown_brand_context_is_none(self):
        assert brand_action_context("competitor") is None
        assert brand_action_context(None) is None


class TestGapTypeSuffixes:
    """The gap-type suffix semantics survive the brand-aware rewrite."""

    @pytest.mark.parametrize("brand", [None, "Kisqali", "Fabhalta", "Remibrutinib"])
    @pytest.mark.parametrize("gap_type", GAP_TYPES)
    def test_suffix_preserved(self, brand, gap_type):
        text = _render(brand, gap_type=gap_type)
        suffix = GAP_TYPE_SUFFIXES.get(gap_type, "")
        assert text.endswith(suffix)
        for other, other_suffix in GAP_TYPE_SUFFIXES.items():
            if other != gap_type:
                assert other_suffix not in text

    def test_suffix_table_is_todays_three(self):
        assert GAP_TYPE_SUFFIXES == {
            "vs_benchmark": " (benchmark-driven)",
            "vs_potential": " (top-decile target)",
            "temporal": " (restore prior performance)",
        }


class TestNodeWiring:
    """The node passes state['brand'] through — the prod-observed identical
    top action must differ once two brands run the same gap."""

    @pytest.mark.asyncio
    async def test_execute_uses_state_brand(self):
        gap = _gap()
        roi = _roi(gap["gap_id"])
        actions = {}
        for brand in ("Kisqali", "Remibrutinib", "Fabhalta"):
            result = await PrioritizerNode().execute(_state(brand, [gap], [roi]))
            assert result["status"] == "completed", result
            actions[brand] = result["prioritized_opportunities"][0]["recommended_action"]
        assert len(set(actions.values())) == 3, actions
        assert "Kisqali" in actions["Kisqali"] and "oncologist" in actions["Kisqali"]

    @pytest.mark.asyncio
    async def test_execute_with_unknown_brand_completes_with_neutral_text(self):
        gap = _gap()  # cost 100k / 20% gap -> the node rates it "medium"
        result = await PrioritizerNode().execute(_state("competitor", [gap], [_roi(gap["gap_id"])]))
        assert result["status"] == "completed", result
        assert result["prioritized_opportunities"][0]["implementation_difficulty"] == "medium"
        # Neutral medium wording: pre-#1835 voice (no brand, no audience), the
        # segment moved trailing by #1854.
        assert result["prioritized_opportunities"][0]["recommended_action"] == (
            "Implement multichannel engagement strategy for HCPs to increase TRx in west "
            "(restore prior performance)"
        )


class TestSsotPins:
    """The audience mirror and vocab maps drift-fail loudly against the SSOTs."""

    def test_target_specialties_mirror_hcp_generator_ordered_by_share(self):
        expected: Dict[Brand, tuple] = {}
        for brand, dist in HCPGenerator.BRAND_SPECIALTY_DIST.items():
            ordered = sorted(dist.items(), key=lambda kv: -kv[1])  # stable: ties keep SSOT order
            expected[brand] = tuple(spec for spec, _share in ordered)
        assert BRAND_TARGET_SPECIALTIES == expected

    def test_every_supported_brand_has_a_target_audience(self):
        assert {b.value.lower() for b in BRAND_TARGET_SPECIALTIES} == set(SUPPORTED_BRANDS)

    def test_every_specialty_enum_member_has_a_practitioner_noun(self):
        assert set(SPECIALTY_PRACTITIONER) == set(SpecialtyEnum)
        for noun in SPECIALTY_PRACTITIONER.values():
            assert noun and noun == noun.strip() and noun.islower() and "/" not in noun

    def test_template_channels_are_catalog_interventions(self):
        assert ACTION_CHANNELS
        assert set(ACTION_CHANNELS) <= SUPPORTED_INTERVENTIONS

    @pytest.mark.parametrize("brand", list(Brand))
    def test_brand_enum_and_supported_brands_agree(self, brand):
        assert brand.value.lower() in SUPPORTED_BRANDS
