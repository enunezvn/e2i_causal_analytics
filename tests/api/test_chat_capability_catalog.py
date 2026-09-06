"""Unit tests for src.services.chat_capability_catalog.

Everything DB-backed is injected through the two loader callables, so these
tests run without Supabase. The KPI registry (YAML) and the agent roster
(factory config) are real: they are code, and the point of the catalog is
that its lists come from code.
"""

from __future__ import annotations

import asyncio
import re as _re
import time
from dataclasses import dataclass as _dataclass
from typing import Any, Dict, List

import pytest

from src.kpi.segmented_history import SEGMENTED_KPI_QUERY_FAMILIES
from src.services import chat_capability_catalog as cat

# =============================================================================
# FIXTURE DATA
# =============================================================================

COVERAGE_ROWS: List[Dict[str, Any]] = [
    {"kpi_id": "WS3-BI-005", "brand": "", "region": "", "points": 24},
    {"kpi_id": "WS3-BI-005", "brand": "Kisqali", "region": "", "points": 24},
    # NBRx: per-brand scopes only, no '' row -> per_brand_only
    {"kpi_id": "WS3-BI-007", "brand": "Kisqali", "region": "", "points": 24},
    {"kpi_id": "WS3-BI-007", "brand": "Fabhalta", "region": "", "points": 24},
    # zero points is not a trend
    {"kpi_id": "WS3-BI-010", "brand": "", "region": "", "points": 0},
    # region-scoped row does not make WS3-BI-010 trendable
    {"kpi_id": "WS3-BI-010", "brand": "", "region": "west", "points": 24},
    # an id that is not in the registry is never offered
    {"kpi_id": "WS3-BI-099", "brand": "", "region": "", "points": 12},
    # junk row is skipped
    {"kpi_id": "", "brand": None, "points": "x"},
]

OUTCOMES: List[str] = [
    "treatment_initiated",
    "persistent_180d",
    "trx_volume",
    "nrx_volume",
    "discontinued_180d",
    "roi",
    "adopted",
    "trx_market_share",
    "nbrx_volume",
    "intent_to_prescribe",
    "adherent_180d",
    "action_taken",
    "low_gap_180d",
    "conversion_flag",
]


async def _coverage() -> List[Dict[str, Any]]:
    return list(COVERAGE_ROWS)


async def _outcomes() -> List[str]:
    return list(OUTCOMES)


async def _boom() -> Any:
    raise RuntimeError("db down")


async def _empty() -> list:
    return []


async def make_catalog(coverage=_coverage, outcomes=_outcomes) -> cat.CapabilityCatalog:
    return await cat.build_capability_catalog(coverage_loader=coverage, outcomes_loader=outcomes)


# =============================================================================
# BUILDER
# =============================================================================


async def test_kpis_come_from_the_registry():
    c = await make_catalog()
    ids = {k.id for k in c.kpis}
    assert "WS3-BI-005" in ids
    assert len(ids) >= 40
    assert c.kpi_name("WS3-BI-005") == "Total Prescriptions (TRx)"
    # unknown ids fall back to the id itself (never KeyError in a prompt)
    assert c.kpi_name("NOPE-1") == "NOPE-1"


async def test_trend_sets_from_coverage_rows():
    c = await make_catalog()
    assert c.trend_kpi_ids == frozenset({"WS3-BI-005", "WS3-BI-007"})
    assert c.per_brand_only_trend_ids == frozenset({"WS3-BI-007"})
    assert "WS3-BI-010" not in c.trend_kpi_ids
    assert "WS3-BI-099" not in c.trend_kpi_ids


async def test_axis_kpis_from_segmented_history_families():
    c = await make_catalog()
    assert c.axis_kpi_ids == frozenset(SEGMENTED_KPI_QUERY_FAMILIES)


async def test_outcomes_sorted_deduped_and_roster_present():
    async def dup() -> List[str]:
        return ["roi", "roi", "adopted", ""]

    c = await make_catalog(outcomes=dup)
    assert c.causal_outcomes == ("adopted", "roi")
    assert "The E2I system has" in c.agent_roster
    assert c.degraded == ()


async def test_loader_failure_marks_degraded_and_does_not_raise():
    c = await make_catalog(coverage=_boom, outcomes=_boom)
    assert set(c.degraded) == {"trend_coverage", "causal_outcomes"}
    assert c.trend_kpi_ids == frozenset()
    assert c.causal_outcomes == ()
    # code-derived fields survive a DB outage
    assert len(c.kpis) >= 40
    assert c.axis_kpi_ids == frozenset(SEGMENTED_KPI_QUERY_FAMILIES)


async def test_empty_results_are_degraded_too(caplog):
    # KPIHistoryRepository.get_coverage returns [] on error AND when it has no
    # client; an empty coverage view is not a realistic prod state.
    with caplog.at_level("WARNING", logger="src.services.chat_capability_catalog"):
        c = await make_catalog(coverage=_empty, outcomes=_empty)
    assert set(c.degraded) == {"trend_coverage", "causal_outcomes"}
    messages = [r.getMessage() for r in caplog.records]
    assert any("trend coverage empty" in m for m in messages)
    assert any("causal outcomes empty" in m for m in messages)


async def test_stalled_loader_times_out_and_degrades(monkeypatch, caplog):
    """A loader that never answers is bounded by CATALOG_LOADER_TIMEOUT_SECONDS and
    marks its field degraded; the other loader still lands."""
    monkeypatch.setattr(cat, "CATALOG_LOADER_TIMEOUT_SECONDS", 0.05)

    async def stalled() -> List[Dict[str, Any]]:
        await asyncio.sleep(10)
        return await _coverage()

    started = time.monotonic()
    with caplog.at_level("WARNING", logger="src.services.chat_capability_catalog"):
        c = await make_catalog(coverage=stalled, outcomes=_outcomes)
    assert time.monotonic() - started < 2.0
    assert c.degraded == ("trend_coverage",)
    assert c.trend_kpi_ids == frozenset()
    assert c.causal_outcomes  # the outcomes loader was not affected
    messages = [r.getMessage() for r in caplog.records]
    assert any("trend coverage unavailable: TimeoutError" in m for m in messages)


# =============================================================================
# RENDERER
# =============================================================================


async def test_render_lists_registry_kpis_by_area():
    c = await make_catalog()
    block = cat.render_catalog_block(c)
    assert block.startswith("WHAT THE ASSISTANT CAN DO")
    assert "A. KPI values" in block
    assert "- Business impact: " in block
    assert "Total Prescriptions (TRx)" in block


def _section(block: str, letter: str) -> str:
    return next(line for line in block.splitlines() if line.startswith(f"{letter}. "))


async def test_render_trend_and_axis_kpis_by_name():
    c = await make_catalog()
    block = cat.render_catalog_block(c)
    b_line = _section(block, "B")
    assert f"{c.kpi_name('WS3-BI-007')} (per brand only)" in b_line
    assert c.kpi_name("WS3-BI-005") in b_line
    # axis KPIs are named in the comparison clause
    for kpi_id in SEGMENTED_KPI_QUERY_FAMILIES:
        assert c.kpi_name(kpi_id) in b_line
    # brand-specific registry KPIs carry their brand in section A
    brand_entry = next(e for e in c.kpis if e.brand)
    assert f"{brand_entry.name} ({brand_entry.brand} only)" in block


async def test_render_empty_trend_set_falls_back_without_dangling_list():
    async def region_only() -> List[Dict[str, Any]]:
        return [{"kpi_id": "WS3-BI-005", "brand": "", "region": "west", "points": 24}]

    c = await make_catalog(coverage=region_only)
    assert c.trend_kpi_ids == frozenset() and c.degraded == ()
    b_line = _section(cat.render_catalog_block(c), "B")
    assert "for ;" not in b_line
    assert "coverage list is unavailable" in b_line


async def test_render_causal_outcomes_as_registry_nodes():
    c = await make_catalog()
    block = cat.render_catalog_block(c)
    assert "for these OUTCOMES only: " + ", ".join(c.causal_outcomes) in block
    assert "registry NODES, not KPIs" in block
    assert "NO time, region or segment dimension" in block


async def test_render_roster_never_block_and_letters():
    c = await make_catalog()
    block = cat.render_catalog_block(c)
    assert "The E2I system has" in block
    assert "NEVER PROPOSE" in block
    assert "model retraining" in block
    assert "per-HCP or per-patient predictions" in block
    for letter in "ABCDEFGH":
        assert f"\n{letter}. " in block or block.startswith(f"{letter}. "), letter
    section_lines = [line for line in block.splitlines() if _re.match(r"^[A-Z]\. ", line)]
    assert len(section_lines) == 8, section_lines


async def test_workstream_order_covers_every_workstream_and_kpi():
    from src.kpi.models import Workstream

    assert {w for w, _ in cat._WORKSTREAM_ORDER} == set(Workstream)
    c = await make_catalog()
    block = cat.render_catalog_block(c)
    for entry in c.kpis:
        assert entry.name in block, entry.id


async def test_render_degraded_fallbacks_invent_nothing():
    c = await make_catalog(coverage=_boom, outcomes=_boom)
    block = cat.render_catalog_block(c)
    assert "coverage list is unavailable" in block
    assert "outcome list is unavailable" in block
    assert "persistent_180d" not in block
    # the Rx-volume trends are code-derived and stay offered
    assert c.kpi_name("WS3-BI-005") in block


def test_axis_vocabulary_matches_kpi_calculate_tool():
    """The axis RULES are prose; pin their vocabulary to the tool's parameters
    so the prompt can never name an axis the tool does not accept."""
    import inspect

    from src.api.routes.chatbot_tools import kpi_calculate_tool

    # kpi_calculate_tool is a LangChain StructuredTool (``@tool(...)``-wrapped);
    # ``.coroutine`` is the original async function inspect.signature needs.
    params = set(inspect.signature(kpi_calculate_tool.coroutine).parameters)
    for axis in cat.AXIS_PARAMETER_NAMES:
        assert axis in params, axis
        assert axis in cat.AXIS_RULES, axis


# =============================================================================
# ROUTE HINTS
# =============================================================================


def test_route_hints_are_normalized_paths_with_a_catalog_letter():
    assert "/" in cat.ROUTE_HINTS
    for path, hint in cat.ROUTE_HINTS.items():
        assert path.startswith("/"), path
        assert path == "/" or not path.endswith("/"), path
        assert hint.strip() == hint and hint.endswith("."), path
        # every hint tells the model which catalog letters fit: "(A/B)", "(C)"
        assert _re.search(r"\([A-H](?:/[A-H])*\)", hint), path


def test_route_hint_lookup_tolerates_query_and_trailing_slash():
    expected = cat.ROUTE_HINTS["/kpi-dictionary"]
    assert cat.route_hint("/kpi-dictionary") == expected
    assert cat.route_hint("/kpi-dictionary/") == expected
    assert cat.route_hint("/kpi-dictionary?tab=ws3") == expected
    assert cat.route_hint("/kpi-dictionary/?tab=ws3#top") == expected
    assert cat.route_hint("/KPI-Dictionary") == expected
    assert cat.route_hint("/") == cat.ROUTE_HINTS["/"]


def test_route_hint_unknown_or_missing_page_is_empty():
    assert cat.route_hint("/nope") == ""
    assert cat.route_hint(None) == ""
    assert cat.route_hint("") == ""


_AUTH_ROUTES = {"/login", "/signup", "/forgot-password", "/reset-password", "/accept-invite"}


def test_route_hints_cover_every_non_auth_frontend_route():
    """A renamed or added frontend route must show up here, not as a silent fallback."""
    from pathlib import Path

    routes_tsx = Path(__file__).resolve().parents[2] / "frontend" / "src" / "router" / "routes.tsx"
    paths = set(_re.findall(r"path:\s*['\"](/[^'\"]*)['\"]", routes_tsx.read_text()))
    assert paths, "no route paths parsed from routes.tsx"
    assert paths - _AUTH_ROUTES == set(cat.ROUTE_HINTS)


# =============================================================================
# VALIDATOR
# =============================================================================


@_dataclass
class _Pill:
    title: str
    message: str


# Live pills graded NO on 2026-09-05, by rule family.
DROP_FIXTURES = [
    (
        "gap_recompute",
        "Market share gap trend",
        "Can you chart how Kisqali's market share gap in the midwest has evolved over the past 12 months?",
    ),
    (
        "shap_or_feature_importance",
        "Champion cohort vs. all HCPs",
        "How do feature importances differ between the fabhalta_hcp_adoption_champion cohort and all Fabhalta-prescribing HCPs?",
    ),
    (
        "individual_prediction",
        "Persistence trend by baseline UAS7",
        "Can you chart the predicted 180-day persistence probability for Remibrutinib across baseline UAS7 severity tiers?",
    ),
    (
        "individual_prediction",
        "Persistence by IgE tier",
        "What is the distribution of predicted 180-day persistence probability for Remibrutinib across IgE tiers?",
    ),
    (
        "individual_prediction",
        "Model performance & calibration",
        "What is the validation accuracy of the patient_persistence model for Remibrutinib, and how reliable is the 61.3% mean predicted probability?",
    ),
    (
        "territory_detail",
        "Why T-114 gained the most",
        "What are the key drivers behind the +6 field force increase recommended for territory T-114 in Fabhalta's optimization?",
    ),
    (
        "territory_detail",
        "Impact of T-072 reduction",
        "What is the expected TRx impact on Fabhalta if we reduce field force in territory T-072 by 4 as suggested?",
    ),
    (
        "uplift_by_segment",
        "Why naive > experienced?",
        "Run a causal analysis to identify the drivers behind biologic-naive patients showing +0.16 CATE versus +0.07 for biologic-experienced on Remibrutinib.",
    ),
    (
        "uplift_by_segment",
        "Validate persistence model",
        "Can we run a sensitivity analysis on the uas7_baseline -> persistent_180d treatment effect for Remibrutinib to test robustness across patient subgroups?",
    ),
    (
        "off_platform_action",
        "Email the summary",
        "Email this TRx summary for Kisqali to the brand team.",
    ),
    (
        "competitor_data",
        "Competitor share",
        "What is Kisqali's TRx versus competitors in the Northeast?",
    ),
    # outcome-as-KPI: the residual family the prototype still produced
    (
        "outcome_as_kpi:persistent_180d",
        "Persistence by region",
        "Chart the persistent_180d rate for Remibrutinib by census region.",
    ),
    (
        "outcome_as_kpi:adherent_180d",
        "Adherence trend",
        "What is the adherent 180d trend for Kisqali over the last 6 months?",
    ),
    (
        "outcome_as_kpi:discontinued_180d",
        "Discontinuation level",
        "What is the discontinued_180d percentage for Fabhalta?",
    ),
    (
        "uplift_by_segment",
        "CATE by tier",
        "How does the Conditional ATE (CATE) differ across severity tiers for Remibrutinib?",
    ),
    (
        "uplift_by_segment",
        "CATE trend",
        "Chart the Conditional ATE (CATE) trend over time for high-severity patients.",
    ),
    (
        "shap_or_feature_importance",
        "Recompute on-screen SHAP",
        "Recompute the on-screen SHAP values for Kisqali by region.",
    ),
    (
        "uplift_by_segment",
        "On-screen CATE trend",
        "Chart the trend of the on-screen CATE for high-severity Remibrutinib patients.",
    ),
    (
        "gap_recompute",
        "Extend on-screen gap",
        "Extend the on-screen gap chart to the west region for Kisqali.",
    ),
    (
        "uplift_by_segment",
        "Monthly Conditional ATE",
        "Show the monthly Conditional ATE (CATE) for Remibrutinib.",
    ),
    (
        "shap_or_feature_importance",
        "Why on-screen SHAP",
        "Why is the on-screen SHAP value for age so high for Fabhalta?",
    ),
    (
        "uplift_by_segment",
        "Explain displayed CATE",
        "Explain why the displayed CATE for high-severity patients is negative.",
    ),
    (
        "gap_recompute",
        "On-screen gap history",
        "What is the on-screen gap for Kisqali over the past 6 months?",
    ),
    (
        "uplift_by_segment",
        "On-screen CATE by IgE",
        "Break the on-screen CATE down by IgE level for Remibrutinib.",
    ),
    (
        "shap_or_feature_importance",
        "Visible SHAP since",
        "Has the visible SHAP feature ranking changed since last month?",
    ),
    (
        "shap_or_feature_importance",
        "Drivers of on-screen SHAP",
        "What are the drivers behind the on-screen SHAP ranking for Fabhalta?",
    ),
    (
        "individual_prediction",
        "HCP-level probability",
        "What is the predicted probability for HCP 12345?",
    ),
    (
        "individual_prediction",
        "Per-HCP by specialty",
        "List each HCP's predicted probability by specialty for Kisqali.",
    ),
    (
        "individual_prediction",
        "Top HCPs by region",
        "Show the top 20 HCPs by predicted 90-day adoption probability by region for Kisqali.",
    ),
    (
        "individual_prediction",
        "Which HCPs by specialty",
        "Which HCPs have the highest predicted probability by specialty for Kisqali?",
    ),
    (
        "individual_prediction",
        "Ranked HCPs per region",
        "List the HCPs with a predicted probability above 0.8 per region.",
    ),
    (
        "competitor_data",
        "Share vs competitors",
        "Compare Kisqali's TRx share against competitors.",
    ),
    (
        "competitor_data",
        "Volume vs competitor brands",
        "How does Kisqali's NBRx volume compare versus competitor brands?",
    ),
    (
        "competitor_data",
        "Perform vs competitors",
        "How does Fabhalta perform against competitors in PNH?",
    ),
    (
        "competitor_data",
        "Kisqali comparison",
        "How does Kisqali compare against competitors on NBRx?",
    ),
    (
        "competitor_data",
        "Scripts vs competitors",
        "Rank Fabhalta vs competitors by total scripts.",
    ),
    (
        "competitor_data",
        "Adoption vs competitors",
        "How does Kisqali's adoption compare versus competitors?",
    ),
    (
        "competitor_data",
        "Share vs the competition",
        "What is Kisqali's market share versus the competition?",
    ),
    (
        "competitor_data",
        "Competition's share",
        "What is the competition's share of PNH prescriptions?",
    ),
    (
        "individual_prediction",
        "Shown HCPs by HCP specialty",
        "Show the predicted probabilities for the HCPs shown, by HCP specialty.",
    ),
    (
        "shap_or_feature_importance",
        "Top model features",
        "What are the top 5 features driving Kisqali adoption predictions?",
    ),
    (
        "competitor_data",
        "Starts vs competitors",
        "How does Kisqali compare against competitors on patient starts?",
    ),
    (
        "competitor_data",
        "Initiations vs the competition",
        "Compare Remibrutinib's treatment initiations versus the competition.",
    ),
    (
        "shap_or_feature_importance",
        "E2I model features",
        "What are the top features of the E2I model?",
    ),
]

# Pills the assistant CAN answer; every one must survive.
KEEP_FIXTURES = [
    (
        "Persistence drivers",
        "What drives persistent_180d for Remibrutinib, and how confident are those paths?",
    ),
    ("Kisqali TRx trend", "Show me the month-over-month trend for Kisqali total TRx."),
    ("TRx by severity", "Chart Fabhalta's TRx trend by severity tier."),
    (
        "Midwest conversion",
        "What is Kisqali's conversion rate in the Midwest over the last 3 months?",
    ),
    (
        "Likely specialties",
        "Which HCP specialties are most likely to increase Fabhalta prescriptions?",
    ),
    ("ROI trend", "Chart the ROI trend for Remibrutinib."),
    ("Action rate uplift", "What is the action rate uplift for Kisqali?"),
    ("TRx volume drivers", "What are the causal drivers of trx_volume for Kisqali?"),
    ("Active agents", "Which agents are active right now and what are they working on?"),
    (
        "Competitive landscape",
        "Give me the competitive landscape context for Fabhalta's PNH indication.",
    ),
    (
        "Effect comparison",
        "How does the nba_trigger_accepted -> persistent_180d effect for Remibrutinib compare in confidence to the uas7_baseline path?",
    ),
    ("Regional TRx", "What is Fabhalta's TRx by census region?"),
    ("Consistency gap trend", "Chart the Geographic Consistency Gap trend for Kisqali."),
    (
        "Monthly consistency gap",
        "Show the monthly Geographic Consistency Gap for Fabhalta over the last 12 months.",
    ),
    ("SHAP coverage", "What is the current SHAP Coverage for the Kisqali model?"),
    ("Conditional ATE", "What is the Conditional ATE (CATE) for Kisqali sample drops?"),
    (
        "Specialty uptake",
        "Which specific HCP specialties are most likely to increase Kisqali prescriptions?",
    ),
    (
        "On-screen SHAP rank",
        "Which of the on-screen SHAP features ranks highest for Fabhalta?",
    ),
    (
        "On-screen SHAP by value",
        "Rank the on-screen SHAP features for Fabhalta by mean |SHAP|.",
    ),
    (
        "On-screen CATE compare",
        "Which of the on-screen segments has the largest CATE for Remibrutinib?",
    ),
    (
        "On-screen gap read",
        "Which of the on-screen gap bars is largest for Kisqali on the chart?",
    ),
    (
        "Chart Conditional ATE",
        "Chart the Conditional ATE (CATE) for Kisqali sample drops.",
    ),
    (
        "Explain on-screen SHAP",
        "Explain what the on-screen SHAP chart shows for Fabhalta.",
    ),
    (
        "Likelihood by specialty",
        "What is the mean predicted probability by specialty for Kisqali?",
    ),
    (
        "Top specialty likelihood",
        "Which specialty has the highest mean predicted probability for Kisqali?",
    ),
    (
        "Mean likelihood for HCPs",
        "What is the mean predicted probability for HCPs by specialty for Kisqali?",
    ),
    (
        "Main on-screen driver",
        "Which on-screen SHAP feature is the main driver of Fabhalta adoption?",
    ),
    (
        "HCP specialties likelihood",
        "Which HCP specialties have the highest mean predicted probability for Kisqali?",
    ),
    (
        "Likelihood by HCP specialty",
        "What is the mean predicted probability by HCP specialty for Kisqali?",
    ),
    (
        "Competitor context",
        "How does Fabhalta compare versus competitors for its PNH indication?",
    ),
    (
        "MoA vs competition",
        "How does Fabhalta compare versus the competition on mechanism of action?",
    ),
    (
        "Differentiators vs competitors",
        "What are Fabhalta's differentiators against competitors in PNH?",
    ),
    (
        "Platform features",
        "What are the top features of the E2I platform?",
    ),
    (
        "Dashboard features",
        "What are the top 3 features of this dashboard?",
    ),
    (
        "Trial endpoint vs competitors",
        "How does Fabhalta compare versus competitors on hemoglobin response in trials?",
    ),
    (
        "E2I features",
        "What are the top features of E2I?",
    ),
]


async def test_journey_outcomes_exclude_kpi_named_outcomes():
    c = await make_catalog()
    journey = set(cat.journey_outcomes(c))
    assert {
        "persistent_180d",
        "discontinued_180d",
        "adherent_180d",
        "low_gap_180d",
        "adopted",
    } <= journey
    # KPI-named outcomes (a trend of ROI or TRx volume IS answerable) stay out
    for kpi_like in ("roi", "trx_volume", "nrx_volume", "nbrx_volume", "trx_market_share"):
        assert kpi_like not in journey, kpi_like


async def test_treatment_initiated_is_left_to_the_prompt():
    """The KPI recognizer reads 'treatment initiated' as a causal-metric KPI
    mention (CM-001), so the outcome rule does not fire on it; the prompt's
    section C carries that case. Pinned so a recognizer change is visible."""
    c = await make_catalog()
    assert "treatment_initiated" not in cat.journey_outcomes(c)
    kept, dropped = cat.filter_unsupported_pills(
        [
            _Pill(
                "LoT depth",
                "What is the treatment_initiated conversion rate for Fabhalta in line-of-therapy 0?",
            )
        ],
        c,
    )
    assert len(kept) == 1 and dropped == []


@pytest.mark.parametrize("rule,title,message", DROP_FIXTURES)
async def test_known_unsupported_pills_are_dropped(rule, title, message):
    c = await make_catalog()
    kept, dropped = cat.filter_unsupported_pills([_Pill(title, message)], c)
    assert kept == []
    assert [r for _, r in dropped] == [rule]


@pytest.mark.parametrize("title,message", KEEP_FIXTURES)
async def test_supported_pills_are_kept(title, message):
    c = await make_catalog()
    kept, dropped = cat.filter_unsupported_pills([_Pill(title, message)], c)
    assert dropped == []
    assert [p.title for p in kept] == [title]


async def test_filter_preserves_order_and_returns_rules():
    c = await make_catalog()
    pills = [
        _Pill("keep-1", "Chart the TRx trend for Kisqali."),
        _Pill("drop", "Which SHAP features matter most for Kisqali adoption?"),
        _Pill("keep-2", "What drives adopted for Kisqali?"),
    ]
    kept, dropped = cat.filter_unsupported_pills(pills, c)
    assert [p.title for p in kept] == ["keep-1", "keep-2"]
    assert [(p.title, r) for p, r in dropped] == [("drop", "shap_or_feature_importance")]


async def test_off_platform_rule_wins_over_outcome_rule():
    """Rules are checked in tuple order; the first off-platform match names the drop."""
    c = await make_catalog()
    kept, dropped = cat.filter_unsupported_pills(
        [
            _Pill(
                "Territory persistence",
                "Chart the predicted persistent_180d probability by territory.",
            )
        ],
        c,
    )
    assert kept == []
    assert [r for _, r in dropped] == ["territory_detail"]


async def test_empty_outcomes_disable_outcome_rule():
    """A degraded catalog with no outcomes leaves the outcome rule inert; off-platform rules still apply."""
    c = await make_catalog(outcomes=_empty)
    assert cat.journey_outcomes(c) == ()
    kept, dropped = cat.filter_unsupported_pills(
        [
            _Pill(
                "Persistence by region",
                "Chart the persistent_180d rate for Remibrutinib by census region.",
            )
        ],
        c,
    )
    assert len(kept) == 1 and dropped == []


# =============================================================================
# CACHE
# =============================================================================


class _Counting:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    async def __call__(self):
        self.calls += 1
        return await self.fn()


async def test_cache_builds_once_within_ttl():
    cov, out = _Counting(_coverage), _Counting(_outcomes)
    cache = cat._CatalogCache()
    first = await cache.get(now=1000.0, coverage_loader=cov, outcomes_loader=out)
    second = await cache.get(
        now=1000.0 + cat.CATALOG_TTL_SECONDS - 1, coverage_loader=cov, outcomes_loader=out
    )
    assert second is first
    assert (cov.calls, out.calls) == (1, 1)


async def test_cache_rebuilds_after_ttl():
    cov, out = _Counting(_coverage), _Counting(_outcomes)
    cache = cat._CatalogCache()
    first = await cache.get(now=1000.0, coverage_loader=cov, outcomes_loader=out)
    second = await cache.get(
        now=1000.0 + cat.CATALOG_TTL_SECONDS + 1, coverage_loader=cov, outcomes_loader=out
    )
    assert second is not first
    assert (cov.calls, out.calls) == (2, 2)


async def test_degraded_catalog_retries_sooner_and_heals():
    cache = cat._CatalogCache()
    broken = await cache.get(now=0.0, coverage_loader=_boom, outcomes_loader=_boom)
    assert broken.degraded
    # still cached inside the short TTL
    same = await cache.get(
        now=cat.DEGRADED_TTL_SECONDS - 1, coverage_loader=_coverage, outcomes_loader=_outcomes
    )
    assert same is broken
    healed = await cache.get(
        now=cat.DEGRADED_TTL_SECONDS + 1, coverage_loader=_coverage, outcomes_loader=_outcomes
    )
    assert healed.degraded == ()
    assert healed.causal_outcomes == tuple(sorted(set(OUTCOMES)))


async def test_refresh_failure_keeps_last_good_fields():
    cache = cat._CatalogCache()
    good = await cache.get(now=0.0, coverage_loader=_coverage, outcomes_loader=_outcomes)
    after = await cache.get(
        now=cat.CATALOG_TTL_SECONDS + 1, coverage_loader=_boom, outcomes_loader=_boom
    )
    assert after is not good
    assert after.causal_outcomes == good.causal_outcomes
    assert after.trend_kpi_ids == good.trend_kpi_ids
    assert after.per_brand_only_trend_ids == good.per_brand_only_trend_ids
    # the failure is recorded (so the cache retries in 60 s and the outage is visible) ...
    assert after.degraded == ("trend_coverage", "causal_outcomes")
    # ... while the prompt still shows the carried-forward lists, not the fallbacks
    block = cat.render_catalog_block(after)
    assert "unavailable right now" not in block
    assert "coverage list is unavailable" not in block
    for outcome in good.causal_outcomes:
        assert outcome in block
    # and the degraded TTL applies: a good DB heals it after 60 s, not 10 min
    healed = await cache.get(
        now=cat.CATALOG_TTL_SECONDS + 1 + cat.DEGRADED_TTL_SECONDS + 1,
        coverage_loader=_coverage,
        outcomes_loader=_outcomes,
    )
    assert healed.degraded == ()
    assert healed is not after


async def test_module_level_accessor_and_reset(monkeypatch, request):
    request.addfinalizer(cat.reset_capability_catalog_cache)
    calls = {"n": 0}
    real_build = cat.build_capability_catalog

    async def fake_build(**kwargs):
        calls["n"] += 1
        return await real_build(coverage_loader=_coverage, outcomes_loader=_outcomes)

    monkeypatch.setattr(cat, "build_capability_catalog", fake_build)
    cat.reset_capability_catalog_cache()
    a = await cat.get_capability_catalog()
    b = await cat.get_capability_catalog()
    assert a is b and calls["n"] == 1
    cat.reset_capability_catalog_cache()
    c = await cat.get_capability_catalog()
    assert c is not a and calls["n"] == 2
    cat.reset_capability_catalog_cache()


@pytest.mark.parametrize(
    "fresh_kind, prev_kind, expect_degraded, expect_trends, expect_outcomes",
    [
        # (fresh, previous) -> this refresh's failed fields, trend count, outcome count
        # "full" = the healthy catalog's own counts, 0 = the field stayed empty
        ("trend_bad", None, ("trend_coverage",), 0, "full"),
        ("trend_bad", "outcomes_bad", ("trend_coverage",), "full", "full"),
        ("outcomes_bad", "trend_bad", ("causal_outcomes",), "full", "full"),
        ("trend_bad", "both_bad", ("trend_coverage",), 0, "full"),
        ("both_bad", "healthy", ("trend_coverage", "causal_outcomes"), "full", "full"),
        ("both_bad", "trend_bad", ("trend_coverage", "causal_outcomes"), 0, "full"),
        ("healthy", "both_bad", (), "full", "full"),
    ],
)
async def test_keep_last_good_fields_partial_cases(
    fresh_kind, prev_kind, expect_degraded, expect_trends, expect_outcomes
):
    """A degraded refresh carries forward the fields the PREVIOUS catalog had
    good; ``degraded`` always records the fields that failed in THIS refresh."""

    async def build(kind):
        if kind == "healthy":
            return await make_catalog()
        if kind == "trend_bad":
            return await make_catalog(coverage=_empty)
        if kind == "outcomes_bad":
            return await make_catalog(outcomes=_empty)
        return await make_catalog(coverage=_empty, outcomes=_empty)

    healthy = await make_catalog()
    full_trends, full_outcomes = len(healthy.trend_kpi_ids), len(healthy.causal_outcomes)
    fresh = await build(fresh_kind)
    prev = await build(prev_kind) if prev_kind else None

    merged = cat._keep_last_good_fields(fresh, prev)

    assert merged.degraded == expect_degraded
    assert len(merged.trend_kpi_ids) == (full_trends if expect_trends == "full" else 0)
    assert len(merged.causal_outcomes) == (full_outcomes if expect_outcomes == "full" else 0)


async def test_concurrent_cold_gets_build_once():
    """Single-flight: concurrent first callers share ONE build - no DB herd, and a
    slower degraded build can never overwrite a faster good one."""
    calls = {"n": 0}

    async def slow_coverage():
        calls["n"] += 1
        await asyncio.sleep(0.01)
        return await _coverage()

    c = cat._CatalogCache()
    results = await asyncio.gather(
        *(c.get(coverage_loader=slow_coverage, outcomes_loader=_outcomes) for _ in range(5))
    )
    assert calls["n"] == 1
    assert all(r is results[0] for r in results)
    assert c._inflight is None


async def test_failed_build_propagates_and_clears_inflight(monkeypatch):
    c = cat._CatalogCache()

    async def boom_build(**kwargs):
        raise RuntimeError("registry broke")

    monkeypatch.setattr(cat, "build_capability_catalog", boom_build)
    with pytest.raises(RuntimeError):
        await c.get()
    assert c._inflight is None
    monkeypatch.undo()
    good = await c.get(coverage_loader=_coverage, outcomes_loader=_outcomes)
    assert good.degraded == ()


async def test_reset_mid_flight_discards_the_stale_build():
    """A build orphaned by reset() still serves its own waiters but must not
    overwrite the cache or clear a newer build's future."""
    c = cat._CatalogCache()

    async def slow_coverage():
        await asyncio.sleep(0.02)
        return await _coverage()

    first = asyncio.ensure_future(c.get(coverage_loader=slow_coverage, outcomes_loader=_outcomes))
    await asyncio.sleep(0)  # the first build is now in flight
    c.reset()
    second = await c.get(coverage_loader=_coverage, outcomes_loader=_outcomes)
    stale = await first
    assert stale is not second
    assert c._catalog is second
    assert c._inflight is None


async def test_eager_task_factory_publishes_and_clears():
    """Under asyncio.eager_task_factory a build whose loaders never suspend
    finishes inside ensure_future(); it must still publish and clear so an
    expired get() rebuilds instead of serving the first build forever."""
    cov, out = _Counting(_coverage), _Counting(_outcomes)
    c = cat._CatalogCache()
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        first = await c.get(now=1000.0, coverage_loader=cov, outcomes_loader=out)
        assert c._catalog is first
        assert c._inflight is None
        second = await c.get(
            now=1000.0 + cat.CATALOG_TTL_SECONDS + 1, coverage_loader=cov, outcomes_loader=out
        )
    finally:
        loop.set_task_factory(None)
    assert second is not first
    assert c._catalog is second
    assert (cov.calls, out.calls) == (2, 2)
