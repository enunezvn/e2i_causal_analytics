"""#1874: deterministic markdown flattener for LM-generated insight prose.

The DSPy insight signatures give the LM no output-format constraint, so it can
emit markdown (``**bold**``, backticks, heading/list markers) that
StrategicInsightCard renders literally (the card is plain text by design).
``flatten_markdown`` is the universal server-side guarantee, applied at the
``_finalize`` seam so STALE Redis-cached payloads are cleaned too.
"""

import importlib

import pytest

from src.insights.common import flatten_markdown


def test_none_passes_through():
    assert flatten_markdown(None) is None


def test_empty_string_passes_through():
    assert flatten_markdown("") == ""


def test_plain_prose_is_untouched():
    s = (
        "For Kisqali / patient, copay_card->adherence_180d: ATE +0.043 "
        "[+0.020, +0.066] excludes 0. Gate distribution: proceed=1."
    )
    assert flatten_markdown(s) == s


def test_bold_asterisks_stripped():
    assert (
        flatten_markdown("**Prioritize** copay_card outreach") == "Prioritize copay_card outreach"
    )


def test_bold_double_underscores_stripped():
    assert flatten_markdown("__Prioritize__ field triggers") == "Prioritize field triggers"


def test_single_asterisk_emphasis_stripped_when_boundary_sane():
    assert flatten_markdown("a *robust* effect") == "a robust effect"


def test_multiplication_and_stray_asterisks_kept():
    assert flatten_markdown("projected 2 * 3 = 6 uplift") == "projected 2 * 3 = 6 uplift"
    assert flatten_markdown("significant at p<0.05*") == "significant at p<0.05*"


def test_snake_case_identifiers_survive_byte_identical():
    s = "acceptance_status -> conversion_flag gates persistent_180d"
    assert flatten_markdown(s) == s


def test_single_underscore_emphasis_is_never_touched():
    # Single underscores are indistinguishable from identifier underscores at
    # this layer — the flattener must NEVER rewrite them.
    s = "_emphasis_ stays as authored"
    assert flatten_markdown(s) == s


def test_inline_backticks_stripped_keeping_inner_text():
    assert (
        flatten_markdown("`copay_card` drives `adherence_180d`")
        == "copay_card drives adherence_180d"
    )


def test_fenced_code_markers_removed():
    assert flatten_markdown("Before\n```python\nx = 1\n```\nAfter") == "Before\nx = 1\nAfter"


def test_leading_heading_markers_stripped_per_line():
    assert flatten_markdown("## Key actions\nDo X") == "Key actions\nDo X"


def test_mid_line_hash_kept():
    s = "segment #4 leads on TRx"
    assert flatten_markdown(s) == s


def test_bullet_markers_normalized_to_bullet_glyph():
    assert flatten_markdown("- item one\n* item two") == "• item one\n• item two"


def test_numbered_list_markers_kept():
    # `1. ` enumeration reads as prose under the card's pre-line rendering.
    s = "1. First action\n2. Second action"
    assert flatten_markdown(s) == s


def test_leading_negative_number_untouched():
    s = "-0.02 is the lower CI bound"
    assert flatten_markdown(s) == s


def test_idempotent_on_its_own_output():
    marked = "## Plan\n- **Prioritize** `copay_card` for *high-decile* HCPs"
    once = flatten_markdown(marked)
    assert flatten_markdown(once) == once


def test_regression_1874_observed_leaderboard_string():
    # The user-observed causal-discovery read (#1874), pinned end to end.
    assert (
        flatten_markdown("1. **`acceptance_status -> conversion_flag`** drives conversions")
        == "1. acceptance_status -> conversion_flag drives conversions"
    )


# ---- Signature-side constraint (defense at the source) --------------------------
# Every insight signature that feeds the StrategicInsightCard surfaces must tell
# the LM to emit plain prose. The flattener remains the universal guarantee;
# this pins the sweep so a future signature edit can't silently drop the
# instruction. DagReviewAssessmentSignature (expert-review DAG verdicts, a
# different surface) is deliberately not in this list.
_SIGNATURE_CLASSES = [
    ("src.insights.causal_discovery", "CausalDiscoveryInsightSignature"),
    ("src.insights.clinical_narrative", "ClinicalNarrativeSignature"),
    ("src.insights.digital_twin", "DigitalTwinInsightSignature"),
    ("src.insights.executive_brief", "ExecutiveBriefInsightSignature"),
    ("src.insights.experiments", "ExperimentsInsightSignature"),
    ("src.insights.feedback_learning", "FeedbackLearningInsightSignature"),
    ("src.insights.home_kpi", "HomeKpiInsightSignature"),
    ("src.insights.hte", "HTEInsightSignature"),
    ("src.insights.knowledge_graph", "KnowledgeGraphInsightSignature"),
    ("src.insights.model_performance", "ModelPerformanceInsightSignature"),
    ("src.insights.predictive_cohort", "PredictiveCohortInsightSignature"),
    ("src.insights.predictive_cohort", "PredictiveWhatIfInsightSignature"),
    ("src.insights.resource_optimization", "ResourceOptimizationInsightSignature"),
    ("src.insights.treatment_effect", "TreatmentEffectInsightSignature"),
]


@pytest.mark.parametrize("mod_name,cls_name", _SIGNATURE_CLASSES)
def test_insight_signatures_carry_the_plain_prose_constraint(mod_name, cls_name):
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    if cls is None:
        pytest.skip("dspy unavailable in this environment")
    doc = (cls.__doc__ or "").lower()
    assert "plain prose" in doc and "markdown" in doc, (
        f"{cls_name} docstring must instruct the LM to write plain prose "
        "with no markdown syntax (#1874)"
    )
