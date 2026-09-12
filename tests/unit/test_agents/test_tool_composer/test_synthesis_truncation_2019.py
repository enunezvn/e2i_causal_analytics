"""Red-first pins for #2019 — the synthesis prompt must not lose a tool's tail.

``ResponseSynthesizer._format_results`` cut every tool's serialized output to its
first 1,000 characters (``output_str[:1000] + "... (truncated)"``) — a byte cut
through whatever happened to be at offset 1,000. Tools put their headline verdict
and their disclosures AFTER the bulky arrays, so the cut removed exactly the
fields the answer must carry.

MEASURED on this branch by calling the REAL registered tool callables on real
DataFrames (nothing hand-written to look plausible); ``chars`` is
``json.dumps(result.model_dump(), indent=2, default=str)``, the synthesizer's own
serialization, and "cut" means the key never reached the synthesis LLM at all:

===============================  =======  ==========================================
tool output (the frames below)    chars    keys the 1,000-char cut removed entirely
===============================  =======  ==========================================
gap_calculator[40 territories]     1,393   top_performer, bottom_performer
gap_calculator[150 territories]    4,925   top_performer, bottom_performer
cohort_builder[800 patients]      11,003   total_evaluated, total_eligible,
                                           eligibility_rate, criteria_breakdown,
                                           execution_time_ms
risk_scorer[50 patients]           5,598   model_version, scored_at
risk_scorer[1,200 patients]      131,546   model_version, scored_at
segment_ranker[16 tiers]           1,189   recommended_targets
cate_analyzer[16 payer tiers]      3,279   excluded_segments (the #1610 disclosure),
                                           effect_by_segment
cate_analyzer[30 payer tiers]      5,723   high_responders, effect_by_segment,
                                           excluded_segments
===============================  =======  ==========================================

Under the projection every one of those top-level keys survives, in all eight
cases, inside the same bounded prompt.

Two distinct harms, both present:

1. **The disclosure is lost.** ``cate_analyzer``'s ``excluded_segments`` (#1610)
   is the LAST field of ``CATEResults``, so a frame with enough segments pushes
   every exclusion past the cut. The synthesis then states per-segment effects
   with no mention that segments were dropped for having no within-segment
   contrast — a verdict without its limitation.
2. **The verdict itself is lost.** ``gap_calculator`` declares ``gap`` first and
   ``top_performer`` / ``bottom_performer`` last, after the per-entity
   ``entity_values`` dict. At 40 territories the LLM receives 1,000 characters of
   float values and never learns which territory is top. ``cohort_builder`` is
   worse: the prompt is 1,000 characters of raw patient IDs and NONE of
   ``total_eligible`` / ``eligibility_rate`` / ``criteria_breakdown``.

The fix is a structured projection, not a bigger byte cut: scalars always
survive, disclosure containers are filled before bulky data containers, and every
trim is stated inline with a count of what was omitted.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    ExecutionStatus,
    ExecutionTrace,
    StepResult,
    SubQuestion,
    SynthesisInput,
    ToolInput,
    ToolOutput,
)
from src.agents.tool_composer.synthesizer import (
    SYNTHESIS_OUTPUT_BUDGET_CHARS,
    ResponseSynthesizer,
    project_tool_output,
)

# ---------------------------------------------------------------------------
# Real frames -> real tool outputs
# ---------------------------------------------------------------------------


def _rng() -> np.random.Generator:
    """A FRESH seeded generator per frame.

    A module-level generator would make every frame depend on how many other
    tests drew from it first, so the same test would see different data under
    ``-p no:randomly``, ``-k``, or a reordering — and an assertion about which
    segments survive the projection would pass or fail on test order alone.
    """
    return np.random.default_rng(7)


def _kisqali_frame(n_patients: int, n_segments: int) -> pd.DataFrame:
    """A Kisqali-shaped patient frame: ``copay_support -> days_to_treatment``.

    Built here, not inside a tool (the anti-mock rule forbids fabricated data in
    tool bodies, not in tests). Every third payer tier is made treated-only so
    ``cate_analyzer`` produces REAL ``excluded_segments`` entries through its
    ``no_within_segment_contrast`` branch, and the first rows carry a null tier
    so the null-key exclusion fires too.

    The treatment effect is HETEROGENEOUS by tier (``-4 + 1.5 * tier_index``), the
    thing a CATE exists to find. It also makes ``high_responders`` non-empty, so
    the assertions over it are not vacuously true on an empty list.
    """
    rng = _rng()
    labels = [f"payer_tier_{i}" for i in range(n_segments)]
    tier = rng.choice(labels, size=n_patients)
    tier_index = np.array([int(str(t).rsplit("_", 1)[1]) for t in tier])
    copay = rng.integers(0, 2, size=n_patients)
    effect = -4.0 + 1.5 * tier_index
    frame = pd.DataFrame(
        {
            "patient_id": [f"PT{i:05d}" for i in range(n_patients)],
            "payer_tier": tier,
            "copay_support": copay,
            "days_to_treatment": 30 + effect * copay + rng.normal(0, 5, size=n_patients),
            "age": rng.integers(35, 85, size=n_patients),
            "prior_lines": rng.integers(0, 4, size=n_patients),
            "discontinuation_flag": rng.integers(0, 2, size=n_patients),
        }
    )
    for i in range(0, n_segments, 3):
        frame.loc[frame["payer_tier"] == labels[i], "copay_support"] = 1
    frame.loc[frame.index[:12], "payer_tier"] = None
    return frame


def _cate_output(n_segments: int = 16, n_patients: int = 1200) -> Dict[str, Any]:
    return tr.cate_analyzer(
        treatment="copay_support",
        outcome="days_to_treatment",
        segments=["payer_tier"],
        estimation_data=_kisqali_frame(n_patients, n_segments),
    ).model_dump()


def _gap_output(n_entities: int = 40) -> Dict[str, Any]:
    frame = pd.DataFrame(
        {
            "territory": [f"T{i:03d}" for i in range(n_entities) for _ in range(4)],
            "market_share": _rng().uniform(0.1, 0.9, size=n_entities * 4),
        }
    )
    return tr.gap_calculator(
        metric="market_share", entity_type="territory", entities=[], estimation_data=frame
    ).model_dump()


def _cohort_output(n_patients: int = 800) -> Dict[str, Any]:
    return tr.cohort_builder(
        brand="Kisqali",
        inclusion_criteria=["age > 40"],
        estimation_data=_kisqali_frame(n_patients, 6),
    ).model_dump()


def _risk_output(n_patients: int = 1200) -> Dict[str, Any]:
    return tr.risk_scorer(
        entity_type="patient",
        risk_type="discontinuation",
        estimation_data=_kisqali_frame(n_patients, 6),
        id_column="patient_id",
        outcome="discontinuation_flag",
    ).model_dump()


# ---------------------------------------------------------------------------
# Prompt plumbing
# ---------------------------------------------------------------------------


def _prompt_for(outputs: Dict[str, Dict[str, Any]], llm: Any) -> str:
    """Render ``{tool_name: result_dict}`` through the real ``_format_results``."""
    sub_questions: List[SubQuestion] = []
    trace = ExecutionTrace(plan_id="plan_2019")
    for i, (tool_name, result) in enumerate(outputs.items(), start=1):
        sq_id = f"sq_{i}"
        sub_questions.append(
            SubQuestion(id=sq_id, question=f"What does {tool_name} report?", intent="CAUSAL")
        )
        trace.add_result(
            StepResult(
                step_id=f"step_{i}",
                sub_question_id=sq_id,
                tool_name=tool_name,
                input=ToolInput(tool_name=tool_name, parameters={}),
                output=ToolOutput(tool_name=tool_name, success=True, result=result),
                status=ExecutionStatus.COMPLETED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
        )
    synthesis_input = SynthesisInput(
        original_query="Which payer tiers respond to copay support?",
        decomposition=DecompositionResult(
            original_query="Which payer tiers respond to copay support?",
            sub_questions=sub_questions,
            decomposition_reasoning="test",
            timestamp=datetime.now(timezone.utc),
        ),
        execution_trace=trace,
    )
    return ResponseSynthesizer(llm_client=llm)._format_results(synthesis_input)


def _rendered_output_block(prompt: str) -> str:
    """The text the LLM sees for the single tool in a one-tool prompt."""
    return prompt.split("Output:\n", 1)[1]


# ---------------------------------------------------------------------------
# (1) the disclosure must survive
# ---------------------------------------------------------------------------


def test_every_cate_excluded_segment_reaches_the_synthesis_prompt(mock_llm_client):
    """#1610's ``excluded_segments`` is the LAST field of ``CATEResults``.

    Pre-fix the 1,000-char cut removed the key entirely at 16 payer tiers, so the
    answer could report per-segment effects while never disclosing that segments
    were dropped for having no within-segment contrast.
    """
    out = _cate_output(n_segments=16)
    excluded = out["excluded_segments"]
    assert len(excluded) >= 5, f"frame must produce several exclusions, got {len(excluded)}"

    prompt = _prompt_for({"cate_analyzer": out}, mock_llm_client)

    assert "excluded_segments" in prompt
    for entry in excluded:
        label = "null" if entry["name"] is None else str(entry["name"])
        assert label in prompt, f"excluded segment {label!r} never reached the prompt"
        assert entry["reason"] in prompt, f"exclusion reason for {label!r} was cut"


def test_cate_headline_effects_survive_alongside_the_disclosures(mock_llm_client):
    """Keeping the caveat must not cost the verdict: both reach the prompt."""
    out = _cate_output(n_segments=16)
    prompt = _prompt_for({"cate_analyzer": out}, mock_llm_client)

    assert "effect_by_segment" in prompt
    assert "high_responders" in prompt
    # Positive control: an empty list would make the loop below vacuously true.
    assert out["high_responders"], "frame must produce high responders for this pin"
    for name in out["high_responders"]:
        assert name in prompt, f"recommended segment {name!r} was cut"


# ---------------------------------------------------------------------------
# (2) the verdict must survive
# ---------------------------------------------------------------------------


def test_gap_top_and_bottom_performer_survive(mock_llm_client):
    """``top_performer``/``bottom_performer`` sit AFTER the per-entity dict.

    Measured pre-fix at 40 territories: the prompt carried 1,000 chars of
    ``entity_values`` floats and neither performer name.
    """
    out = _gap_output(n_entities=40)
    prompt = _prompt_for({"gap_calculator": out}, mock_llm_client)

    assert "top_performer" in prompt
    assert out["top_performer"] in prompt
    assert "bottom_performer" in prompt
    assert out["bottom_performer"] in prompt
    assert str(round(out["gap"], 6))[:6] in prompt or "gap" in prompt


def test_cohort_counts_survive_a_long_patient_id_list(mock_llm_client):
    """Pre-fix the whole prompt was patient IDs; every count was cut."""
    out = _cohort_output(n_patients=800)
    prompt = _prompt_for({"cohort_builder": out}, mock_llm_client)

    for key in ("total_evaluated", "total_eligible", "eligibility_rate", "criteria_breakdown"):
        assert key in prompt, f"{key} never reached the synthesis prompt"
    assert str(out["total_eligible"]) in prompt


def test_risk_scorer_metadata_survives_a_huge_score_array(mock_llm_client):
    out = _risk_output(n_patients=1200)
    prompt = _prompt_for({"risk_scorer": out}, mock_llm_client)

    assert "model_version" in prompt
    assert out["model_version"] in prompt
    assert "scored_at" in prompt


# ---------------------------------------------------------------------------
# (3) every omission is explicit, with a count
# ---------------------------------------------------------------------------


def test_a_trimmed_array_states_how_many_entries_were_omitted(mock_llm_client):
    """Never a silent cut: the prompt must name the omitted count."""
    out = _risk_output(n_patients=1200)
    total = len(out["scores"])
    prompt = _prompt_for({"risk_scorer": out}, mock_llm_client)

    block = _rendered_output_block(prompt)
    assert "_omitted" in block, "the trim is not disclosed in the prompt"
    counts = [int(m) for m in re.findall(r"_omitted[^\n]*?(\d+) of (\d+)", block)[0]]
    omitted, stated_total = counts
    assert stated_total == total, f"stated total {stated_total} != real {total}"
    assert omitted > 0


def test_nothing_is_cut_mid_json(mock_llm_client):
    """The old cut sliced through whatever sat at offset 1,000.

    Whatever the projection emits must still parse as JSON, so no consumer (and
    no model) reads a half-written value as a complete one.
    """
    for name, out in (
        ("risk_scorer", _risk_output(1200)),
        ("cate_analyzer", _cate_output(30)),
        ("cohort_builder", _cohort_output(800)),
        ("gap_calculator", _gap_output(150)),
    ):
        block = _rendered_output_block(_prompt_for({name: out}, mock_llm_client))
        payload = block.split("\n(structured summary", 1)[0].strip()
        json.loads(payload)  # pre-fix: raises on the mid-value cut


# ---------------------------------------------------------------------------
# (4) the prompt stays inside a measured budget
# ---------------------------------------------------------------------------


def test_each_tool_output_stays_within_the_budget(mock_llm_client):
    """The cap exists to bound prompt size; the fix must keep a bound.

    ``SYNTHESIS_OUTPUT_BUDGET_CHARS`` is pinned to the composer's existing
    ``_MAX_FAILURE_REASON_CHARS`` carry limit (2,000) — a tool's FAILURE reason
    already gets 2,000 characters carried into the user-visible answer, so its
    SUCCESS output getting less was the incoherent part.

    Asserted on the WHOLE rendered string, body AND trailing summary line: the
    whole string is what costs prompt budget, so accounting that excluded the
    footer would let the real cost drift ~180 chars past the cap per tool.
    """
    for name, out in (
        ("risk_scorer", _risk_output(1200)),
        ("cate_analyzer", _cate_output(30)),
        ("cohort_builder", _cohort_output(800)),
        ("gap_calculator", _gap_output(150)),
    ):
        rendered = project_tool_output(out)
        assert "structured summary" in rendered, f"{name} was expected to be projected"
        assert len(rendered) <= SYNTHESIS_OUTPUT_BUDGET_CHARS, (
            f"{name} rendered {len(rendered)} chars TOTAL, over the "
            f"{SYNTHESIS_OUTPUT_BUDGET_CHARS} budget"
        )
        # ...and that exact string is what lands in the prompt.
        block = _rendered_output_block(_prompt_for({name: out}, mock_llm_client))
        assert block.strip() == rendered.strip()


def test_the_budget_covers_the_footer_not_just_the_json_body():
    """Regression pin: the budget is accounted on the TOTAL string.

    The first cut of this fix enforced the cap on the JSON body only and then
    appended the ~180-char summary line on top, so an output reported as "within
    budget" actually handed the model 2,181 chars against a 2,000 cap. The footer
    is now sized first and charged to the budget.
    """
    out = _gap_output(150)
    rendered = project_tool_output(out)
    body, sep, tail = rendered.partition("\n(structured summary")
    assert sep, "expected the summary footer"
    footer_len = len(sep + tail)
    assert footer_len > 100, "footer is big enough that ignoring it would matter"
    assert len(body) + footer_len == len(rendered) <= SYNTHESIS_OUTPUT_BUDGET_CHARS

    # An explicitly passed budget is honoured on the total too, not just default.
    for budget in (600, 1000, 2000, 4000):
        assert len(project_tool_output(out, budget)) <= budget, f"budget={budget}"


def test_whole_prompt_is_bounded_by_steps_times_budget(mock_llm_client):
    """Six sub-questions is the decomposer's ceiling, so this is the worst case."""
    outputs = {
        "risk_scorer": _risk_output(1200),
        "cate_analyzer": _cate_output(30),
        "cohort_builder": _cohort_output(800),
        "gap_calculator": _gap_output(150),
        "segment_ranker": tr.segment_ranker(cate_results=_cate_output(30)).model_dump(),
        "propensity_estimator": tr.propensity_estimator(
            treatment="copay_support",
            covariates=["age", "prior_lines"],
            estimation_data=_kisqali_frame(400, 5),
        ).model_dump(),
    }
    prompt = _prompt_for(outputs, mock_llm_client)
    # 6 outputs x budget, plus per-step headers (sub-question, tool, status).
    ceiling = 6 * SYNTHESIS_OUTPUT_BUDGET_CHARS + 6 * 500
    assert len(prompt) <= ceiling, f"prompt is {len(prompt)} chars, ceiling {ceiling}"


# ---------------------------------------------------------------------------
# (5) no regression for outputs that already fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,builder",
    [
        ("gap_calculator", lambda: _gap_output(5)),
        (
            "sensitivity_analyzer",
            lambda: tr.sensitivity_analyzer(ate=0.12, ci_lower=0.04, ci_upper=0.20, naive_ate=0.19),
        ),
    ],
)
def test_an_output_inside_the_budget_is_rendered_byte_identically(mock_llm_client, name, builder):
    """A small output must reach the LLM exactly as before — no new markers."""
    out = builder()
    expected = json.dumps(out, indent=2, default=str)
    assert len(expected) <= SYNTHESIS_OUTPUT_BUDGET_CHARS

    block = _rendered_output_block(_prompt_for({name: out}, mock_llm_client))
    assert block.strip() == expected.strip()
    assert "_omitted" not in block
    assert "structured summary" not in block
