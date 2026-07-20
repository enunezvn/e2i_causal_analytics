"""Home-KPI insight: two-channel triage wiring + fail-closed digit-subset guard
(constraint-aware insights plan §4, 2026-07-20).

* HomeKpiInsightSignature gains data_constraint_context (input) and
  structural_considerations (output — channel 2: escalation/investment
  considerations rendered separately so recommendations are not diluted).
* generate_insight enforces a DIGIT-SUBSET guard: every digit sequence in the
  LM output must literally appear in the grounding (kpi_table/scope/coverage/
  status_summary) or the constraint context. One retry with lm_cache=False,
  then the factual fallback — mirroring the exec-brief lesson (instruction-only
  proved insufficient there) without its placeholder machinery.
* build_grounding renders a [lower is better] hint from the registry direction
  field (the WS1-DQ-006 rename's second half).
"""

from types import SimpleNamespace
from unittest.mock import patch

from src.insights import home_kpi


def _g(context: str = "Constraint context line with 10.2% and 1-3 months.") -> dict:
    return {
        "scope": "Remibrutinib / All US",
        "kpi_table": "Trigger Recall [ws2_triggers]: 67.5% (good)\nTRx Share [ws3_business]: 25.6% (warning)",
        "status_summary": "warning=1, good=1",
        "coverage": "2 of 44 defined KPIs computed for this scope",
        "grounding": [{"label": "Brand", "value": "Remibrutinib"}],
        "data_constraint_context": context,
    }


def _pred(interpretation, takeaways=None, structural=""):
    return SimpleNamespace(
        interpretation=interpretation,
        key_takeaways=takeaways or [],
        structural_considerations=structural,
    )


def test_signature_carries_the_two_new_fields():
    assert "data_constraint_context" in home_kpi.HomeKpiInsightSignature.input_fields
    assert "structural_considerations" in home_kpi.HomeKpiInsightSignature.output_fields


def test_grounded_digits_pass_and_structural_channel_flows_through():
    ok = _pred(
        "Trigger Recall at 67.5% is good; TRx Share at 25.6% needs commercial focus.",
        ["Focus call-plan coverage to lift TRx Share from 25.6%."],
        structural="Claims lag of 1-3 months gates outcome metrics.",
    )
    with patch.object(home_kpi, "run_signature", return_value=ok) as rs:
        out = home_kpi.generate_insight(_g())
    assert out["is_fallback"] is False
    assert out["structural_considerations"] == "Claims lag of 1-3 months gates outcome metrics."
    assert rs.call_count == 1


def test_invented_digit_retries_once_then_falls_back():
    bad = _pred("Recall improved 42% quarter over quarter.")
    with patch.object(home_kpi, "run_signature", return_value=bad) as rs:
        out = home_kpi.generate_insight(_g())
    assert out["is_fallback"] is True
    assert rs.call_count == 2  # one retry with a fresh sample, then fallback
    # the retry must force a fresh sample past the in-process DSPy cache
    assert rs.call_args_list[1].kwargs.get("lm_cache") is False


def test_invented_digit_in_structural_channel_is_also_guarded():
    bad = _pred(
        "Trigger Recall at 67.5% is good.",
        structural="Reducing lag would lift precision by 12 points.",
    )
    with patch.object(home_kpi, "run_signature", return_value=bad) as rs:
        out = home_kpi.generate_insight(_g())
    assert out["is_fallback"] is True
    assert rs.call_count == 2


def test_retry_that_recovers_is_served():
    bad = _pred("Recall improved 42% quarter over quarter.")
    ok = _pred("Trigger Recall at 67.5% is good.")
    with patch.object(home_kpi, "run_signature", side_effect=[bad, ok]):
        out = home_kpi.generate_insight(_g())
    assert out["is_fallback"] is False
    assert "67.5%" in out["insight"]


def test_digits_from_constraint_context_are_allowed():
    ok = _pred("The 10.2% gap reads within claims-lag expectations (1-3 months).")
    with patch.object(home_kpi, "run_signature", return_value=ok):
        out = home_kpi.generate_insight(_g())
    assert out["is_fallback"] is False


def test_build_grounding_renders_lower_is_better_hint():
    meta = SimpleNamespace(
        id="WS1-DQ-006",
        name="Geographic Consistency Gap",
        workstream=SimpleNamespace(value="ws1_data_quality"),
        unit=None,
        value_format="percent",
        brand=None,
        direction="lower_is_better",
    )
    result = SimpleNamespace(kpi_id="WS1-DQ-006", value=0.102, error=None, status="warning")
    g = home_kpi.build_grounding("Remibrutinib", None, [meta], [result])
    assert "[lower is better]" in g["kpi_table"]
