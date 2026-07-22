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
* Brand scoping (frontend review 2026-07-22): under a selected brand, another
  brand's hard-bound KPIs are EXCLUDED from the grounding (rows, coverage
  denominator, chips) — the dashboard grid applies the identical filter, so
  the narrative can never cite a card that is not on screen. Supersedes the
  "[sibling brand: X]" tagging.
* Lag-leak guard (same review): the claims adjudication/runout constraint is
  stated once, in structural_considerations only. Channel-1 leakage earns one
  fresh-sample retry; a persistent leak is SERVED (logged) — a repeated caveat
  is a style defect, never worth the factual fallback.
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


def test_thousands_separator_forms_are_equivalent():
    """_fmt_value renders volumes with commas ('12,345'); the LM may cite
    either '12,345' or '12345' — both must pass, and neither may leak the
    fragment tokens ('12'/'345') as independently-allowed digits."""
    g = _g()
    g["kpi_table"] = "Total Prescriptions (TRx) [ws3_business]: 12,345 (warning)"
    for quote in ("12,345", "12345"):
        ok = _pred(f"TRx volume stands at {quote} scripts.")
        with patch.object(home_kpi, "run_signature", return_value=ok):
            out = home_kpi.generate_insight(g)
        assert out["is_fallback"] is False, quote
    fragment = _pred("Volume rose by 345 scripts.")
    with patch.object(home_kpi, "run_signature", return_value=fragment):
        out = home_kpi.generate_insight(g)
    assert out["is_fallback"] is True


def test_sign_stripped_negative_is_rejected():
    """Codex H1: a negative grounded value quoted WITHOUT its sign flips a
    decline into a gain — the wrong-direction narrative this surface exists to
    prevent. '-8.0' and '8.0' are distinct tokens."""
    g = _g()
    g["kpi_table"] = "Action Rate Uplift [ws2_triggers]: -8.0% (warning)"
    stripped = _pred("Action Rate Uplift shows an 8.0% lift.")
    with patch.object(home_kpi, "run_signature", return_value=stripped) as rs:
        out = home_kpi.generate_insight(g)
    assert out["is_fallback"] is True
    assert rs.call_count == 2


def test_verbatim_negative_and_unicode_minus_pass():
    g = _g()
    g["kpi_table"] = "Action Rate Uplift [ws2_triggers]: -8.0% (warning)"
    for quote in ("-8.0%", "−8.0%"):  # ASCII hyphen and U+2212 minus
        ok = _pred(f"Action Rate Uplift sits at {quote} — a decline needing attention.")
        with patch.object(home_kpi, "run_signature", return_value=ok):
            out = home_kpi.generate_insight(g)
        assert out["is_fallback"] is False, quote


def test_range_digits_stay_individually_quotable():
    """Deliberate stance (codex M2 declined): '1-3 months' licenses '1' and
    '3' individually — honest phrasings like 'up to 3 months' must not be
    rejected; range fidelity is the prompt's job, not the subset guard's."""
    ok = _pred("Claims lag can reach 3 months for these outcome metrics.")
    with patch.object(home_kpi, "run_signature", return_value=ok):
        out = home_kpi.generate_insight(_g())
    assert out["is_fallback"] is False


def test_lag_leak_in_channel_1_retries_once_with_fresh_sample():
    leaking = _pred(
        "Trigger Recall at 67.5% is good; the adjudication/runout lag under-counts recent windows.",
        structural="Claims adjudication lag of 1-3 months gates outcome metrics.",
    )
    clean = _pred(
        "Trigger Recall at 67.5% is good.",
        structural="Claims adjudication lag of 1-3 months gates outcome metrics.",
    )
    with patch.object(home_kpi, "run_signature", side_effect=[leaking, clean]) as rs:
        out = home_kpi.generate_insight(_g())
    assert out["is_fallback"] is False
    assert "adjudication" not in out["insight"]
    assert rs.call_count == 2
    assert rs.call_args_list[1].kwargs.get("lm_cache") is False


def test_persistent_lag_leak_is_served_not_degraded_to_fallback():
    leaking = _pred(
        "Trigger Recall at 67.5% is good; runout gates the recent windows.",
        structural="Claims adjudication lag of 1-3 months gates outcome metrics.",
    )
    with patch.object(home_kpi, "run_signature", return_value=leaking) as rs:
        out = home_kpi.generate_insight(_g())
    assert out["is_fallback"] is False  # style defect, not an ungrounded figure
    assert rs.call_count == 2
    assert "runout" in out["insight"]


def test_lag_terms_in_structural_channel_alone_are_not_a_leak():
    ok = _pred(
        "Trigger Recall at 67.5% is good; TRx Share at 25.6% needs commercial focus.",
        ["Focus call-plan coverage to lift TRx Share from 25.6%."],
        structural="A 1-3 months adjudication/runout lag gates the outcome metrics.",
    )
    with patch.object(home_kpi, "run_signature", return_value=ok) as rs:
        out = home_kpi.generate_insight(_g())
    assert out["is_fallback"] is False
    assert rs.call_count == 1


def _brand_metas():
    own = SimpleNamespace(
        id="FAB-001",
        name="Fabhalta - % PNH Tested",
        workstream=SimpleNamespace(value="brand_specific"),
        unit=None,
        value_format="percent",
        brand="Fabhalta",
        direction=None,
    )
    sibling = SimpleNamespace(
        id="KIS-001",
        name="Kisqali - Dx Adoption",
        workstream=SimpleNamespace(value="brand_specific"),
        unit=None,
        value_format="percent",
        brand="Kisqali",
        direction=None,
    )
    portfolio = SimpleNamespace(
        id="WS3-BI-001",
        name="Total TRx",
        workstream=SimpleNamespace(value="ws3_business"),
        unit=None,
        value_format=None,
        brand=None,
        direction=None,
    )
    results = [
        SimpleNamespace(kpi_id="FAB-001", value=0.63, error=None, status="good"),
        SimpleNamespace(kpi_id="KIS-001", value=0.21, error=None, status="good"),
        SimpleNamespace(kpi_id="WS3-BI-001", value=11634.0, error=None, status="informational"),
    ]
    return [own, sibling, portfolio], results


def test_build_grounding_scopes_out_other_brands_hard_bound_kpis():
    metas, results = _brand_metas()
    g = home_kpi.build_grounding("Fabhalta", None, metas, results)
    assert "Fabhalta - % PNH Tested" in g["kpi_table"]
    assert "Total TRx" in g["kpi_table"]
    # The other brand's row leaves the table, the coverage denominator, and
    # the chips together — never a tagged leftover.
    assert "Kisqali" not in g["kpi_table"]
    assert "sibling brand" not in g["kpi_table"]
    assert g["coverage"] == "2 of 2 defined KPIs computed for this scope"
    chips = {c["label"]: c["value"] for c in g["grounding"]}
    assert chips["Computed"] == "2/2"


def test_build_grounding_keeps_every_brand_first_class_under_all():
    metas, results = _brand_metas()
    g = home_kpi.build_grounding("All", None, metas, results)
    assert "Fabhalta - % PNH Tested" in g["kpi_table"]
    assert "Kisqali - Dx Adoption" in g["kpi_table"]
    assert g["coverage"] == "3 of 3 defined KPIs computed for this scope"


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
