from src.insights.feedback_learning import build_grounding, generate_insight


def _grounding(**overrides):
    defaults = dict(
        cycles_24h=4,
        last_cycle_at="2026-07-07T00:00:00+00:00",
        thumbs_7d=3,
        signals_7d=36,
        avg_reward_7d=0.87,
        patterns=[
            {"severity": "high", "pattern_type": "accuracy_issue", "description": "low reward"},
            {"severity": "medium", "pattern_type": "latency_issue", "description": "slow hops"},
        ],
        updates=[
            {"status": "applied", "update_type": "prompt_refinement"},
            {"status": "proposed", "update_type": "prompt_refinement"},
        ],
        low_reward_agents=[("cognitive_investigator", 0.42)],
    )
    defaults.update(overrides)
    return build_grounding(**defaults)


def test_build_grounding_summaries_and_chips():
    g = _grounding()
    assert "4 learning cycle(s)" in g["activity_summary"]
    assert "39 items" in g["activity_summary"]  # 3 thumbs + 36 signals
    assert "avg reward 0.87" in g["activity_summary"]
    assert "2 pattern(s)" in g["patterns_summary"]
    assert "1 high" in g["patterns_summary"]
    assert "1 applied, 1 pending review" in g["updates_summary"]
    assert "cognitive_investigator (0.42)" in g["signal_quality_summary"]
    labels = {c["label"]: c["value"] for c in g["grounding"]}
    assert labels["Cycles 24h"] == "4"
    assert labels["Feedback 7d"] == "39"
    assert labels["Patterns"] == "2"
    assert labels["Avg reward"] == "0.87"


def test_build_grounding_honest_empty_state():
    g = _grounding(
        cycles_24h=0,
        last_cycle_at=None,
        thumbs_7d=0,
        signals_7d=0,
        avg_reward_7d=None,
        patterns=[],
        updates=[],
        low_reward_agents=[],
    )
    assert "last cycle never" in g["activity_summary"]
    assert g["patterns_summary"] == "no patterns detected yet"
    assert g["updates_summary"] == "no knowledge updates proposed yet"
    assert g["signal_quality_summary"] == "no reward signals in the window"
    labels = {c["label"] for c in g["grounding"]}
    assert "Avg reward" not in labels  # never a fabricated average


def test_generate_insight_fallback_grounded():
    g = _grounding()
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "4 learning cycle(s)" in out["insight"]
    assert "2 pattern(s)" in out["insight"]
    assert out["grounding"] == g["grounding"]
