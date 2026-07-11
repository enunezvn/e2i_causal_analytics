"""src/insights/experiments.py — A/B portfolio strategic insight (2026-07-11).

All arithmetic (per-channel means, significance shares, ranking, null-channel
detection) is server-side in build_grounding; the LM only narrates. The
fallback must stay deterministic and never fabricate."""

from src.insights.experiments import _fallback, build_grounding


def _row(channel, effect, significant):
    return {
        "intervention_channel": channel,
        "ab_experiment_results": [
            {"effect_estimate": effect, "is_significant": significant}
        ],
    }


class TestBuildGrounding:
    def test_aggregates_per_channel_and_ranks_by_effect(self):
        g = build_grounding(
            "Fabhalta",
            [
                _row("speaker_program_invitation", 0.15, True),
                _row("speaker_program_invitation", 0.17, True),
                _row("digital_engagement", 0.004, False),
                _row("email_campaign", 0.03, False),
            ],
        )
        assert g["n_experiments"] == 4
        assert [c["channel"] for c in g["channels"]][0] == "speaker_program_invitation"
        top = g["channels"][0]
        assert top["mean_effect_pp"] == 16.0
        assert top["n"] == 2 and top["significant"] == 2
        # Human labels, not enum values, in the narration inputs
        assert "Speaker Program Invitation" in g["channel_effects"]
        assert "digital_engagement" not in g["channel_effects"]

    def test_null_channels_called_out_and_winner_scaled_per_100_hcps(self):
        g = build_grounding(
            "All",
            [
                _row("speaker_program_invitation", 0.16, True),
                _row("digital_engagement", 0.002, False),
            ],
        )
        assert "Digital Engagement" in g["highlights"]
        assert "No significant effect" in g["highlights"]
        # Value framing stays in percentage points per 100 HCPs — no dollars
        assert "per 100" in g["highlights"]
        assert "$" not in g["highlights"]
        chips = {c["label"]: c["value"] for c in g["grounding"]}
        assert chips["Null channels"] == "Digital Engagement"
        assert "Speaker Program Invitation" in chips["Top channel"]

    def test_rows_without_channel_or_results_are_excluded(self):
        g = build_grounding(
            "All",
            [
                {"intervention_channel": None, "ab_experiment_results": [
                    {"effect_estimate": 0.9, "is_significant": True}
                ]},
                {"intervention_channel": "email_campaign", "ab_experiment_results": []},
                {"intervention_channel": "email_campaign", "ab_experiment_results": [
                    {"effect_estimate": None}
                ]},
            ],
        )
        assert g["n_experiments"] == 0
        assert g["channels"] == []

    def test_minority_significant_channel_is_not_a_winner(self):
        g = build_grounding(
            "All",
            [
                _row("email_campaign", 0.05, True),
                _row("email_campaign", 0.02, False),
                _row("email_campaign", 0.01, False),
            ],
        )
        chips = {c["label"]: c["value"] for c in g["grounding"]}
        assert "Top channel" not in chips
        assert "No channel has majority-significant evidence" in g["highlights"]


class TestFallback:
    def test_fallback_is_deterministic_and_flagged(self):
        g = build_grounding("Fabhalta", [_row("sample_distribution", 0.11, True)])
        fb = _fallback(g)
        assert fb["is_fallback"] is True
        assert "Sample Distribution" in fb["insight"]
        assert "(Factual summary — LLM interpretation unavailable.)" in fb["insight"]
        assert fb["key_takeaways"]  # per-channel takeaways, not empty

    def test_empty_portfolio_degrades_honestly(self):
        fb = _fallback(build_grounding("Kisqali", []))
        assert fb["is_fallback"] is True
        assert "No running A/B experiments" in fb["insight"]
        assert fb["key_takeaways"] == []
