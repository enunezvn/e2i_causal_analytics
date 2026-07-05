"""Unit tests for the HTE strategic insight (grounding + fail-closed guard).

NOTE: CI's unit job does NOT run tests/insights/ (known blind spot) — run these
scoped locally when touching src/insights/hte.py.
"""

from types import SimpleNamespace

import pytest

from src.insights import hte


def _record(**overrides):
    base = {
        "treatment_var": "treatment_arm",
        "outcome_var": "persistent_180d",
        "brand": "Remibrutinib",
        "confidence_level": 0.95,
        "overall_ate": 0.1106,
        "heterogeneity_score": 0.26,
        "expected_lift_pp": 0.0,
        "optimal_allocation_summary": "No reliable differential-targeting opportunity.",
        "cate_by_segment": {
            "disease_severity_band": [
                {
                    "segment_value": "high",
                    "cate_estimate": 0.1772,
                    "cate_ci_lower": 0.1267,
                    "cate_ci_upper": 0.2277,
                    "sample_size": 1385,
                    "statistical_significance": True,
                },
                {
                    "segment_value": "low",
                    "cate_estimate": 0.0338,
                    "cate_ci_lower": -0.0280,
                    "cate_ci_upper": 0.0955,
                    "sample_size": 2498,
                    "statistical_significance": False,
                },
            ],
            "age_band": [
                {
                    "segment_value": "50-65",
                    "cate_estimate": 0.1375,
                    "cate_ci_lower": 0.0796,
                    "cate_ci_upper": 0.1955,
                    "sample_size": 2015,
                    "statistical_significance": True,
                },
            ],
        },
    }
    base.update(overrides)
    return base


class TestBuildGrounding:
    def test_scope_and_effect_summary_render_real_figures(self):
        g = hte.build_grounding(_record())
        assert "treatment_arm -> persistent_180d" in g["scope"]
        assert "Remibrutinib" in g["scope"]
        assert "+11.1pp" in g["effect_summary"]
        assert "2 of 3 segments" in g["effect_summary"]
        assert g["sig_count"] == 2 and g["total_count"] == 3

    def test_cohort_n_is_per_dimension_sum_not_all_rows(self):
        # severity dim: 1385+2498=3883; age dim: 2015 -> cohort = max = 3883,
        # NOT 1385+2498+2015=5898 (dimensions partition the same cohort).
        g = hte.build_grounding(_record())
        assert "n=3,883" in g["scope"]
        assert "5,898" not in g["scope"]

    def test_segments_sorted_desc_with_pp_and_significance(self):
        g = hte.build_grounding(_record())
        lines = g["segments"].splitlines()
        assert lines[0].startswith("disease_severity_band=high: +17.7pp")
        assert "not significant" in lines[-1]
        assert "n=1,385" in lines[0]

    def test_expected_lift_fraction_renders_true_pp(self):
        # The producer stores expected_lift_pp as a probability FRACTION
        # despite the name (policy_learner validates [0,1] and multiplies by
        # 100 only at display) — a 0.021 lift is +2.1pp, not +0.0pp.
        g = hte.build_grounding(_record(expected_lift_pp=0.021))
        assert "+2.1pp" in g["targeting"]
        assert "+0.0pp" not in g["targeting"]

    def test_zero_expected_lift_still_stated(self):
        g = hte.build_grounding(_record())
        assert "+0.0pp" in g["targeting"]

    def test_no_signal_when_no_cate_rows(self):
        g = hte.build_grounding(_record(cate_by_segment={}, overall_ate=None))
        assert g["has_signal"] is False


class TestGuard:
    def test_faithful_output_passes(self):
        g = hte.build_grounding(_record())
        text = (
            "The overall effect is +11.1pp and 2 of 3 segments clear zero. "
            "High severity responds strongest at +17.7pp [CI +12.7pp to +22.8pp] "
            "(n=1,385), while the low band (+3.4pp) is not distinguishable from "
            "no effect. Expected lift from differential targeting is +0.0pp."
        )
        assert hte._is_grounded(text, g) is True

    def test_fabricated_pp_rejected(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The high band gains +24.9pp of persistence.", g) is False

    def test_rederived_delta_rejected(self):
        # 17.7 - 3.4 = 14.3pp spread is a RE-DERIVED figure, never rendered.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The spread between bands is 14.3pp.", g) is False

    def test_fabricated_count_fraction_rejected(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Fully 3 of 3 segments are significant.", g) is False

    def test_true_count_fraction_passes(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("2 of 3 segments are significant.", g) is True

    def test_comma_formatted_n_passes(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The strongest segment holds 1,385 patients.", g) is True

    def test_unvouched_integer_rejected(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Roll out to 500 more HCPs.", g) is False

    def test_sign_flip_rejected(self):
        # codex round-1 HIGH: grounded "+11.1pp" must not vouch "-11.1pp".
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Overall ATE is -11.1pp.", g) is False

    def test_unit_swap_rejected(self):
        # codex round-1 HIGH: grounded "+11.1pp" must not vouch "+11.1%".
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Overall ATE is +11.1%.", g) is False

    def test_signed_pp_and_percentage_point_wording_pass(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Overall ATE is +11.1pp.", g) is True
        assert hte._is_grounded("An 11.1 percentage-point gain overall.", g) is True

    def test_negative_ci_bound_keeps_its_sign(self):
        # Low band CI lower is -2.8pp: the negative form passes, the flipped
        # positive form is a different claim and must not.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The low band CI dips to -2.8pp.", g) is True
        assert hte._is_grounded("The low band gains +2.8pp.", g) is False

    def test_markdown_bullet_hyphen_is_not_a_sign(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("- 11.1pp overall effect", g) is True

    def test_fraction_variants_all_checked(self):
        # codex round-1 HIGH: "3 out of 3" / "3-of-3" bypassed the fraction
        # rule; codex round-2 HIGH: so did hyphenated "3-out-of-3".
        g = hte.build_grounding(_record())
        for wrong in ("3 of 3", "3/3", "3 out of 3", "3-of-3", "3-out-of-3", "3 out-of 3"):
            assert hte._is_grounded(f"Fully {wrong} segments are significant.", g) is False
        assert hte._is_grounded("2 out of 3 segments are significant.", g) is True

    def test_unit_bearing_number_not_vouched_bare(self):
        # codex round-2 HIGH: "95" is rendered only as "95% CIs" — re-using it
        # bare as a count ("95 segments") must reject; the %-form stays fine.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("95 segments have CIs excluding zero.", g) is False
        assert hte._is_grounded("Reach out to 95 HCPs first.", g) is False
        assert hte._is_grounded("At the 95% confidence level, 2 of 3 clear zero.", g) is True

    def test_vouched_number_misattributed_as_segment_count_rejected(self):
        # codex round-2 HIGH: a genuinely vouched unitless number (n=1,385)
        # re-attributed as a segment count must reject; its true attribution
        # (patients) still passes.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("There are 1,385 significant segments.", g) is False
        assert hte._is_grounded("2 significant segments emerged from 3.", g) is True

    def test_unanchored_range_phrase_is_a_quantity_claim(self):
        # codex round-2 HIGH: "50 to 65 significant segments clear zero." used
        # the age band's spelled-out range as a segment-count range. Anchored
        # band mentions keep passing.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("50 to 65 significant segments clear zero.", g) is False
        assert hte._is_grounded("Patients 50 to 65 respond at +13.8pp.", g) is True
        assert hte._is_grounded("The 50 to 65 band responds strongest.", g) is True

    def test_worded_or_spaced_sign_flip_rejected(self):
        # codex round-3 HIGH: grounded "+11.1pp" must not vouch word-sign or
        # word-preceded spaced-hyphen negations (nor "-2.8pp" a worded plus).
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Overall ATE is negative 11.1pp.", g) is False
        assert hte._is_grounded("Overall ATE is minus 11.1pp.", g) is False
        assert hte._is_grounded("Overall ATE is - 11.1pp.", g) is False
        assert hte._is_grounded("The low band CI dips to positive 2.8pp.", g) is False
        assert hte._is_grounded("A positive 11.1pp overall effect.", g) is True
        assert hte._is_grounded("The low band CI dips to negative 2.8pp.", g) is True

    def test_segment_count_synonyms_and_long_modifiers_rejected(self):
        # codex round-3 HIGH: count-noun synonyms (subgroups/cohorts/bands)
        # and 3+ modifier words bypassed the segment-count rule. Population
        # words keep their own attribution and stay exempt.
        g = hte.build_grounding(_record())
        for wrong in (
            "1,385 significant clinically relevant priority segments.",
            "1,385 significant subgroups emerged.",
            "1,385 significant cohorts emerged.",
            "1,385 significant bands emerged.",
        ):
            assert hte._is_grounded(wrong, g) is False
        assert hte._is_grounded("Analysis spans 3 cohorts.", g) is True
        assert hte._is_grounded("The strongest segment holds 1,385 patients.", g) is True

    def test_compact_range_reused_as_count_rejected(self):
        # codex round-3 HIGH: compact dash forms of the age band ("50-65",
        # en/em dash) were unconditionally stripped. Anchored name mentions
        # (singular noun, aged/patients, dimension=value) keep passing.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("50-65 significant segments clear zero.", g) is False
        assert hte._is_grounded("50–65 significant segments clear zero.", g) is False
        assert hte._is_grounded("50—65 significant segments clear zero.", g) is False
        assert hte._is_grounded("The 50-65 segment responds strongest.", g) is True
        assert hte._is_grounded("Patients aged 50-65 respond at +13.8pp.", g) is True
        assert hte._is_grounded("age_band=50-65 leads at +13.8pp.", g) is True

    def test_anchor_word_cannot_launder_a_count_claim(self):
        # codex round-3 HIGH: "Patients 50 to 65 significant segments ..."
        # stripped the range via the Patients anchor, leaving no claim.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Patients 50 to 65 significant segments clear zero.", g) is False
        assert hte._is_grounded("Groups 50 to 65 significant segments clear zero.", g) is False
        assert hte._is_grounded("Patients 50 to 65 form the strongest segment.", g) is True

    def test_unicode_dash_fraction_separators_checked(self):
        # codex round-3 HIGH: "3–out–of–3" (en dashes) bypassed the fraction
        # rule while both bare 3s were individually vouched.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Fully 3–out–of–3 segments are significant.", g) is False
        assert hte._is_grounded("2–out–of–3 segments are significant.", g) is True

    def test_typography_cannot_hide_a_sign_flip(self):
        # codex round-4 HIGH: extra spaces, a colon, or a hyphen-attached word
        # sign hid the flip; a spaced "+" hid the -2.8pp flip. Bulleted lists
        # stay unsigned.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Overall ATE is  - 11.1pp.", g) is False
        assert hte._is_grounded("Overall ATE: - 11.1pp.", g) is False
        assert hte._is_grounded("Overall ATE is minus-11.1pp.", g) is False
        assert hte._is_grounded("The low band CI dips to + 2.8pp.", g) is False
        assert hte._is_grounded("- +17.7pp in high severity\n- 11.1pp overall", g) is True

    def test_punctuated_or_adjectival_segment_counts_rejected(self):
        # codex round-4 HIGH: 5+ modifiers, a comma inside the window, and
        # exempt words used adjectivally ("patient segments") bypassed the
        # fixed-width segment-count window.
        g = hte.build_grounding(_record())
        for wrong in (
            "1,385 significant clinically relevant priority target segments emerged.",
            "1,385 significant, clinically relevant segments emerged.",
            "1,385 patient segments are significant.",
            "1,385 patients-adjacent segments are significant.",
        ):
            assert hte._is_grounded(wrong, g) is False
        assert hte._is_grounded("The strongest segment holds 1,385 patients.", g) is True

    def test_parenthesized_or_attached_range_cannot_launder(self):
        # codex round-4 HIGH: "(50-65)" and "age_band=50-65" were stripped
        # before the count guard could see the plural count phrase they head.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Patients (50-65) significant segments clear zero.", g) is False
        assert hte._is_grounded("Groups (50 to 65) significant segments clear zero.", g) is False
        assert hte._is_grounded("age_band=50-65 significant segments clear zero.", g) is False
        assert hte._is_grounded("Patients (50-65) respond at +13.8pp.", g) is True
        assert hte._is_grounded("age_band=50-65 leads at +13.8pp.", g) is True

    def test_exotic_fraction_separators_checked(self):
        # codex round-4 HIGH: non-breaking hyphens, the fraction slash, and
        # "over" all bypassed the fraction rule.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Fully 3‑out‑of‑3 segments are significant.", g) is False
        assert hte._is_grounded("Fully 3⁄3 segments are significant.", g) is False
        assert hte._is_grounded("Fully 3 over 3 segments are significant.", g) is False
        assert hte._is_grounded("2 over 3 segments are significant.", g) is True

    def test_unicode_plus_and_bridged_sign_words_rejected(self):
        # codex round-5 HIGH: fullwidth/emoji plus glyphs and "±" read as
        # unsigned; sign words separated by a short bridge went unbound.
        # "Plus," as a discourse marker and adjectival "positive in 2 of 3"
        # (unitless) keep passing.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The low band CI dips to ＋2.8pp.", g) is False
        assert hte._is_grounded("The low band CI dips to ➕2.8pp.", g) is False
        assert hte._is_grounded("The low band CI dips to ±2.8pp.", g) is False
        assert hte._is_grounded("Overall ATE is negative net 11.1pp.", g) is False
        assert hte._is_grounded("The low band CI dips to positive, 2.8pp.", g) is False
        assert hte._is_grounded("Plus, 2 of 3 segments are significant.", g) is True
        assert hte._is_grounded("The effect is positive in 2 of 3 segments.", g) is True

    def test_punctuated_modifier_tokens_cannot_hide_segment_counts(self):
        # codex round-5 HIGH: parens/brackets/slashes/attached hyphens inside
        # the modifier run bypassed the segment-count rule; a parenthetical
        # "(n=1,385)," must NOT be counted against later segment nouns.
        g = hte.build_grounding(_record())
        for wrong in (
            "1,385 significant (clinically relevant) segments emerged.",
            "1,385 significant [priority] segments emerged.",
            "1,385 significant/priority segments emerged.",
            "1,385-significant segments emerged.",
            "1,385 patient(s) segments are significant.",
        ):
            assert hte._is_grounded(wrong, g) is False
        assert (
            hte._is_grounded(
                "High severity responds strongest (n=1,385), while 2 of 3 segments clear zero.",
                g,
            )
            is True
        )

    def test_phrase_matches_whole_tokens_only(self):
        # codex round-5 HIGH: "50-65+" had its "50-65" substring stripped,
        # re-attributing the effect to an ungrounded 65+-inclusive label.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("age_band=50-65+ leads at +13.8pp.", g) is False
        assert hte._is_grounded("Patients aged 50-65+ respond at +13.8pp.", g) is False
        assert hte._is_grounded("age_band = 50-65 leads at +13.8pp.", g) is True
        assert hte._is_grounded("Patients 50 -65 respond at +13.8pp.", g) is True

    def test_fullwidth_slash_and_word_fraction_separators_checked(self):
        # codex round-5 HIGH: "3／3" (fullwidth slash) and in/per/among
        # separator words bypassed the fraction rule.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Fully 3／3 segments are significant.", g) is False
        assert hte._is_grounded("Fully 3 in 3 segments are significant.", g) is False
        assert hte._is_grounded("Fully 3 per 3 segments are significant.", g) is False
        assert hte._is_grounded("Fully 3 among 3 segments are significant.", g) is False
        assert hte._is_grounded("2 in 3 segments are significant.", g) is True

    def test_vouched_numbers_cannot_swap_segments(self):
        # codex round-6 HIGH: vouching was global, so one segment's figures
        # could be served under another segment's name. A sentence naming
        # exactly one segment may only carry that row's figures plus globals;
        # comparisons naming several segments fall back to global vouching.
        g = hte.build_grounding(_record())
        assert (
            hte._is_grounded(
                "The age_band=50-65 segment responds strongest at +17.7pp "
                "[CI +12.7pp to +22.8pp], n=1,385.",
                g,
            )
            is False
        )
        assert hte._is_grounded("The low band responds at +17.7pp.", g) is False
        assert hte._is_grounded("High severity responds at +13.8pp (n=2,015).", g) is False
        assert (
            hte._is_grounded("The 50-65 band responds at +13.8pp [CI +8.0pp to +19.6pp].", g)
            is True
        )
        assert (
            hte._is_grounded("High severity leads at +17.7pp while the 50-65 band adds +13.8pp.", g)
            is True
        )
        assert hte._is_grounded("The 50-65 band nearly matches the +11.1pp overall ATE.", g) is True

    def test_vouched_numbers_cannot_swap_metrics(self):
        # codex round-6 HIGH: the ATE could be served as the expected lift
        # (and any vouched pp figure as either metric).
        g = hte.build_grounding(_record())
        assert (
            hte._is_grounded(
                "Expected lift from differential targeting is +11.1pp, matching the overall ATE.",
                g,
            )
            is False
        )
        assert (
            hte._is_grounded(
                "The targeting verdict is no opportunity, with expected lift +17.7pp.", g
            )
            is False
        )
        assert hte._is_grounded("The overall ATE is +17.7pp.", g) is False
        assert hte._is_grounded("Expected lift from differential targeting is +0.0pp.", g) is True
        assert hte._is_grounded("The ATE of +11.1pp exceeds the +0.0pp lift.", g) is True
        assert hte._is_grounded("The overall effect is +11.1pp with heterogeneity 0.26.", g) is True

    def test_metric_anchor_variants_and_long_clauses_bound(self):
        # codex round-7 HIGH: "treatment effect overall" / "Overall, the
        # treatment effect" missed the ATE anchor, and a long parenthetical
        # clause escaped the fixed 60-char window. Binding now follows the
        # copula: the first figure after the anchor in the clause.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The treatment effect overall is +17.7pp.", g) is False
        assert hte._is_grounded("Overall, the treatment effect is +17.7pp.", g) is False
        assert (
            hte._is_grounded(
                "The expected lift from differential targeting, after weighing the "
                "no-opportunity verdict against the high-severity response, is +17.7pp.",
                g,
            )
            is False
        )
        assert hte._is_grounded("Overall, the treatment effect is +11.1pp.", g) is True
        assert (
            hte._is_grounded(
                "Expected lift is +0.0pp because high severity (+17.7pp) already leads.", g
            )
            is True
        )
        assert (
            hte._is_grounded("High severity (+17.7pp) drives the overall effect of +11.1pp.", g)
            is True
        )

    def test_segment_noun_synonyms_count_as_segment_context(self):
        # codex round-7 HIGH: "the high category/bucket" was not recognized
        # as a mention of the high segment, skipping attribution.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The high category responds at +13.8pp (n=2,015).", g) is False
        assert hte._is_grounded("The high bucket responds at +13.8pp (n=2,015).", g) is False
        assert hte._is_grounded("The low category responds at +17.7pp.", g) is False
        assert hte._is_grounded("The high category responds at +17.7pp (n=1,385).", g) is True

    def test_three_digit_sample_sizes_cannot_cross_attribute(self):
        # codex round-7 HIGH: unitless integers were governed only at 4+
        # digits, so records with 3-digit segment n could swap sample sizes.
        record = _record()
        record["cate_by_segment"]["disease_severity_band"][0]["sample_size"] = 815
        record["cate_by_segment"]["disease_severity_band"][1]["sample_size"] = 985
        record["cate_by_segment"]["age_band"][0]["sample_size"] = 765
        g = hte.build_grounding(record)
        assert (
            hte._is_grounded("The low severity segment has n=815 and responds at +3.4pp.", g)
            is False
        )
        assert hte._is_grounded("The 50-65 band responds at +13.8pp with n=985.", g) is False
        assert (
            hte._is_grounded("The low severity segment has n=985 and responds at +3.4pp.", g)
            is True
        )
        assert hte._is_grounded("The 50-65 band responds at +13.8pp with n=765.", g) is True

    def test_metric_paraphrase_families_bound(self):
        # codex round-8 HIGH: "population-level treatment effect" and
        # targeting-benefit paraphrases ("incremental gain from targeting",
        # "targeting offers ...") escaped the metric anchors.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The population-level treatment effect is +17.7pp.", g) is False
        assert (
            hte._is_grounded("The incremental gain from differential targeting is +17.7pp.", g)
            is False
        )
        assert hte._is_grounded("Segment-based targeting offers a +17.7pp improvement.", g) is False
        assert hte._is_grounded("The population-level treatment effect is +11.1pp.", g) is True
        assert (
            hte._is_grounded(
                "Differential targeting offers no advantage; the expected lift is +0.0pp.", g
            )
            is True
        )

    def test_dimension_context_and_verb_headed_segment_mentions(self):
        # codex round-8 HIGH: table-like prose ("Within disease_severity_band,
        # high responds ...", "High responds at ...") escaped the mention
        # detector's adjacency requirement.
        g = hte.build_grounding(_record())
        assert (
            hte._is_grounded("Within disease_severity_band, high responds at +13.8pp (n=2,015).", g)
            is False
        )
        assert (
            hte._is_grounded("For disease severity, high responds at +13.8pp (n=2,015).", g)
            is False
        )
        assert (
            hte._is_grounded(
                "Disease severity separates the response. High responds at +13.8pp (n=2,015).", g
            )
            is False
        )
        assert (
            hte._is_grounded(
                "Disease severity separates the response. High responds at +17.7pp (n=1,385).", g
            )
            is True
        )

    def test_appositive_and_far_sign_words_bind(self):
        # codex round-8 HIGH: appositive ("11.1pp, a negative effect") and
        # far-prepositive ("a net negative treatment effect of 11.1pp") sign
        # words went unbound.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The overall ATE was 11.1pp, a negative effect.", g) is False
        assert (
            hte._is_grounded("The overall ATE is a net negative treatment effect of 11.1pp.", g)
            is False
        )
        assert (
            hte._is_grounded("The high severity effect is 17.7pp, a negative result.", g) is False
        )
        assert hte._is_grounded("The overall ATE was 11.1pp, a positive effect.", g) is True

    def test_total_count_cannot_vouch_significance_claims(self):
        # codex round-9 HIGH: "All 3 segments have 95% CIs excluding zero."
        # passed because the total count vouched any segment-count claim,
        # even inside a significance predicate. Fraction forms stay exempt.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("All 3 segments have 95% CIs excluding zero.", g) is False
        assert hte._is_grounded("There are 3 segments with 95% CIs excluding zero.", g) is False
        assert hte._is_grounded("Significant segments: 3.", g) is False
        assert hte._is_grounded("2 of 3 segments have 95% CIs excluding zero.", g) is True
        assert hte._is_grounded("Only 2 segments are significant.", g) is True
        assert hte._is_grounded("Significant segments: 2/3.", g) is True
        assert hte._is_grounded("All 3 segments were tested; 2 are significant.", g) is True

    def test_targeting_opportunity_and_improves_by_bound_to_lift(self):
        # codex round-9 HIGH: "differential-targeting opportunity" (rendered
        # verbatim in the grounding) and "targeting improves ... by" escaped
        # the lift anchor.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The differential-targeting opportunity is +17.7pp.", g) is False
        assert (
            hte._is_grounded("There is a +17.7pp differential-targeting opportunity.", g) is False
        )
        assert hte._is_grounded("Differential targeting improves outcomes by +17.7pp.", g) is False
        assert (
            hte._is_grounded(
                "There is no reliable differential-targeting opportunity, and the "
                "expected lift is +0.0pp.",
                g,
            )
            is True
        )
        assert (
            hte._is_grounded("No reliable differential-targeting opportunity (+0.0pp).", g) is True
        )

    def test_has_had_segment_predicates_and_sign_linkers(self):
        # codex round-9 HIGH: "High has a +13.8pp effect" escaped the verb
        # list, and "11.1pp, indicating a negative effect" escaped the
        # postpositive sign binder.
        g = hte.build_grounding(_record())
        assert (
            hte._is_grounded(
                "Disease severity separates the response. High has a +13.8pp effect "
                "(n=2,015), while overall ATE is +11.1pp.",
                g,
            )
            is False
        )
        assert (
            hte._is_grounded("The overall ATE is 11.1pp, indicating a negative effect.", g) is False
        )
        assert (
            hte._is_grounded("The high severity effect is 17.7pp, indicating a negative result.", g)
            is False
        )
        assert (
            hte._is_grounded(
                "Disease severity separates the response. High has a +17.7pp effect (n=1,385).", g
            )
            is True
        )
        assert (
            hte._is_grounded("The overall ATE is 11.1pp, indicating a positive effect.", g) is True
        )

    def test_heterogeneity_and_cohort_n_role_bound(self):
        # codex round-11 HIGH: role binding existed only for ATE/lift, so the
        # scale endpoint could pose as the heterogeneity score and a segment
        # n as the cohort n.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Heterogeneity score is 1 on a 0-1 scale.", g) is False
        assert hte._is_grounded("Heterogeneity score is 0.26 on a 0-1 scale.", g) is True
        assert hte._is_grounded("Heterogeneity (0-1 scale) is 0.26.", g) is True
        assert (
            hte._is_grounded(
                "For treatment_arm -> persistent_180d, brand filter Remibrutinib, "
                "cohort n=1,385 with 95% CIs, the overall ATE is +11.1pp.",
                g,
            )
            is False
        )
        assert (
            hte._is_grounded(
                "For treatment_arm -> persistent_180d, brand filter Remibrutinib, "
                "cohort n=3,883 with 95% CIs, the overall ATE is +11.1pp.",
                g,
            )
            is True
        )

    def test_totality_wording_forces_true_total(self):
        # codex round-11 HIGH: the significant count (2) could pose as the
        # total ("2 total segments in the analysis"). Sig-predicated counts
        # and source prepositions ("from 3 tested segments") stay exempt.
        g = hte.build_grounding(_record())
        assert (
            hte._is_grounded("Overall ATE is +11.1pp, with 2 total segments in the analysis.", g)
            is False
        )
        assert (
            hte._is_grounded(
                "Overall ATE is +11.1pp, with 3 total segments in the analysis; 2 are significant.",
                g,
            )
            is True
        )
        assert hte._is_grounded("There are 2 significant segments in the analysis.", g) is True
        assert (
            hte._is_grounded("Two significant segments emerged from 3 tested segments.", g) is True
        )

    def test_impact_and_targeting_benefit_anchors(self):
        # codex round-11 HIGH: "Overall impact" and "differential-targeting
        # benefit/gain" escaped the metric anchors.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Overall impact is +17.7pp across the cohort.", g) is False
        assert hte._is_grounded("The differential-targeting benefit is +17.7pp.", g) is False
        assert hte._is_grounded("The differential-targeting gain is +17.7pp.", g) is False
        assert hte._is_grounded("Overall impact is +11.1pp across the cohort.", g) is True
        assert (
            hte._is_grounded("There is no reliable differential-targeting benefit (+0.0pp).", g)
            is True
        )

    def test_round11_medium_faithful_interpretation_passes(self):
        # codex round-11 MEDIUM over-rejection, fixed via the source-
        # preposition exemption ("from 3 tested segments").
        g = hte.build_grounding(_record())
        text = (
            "The analysis shows an 11.1 percentage-point positive effect overall. "
            "Two significant segments emerged from 3 tested segments. age_band=50-65 "
            "is +13.8pp [CI +8.0pp to +19.6pp], n=2,015, significant. No reliable "
            "differential-targeting opportunity is shown (+0.0pp)."
        )
        assert hte._is_grounded(text, g) is True

    def test_elided_subject_significance_counts_bind(self):
        # codex round-12 HIGH: "3 are significant" (subject elided) fell
        # through as a vouched bare 3. Negated forms bind the complement.
        g = hte.build_grounding(_record())
        assert (
            hte._is_grounded(
                "Overall ATE is +11.1pp, with 3 total segments in the analysis; 3 are significant.",
                g,
            )
            is False
        )
        assert (
            hte._is_grounded(
                "Overall ATE is +11.1pp, with 3 total segments in the analysis; 2 are significant.",
                g,
            )
            is True
        )
        assert (
            hte._is_grounded("Of the 3 segments, 2 are significant and 1 is not significant.", g)
            is True
        )
        assert hte._is_grounded("All 3 are significant.", g) is False

    def test_cohort_prose_and_analysis_included_bind(self):
        # codex round-12 HIGH: "The cohort includes 1,385 patients" and "The
        # analysis included 2 segments" escaped the cohort-n and totality
        # bindings.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The cohort includes 1,385 patients, with 95% CIs.", g) is False
        assert hte._is_grounded("This is an observational cohort of 1,385 patients.", g) is False
        assert hte._is_grounded("The cohort includes 3,883 patients, with 95% CIs.", g) is True
        assert hte._is_grounded("This is an observational cohort of 3,883 patients.", g) is True
        assert (
            hte._is_grounded(
                "The analysis included 2 segments and reports an overall ATE of +11.1pp.", g
            )
            is False
        )
        assert (
            hte._is_grounded(
                "The analysis included 3 segments and reports an overall ATE of +11.1pp.", g
            )
            is True
        )

    def test_round12_medium_faithful_interpretation_passes(self):
        # codex round-12 MEDIUM over-rejection: the "1" in "(0-1 scale)" was
        # read as a segment count for the later "cohort" noun.
        g = hte.build_grounding(_record())
        text = (
            "There are 2 significant segments in the analysis. Overall ATE is "
            "+11.1pp, heterogeneity score 0.26 (0-1 scale), and cohort n=3,883. "
            "Expected lift from differential targeting is +0.0pp."
        )
        assert hte._is_grounded(text, g) is True

    def test_elided_ci_predicates_and_fraction_denominators_bind(self):
        # codex round-13 HIGHs: elided counts with CI phrasing ("3 have 95%
        # CIs excluding zero"), wrong fraction DENOMINATORS ("2 of 2 segments
        # ..."), and "the analysis has 2 segments" all escaped. Negated
        # fractions count the complement.
        g = hte.build_grounding(_record())
        assert (
            hte._is_grounded(
                "Overall ATE is +11.1pp, with 3 total segments in the analysis; "
                "3 have 95% CIs excluding zero.",
                g,
            )
            is False
        )
        assert hte._is_grounded("Of the 3 segments, 3 clear zero.", g) is False
        assert (
            hte._is_grounded(
                "Overall ATE is +11.1pp; 2 of 2 segments have 95% CIs excluding zero.", g
            )
            is False
        )
        assert hte._is_grounded("Significant segments: 2/2.", g) is False
        assert (
            hte._is_grounded(
                "The analysis has 2 segments and reports an overall ATE of +11.1pp.", g
            )
            is False
        )
        assert (
            hte._is_grounded(
                "Overall ATE is +11.1pp, with 3 total segments in the analysis; "
                "2 have 95% CIs excluding zero.",
                g,
            )
            is True
        )
        assert hte._is_grounded("Of the 3 segments, 2 clear zero.", g) is True
        assert hte._is_grounded("Significant segments: 2/3.", g) is True
        assert (
            hte._is_grounded(
                "The analysis has 3 segments and reports an overall ATE of +11.1pp.", g
            )
            is True
        )
        assert hte._is_grounded("1 of 3 segments is not significant.", g) is True
        assert hte._is_grounded("2 of 3 segments are not significant.", g) is False

    def test_copula_headed_segment_figures_bind(self):
        # codex round-10 HIGH (single finding): "High is +13.8pp" escaped the
        # mention grammar. The copula binds only when it heads a figure, so
        # predicate uses ("confidence is high") cannot false-fire.
        g = hte.build_grounding(_record())
        assert (
            hte._is_grounded(
                "Disease severity separates the response. High is +13.8pp (n=2,015).", g
            )
            is False
        )
        assert hte._is_grounded("High was +13.8pp in this analysis.", g) is False
        assert hte._is_grounded("High: +13.8pp (n=2,015).", g) is False
        assert (
            hte._is_grounded(
                "Disease severity separates the response. High is +17.7pp (n=1,385).", g
            )
            is True
        )
        assert hte._is_grounded("Confidence is high; 2 of 3 segments clear zero.", g) is True
        assert hte._is_grounded("Heterogeneity is high at 0.26 on a 0-1 scale.", g) is True

    def test_explicit_metric_comparison_stays_legal(self):
        # round-9 over-rejection fix: "overall ATE: +17.7pp versus +11.1pp"
        # names both sides of a comparison — legal when the metric's true
        # value is one of them.
        g = hte.build_grounding(_record())
        assert (
            hte._is_grounded(
                "The high severity segment is above the overall ATE: +17.7pp versus +11.1pp.", g
            )
            is True
        )
        assert (
            hte._is_grounded("The expected lift is +17.7pp versus the observed response.", g)
            is False
        )

    def test_postpositive_sign_words_bind_to_unit_figures(self):
        # codex round-6 HIGH: "an 11.1pp negative effect" read as unsigned.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The overall ATE is an 11.1pp negative effect.", g) is False
        assert (
            hte._is_grounded("The overall effect is an 11.1 percentage-point negative effect.", g)
            is False
        )
        assert hte._is_grounded("An 11.1pp positive effect overall.", g) is True

    def test_ascii_plus_minus_never_vouches(self):
        # codex round-6 HIGH: "+/-2.8pp" parsed as a signed -2.8pp and passed.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The low band CI dips to +/-2.8pp.", g) is False
        assert (
            hte._is_grounded("The low band confidence interval is +/-2.8pp to +9.6pp.", g) is False
        )

    def test_segment_count_rule_cannot_start_mid_decimal(self):
        # codex round-6 MEDIUM over-rejection: _SEG_COUNT_RE matched "1pp ..."
        # inside "+11.1pp" and rejected faithful prose mentioning "cohort".
        g = hte.build_grounding(_record())
        text = (
            "Overall ATE is +11.1pp for persistent_180d, with cohort n=3,883 and 95% CIs. "
            "High severity responds strongest at +17.7pp, while low severity is weaker at "
            "+3.4pp with CI -2.8pp to +9.6pp. These are model-based estimates from one "
            "causal-forest analysis on an observational cohort."
        )
        assert hte._is_grounded(text, g) is True

    def test_variable_name_digits_pass_only_in_context(self):
        # codex round-1 HIGH: "Treat 180 patients." re-used persistent_180d's
        # digits bare. In-context uses (the name itself, "180-day") stay fine.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("persistent_180d (180-day persistence) improves.", g) is True
        assert hte._is_grounded("Persistence at 180 days improves.", g) is True
        assert hte._is_grounded("Treat 180 patients.", g) is False

    def test_segment_value_digits_pass_only_in_context(self):
        # codex round-1 HIGH: "2 of 65 segments" passed because the 50-65 age
        # band vouched a free-floating 65.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Patients 50 to 65 (the 50-65 band) respond.", g) is True
        assert hte._is_grounded("2 of 65 segments clear zero.", g) is False

    def test_phrase_followed_by_unit_is_a_numeric_claim(self):
        # An age-band range re-used as a quantity ("50 to 65 percent") must
        # NOT be stripped as a name mention — its digits face the guard.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Share rises 50 to 65 percent in responders.", g) is False


class TestGenerateInsight:
    def test_no_signal_returns_honest_empty_fallback(self):
        g = hte.build_grounding(_record(cate_by_segment={}, overall_ate=None))
        out = hte.generate_insight(g)
        assert out["is_fallback"] is True
        assert "no per-segment CATE results" in out["insight"]

    def test_lm_unavailable_returns_factual_fallback(self, monkeypatch):
        monkeypatch.setattr("src.insights.hte.run_signature", lambda *a, **k: None)
        g = hte.build_grounding(_record())
        out = hte.generate_insight(g)
        assert out["is_fallback"] is True
        assert "+11.1pp" in out["insight"]
        assert "Factual summary" in out["insight"]

    def test_grounded_lm_output_served(self, monkeypatch):
        monkeypatch.setattr(
            "src.insights.hte.run_signature",
            lambda *a, **k: SimpleNamespace(
                interpretation=(
                    "Treatment lifts persistence by +11.1pp overall; high severity "
                    "(+17.7pp, n=1,385) responds most."
                ),
                key_takeaways=["2 of 3 segments clear zero", "Uniform rollout is appropriate"],
            ),
        )
        out = hte.generate_insight(hte.build_grounding(_record()))
        assert out["is_fallback"] is False
        assert "+17.7pp" in out["insight"]

    def test_ungrounded_lm_output_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            "src.insights.hte.run_signature",
            lambda *a, **k: SimpleNamespace(
                interpretation="Persistence improves 42.5pp in responders.",
                key_takeaways=[],
            ),
        )
        out = hte.generate_insight(hte.build_grounding(_record()))
        assert out["is_fallback"] is True
        assert "42.5" not in out["insight"]

    def test_ungrounded_takeaway_also_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            "src.insights.hte.run_signature",
            lambda *a, **k: SimpleNamespace(
                interpretation="Overall ATE is +11.1pp.",
                key_takeaways=["Target the 900 highest-value HCPs"],
            ),
        )
        out = hte.generate_insight(hte.build_grounding(_record()))
        assert out["is_fallback"] is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.1106, "+11.1pp"), (-0.028, "-2.8pp"), (None, None), (float("nan"), None)],
)
def test_pp_formatting(value, expected):
    assert hte._pp(value) == expected
