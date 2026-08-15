"""#1639: an infeasible design must be flagged where a reader will see it.

Turn 3.6 of the 2026-08-15 eval emitted a **257.7-year** study duration
(``duration_estimate_days: 94115``) printed verbatim inside the tool's own
pre-registration document, beside a sample size of 672,206.

THE FILED DIAGNOSIS WAS WRONG, and keeping the disproof matters because it says
what NOT to fix. The issue called the MDE "20x smaller than the effect being
designed for" and treated that as evidence the solve was broken. It is not:

    binary_outcome_power(effect_size=0.030, alpha=0.05, power=0.8, baseline_rate=0.05)
      -> sample_size = 672206     # observed: 672206  exact
         mde         = 0.0015     # observed: 0.0015  exact

Both figures come from ONE call, on two different scales. ``effect_size`` is a
RELATIVE change (``p2 = p1 * (1 + effect)``, ``power_analysis_lib.py:98``) and
``mde`` is the ABSOLUTE risk difference (``|p2 - p1|``, line 126). On a 5%
baseline a 3% relative lift IS a 0.15pp absolute difference, and
``0.0015 / 0.030`` recovers the baseline exactly.

So the arithmetic is correct end to end, and n = 672,206 is the honest sample
size for detecting 0.15pp on a 5% base rate — an infeasible design faithfully
computed. The defects are therefore about REPORTING, not solving:

1. two scales printed side by side with no unit on either, which reads as a
   contradiction to every reader (the answer layer, a user opening the document,
   and the person who filed the issue);
2. nothing on the OUTPUT path bounds or flags the duration — the
   ``7 <= duration <= 365`` range at ``dspy_integration.py:112`` is a GEPA reward
   term for optimizer scoring, not a validator;
3. ``overall_validity_score: 0.0`` with an empty ``validity_threats`` list, empty
   because the audit timed out, reads identically to "no threats found".
"""

import pytest

from src.utils.power_analysis_lib import binary_outcome_power

pytestmark = pytest.mark.unit


class TestTheSolveIsCorrectAndMustNotBeChanged:
    """Pinned so a future reader does not "fix" a number that is already right."""

    def test_observed_payload_reproduces_exactly(self):
        r = binary_outcome_power(0.030, 0.05, 0.8, 0.05)
        assert r.sample_size == 672206
        assert round(r.mde, 6) == 0.0015

    def test_mde_and_effect_are_the_same_quantity_on_two_scales(self):
        """The "20x mismatch" is baseline_rate, nothing more."""
        baseline, relative = 0.05, 0.030
        r = binary_outcome_power(relative, 0.05, 0.8, baseline)
        assert round(r.mde / relative, 6) == baseline


class TestFeasibilityIsFlaggedOnTheOutputPath:
    def _run(self, constraints=None, outcomes=None):
        import asyncio

        from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode

        state = {
            "design_type": "RCT",
            "constraints": constraints or {},
            "outcomes": outcomes
            or [
                {
                    "is_primary": True,
                    "metric_type": "binary",
                    "expected_effect_size": 0.030,
                    "baseline_value": 0.05,
                }
            ],
        }
        return asyncio.run(PowerAnalysisNode().execute(state))

    def test_absurd_duration_is_flagged_not_merely_emitted(self):
        """The 3.6 payload. 94,115 days must not leave the node unmarked."""
        out = self._run()
        assert out["duration_estimate_days"] > 365
        assert out.get("feasibility_warnings"), (
            "a 257-year duration left the node with no feasibility warning: "
            f"{out.get('duration_estimate_days')} days"
        )
        blob = " ".join(out["feasibility_warnings"]).lower()
        assert "duration" in blob

    def test_a_feasible_design_carries_no_warning(self):
        """The flag must discriminate, not fire on everything."""
        out = self._run(
            outcomes=[
                {
                    "is_primary": True,
                    "metric_type": "binary",
                    "expected_effect_size": 0.5,
                    "baseline_value": 0.3,
                }
            ],
            constraints={"weekly_accrual": 200},
        )
        assert out["duration_estimate_days"] <= 365
        # Presence AND emptiness. `assert not out.get(...)` would pass just as
        # happily if a future change stopped setting the key at all, which is the
        # "checked, feasible" vs "never checked" distinction this field exists for.
        assert "feasibility_warnings" in out, sorted(out)
        assert out["feasibility_warnings"] == [], out["feasibility_warnings"]

    def test_mde_carries_its_scale(self):
        """0.0015 beside a relative 0.030 reads as a contradiction unless the
        output says which is which."""
        out = self._run()
        power = out["power_analysis"]
        assert "minimum_detectable_effect_scale" in power, sorted(power)
        assert power["minimum_detectable_effect_scale"] in {
            "absolute_risk_difference",
            "cohens_d",
            "relative_change",
            "hazard_ratio",
        }

    def test_sample_size_is_unchanged_by_the_flagging(self):
        """Flagging must not alter the arithmetic — the number is correct."""
        out = self._run()
        assert out["power_analysis"]["required_sample_size"] == 672206


class TestThePreRegistrationDocumentTellsTheTruth:
    """The document is the artifact a human actually reads and files.

    Turn 3.6 printed ``- **Duration:** 94115 days`` into a pre-registration with
    no qualification whatever, under a ``Validity Score: 0.00`` that was 0.00
    only because the audit had not run.
    """

    def _doc(self, state_extra=None, formality="heavy"):
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        state = {
            "design_type": "RCT",
            "preregistration_formality": formality,
            "treatments": [{"name": "detailing", "description": "rep visits"}],
            "outcomes": [{"name": "conversion", "is_primary": True}],
            "power_analysis": {
                "required_sample_size": 672206,
                "required_sample_size_per_arm": 336103,
                "achieved_power": 0.8,
                "minimum_detectable_effect": 0.0015,
                "minimum_detectable_effect_scale": "absolute_risk_difference",
                "effect_size_type": "rate_ratio",
                "alpha": 0.05,
                "assumptions": ["Two-sided test"],
            },
            "duration_estimate_days": 94115,
        }
        state.update(state_extra or {})
        return TemplateGeneratorNode()._generate_preregistration(state)

    def test_infeasible_duration_is_qualified_in_the_document(self):
        doc = self._doc({"feasibility_warnings": ["Estimated duration 94,115 days exceeds ..."]})
        assert "94115" in doc or "94,115" in doc, "duration must still be reported honestly"
        assert "Feasibility" in doc, (
            "a 257-year duration was printed with nothing marking it infeasible"
        )

    def test_a_feasible_design_gets_no_feasibility_section(self):
        doc = self._doc({"duration_estimate_days": 56, "feasibility_warnings": []})
        assert "Feasibility" not in doc

    def test_mde_is_not_labelled_with_the_input_effect_type(self):
        """``0.0015 (rate_ratio)`` is a false statement: 0.0015 is a risk
        DIFFERENCE. The document must label the MDE with the MDE's own scale."""
        doc = self._doc()
        assert "0.0015 (rate_ratio)" not in doc, (
            "the absolute MDE is labelled with the INPUT effect's type"
        )
        assert "absolute_risk_difference" in doc

    def test_an_audit_that_never_ran_does_not_read_as_no_threats_found(self):
        """``validity_threats: []`` + ``overall_validity_score: 0.0`` is what a
        skipped/timed-out audit leaves behind, and it renders identically to a
        clean bill of health."""
        doc = self._doc(
            {
                "validity_threats": [],
                "overall_validity_score": 0.0,
                "validity_audit_status": "timed_out",
            }
        )
        assert "None identified" not in doc, "an audit that timed out claimed no threats"
        assert "No significant threats identified" not in doc
        assert "timed_out" in doc or "not assessed" in doc.lower()

    def test_a_completed_audit_finding_nothing_still_says_so(self):
        """The distinction must cut both ways, or it is just a blanket hedge."""
        doc = self._doc(
            {
                "validity_threats": [],
                "overall_validity_score": 0.82,
                "validity_audit_status": "completed",
            }
        )
        assert "None identified" in doc or "No significant threats identified" in doc


class TestTheAuditRecordsWhetherItRan:
    def _audit_state(self, **extra):
        state = {"status": "auditing", "enable_validity_audit": False}
        state.update(extra)
        return state

    def test_a_skipped_audit_is_recorded_as_skipped(self):
        import asyncio

        from src.agents.experiment_designer.nodes.validity_audit import ValidityAuditNode

        out = asyncio.run(ValidityAuditNode().execute(self._audit_state()))
        assert out.get("validity_audit_status") == "skipped", sorted(out)


class TestTheWarningSurvivesToTheCaller:
    """The document is not the only reader.

    ``extract_narrative("experiment_designer", output)`` finds neither
    ``design_summary`` nor ``narrative`` on ``ExperimentDesignerOutput``, so the
    orchestrator's synthesizer falls back to stringifying the whole output dict
    — which is how turn 3.6's answer came to quote ``duration_estimate_days``
    verbatim. A warning that exists only inside the pre-registration markdown
    would be invisible on that path.
    """

    def test_output_model_carries_the_feasibility_verdict(self):
        from src.agents.experiment_designer.agent import ExperimentDesignerOutput

        fields = ExperimentDesignerOutput.model_fields
        assert "feasibility_warnings" in fields, sorted(fields)
        assert "validity_audit_status" in fields, sorted(fields)

    def test_create_output_propagates_both(self):
        from src.agents.experiment_designer.agent import ExperimentDesignerAgent

        out = ExperimentDesignerAgent._create_output(
            ExperimentDesignerAgent.__new__(ExperimentDesignerAgent),
            {
                "duration_estimate_days": 94115,
                "feasibility_warnings": ["Estimated duration 94,115 days ..."],
                "validity_audit_status": "timed_out",
                "validity_threats": [],
            },
        )
        assert out.feasibility_warnings == ["Estimated duration 94,115 days ..."]
        assert out.validity_audit_status == "timed_out"

    def test_the_generic_warnings_channel_also_carries_it(self):
        """``warnings`` is the field every downstream reader already knows."""
        import asyncio

        from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode

        out = asyncio.run(
            PowerAnalysisNode().execute(
                {
                    "design_type": "RCT",
                    "constraints": {},
                    "outcomes": [
                        {
                            "is_primary": True,
                            "metric_type": "binary",
                            "expected_effect_size": 0.030,
                            "baseline_value": 0.05,
                        }
                    ],
                }
            )
        )
        assert any("duration" in w.lower() for w in out.get("warnings", [])), out.get("warnings")


class TestReadablePrecision:
    def test_mde_is_printed_at_readable_precision(self):
        """``0.0015000000000000013`` -- the exact value the solve returns -- is
        binary-float noise, not six extra significant digits, and printing it
        verbatim made the figure look unserious beside a clean 0.030."""
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )
        from src.utils.power_analysis_lib import binary_outcome_power

        raw = binary_outcome_power(0.030, 0.05, 0.8, 0.05).mde
        assert repr(raw) == "0.0015000000000000013", repr(raw)

        doc = TemplateGeneratorNode()._generate_preregistration(
            {
                "preregistration_formality": "heavy",
                "power_analysis": {
                    "minimum_detectable_effect": raw,
                    "minimum_detectable_effect_scale": "absolute_risk_difference",
                    "effect_size_type": "rate_ratio",
                },
                "validity_audit_status": "completed",
            }
        )
        assert "0.0015 (absolute_risk_difference)" in doc
        assert "0.0015000000000000013" not in doc

    def test_a_non_numeric_effect_still_renders(self):
        """``TBD`` is the documented fallback and must survive formatting."""
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        doc = TemplateGeneratorNode()._generate_preregistration(
            {"preregistration_formality": "heavy", "validity_audit_status": "completed"}
        )
        assert "**Minimum Detectable Effect:** TBD" in doc


class TestTheLightPreRegistrationIsNotAnEscapeHatch:
    """Feasibility is not a formality-level concern.

    ``light`` is a separate template, not a truncation of ``medium`` -- so
    patching medium (which heavy inherits) left the shortest, most quotable
    artifact silently reporting an unrunnable design.
    """

    def _doc(self, formality):
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        return TemplateGeneratorNode()._generate_preregistration(
            {
                "preregistration_formality": formality,
                "treatments": [{"name": "detailing"}],
                "outcomes": [{"name": "conversion", "is_primary": True}],
                "power_analysis": {"required_sample_size": 672206},
                "duration_estimate_days": 94115,
                "feasibility_warnings": [
                    "Estimated duration 94,115 days (257.7 years) exceeds ..."
                ],
                "validity_audit_status": "completed",
            }
        )

    def test_every_formality_carries_the_feasibility_warning(self):
        missing = [f for f in ("light", "medium", "heavy") if "Feasibility" not in self._doc(f)]
        assert not missing, f"formalities that hid an unrunnable design: {missing}"


class TestTheScaleReachesThePublicOutputModel:
    """The state dict is not the public surface.

    ``PowerAnalysisOutput`` is an explicit field list, so a key added to the
    node's dict is DROPPED unless the model is widened too -- which recreates
    the labelling ambiguity on exactly the path this fix argued was the one
    that mattered.
    """

    def test_power_analysis_output_carries_the_mde_scale(self):
        from src.agents.experiment_designer.agent import PowerAnalysisOutput

        assert "minimum_detectable_effect_scale" in PowerAnalysisOutput.model_fields, sorted(
            PowerAnalysisOutput.model_fields
        )

    def test_create_output_propagates_the_scale(self):
        from src.agents.experiment_designer.agent import ExperimentDesignerAgent

        out = ExperimentDesignerAgent._create_output(
            ExperimentDesignerAgent.__new__(ExperimentDesignerAgent),
            {
                "power_analysis": {
                    "required_sample_size": 672206,
                    "required_sample_size_per_arm": 336103,
                    "achieved_power": 0.8,
                    "minimum_detectable_effect": 0.0015,
                    "minimum_detectable_effect_scale": "absolute_risk_difference",
                    "alpha": 0.05,
                    "effect_size_type": "rate_ratio",
                    "assumptions": [],
                }
            },
        )
        assert out.power_analysis is not None
        assert out.power_analysis.minimum_detectable_effect_scale == "absolute_risk_difference"

    def test_the_serialized_output_is_not_ambiguous(self):
        """The synthesizer stringifies the whole dump -- 0.0015 must not appear
        there beside 'rate_ratio' with nothing to separate them."""
        from src.agents.experiment_designer.agent import ExperimentDesignerAgent

        out = ExperimentDesignerAgent._create_output(
            ExperimentDesignerAgent.__new__(ExperimentDesignerAgent),
            {
                "power_analysis": {
                    "minimum_detectable_effect": 0.0015,
                    "minimum_detectable_effect_scale": "absolute_risk_difference",
                    "effect_size_type": "rate_ratio",
                }
            },
        )
        blob = str(out.model_dump())
        assert "absolute_risk_difference" in blob
