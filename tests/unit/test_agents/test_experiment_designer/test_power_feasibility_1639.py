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


class TestTheBoundDoesNotSlanderLegitimateTrials:
    """codex iter-2 HIGH: a FALSE "not executable" is worse than a missing one.

    My first bound reused the ``7 <= duration <= 365`` GEPA reward term from
    dspy_integration.py:112 as an output-path validator. That reasoning was
    wrong: the reward term SCORES an optimizer, where preferring short designs
    is a legitimate bias, whereas the output path makes an ASSERTION ABOUT
    REALITY, where a 16-month oncology trial is routine.

    Measured on the real node: effect 0.15 on a 0.30 baseline at 50/week is
    n=3,388 over 476 days. That is a normal study, and it was being labelled
    not executable.
    """

    def _run(self, effect, baseline, accrual, constraints=None):
        import asyncio

        from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode

        c = {"weekly_accrual": accrual}
        c.update(constraints or {})
        return asyncio.run(
            PowerAnalysisNode().execute(
                {
                    "design_type": "RCT",
                    "constraints": c,
                    "outcomes": [
                        {
                            "is_primary": True,
                            "metric_type": "binary",
                            "expected_effect_size": effect,
                            "baseline_value": baseline,
                        }
                    ],
                }
            )
        )

    def test_a_sixteen_month_trial_is_not_slandered(self):
        out = self._run(0.15, 0.30, 50)
        assert 366 <= out["duration_estimate_days"] <= 3650, out["duration_estimate_days"]
        assert out["feasibility_warnings"] == [], out["feasibility_warnings"]

    def test_a_three_year_trial_is_not_slandered(self):
        out = self._run(0.10, 0.30, 50)
        assert 1000 <= out["duration_estimate_days"] <= 3650, out["duration_estimate_days"]
        assert out["feasibility_warnings"] == [], out["feasibility_warnings"]

    def test_the_257_year_design_is_still_flagged(self):
        out = self._run(0.030, 0.05, 50)
        assert out["duration_estimate_days"] == 94115
        assert out["feasibility_warnings"], "the absurd case must survive the loosened bound"

    def test_a_stated_maximum_is_enforced_however_short(self):
        """When the CALLER states a limit, exceeding it is a fact, not a guess —
        and that is the only way a 476-day design should ever be flagged."""
        out = self._run(0.15, 0.30, 50, {"max_duration_days": 180})
        assert out["duration_estimate_days"] > 180
        assert out["feasibility_warnings"], out["duration_estimate_days"]
        blob = " ".join(out["feasibility_warnings"])
        assert "180" in blob, blob

    def test_a_stated_maximum_that_is_met_stays_quiet(self):
        out = self._run(0.15, 0.30, 50, {"max_duration_days": 720})
        assert out["feasibility_warnings"] == [], out["feasibility_warnings"]

    def test_the_timeline_constraint_is_also_read(self):
        """``timeline`` is the documented constraint key (agent.py valid_keys)."""
        out = self._run(0.15, 0.30, 50, {"timeline": {"max_duration_days": 180}})
        assert out["feasibility_warnings"], out["duration_estimate_days"]


class TestAnAuditWithEvidenceIsNotCalledNeverRun:
    """codex iter-2 HIGH: a checkpoint written before this change carries audit
    RESULTS but no status, and defaulting to "not_run" is then a false
    provenance claim — the mirror image of the defect being fixed."""

    def _doc(self, state_extra):
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        state = {"preregistration_formality": "heavy", "power_analysis": {}}
        state.update(state_extra)
        return TemplateGeneratorNode()._generate_preregistration(state)

    def test_threats_present_without_status_is_not_never_ran(self):
        doc = self._doc(
            {
                "validity_threats": [{"threat_name": "selection bias", "severity": "high"}],
                "overall_validity_score": 0.62,
            }
        )
        assert "never ran" not in doc
        assert "selection bias" in doc
        assert "0.62" in doc

    def test_a_score_without_threats_or_status_is_not_never_ran(self):
        doc = self._doc({"validity_threats": [], "overall_validity_score": 0.82})
        assert "never ran" not in doc

    def test_a_genuinely_empty_state_still_says_never_ran(self):
        doc = self._doc({"validity_threats": [], "overall_validity_score": 0.0})
        assert "never ran" in doc


class TestTheOtherArtifactsCarryTheCaveat:
    """codex iter-2 HIGH: the monitoring spec and experiment template
    re-project the infeasible sample size and duration into an execution plan.
    A consumer of only those gets enrollment target 672,206 and a timeline
    centuries out, with no caveat anywhere."""

    def _state(self):
        return {
            "power_analysis": {"required_sample_size": 672206},
            "duration_estimate_days": 94115,
            "feasibility_warnings": ["Estimated duration 94,115 days ..."],
            "validity_audit_status": "skipped",
            "stratification_variables": [],
        }

    def test_monitoring_spec_carries_the_warnings(self):
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        spec = TemplateGeneratorNode()._generate_monitoring_spec(self._state())
        assert spec.get("feasibility_warnings"), sorted(spec)

    def test_experiment_template_carries_the_warnings(self):
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        tpl = TemplateGeneratorNode()._build_experiment_template(self._state())
        assert dict(tpl).get("feasibility_warnings"), sorted(dict(tpl))


class TestEveryReProjectionCarriesTheProvenance:
    """codex iter-2: each surface that re-projects the design is its own chance
    to drop the caveat. Four were found; this pins all four so the next one
    added has to answer the same question."""

    def test_training_signal_carries_scale_and_reasons(self):
        from src.agents.experiment_designer.dspy_integration import (
            ExperimentDesignTrainingSignal,
        )

        sig = ExperimentDesignTrainingSignal()
        sig.minimum_detectable_effect = 0.0015
        sig.minimum_detectable_effect_scale = "absolute_risk_difference"
        sig.duration_estimate_days = 94115
        sig.feasibility_warnings = ["Estimated duration 94,115 days ..."]
        sig.validity_audit_status = "skipped"
        d = sig.to_dict()
        assert d["power_analysis"]["minimum_detectable_effect_scale"] == "absolute_risk_difference"
        assert d["power_analysis"]["feasibility_warnings"]
        assert d["validity_audit"]["validity_audit_status"] == "skipped"

    def test_mlflow_metrics_count_the_warnings(self):
        from src.agents.experiment_designer.mlflow_tracker import (
            ExperimentDesignerMLflowTracker,
        )

        tracker = ExperimentDesignerMLflowTracker.__new__(ExperimentDesignerMLflowTracker)
        metrics = tracker._extract_metrics(
            {
                "duration_estimate_days": 94115,
                "feasibility_warnings": ["a", "b"],
                "validity_threats": [],
            },
            None,
        )
        assert metrics.feasibility_warnings_count == 2


class TestEveryStatedTimelineShapeIsRead:
    """codex iter-3 HIGH, and the sharpest miss of the run: I invented
    ``max_duration_days``, tested the shape I invented, and it passed.

    The shapes callers actually use are FOUR, and three of them predate this
    change:

    * ``timeline_weeks: 12`` — the contract's own worked example
      (.claude/contracts/tier3-contracts.md)
    * ``timeline: "3 months"`` — free text, used across the agent tests
    * ``timeline: {"max_duration_days": 90}`` — src/testing/tier0_output_mapper.py
    * ``max_duration_days: 180`` — the one I added

    A caller stating ``timeline_weeks: 12`` against a 476-day design was told
    nothing, because 476 days is under the 10-year absurdity bound.
    """

    def _warnings(self, constraints):
        import asyncio

        from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode

        c = {"weekly_accrual": 50}
        c.update(constraints)
        out = asyncio.run(
            PowerAnalysisNode().execute(
                {
                    "design_type": "RCT",
                    "constraints": c,
                    "outcomes": [
                        {
                            "is_primary": True,
                            "metric_type": "binary",
                            "expected_effect_size": 0.15,
                            "baseline_value": 0.30,
                        }
                    ],
                }
            )
        )
        assert out["duration_estimate_days"] == 476, out["duration_estimate_days"]
        return out["feasibility_warnings"]

    def test_timeline_weeks_from_the_contract_example(self):
        assert self._warnings({"timeline_weeks": 12}), "the contract's own shape was ignored"

    def test_timeline_as_free_text_is_deliberately_NOT_enforced(self):
        """This asserted enforcement for four rounds. Prose parsing was removed
        after five rounds of counter-examples — see
        ``TestFreeTextTimelinesAreNotParsedAtAll`` for the evidence and the
        reasoning. The capability survives in the structured shapes below."""
        assert self._warnings({"timeline": "3 months"}) == []

    def test_timeline_as_a_dict(self):
        assert self._warnings({"timeline": {"max_duration_days": 90}})

    def test_direct_max_duration_days(self):
        assert self._warnings({"max_duration_days": 90})

    def test_a_generous_stated_timeline_stays_quiet(self):
        assert self._warnings({"timeline_weeks": 100}) == []
        assert self._warnings({"max_duration_days": 720}) == []

    def test_no_prose_timeline_is_guessed_at(self):
        """Originally about unparseable strings; now the rule for ALL prose."""
        assert self._warnings({"timeline": "as soon as possible"}) == []
        assert self._warnings({"timeline": "Q3"}) == []
        assert self._warnings({"timeline": "3 months"}) == []


class TestEpisodicPrecedentsCarryTheCaveat:
    """codex iter-3 HIGH: an infeasible design is stored as a PRECEDENT that
    later designs learn from. Storing 94,115 days with no first-class reason
    teaches the next design that it was acceptable."""

    def test_record_carries_the_structured_fields(self):
        from src.agents.experiment_designer.memory_hooks import ExperimentDesignRecord

        names = set(ExperimentDesignRecord.__dataclass_fields__)
        assert {"feasibility_warnings", "validity_audit_status"} <= names, sorted(names)

    def test_to_dict_serializes_them(self):
        from src.agents.experiment_designer.memory_hooks import ExperimentDesignRecord

        fields = ExperimentDesignRecord.__dataclass_fields__
        kwargs = {}
        for name, f in fields.items():
            if name == "timestamp":
                from datetime import datetime, timezone

                kwargs[name] = datetime.now(timezone.utc)
            elif f.type in ("int",):
                kwargs[name] = 0
            elif f.type in ("float",):
                kwargs[name] = 0.0
            else:
                kwargs[name] = None
        kwargs["feasibility_warnings"] = ["Estimated duration 94,115 days ..."]
        kwargs["validity_audit_status"] = "skipped"
        kwargs["warnings"] = []
        kwargs["constraints"] = {}
        d = ExperimentDesignRecord(**kwargs).to_dict()
        assert d["feasibility_warnings"] == ["Estimated duration 94,115 days ..."]
        assert d["validity_audit_status"] == "skipped"


class TestTheContractNamesWhatTheRuntimeActuallyEmits:
    """codex iter-3 HIGH: I documented ``achievable_mde_scale``, a name that
    exists nowhere in the runtime. Inventing a contract name and then marking
    it compliant by translation is a labeling fix, not a functional one."""

    def test_contract_uses_the_runtime_field_name(self):
        from pathlib import Path

        text = Path(".claude/contracts/tier3-contracts.md").read_text()
        assert "achievable_mde_scale" not in text, "an invented contract-only name"
        assert "minimum_detectable_effect_scale" in text

    def test_the_record_is_built_from_the_result_not_left_at_defaults(self):
        """A field wired into the dataclass but never populated is worse than
        absent: it reports "no warnings" for every design ever stored."""
        import inspect

        from src.agents.experiment_designer import memory_hooks

        src = inspect.getsource(memory_hooks)
        assert 'feasibility_warnings=result.get("feasibility_warnings")' in src
        assert 'validity_audit_status=result.get("validity_audit_status"' in src


class TestTheAnalysisCodeTemplateCarriesTheCaveat:
    """codex iter-4 HIGH: the generated analysis script prints
    ``Sample Size: 672206`` in its header. Someone who opens only that file gets
    the uncaveated execution artifact this issue exists to prevent."""

    def test_analysis_code_header_flags_an_infeasible_design(self):
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        node = TemplateGeneratorNode()
        state = {
            "power_analysis": {"required_sample_size": 672206},
            "duration_estimate_days": 94115,
            "feasibility_warnings": ["Estimated duration 94,115 days (257.7 years) ..."],
            "treatments": [{"name": "detailing"}],
            "outcomes": [{"name": "conversion", "is_primary": True}],
        }
        code = node._generate_analysis_code(state, node._build_dowhy_spec(state))
        assert "672206" in code
        assert "NOT EXECUTABLE" in code.upper() or "FEASIBILITY" in code.upper(), code[:500]

    def test_a_feasible_design_gets_no_banner(self):
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        node = TemplateGeneratorNode()
        state = {
            "power_analysis": {"required_sample_size": 1930},
            "duration_estimate_days": 70,
            "feasibility_warnings": [],
            "treatments": [{"name": "detailing"}],
            "outcomes": [{"name": "conversion", "is_primary": True}],
        }
        code = node._generate_analysis_code(state, node._build_dowhy_spec(state))
        assert "NOT EXECUTABLE" not in code.upper()


class TestTheAuditorSeesTheCaveat:
    """codex iter-5 HIGH: the validity-audit prompt re-renders the sample size
    and asks an LLM whether the design is sound. Without the caveat it can drive
    a REDESIGN decision from an uncaveated projection."""

    def test_audit_prompt_carries_duration_and_feasibility(self):
        from src.agents.experiment_designer.nodes.validity_audit import ValidityAuditNode

        prompt = ValidityAuditNode._build_audit_prompt(
            ValidityAuditNode.__new__(ValidityAuditNode),
            {
                "power_analysis": {"required_sample_size": 672206},
                "duration_estimate_days": 94115,
                "feasibility_warnings": ["Estimated duration 94,115 days (257.7 years) ..."],
            },
        )
        assert "672206" in prompt
        assert "94115" in prompt or "94,115" in prompt
        assert "257.7 years" in prompt

    def test_a_feasible_design_gets_no_feasibility_block(self):
        from src.agents.experiment_designer.nodes.validity_audit import ValidityAuditNode

        prompt = ValidityAuditNode._build_audit_prompt(
            ValidityAuditNode.__new__(ValidityAuditNode),
            {
                "power_analysis": {"required_sample_size": 1930},
                "duration_estimate_days": 70,
                "feasibility_warnings": [],
            },
        )
        assert "NOT EXECUTABLE" not in prompt.upper()


class TestStaleWarningsDoNotSurviveARedesign:
    def _run(self, state):
        import asyncio

        from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode

        return asyncio.run(PowerAnalysisNode().execute(state))

    def test_the_error_path_does_not_leave_a_stale_verdict(self):
        """codex iter-5 MED: iteration N flags infeasible, the redesign changes
        inputs, iteration N+1 errors before assignment — a consumer then reads
        the OLD warning attached to a different, failed design."""
        out = self._run(
            {
                "design_type": "RCT",
                "constraints": {"weekly_accrual": "not-a-number"},
                "outcomes": [{"is_primary": True, "metric_type": "binary"}],
                "feasibility_warnings": ["STALE from iteration N"],
            }
        )
        assert out["status"] == "failed"
        assert "STALE from iteration N" not in out["feasibility_warnings"]
        assert out["feasibility_warnings"], "a failed assessment is not a clean bill of health"

    def test_the_warning_is_not_duplicated_across_iterations(self):
        """codex iter-5 LOW: power analysis appends to the generic ``warnings``
        channel, so a redesign that leaves power inputs unchanged stacks the
        same sentence twice."""
        state = {
            "design_type": "RCT",
            "constraints": {"weekly_accrual": 50},
            "outcomes": [
                {
                    "is_primary": True,
                    "metric_type": "binary",
                    "expected_effect_size": 0.030,
                    "baseline_value": 0.05,
                }
            ],
        }
        first = self._run(state)
        second = self._run(dict(first, status="calculating"))
        duration_warnings = [w for w in second.get("warnings", []) if "duration" in w.lower()]
        assert len(duration_warnings) == 1, duration_warnings


class TestARedesignClearsTheOldVerdict:
    """codex iter-6 HIGH x2: `feasibility_warnings` is replaced on each run, but
    the copy pushed onto the generic `warnings` channel was not — so an answer
    could show a design that is now feasible beside iteration 1's
    "94,115 days" sentence, or show "not assessed" beside an obsolete verdict."""

    def _run(self, state):
        import asyncio

        from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode

        return asyncio.run(PowerAnalysisNode().execute(state))

    def _state(self, effect, baseline, **extra):
        s = {
            "design_type": "RCT",
            "constraints": {"weekly_accrual": 50},
            "outcomes": [
                {
                    "is_primary": True,
                    "metric_type": "binary",
                    "expected_effect_size": effect,
                    "baseline_value": baseline,
                }
            ],
        }
        s.update(extra)
        return s

    def test_becoming_feasible_removes_the_old_warning(self):
        infeasible = self._run(self._state(0.030, 0.05))
        assert infeasible["feasibility_warnings"]
        assert any("94,115" in w for w in infeasible["warnings"])

        redesigned = self._run(
            self._state(
                0.15,
                0.30,
                **{
                    "warnings": infeasible["warnings"],
                    "feasibility_warnings": infeasible["feasibility_warnings"],
                    "status": "calculating",
                },
            )
        )
        assert redesigned["feasibility_warnings"] == []
        assert not any("94,115" in w for w in redesigned["warnings"]), redesigned["warnings"]

    def test_unrelated_warnings_are_preserved(self):
        out = self._run(
            self._state(
                0.15,
                0.30,
                **{
                    "warnings": ["Validity audit skipped (disabled)"],
                    "feasibility_warnings": ["Estimated duration 94,115 days ..."],
                    "status": "calculating",
                },
            )
        )
        assert "Validity audit skipped (disabled)" in out["warnings"]
        assert not any("94,115" in w for w in out["warnings"])

    def test_the_error_path_also_clears_the_old_generic_warning(self):
        out = self._run(
            self._state(
                0.15,
                0.30,
                **{
                    "constraints": {"weekly_accrual": "not-a-number"},
                    "warnings": ["Estimated duration 94,115 days ...", "keep me"],
                    "feasibility_warnings": ["Estimated duration 94,115 days ..."],
                },
            )
        )
        assert out["status"] == "failed"
        assert not any("94,115" in w for w in out["warnings"]), out["warnings"]
        assert "keep me" in out["warnings"]
        assert any("not assessed" in w for w in out["feasibility_warnings"])


class TestFreeTextTimelinesAreNotParsedAtAll:
    """The decision that ended five rounds of parser defects.

    Every attempt to infer a bound's DIRECTION from prose produced a new
    counter-example, in both directions:

    ======================================  =========================================
    phrasing                                what the parser of the day did
    ======================================  =========================================
    "2 month ramp; no longer than 24 months" first-match took 61 days -> FALSE warning
    "12 months recruitment plus 6 months     MAX took 12, but the stated window is
     follow-up"                              the SUM (18) -> FALSE warning
    "at least 3 months"                      single-duration read it as a cap
    "patients under observation for at       "under" matched inside "under
     least 3 months"                          observation" -> FALSE warning
    "not under 3 months"                     "under" matched -> FALSE warning
    "no more than 3 months"                  dead: floor list's "more than" won first
    "at least 2 and at most 6 months"        floor veto dropped the real 6-month cap
    "3 months maximum" / "3 months or less"  postposed cap ignored
    ======================================  =========================================

    A false warning is the worst outcome available here: it tells a caller their
    design is unrunnable against a limit they never set, and it discredits every
    real warning printed beside it. Free text is also not load-bearing — eval
    turn 3.6 stated no timeline at all and is caught by the plausibility bound.

    So prose is treated as "no stated limit". These cases are kept as tests
    because they are the evidence for the decision, not because prose parsing is
    coming back.
    """

    def _warnings(self, constraints):
        import asyncio

        from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode

        c = {"weekly_accrual": 50}
        c.update(constraints)
        out = asyncio.run(
            PowerAnalysisNode().execute(
                {
                    "design_type": "RCT",
                    "constraints": c,
                    "outcomes": [
                        {
                            "is_primary": True,
                            "metric_type": "binary",
                            "expected_effect_size": 0.15,
                            "baseline_value": 0.30,
                        }
                    ],
                }
            )
        )
        assert out["duration_estimate_days"] == 476, out["duration_estimate_days"]
        return out["feasibility_warnings"]

    @pytest.mark.parametrize(
        "timeline",
        [
            "3 months",
            "within 3 months",
            "no longer than 3 months",
            "no more than 3 months",
            "3 months maximum",
            "3 months or less",
            "at least 2 and at most 6 months",
            "at least 3 months",
            "not under 3 months",
            "patients under observation for at least 3 months",
            "12 months recruitment plus 6 months follow-up",
            "2 month recruitment ramp; total study no longer than 24 months",
            "as soon as possible",
            "Q3",
        ],
    )
    def test_no_prose_timeline_is_ever_enforced(self, timeline):
        assert self._warnings({"timeline": timeline}) == [], timeline

    def test_the_structured_shapes_are_still_enforced(self):
        """The capability is not lost — it is only expressed unambiguously."""
        assert self._warnings({"max_duration_days": 90})
        assert self._warnings({"timeline": {"max_duration_days": 90}})
        assert self._warnings({"timeline_weeks": 12})

    def test_a_generous_structured_limit_stays_quiet(self):
        assert self._warnings({"max_duration_days": 720}) == []
        assert self._warnings({"timeline_weeks": 100}) == []

    def test_the_absurd_design_is_still_caught_with_no_timeline_at_all(self):
        """#1639's actual payload stated no timeline. This is why prose parsing
        was never load-bearing."""
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
        assert out["duration_estimate_days"] == 94115
        assert out["feasibility_warnings"]


class TestOnlyDeclaredConstraintShapesAreRead:
    """Two of the four shapes I claimed to "read" were my own inventions.

    codex caught this at iter-3 for ``max_duration_days``; I then added
    ``timeline: {"weeks": 12}`` speculatively and it survived four more rounds
    because my own test was the only thing that wrote it. Grep is the check:

    * ``timeline_weeks``                  — the Tier 3 contract's worked example
    * ``timeline: {"max_duration_days"}`` — src/testing/tier0_output_mapper.py
    * ``max_duration_days``               — invented; KEPT, but now declared in
      ``validate_constraints`` and the contract example, so it is a supported
      key rather than a private convention
    * ``timeline: {"weeks"}``             — invented; REMOVED
    """

    def test_max_duration_days_is_a_declared_constraint_key(self):
        import inspect

        from src.agents.experiment_designer.agent import ExperimentDesignerInput

        src = inspect.getsource(ExperimentDesignerInput.validate_constraints)
        assert '"max_duration_days"' in src, "read but never declared"

    def test_the_contract_example_shows_it(self):
        from pathlib import Path

        text = Path(".claude/contracts/tier3-contracts.md").read_text()
        assert "max_duration_days" in text

    def test_the_invented_nested_weeks_shape_is_gone(self):
        import inspect

        from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode

        # It is a staticmethod on the node, not a module-level function — the
        # first version of this test looked it up on the module, so it failed
        # with AttributeError and would have "caught" anything at all.
        src = inspect.getsource(PowerAnalysisNode._stated_max_duration_days)
        assert 'timeline.get("weeks")' not in src, "a shape nobody writes is still read"
        assert 'constraints.get("timeline_weeks")' in src, "a real shape was dropped"


class TestAllStatedCapsMustBeSatisfied:
    """codex iter-9 HIGH: precedence silently ignored the stricter limit.

    A caller supplying BOTH ``timeline_weeks: 8`` (56 days) and
    ``timeline: {"max_duration_days": 90}`` had the 90 returned, so a 70-day
    design passed clean while violating the explicitly stated 8-week limit.

    These are MAXIMA. Satisfying all of them means the binding one is the
    smallest, not the first in some arbitrary lookup order.
    """

    def _out(self, constraints, effect=0.15, baseline=0.30, accrual=50):
        import asyncio

        from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode

        c = {"weekly_accrual": accrual}
        c.update(constraints)
        return asyncio.run(
            PowerAnalysisNode().execute(
                {
                    "design_type": "RCT",
                    "constraints": c,
                    "outcomes": [
                        {
                            "is_primary": True,
                            "metric_type": "binary",
                            "expected_effect_size": effect,
                            "baseline_value": baseline,
                        }
                    ],
                }
            )
        )

    def test_the_strictest_cap_binds(self):
        """Measured: effect 0.20 on a 0.30 baseline at 200/week is n=1,930 over
        70 days — inside the 90-day cap, outside the 8-week (56-day) one."""
        out = self._out(
            {"timeline_weeks": 8, "timeline": {"max_duration_days": 90}},
            effect=0.20,
            accrual=200,
        )
        assert out["duration_estimate_days"] == 70
        assert out["feasibility_warnings"], "the 8-week cap was ignored"
        assert "56" in " ".join(out["feasibility_warnings"])

    def test_a_design_inside_every_cap_stays_quiet(self):
        out = self._out(
            {"timeline_weeks": 8, "timeline": {"max_duration_days": 90}},
            effect=0.20,
            accrual=300,
        )
        assert out["duration_estimate_days"] <= 56, out["duration_estimate_days"]
        assert out["feasibility_warnings"] == []

    def test_a_single_cap_is_unchanged(self):
        assert self._out({"max_duration_days": 90})["feasibility_warnings"]
        assert self._out({"max_duration_days": 720})["feasibility_warnings"] == []

    def test_all_three_shapes_together_take_the_minimum(self):
        out = self._out(
            {
                "max_duration_days": 300,
                "timeline_weeks": 8,
                "timeline": {"max_duration_days": 120},
            },
            effect=0.20,
            accrual=200,
        )
        assert out["feasibility_warnings"]
        assert "56" in " ".join(out["feasibility_warnings"])


class TestTheMachineFieldCarriesTheStatusNotTheProse:
    """codex iter-10 HIGH: I reused the human phrasing as the machine value.

    ``_audit_verdict`` returns ``(completed, phrasing)`` where phrasing is prose
    for a sentence — "was skipped", "reported status 'completed'". The
    monitoring spec put that phrasing into ``validity_audit_status``, whose
    documented values are ``completed | skipped | timed_out | failed |
    not_run``. A consumer filtering on the enum matched nothing.
    """

    def _spec(self, state_extra):
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        state = {"power_analysis": {}, "stratification_variables": []}
        state.update(state_extra)
        return TemplateGeneratorNode()._generate_monitoring_spec(state)

    @pytest.mark.parametrize("status", ["completed", "skipped", "timed_out", "failed", "not_run"])
    def test_every_documented_status_survives_verbatim(self, status):
        assert self._spec({"validity_audit_status": status})["validity_audit_status"] == status

    def test_an_absent_status_is_inferred_not_phrased(self):
        """The same evidence-based inference as the document, but the VALUE."""
        assert (
            self._spec({"validity_threats": [{"threat_name": "x"}], "overall_validity_score": 0.6})[
                "validity_audit_status"
            ]
            == "completed"
        )
        assert (
            self._spec({"validity_threats": [], "overall_validity_score": 0.0})[
                "validity_audit_status"
            ]
            == "not_run"
        )

    def test_the_document_still_reads_as_prose(self):
        """The prose path must not regress into emitting a bare enum value."""
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        doc = TemplateGeneratorNode()._generate_preregistration(
            {
                "preregistration_formality": "heavy",
                "power_analysis": {},
                "validity_audit_status": "skipped",
            }
        )
        assert "was skipped" in doc, "the human sentence lost its phrasing"


class TestAnAuditThatDidNotCompleteRetractsTheOldVerdict:
    """codex iter-11 HIGH: the same staleness class I fixed in power_analysis,
    left unfixed one node over.

    Iteration 0 completes, finds "selection bias", triggers a redesign.
    Iteration 1 changes the design and its audit TIMES OUT. The status became
    "timed_out" while ``validity_threats``, ``mitigations``,
    ``overall_validity_score`` and ``redesign_recommendations`` still held
    iteration 0's findings — so `_create_output` published "timed_out" beside
    threats belonging to a design that no longer exists.
    """

    def _run(self, **extra):
        import asyncio

        from src.agents.experiment_designer.nodes.validity_audit import ValidityAuditNode

        state = {
            "status": "auditing",
            "enable_validity_audit": False,
            "validity_threats": [{"threat_name": "selection bias", "severity": "high"}],
            "mitigations": [{"threat_addressed": "selection bias", "strategy": "block"}],
            "overall_validity_score": 0.62,
            "redesign_recommendations": ["stratify by region"],
        }
        state.update(extra)
        return asyncio.run(ValidityAuditNode().execute(state))

    def test_a_skipped_audit_retracts_the_previous_findings(self):
        out = self._run()
        assert out["validity_audit_status"] == "skipped"
        assert out["validity_threats"] == [], out["validity_threats"]
        assert out.get("mitigations") == []
        assert out["overall_validity_score"] == 0.0
        assert out.get("redesign_recommendations") == []

    def test_the_first_pass_is_unaffected(self):
        """Nothing to retract when there was no previous verdict."""
        out = self._run(
            validity_threats=[],
            mitigations=[],
            overall_validity_score=0.0,
            redesign_recommendations=[],
        )
        assert out["validity_threats"] == []
        assert out["overall_validity_score"] == 0.0


class TestTheStatusFieldRejectsValuesOutsideItsEnum:
    """codex iter-11 HIGH: `_audit_status` returned whatever was in state.

    A checkpoint carrying the previous BAD value ``"was skipped"`` — the exact
    prose bug fixed one round earlier — or a typo like ``"timeout"`` flowed
    straight into the documented machine enum, recreating the failure for any
    consumer filtering on it.
    """

    def _status(self, value):
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        return TemplateGeneratorNode._audit_status({"validity_audit_status": value})

    @pytest.mark.parametrize("status", ["completed", "skipped", "timed_out", "failed", "not_run"])
    def test_documented_values_pass_through(self, status):
        assert self._status(status) == status

    @pytest.mark.parametrize("bogus", ["was skipped", "timeout", "", "COMPLETED", 7, None])
    def test_anything_else_becomes_unknown_not_a_guess(self, bogus):
        if bogus is None:
            pytest.skip("absent status is the inference path, covered elsewhere")
        assert self._status(bogus) == "unknown"

    def test_unknown_is_a_documented_value(self):
        from src.agents.experiment_designer.nodes.template_generator import (
            TemplateGeneratorNode,
        )

        completed, phrasing = TemplateGeneratorNode._audit_verdict(
            {"validity_audit_status": "timeout"}
        )
        # Exact: pre-fix this said "reported status 'timeout'" — the typo
        # echoed straight back. The `or "reported status"` I first wrote here
        # passed both ways and proved nothing.
        assert completed is False
        assert phrasing == "reported status 'unknown'", phrasing


class TestTheAuditStatusIsNormalizedWhereItLeaves:
    """codex iter-12 HIGH: I validated the leaf, not the source.

    `_audit_status` guards the monitoring spec, but `_create_output` passed
    `state["validity_audit_status"]` through raw — so the PRIMARY output, the
    one every consumer reads, still published `"was skipped"` or `"timeout"`.
    """

    @pytest.mark.parametrize(
        "bogus, expected", [("was skipped", "unknown"), ("timeout", "unknown")]
    )
    def test_the_public_output_cannot_carry_an_out_of_band_status(self, bogus, expected):
        from src.agents.experiment_designer.state import normalize_audit_status

        assert normalize_audit_status(bogus) == expected

    def test_the_output_boundary_uses_the_normalizer(self):
        import ast
        import inspect

        from src.agents.experiment_designer.agent import ExperimentDesignerAgent

        src = inspect.getsource(ExperimentDesignerAgent._create_output)
        tree = ast.parse(src.lstrip())
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "normalize_audit_status" in called, (
            "_create_output publishes validity_audit_status without normalizing it"
        )

    def test_one_definition_of_the_enum(self):
        """Two copies drift; the leaf guard and the output guard must agree."""
        from src.agents.experiment_designer.nodes.template_generator import AUDIT_STATUSES
        from src.agents.experiment_designer.state import AUDIT_STATUSES as CANONICAL

        assert AUDIT_STATUSES is CANONICAL


class TestRetractionAlsoDropsTheAuditsProse:
    """codex iter-12 HIGH: the structured verdict was retracted, the words were not.

    A completed iteration-0 audit appends DAG findings to `state["warnings"]`
    ("Assumed confounder X was NOT discovered in causal DAG"). After a
    redesign whose audit does not complete, clearing only the structured
    fields left those sentences attached — so the user reads a DAG verdict
    about a design that no longer exists, beside status "timed_out".
    """

    def _run_skipped_after_a_completed_audit(self):
        import asyncio

        from src.agents.experiment_designer.nodes.validity_audit import ValidityAuditNode

        dag_warning = "Assumed confounder region was NOT discovered in causal DAG"
        state = {
            "status": "auditing",
            "enable_validity_audit": False,
            "validity_threats": [{"threat_name": "selection bias", "severity": "high"}],
            "overall_validity_score": 0.62,
            "dag_validation_warnings": [dag_warning],
            "dag_missing_confounders": ["region"],
            "warnings": ["An unrelated upstream warning", dag_warning],
        }
        return dag_warning, asyncio.run(ValidityAuditNode().execute(state))

    def test_the_previous_audits_warnings_are_withdrawn(self):
        dag_warning, out = self._run_skipped_after_a_completed_audit()
        assert dag_warning not in out.get("warnings", []), out.get("warnings")

    def test_warnings_from_other_sources_survive(self):
        """Retraction, not a blanket wipe — the audit only withdraws its own."""
        _, out = self._run_skipped_after_a_completed_audit()
        assert "An unrelated upstream warning" in out.get("warnings", [])

    def test_the_dag_findings_are_cleared_too(self):
        _, out = self._run_skipped_after_a_completed_audit()
        assert out.get("dag_validation_warnings") == []
        assert out.get("dag_missing_confounders") == []


class TestANonCompletedAuditEarnsNoValidityCredit:
    """codex iter-12 MED: the training signal could not tell silence apart.

    `compute_reward` gave the same partial validity credit for "the audit
    found no threats" and "the audit never ran", so selection could learn to
    prefer a design whose audit timed out.
    """

    def _reward(self, status):
        from src.agents.experiment_designer.dspy_integration import (
            ExperimentDesignTrainingSignal,
        )

        signal = ExperimentDesignTrainingSignal()
        signal.validity_threats_identified = 0
        signal.overall_validity_score = 0.0
        signal.validity_audit_status = status
        return signal.compute_reward()

    def test_a_completed_audit_with_no_threats_scores_higher_than_one_that_never_ran(self):
        assert self._reward("completed") > self._reward("timed_out")
        assert self._reward("completed") > self._reward("skipped")
        assert self._reward("completed") > self._reward("failed")


class TestMlflowCanTellARetractedZeroFromARealOne:
    """codex iter-12 MED: 0.0 means two different things without the status.

    A dashboard averaging `overall_validity_score` counts a retracted
    non-completing audit as a genuine zero-score verdict.
    """

    def test_the_status_is_logged_beside_the_score(self):
        import inspect

        from src.agents.experiment_designer import mlflow_tracker

        src = inspect.getsource(mlflow_tracker)
        assert "validity_audit_status" in src, (
            "mlflow logs overall_validity_score with no way to tell a retracted 0.0 apart"
        )
